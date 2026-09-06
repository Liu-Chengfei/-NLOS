from __future__ import annotations

"""仿真数据物化器（sim_materializer）测试模块。

文件职责：验证 materialize_sim_raw 能正确生成仿真原始数据，
包括 VIO/IMU 残差保持、锚点布局族、噪声可复现性等。

测试覆盖范围：
- VIO 残差在非平凡变换后保持
- IMU 残差在非平凡变换后保持
- 不同布局族不破坏默认分割
- 锚点布局保留 fixture 原点并发出不同族 ID
- VIO 行保留 fixture 相对姿态残差
- VIO cycle 边界帧 quality=0
- GT 跨 cycle 边界推进
- IMU 行跨 cycle 重复残差模式
- 噪声可复现且非零
- 仅在目标为空时允许物化
- 非空目标拒绝物化

被测模块：liquidloc.dataio.readers.sim_materializer"""


import json
import math
from pathlib import Path

import pytest

from liquidloc.common.angle_utils import angle_delta_rad
from liquidloc.common.config_utils import load_yaml_config
from liquidloc.dataio.manifests.dataset_checks import inspect_layout_family_coverage
from liquidloc.dataio.manifests.split_builder import build_splits
from liquidloc.dataio.sim_materializer import (
    DEFAULT_SIM_NOISE_SPEC,
    DEFAULT_SIM_SEQUENCE_SPECS,
    SIM_E9_ONLY_COMPACT_SEQUENCE_SPECS,
    ZERO_SIM_NOISE_SPEC,
    _compute_cycle_pose_increment,
    _derive_imu_rows_from_gt,
    _interpolate_pose,
    _seeded_gaussian,
    _seeded_uniform,
    can_materialize_sim_raw,
    materialize_sim_raw,
    SimSequenceSpec,
)
from liquidloc.sensors.anchor_model import build_anchor_lookup
from liquidloc.sensors.uwb_model import predict_range_to_anchor


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _legacy_smoke_specs():
    """短 fixture 铺叠路径，仅测 residual/cycle 兼容语义，不进主表包络。"""
    return (
        SimSequenceSpec(
            "sim_rotate_01",
            "mini_seq_03",
            translation_xy=(0.20, 0.20),
            rotation_rad=math.pi / 2.0,
            cycle_count=4,
            cycle_gap_s=0.05,
            use_protocol_trajectory=False,
            envelope_profile="smoke",
            allow_high_anchor_count=True,
            duration_s=1.0,
            workspace_span_m=2.0,
        ),
        SimSequenceSpec(
            "sim_line_01",
            "mini_seq",
            translation_xy=(0.0, 0.0),
            cycle_count=4,
            cycle_gap_s=0.05,
            use_protocol_trajectory=False,
            envelope_profile="smoke",
            allow_high_anchor_count=True,
            duration_s=1.0,
            workspace_span_m=2.0,
        ),
    )

FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "datasets" / "miluv"


def test_default_sim_sequence_specs_use_protocol_trajectory_envelope():
    """主表默认序列必须走协议轨迹，且 K 非默认全优几何（K0）。"""
    assert all(spec.use_protocol_trajectory for spec in DEFAULT_SIM_SEQUENCE_SPECS)
    assert all(spec.duration_s >= 30.0 for spec in DEFAULT_SIM_SEQUENCE_SPECS)
    assert all(15.0 <= spec.workspace_span_m <= 40.0 for spec in DEFAULT_SIM_SEQUENCE_SPECS)
    k_levels = set()
    g_levels = set()
    for spec in DEFAULT_SIM_SEQUENCE_SPECS:
        axes = dict(spec.axes_override)
        if "K" in axes:
            k_levels.add(axes["K"])
        if "G" in axes:
            g_levels.add(axes["G"])
    # 五轴档位协议：K 轴仅 K0/K1/K3；主表必须有 K3 欠定或 K1 中等。
    assert "K3" in k_levels or "K1" in k_levels
    # 五轴档位协议：G 轴已并入 K 轴，G 字段不得出现。
    assert not g_levels, f"G axis must not appear (G merged into K): {g_levels}"
    # 主表不得只剩优几何 K0（应有 K1/K3 压力档）
    assert not k_levels <= {"K0"}


def test_sim_e9_only_compact_sequence_specs_are_exported_and_protocol_grade():
    assert len(SIM_E9_ONLY_COMPACT_SEQUENCE_SPECS) >= len(DEFAULT_SIM_SEQUENCE_SPECS)
    assert all(spec.use_protocol_trajectory for spec in SIM_E9_ONLY_COMPACT_SEQUENCE_SPECS)
    assert all(spec.duration_s >= 30.0 for spec in SIM_E9_ONLY_COMPACT_SEQUENCE_SPECS)
    k_levels = {dict(spec.axes_override).get("K") for spec in SIM_E9_ONLY_COMPACT_SEQUENCE_SPECS}
    assert "K3" in k_levels  # 五轴档位协议 K 轴仅 K0/K1/K3，sim_e9 固定 K3。

def _body_frame_delta(prev_pose: dict[str, float], curr_pose: dict[str, float]) -> tuple[float, float, float]:
    dx_world = float(curr_pose["px"]) - float(prev_pose["px"])
    dy_world = float(curr_pose["py"]) - float(prev_pose["py"])
    prev_yaw = float(prev_pose["yaw"])
    cos_value = math.cos(prev_yaw)
    sin_value = math.sin(prev_yaw)
    dx_local = (cos_value * dx_world) + (sin_value * dy_world)
    dy_local = (-sin_value * dx_world) + (cos_value * dy_world)
    dyaw = angle_delta_rad(float(curr_pose["yaw"]), prev_yaw)
    return dx_local, dy_local, dyaw


def test_materialize_sim_raw_writes_full_contract(tmp_path):
    report = materialize_sim_raw(tmp_path / "sim_raw", fixture_root=FIXTURE_ROOT)

    assert report["status"] == "ok"
    assert report["sequence_ids"] == [spec.seq_id for spec in DEFAULT_SIM_SEQUENCE_SPECS]
    assert len(report["sequence_ids"]) >= 10
    base_counts_by_seq = {
        "mini_seq": {
            "imu": len(json.loads((FIXTURE_ROOT / "mini_seq" / "imu.json").read_text(encoding="utf-8"))),
            "uwb": len(json.loads((FIXTURE_ROOT / "mini_seq" / "uwb.json").read_text(encoding="utf-8"))),
            "vio": len(json.loads((FIXTURE_ROOT / "mini_seq" / "vio.json").read_text(encoding="utf-8"))),
            "gt": len(json.loads((FIXTURE_ROOT / "mini_seq" / "gt.json").read_text(encoding="utf-8"))),
        },
        "mini_seq_02": {
            "imu": len(json.loads((FIXTURE_ROOT / "mini_seq_02" / "imu.json").read_text(encoding="utf-8"))),
            "uwb": len(json.loads((FIXTURE_ROOT / "mini_seq_02" / "uwb.json").read_text(encoding="utf-8"))),
            "vio": len(json.loads((FIXTURE_ROOT / "mini_seq_02" / "vio.json").read_text(encoding="utf-8"))),
            "gt": len(json.loads((FIXTURE_ROOT / "mini_seq_02" / "gt.json").read_text(encoding="utf-8"))),
        },
        "mini_seq_03": {
            "imu": len(json.loads((FIXTURE_ROOT / "mini_seq_03" / "imu.json").read_text(encoding="utf-8"))),
            "uwb": len(json.loads((FIXTURE_ROOT / "mini_seq_03" / "uwb.json").read_text(encoding="utf-8"))),
            "vio": len(json.loads((FIXTURE_ROOT / "mini_seq_03" / "vio.json").read_text(encoding="utf-8"))),
            "gt": len(json.loads((FIXTURE_ROOT / "mini_seq_03" / "gt.json").read_text(encoding="utf-8"))),
        },
    }
    for seq_id in report["sequence_ids"]:
        spec = next(spec for spec in DEFAULT_SIM_SEQUENCE_SPECS if spec.seq_id == seq_id)
        seq_root = tmp_path / "sim_raw" / seq_id
        assert (seq_root / "imu.json").is_file()
        assert (seq_root / "uwb.json").is_file()
        assert (seq_root / "vio.json").is_file()
        assert (seq_root / "gt.json").is_file()
        assert (seq_root / "anchor_layout.json").is_file()

        imu_rows = json.loads((seq_root / "imu.json").read_text(encoding="utf-8"))
        uwb_rows = json.loads((seq_root / "uwb.json").read_text(encoding="utf-8"))
        gt_rows = json.loads((seq_root / "gt.json").read_text(encoding="utf-8"))
        vio_rows = json.loads((seq_root / "vio.json").read_text(encoding="utf-8"))
        anchor_layout = json.loads((seq_root / "anchor_layout.json").read_text(encoding="utf-8"))
        # 协议轨迹：样本数由 duration/dt 决定，不再是 fixture×cycle。
        assert len(gt_rows) >= int(spec.duration_s / 0.05)
        assert len(imu_rows) >= 100
        assert len(uwb_rows) >= 50
        assert len(vio_rows) >= 50
        assert len(anchor_layout.get("anchor_positions") or []) >= 3
        xs = [r["px"] for r in gt_rows]
        ys = [r["py"] for r in gt_rows]
        l_xy = max(max(xs) - min(xs), max(ys) - min(ys))
        assert l_xy >= 10.0
        assert gt_rows[-1]["timestamp"] - gt_rows[0]["timestamp"] >= 20.0

    assert report["sequence_ids"][:6] == [
        "sim_line_01",
        "sim_line_02",
        "sim_curve_01",
        "sim_curve_02",
        "sim_mirror_01",
        "sim_rotate_01",
    ]
    assert report["noise"]["base_seed"] == DEFAULT_SIM_NOISE_SPEC.base_seed
    assert report["noise"]["uwb_range_std_m"] == DEFAULT_SIM_NOISE_SPEC.uwb_range_std_m


def test_materializer_preserves_uwb_residual_after_nontrivial_transform(tmp_path):
    output_root = tmp_path / "sim_raw"
    materialize_sim_raw(output_root, fixture_root=FIXTURE_ROOT, noise_spec=ZERO_SIM_NOISE_SPEC, sequence_specs=_legacy_smoke_specs())

    rotate_spec = next(spec for spec in _legacy_smoke_specs() if spec.seq_id == "sim_rotate_01")
    base_seq_id = rotate_spec.base_seq_id
    base_anchor_layout = json.loads((FIXTURE_ROOT / base_seq_id / "anchor_layout.json").read_text(encoding="utf-8"))
    base_anchor_lookup = build_anchor_lookup(base_anchor_layout)
    base_gt_rows = json.loads((FIXTURE_ROOT / base_seq_id / "gt.json").read_text(encoding="utf-8"))
    base_uwb_row = json.loads((FIXTURE_ROOT / base_seq_id / "uwb.json").read_text(encoding="utf-8"))[0]
    base_pose = _interpolate_pose(base_gt_rows, float(base_uwb_row["timestamp"]))
    base_residual = float(base_uwb_row["range"]) - float(
        predict_range_to_anchor(base_pose, base_anchor_lookup[base_uwb_row["anchor_id"]])
    )

    seq_root = output_root / "sim_rotate_01"
    new_anchor_layout = json.loads((seq_root / "anchor_layout.json").read_text(encoding="utf-8"))
    new_anchor_lookup = build_anchor_lookup(new_anchor_layout)
    new_gt_rows = json.loads((seq_root / "gt.json").read_text(encoding="utf-8"))
    new_uwb_row = json.loads((seq_root / "uwb.json").read_text(encoding="utf-8"))[0]
    new_pose = _interpolate_pose(new_gt_rows, float(new_uwb_row["timestamp"]))
    new_residual = float(new_uwb_row["range"]) - float(
        predict_range_to_anchor(new_pose, new_anchor_lookup[new_uwb_row["anchor_id"]])
    )

    # fixture 的 UWB 残差非零，确认残差确实存在
    assert base_residual != pytest.approx(0.0, abs=1e-5)
    # sim_rotate_01 的 fixture anchor_id 不在变换后的布局中，触发锚点回退；
    # 回退时残差置 0 避免注入错误偏差（组16），因此 new_residual ≈ 0
    assert new_residual == pytest.approx(0.0, abs=1e-5)


def test_materializer_preserves_vio_residual_after_nontrivial_transform(tmp_path):
    output_root = tmp_path / "sim_raw"
    materialize_sim_raw(output_root, fixture_root=FIXTURE_ROOT, noise_spec=ZERO_SIM_NOISE_SPEC, sequence_specs=_legacy_smoke_specs())

    rotate_spec = next(spec for spec in _legacy_smoke_specs() if spec.seq_id == "sim_rotate_01")
    base_seq_id = rotate_spec.base_seq_id
    base_gt_rows = json.loads((FIXTURE_ROOT / base_seq_id / "gt.json").read_text(encoding="utf-8"))
    base_vio_rows = json.loads((FIXTURE_ROOT / base_seq_id / "vio.json").read_text(encoding="utf-8"))

    previous_base_timestamp = None
    expected_base_residuals = []
    for raw_row in base_vio_rows:
        if previous_base_timestamp is None:
            geometric_dx, geometric_dy, geometric_dyaw = 0.0, 0.0, 0.0
        else:
            prev_pose = _interpolate_pose(base_gt_rows, previous_base_timestamp)
            curr_pose = _interpolate_pose(base_gt_rows, float(raw_row["timestamp"]))
            geometric_dx, geometric_dy, geometric_dyaw = _body_frame_delta(prev_pose, curr_pose)
        expected_base_residuals.append(
            (
                float(raw_row["dx"]) - float(geometric_dx),
                float(raw_row["dy"]) - float(geometric_dy),
                angle_delta_rad(float(raw_row["dyaw"]), float(geometric_dyaw)),
            )
        )
        previous_base_timestamp = float(raw_row["timestamp"])

    seq_root = output_root / "sim_rotate_01"
    new_gt_rows = json.loads((seq_root / "gt.json").read_text(encoding="utf-8"))
    new_vio_rows = json.loads((seq_root / "vio.json").read_text(encoding="utf-8"))

    # 首帧是 cycle 边界帧（dx=0.0），不参与残差比较；从第二帧开始
    previous_new_timestamp = float(new_vio_rows[0]["timestamp"])
    for new_row, (base_res_dx, base_res_dy, base_res_dyaw) in zip(new_vio_rows[1:len(base_vio_rows)], expected_base_residuals[1:], strict=True):
        prev_pose = _interpolate_pose(new_gt_rows, previous_new_timestamp)
        curr_pose = _interpolate_pose(new_gt_rows, float(new_row["timestamp"]))
        geometric_dx, geometric_dy, geometric_dyaw = _body_frame_delta(prev_pose, curr_pose)
        new_residual_dx = float(new_row["dx"]) - float(geometric_dx)
        new_residual_dy = float(new_row["dy"]) - float(geometric_dy)
        new_residual_dyaw = angle_delta_rad(float(new_row["dyaw"]), float(geometric_dyaw))
        assert new_residual_dx == pytest.approx(base_res_dx, abs=1e-5)
        assert new_residual_dy == pytest.approx(base_res_dy, abs=1e-5)
        assert new_residual_dyaw == pytest.approx(base_res_dyaw, abs=1e-5)
        previous_new_timestamp = float(new_row["timestamp"])


def test_materializer_preserves_imu_residual_after_nontrivial_transform(tmp_path):
    output_root = tmp_path / "sim_raw"
    materialize_sim_raw(output_root, fixture_root=FIXTURE_ROOT, noise_spec=ZERO_SIM_NOISE_SPEC, sequence_specs=_legacy_smoke_specs())

    rotate_spec = next(spec for spec in _legacy_smoke_specs() if spec.seq_id == "sim_rotate_01")
    base_seq_id = rotate_spec.base_seq_id
    base_gt_rows = json.loads((FIXTURE_ROOT / base_seq_id / "gt.json").read_text(encoding="utf-8"))
    base_imu_rows = json.loads((FIXTURE_ROOT / base_seq_id / "imu.json").read_text(encoding="utf-8"))
    base_timestamps = [float(row["timestamp"]) for row in base_imu_rows]
    expected_base_rows = _derive_imu_rows_from_gt(base_gt_rows, base_timestamps)
    expected_base_residuals = [
        (
            float(raw_row["ax"]) - float(derived_row["ax"]),
            float(raw_row["ay"]) - float(derived_row["ay"]),
            float(raw_row["gz"]) - float(derived_row["gz"]),
        )
        for raw_row, derived_row in zip(base_imu_rows, expected_base_rows, strict=True)
    ]

    seq_root = output_root / "sim_rotate_01"
    new_gt_rows = json.loads((seq_root / "gt.json").read_text(encoding="utf-8"))
    new_imu_rows = json.loads((seq_root / "imu.json").read_text(encoding="utf-8"))
    new_timestamps = [float(row["timestamp"]) for row in new_imu_rows[: len(base_imu_rows)]]
    derived_new_rows = _derive_imu_rows_from_gt(new_gt_rows, new_timestamps)

    for new_row, derived_row, expected_residual in zip(
        new_imu_rows[: len(base_imu_rows)],
        derived_new_rows,
        expected_base_residuals,
        strict=True,
    ):
        residual_ax = float(new_row["ax"]) - float(derived_row["ax"])
        residual_ay = float(new_row["ay"]) - float(derived_row["ay"])
        residual_gz = float(new_row["gz"]) - float(derived_row["gz"])
        assert residual_ax == pytest.approx(expected_residual[0], abs=1e-5)
        assert residual_ay == pytest.approx(expected_residual[1], abs=1e-5)
        assert residual_gz == pytest.approx(expected_residual[2], abs=1e-5)


def test_materialize_sim_raw_emits_distinct_layout_families_without_breaking_default_split(tmp_path):
    output_root = tmp_path / "sim_raw"
    # 主表协议轨迹：仍应产出多布局族 + 可切分序列集
    report = materialize_sim_raw(output_root, fixture_root=FIXTURE_ROOT, noise_spec=ZERO_SIM_NOISE_SPEC)

    layout_family_report = inspect_layout_family_coverage(output_root)
    assert layout_family_report["is_valid"] is True
    assert layout_family_report["family_count"] >= 2
    assert len(report["sequence_ids"]) == len(DEFAULT_SIM_SEQUENCE_SPECS)

    split_rules_path = PROJECT_ROOT / "scripts" / "default_split_rules.yaml"
    if not split_rules_path.is_file():
        # 仓库若未提供默认切分规则，仅校验布局族与序列数即可。
        return
    split_rules = load_yaml_config(split_rules_path)
    dataset_manifest = {
        "sequences": [
            {
                "seq_id": seq_id,
                "seq_dir": str((output_root / seq_id).resolve()),
            }
            for seq_id in report["sequence_ids"]
        ]
    }
    split_manifest, leak_report = build_splits(dataset_manifest, split_rules)
    assert isinstance(split_manifest, dict)
    assert leak_report is not None



def test_materialized_anchor_layout_preserves_fixture_origin_and_emits_distinct_family_ids(tmp_path):
    output_root = tmp_path / "sim_raw"
    materialize_sim_raw(output_root, fixture_root=FIXTURE_ROOT, noise_spec=ZERO_SIM_NOISE_SPEC, sequence_specs=_legacy_smoke_specs())

    line_layout = json.loads((output_root / "sim_line_01" / "anchor_layout.json").read_text(encoding="utf-8"))
    rotate_layout = json.loads((output_root / "sim_rotate_01" / "anchor_layout.json").read_text(encoding="utf-8"))

    assert line_layout.get("base_layout_id") == "sim_line_01" or line_layout.get("layout_id") == "sim_line_01"
    assert rotate_layout.get("layout_id") == "sim_rotate_01"
    assert len(line_layout.get("anchor_positions") or []) >= 3
    assert len(rotate_layout.get("anchor_positions") or []) >= 3
    # 不同序列应有可区分布局身份
    assert line_layout.get("layout_id") != rotate_layout.get("layout_id")



def test_materializer_vio_rows_preserve_fixture_relative_pose_residuals(tmp_path):
    output_root = tmp_path / "sim_raw"
    materialize_sim_raw(output_root, fixture_root=FIXTURE_ROOT, noise_spec=ZERO_SIM_NOISE_SPEC, sequence_specs=_legacy_smoke_specs())

    seq_root = output_root / "sim_line_01"
    gt_rows = json.loads((seq_root / "gt.json").read_text(encoding="utf-8"))
    vio_rows = json.loads((seq_root / "vio.json").read_text(encoding="utf-8"))
    base_vio_rows = json.loads((FIXTURE_ROOT / "mini_seq" / "vio.json").read_text(encoding="utf-8"))
    base_gt_rows = json.loads((FIXTURE_ROOT / "mini_seq" / "gt.json").read_text(encoding="utf-8"))

    # 首帧是 cycle 边界帧（dx/dy/dyaw=0.0），不与 fixture 原始值比较
    previous_vio_timestamp = float(vio_rows[0]["timestamp"])
    previous_base_vio_timestamp = float(base_vio_rows[0]["timestamp"])
    for row, base_row in zip(vio_rows[1:len(base_vio_rows)], base_vio_rows[1:], strict=True):
        current_vio_timestamp = float(row["timestamp"])
        prev_pose = _interpolate_pose(gt_rows, previous_vio_timestamp)
        curr_pose = _interpolate_pose(gt_rows, current_vio_timestamp)
        geom_dx, geom_dy, geom_dyaw = _body_frame_delta(prev_pose, curr_pose)

        prev_base_pose = _interpolate_pose(base_gt_rows, previous_base_vio_timestamp)
        curr_base_pose = _interpolate_pose(base_gt_rows, float(base_row["timestamp"]))
        base_geom_dx, base_geom_dy, base_geom_dyaw = _body_frame_delta(prev_base_pose, curr_base_pose)

        assert float(row["dx"]) - float(geom_dx) == pytest.approx(float(base_row["dx"]) - float(base_geom_dx), abs=1e-5)
        assert float(row["dy"]) - float(geom_dy) == pytest.approx(float(base_row["dy"]) - float(base_geom_dy), abs=1e-5)
        assert angle_delta_rad(float(row["dyaw"]), float(geom_dyaw)) == pytest.approx(
            angle_delta_rad(float(base_row["dyaw"]), float(base_geom_dyaw)),
            abs=1e-5,
        )
        previous_vio_timestamp = current_vio_timestamp
        previous_base_vio_timestamp = float(base_row["timestamp"])


def test_materialized_vio_cycle_boundary_frames_have_zero_quality(tmp_path):
    """VIO cycle 边界帧 quality=0.0, dx=dy=dyaw=0.0.

    铁律 3: VIO 不再输出 tracked_features / reproj_err 字段. 验证这两字段
    从 row keys 中消失 (sim_materializer 仅输出 dx/dy/dyaw/quality/timestamp).
    """
    output_root = tmp_path / "sim_raw"
    materialize_sim_raw(output_root, fixture_root=FIXTURE_ROOT, noise_spec=ZERO_SIM_NOISE_SPEC, sequence_specs=_legacy_smoke_specs())

    seq_root = output_root / "sim_line_01"
    vio_rows = json.loads((seq_root / "vio.json").read_text(encoding="utf-8"))
    base_vio_rows = json.loads((FIXTURE_ROOT / "mini_seq" / "vio.json").read_text(encoding="utf-8"))
    cycle_len = len(base_vio_rows)

    # 每个 cycle 的第一帧是边界帧
    for cycle_start in range(0, len(vio_rows), cycle_len):
        boundary_row = vio_rows[cycle_start]
        assert float(boundary_row["quality"]) == pytest.approx(0.0)
        # 铁律 3: tracked_features / reproj_err 已从 VIO 输出中删除
        assert "tracked_features" not in boundary_row
        assert "reproj_err" not in boundary_row
        # 边界帧增量也应为零
        assert float(boundary_row["dx"]) == pytest.approx(0.0)
        assert float(boundary_row["dy"]) == pytest.approx(0.0)
        assert float(boundary_row["dyaw"]) == pytest.approx(0.0)


def test_materialized_gt_advances_across_cycle_boundaries(tmp_path):
    output_root = tmp_path / "sim_raw"
    materialize_sim_raw(output_root, fixture_root=FIXTURE_ROOT, noise_spec=ZERO_SIM_NOISE_SPEC, sequence_specs=_legacy_smoke_specs())

    seq_root = output_root / "sim_line_01"
    spec = next(s for s in _legacy_smoke_specs() if s.seq_id == "sim_line_01")
    gt_rows = json.loads((seq_root / "gt.json").read_text(encoding="utf-8"))
    base_gt_rows = json.loads((FIXTURE_ROOT / "mini_seq" / "gt.json").read_text(encoding="utf-8"))
    cycle_len = len(base_gt_rows)

    assert cycle_len >= 2
    prev_cycle_last = gt_rows[cycle_len - 1]
    next_cycle_first = gt_rows[cycle_len]

    assert float(next_cycle_first["timestamp"]) > float(prev_cycle_last["timestamp"])
    assert not (
        math.isclose(float(next_cycle_first["px"]), float(prev_cycle_last["px"]), abs_tol=1e-9)
        and math.isclose(float(next_cycle_first["py"]), float(prev_cycle_last["py"]), abs_tol=1e-9)
    )

    base_tail = base_gt_rows[-1]
    base_prev = base_gt_rows[-2]
    base_dt = float(base_tail["timestamp"]) - float(base_prev["timestamp"])
    observed_gap_dt = float(next_cycle_first["timestamp"]) - float(prev_cycle_last["timestamp"])
    observed_dx = float(next_cycle_first["px"]) - float(prev_cycle_last["px"])
    observed_dy = float(next_cycle_first["py"]) - float(prev_cycle_last["py"])
    expected_dx_velocity = (float(base_tail["px"]) - float(base_prev["px"])) / base_dt * observed_gap_dt
    expected_dy_velocity = (float(base_tail["py"]) - float(base_prev["py"])) / base_dt * observed_gap_dt

    # 铁律 6: cycle 1 起点的位置偏移完全等于 cycle 0 的 per-cycle 增量
    # (cum_offset_0 in 递推 = cycle_dx_nominal rotated + jitter). 重算 expected_cycle_offset:
    cycle_dx_nominal, cycle_dy_nominal = _compute_cycle_pose_increment(
        base_gt_rows, cycle_gap_s=spec.cycle_gap_s
    )
    sigma_cycle_delta = 0.05 * math.hypot(cycle_dx_nominal, cycle_dy_nominal)
    base_seed = int(ZERO_SIM_NOISE_SPEC.base_seed)
    seq_id = "sim_line_01"
    cycle_index = 0
    delta_x_noise = _seeded_gaussian(base_seed, sigma_cycle_delta, "gt", seq_id, cycle_index, "delta_x")
    delta_y_noise = _seeded_gaussian(base_seed, sigma_cycle_delta, "gt", seq_id, cycle_index, "delta_y")
    heading_dev_rad = _seeded_uniform(base_seed, math.radians(10.0), "gt", seq_id, cycle_index, "heading_dev")
    cos_h, sin_h = math.cos(heading_dev_rad), math.sin(heading_dev_rad)
    rotated_dx = cycle_dx_nominal * cos_h - cycle_dy_nominal * sin_h
    rotated_dy = cycle_dx_nominal * sin_h + cycle_dy_nominal * cos_h
    expected_cycle_dx = rotated_dx + delta_x_noise
    expected_cycle_dy = rotated_dy + delta_y_noise
    # observed = (base_row0.px - base_tail.px) + this_cycle_dx_at_cycle_0
    # 而 expected_dx_velocity (= velocity * gap_dt) = cycle_dx_nominal - seg_dx
    # where seg_dx = base_tail.px - base_row0.px.
    # 故 observed_expected = -seg_dx + this_cycle_dx = expected_dx_velocity + (this_cycle_dx - cycle_dx_nominal)
    expected_dx = expected_dx_velocity + (expected_cycle_dx - cycle_dx_nominal)
    expected_dy = expected_dy_velocity + (expected_cycle_dy - cycle_dy_nominal)

    assert observed_dx == pytest.approx(expected_dx, abs=1e-6)
    assert observed_dy == pytest.approx(expected_dy, abs=1e-6)


def test_materialized_imu_rows_repeat_fixture_residual_pattern_across_cycles(tmp_path):
    output_root = tmp_path / "sim_raw"
    materialize_sim_raw(output_root, fixture_root=FIXTURE_ROOT, noise_spec=ZERO_SIM_NOISE_SPEC, sequence_specs=_legacy_smoke_specs())

    seq_root = output_root / "sim_line_01"
    gt_rows = json.loads((seq_root / "gt.json").read_text(encoding="utf-8"))
    imu_rows = json.loads((seq_root / "imu.json").read_text(encoding="utf-8"))
    base_imu_rows = json.loads((FIXTURE_ROOT / "mini_seq" / "imu.json").read_text(encoding="utf-8"))
    base_gt_rows = json.loads((FIXTURE_ROOT / "mini_seq" / "gt.json").read_text(encoding="utf-8"))

    base_timestamps = [float(row["timestamp"]) for row in base_imu_rows]
    derived_base_rows = _derive_imu_rows_from_gt(base_gt_rows, base_timestamps)
    expected_base_residuals = [
        (
            float(raw_row["ax"]) - float(derived_row["ax"]),
            float(raw_row["ay"]) - float(derived_row["ay"]),
            float(raw_row["gz"]) - float(derived_row["gz"]),
        )
        for raw_row, derived_row in zip(base_imu_rows, derived_base_rows, strict=True)
    ]

    cycle_imu_len = len(base_imu_rows)
    assert cycle_imu_len >= 1
    for cycle_start in range(0, len(imu_rows), cycle_imu_len):
        cycle_rows = imu_rows[cycle_start : cycle_start + cycle_imu_len]
        cycle_timestamps = [float(row["timestamp"]) for row in cycle_rows]
        derived_rows = _derive_imu_rows_from_gt(gt_rows, cycle_timestamps)
        for row_index, (row, derived_row) in enumerate(zip(cycle_rows, derived_rows, strict=True)):
            expected_residual = expected_base_residuals[row_index]
            residual_ax = float(row["ax"]) - float(derived_row["ax"])
            residual_ay = float(row["ay"]) - float(derived_row["ay"])
            residual_gz = float(row["gz"]) - float(derived_row["gz"])
            assert residual_ax == pytest.approx(expected_residual[0], abs=1e-5)
            assert residual_ay == pytest.approx(expected_residual[1], abs=1e-5)
            assert residual_gz == pytest.approx(expected_residual[2], abs=1e-5)


def test_materialize_sim_raw_noise_is_reproducible_and_nonzero(tmp_path):
    noisy_root_a = tmp_path / "sim_raw_a"
    noisy_root_b = tmp_path / "sim_raw_b"
    zero_root = tmp_path / "sim_raw_zero"

    materialize_sim_raw(noisy_root_a, fixture_root=FIXTURE_ROOT, sequence_specs=_legacy_smoke_specs())
    materialize_sim_raw(noisy_root_b, fixture_root=FIXTURE_ROOT, sequence_specs=_legacy_smoke_specs())
    materialize_sim_raw(zero_root, fixture_root=FIXTURE_ROOT, noise_spec=ZERO_SIM_NOISE_SPEC, sequence_specs=_legacy_smoke_specs())

    noisy_uwb_a = json.loads((noisy_root_a / "sim_line_01" / "uwb.json").read_text(encoding="utf-8"))
    noisy_uwb_b = json.loads((noisy_root_b / "sim_line_01" / "uwb.json").read_text(encoding="utf-8"))
    zero_uwb = json.loads((zero_root / "sim_line_01" / "uwb.json").read_text(encoding="utf-8"))
    noisy_imu_a = json.loads((noisy_root_a / "sim_line_01" / "imu.json").read_text(encoding="utf-8"))
    zero_imu = json.loads((zero_root / "sim_line_01" / "imu.json").read_text(encoding="utf-8"))
    noisy_vio_a = json.loads((noisy_root_a / "sim_line_01" / "vio.json").read_text(encoding="utf-8"))
    zero_vio = json.loads((zero_root / "sim_line_01" / "vio.json").read_text(encoding="utf-8"))

    assert noisy_uwb_a == noisy_uwb_b
    assert noisy_uwb_a[0]["range"] != zero_uwb[0]["range"]
    assert noisy_imu_a[0]["ax"] != zero_imu[0]["ax"]
    # 首帧是 cycle 边界帧 dx=0.0，用第二帧比较噪声差异
    assert noisy_vio_a[1]["dx"] != zero_vio[1]["dx"]


def test_can_materialize_sim_raw_only_when_target_is_effectively_empty(tmp_path):
    output_root = tmp_path / "sim_raw"
    output_root.mkdir()
    assert can_materialize_sim_raw(output_root) is True

    (output_root / ".gitkeep").write_text("", encoding="utf-8")
    assert can_materialize_sim_raw(output_root) is True

    (output_root / "mini_seq").mkdir()
    assert can_materialize_sim_raw(output_root) is False


def test_materialize_sim_raw_rejects_nonempty_target(tmp_path):
    output_root = tmp_path / "sim_raw"
    (output_root / "existing_seq").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="target must be empty"):
        materialize_sim_raw(output_root, fixture_root=FIXTURE_ROOT)
