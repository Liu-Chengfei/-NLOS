"""Channel Impulse Response (CIR) 统计特征单元测试。

文件职责：验证 cir_features 模块中 CIR 特征计算的正确性，
使用构造的 LOS/NLOS CIR 模型进行核对。

测试覆盖范围：
- compute_cir_features: 首径幅度、峰值幅度、延迟扩展正确性
- generate_los_cir: LOS CIR 生成符合预期特征
- generate_nlos_cir: NLOS CIR 生成符合预期特征
- validate_cir_features_array: 形状与取值区间断言
- compute_all_cir_features: 批量计算与单帧一致
- SimSequenceSpec cir_features 字段校验
- materialize_sim_raw cir_features=on 时输出 npz
"""

from __future__ import annotations

import numpy as np
import pytest

from liquidloc.dataio.sim_materializer import (
    SimSequenceSpec,
    materialize_sim_raw,
)
from liquidloc.sensors.cir_features import (
    compute_cir_features,
    generate_los_cir,
    generate_nlos_cir,
    validate_cir_features_array,
    compute_all_cir_features,
)


FIXTURE_ROOT = "tests/fixtures/datasets/miluv"


class TestComputeCirFeatures:
    """测试 compute_cir_features 单帧特征计算。"""

    def test_los_like_cir_has_high_first_path_and_low_delay_spread(self):
        """LOS 型 CIR：首径占主导，延迟扩展较小。"""
        cir = generate_los_cir(length=64, first_path_amp=0.85, seed=42)
        fp, pk, ds = compute_cir_features(cir)

        assert 0.0 <= fp <= 1.0, f"first_path_amplitude out of [0,1]: {fp}"
        assert 0.0 <= pk <= 1.0, f"peak_amplitude out of [0,1]: {pk}"
        assert ds >= 0.0, f"delay_spread must be >= 0, got {ds}"
        # LOS: 首径幅度较高
        assert fp >= 0.5, f"LOS CIR should have high first_path, got {fp}"
        # 峰值应为最大幅度
        assert pk >= fp, f"peak_amplitude {pk} should be >= first_path {fp}"

    def test_nlos_like_cir_has_low_first_path_and_high_delay_spread(self):
        """NLOS 型 CIR：首径较弱，延迟扩展较大。"""
        cir = generate_nlos_cir(length=64, first_path_amp=0.2, seed=42)
        fp, pk, ds = compute_cir_features(cir)

        assert 0.0 <= fp <= 1.0
        assert 0.0 <= pk <= 1.0
        assert ds >= 0.0
        # NLOS: 首径幅度较低
        assert fp < 0.5, f"NLOS CIR should have low first_path, got {fp}"

    def test_los_nlos_first_path_contrast(self):
        """LOS vs NLOS：LOS 首径幅度应显著高于 NLOS。"""
        los_cir = generate_los_cir(length=64, seed=100)
        nlos_cir = generate_nlos_cir(length=64, seed=100)
        los_fp, _, _ = compute_cir_features(los_cir)
        nlos_fp, _, _ = compute_cir_features(nlos_cir)
        assert los_fp > nlos_fp, f"LOS first_path ({los_fp}) should be > NLOS ({nlos_fp})"

    def test_los_nlos_delay_spread_contrast(self):
        """LOS vs NLOS：NLOS 延迟扩展应显著高于 LOS。"""
        los_cir = generate_los_cir(length=64, seed=100)
        nlos_cir = generate_nlos_cir(length=64, seed=100)
        _, _, los_ds = compute_cir_features(los_cir)
        _, _, nlos_ds = compute_cir_features(nlos_cir)
        assert nlos_ds > los_ds, f"NLOS delay_spread ({nlos_ds}) should be > LOS ({los_ds})"

    def test_empty_cir_returns_zeros(self):
        """空 CIR 返回全零。"""
        cir = np.array([], dtype=np.float32)
        fp, pk, ds = compute_cir_features(cir)
        assert fp == 0.0 and pk == 0.0 and ds == 0.0

    def test_constant_cir(self):
        """恒定 CIR：首径 = 峰值，延迟扩展 = 0（单点等效）。"""
        cir = np.zeros(64, dtype=np.float32)
        cir[0] = 1.0  # 单峰 = 常数等效，延迟扩展为 0
        fp, pk, ds = compute_cir_features(cir)
        assert fp == 1.0
        assert pk == 1.0
        assert ds == 0.0

    def test_single_peak_cir(self):
        """单峰 CIR：首径 = 峰值 = 单峰幅度。"""
        cir = np.zeros(64, dtype=np.float32)
        cir[16] = 0.7
        cir = cir / np.max(cir)
        fp, pk, ds = compute_cir_features(cir)
        assert fp == 1.0
        assert pk == 1.0
        assert ds >= 0.0

    def test_2d_cir_raises_value_error(self):
        """2D CIR 应抛出 ValueError。"""
        cir = np.zeros((10, 64), dtype=np.float32)
        with pytest.raises(ValueError, match="must be 1D"):
            compute_cir_features(cir)

    def test_threshold_affects_first_path(self):
        """首径阈值影响：提高阈值可能跳过弱首径，改用峰值。"""
        # 低首径 CIR
        cir = np.zeros(64, dtype=np.float32)
        cir[4] = 0.005  # 弱首径
        cir[20] = 0.8  # 强多径
        cir = cir / np.max(cir)
        # 默认阈值 0.01：弱首径 < 阈值，首径用峰值位置
        fp1, pk1, _ = compute_cir_features(cir, amplitude_threshold=0.01)
        # 低阈值 0.001：弱首径 > 阈值
        fp2, _, _ = compute_cir_features(cir, amplitude_threshold=0.001)
        assert fp1 == 1.0  # 用峰值位置
        assert fp2 == cir[4]  # 检测到弱首径


class TestValidateCirFeaturesArray:
    """测试 validate_cir_features_array 形状与取值区间断言。"""

    def test_valid_los_features_pass(self):
        """有效 LOS 特征通过验证。"""
        features = np.array([
            [0.85, 0.85, 2.1],
            [0.80, 0.90, 3.5],
            [0.75, 0.82, 1.8],
        ], dtype=np.float32)
        assert validate_cir_features_array(features) is True
        assert features.shape == (3, 3)

    def test_valid_nlos_features_pass(self):
        """有效 NLOS 特征通过验证。"""
        features = np.array([
            [0.15, 0.60, 12.3],
            [0.20, 0.65, 15.1],
            [0.10, 0.55, 10.8],
        ], dtype=np.float32)
        assert validate_cir_features_array(features) is True

    def test_wrong_dimension_raises(self):
        """非 2D 数组抛出 ValueError。"""
        with pytest.raises(ValueError, match="must be 2D"):
            validate_cir_features_array(np.array([0.5, 0.6, 0.7]))

    def test_wrong_second_dim_raises(self):
        """第二维非 3 抛出 ValueError。"""
        with pytest.raises(ValueError, match="second dim must be 3"):
            validate_cir_features_array(np.zeros((10, 2)))

    def test_empty_array_raises(self):
        """空数组抛出 ValueError。"""
        with pytest.raises(ValueError, match="at least 1 frame"):
            validate_cir_features_array(np.zeros((0, 3), dtype=np.float32))

    def test_expected_frames_mismatch_raises(self):
        """expected_frames 不匹配抛出 ValueError。"""
        features = np.zeros((5, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="frame count mismatch"):
            validate_cir_features_array(features, expected_frames=10)

    def test_first_path_out_of_range_raises(self):
        """first_path_amplitude 超出 [0,1] 抛出 ValueError。"""
        features = np.array([[1.5, 0.8, 2.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="first_path_amplitude out of range"):
            validate_cir_features_array(features)

    def test_negative_delay_spread_raises(self):
        """delay_spread 为负抛出 ValueError。"""
        features = np.array([[0.5, 0.8, -1.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="delay_spread must be >= 0"):
            validate_cir_features_array(features)

    def test_valid_with_expected_frames(self):
        """指定 expected_frames 时通过。"""
        features = np.array([[0.7, 0.9, 3.0]], dtype=np.float32)
        assert validate_cir_features_array(features, expected_frames=1) is True


class TestComputeAllCirFeatures:
    """测试 compute_all_cir_features 批量计算。"""

    def test_batch_matches_single(self):
        """批量计算与单帧计算结果一致。"""
        n_frames = 10
        cir_array = np.zeros((n_frames, 64), dtype=np.float32)
        for i in range(n_frames):
            cir_array[i] = generate_los_cir(length=64, seed=i * 7)

        batched = compute_all_cir_features(cir_array)
        assert batched.shape == (n_frames, 3)

        for i in range(n_frames):
            single = compute_cir_features(cir_array[i])
            np.testing.assert_allclose(batched[i], single, rtol=1e-5)

    def test_los_batch_stats(self):
        """LOS 批次的统计特征符合预期。"""
        n_frames = 5
        cir_array = np.zeros((n_frames, 64), dtype=np.float32)
        for i in range(n_frames):
            cir_array[i] = generate_los_cir(length=64, seed=i * 13)
        batched = compute_all_cir_features(cir_array)
        assert np.all(batched[:, 0] >= 0.5)  # LOS 首径较高
        assert np.all(batched[:, 1] >= batched[:, 0])  # 峰值 >= 首径
        assert np.all(batched[:, 2] >= 0)  # 延迟扩展非负


class TestSimSequenceSpecCirFeatures:
    """测试 SimSequenceSpec cir_features 字段校验。"""

    def test_cir_features_default_is_off(self):
        """默认 cir_features 为 'off'。"""
        spec = SimSequenceSpec(
            "test_seq", "mini_seq",
            translation_xy=(0.0, 0.0),
        )
        assert spec.cir_features == "off"

    def test_cir_features_on_accepted(self):
        """cir_features='on' 接受。"""
        spec = SimSequenceSpec(
            "test_seq", "mini_seq",
            translation_xy=(0.0, 0.0),
            cir_features="on",
        )
        assert spec.cir_features == "on"

    def test_cir_features_off_accepted(self):
        """cir_features='off' 接受。"""
        spec = SimSequenceSpec(
            "test_seq", "mini_seq",
            translation_xy=(0.0, 0.0),
            cir_features="off",
        )
        assert spec.cir_features == "off"

    def test_cir_features_invalid_raises(self):
        """无效 cir_features 值抛出 ValueError。"""
        with pytest.raises(ValueError, match="must be 'on' or 'off'"):
            SimSequenceSpec(
                "test_seq", "mini_seq",
                translation_xy=(0.0, 0.0),
                cir_features="maybe",
            )

    def test_cir_features_case_insensitive(self):
        """大小写不敏感：'ON' → 'on'。"""
        spec = SimSequenceSpec(
            "test_seq", "mini_seq",
            translation_xy=(0.0, 0.0),
            cir_features="ON",
        )
        assert spec.cir_features == "on"


class TestMaterializeSimRawWithCirFeatures:
    """测试 materialize_sim_raw 在 cir_features=on 时的输出。"""

    def test_cir_features_off_no_npz(self, tmp_path):
        """cir_features=off 时不生成 cir_features.npz。"""
        report = materialize_sim_raw(
            tmp_path / "sim_off",
            fixture_root=FIXTURE_ROOT,
            sequence_specs=(
                SimSequenceSpec(
                    "cir_off_test", "mini_seq",
                    translation_xy=(0.0, 0.0),
                    cir_features="off",
                    use_protocol_trajectory=True,
                    duration_s=60.0,
                    workspace_span_m=20.0,
                ),
            ),
        )
        assert report["status"] == "ok"
        seq_root = tmp_path / "sim_off" / "cir_off_test"
        assert (seq_root / "cir_features.npz").is_file() is False
        assert report["sequences"]["cir_off_test"]["cir_features"]["enabled"] is False

    def test_cir_features_on_generates_npz(self, tmp_path):
        """cir_features=on 时生成 cir_features.npz。"""
        report = materialize_sim_raw(
            tmp_path / "sim_on",
            fixture_root=FIXTURE_ROOT,
            sequence_specs=(
                SimSequenceSpec(
                    "cir_on_test", "mini_seq",
                    translation_xy=(0.0, 0.0),
                    cir_features="on",
                    use_protocol_trajectory=True,
                    duration_s=60.0,
                    workspace_span_m=20.0,
                    axes_override=(("N", "N0"),),
                ),
            ),
        )
        assert report["status"] == "ok"
        seq_root = tmp_path / "sim_on" / "cir_on_test"
        npz_path = seq_root / "cir_features.npz"
        assert npz_path.is_file(), f"cir_features.npz not found at {npz_path}"

        data = np.load(npz_path)
        n = len(data["first_path_amplitude"])
        assert n >= 1
        assert data["first_path_amplitude"].shape == (n,)
        assert data["peak_amplitude"].shape == (n,)
        assert data["delay_spread"].shape == (n,)

        validate_cir_features_array(
            np.column_stack([
                data["first_path_amplitude"],
                data["peak_amplitude"],
                data["delay_spread"],
            ]),
            expected_frames=n,
        )

        # 检查 sim_meta.json 标注
        import json as _json
        meta_path = seq_root / "sim_meta.json"
        assert meta_path.is_file()
        meta = _json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta.get("cir_features_enabled") is True

    def test_cir_features_reproducible(self, tmp_path):
        """同 seed 同配置，CIR 特征可复现。"""
        report1 = materialize_sim_raw(
            tmp_path / "sim_rep1",
            fixture_root=FIXTURE_ROOT,
            sequence_specs=(
                SimSequenceSpec(
                    "cir_rep", "mini_seq",
                    translation_xy=(0.0, 0.0),
                    cir_features="on",
                    use_protocol_trajectory=True,
                    duration_s=60.0,
                    workspace_span_m=20.0,
                    axes_override=(("N", "N0"),),
                ),
            ),
        )
        report2 = materialize_sim_raw(
            tmp_path / "sim_rep2",
            fixture_root=FIXTURE_ROOT,
            sequence_specs=(
                SimSequenceSpec(
                    "cir_rep", "mini_seq",
                    translation_xy=(0.0, 0.0),
                    cir_features="on",
                    use_protocol_trajectory=True,
                    duration_s=60.0,
                    workspace_span_m=20.0,
                    axes_override=(("N", "N0"),),
                ),
            ),
        )
        assert report1["status"] == "ok" and report2["status"] == "ok"

        npz1 = np.load(tmp_path / "sim_rep1" / "cir_rep" / "cir_features.npz")
        npz2 = np.load(tmp_path / "sim_rep2" / "cir_rep" / "cir_features.npz")

        np.testing.assert_array_equal(npz1["first_path_amplitude"], npz2["first_path_amplitude"])
        np.testing.assert_array_equal(npz1["peak_amplitude"], npz2["peak_amplitude"])
        np.testing.assert_array_equal(npz1["delay_spread"], npz2["delay_spread"])

    def test_cir_features_nlos_level_affects_delay_spread(self, tmp_path):
        """N0 vs N3：N3 序列延迟扩展应更大。"""
        def _run(nlos_level):
            report = materialize_sim_raw(
                tmp_path / f"sim_n{int(nlos_level[1:])}",
                fixture_root=FIXTURE_ROOT,
                sequence_specs=(
                    SimSequenceSpec(
                        f"cir_n{nlos_level}", "mini_seq",
                        translation_xy=(0.0, 0.0),
                        cir_features="on",
                        use_protocol_trajectory=True,
                        duration_s=60.0,
                        workspace_span_m=20.0,
                        axes_override=(("N", nlos_level),),
                    ),
                ),
            )
            return report

        report_n0 = _run("N0")
        report_n3 = _run("N3")
        assert report_n0["status"] == "ok" and report_n3["status"] == "ok"

        ds_n0 = np.load(tmp_path / "sim_n0" / "cir_nN0" / "cir_features.npz")["delay_spread"]
        ds_n3 = np.load(tmp_path / "sim_n3" / "cir_nN3" / "cir_features.npz")["delay_spread"]

        # N3 的平均延迟扩展应显著大于 N0
        assert ds_n3.mean() > ds_n0.mean(), (
            f"N3 delay_spread mean ({ds_n3.mean():.4f}) should be > N0 ({ds_n0.mean():.4f})"
        )
