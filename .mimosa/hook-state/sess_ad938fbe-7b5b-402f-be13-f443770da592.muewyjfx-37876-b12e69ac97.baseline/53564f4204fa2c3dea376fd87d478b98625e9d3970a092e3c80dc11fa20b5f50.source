from __future__ import annotations

"""任务合同（task_contract）测试模块。

测试覆盖范围：
- 任务定义的必填字段验证
- 任务 ID/场景 ID/序列 ID 的约束
- 任务参数的合同一致性

被测模块：liquidloc.protocol.task_contract"""

import pytest

from liquidloc.common.config_utils import find_project_root
from liquidloc.protocol.task_contract import (
    get_sensor_roles,
    get_state_definition,
    get_state_items,
    get_task_name,
    get_vio_update_contract,
    load_task_contract,
)

_PROJECT_ROOT = find_project_root()


def _write_task_contract(
    tmp_path,
    *,
    task_name: str = "uwb_imu_vio_localization",
    frame: str = "planar_xy_yaw",
    state_items: str = "[px, py, vx, vy, yaw, bax, bay, bg, uwb_clock_bias, vio_scale]",
    uwb_role: str = "absolute_range_constraint",
    imu_role: str = "high_rate_propagation",
    vio_role: str = "relative_pose_constraint",
    measurement_items: str = "[dx, dy, dyaw]",
    updated_state_items: str = "[px, py, yaw, uwb_clock_bias, vio_scale]",
    learned_control_entry: str = "noise_multiplier",
    forbid_rewrite: str = "true",
):
    # Use project-root-relative temp dir to satisfy path security check
    tmp_dir = _PROJECT_ROOT / "outputs" / "_test_tmp" / "task_contract"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    config_path = tmp_dir / "task.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"task_name: {task_name}",
                "state_definition:",
                f"  frame: {frame}",
                f"  state_items: {state_items}",
                "sensor_roles:",
                f"  uwb: {uwb_role}",
                f"  imu: {imu_role}",
                f"  vio: {vio_role}",
                "vio_update_contract:",
                f"  measurement_items: {measurement_items}",
                f"  updated_state_items: {updated_state_items}",
                f"  learned_control_entry: {learned_control_entry}",
                f"  forbid_rewrite_measurement_fields_inside_update: {forbid_rewrite}",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def test_task_contract_matches_frozen_yaml_defaults():
    """匹配测试：task contract。\n\n验证 task contract 的输出与预期一致，\n确保合同合规。
    """
    assert get_task_name() == "uwb_imu_vio_localization"
    assert get_state_items() == ("px", "py", "vx", "vy", "yaw", "bax", "bay", "bg", "uwb_clock_bias", "vio_scale")
    assert get_state_definition() == {
        "frame": "planar_xy_yaw",
        "state_items": ("px", "py", "vx", "vy", "yaw", "bax", "bay", "bg", "uwb_clock_bias", "vio_scale"),
    }
    assert get_sensor_roles() == {
        "uwb": "absolute_range_constraint",
        "imu": "high_rate_propagation",
        "vio": "relative_pose_constraint",
    }
    assert get_vio_update_contract() == {
        "measurement_items": ("dx", "dy", "dyaw"),
        "updated_state_items": ("px", "py", "yaw", "uwb_clock_bias", "vio_scale"),
        "learned_control_entry": "noise_multiplier",
        "forbid_rewrite_measurement_fields_inside_update": True,
    }


def test_task_contract_rejects_wrong_task_name(tmp_path):
    """拒绝测试：task contract。\n\n验证被测功能对 task contract 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    config_path = _write_task_contract(tmp_path, task_name="other_task")
    with pytest.raises(ValueError, match="task_name must be"):
        load_task_contract(config_path)


def test_task_contract_rejects_wrong_frame(tmp_path):
    """拒绝测试：task contract。\n\n验证被测功能对 task contract 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    config_path = _write_task_contract(tmp_path, frame="full_3d")
    with pytest.raises(ValueError, match="state_definition.frame must be"):
        load_task_contract(config_path)


def test_task_contract_rejects_wrong_state_items(tmp_path):
    """拒绝测试：task contract。\n\n验证被测功能对 task contract 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    config_path = _write_task_contract(tmp_path, state_items="[px, py, yaw]")
    with pytest.raises(ValueError, match="state_definition.state_items must be"):
        load_task_contract(config_path)


def test_task_contract_rejects_duplicate_state_items(tmp_path):
    """拒绝测试：task contract。\n\n验证被测功能对 task contract 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    config_path = _write_task_contract(tmp_path, state_items="[px, py, vx, vy, yaw, bax, bay, px]")
    with pytest.raises(ValueError, match="state_definition.state_items must not contain duplicate entries"):
        load_task_contract(config_path)


def test_task_contract_rejects_sensor_role_drift(tmp_path):
    """拒绝测试：task contract。\n\n验证被测功能对 task contract 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    config_path = _write_task_contract(tmp_path, vio_role="pose_constraint")
    with pytest.raises(ValueError, match="sensor_roles.vio must be"):
        load_task_contract(config_path)


def test_task_contract_rejects_vio_measurement_items_drift(tmp_path):
    """拒绝测试：task contract。\n\n验证被测功能对 task contract 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    config_path = _write_task_contract(tmp_path, measurement_items="[dx, dy]")
    with pytest.raises(ValueError, match="vio_update_contract.measurement_items must be"):
        load_task_contract(config_path)


def test_task_contract_rejects_vio_updated_state_items_drift(tmp_path):
    """拒绝测试：task contract。\n\n验证被测功能对 task contract 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    config_path = _write_task_contract(tmp_path, updated_state_items="[px, py, vx]")
    with pytest.raises(ValueError, match="vio_update_contract.updated_state_items must be"):
        load_task_contract(config_path)


def test_task_contract_rejects_vio_control_entry_drift(tmp_path):
    """拒绝测试：task contract。\n\n验证被测功能对 task contract 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    config_path = _write_task_contract(tmp_path, learned_control_entry="residual_scale")
    with pytest.raises(ValueError, match="vio_update_contract.learned_control_entry must be"):
        load_task_contract(config_path)


def test_task_contract_rejects_vio_forbid_rewrite_drift(tmp_path):
    """拒绝测试：task contract。\n\n验证被测功能对 task contract 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    config_path = _write_task_contract(tmp_path, forbid_rewrite="false")
    with pytest.raises(ValueError, match="forbid_rewrite_measurement_fields_inside_update"):
        load_task_contract(config_path)
