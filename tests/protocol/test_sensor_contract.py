from __future__ import annotations

"""传感器合同（sensor_contract）测试模块。

测试覆盖范围：
- IMU/UWB/VIO 传感器数据的字段合同
- 时间戳/质量/有效性等必填字段
- 传感器载荷的结构验证

被测模块：liquidloc.protocol.sensor_contract"""

import pytest

from liquidloc.common.constants import MODALITY_IMU, MODALITY_UWB, MODALITY_VIO
from liquidloc.common.config_utils import find_project_root
from liquidloc.protocol.sensor_contract import (
    get_feature_missing_policy,
    get_required_payload_fields,
    load_sensor_contract,
)

_PROJECT_ROOT = find_project_root()


def test_sensor_contract_matches_frozen_yaml_defaults():
    """匹配测试：sensor contract。\n\n验证 sensor contract 的输出与预期一致，\n确保合同合规。
    """
    required_fields = get_required_payload_fields()
    missing_policy = get_feature_missing_policy()

    assert required_fields[MODALITY_IMU] == ("ax", "ay", "gz")
    assert required_fields[MODALITY_UWB] == ("anchor_id", "range", "valid", "quality")
    assert required_fields[MODALITY_VIO] == ("dx", "dy", "dyaw", "quality")
    assert missing_policy["source_precedence"] == ("event_root", "modality_payload", "state_ctx")
    assert missing_policy["numeric_fill_value"] == pytest.approx(0.0)
    assert missing_policy["require_missing_mask"] is True
    assert missing_policy["missing_semantics_carrier"] == "missing_mask"


def test_sensor_contract_rejects_duplicate_payload_fields(tmp_path):
    """拒绝测试：sensor contract。\n\n验证被测功能对 sensor contract 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    tmp_dir = _PROJECT_ROOT / "outputs" / "_test_tmp" / "sensor_contract"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    config_path = tmp_dir / "sensors.yaml"
    config_path.write_text(
        "\n".join(
            [
                "imu_fields: [ax, ax, gz]",
                "uwb_fields: [anchor_id, range, valid, quality]",
                "vio_fields: [dx, dy, dyaw, quality, tracked_features, reproj_err]",
                "feature_missing_policy:",
                "  source_precedence: [event_root, modality_payload, state_ctx]",
                "  numeric_fill_value: 0.0",
                "  require_missing_mask: true",
                "  missing_semantics_carrier: missing_mask",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="imu_fields must not contain duplicate fields"):
        load_sensor_contract(config_path)


def test_sensor_contract_requires_missing_mask_carrier(tmp_path):
    """合同测试：sensor。\n\n验证 sensor 的接口合同，\n确保输入输出符合协议约定。
    """
    tmp_dir = _PROJECT_ROOT / "outputs" / "_test_tmp" / "sensor_contract"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    config_path = tmp_dir / "sensors.yaml"
    config_path.write_text(
        "\n".join(
            [
                "imu_fields: [ax, ay, gz]",
                "uwb_fields: [anchor_id, range, valid, quality]",
                "vio_fields: [dx, dy, dyaw, quality, tracked_features, reproj_err]",
                "flow_fields: [dx, dy, quality]",
                "tof_fields: [range, quality]",
                "feature_missing_policy:",
                "  source_precedence: [event_root, modality_payload, state_ctx]",
                "  numeric_fill_value: 0.0",
                "  require_missing_mask: true",
                "  missing_semantics_carrier: nan",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing_semantics_carrier must be one of"):
        load_sensor_contract(config_path)
