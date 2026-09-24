from __future__ import annotations

"""常量注册表（constants）测试模块。

文件职责：验证常量注册表的完整性和一致性。

测试覆盖范围：
- 正常场景：注册表完整且无合同错误
- payload 键顺序无关
- 异常场景：PAYLOAD_KEYS 与 ALLOWED_MODALITIES 不匹配

被测模块：liquidloc.common.constants"""


import liquidloc.common.constants as constants
from liquidloc.common.constants import (
    ALLOWED_MODALITIES,
    DEFAULT_OUTPUT_DIRS,
    validate_constant_registry,
)


def test_normal_case():
    report = validate_constant_registry()
    assert report["is_complete"] is True
    assert report["contract_errors"] == []
    assert set(ALLOWED_MODALITIES) == {"imu", "uwb", "vio", "flow", "tof"}
    assert "metrics" in DEFAULT_OUTPUT_DIRS


def test_payload_key_order_does_not_matter(monkeypatch):
    monkeypatch.setattr(
        constants,
        "PAYLOAD_KEYS",
        {"vio": "vio_payload", "imu": "imu_payload", "uwb": "uwb_payload", "flow": "flow_payload", "tof": "tof_payload"},
    )
    report = validate_constant_registry()
    assert report["missing_names"] == []
    assert report["contract_errors"] == []
    assert report["is_complete"] is True


def test_invalid_case(monkeypatch):
    monkeypatch.setattr(
        constants,
        "PAYLOAD_KEYS",
        {"imu": "imu_payload", "uwb": "uwb_payload"},
    )
    report = validate_constant_registry()
    assert report["missing_names"] == []
    assert report["is_complete"] is False
    assert any("payload_keys_do_not_match_allowed_modalities" in err for err in report["contract_errors"])
