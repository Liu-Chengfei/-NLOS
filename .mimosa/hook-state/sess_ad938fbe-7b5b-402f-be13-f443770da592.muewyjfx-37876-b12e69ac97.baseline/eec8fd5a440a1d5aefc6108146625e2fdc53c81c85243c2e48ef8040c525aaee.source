from __future__ import annotations

"""质量模型（quality_model）测试模块。

文件职责：验证 normalize_quality_value 和 get_event_quality
能正确归一化和提取事件质量值。

测试覆盖范围：
- normalize_quality_value：正常值、None 回退、裁剪、类型拒绝、NaN/inf 拒绝
- get_event_quality：UWB/VIO 字典和对象事件、模态校验、payload 缺失、
  quality 类型非法、返回类型

被测模块：liquidloc.sensors.quality_model"""


from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from liquidloc.sensors.quality_model import get_event_quality, normalize_quality_value


# ── 辅助工厂 ──────────────────────────────────────────────────────────


def _uwb_event(*, quality: Any, anchor_id: str = "a0", range_val: float = 2.4) -> dict:
    """构造 UWB 字典事件。"""
    return {
        "modality": "uwb",
        "uwb_payload": {
            "anchor_id": anchor_id,
            "range": range_val,
            "valid": True,
            "quality": quality,
        },
    }


def _vio_event(*, quality: Any, dx: float = 0.1, dy: float = -0.2, dyaw: float = 0.03,
               tracked_features: int = 24, reproj_err: float = 0.4) -> dict:
    """构造 VIO 字典事件。"""
    return {
        "modality": "vio",
        "vio_payload": {
            "dx": dx,
            "dy": dy,
            "dyaw": dyaw,
            "quality": quality,
            "tracked_features": tracked_features,
            "reproj_err": reproj_err,
        },
    }


def _uwb_object_event(*, quality: Any, anchor_id: str = "a0", range_val: float = 2.4) -> SimpleNamespace:
    """构造 UWB 对象事件。"""
    return SimpleNamespace(
        modality="uwb",
        uwb_payload=SimpleNamespace(anchor_id=anchor_id, range=range_val, valid=True, quality=quality),
    )


def _vio_object_event(*, quality: Any) -> SimpleNamespace:
    """构造 VIO 对象事件。"""
    return SimpleNamespace(
        modality="vio",
        vio_payload=SimpleNamespace(dx=0.1, dy=-0.2, dyaw=0.03, quality=quality,
                                    tracked_features=24, reproj_err=0.4),
    )


# ── normalize_quality_value 测试 ──────────────────────────────────────


class TestNormalizeQualityValue:
    """normalize_quality_value 的全面测试。"""

    # --- 正常值 ---

    def test_float_in_range(self):
        assert normalize_quality_value(0.5) == 0.5

    def test_int_in_range(self):
        assert normalize_quality_value(1) == 1.0

    def test_zero(self):
        assert normalize_quality_value(0.0) == 0.0

    def test_one(self):
        assert normalize_quality_value(1.0) == 1.0

    def test_numpy_float64(self):
        assert normalize_quality_value(np.float64(0.7)) == pytest.approx(0.7)

    def test_numpy_int64(self):
        assert normalize_quality_value(np.int64(1)) == 1.0

    # --- None 回退 ---

    @pytest.mark.parametrize(
        ("default_quality", "expected"),
        [
            (0.25, 0.25),
            (0.5, 0.5),
            (0.0, 0.0),
            (1.0, 1.0),
        ],
    )
    def test_none_falls_back_to_default(self, default_quality, expected):
        assert normalize_quality_value(None, default_quality=default_quality) == expected

    # --- 裁剪 ---

    def test_negative_clipped_to_zero(self):
        assert normalize_quality_value(-0.2) == 0.0

    def test_above_one_clipped_to_one(self):
        assert normalize_quality_value(1.5) == 1.0

    def test_very_negative_clipped(self):
        assert normalize_quality_value(-100.0) == 0.0

    def test_very_large_clipped(self):
        assert normalize_quality_value(100.0) == 1.0

    # --- 类型拒绝 ---

    def test_rejects_bool_true(self):
        with pytest.raises(TypeError, match="quality must be numeric or None"):
            normalize_quality_value(True)

    def test_rejects_bool_false(self):
        with pytest.raises(TypeError, match="quality must be numeric or None"):
            normalize_quality_value(False)

    def test_rejects_numpy_bool_true(self):
        with pytest.raises(TypeError, match="quality must be numeric or None"):
            normalize_quality_value(np.bool_(True))

    def test_rejects_numpy_bool_false(self):
        with pytest.raises(TypeError, match="quality must be numeric or None"):
            normalize_quality_value(np.bool_(False))

    def test_rejects_string(self):
        with pytest.raises(TypeError, match="quality must be numeric or None"):
            normalize_quality_value("0.5")

    def test_rejects_bytes(self):
        with pytest.raises(TypeError, match="quality must be numeric or None"):
            normalize_quality_value(b"0.5")

    def test_rejects_bytearray(self):
        with pytest.raises(TypeError, match="quality must be numeric or None"):
            normalize_quality_value(bytearray(b"0.5"))

    def test_rejects_list(self):
        with pytest.raises(TypeError, match="quality must be numeric, got list"):
            normalize_quality_value([0.5])

    # --- NaN/inf 拒绝 ---

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="quality must be finite"):
            normalize_quality_value(float("nan"))

    def test_rejects_inf(self):
        with pytest.raises(ValueError, match="quality must be finite"):
            normalize_quality_value(float("inf"))

    def test_rejects_negative_inf(self):
        with pytest.raises(ValueError, match="quality must be finite"):
            normalize_quality_value(float("-inf"))

    def test_rejects_numpy_nan(self):
        with pytest.raises(ValueError, match="quality must be finite"):
            normalize_quality_value(np.float64("nan"))

    def test_rejects_numpy_inf(self):
        with pytest.raises(ValueError, match="quality must be finite"):
            normalize_quality_value(np.float64("inf"))

    # --- 返回类型 ---

    def test_returns_float(self):
        result = normalize_quality_value(0.5)
        assert isinstance(result, float)

    def test_returns_float_from_int(self):
        result = normalize_quality_value(1)
        assert isinstance(result, float)


# ── get_event_quality 测试 ────────────────────────────────────────────


class TestGetEventQuality:
    """get_event_quality 的全面测试。"""

    # --- UWB 字典事件 ---

    def test_uwb_dict_normal(self):
        assert get_event_quality(_uwb_event(quality=0.8)) == pytest.approx(0.8)

    def test_uwb_dict_quality_none_uses_default(self):
        assert get_event_quality(_uwb_event(quality=None), default_quality=0.3) == pytest.approx(0.3)

    def test_uwb_dict_quality_none_default_zero(self):
        assert get_event_quality(_uwb_event(quality=None)) == 0.0

    def test_uwb_dict_quality_clipped_high(self):
        assert get_event_quality(_uwb_event(quality=1.5)) == 1.0

    def test_uwb_dict_quality_clipped_low(self):
        assert get_event_quality(_uwb_event(quality=-0.5)) == 0.0

    # --- VIO 字典事件 ---

    def test_vio_dict_normal(self):
        assert get_event_quality(_vio_event(quality=0.6)) == pytest.approx(0.6)

    def test_vio_dict_quality_none_uses_default(self):
        assert get_event_quality(_vio_event(quality=None), default_quality=0.4) == pytest.approx(0.4)

    # --- 对象事件 ---

    def test_uwb_object_normal(self):
        assert get_event_quality(_uwb_object_event(quality=0.9)) == pytest.approx(0.9)

    def test_vio_object_normal(self):
        assert get_event_quality(_vio_object_event(quality=0.7)) == pytest.approx(0.7)

    def test_uwb_object_quality_none(self):
        assert get_event_quality(_uwb_object_event(quality=None), default_quality=0.2) == pytest.approx(0.2)

    # --- 模态校验 ---

    def test_rejects_imu_modality(self):
        with pytest.raises(ValueError, match="Unsupported modality for quality extraction"):
            get_event_quality({"modality": "imu"})

    def test_rejects_gps_modality(self):
        with pytest.raises(ValueError, match="Unsupported modality for quality extraction"):
            get_event_quality({"modality": "gps"})

    def test_rejects_none_modality(self):
        with pytest.raises(ValueError, match="Unsupported modality for quality extraction: None"):
            get_event_quality({"uwb_payload": {"quality": 0.5}})

    def test_rejects_empty_string_modality(self):
        with pytest.raises(ValueError, match="Unsupported modality"):
            get_event_quality({"modality": "", "uwb_payload": {"quality": 0.5}})

    def test_rejects_int_modality(self):
        with pytest.raises(ValueError, match="Unsupported modality"):
            get_event_quality({"modality": 42, "uwb_payload": {"quality": 0.5}})

    # --- 对象无 modality 属性 ---

    def test_object_without_modality_rejected(self):
        event = SimpleNamespace(uwb_payload=SimpleNamespace(quality=0.5))
        with pytest.raises(ValueError, match="Unsupported modality for quality extraction: None"):
            get_event_quality(event)

    # --- payload 缺失 ---

    def test_rejects_missing_uwb_payload(self):
        with pytest.raises(ValueError, match="Missing payload for modality=uwb: uwb_payload"):
            get_event_quality({"modality": "uwb"})

    def test_rejects_missing_vio_payload(self):
        with pytest.raises(ValueError, match="Missing payload for modality=vio: vio_payload"):
            get_event_quality({"modality": "vio"})

    def test_rejects_none_payload(self):
        with pytest.raises(ValueError, match="Missing payload"):
            get_event_quality({"modality": "uwb", "uwb_payload": None})

    # --- 对象无 payload 属性 ---

    def test_object_without_payload_rejected(self):
        event = SimpleNamespace(modality="uwb")
        with pytest.raises(ValueError, match="Missing payload"):
            get_event_quality(event)

    # --- payload 类型校验 ---

    def test_rejects_non_dict_payload_int(self):
        with pytest.raises(TypeError, match="uwb_payload must be a mapping or expose a quality attribute"):
            get_event_quality({"modality": "uwb", "uwb_payload": 1.0})

    def test_rejects_non_dict_payload_string(self):
        with pytest.raises(TypeError, match="uwb_payload must be a mapping or expose a quality attribute"):
            get_event_quality({"modality": "uwb", "uwb_payload": "bad"})

    def test_rejects_non_dict_payload_list(self):
        with pytest.raises(TypeError, match="uwb_payload must be a mapping or expose a quality attribute"):
            get_event_quality({"modality": "uwb", "uwb_payload": [0.5]})

    # --- payload 中 quality 缺失（返回 None → 走 default） ---

    def test_payload_missing_quality_key_uses_default(self):
        event = {"modality": "uwb", "uwb_payload": {"anchor_id": "a0", "range": 2.0, "valid": True}}
        assert get_event_quality(event, default_quality=0.25) == pytest.approx(0.25)

    def test_payload_object_without_quality_raises_type_error(self):
        """对象 payload 没有 quality 属性时，hasattr 返回 False，走到 else 分支抛 TypeError。
        这与 dict payload 不同：dict.get("quality") 缺失返回 None → 走 default，
        但对象 payload 没有 quality 属性被视为类型错误。
        """
        event = SimpleNamespace(
            modality="uwb",
            uwb_payload=SimpleNamespace(anchor_id="a0", range=2.0, valid=True),
        )
        with pytest.raises(TypeError, match="uwb_payload must be a mapping or expose a quality attribute"):
            get_event_quality(event, default_quality=0.35)

    # --- quality 类型非法 ---

    def test_rejects_string_quality(self):
        with pytest.raises(TypeError, match="quality must be numeric or None"):
            get_event_quality(_uwb_event(quality="bad"))

    def test_rejects_bool_quality(self):
        with pytest.raises(TypeError, match="quality must be numeric or None"):
            get_event_quality(_uwb_event(quality=True))

    def test_rejects_nan_quality(self):
        with pytest.raises(ValueError, match="quality must be finite"):
            get_event_quality(_uwb_event(quality=float("nan")))

    def test_rejects_inf_quality(self):
        with pytest.raises(ValueError, match="quality must be finite"):
            get_event_quality(_uwb_event(quality=float("inf")))

    # --- 返回类型 ---

    def test_returns_float(self):
        result = get_event_quality(_uwb_event(quality=0.5))
        assert isinstance(result, float)

    # --- default_quality 本身非法 ---

    def test_invalid_default_quality_raises_type_error(self):
        """当 default_quality 是字符串时，normalize_quality_value 应报 TypeError。"""
        with pytest.raises(TypeError, match="quality must be numeric or None"):
            get_event_quality(_uwb_event(quality=None), default_quality="bad")  # type: ignore[arg-type]

    def test_invalid_default_quality_nan_raises_value_error(self):
        """当 default_quality 是 NaN 时，normalize_quality_value 应报 ValueError。"""
        with pytest.raises(ValueError, match="quality must be finite"):
            get_event_quality(_uwb_event(quality=None), default_quality=float("nan"))
