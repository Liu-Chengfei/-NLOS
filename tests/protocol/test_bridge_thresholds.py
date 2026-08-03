"""桥接层阈值协议测试模块。

验证 §11.2 修复（Hazard §11-2）：imu_missing_inflation 必须走协议单源真相，
而非 estimator 内部硬编码。
"""

from __future__ import annotations

import pytest

from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS


class TestImuMissingInflationProtocolConstant:
    """imu_missing_inflation 协议常量测试。"""

    def test_imu_missing_inflation_exists_in_bridge_thresholds(self):
        """协议层必须存在 imu_missing_inflation 常量。"""
        assert "imu_missing_inflation" in BRIDGE_THRESHOLDS

    def test_imu_missing_inflation_eq_10(self):
        """协议层 imu_missing_inflation 必须等于 10.0（与 estimator 内部硬编码一致）。

        §11.2 要求 Q 不在门控层被单方偷偷加大；imu_missing_inflation 是 IMU 字段
        缺失这一物理事件的方差膨胀系数，必须协议写死而非 estimator 私调。
        """
        assert BRIDGE_THRESHOLDS["imu_missing_inflation"] == pytest.approx(10.0, abs=1e-12)

    def test_imu_missing_inflation_is_readonly(self):
        """BRIDGE_THRESHOLDS 是 _FrozenDict，运行时不可篡改。"""
        with pytest.raises(TypeError):
            BRIDGE_THRESHOLDS["imu_missing_inflation"] = 99.0  # type: ignore[misc]

    def test_imu_missing_inflation_is_readonly_delete(self):
        """BRIDGE_THRESHOLDS 是 _FrozenDict，运行时不可删除。"""
        with pytest.raises(TypeError):
            del BRIDGE_THRESHOLDS["imu_missing_inflation"]  # type: ignore[misc]

    def test_imu_missing_inflation_is_readonly_pop(self):
        """BRIDGE_THRESHOLDS 是 _FrozenDict，运行时不可 pop。"""
        with pytest.raises(TypeError):
            BRIDGE_THRESHOLDS.pop("imu_missing_inflation")  # type: ignore[misc]
