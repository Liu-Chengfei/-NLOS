from __future__ import annotations

"""锚点模型（anchor_model）测试模块。

文件职责：验证锚点布局校验、查找表构建和
几何评分计算的正确性。

测试覆盖范围：
- validate_anchor_layout：合法字典/对象、字段缺失、类型非法、
  数量校验、id 唯一性、坐标校验
- build_anchor_lookup：正方形/单锚点/整数坐标/生成器输入
- compute_geometry_report：正方形/单锚点/共线/等边三角形评分

被测模块：liquidloc.sensors.anchor_model"""


import math
from types import SimpleNamespace

import numpy as np
import pytest

from liquidloc.sensors.anchor_model import (
    build_anchor_lookup,
    compute_geometry_report,
    validate_anchor_layout,
)


# ---------------------------------------------------------------------------
# 辅助工厂
# ---------------------------------------------------------------------------

def _square_layout():
    """正方形四锚点布局。"""
    return {
        "anchor_positions": [(0, 0), (1, 0), (1, 1), (0, 1)],
        "anchor_ids": ["a0", "a1", "a2", "a3"],
        "layout_id": "square_k4",
    }


def _single_anchor_layout():
    """单锚点布局。"""
    return {
        "anchor_positions": [(2, 3)],
        "anchor_ids": ["solo"],
        "layout_id": "single",
    }


def _collinear_layout():
    """共线三点布局（x 轴上均匀分布）。"""
    return {
        "anchor_positions": [(0, 0), (1, 0), (2, 0)],
        "anchor_ids": ["c0", "c1", "c2"],
        "layout_id": "collinear",
    }


def _two_anchor_layout():
    """两个锚点布局。"""
    return {
        "anchor_positions": [(0, 0), (1, 0)],
        "anchor_ids": ["t0", "t1"],
        "layout_id": "two",
    }


# ===========================================================================
# TestValidateAnchorLayout
# ===========================================================================

class TestValidateAnchorLayout:
    """validate_anchor_layout 的单元测试。"""

    def test_valid_dict(self):
        """合法字典布局校验通过。"""
        validate_anchor_layout(_square_layout())

    def test_valid_object(self):
        """合法对象布局校验通过。"""
        layout = SimpleNamespace(
            anchor_positions=[(0, 0), (1, 0)],
            anchor_ids=["a0", "a1"],
            layout_id="obj",
        )
        validate_anchor_layout(layout)

    # -- 字段缺失 --

    def test_missing_positions_rejected(self):
        """缺少 anchor_positions 被拒绝。"""
        layout = {"anchor_ids": ["a0"], "layout_id": "x"}
        with pytest.raises(ValueError, match="must provide anchor_positions"):
            validate_anchor_layout(layout)

    def test_missing_ids_rejected(self):
        """缺少 anchor_ids 被拒绝。"""
        layout = {"anchor_positions": [(0, 0)], "layout_id": "x"}
        with pytest.raises(ValueError, match="must provide anchor_ids"):
            validate_anchor_layout(layout)

    def test_missing_layout_id_passes(self):
        """缺少 layout_id 允许通过（可选字段）。"""
        layout = {"anchor_positions": [(0, 0)], "anchor_ids": ["a0"]}
        validate_anchor_layout(layout)

    # -- 类型非法 --

    def test_string_positions_rejected(self):
        """字符串 anchor_positions 被拒绝。"""
        layout = {"anchor_positions": "bad", "anchor_ids": ["a0"]}
        with pytest.raises(TypeError, match="must be a coordinate collection"):
            validate_anchor_layout(layout)

    def test_bytes_positions_rejected(self):
        """bytes anchor_positions 被拒绝。"""
        layout = {"anchor_positions": b"bad", "anchor_ids": ["a0"]}
        with pytest.raises(TypeError, match="must be a coordinate collection"):
            validate_anchor_layout(layout)

    def test_string_ids_rejected(self):
        """字符串 anchor_ids 被拒绝。"""
        layout = {"anchor_positions": [(0, 0)], "anchor_ids": "bad"}
        with pytest.raises(TypeError, match="must be a collection of ids"):
            validate_anchor_layout(layout)

    # -- 数量校验 --

    def test_empty_positions_rejected(self):
        """空坐标列表被拒绝。"""
        layout = {"anchor_positions": [], "anchor_ids": []}
        with pytest.raises(ValueError, match="anchor_count must be >= 1"):
            validate_anchor_layout(layout)

    def test_length_mismatch_rejected(self):
        """坐标和 id 数量不一致被拒绝。"""
        layout = {"anchor_positions": [(0, 0), (1, 0)], "anchor_ids": ["a0"]}
        with pytest.raises(ValueError, match="must have the same length"):
            validate_anchor_layout(layout)

    # -- id 唯一性 --

    def test_duplicate_ids_rejected(self):
        """重复 id 被拒绝。"""
        layout = {
            "anchor_positions": [(0, 0), (1, 0)],
            "anchor_ids": ["dup", "dup"],
        }
        with pytest.raises(ValueError, match="must be unique"):
            validate_anchor_layout(layout)

    def test_unhashable_ids_rejected(self):
        """不可哈希 id 被拒绝。"""
        layout = {
            "anchor_positions": [(0, 0), (1, 0)],
            "anchor_ids": [["list0"], ["list1"]],
        }
        with pytest.raises(TypeError, match="must be hashable"):
            validate_anchor_layout(layout)

    # -- 坐标校验 --

    def test_string_position_rejected(self):
        """字符串坐标被拒绝。"""
        layout = {"anchor_positions": ["bad"], "anchor_ids": ["a0"]}
        with pytest.raises(TypeError, match="must be a 2D coordinate"):
            validate_anchor_layout(layout)

    def test_1d_position_rejected(self):
        """一维坐标被拒绝。"""
        layout = {"anchor_positions": [(1.0,)], "anchor_ids": ["a0"]}
        with pytest.raises(ValueError, match="exactly two coordinates"):
            validate_anchor_layout(layout)

    def test_3d_position_rejected(self):
        """三维坐标被拒绝。"""
        layout = {"anchor_positions": [(1.0, 2.0, 3.0)], "anchor_ids": ["a0"]}
        with pytest.raises(ValueError, match="exactly two coordinates"):
            validate_anchor_layout(layout)

    def test_nan_coordinate_rejected(self):
        """NaN 坐标被拒绝。"""
        layout = {"anchor_positions": [(float("nan"), 0.0)], "anchor_ids": ["a0"]}
        with pytest.raises(ValueError, match="must be finite"):
            validate_anchor_layout(layout)

    def test_inf_coordinate_rejected(self):
        """inf 坐标被拒绝。"""
        layout = {"anchor_positions": [(float("inf"), 0.0)], "anchor_ids": ["a0"]}
        with pytest.raises(ValueError, match="must be finite"):
            validate_anchor_layout(layout)

    def test_neg_inf_coordinate_rejected(self):
        """-inf 坐标被拒绝。"""
        layout = {"anchor_positions": [(0.0, float("-inf"))], "anchor_ids": ["a0"]}
        with pytest.raises(ValueError, match="must be finite"):
            validate_anchor_layout(layout)

    def test_string_coordinate_rejected(self):
        """字符串坐标值被拒绝。"""
        layout = {"anchor_positions": [("x", 0.0)], "anchor_ids": ["a0"]}
        with pytest.raises(TypeError, match="must be numeric"):
            validate_anchor_layout(layout)

    def test_negative_coordinates_pass(self):
        """负坐标允许通过。"""
        layout = {"anchor_positions": [(-1.0, -2.0)], "anchor_ids": ["a0"]}
        validate_anchor_layout(layout)

    def test_numpy_coordinates_converted(self):
        """numpy 坐标被转换为 float。"""
        layout = {
            "anchor_positions": [(np.float64(1.5), np.float64(2.5))],
            "anchor_ids": ["a0"],
        }
        validate_anchor_layout(layout)

    # -- 副作用：字典输入被原地修改 --

    def test_dict_input_mutated(self):
        """字典输入被原地修改（归一化后的值写回）。"""
        layout = {"anchor_positions": [(0, 0), (1, 0)], "anchor_ids": ["a0", "a1"]}
        validate_anchor_layout(layout)
        # 归一化后 positions 变成浮点元组列表
        assert layout["anchor_positions"] == [(0.0, 0.0), (1.0, 0.0)]


# ===========================================================================
# TestBuildAnchorLookup
# ===========================================================================

class TestBuildAnchorLookup:
    """build_anchor_lookup 的单元测试。"""

    def test_square_layout(self):
        """正方形布局构建正确的查表。"""
        lookup = build_anchor_lookup(_square_layout())
        assert lookup == {
            "a0": (0.0, 0.0),
            "a1": (1.0, 0.0),
            "a2": (1.0, 1.0),
            "a3": (0.0, 1.0),
        }

    def test_single_anchor(self):
        """单锚点布局。"""
        lookup = build_anchor_lookup(_single_anchor_layout())
        assert lookup == {"solo": (2.0, 3.0)}

    def test_integer_coordinates_to_float(self):
        """整数坐标被转为浮点。"""
        layout = {
            "anchor_positions": [(1, 2)],
            "anchor_ids": ["a0"],
        }
        lookup = build_anchor_lookup(layout)
        assert lookup["a0"] == (1.0, 2.0)

    def test_integer_ids(self):
        """整数 id 允许。"""
        layout = {
            "anchor_positions": [(0, 0), (1, 0)],
            "anchor_ids": [0, 1],
        }
        lookup = build_anchor_lookup(layout)
        assert lookup[0] == (0.0, 0.0)
        assert lookup[1] == (1.0, 0.0)

    def test_return_type(self):
        """返回类型正确。"""
        lookup = build_anchor_lookup(_square_layout())
        assert isinstance(lookup, dict)
        for key, value in lookup.items():
            assert isinstance(value, tuple)
            assert len(value) == 2
            assert isinstance(value[0], float)
            assert isinstance(value[1], float)

    def test_generator_input(self):
        """生成器输入正确处理。"""
        layout = {
            "anchor_positions": ((v, v + 1) for v in range(2)),
            "anchor_ids": (f"a{v}" for v in range(2)),
            "layout_id": "gen",
        }
        lookup = build_anchor_lookup(layout)
        assert lookup == {"a0": (0.0, 1.0), "a1": (1.0, 2.0)}


# ===========================================================================
# TestComputeGeometryReport
# ===========================================================================

class TestComputeGeometryReport:
    """compute_geometry_report 的单元测试。"""

    def test_square_layout_score_1(self):
        """正方形布局几何评分为 1.0。"""
        report = compute_geometry_report(_square_layout())
        assert report["anchor_count"] == 4
        assert report["layout_id"] == "square_k4"
        assert report["geom_score"] == 1.0

    def test_single_anchor_score_0(self):
        """单锚点几何评分为 0.0。"""
        report = compute_geometry_report(_single_anchor_layout())
        assert report["anchor_count"] == 1
        assert report["geom_score"] == 0.0

    def test_two_anchors_collinear_score_0(self):
        """两个共线锚点几何评分为 0.0。"""
        report = compute_geometry_report(_two_anchor_layout())
        assert report["anchor_count"] == 2
        assert report["geom_score"] == 0.0

    def test_three_collinear_score_0(self):
        """三个共线锚点几何评分为 0.0。"""
        report = compute_geometry_report(_collinear_layout())
        assert report["geom_score"] == 0.0

    def test_equilateral_triangle_score(self):
        """等边三角形几何评分接近 1.0。"""
        # 等边三角形的特征值比 = 1.0（均匀分布）
        layout = {
            "anchor_positions": [(0, 0), (1, 0), (0.5, math.sqrt(3) / 2)],
            "anchor_ids": ["t0", "t1", "t2"],
            "layout_id": "equilateral",
        }
        report = compute_geometry_report(layout)
        assert report["geom_score"] == pytest.approx(1.0, abs=1e-6)

    def test_layout_id_none(self):
        """layout_id 为 None 时正确记录。"""
        layout = {"anchor_positions": [(0, 0)], "anchor_ids": ["a0"]}
        report = compute_geometry_report(layout)
        assert report["layout_id"] is None

    def test_return_keys(self):
        """返回字典包含 3 个固定键。"""
        report = compute_geometry_report(_square_layout())
        assert set(report.keys()) == {"anchor_count", "layout_id", "geom_score"}

    def test_return_types(self):
        """返回值类型正确。"""
        report = compute_geometry_report(_square_layout())
        assert isinstance(report["anchor_count"], int)
        assert isinstance(report["geom_score"], float)

    def test_geom_score_range(self):
        """几何评分在 [0, 1] 范围内。"""
        # 用一个不等边布局
        layout = {
            "anchor_positions": [(0, 0), (10, 0), (5, 1)],
            "anchor_ids": ["a0", "a1", "a2"],
        }
        report = compute_geometry_report(layout)
        assert 0.0 <= report["geom_score"] <= 1.0

    def test_coincident_points_score_0(self):
        """重合锚点几何评分为 0.0。"""
        layout = {
            "anchor_positions": [(1, 1), (1, 1)],
            "anchor_ids": ["a0", "a1"],
        }
        report = compute_geometry_report(layout)
        assert report["geom_score"] == 0.0
