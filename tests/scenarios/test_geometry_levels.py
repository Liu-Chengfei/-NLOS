from __future__ import annotations

"""几何等级（geometry_levels）测试模块。

文件职责：验证 build_anchor_layout 能根据几何条件等级
生成锚点布局，以及 project_anchor_layout_to_reference
能将布局投影到参考布局上。

测试覆盖范围：
- 正常场景：K3 等级生成 4 锚点布局
- 边界场景：K1 等级单锚点布局
- 投影保持参考质心和尺度
- 异常场景：缺失 geom_condition、非法锚点数量

被测模块：liquidloc.scenarios.geometry_levels"""


import pytest

from liquidloc.scenarios.geometry_levels import build_anchor_layout, project_anchor_layout_to_reference


def _geometry_cfg():
    """五轴档位协议 fixtures：使用协议归一化边界（min=1.0, max=10.0）。
    协议层 K 轴仅支持 K0/K1/K3 三档，G 轴已并入 K。
    """
    return {
        "levels": {
            "K0": {"geom_condition": 1.0},
            "K1": {"geom_condition": 2.0},
            "K3": {"geom_condition": 3.5},
        }
    }


def test_normal_case():
    anchor_layout, geometry_report = build_anchor_layout(4, 'K3', _geometry_cfg())

    # 按「五轴档位协议定义」K3 = 4 锚近共线退化（3 锚近似共线 + 1 锚孤立）。
    assert anchor_layout['layout_id'] == 'K3_K4'
    assert anchor_layout['anchor_count'] == 4
    assert anchor_layout['anchor_positions'] == [
        [-1.0, 0.0],
        [-0.33, 0.0],
        [0.33, 0.0],
        [0.0, 1.5],
    ]
    assert geometry_report['layout_id'] == 'K3_K4'
    assert geometry_report['geom_condition'] == pytest.approx(3.5)
    assert geometry_report['layout_strength'] == pytest.approx(1.0)
    # geom_score 由 build_anchor_layout 实际计算为准（K3 严重退化，分数低）。
    assert 0.0 <= geometry_report['geom_score'] <= 1.0


def test_boundary_case():
    # 按「五轴档位协议定义」K 轴锚数全档固定 4，故 K0 用 4 锚测试。
    anchor_layout, geometry_report = build_anchor_layout(4, 'K0', _geometry_cfg())

    # 按「五轴档位协议定义」K0 = 4 锚对称矩形四角。
    assert anchor_layout['anchor_positions'] == [
        [-1.0, -1.0],
        [1.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
    ]
    assert anchor_layout['layout_id'] == 'K0_K4'
    assert geometry_report['anchor_count'] == 4
    assert geometry_report['geom_score'] == pytest.approx(1.0)
    assert geometry_report['layout_strength'] == pytest.approx(0.0)


def test_project_anchor_layout_to_reference_preserves_reference_centroid_and_scale():
    anchor_layout, _ = build_anchor_layout(4, 'K0', _geometry_cfg())
    projected_layout = project_anchor_layout_to_reference(
        anchor_layout,
        {
            'anchor_ids': [0, 1],
            'anchor_positions': [[2.0, 0.0], [2.1, 0.0]],
            'layout_id': 'reference_layout',
        },
    )

    xs = [position[0] for position in projected_layout['anchor_positions']]
    ys = [position[1] for position in projected_layout['anchor_positions']]
    assert sum(xs) / len(xs) == pytest.approx(2.05)
    assert sum(ys) / len(ys) == pytest.approx(0.0)
    assert projected_layout['reference_layout_id'] == 'reference_layout'
    assert projected_layout['scale_to_reference'] >= 0.0


@pytest.mark.parametrize(
    ('anchor_count', 'geometry_level', 'geometry_cfg', 'exc_type', 'match'),
    [
        (4, 'K3', {'levels': {'K3': {'label': 'bad'}}}, ValueError, 'geom_condition'),
        (True, 'K3', {'levels': {'K3': {'geom_condition': 1.0}}}, TypeError, 'integer'),
    ],
)
def test_invalid_case(anchor_count, geometry_level, geometry_cfg, exc_type, match):
    with pytest.raises(exc_type, match=match):
        build_anchor_layout(anchor_count, geometry_level, geometry_cfg)


# V1 修复（§4.2.2 GDOP 占比显式断言）：验证 build_anchor_layout 在 geometry_report 中
# 显式暴露 gdop_value / gdop_floor / gdop_above_floor_ratio 字段，并把
# "本布局是否落入差 GDOP 族" 作为 consistency_checks 显式断言。
# 协议层 `configs/base/scene_axis_protocol.yaml` 已固化 geom_condition 即 GDOP 量级语义
# （"数值越大，GDOP 越大"），`前提指导.md:894` "差 GDOP、病态观测时段达到足够占比"
# 要求评估层可时序统计差 GDOP 占比，本测试验证单布局层已正确暴露所需字段。
def _gdop_geometry_cfg():
    return {
        'levels': {
            'K0': {'geom_condition': 1.0},  # 优 GDOP
            'K1': {'geom_condition': 2.0},  # 中等 GDOP
            'K3': {'geom_condition': 3.5},  # 差 GDOP
        }
    }


def test_gdop_fields_present_in_geometry_report():
    """geometry_report 必须包含 gdop_value / gdop_floor / gdop_above_floor_ratio 三个 V1 修复字段。"""
    layout, report = build_anchor_layout(4, 'K3', _gdop_geometry_cfg())
    assert 'gdop_value' in report
    assert 'gdop_floor' in report
    assert 'gdop_above_floor_ratio' in report
    assert isinstance(report['gdop_value'], float)
    assert isinstance(report['gdop_floor'], float)
    assert isinstance(report['gdop_above_floor_ratio'], float)


def test_gdop_default_floor_is_1_5():
    """协议顶层未显式给 gdop_floor 时，build_anchor_layout 应取默认 1.5（与 G0=1.0 < floor + G2=3.5 ≥ floor 语义对齐）。"""
    layout, report = build_anchor_layout(4, 'K1', _gdop_geometry_cfg())
    assert report['gdop_floor'] == 1.5


def test_gdop_explicit_floor_loaded_from_protocol():
    """协议顶层显式给 gdop_floor 时，build_anchor_layout 应读入并归一化为非负浮点。"""
    cfg = {**_gdop_geometry_cfg(), 'gdop_floor': 2.5}
    layout, report = build_anchor_layout(4, 'K3', cfg)
    assert report['gdop_floor'] == 2.5
    assert report['gdop_value'] == 3.5
    assert report['gdop_above_floor_ratio'] == 1.0


def test_gdop_above_floor_ratio_for_k0_below_default_floor():
    """K0 (geom_condition=1.0) < 默认 floor 1.5 → ratio=0.0 + meets_floor=False。"""
    layout, report = build_anchor_layout(4, 'K0', _gdop_geometry_cfg())
    assert report['gdop_value'] == 1.0
    assert report['gdop_above_floor_ratio'] == 0.0
    assert report['consistency_checks']['gdop_occupancy_meets_floor'] is False


def test_gdop_above_floor_ratio_for_k3_above_default_floor():
    """K3 (geom_condition=3.5) ≥ 默认 floor 1.5 → ratio=1.0 + meets_floor=True。"""
    layout, report = build_anchor_layout(4, 'K3', _gdop_geometry_cfg())
    assert report['gdop_value'] == 3.5
    assert report['gdop_above_floor_ratio'] == 1.0
    assert report['consistency_checks']['gdop_occupancy_meets_floor'] is True


def test_gdop_consistency_checks_keys_present():
    """consistency_checks 中必须显式包含 V1 修复后新增的 GDOP 两条断言键。"""
    layout, report = build_anchor_layout(4, 'K3', _gdop_geometry_cfg())
    assert 'gdop_floor_loaded' in report['consistency_checks']
    assert 'gdop_occupancy_meets_floor' in report['consistency_checks']
    assert report['consistency_checks']['gdop_floor_loaded'] is True


def test_gdop_negative_floor_rejected():
    """显式给负的 gdop_floor 必须被 coerce_finite_scalar(min_value=0.0) 拒绝。"""
    cfg = {**_gdop_geometry_cfg(), 'gdop_floor': -0.5}
    with pytest.raises(ValueError, match='gdop_floor'):
        build_anchor_layout(4, 'K3', cfg)
