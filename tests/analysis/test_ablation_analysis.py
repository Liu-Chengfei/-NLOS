"""消融分析模块测试。

本模块验证 build_ablation_table 函数的正确性，该函数对比完整模型
与消融变体的指标差异，生成消融分析表。

测试覆盖范围：
  - 正常情况：完整模型与消融变体的指标差异计算
  - 边界情况：基线值为 0 且消融值也为 0 时，delta 和 relative_delta 均为 0
  - 异常情况：基线值为 0 但消融值非零时，relative_delta 未定义，抛出 ValueError
  - 异常情况：非数值类型 metric 值被拒绝

被测模块：
  - liquidloc.analysis.ablation_analysis
"""

import pytest

from liquidloc.analysis.ablation_analysis import build_ablation_table


def test_normal_case():
    """验证完整模型与消融变体的指标差异计算。

    测试场景：完整模型 acc=0.8、loss=1.0，消融变体 drop_aug 的
    acc=0.7、loss=1.4。
    预期行为：生成两条记录，分别对应 acc 和 loss 的差异，
    delta 为消融值减去完整值，relative_delta 为 delta 除以完整值。
    """
    full_metrics = {"method_name": "full", "acc": 0.8, "loss": 1.0}
    ablation_metrics = {"drop_aug": {"method_name": "drop_aug", "acc": 0.7, "loss": 1.4}}

    table = build_ablation_table(full_metrics, ablation_metrics, ["acc", "loss"])

    # 验证 acc 的差异：0.7 - 0.8 = -0.1，relative_delta = -0.1/0.8 = -0.125
    assert table == [
        {"method_name": "drop_aug", "metric_name": "acc", "delta": -0.10000000000000009, "relative_delta": -0.1250000000000001},
        {"method_name": "drop_aug", "metric_name": "loss", "delta": 0.3999999999999999, "relative_delta": 0.3999999999999999},
    ]


def test_boundary_case():
    """验证基线值和消融值均为 0 时，delta 和 relative_delta 均为 0。

    测试场景：完整模型 acc=0.0，消融变体 acc=0.0。
    预期行为：delta=0.0，relative_delta=0.0（0/0 的特殊情况，
    函数内部将 0/0 视为 0.0）。
    """
    full_metrics = {"method_name": "full", "acc": 0.0}
    ablation_metrics = {"edge_case": {"acc": 0.0}}

    table = build_ablation_table(full_metrics, ablation_metrics, ["acc"])

    # 0/0 的特殊情况，函数内部视为 0.0
    assert table == [
        {"method_name": "edge_case", "metric_name": "acc", "delta": 0.0, "relative_delta": 0.0},
    ]


def test_zero_baseline_nonzero_delta_is_rejected():
    """验证基线值为 0 但消融值非零时，relative_delta 未定义，抛出 ValueError。

    测试场景：完整模型 acc=0.0，消融变体 acc=0.1。
    预期行为：0.1/0.0 的 relative_delta 未定义，抛出 ValueError。
    """
    full_metrics = {"method_name": "full", "acc": 0.0}
    ablation_metrics = {"edge_case": {"acc": 0.1}}

    with pytest.raises(ValueError, match=r"relative_delta is undefined"):
        build_ablation_table(full_metrics, ablation_metrics, ["acc"])


def test_invalid_case():
    """验证非数值类型 metric 值被拒绝。

    测试场景：消融变体的 acc 值为字符串 "oops"。
    预期行为：抛出 TypeError，提示 metric 值必须为数值类型。
    """
    full_metrics = {"method_name": "full", "acc": 0.8}
    ablation_metrics = {"bad_case": {"method_name": "bad_case", "acc": "oops"}}

    with pytest.raises(TypeError, match=r"must be numeric"):
        build_ablation_table(full_metrics, ablation_metrics, ["acc"])
