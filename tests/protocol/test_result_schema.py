from __future__ import annotations

"""结果模式（result_schema）测试模块。

测试覆盖范围：
- StageResult 的结构与字段验证
- 元数据与产物的合同一致性

被测模块：liquidloc.protocol.result_schema"""

import pytest

from liquidloc.protocol.metric_schema import get_metric_order
from liquidloc.protocol.result_schema import (
    ExperimentResult,
    SequenceResult,
    SummaryResult,
    validate_experiment_result,
    validate_sequence_result,
    validate_summary_result,
)


def _metric_dict() -> dict[str, float]:
    return {metric_name: float(index + 1) for index, metric_name in enumerate(get_metric_order())}


def test_normal_case():
    """正常场景测试。
    
    验证被测功能在标准输入下的正确行为，
    确保核心路径能正常执行并返回预期结果。
    """
    sequence_result = SequenceResult(
        seq_id="seq_001",
        scene_code="S(A0,N0,V0,K0,M0)",
        model_name="liquid_ekf",
        metric_dict=_metric_dict(),
    )
    experiment_result = ExperimentResult(
        experiment_id="exp_main_table",
        model_results=[sequence_result],
        aggregate_metrics=_metric_dict(),
    )
    summary_result = SummaryResult(
        case_refs=["main_cases/seq_001"],
        summary_stats={"rmse_mean": 1.0},
    )

    validate_sequence_result(sequence_result)
    validate_experiment_result(experiment_result)
    validate_summary_result(summary_result)

    assert list(sequence_result.to_dict().keys()) == [
        "seq_id",
        "scene_code",
        "model_name",
        "metric_dict",
    ]
    assert list(experiment_result.to_dict().keys()) == [
        "snapshot_version",
        "experiment_id",
        "non_main_table",
        "model_results",
        "aggregate_metrics",
    ]
    assert experiment_result.to_dict()["snapshot_version"] == 1  # 快照版本号必须为 SNAPSHOT_VERSION 的值。
    assert list(summary_result.to_dict().keys()) == [
        "case_refs",
        "summary_stats",
    ]
    assert list(sequence_result.to_dict()["metric_dict"].keys()) == list(_metric_dict().keys())
    assert list(experiment_result.to_dict()["aggregate_metrics"].keys()) == list(_metric_dict().keys())
    assert sequence_result.to_dict()["metric_dict"]["rmse"] == 1.0
    assert experiment_result.to_dict()["experiment_id"] == "exp_main_table"
    assert summary_result.to_dict()["case_refs"] == ["main_cases/seq_001"]


def test_boundary_case():
    """边界场景测试。
    
    验证被测功能在边界条件下的行为，
    确保极端输入不会导致异常或错误结果。
    """
    sequence_result = {
        "seq_id": "seq_boundary",
        "scene_code": "S(A3,N3,V3,K0,M0)",
        "model_name": "robust_ekf",
        "metric_dict": _metric_dict(),
    }
    experiment_result = {
        "experiment_id": "exp_boundary",
        "model_results": (sequence_result,),
        "aggregate_metrics": _metric_dict(),
    }
    summary_result = {
        "case_refs": [],
        "summary_stats": {},
    }

    validate_sequence_result(sequence_result)
    validate_experiment_result(experiment_result)
    validate_summary_result(summary_result)


def test_invalid_case():
    """无效输入测试。
    
    验证被测功能对无效输入的拒绝行为，
    确保缺少必要参数时抛出 ValueError。
    """
    missing_metric_result = {
        "seq_id": "seq_bad",
        "scene_code": "S(A0,N0,V0,K0,M0)",
        "model_name": "liquid_ekf",
        "metric_dict": {
            metric_name: value
            for metric_name, value in _metric_dict().items()
            if metric_name != "rmse"
        },
    }
    bad_experiment_result = {
        "experiment_id": "exp_bad",
        "model_results": [42],
        "aggregate_metrics": _metric_dict(),
    }
    bad_summary_result = {
        "case_refs": ["  "],
        "summary_stats": {"rmse_mean": 1.0},
    }

    with pytest.raises(KeyError, match="metric_dict"):
        validate_sequence_result(missing_metric_result)

    with pytest.raises(TypeError, match="model_results\\[0\\]"):
        validate_experiment_result(bad_experiment_result)

    with pytest.raises(ValueError, match="case_refs\\[0\\]"):
        validate_summary_result(bad_summary_result)


def test_model_results_rejects_non_canonical_sequence_shape():
    """拒绝测试：model results。\n\n验证被测功能对 model results 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    experiment_result = {
        "experiment_id": "exp_bad_shape",
        "model_results": [
            {
                "seq_id": "seq_bad_shape",
                "scene_code": "S(A0,N0,V0,K0,M0)",
                "model_name": "liquid_ekf",
                "metric_dict": _metric_dict(),
                "extra_field": "unexpected",
            }
        ],
        "aggregate_metrics": _metric_dict(),
    }

    with pytest.raises(ValueError, match="model_results\\[0\\]"):
        validate_experiment_result(experiment_result)


def test_experiment_result_rejects_top_level_extra_fields():
    """拒绝测试：experiment result。\n\n验证被测功能对 experiment result 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    experiment_result = {
        "experiment_id": "exp_extra_top_level",
        "model_results": [
            {
                "seq_id": "seq_001",
                "scene_code": "S(A0,N0,V0,K0,M0)",
                "model_name": "liquid_ekf",
                "metric_dict": _metric_dict(),
            }
        ],
        "aggregate_metrics": _metric_dict(),
        "extra_field": "unexpected",
    }

    with pytest.raises(ValueError, match="unsupported fields"):
        validate_experiment_result(experiment_result)


def test_summary_result_rejects_top_level_extra_fields():
    """拒绝测试：summary result。\n\n验证被测功能对 summary result 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    summary_result = {
        "case_refs": ["main_cases/seq_001"],
        "summary_stats": {"rmse_mean": 1.0},
        "extra_field": "unexpected",
    }

    with pytest.raises(ValueError, match="unsupported fields"):
        validate_summary_result(summary_result)


def test_sequence_result_rejects_non_finite_metric_value():
    """拒绝测试：sequence result。\n\n验证被测功能对 sequence result 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    metric_dict = _metric_dict()
    metric_dict["rmse"] = float("nan")

    with pytest.raises(ValueError, match="sequence_result.metric_dict.rmse must be finite"):
        validate_sequence_result(
            {
                "seq_id": "seq_nan",
                "scene_code": "S(A0,N0,V0,K0,M0)",
                "model_name": "liquid_ekf",
                "metric_dict": metric_dict,
            }
        )


def test_experiment_result_rejects_non_finite_aggregate_metric_value():
    """拒绝测试：experiment result。\n\n验证被测功能对 experiment result 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    aggregate_metrics = _metric_dict()
    aggregate_metrics["rmse"] = float("inf")

    with pytest.raises(ValueError, match="experiment_result.aggregate_metrics.rmse must be finite"):
        validate_experiment_result(
            {
                "experiment_id": "exp_inf",
                "model_results": [
                    {
                        "seq_id": "seq_ok",
                        "scene_code": "S(A0,N0,V0,K0,M0)",
                        "model_name": "liquid_ekf",
                        "metric_dict": _metric_dict(),
                    }
                ],
                "aggregate_metrics": aggregate_metrics,
            }
        )


def test_summary_result_rejects_non_finite_summary_stat_value():
    """拒绝测试：summary result。\n\n验证被测功能对 summary result 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    with pytest.raises(ValueError, match=r"summary_result\.summary_stats\['rmse_mean'\] must be finite"):
        validate_summary_result(
            {
                "case_refs": ["case_a"],
                "summary_stats": {"rmse_mean": float("nan")},
            }
        )


def test_summary_result_rejects_nested_non_finite_summary_stat_value():
    """拒绝测试：summary result。\n\n验证被测功能对 summary result 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    with pytest.raises(
        ValueError,
        match=r"summary_result\.summary_stats\['main_table_ref'\]\[0\]\.rmse must be finite",
    ):
        validate_summary_result(
            {
                "case_refs": ["case_a"],
                "summary_stats": {
                    "main_table_ref": [
                        {"rmse": float("inf")},
                    ]
                },
            }
        )


def test_summary_result_accepts_bool_summary_stat_value():
    """接受测试：summary result。

    验证被测功能对 bool summary stat 值的接受行为，
    确保统计表里的布尔标志（如 same_tier / strict_better）能通过校验。
    """
    # 顶层 bool、嵌套 dict 内的 bool、list 内的 bool 都应被接受。
    validate_summary_result(
        {
            "case_refs": ["case_a"],
            "summary_stats": {
                "seed_check": {
                    "single_seed_no_conclusion": True,
                    "passed": False,
                },
                "section14_pairwise": [
                    {"same_tier": True, "strict_better": False, "rmse": 0.01},
                ],
            },
        }
    )


def test_summary_result_still_rejects_unsupported_value_type():
    """拒绝测试：summary result。

    验证被测功能对不支持类型的拒绝行为，
    确保 bool 接受不会破坏对 set / bytes 等不稳定类型的拦截。
    """
    with pytest.raises(TypeError, match=r"must be a finite number, bool"):
        validate_summary_result(
            {
                "case_refs": ["case_a"],
                "summary_stats": {"bad_key": {1, 2, 3}},  # set 不是合法 summary stat 值。
            }
        )


def test_experiment_result_validation_preserves_generator_model_results():
    """保持性测试：experiment result validation。\n\n验证 experiment result validation 在处理过程中不修改输入，\n校验函数应无副作用。
    """
    model_results = [
        {
            "seq_id": "seq_gen",
            "scene_code": "S(A0,N0,V0,K0,M0)",
            "model_name": "liquid_ekf",
            "metric_dict": _metric_dict(),
        }
    ]

    payload = {
        "experiment_id": "exp_gen",
        "model_results": model_results,
        "aggregate_metrics": _metric_dict(),
    }

    validate_experiment_result(payload)
    # 校验函数不应修改输入，payload 键和内容仍存在。
    assert "model_results" in payload
    assert len(payload["model_results"]) == 1


def test_summary_result_validation_preserves_generator_case_refs():
    """保持性测试：summary result validation。\n\n验证 summary result validation 在处理过程中不修改输入，\n校验函数应无副作用。
    """
    payload = {
        "case_refs": ["case_a"],
        "summary_stats": {"rmse_mean": 1.0},
    }

    validate_summary_result(payload)
    # 校验函数不应修改输入，payload 键和内容仍存在。
    assert "case_refs" in payload
    assert payload["case_refs"] == ["case_a"]
