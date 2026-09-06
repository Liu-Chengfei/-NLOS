"""评估管线与高层消费者集成测试模块。

本模块验证 eval_pipeline 的输出能否被 script 19（高层消费者验证脚本）
正确消费，确保评估产物（metric_table、statistics_table、trajectory_bundle 等）
与下游验证脚本的输入合同一致。

测试覆盖范围：
  - eval_pipeline 输出产物被 script19 正确消费，生成完整验证报告和图表
  - script19 解析 prediction_index 中的相对路径（基于 index 文件所在目录）
  - metric_table 中包含 runtime 组指标

被测模块：
  - liquidloc.pipelines.eval_pipeline
  - scripts.19_verify_high_level_consumers
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import shutil
from pathlib import Path

from liquidloc.pipelines.eval_pipeline import run as run_eval_pipeline

# fixture 数据根目录，用于加载 ground truth
FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "datasets" / "miluv"
# 仓库根目录，用于定位 scripts 目录
ROOT = Path(__file__).resolve().parents[2]


def _load_stdout_json(stdout: str) -> dict:
    """从脚本 stdout 中提取末尾的 JSON 对象。

    被测脚本在打印最终 JSON 有效负载之前会先打印大量 ``[配置]`` 日志行;
    不能直接对整个 stdout 调 ``json.loads``. 这里跳过所有以 ``[配置]``
    开头的日志行, 然后在剩余文本中通过括号平衡定位最右侧完整的 JSON 对象
    (即脚本最终输出的有效负载).
    """
    lines = stdout.splitlines()
    remaining = "\n".join(line for line in lines if not line.startswith("[配置]"))
    decoder = json.JSONDecoder()
    text = remaining.rstrip()
    # 从右往左寻找能成功 raw_decode 到结尾的 '{', 取最近的一个; 这是脚本末尾
    # 打印的最终 JSON 有效负载.
    idx = text.rfind("{")
    while idx >= 0:
        try:
            obj, end = decoder.raw_decode(text[idx:])
            if end == len(text[idx:]):
                return obj
        except json.JSONDecodeError:
            pass
        idx = text.rfind("{", 0, idx)
    raise AssertionError(f"no JSON payload found in stdout tail: {text[-200:]!r}")



def _load_script19():
    """动态加载 script 19（高层消费者验证脚本）。

    Returns:
        加载后的模块对象，可直接调用其 main 函数和修改模块级变量。
    """
    script_path = ROOT / "scripts" / "19_verify_high_level_consumers.py"
    spec = importlib.util.spec_from_file_location("high_level_consumer_verify_for_integration", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _bundle(method_name: str, *, second_px: float) -> dict:
    """构造一个标准 prediction bundle 字典，用于测试。

    Args:
        method_name: 方法名称（如 "ekf"、"robust_ekf"）。
        second_px: 第二个状态点的 px 坐标，用于控制轨迹差异。

    Returns:
        包含 seq_id、scene_id、method_name、states、timestamps、
        diagnostics、runtime_log、scenario_context 的完整 bundle。
    """
    return {
        "seq_id": "mini_seq",
        "scene_id": "S(A2,N2,V2,K0,M0)",
        "method_name": method_name,
        "task_id": "scene_00",
        # 两个状态点，第二个 px 由参数控制
        "states": [{"px": 0.0, "py": 0.0}, {"px": second_px, "py": 0.0}],
        "timestamps": [0.0, 0.1],
        "diagnostics": {
            "risk_trace": [0.0, 0.0],
            "bias_trace": [0.0, 0.0],
            # scaling_trace 的第二个值与 second_px 对应
            "scaling_trace": [0.0, second_px],
        },
        "runtime_log": {"latency": [1.0, 2.0], "params": 0.0, "ram_peak": 0.0},
        "scenario_context": {
            "scenario_reports": {
                "V": {
                    "protocol_consistent": True,
                    "consistency_checks": {
                        "tracked_features_range": True,
                        "reproj_err_max": True,
                        "blackout": True,
                        "drift_scale": True,
                    },
                }
            }
        },
    }


def _write_fixed_eval_root(eval_root: Path) -> None:
    """在给定目录下写入固定的评估产物文件，用于相对路径测试。

    写入 metric_table.csv、statistics_table.json、selected_cases.json、
    eval_audit.json、runtime_table.json、sweep_table.json、
    trajectory_bundle.json、gt_bundle.json。

    Args:
        eval_root: 评估产物根目录。
    """
    metrics_root = eval_root / "metrics"
    statistics_root = eval_root / "statistics"
    cases_root = eval_root / "cases"
    audits_root = eval_root / "audits"
    plotting_inputs_root = eval_root / "plotting_inputs"
    for path in (metrics_root, statistics_root, cases_root, audits_root, plotting_inputs_root):
        path.mkdir(parents=True, exist_ok=True)

    # 写入 metric_table.csv，包含 primary 和 mechanism 两组指标
    (metrics_root / "metric_table.csv").write_text(
        "\n".join(
            [
                "case_ref,seq_id,scene_id,method_name,metric,value,group",
                'case-1,mini_seq,"S(A2,N2,V2,K0,M0)",ekf,rmse,0.80,primary',
                'case-1,mini_seq,"S(A2,N2,V2,K0,M0)",ekf,risk_error_corr,0.40,mechanism',
                'case-1,mini_seq,"S(A2,N2,V2,K0,M0)",ekf,coverage,0.85,mechanism',
                'case-1,mini_seq,"S(A2,N2,V2,K0,M0)",ekf,bias_alignment,0.30,mechanism',
                'case-1,mini_seq,"S(A2,N2,V2,K0,M0)",ekf,corr_scaling_error,0.25,mechanism',
            ]
        ),
        encoding="utf-8",
    )
    # 写入统计检验表
    (statistics_root / "statistics_table.json").write_text(
        json.dumps({"method_summary": {"ekf": {"rmse": 0.80}}, "pairwise_tests": []}),
        encoding="utf-8",
    )
    # 写入选中案例
    (cases_root / "selected_cases.json").write_text(
        json.dumps(
            {
                "main_cases": [{"case_ref": "case-1"}],
                "failure_cases": [{"case_ref": "case-2"}],
                "boundary_cases": [{"case_ref": "case-3"}],
            }
        ),
        encoding="utf-8",
    )
    # 写入评估审计信息
    (audits_root / "eval_audit.json").write_text(
        json.dumps(
            {
                "num_prediction_bundles": 1,
                "num_metric_rows": 5,
                "best_method": "ekf",
                "best_method_by_priority": "ekf",
                "conclusion_priority": ["rmse"],
                "protocol_gate": {
                    "num_prediction_bundles": 1,
                    "failure_threshold": 1.0,
                    "aggregation_order": [
                        "single_run",
                        "repeat_summary",
                        "scene_summary",
                        "experiment_conclusion",
                    ],
                    "conclusion_priority": ["rmse"],
                },
            }
        ),
        encoding="utf-8",
    )
    # 写入运行时指标表
    (plotting_inputs_root / "runtime_table.json").write_text(
        json.dumps(
            [
                {
                    "case_ref": "scene_00::ekf",
                    "latency_mean": 1.0,
                    "latency_p50": 0.9,
                    "latency_p95": 1.1,
                    "params": 10,
                    "ram_peak": 20,
                }
            ]
        ),
        encoding="utf-8",
    )
    # 写入主表聚合视图
    (plotting_inputs_root / "main_table.json").write_text(
        json.dumps(
            [
                {
                    "method_name": "ekf",
                    "rmse": 0.80,
                }
            ]
        ),
        encoding="utf-8",
    )
    # 写入扫描表（场景轴展开后的指标）
    (plotting_inputs_root / "sweep_table.json").write_text(
        json.dumps(
            [
                {
                    "case_ref": "scene_00::ekf",
                    "seq_id": "mini_seq",
                    "scene_id": "S(A2,N2,V2,K0,M0)",
                    "method_name": "ekf",
                    "A": "A2",
                    "N": "N2",
                    "V": "V2",
                    "G": "K1",
                    "K": "K0",
                    "rmse": 0.8,
                }
            ]
        ),
        encoding="utf-8",
    )
    # 写入轨迹 bundle（预测 + 真值）
    (plotting_inputs_root / "trajectory_bundle.json").write_text(
        json.dumps(
            {
                "prediction_bundle": [_bundle("ekf", second_px=0.03)],
                "gt_bundle": [
                    {
                        "seq_id": "mini_seq",
                        "scene_id": "S(A2,N2,V2,K0,M0)",
                        "method_name": "ekf",
                        "task_id": "scene_00",
                        "states": [{"px": 0.0, "py": 0.0}, {"px": 0.0, "py": 0.0}],
                        "timestamps": [0.0, 0.1],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    # 写入真值 bundle
    (plotting_inputs_root / "gt_bundle.json").write_text(
        json.dumps(
            [
                {
                    "seq_id": "mini_seq",
                    "scene_id": "S(A2,N2,V2,K0,M0)",
                    "method_name": "ekf",
                    "task_id": "scene_00",
                    "states": [{"px": 0.0, "py": 0.0}, {"px": 0.0, "py": 0.0}],
                    "timestamps": [0.0, 0.1],
                }
            ]
        ),
        encoding="utf-8",
    )


def test_eval_pipeline_outputs_feed_script19_high_level_consumers(tmp_path, capsys):
    """验证 eval_pipeline 的输出产物能被 script19 正确消费。

    测试场景：运行 eval_pipeline 生成评估产物，然后调用 script19
    验证高层消费者能否正确读取这些产物并生成图表和验证报告。
    预期行为：eval_pipeline 输出所有必需产物文件，
    script19 返回退出码 0，输出 JSON 中 status 为 ok，
    图表文件（sweep.svg、trajectory.png、main_table.png 等）均存在，
    metric_table 中包含 runtime 组指标。
    """
    route_root = tmp_path / "fixed_route"
    eval_root = route_root / "eval"
    verification_root = route_root / "verification"

    # 运行 eval_pipeline，生成评估产物
    result = run_eval_pipeline(
        {
            "prediction_bundles": [
                _bundle("ekf", second_px=0.03),
                _bundle("robust_ekf", second_px=0.30),
            ],
            "ground_truth_root": FIXTURE_ROOT,
            "output_root": eval_root,
        }
    )

    # 构造 prediction index 文件，供 script19 读取
    prediction_root = route_root / "predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    prediction_index_path = route_root / "audits" / "prediction_index.json"
    prediction_index_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_bundles = [
        _bundle("ekf", second_px=0.03),
        _bundle("robust_ekf", second_px=0.30),
    ]
    prediction_index = []
    for index, bundle in enumerate(prediction_bundles):
        prediction_path = prediction_root / f"prediction_{index}.json"
        prediction_path.write_text(json.dumps(bundle), encoding="utf-8")
        prediction_index.append(
            {
                "seq_id": bundle["seq_id"],
                "scene_id": bundle["scene_id"],
                "method_name": bundle["method_name"],
                "prediction_path": str(prediction_path),
            }
        )
    prediction_index_path.write_text(json.dumps(prediction_index), encoding="utf-8")

    # 验证 eval_pipeline 输出的关键产物文件存在
    metric_table_path = eval_root / "metrics" / "metric_table.csv"
    statistics_path = eval_root / "statistics" / "statistics_table.json"
    cases_path = eval_root / "cases" / "selected_cases.json"
    runtime_table_path = eval_root / "plotting_inputs" / "runtime_table.json"
    sweep_table_path = eval_root / "plotting_inputs" / "sweep_table.json"
    trajectory_bundle_path = eval_root / "plotting_inputs" / "trajectory_bundle.json"

    assert result.stage_name == "eval_pipeline"
    assert metric_table_path.is_file()
    assert statistics_path.is_file()
    assert cases_path.is_file()
    assert runtime_table_path.is_file()
    assert sweep_table_path.is_file()
    assert trajectory_bundle_path.is_file()

    # 验证 sweep_table 和 trajectory_bundle 的内容结构
    sweep_rows = json.loads(sweep_table_path.read_text(encoding="utf-8"))
    trajectory_bundle = json.loads(trajectory_bundle_path.read_text(encoding="utf-8"))
    assert sweep_rows and sweep_rows[0]["case_ref"].startswith("scene_00::")
    assert "prediction_bundle" in trajectory_bundle and "gt_bundle" in trajectory_bundle

    # 加载并运行 script19
    script19 = _load_script19()
    script19.INPUT_INDEX_PATH = prediction_index_path
    script19.GROUND_TRUTH_ROOT = FIXTURE_ROOT
    script19.OUTPUT_ROOT = verification_root

    # 验证 script19 正常退出
    assert script19.main() == 0
    stdout_payload = _load_stdout_json(capsys.readouterr().out)

    # 验证 script19 输出的状态和图表清单
    plotting_root = verification_root / "plotting_consume_check"
    assert stdout_payload["status"] == "ok"
    assert stdout_payload["plot_manifests"]["sweep"]["kind"] == "bar"
    assert stdout_payload["plot_manifests"]["trajectory"]["figure_path"] == str(plotting_root / "trajectory.png")
    # 验证产物路径正确
    assert stdout_payload["artifacts"]["metric_table"] == str(verification_root / "metrics" / "metric_table.csv")
    assert stdout_payload["artifacts"]["sweep_table"] == str(verification_root / "plotting_inputs" / "sweep_table.json")
    assert stdout_payload["artifacts"]["trajectory_bundle"] == str(verification_root / "plotting_inputs" / "trajectory_bundle.json")
    # 验证所有图表文件已生成
    assert (plotting_root / "sweep.svg").is_file()
    assert (plotting_root / "trajectory.png").is_file()
    assert (plotting_root / "main_table.png").is_file()
    assert (plotting_root / "runtime.svg").is_file()
    assert (plotting_root / "calibration.svg").is_file()
    assert (plotting_root / "cases_main_cases.png").is_file()

    # 验证 metric_table 中包含 runtime 组指标
    with metric_table_path.open("r", encoding="utf-8", newline="") as handle:
        metric_rows = list(csv.DictReader(handle))
    assert {row["metric"] for row in metric_rows if row["group"] == "runtime"}


def test_script19_resolves_relative_prediction_paths_from_index_directory(tmp_path, capsys, monkeypatch):
    """验证 script19 能正确解析 prediction_index 中的相对路径。

    测试场景：prediction_index 中的 prediction_path 使用相对路径
    （如 "../predictions/prediction_0.json"），且当前工作目录
    不在 index 文件所在目录。
    预期行为：script19 基于 index 文件所在目录解析相对路径，
    正确读取 prediction 数据，返回退出码 0 和 status=ok。
    """
    module = _load_script19()
    route_root = tmp_path / "relative_route"
    eval_root = route_root / "eval"
    prediction_root = route_root / "predictions"
    audits_root = route_root / "audits"
    cwd_root = tmp_path / "cwd"

    # 写入固定的评估产物
    _write_fixed_eval_root(eval_root)
    prediction_root.mkdir(parents=True, exist_ok=True)
    audits_root.mkdir(parents=True, exist_ok=True)
    cwd_root.mkdir(parents=True, exist_ok=True)

    # 写入 prediction 文件
    prediction_path = prediction_root / "prediction_0.json"
    prediction_path.write_text(json.dumps(_bundle("ekf", second_px=0.03)), encoding="utf-8")
    # prediction_index 中使用相对路径
    (audits_root / "prediction_index.json").write_text(
        json.dumps(
            [
                {
                    "seq_id": "mini_seq",
                    "scene_id": "S(A2,N2,V2,K0,M0)",
                    "method_name": "ekf",
                    # 相对路径，基于 index 文件所在目录解析
                    "prediction_path": "../predictions/prediction_0.json",
                }
            ]
        ),
        encoding="utf-8",
    )

    # 切换工作目录到 cwd_root，验证相对路径解析不依赖 cwd
    monkeypatch.chdir(cwd_root)
    module.INPUT_INDEX_PATH = audits_root / "prediction_index.json"
    module.GROUND_TRUTH_ROOT = FIXTURE_ROOT
    module.OUTPUT_ROOT = route_root / "verification"

    # 验证 script19 能正确解析相对路径并正常退出
    assert module.main() == 0
    stdout_payload = _load_stdout_json(capsys.readouterr().out)
    assert stdout_payload["status"] == "ok"
    # 验证 route_bridge 中的 prediction_index 路径正确
    assert stdout_payload["route_bridge"]["prediction_index"] == str(audits_root / "prediction_index.json")
