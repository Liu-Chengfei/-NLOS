from __future__ import annotations

"""公开基准测试脚本（run_public_benchmarks）测试模块。

测试覆盖范围：
- 公开基准测试脚本的入口参数验证
- 数据集选择与协议门控

被测模块：scripts.run_public_benchmarks"""

import importlib.util
import json
from pathlib import Path

import pytest

from liquidloc.pipelines import public_benchmark_pipeline as public_pipeline


ROOT = Path(__file__).resolve().parents[2]


def _load_script():
    script_path = ROOT / "scripts" / "18_run_public_benchmarks.py"
    spec = importlib.util.spec_from_file_location("run_public_benchmarks_script", script_path)
    script = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(script)
    return script


def _extract_stdout_json(stdout: str):
    """从混合了诊断日志的 stdout 中提取脚本打印的 JSON 摘要。

    18_run_public_benchmarks.py 脚本会向 stdout 打印诊断日志（print_args/
    print_dict/阶段标记），机器可读的 JSON 摘要以独立的多行块形式打印（行首
    为 '{'）。这里扫描行首 '{' 并用 raw_decode 解析，返回最后一个解析成功的
    dict，兼容纯净 stdout。
    """
    decoder = json.JSONDecoder()
    payloads: list = []
    for idx in range(len(stdout)):
        if stdout[idx] != "{":
            continue
        if idx > 0 and stdout[idx - 1] != "\n":
            continue
        try:
            value, _end = decoder.raw_decode(stdout[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            payloads.append(value)
    assert payloads, f"no JSON object found in stdout: {stdout!r}"
    return payloads[-1]


def _prediction_bundle(
    *,
    seq_id: str = "mini_seq",
    scene_id: str = "miluv:mini_seq",
    method_name: str = "ekf",
    task_id: str = "public_00",
) -> dict[str, object]:
    return {
        "seq_id": seq_id,
        "scene_id": scene_id,
        "method_name": method_name,
        "task_id": task_id,
        "states": [{"px": 0.0, "py": 0.0}, {"px": 0.1, "py": 0.0}],
        "timestamps": [0.0, 0.1],
        "axes": {"dataset": "miluv", "split": "frozen_public_eval"},
        "scenario_context": {"split": "frozen_public_eval"},
        "runtime_log": {"latency": [1.0, 1.0], "params": 0.0, "ram_peak": 0.0},
    }


def test_script_smoke(tmp_path):
    """冒烟测试：script。\n\n快速验证 script 的基本功能可用，\n不深入检查细节，仅确认流程不崩溃。
    """
    script = _load_script()
    captured = {}
    public_result = type(
        "Result",
        (),
        {
            "stage_name": "public_benchmark_pipeline",
            "artifacts": [
                str(tmp_path / "public" / "reports" / "public_benchmark_report.json"),
                str(tmp_path / "public" / "core" / "predictions" / "public_00__ekf.json"),
            ],
            "metadata": {
                "public_benchmark_report": {"status": "ok"},
                "prediction_bundles": [_prediction_bundle()],
            },
        },
    )()

    class DummyPublicPipeline:
        def run(self, payload):
            captured["public"] = payload
            assert payload["seq_ids"] == ["mini_seq", "mini_seq_02"]
            return public_result

    class DummyEvalPipeline:
        def run(self, payload):
            captured["eval"] = payload
            assert payload["prediction_bundles"] == public_result.metadata["prediction_bundles"]
            assert payload["prediction_bundles"][0]["task_id"] == "public_00"
            assert payload["prediction_bundles"][0]["states"]
            return type(
                "Result",
                (),
                {
                    "stage_name": "eval_pipeline",
                    "artifacts": ["metric-table"],
                    "metadata": {"audit_report": {"best_method_by_rmse": "ekf"}},
                },
            )()

    script.PublicBenchmarkPipeline = lambda: DummyPublicPipeline()
    script.EvalPipeline = lambda: DummyEvalPipeline()

    output_root = tmp_path / "public"
    assert script.main(["--output-root", str(output_root)]) == 0
    assert captured["public"]["output_root"] == str(output_root)
    assert captured["eval"]["output_root"] == str(output_root / "eval")
    assert captured["eval"]["mode"] == "quick"
    assert (output_root / "eval").is_dir()
    assert public_result.artifacts[0].endswith("public_benchmark_report.json")


@pytest.mark.parametrize(
    ("mode", "expected_seq_ids"),
    [
        ("quick", ["mini_seq", "mini_seq_02"]),
        ("full", ["mini_seq", "mini_seq_02"]),
    ],
)
def test_script_real_prediction_bundles_smoke(mode, expected_seq_ids, tmp_path, monkeypatch, capsys):
    """冒烟测试：script real prediction bundles。\n\n快速验证 script real prediction bundles 的基本功能可用，\n不深入检查细节，仅确认流程不崩溃。
    """
    script = _load_script()
    captured = {}
    real_public_pipeline_cls = script.PublicBenchmarkPipeline
    output_root = tmp_path / f"public_real_{mode}"

    class PassthroughPublicPipeline:
        def run(self, payload):
            result = real_public_pipeline_cls().run(payload)
            captured["public_result"] = result
            captured["public_payload"] = payload
            return result

    class DummyEvalPipeline:
        def run(self, payload):
            captured["eval"] = payload
            bundles = payload["prediction_bundles"]
            assert bundles
            assert bundles == captured["public_result"].metadata["prediction_bundles"]
            assert list(dict.fromkeys(bundle["seq_id"] for bundle in bundles)) == expected_seq_ids
            prediction_artifacts = [
                Path(artifact_path)
                for artifact_path in captured["public_result"].artifacts
                if "predictions" in Path(artifact_path).parts
            ]
            assert prediction_artifacts
            first_bundle = bundles[0]
            assert first_bundle["task_id"].startswith("public_")
            assert first_bundle["method_name"]
            assert first_bundle["states"]
            matched_artifact = next(
                (
                    artifact_path
                    for artifact_path in prediction_artifacts
                    if artifact_path.is_file()
                    and artifact_path.stat().st_size > 0
                    and artifact_path.resolve().is_relative_to(output_root.resolve())
                    and artifact_path.stem.startswith(f"{first_bundle['task_id']}__{first_bundle['method_name']}")
                ),
                None,
            )
            assert matched_artifact is not None
            return type(
                "Result",
                (),
                {
                    "stage_name": "eval_pipeline",
                    "artifacts": ["metric-table"],
                    "metadata": {"audit_report": {"best_method_by_rmse": "ekf"}},
                },
            )()

    monkeypatch.setattr(script, "PublicBenchmarkPipeline", lambda: PassthroughPublicPipeline())
    monkeypatch.setattr(script, "EvalPipeline", lambda: DummyEvalPipeline())

    assert script.main(["--output-root", str(output_root), "--mode", mode]) == 0
    stdout_payload = _extract_stdout_json(capsys.readouterr().out)

    assert (output_root / "reports" / "public_benchmark_report.json").is_file()
    assert (output_root / "audits" / "public_protocol_gate.json").is_file()
    assert captured["public_result"].metadata["public_benchmark_report"]["prediction_bundle_count"] > 0
    assert captured["public_result"].artifacts
    assert all(str(Path(artifact_path)).startswith(str(output_root)) for artifact_path in captured["public_result"].artifacts)
    assert list(dict.fromkeys(bundle["seq_id"] for bundle in captured["public_result"].metadata["prediction_bundles"])) == expected_seq_ids
    assert captured["eval"]["ground_truth_root"] == str(ROOT / "tests" / "fixtures" / "datasets" / "miluv")
    assert captured["eval"]["output_root"] == str(output_root / "eval")
    assert captured["eval"]["mode"] == mode
    assert captured["eval"]["prediction_bundles"] == captured["public_result"].metadata["prediction_bundles"]
    assert stdout_payload["stage_name"] == "public_benchmark_closed_loop"
    assert stdout_payload["public_stage_name"] == "public_benchmark_pipeline"
    assert stdout_payload["eval_stage_name"] == "eval_pipeline"


def test_script_uses_dataset_specific_config_and_raw_root(monkeypatch, tmp_path):
    """使用测试：script。\n\n验证被测功能正确使用 script，\n确保内部依赖被正确调用。
    """
    script = _load_script()

    captured = {"public": None, "eval": None}

    class DummyPipeline:
        def run(self, payload):
            captured["public"] = payload
            assert payload["seq_ids"] == ["mini_seq", "mini_seq_02"]
            return type(
                "Result",
                (),
                {
                    "stage_name": "stage",
                    "artifacts": [],
                    "metadata": {
                        "public_benchmark_report": {},
                        "prediction_bundles": [_prediction_bundle()],
                    },
                },
            )()

    class DummyEvalPipeline:
        def run(self, payload):
            captured["eval"] = payload
            return type(
                "Result",
                (),
                {
                    "stage_name": "eval_pipeline",
                    "artifacts": [],
                    "metadata": {"audit_report": {}},
                },
            )()

    monkeypatch.setattr(script, "PublicBenchmarkPipeline", lambda: DummyPipeline())
    monkeypatch.setattr(script, "EvalPipeline", lambda: DummyEvalPipeline())
    monkeypatch.setattr(
        script,
        "_inspect_public_raw_readiness",
        lambda dataset_name, raw_root: {
            "dataset_name": dataset_name,
            "raw_root": str(raw_root),
            "status": "ready",
            "gate_action": "pass",
            "reasons": [],
            "sequence_count": 1,
            "ready_sequence_count": 1,
        },
    )

    output_root = tmp_path / "out"
    script.main(["--dataset-name", "miluv", "--output-root", str(output_root)])

    assert captured["public"]["dataset_name"] == "miluv"
    assert captured["public"]["raw_root"] == str(ROOT / "tests" / "fixtures" / "datasets" / "miluv")
    assert captured["public"]["methods"] == ["ekf", "robust_ekf", "fgo", "lstm_ekf", "liquid_ekf"]
    assert captured["eval"]["ground_truth_root"] == str(ROOT / "tests" / "fixtures" / "datasets" / "miluv")
    assert captured["eval"]["output_root"] == str(output_root / "eval")
    assert captured["eval"]["mode"] == "quick"


def test_script_normalizes_public_dataset_name_case(monkeypatch, tmp_path):
    """归一化测试：script。\n\n验证 script 的归一化处理，\n确保空白/重复等输入被正确清洗。
    """
    script = _load_script()

    captured = {"public": None, "eval": None}

    class DummyPipeline:
        def run(self, payload):
            captured["public"] = payload
            return type(
                "Result",
                (),
                {
                    "stage_name": "stage",
                    "artifacts": [],
                    "metadata": {
                        "public_benchmark_report": {},
                        "prediction_bundles": [_prediction_bundle()],
                    },
                },
            )()

    class DummyEvalPipeline:
        def run(self, payload):
            captured["eval"] = payload
            return type(
                "Result",
                (),
                {
                    "stage_name": "eval_pipeline",
                    "artifacts": [],
                    "metadata": {"audit_report": {}},
                },
            )()

    monkeypatch.setattr(script, "PublicBenchmarkPipeline", lambda: DummyPipeline())
    monkeypatch.setattr(script, "EvalPipeline", lambda: DummyEvalPipeline())
    monkeypatch.setattr(
        script,
        "_inspect_public_raw_readiness",
        lambda dataset_name, raw_root: {
            "dataset_name": dataset_name,
            "raw_root": str(raw_root),
            "status": "ready",
            "gate_action": "pass",
            "reasons": [],
            "sequence_count": 1,
            "ready_sequence_count": 1,
        },
    )

    output_root = tmp_path / "out"
    script.main(["--dataset-name", "MILUV", "--output-root", str(output_root)])

    assert captured["public"]["dataset_name"] == "miluv"
    assert captured["eval"]["ground_truth_root"] == str(ROOT / "tests" / "fixtures" / "datasets" / "miluv")


def test_script_blocks_unknown_public_dataset_before_pipeline(monkeypatch, tmp_path, capsys):
    """前置验证测试：script blocks unknown public dataset。\n\n验证 script blocks unknown public dataset 在后续操作前被正确检查，\n确保早期拦截无效输入。
    """
    script = _load_script()

    def _unexpected_load_yaml_config(_path):
        raise AssertionError("public benchmark script must reject unsupported datasets before loading configs")

    class BrokenPublicPipeline:
        def run(self, payload):
            raise AssertionError("public pipeline should not run for unsupported datasets")

    monkeypatch.setattr(script, "load_yaml_config", _unexpected_load_yaml_config)
    monkeypatch.setattr(script, "PublicBenchmarkPipeline", lambda: BrokenPublicPipeline())

    output_root = tmp_path / "unsupported_blocked"
    assert script.main(["--dataset-name", "util", "--raw-root", str(tmp_path / "raw"), "--output-root", str(output_root)]) == 2
    stdout_payload = _extract_stdout_json(capsys.readouterr().out)

    assert stdout_payload["status"] == "blocked"
    assert stdout_payload["stage"] == "data_readiness"
    assert stdout_payload["dataset_name"] == "util"
    assert stdout_payload["reasons"] == ["unsupported_public_dataset"]
