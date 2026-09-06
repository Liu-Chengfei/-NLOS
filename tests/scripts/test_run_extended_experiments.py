from __future__ import annotations

"""扩展实验脚本（run_extended_experiments）测试模块。

测试覆盖范围：
- 扩展实验的流程与配置
- 实验参数的验证

被测模块：scripts.run_extended_experiments"""

import importlib.util
import json
from pathlib import Path

import pytest

from liquidloc.common.config_utils import load_yaml_config
from liquidloc.protocol.experiment_gates import get_public_benchmark_allowed_datasets
from liquidloc.scenarios.scene_sampler import sample_scenes


ROOT = Path(__file__).resolve().parents[2]


def _load_script():
    script_path = ROOT / "scripts" / "09_run_extended_experiments.py"
    spec = importlib.util.spec_from_file_location("run_extended_experiments_script", script_path)
    script = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(script)
    return script


def _extract_stdout_json(stdout: str):
    """从混合了诊断日志的 stdout 中提取脚本打印的 JSON 摘要。

    09_run_extended_experiments.py 脚本会向 stdout 打印诊断日志（print_args/
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


def _fake_core_result(methods: list[str]):
    return type(
        "Result",
        (),
        {
            "stage_name": "core_pipeline",
            "artifacts": [],
            "metadata": {
                "prediction_index": [{"method_name": method_name} for method_name in methods],
                "prediction_bundles": [{"method_name": method_name} for method_name in methods],
            },
        },
    )()


def _fake_public_result(methods: list[str]):
    return type(
        "Result",
        (),
        {
            "stage_name": "public_benchmark_pipeline",
            "artifacts": [],
            "metadata": {
                "prediction_bundles": [{"method_name": method_name, "seq_id": "mini_seq"} for method_name in methods],
                "public_benchmark_report": {"prediction_bundle_count": len(methods)},
            },
        },
    )()


def test_public_route_rejects_unsupported_dataset(tmp_path, monkeypatch):
    """拒绝测试：public route。\n\n验证被测功能对 public route 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    script = _load_script()

    class DummyPublicPipeline:
        def run(self, payload):
            raise AssertionError("unsupported public dataset should be rejected before public pipeline runs")

    monkeypatch.setattr(script, "PublicBenchmarkPipeline", lambda: DummyPublicPipeline())
    monkeypatch.setattr(script, "CorePipeline", lambda: (_ for _ in ()).throw(AssertionError("core pipeline should not run")))
    monkeypatch.setattr(script, "load_public_dataset_registry", lambda: {"datasets": {"miluv": {}}})
    with pytest.raises(ValueError, match=r"registered public dataset required"):
        script.main(
            [
                "--config",
                str(ROOT / "configs" / "experiments" / "e7_miluv.yaml"),
                "--dataset-name",
                "unsupported_dataset",
                "--seq-ids",
                "mini_seq",
                "--output-root",
                str(tmp_path / "out"),
            ]
        )


def test_public_route_normalizes_dataset_name_case(tmp_path, monkeypatch, capsys):
    """归一化测试：public route。\n\n验证 public route 的归一化处理，\n确保空白/重复等输入被正确清洗。
    """
    script = _load_script()
    captured = {}

    class DummyPublicPipeline:
        def run(self, payload):
            captured["payload"] = payload
            return _fake_public_result(payload["methods"])

    monkeypatch.setattr(script, "PublicBenchmarkPipeline", lambda: DummyPublicPipeline())
    monkeypatch.setattr(script, "CorePipeline", lambda: (_ for _ in ()).throw(AssertionError("core pipeline should not run")))

    assert script.main(
        [
            "--config",
            str(ROOT / "configs" / "experiments" / "e7_miluv.yaml"),
            "--dataset-name",
            "MILUV",
            "--seq-ids",
            "mini_seq",
            "--output-root",
            str(tmp_path / "out"),
        ]
    ) == 0

    payload = _extract_stdout_json(capsys.readouterr().out)
    assert payload["route"] == "public"
    assert payload["stage_name"] == "public_benchmark_pipeline"
    assert captured["payload"]["dataset_name"] == "miluv"


@pytest.mark.parametrize(
    "config_name",
    ["e1_main_table.yaml", "e6_geometry.yaml", "e8_runtime.yaml", "e9_dual_degradation.yaml"],
)
def test_core_family_configs_use_declared_scene_surface(config_name, tmp_path, monkeypatch):
    script = _load_script()
    captured = {}
    experiment_cfg = load_yaml_config(ROOT / "configs" / "experiments" / config_name)
    experiment_cfg["mode"] = "quick"
    expected_scene_tasks = sample_scenes(experiment_cfg)

    class DummyCorePipeline:
        def run(self, payload):
            captured["payload"] = payload
            return _fake_core_result(payload["methods"])

    monkeypatch.setattr(script, "CorePipeline", lambda: DummyCorePipeline())
    monkeypatch.setattr(script, "PublicBenchmarkPipeline", lambda: (_ for _ in ()).throw(AssertionError("public pipeline should not run")))
    assert script.main(["--config", str(ROOT / "configs" / "experiments" / config_name), "--output-root", str(tmp_path / "out")]) == 0

    assert captured["payload"]["scene_tasks"] == expected_scene_tasks
    assert set(captured["payload"]["events_by_scene_id"]) == {task["scene_id"] for task in expected_scene_tasks}


def test_e5_ablation_alias_kept(tmp_path, monkeypatch):
    """别名测试：e5 ablation。\n\n验证 e5 ablation 的别名兼容性，\n确保旧参数名仍可使用。
    """
    script = _load_script()
    captured = {}
    experiment_cfg = load_yaml_config(ROOT / "configs" / "experiments" / "e5_ablation.yaml")
    experiment_cfg["mode"] = "quick"
    expected_axes = dict(experiment_cfg["frozen_axes"])

    class DummyCorePipeline:
        def run(self, payload):
            captured["payload"] = payload
            return _fake_core_result(payload["methods"])

    monkeypatch.setattr(script, "CorePipeline", lambda: DummyCorePipeline())
    monkeypatch.setattr(script, "PublicBenchmarkPipeline", lambda: (_ for _ in ()).throw(AssertionError("public pipeline should not run")))
    assert script.main(["--config", str(ROOT / "configs" / "experiments" / "e5_ablation.yaml"), "--output-root", str(tmp_path / "out")]) == 0

    assert captured["payload"]["scene_tasks"][0]["axes"] == expected_axes
    assert captured["payload"]["methods"] == experiment_cfg["methods"]


def test_full_mode_runs_real_pipeline(tmp_path, capsys, monkeypatch):
    """模式测试：full。\n\n验证 full 的模式验证，\n确保仅允许 quick/full 模式。
    """
    script = _load_script()
    captured = {}

    class DummyCorePipeline:
        def run(self, payload):
            captured["payload"] = payload
            return _fake_core_result(payload["methods"])

    monkeypatch.setattr(script, "CorePipeline", lambda: DummyCorePipeline())
    monkeypatch.setattr(script, "PublicBenchmarkPipeline", lambda: (_ for _ in ()).throw(AssertionError("public pipeline should not run")))
    assert script.main(
        [
            "--config",
            str(ROOT / "configs" / "experiments" / "e8_runtime.yaml"),
            "--mode",
            "full",
            "--output-root",
            str(tmp_path / "out"),
        ]
    ) == 0

    payload = _extract_stdout_json(capsys.readouterr().out)
    assert payload["route"] == "core"
    assert payload["stage_name"] == "core_pipeline"
    assert captured["payload"]["experiment_cfg"]["mode"] == "full"
