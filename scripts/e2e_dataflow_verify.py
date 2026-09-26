"""验证配置、协议和场景参数之间的最小端到端数据流。

该入口故意保持为可导入模块：测试和其他脚本可以读取模块级结果，
命令行执行时则通过 :func:`main` 打印逐步报告。所有路径均由统一
bootstrap 解析，避免从不同当前工作目录启动时加载错误配置。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:
    from scripts._bootstrap import bootstrap_project
except ModuleNotFoundError:  # 直接以 ``python scripts/e2e_dataflow_verify.py`` 启动
    from _bootstrap import bootstrap_project

_LAYOUT = bootstrap_project(__file__)
PROJECT_ROOT = _LAYOUT.root
CONFIG_ROOT = _LAYOUT.configs / "base"

EXPECTED_YAMLS = (
    "task.yaml",
    "sensors.yaml",
    "experiment_protocol.yaml",
    "metrics.yaml",
    "scene_axis_protocol.yaml",
)

# 导出结果保持稳定，供 tests/scripts/test_e2e_dataflow_verify.py 和外部审计使用。
step1_results: dict[str, bool] = {}
step2_results: dict[str, bool] = {}
step3_results: dict[str, bool] = {}
yaml_files: dict[str, Path] = {}
loaded_configs: dict[str, Any] = {}
scene_params: Any | None = None


def mark(ok: bool) -> str:
    """将校验布尔值格式化为稳定的 PASS/FAIL 标记。"""
    return "[PASS]" if bool(ok) else "[FAIL]"


def _load_yaml(name: str) -> bool:
    path = CONFIG_ROOT / name
    yaml_files[name] = path
    try:
        from liquidloc.common.config_utils import load_yaml_config

        loaded_configs[name] = load_yaml_config(path)
        return True
    except Exception as exc:  # 单项失败要记录，剩余步骤仍可给出完整报告。
        loaded_configs[name] = None
        print(f"[FAIL] {name}: {exc}", file=sys.stderr)
        return False


def _validate_protocols() -> None:
    """通过协议层公开 loader 校验四类冻结配置。"""
    step2_results.clear()
    validators = (
        ("validate_task_config", "liquidloc.protocol.task_contract", "load_task_contract", "task.yaml"),
        ("validate_sensor_config", "liquidloc.protocol.sensor_contract", "load_sensor_contract", "sensors.yaml"),
        ("load_experiment_protocol", "liquidloc.protocol.experiment_gates", "load_experiment_protocol", "experiment_protocol.yaml"),
        ("load_scene_axis_protocol", "liquidloc.protocol.scene_axis_protocol", "load_scene_axis_protocol", "scene_axis_protocol.yaml"),
    )
    for key, module_name, function_name, filename in validators:
        try:
            module = __import__(module_name, fromlist=[function_name])
            loader = getattr(module, function_name)
            loader(yaml_files.get(filename, CONFIG_ROOT / filename))
            step2_results[key] = True
        except Exception as exc:
            step2_results[key] = False
            print(f"[FAIL] {key}: {exc}", file=sys.stderr)


def _generate_scene_params() -> None:
    global scene_params
    step3_results.clear()
    step3_results.update({"attach_scene_parameters": False, "轴值验证": False, "元数据验证": False})
    scene_params = None
    try:
        from liquidloc.protocol.scene_axis_protocol import (
            attach_scene_parameters,
            get_nominal_levels,
            load_scene_axis_protocol,
        )

        cfg = load_scene_axis_protocol(yaml_files.get("scene_axis_protocol.yaml", CONFIG_ROOT / "scene_axis_protocol.yaml"))
        scene_params = attach_scene_parameters(get_nominal_levels(), cfg)
        step3_results["attach_scene_parameters"] = True
        step3_results["轴值验证"] = set(scene_params.axes) == {"A", "N", "V", "K", "M"}
        metadata = scene_params.axis_metadata
        step3_results["元数据验证"] = (
            isinstance(metadata.get("V"), dict)
            and metadata["V"].get("range_semantics") == "sampling_envelope"
        )
    except Exception as exc:
        print(f"[FAIL] scene parameter generation: {exc}", file=sys.stderr)


def _run() -> None:
    step1_results.clear()
    for filename in EXPECTED_YAMLS:
        step1_results[filename] = _load_yaml(filename)
    _validate_protocols()
    _generate_scene_params()


def main() -> int:
    _run()
    print("=== LiquidLoc 端到端数据流验证 ===")
    for label, results in (("步骤1", step1_results), ("步骤2", step2_results), ("步骤3", step3_results)):
        print(f"\n[{label}]")
        for name, ok in results.items():
            print(f"  {name}: {mark(ok)}")
    all_ok = all(step1_results.values()) and all(step2_results.values()) and all(step3_results.values())
    print(f"\n=== 结论 ===\n{mark(all_ok)} 端到端验证{'' if all_ok else '未'}全部通过")
    return 0 if all_ok else 1


_run()

if __name__ == "__main__":
    raise SystemExit(main())
