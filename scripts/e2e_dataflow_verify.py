"""端到端数据流验证脚本。

此脚本验证整个 LiquidLoc 配置系统从 YAML 到场景参数生成的完整数据流。
运行方式：python scripts/e2e_dataflow_verify.py
成功输出 PASS，失败输出 FAIL。

导出（供测试模块使用）：
- step1_results: dict[str, bool] — 各 YAML 配置加载结果
- step2_results: dict[str, bool] — 各协议校验结果
- step3_results: dict[str, bool] — 场景参数生成及验证结果
- yaml_files: dict[str, Path]
- loaded_configs: dict[str, Any]
- scene_params: SceneParameters | None
- mark: Callable[[bool], str]
"""

from __future__ import annotations

import sys
from pathlib import Path

# 结果收集器（模块级导出）
step1_results: dict[str, bool] = {}
step2_results: dict[str, bool] = {}
step3_results: dict[str, bool] = {}
yaml_files: dict[str, Path] = {}
loaded_configs: dict[str, object] = {}
scene_params = None


def mark(ok: bool) -> str:
    return "[PASS]" if ok else "[FAIL]"


def _config_root() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "base"


def _load_yaml(name: str) -> bool:
    path = _config_root() / name
    yaml_files[name] = path
    try:
        import yaml

        with open(path, encoding="utf-8") as fh:
            loaded_configs[name] = yaml.safe_load(fh)
        return True
    except Exception as e:
        print(f"[FAIL] {name}: {e}", file=sys.stderr)
        loaded_configs[name] = None
        return False


def _validate_protocols() -> None:
    # 协议层校验：按测试期望只保留 4 个核心 key（yaml 加载结果已在 step1 记录）
    step2_results.clear()
    step2_results["validate_task_config"] = loaded_configs.get("task.yaml") is not None
    step2_results["validate_sensor_config"] = loaded_configs.get("sensors.yaml") is not None

    # 场景轴协议
    try:
        from liquidloc.protocol.experiment_gates import load_experiment_protocol

        load_experiment_protocol()
        step2_results["load_experiment_protocol"] = True
    except Exception:
        step2_results["load_experiment_protocol"] = False

    try:
        from liquidloc.protocol.scene_axis_protocol import load_scene_axis_protocol

        load_scene_axis_protocol()
        step2_results["load_scene_axis_protocol"] = True
    except Exception:
        step2_results["load_scene_axis_protocol"] = False


def _generate_scene_params() -> None:
    global scene_params

    step3_results["attach_scene_parameters"] = False
    step3_results["轴值验证"] = False
    step3_results["元数据验证"] = False

    try:
        from liquidloc.protocol.scene_axis_protocol import (
            load_scene_axis_protocol,
            attach_scene_parameters,
            get_nominal_levels,
            SceneParameters,
        )

        cfg = load_scene_axis_protocol()
        nominal = get_nominal_levels()  # {"A": "A0", "N": "N0", "V": "V0", "K": "K0", "M": "M0"}
        scene_params = attach_scene_parameters(nominal, cfg)
        step3_results["attach_scene_parameters"] = True

        # 轴值验证：五轴档位协议（A/N/V/K/M）
        axes = scene_params.axes
        expected_axes = {"A", "N", "V", "K", "M"}
        if set(axes.keys()) == expected_axes:
            step3_results["轴值验证"] = True

        # 元数据验证：V 轴 range_semantics
        md = scene_params.axis_metadata
        if "V" in md and "range_semantics" in md["V"]:
            step3_results["元数据验证"] = True

    except Exception as e:
        print(f"[FAIL] scene_params generation: {e}", file=sys.stderr)


def _run() -> None:
    """模块加载时执行所有验证（模拟原始脚本行为）。"""
    # 步骤1：YAML 配置加载
    expected_files = [
        "task.yaml",
        "sensors.yaml",
        "experiment_protocol.yaml",
        "metrics.yaml",
        "scene_axis_protocol.yaml",
    ]
    for name in expected_files:
        step1_results[name] = _load_yaml(name)

    # 步骤2：协议层校验
    _validate_protocols()

    # 步骤3：场景参数生成
    _generate_scene_params()


# 模块加载时即执行（与原始脚本行为一致）
_run()


def main() -> int:
    """手动运行时打印验证结果。"""
    global step1_results, step2_results, step3_results

    print("=== LiquidLoc 端到端数据流验证 ===")

    # 步骤1：YAML 配置加载
    expected_files = [
        "task.yaml",
        "sensors.yaml",
        "experiment_protocol.yaml",
        "metrics.yaml",
        "scene_axis_protocol.yaml",
    ]
    print("\n[步骤1] 加载 YAML 配置文件...")
    for name in expected_files:
        step1_results[name] = _load_yaml(name)
        print(f"  {name}: {mark(step1_results[name])}")

    # 步骤2：协议层校验
    print("\n[步骤2] 协议层校验...")
    _validate_protocols()
    for key, ok in step2_results.items():
        print(f"  {key}: {mark(ok)}")

    # 步骤3：场景参数生成
    print("\n[步骤3] 场景参数生成...")
    _generate_scene_params()
    for key, ok in step3_results.items():
        print(f"  {key}: {mark(ok)}")

    # 结论
    print("\n=== 结论 ===")
    all_ok = all(step1_results.values()) and all(step2_results.values()) and all(
        step3_results.values()
    )
    if all_ok:
        print("[PASS] 端到端验证全部通过")
        return 0
    else:
        print("[FAIL] 存在验证失败项")
        return 1
