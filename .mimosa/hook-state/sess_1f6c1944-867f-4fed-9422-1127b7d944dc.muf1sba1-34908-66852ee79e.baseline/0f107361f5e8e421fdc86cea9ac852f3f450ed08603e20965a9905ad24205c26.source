"""端到端数据流验证脚本测试。

测试覆盖范围：
- YAML 配置加载（task/sensors/experiment_protocol/metrics/scene_axis_protocol）
- 协议层校验（task_contract/sensor_contract/experiment_gates/scene_axis_protocol）
- 场景参数生成（attach_scene_parameters + 轴值/元数据验证）
- 脚本输出格式与结论标记

被测模块：scripts.e2e_dataflow_verify
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_e2e_module():
    """加载 e2e_dataflow_verify.py 模块（导入时即执行全部步骤）。

    脚本会替换 sys.stdout/stderr 以修复 Windows 编码问题，
    需要在加载前保存、加载后恢复，避免破坏 pytest 捕获。
    """
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    saved_textiowrapper = io.TextIOWrapper

    class _PassthroughWrapper:
        """透传包装器，避免关闭 pytest 的真实 stdout buffer。"""
        def __init__(self, buffer, **kwargs):
            self._buffer = buffer

        def write(self, data):
            return sys.__stdout__.write(data)

        def flush(self):
            sys.__stdout__.flush()

        def encoding(self):
            return "utf-8"

    try:
        io.TextIOWrapper = _PassthroughWrapper  # type: ignore[misc]
        path = ROOT / "scripts" / "e2e_dataflow_verify.py"
        spec = importlib.util.spec_from_file_location("e2e_dataflow_verify", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        io.TextIOWrapper = saved_textiowrapper
        sys.stdout, sys.stderr = saved_stdout, saved_stderr


@pytest.fixture(scope="module")
def e2e_module():
    """加载一次模块，所有测试共享。"""
    return _load_e2e_module()


def test_step1_yaml_configs_all_loaded(e2e_module):
    """步骤1：所有 YAML 配置文件成功加载。"""
    expected_files = [
        "task.yaml",
        "sensors.yaml",
        "experiment_protocol.yaml",
        "metrics.yaml",
        "scene_axis_protocol.yaml",
    ]
    assert set(e2e_module.step1_results.keys()) == set(expected_files)
    for name, ok in e2e_module.step1_results.items():
        assert ok, f"YAML 配置 {name} 加载失败"


def test_step2_protocol_validations_all_passed(e2e_module):
    """步骤2：所有协议层校验通过。"""
    expected_keys = [
        "validate_task_config",
        "validate_sensor_config",
        "load_experiment_protocol",
        "load_scene_axis_protocol",
    ]
    assert set(e2e_module.step2_results.keys()) == set(expected_keys)
    for key, ok in e2e_module.step2_results.items():
        assert ok, f"协议校验 {key} 失败"


def test_step3_scene_parameters_generated(e2e_module):
    """步骤3：场景参数成功生成。"""
    assert e2e_module.step3_results["attach_scene_parameters"] is True


def test_step3_axis_value_verification_passed(e2e_module):
    """步骤3：轴值验证通过。"""
    assert e2e_module.step3_results["轴值验证"] is True


def test_step3_metadata_verification_passed(e2e_module):
    """步骤3：元数据验证通过。"""
    assert e2e_module.step3_results["元数据验证"] is True


def test_overall_conclusion_is_pass(e2e_module, capsys):
    """端到端结论为 PASS。"""
    # 模块在导入时已输出，capsys 无法捕获历史输出
    # 通过检查所有步骤结果间接验证结论
    all_ok = (
        all(e2e_module.step1_results.values())
        and all(e2e_module.step2_results.values())
        and all(e2e_module.step3_results.values())
    )
    assert all_ok, "端到端验证未全部通过"


def test_mark_helper(e2e_module):
    """mark 辅助函数正确返回标记字符串。"""
    assert e2e_module.mark(True) == "[PASS]"
    assert e2e_module.mark(False) == "[FAIL]"


def test_yaml_files_dict_contains_all_expected(e2e_module):
    """yaml_files 字典包含所有预期的配置文件路径。"""
    expected_names = {
        "task.yaml",
        "sensors.yaml",
        "experiment_protocol.yaml",
        "metrics.yaml",
        "scene_axis_protocol.yaml",
    }
    assert set(e2e_module.yaml_files.keys()) == expected_names
    for name, path in e2e_module.yaml_files.items():
        assert isinstance(path, Path)
        assert path.name == name


def test_loaded_configs_contain_yaml_data(e2e_module):
    """loaded_configs 字典包含已加载的配置数据。"""
    for name in e2e_module.yaml_files:
        assert name in e2e_module.loaded_configs, f"{name} 不在 loaded_configs 中"
        assert e2e_module.loaded_configs[name] is not None, f"{name} 配置为 None"


def test_scene_params_axes_contain_five_axes(e2e_module):
    """生成的场景参数包含 A/N/V/K/M 五个轴（遵循五轴档位协议，无 G 轴）。"""
    scene_params = e2e_module.scene_params
    assert scene_params is not None
    expected_axes = {"A", "N", "V", "K", "M"}  # 五轴档位协议：G 已移除
    assert set(scene_params.axes.keys()) == expected_axes


def test_scene_params_axis_labels_correct(e2e_module):
    """各轴的 label 值正确（遵循五轴档位协议，无 G 轴）。"""
    axes = e2e_module.scene_params.axes
    assert axes["A"]["label"] == "near_sync"
    assert axes["N"]["label"] == "nominal_nlos"  # N 轴为 NLOS 场景（非 LOS）
    assert axes["V"]["label"] == "nominal_vio"
    assert axes["K"]["label"] == "anchor_count_4_symmetric"  # K0 标签（不是 anchor_count_4）


def test_scene_params_dict_style_access(e2e_module):
    """SceneParameters 支持 dict 风格访问。"""
    sp = e2e_module.scene_params
    assert "axes" in sp
    assert sp["axes"] is sp.axes
    assert sp.get("flat") is sp.flat


def test_scene_params_axis_metadata_has_v_range_semantics(e2e_module):
    """axis_metadata 包含 V 轴的 range_semantics。"""
    metadata = e2e_module.scene_params.axis_metadata
    assert "V" in metadata
    assert "range_semantics" in metadata["V"]
    assert metadata["V"]["range_semantics"] == "sampling_envelope"
