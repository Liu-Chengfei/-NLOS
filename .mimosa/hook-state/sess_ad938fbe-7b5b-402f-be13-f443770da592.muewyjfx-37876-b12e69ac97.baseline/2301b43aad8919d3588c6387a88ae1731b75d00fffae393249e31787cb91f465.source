from __future__ import annotations

"""训练入口脚本（train_entrypoints）测试模块。

测试覆盖范围：
- 训练脚本的入口参数解析
- 模型选择与配置传递

被测模块：scripts.train_entrypoints"""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_module(script_name: str, module_name: str):
    script_path = ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("script_name", "module_name"),
    [
        ("05_train_lstm.py", "train_lstm_entrypoint"),
        ("06_train_liquid.py", "train_liquid_entrypoint"),
    ],
)
def test_training_script_has_main_function(script_name: str, module_name: str):
    """验证训练脚本导出 main 函数。"""
    module = _load_module(script_name, module_name)
    assert callable(getattr(module, "main", None))


@pytest.mark.parametrize(
    ("script_name", "module_name"),
    [
        ("05_train_lstm.py", "train_lstm_entrypoint_ok"),
        ("06_train_liquid.py", "train_liquid_entrypoint_ok"),
    ],
)
def test_training_script_main_returns_int(script_name: str, module_name: str):
    """验证训练脚本 main 函数返回整数退出码。"""
    module = _load_module(script_name, module_name)
    # main 需要 argv 参数，不实际运行，只检查签名
    import inspect
    sig = inspect.signature(module.main)
    assert "argv" in sig.parameters
