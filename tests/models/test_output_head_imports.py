"""输出头的分层与向后兼容导入测试。"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_model_output_heads_do_not_import_factory_and_factory_reexports_them():
    """模型层可独立导入，工厂旧路径仍重导出同一批类。"""
    project_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root / "src")
    script = """
import sys

from liquidloc.models.liquid import output_head
from liquidloc.models.transformer import network as transformer_network

assert "liquidloc.factories.model_factory" not in sys.modules
assert transformer_network.LiquidOutputHeadLinear is output_head.LiquidOutputHeadLinear

from liquidloc.factories import model_factory

assert model_factory.LiquidOutputHead is output_head.LiquidOutputHead
assert model_factory.LiquidOutputBackbone is output_head.LiquidOutputBackbone
assert model_factory.LiquidOutputHeadLinear is output_head.LiquidOutputHeadLinear
assert model_factory.RiskCalibration is output_head.RiskCalibration
assert model_factory._LiquidOutputHead is output_head.LiquidOutputHead
assert model_factory._LiquidOutputBackbone is output_head.LiquidOutputBackbone
assert model_factory._LiquidOutputHeadLinear is output_head.LiquidOutputHeadLinear
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
