from __future__ import annotations

"""NTU VIRAL 数据准备脚本（prepare_ntu_viral_data）测试模块。

测试覆盖范围：
- NTU VIRAL 数据的下载与准备流程
- 数据路径与格式验证

被测模块：scripts.prepare_ntu_viral_data"""

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_module(script_name: str, alias: str):
    path = ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _wrap_build_cfg_with_smoke(module):
    """Inject smoke_mode=True into pipeline_cfg for smoke tests."""
    original = module._build_pipeline_cfg
    def wrapped(project_root, raw_root, output_root):
        cfg = original(project_root, raw_root, output_root)
        cfg['smoke_mode'] = True
        return cfg
    module._build_pipeline_cfg = wrapped


def test_script_smoke(tmp_path):
    """冒烟测试：script。

    快速验证 script 的基本功能可用，
    不深入检查细节，仅确认流程不崩溃。
    """
    module = _load_module("03_prepare_ntu_viral_data.py", "prepare_ntu_viral_script")
    _wrap_build_cfg_with_smoke(module)
    project_root = tmp_path / "project"
    seq_dir = project_root / "data" / "raw" / "ntu_viral" / "mini_seq"
    seq_dir.mkdir(parents=True)
    (project_root / "configs" / "datasets").mkdir(parents=True)
    (project_root / "configs" / "datasets" / "ntu_viral.yaml").write_text(
        "dataset_name: ntu_viral\nraw_root: data/raw/ntu_viral\nfield_mapping:\n  imu:\n    timestamp: timestamp\n    ax: ax\n    ay: ay\n    gz: gz\n  uwb:\n    timestamp: timestamp\n    anchor_id: anchor_id\n    range: range\n    valid: valid\n    quality: quality\n  vio:\n    timestamp: timestamp\n    dx: dx\n    dy: dy\n    dyaw: dyaw\n    quality: quality\n    tracked_features: tracked_features\n    reproj_err: reproj_err\n  gt:\n    timestamp: timestamp\n    px: px\n    py: py\n    yaw: yaw\n",
        encoding="utf-8",
    )
    for filename, rows in {
        "imu.json": [{"timestamp": 0.0, "ax": 0.1, "ay": 0.0, "gz": 0.01}],
        "uwb.json": [{"timestamp": 0.1, "anchor_id": 1, "range": 2.0, "valid": True, "quality": 0.9}],
        "vio.json": [{"timestamp": 0.2, "dx": 0.05, "dy": 0.0, "dyaw": 0.0, "quality": 0.8, "tracked_features": 40, "reproj_err": 0.5}],
        "gt.json": [{"timestamp": 0.3, "px": 0.0, "py": 0.0, "yaw": 0.0}],
    }.items():
        (seq_dir / filename).write_text(json.dumps(rows), encoding="utf-8")

    output_root = tmp_path / "prepare_ntu_viral"
    exit_code = module.main(["--project-root", str(project_root), "--output-root", str(output_root)])

    assert exit_code == 0
    payload = json.loads((output_root / "prepare_manifest.json").read_text(encoding="utf-8"))
    assert "mini_seq" in payload["sequences"]
    assert payload["sequences"]["mini_seq"]["event_count"] > 0
    assert (output_root / "mini_seq_events.pkl.gz").is_file()
