from __future__ import annotations

"""MILUV 数据准备脚本（prepare_miluv_data）测试模块。

测试覆盖范围：
- MILUV 数据的下载与准备流程
- 数据路径与格式验证

被测模块：scripts.prepare_miluv_data"""

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


def test_script_smoke(tmp_path):
    """冒烟测试：script。\n\n快速验证 script 的基本功能可用，\n不深入检查细节，仅确认流程不崩溃。
    """
    module = _load_module("03_prepare_miluv_data.py", "prepare_miluv_script")
    project_root = tmp_path / "project"
    seq_dir = project_root / "data" / "raw" / "miluv" / "mini_seq"
    seq_dir.mkdir(parents=True)
    (seq_dir / "imu.json").write_text('[{"timestamp": 0.0, "ax": 0.1, "ay": 0.0, "gz": 0.01}]', encoding="utf-8")
    (seq_dir / "uwb.json").write_text('[{"timestamp": 0.05, "anchor_id": 0, "range": 2.0, "valid": true, "quality": 0.95}]', encoding="utf-8")
    (seq_dir / "vio.json").write_text('[{"timestamp": 0.08, "dx": 0.03, "dy": 0.0, "dyaw": 0.0, "quality": 0.85, "tracked_features": 60, "reproj_err": 0.4}]', encoding="utf-8")
    (seq_dir / "gt.json").write_text('[{"timestamp": 0.0, "px": 0.0, "py": 0.0, "yaw": 0.0}]', encoding="utf-8")
    cfg_dir = project_root / "configs" / "datasets"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "miluv.yaml").write_text(
        "dataset_name: miluv\nraw_root: data/raw/miluv\ninterim_root: data/interim/miluv\nprocessed_root: data/processed\nfield_mapping:\n  imu:\n    timestamp: timestamp\n    ax: ax\n    ay: ay\n    gz: gz\n  uwb:\n    timestamp: timestamp\n    anchor_id: anchor_id\n    range: range\n    valid: valid\n    quality: quality\n  vio:\n    timestamp: timestamp\n    dx: dx\n    dy: dy\n    dyaw: dyaw\n    quality: quality\n    tracked_features: tracked_features\n    reproj_err: reproj_err\n  gt:\n    timestamp: timestamp\n    px: px\n    py: py\n    yaw: yaw\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "prepare_miluv"
    exit_code = module.main(["--project-root", str(project_root), "--output-root", str(output_root)])

    assert exit_code == 0
    payload = json.loads((output_root / "prepare_manifest.json").read_text(encoding="utf-8"))
    assert "mini_seq" in payload["sequences"]
    assert payload["sequences"]["mini_seq"]["event_count"] > 0
    assert (output_root / "mini_seq_events.pkl.gz").is_file()


def test_default_raw_root_uses_dataset_config(tmp_path):
    """使用测试：default raw root。\n\n验证被测功能正确使用 default raw root，\n确保内部依赖被正确调用。
    """
    module = _load_module("03_prepare_miluv_data.py", "prepare_miluv_script_default_raw")
    project_root = tmp_path / "project"
    seq_dir = project_root / "data" / "raw" / "miluv" / "mini_seq"
    seq_dir.mkdir(parents=True)
    (seq_dir / "imu.json").write_text('[{"timestamp": 0.0, "ax": 0.1, "ay": 0.0, "gz": 0.01}]', encoding="utf-8")
    (seq_dir / "uwb.json").write_text('[{"timestamp": 0.05, "anchor_id": 0, "range": 2.0, "valid": true, "quality": 0.95}]', encoding="utf-8")
    (seq_dir / "vio.json").write_text('[{"timestamp": 0.08, "dx": 0.03, "dy": 0.0, "dyaw": 0.0, "quality": 0.85, "tracked_features": 60, "reproj_err": 0.4}]', encoding="utf-8")
    (seq_dir / "gt.json").write_text('[{"timestamp": 0.0, "px": 0.0, "py": 0.0, "yaw": 0.0}]', encoding="utf-8")
    cfg_dir = project_root / "configs" / "datasets"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "miluv.yaml").write_text(
        "dataset_name: miluv\nraw_root: data/raw/miluv\ninterim_root: data/interim/miluv\nprocessed_root: data/processed\nfield_mapping:\n  imu:\n    timestamp: timestamp\n    ax: ax\n    ay: ay\n    gz: gz\n  uwb:\n    timestamp: timestamp\n    anchor_id: anchor_id\n    range: range\n    valid: valid\n    quality: quality\n  vio:\n    timestamp: timestamp\n    dx: dx\n    dy: dy\n    dyaw: dyaw\n    quality: quality\n    tracked_features: tracked_features\n    reproj_err: reproj_err\n  gt:\n    timestamp: timestamp\n    px: px\n    py: py\n    yaw: yaw\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "prepare_miluv_default"
    exit_code = module.main(["--project-root", str(project_root), "--output-root", str(output_root)])

    assert exit_code == 0
    payload = json.loads((output_root / "prepare_manifest.json").read_text(encoding="utf-8"))
    assert "mini_seq" in payload["sequences"]
    assert payload["sequences"]["mini_seq"]["event_count"] > 0
    assert (output_root / "mini_seq_events.pkl.gz").is_file()


def test_default_raw_root_ignores_config_and_hidden_dirs(tmp_path):
    module = _load_module("03_prepare_miluv_data.py", "prepare_miluv_script_dir_filter")
    project_root = tmp_path / "project"
    raw_root = project_root / "data" / "raw" / "miluv"
    seq_dir = raw_root / "mini_seq"
    seq_dir.mkdir(parents=True)
    (seq_dir / "imu.json").write_text('[{"timestamp": 0.0, "ax": 0.1, "ay": 0.0, "gz": 0.01}]', encoding="utf-8")
    (seq_dir / "uwb.json").write_text('[{"timestamp": 0.05, "anchor_id": 0, "range": 2.0, "valid": true, "quality": 0.95}]', encoding="utf-8")
    (seq_dir / "vio.json").write_text('[{"timestamp": 0.08, "dx": 0.03, "dy": 0.0, "dyaw": 0.0, "quality": 0.85, "tracked_features": 60, "reproj_err": 0.4}]', encoding="utf-8")
    (seq_dir / "gt.json").write_text('[{"timestamp": 0.0, "px": 0.0, "py": 0.0, "yaw": 0.0}]', encoding="utf-8")
    (raw_root / "config").mkdir()
    (raw_root / ".cache").mkdir()
    cfg_dir = project_root / "configs" / "datasets"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "miluv.yaml").write_text(
        "dataset_name: miluv\nraw_root: data/raw/miluv\nfield_mapping:\n  imu:\n    timestamp: timestamp\n    ax: ax\n    ay: ay\n    gz: gz\n  uwb:\n    timestamp: timestamp\n    anchor_id: anchor_id\n    range: range\n    valid: valid\n    quality: quality\n  vio:\n    timestamp: timestamp\n    dx: dx\n    dy: dy\n    dyaw: dyaw\n    quality: quality\n    tracked_features: tracked_features\n    reproj_err: reproj_err\n  gt:\n    timestamp: timestamp\n    px: px\n    py: py\n    yaw: yaw\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "prepare_miluv_filtered"
    exit_code = module.main(["--project-root", str(project_root), "--output-root", str(output_root)])

    assert exit_code == 0
    payload = json.loads((output_root / "prepare_manifest.json").read_text(encoding="utf-8"))
    assert list(payload["sequences"]) == ["mini_seq"]


def test_blank_paths_and_blank_dataset_raw_root_are_rejected(tmp_path):
    """拒绝测试：blank paths and blank dataset raw root are。\n\n验证被测功能对不合法的 blank paths and blank dataset raw root are 输入正确返回失败，\n防止无效参数通过验证。
    """
    module = _load_module("03_prepare_miluv_data.py", "prepare_miluv_script_validation")
    project_root = tmp_path / "project"
    cfg_dir = project_root / "configs" / "datasets"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "miluv.yaml").write_text(
        "dataset_name: miluv\nraw_root: \"  \"\nfield_mapping: {}\n",
        encoding="utf-8",
    )

    for argv in (
        ["--project-root", " ", "--output-root", str(tmp_path / "out")],
        ["--project-root", str(project_root), "--raw-root", " ", "--output-root", str(tmp_path / "out")],
        ["--project-root", str(project_root), "--output-root", " "],
    ):
        exit_code = module.main(argv)
        assert exit_code == 1, f"expected exit_code 1 for argv={argv!r}"

    exit_code = module.main(["--project-root", str(project_root), "--output-root", str(tmp_path / "out")])
    assert exit_code == 1, "expected exit_code 1 for blank dataset raw_root"


def test_blank_field_mapping_is_rejected_before_pipeline_run(tmp_path):
    """拒绝测试：blank field mapping is。\n\n验证被测功能对不合法的 blank field mapping is 输入正确返回失败，\n防止无效参数通过验证。
    """
    module = _load_module("03_prepare_miluv_data.py", "prepare_miluv_script_field_mapping")
    project_root = tmp_path / "project"
    seq_dir = project_root / "data" / "raw" / "miluv" / "mini_seq"
    cfg_dir = project_root / "configs" / "datasets"
    seq_dir.mkdir(parents=True)
    (seq_dir / "imu.json").write_text('[{"timestamp": 0.0, "ax": 0.1, "ay": 0.0, "gz": 0.01}]', encoding="utf-8")
    (seq_dir / "uwb.json").write_text('[{"timestamp": 0.05, "anchor_id": 0, "range": 2.0, "valid": true, "quality": 0.95}]', encoding="utf-8")
    (seq_dir / "vio.json").write_text('[{"timestamp": 0.08, "dx": 0.03, "dy": 0.0, "dyaw": 0.0, "quality": 0.85, "tracked_features": 60, "reproj_err": 0.4}]', encoding="utf-8")
    (seq_dir / "gt.json").write_text('[{"timestamp": 0.0, "px": 0.0, "py": 0.0, "yaw": 0.0}]', encoding="utf-8")
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "miluv.yaml").write_text(
        "dataset_name: miluv\nraw_root: data/raw/miluv\nfield_mapping: {}\n",
        encoding="utf-8",
    )

    exit_code = module.main(["--project-root", str(project_root), "--output-root", str(tmp_path / "out")])
    assert exit_code == 1, "expected exit_code 1 for blank field_mapping"


def test_missing_raw_root_is_rejected_at_prepare_gate(tmp_path):
    """拒绝测试：missing raw root is。\n\n验证被测功能对不合法的 missing raw root is 输入正确返回失败，\n防止无效参数通过验证。
    """
    module = _load_module("03_prepare_miluv_data.py", "prepare_miluv_script_missing_raw_root")
    project_root = tmp_path / "project"
    cfg_dir = project_root / "configs" / "datasets"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "miluv.yaml").write_text(
        "dataset_name: miluv\nraw_root: data/raw/miluv\nfield_mapping:\n  imu:\n    timestamp: timestamp\n",
        encoding="utf-8",
    )

    exit_code = module.main(["--project-root", str(project_root), "--output-root", str(tmp_path / "out")])
    assert exit_code == 1, "expected exit_code 1 for missing raw_root"
