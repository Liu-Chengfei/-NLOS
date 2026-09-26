"""论文推理入口 checkpoint 配置边界测试。"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts._paper_inference import resolve_checkpoint_overrides


ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINTS = (
    ROOT / "scripts" / "24_direct_paper_inference.py",
    ROOT / "scripts" / "24_min_paper_inference.py",
    ROOT / "scripts" / "24_run_paper_inference.py",
)


def test_resolve_checkpoint_overrides_prefers_named_checkpoint(tmp_path: Path):
    checkpoint_dir = tmp_path / "seed3" / "liquid_ekf" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    preferred = checkpoint_dir / "liquid_ekf_resume_checkpoint.pt"
    preferred.write_bytes(b"checkpoint")
    (checkpoint_dir / "liquid_ekf_epoch_999.pt").write_bytes(b"fallback")

    overrides = resolve_checkpoint_overrides(("ekf", "liquid_ekf"), tmp_path, seed=3)

    assert overrides == {"liquid_ekf": {"checkpoint_path": preferred.resolve().as_posix()}}


def test_resolve_checkpoint_overrides_fails_before_untrained_inference(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="liquid_ekf"):
        resolve_checkpoint_overrides(("liquid_ekf",), tmp_path)


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS, ids=lambda path: path.name)
def test_paper_inference_entrypoints_never_rewrite_model_yaml(entrypoint: Path):
    source = entrypoint.read_text(encoding="utf-8")

    assert "model_cfgs" in source
    assert "25_fix_yaml" not in source
    assert "safe_dump" not in source
    assert "yaml_path.write_text" not in source
