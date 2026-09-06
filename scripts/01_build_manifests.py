"""Build dataset and scene manifests from raw data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from liquidloc.common.config_utils import load_yaml_config
from liquidloc.common.io_utils import dumps_json_text, write_json
from liquidloc.dataio.manifests.build_manifests import (
    REQUIRED_STREAMS,
    UTIL_REQUIRED_STREAMS,
    build_manifests,
)


def _dataset_config_raw_root_anchor(dataset_cfg_path: Path) -> Path:
    """Resolve the anchor used for relative raw_root values."""
    repo_dataset_cfg_dir = (ROOT / "configs" / "datasets").resolve()
    if dataset_cfg_path.is_relative_to(repo_dataset_cfg_dir):
        return ROOT
    return dataset_cfg_path.parent


def _resolve_cli_path(path_value: str | None, *, arg_name: str) -> Path | None:
    """Normalize an optional CLI path argument."""
    if path_value is None:
        return None
    if str(path_value).strip() == "":
        raise ValueError(f"{arg_name} must be a non-empty path")
    return Path(path_value).resolve()


def _resolve_data_root(dataset_cfg_path: Path, raw_root_override: str | None) -> Path:
    """Resolve the final raw data root."""
    override_root = _resolve_cli_path(raw_root_override, arg_name="--data-root")
    if override_root is not None:
        return override_root

    dataset_cfg_path = dataset_cfg_path.resolve()
    dataset_cfg = load_yaml_config(dataset_cfg_path)
    raw_root_value = dataset_cfg.get("raw_root")
    if raw_root_value is None or str(raw_root_value).strip() == "":
        raise ValueError(
            f"Dataset config '{dataset_cfg_path}' must define a non-empty raw_root when --data-root is not provided."
        )

    raw_root = Path(raw_root_value)
    if not raw_root.is_absolute():
        raw_root = _dataset_config_raw_root_anchor(dataset_cfg_path) / raw_root
    return raw_root.resolve()


def _resolve_default_output_root(dataset_cfg: dict) -> Path:
    """Resolve the default manifest output root for one dataset."""
    dataset_name = str(dataset_cfg.get("dataset_name") or "").strip()
    if not dataset_name:
        raise ValueError("dataset config must define a non-empty dataset_name")
    return (ROOT / "outputs" / "data_prep" / "manifests" / dataset_name).resolve()


def main(argv: list[str] | None = None) -> int:
    """Build and persist dataset / scene manifests."""
    print("[01_manifests] 开始 | 构建数据集/场景清单", flush=True)
    parser = argparse.ArgumentParser(description="Build dataset and scene manifests")
    parser.add_argument("--dataset-config", default=str(ROOT / "configs" / "datasets" / "miluv.yaml"))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)
    from liquidloc.common.tee_logger import print_args, print_dict
    print_args(args, "01_build_manifests")

    output_root = _resolve_cli_path(args.output_root, arg_name="--output-root")
    dataset_cfg_path = Path(args.dataset_config)
    data_root = _resolve_data_root(dataset_cfg_path, args.data_root)
    dataset_cfg = load_yaml_config(dataset_cfg_path)
    print_dict(dataset_cfg, "数据集配置 (miluv.yaml)")
    required_streams = UTIL_REQUIRED_STREAMS if dataset_cfg.get("dataset_name") == "util" else REQUIRED_STREAMS
    print_dict(
        {
            "required_streams": list(required_streams),
            "dataset_name": dataset_cfg.get("dataset_name"),
            "data_root": str(data_root),
            "output_root": str(output_root) if output_root else None,
        },
        "派生路径与流",
    )
    print("[01_manifests] 构建清单", flush=True)
    dataset_manifest, scene_manifest = build_manifests(data_root, required_streams=required_streams)
    if output_root is None:
        output_root = _resolve_default_output_root(dataset_cfg)

    print("[01_manifests] 写入产物", flush=True)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_manifest_path = output_root / "dataset_manifest.json"
    scene_manifest_path = output_root / "scene_manifest.json"
    write_json(dataset_manifest_path, dataset_manifest)
    write_json(scene_manifest_path, scene_manifest)
    print(
        dumps_json_text(
            {
                "data_root": str(data_root),
                "dataset_manifest": str(dataset_manifest_path),
                "scene_manifest": str(scene_manifest_path),
            }
        )
    )
    print("[01_manifests] 完成 | 返回码=0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
