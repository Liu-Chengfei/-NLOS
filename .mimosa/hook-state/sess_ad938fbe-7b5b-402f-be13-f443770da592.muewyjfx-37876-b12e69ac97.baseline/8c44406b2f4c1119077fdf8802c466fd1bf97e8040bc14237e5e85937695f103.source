"""Build split manifests and leak reports from manifests plus split rules."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_SPLIT_RULES = ROOT / "scripts" / "default_split_rules.yaml"
DEFAULT_MANIFESTS_ROOT = ROOT / "outputs" / "data_prep" / "manifests" / "miluv"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from liquidloc.common.config_utils import load_yaml_config
from liquidloc.common.io_utils import dumps_json_text, read_json, write_json
from liquidloc.dataio.manifests.split_builder import (
    _collect_record_scope_overlap_items,
    _collect_semantic_overlap_items,
    build_splits,
)


def _load_json(path: Path) -> dict:
    """Read one JSON object from disk."""
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _resolve_cli_path(raw_path: str | None, *, arg_name: str) -> Path | None:
    """Normalize an optional CLI path."""
    if raw_path is None:
        return None
    normalized = raw_path.strip()
    if not normalized:
        raise ValueError(f"{arg_name} must be a non-empty path")
    return Path(normalized).resolve()


def _load_manifests(manifests_root: Path) -> tuple[dict, dict]:
    """Load standard manifests or fall back to prepare_manifest."""
    dataset_manifest_path = manifests_root / "dataset_manifest.json"
    scene_manifest_path = manifests_root / "scene_manifest.json"
    if dataset_manifest_path.is_file() and scene_manifest_path.is_file():
        return _load_json(dataset_manifest_path), _load_json(scene_manifest_path)

    prepare_manifest_path = manifests_root / "prepare_manifest.json"
    prepare_manifest = _load_json(prepare_manifest_path)
    dataset_manifest = prepare_manifest.get("dataset_manifest")
    scene_manifest = prepare_manifest.get("scene_manifest")
    if not isinstance(dataset_manifest, dict) or not isinstance(scene_manifest, dict):
        raise TypeError(f"Expected dataset_manifest and scene_manifest mappings in {prepare_manifest_path}")
    return dataset_manifest, scene_manifest


def _normalize_explicit_split_rules(split_rules: dict) -> dict:
    """Normalize explicit split IDs into stable lists."""
    normalized_rules = dict(split_rules)
    explicit_ids = normalized_rules.get("explicit_ids")
    if not isinstance(explicit_ids, dict):
        return normalized_rules
    normalized_rules["explicit_ids"] = {
        "train": list(explicit_ids.get("train", [])),
        "val": list(explicit_ids.get("val", [])),
        "test": list(explicit_ids.get("test", [])),
    }
    return normalized_rules


def _build_explicit_split_manifest(dataset_manifest: dict, split_rules: dict) -> tuple[dict, dict] | None:
    """Build splits directly from explicit IDs when present."""
    explicit_ids = split_rules.get("explicit_ids")
    if not isinstance(explicit_ids, dict):
        return None
    explicit_rules = {
        "explicit_ids": {
            "train": list(explicit_ids.get("train", [])),
            "val": list(explicit_ids.get("val", [])),
            "test": list(explicit_ids.get("test", [])),
        }
    }
    return build_splits(dataset_manifest, explicit_rules)


def main(argv: list[str] | None = None) -> int:
    """Build split manifest and leak report."""
    print("[04_splits] 开始 | 构建划分清单", flush=True)
    parser = argparse.ArgumentParser(description="Build split manifest and leak report")
    parser.add_argument("--manifests-root", default=str(DEFAULT_MANIFESTS_ROOT))
    parser.add_argument("--split-config", default=str(DEFAULT_SPLIT_RULES))
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)
    from liquidloc.common.tee_logger import print_args, print_dict
    print_args(args, "04_build_splits")

    manifests_root = _resolve_cli_path(args.manifests_root, arg_name="--manifests-root")
    print("[04_splits] 加载清单和划分规则", flush=True)
    dataset_manifest, scene_manifest = _load_manifests(manifests_root)
    split_rules = _normalize_explicit_split_rules(
        load_yaml_config(_resolve_cli_path(args.split_config, arg_name="--split-config"))
    )
    print_dict(split_rules, "划分规则 (split_rules)")

    print("[04_splits] 构建划分", flush=True)
    explicit_split_result = _build_explicit_split_manifest(dataset_manifest, split_rules)
    if explicit_split_result is not None:
        split_manifest, leak_report = explicit_split_result
    else:
        split_manifest, leak_report = build_splits(dataset_manifest, split_rules)

    output_root = _resolve_cli_path(args.output_root, arg_name="--output-root") or manifests_root
    print("[04_splits] 写入产物", flush=True)
    output_root.mkdir(parents=True, exist_ok=True)
    split_manifest_path = output_root / "split_manifest.json"
    leak_report_path = output_root / "leak_report.json"
    print_dict(
        {
            "manifests_root": str(manifests_root),
            "output_root": str(output_root),
            "split_manifest_path": str(split_manifest_path),
            "leak_report_path": str(leak_report_path),
        },
        "路径配置",
    )
    write_json(split_manifest_path, split_manifest)
    write_json(leak_report_path, leak_report)
    print(
        dumps_json_text(
            {
                "split_manifest": str(split_manifest_path),
                "leak_report": str(leak_report_path),
                "is_clean": leak_report["is_clean"],
            }
        )
    )
    _rc = 0 if leak_report["is_clean"] else 2
    print(f"[04_splits] 完成 | 返回码={_rc}", flush=True)
    return _rc


if __name__ == "__main__":
    raise SystemExit(main())
