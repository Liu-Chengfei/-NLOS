"""运行 DTW/形状特征距离泄漏检查，并写入 sim_leak_check manifest。

本脚本读取已构建的 dataset_manifest 和 split_manifest，
对所有 train/test 序列对计算归一化 DTW 距离（基于 gt.json 的 (px,py) 轨迹坐标），
若距离低于阈值（默认 0.1），则判定为高相似度泄漏并记录到 sim_leak_check manifest。

上游依赖:
- dataset_manifest.json（由 scripts/01_build_manifests.py 产出）
- split_manifest.json（由 scripts/04_build_splits.py 产出）
- 各序列目录下的 gt.json（轨迹坐标数据）

下游调用者:
- 实验管线（读取 sim_leak_check.json 判断 train-test 泄漏）
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts._bootstrap import bootstrap_project
except ModuleNotFoundError:
    from _bootstrap import bootstrap_project

_LAYOUT = bootstrap_project(__file__)
ROOT = _LAYOUT.root
SRC = _LAYOUT.src

from liquidloc.common.config_utils import load_yaml_config
from liquidloc.common.io_utils import dumps_json_text, read_json, write_json
from liquidloc.dataio.manifests.sim_leak_check import (
    compute_sim_leak_manifest,
    write_sim_leak_manifest,
)


def _resolve_cli_path(raw_path: str | None, *, arg_name: str) -> Path | None:
    """Normalize an optional CLI path argument."""
    if raw_path is None:
        return None
    normalized = raw_path.strip()
    if not normalized:
        raise ValueError(f"{arg_name} must be a non-empty path")
    return Path(normalized).resolve()


def main(argv: list[str] | None = None) -> int:
    """Run DTW/shape-feature leak check and write sim_leak_check manifest."""
    print("[26_sim_leak] 开始 | DTW/形状特征泄漏检查", flush=True)
    parser = argparse.ArgumentParser(description="DTW/shape-feature leak check")
    parser.add_argument("--manifests-root", default=str(ROOT / "outputs" / "data_prep" / "manifests" / "miluv"))
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)

    manifests_root = _resolve_cli_path(args.manifests_root, arg_name="--manifests-root")
    if manifests_root is None:
        raise ValueError("--manifests-root must be a non-empty path")

    print("[26_sim_leak] 加载清单", flush=True)
    dataset_manifest_path = manifests_root / "dataset_manifest.json"
    split_manifest_path = manifests_root / "split_manifest.json"

    if not dataset_manifest_path.is_file():
        raise FileNotFoundError(f"dataset_manifest.json not found at {dataset_manifest_path}")
    if not split_manifest_path.is_file():
        raise FileNotFoundError(f"split_manifest.json not found at {split_manifest_path}")

    dataset_manifest = read_json(dataset_manifest_path)
    split_manifest = read_json(split_manifest_path)

    if not isinstance(dataset_manifest, dict):
        raise TypeError(f"dataset_manifest must be a dict, got {type(dataset_manifest).__name__}")
    if not isinstance(split_manifest, dict):
        raise TypeError(f"split_manifest must be a dict, got {type(split_manifest).__name__}")

    print(f"[26_sim_leak] DTW 阈值: {args.threshold}", flush=True)

    sim_leak_manifest = compute_sim_leak_manifest(
        dataset_manifest,
        split_manifest,
        threshold=args.threshold,
    )

    output_root = _resolve_cli_path(args.output_root, arg_name="--output-root") or manifests_root
    output_root.mkdir(parents=True, exist_ok=True)

    output_path = write_sim_leak_manifest(sim_leak_manifest, output_root)

    print(
        dumps_json_text(
            {
                "sim_leak_check": str(output_path),
                "is_clean": sim_leak_manifest["is_clean"],
                "leak_count": len(sim_leak_manifest["leak_items"]),
                "threshold": sim_leak_manifest["threshold"],
            }
        )
    )

    _rc = 0 if sim_leak_manifest["is_clean"] else 2
    print(f"[26_sim_leak] 完成 | 返回码={_rc}", flush=True)
    return _rc


if __name__ == "__main__":
    raise SystemExit(main())
