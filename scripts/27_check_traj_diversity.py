"""运行轨迹多样性校验，并写入 traj_diversity manifest。

本脚本读取已构建的 dataset_manifest 和 split_manifest，
对 train/val/test 切分分别计算方位角直方图（八方向，≤20% 差断言）
与速度分布（0.5–1.5 m/s 全区间覆盖、训练/测试占比一致性断言），
结果写入 traj_diversity.json。

上游依赖:
- dataset_manifest.json（由 scripts/01_build_manifests.py 产出）
- split_manifest.json（由 scripts/04_build_splits.py 产出）
- 各序列目录下的 gt.json（轨迹坐标数据）

下游调用者:
- 实验管线（读取 traj_diversity.json 判断测试集代表性）
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

from liquidloc.common.io_utils import dumps_json_text, read_json
from liquidloc.analysis.traj_diversity import (
    compute_traj_diversity_manifest,
    write_traj_diversity_manifest,
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
    """Run trajectory diversity check and write traj_diversity manifest."""
    print("[27_traj_diversity] 开始 | 轨迹多样性校验", flush=True)
    parser = argparse.ArgumentParser(description="Trajectory diversity check")
    parser.add_argument("--manifests-root", default=str(ROOT / "outputs" / "data_prep" / "manifests" / "miluv"))
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)

    manifests_root = _resolve_cli_path(args.manifests_root, arg_name="--manifests-root")
    if manifests_root is None:
        raise ValueError("--manifests-root must be a non-empty path")

    print("[27_traj_diversity] 加载清单", flush=True)
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

    traj_diversity_manifest = compute_traj_diversity_manifest(
        dataset_manifest,
        split_manifest,
    )

    output_root = _resolve_cli_path(args.output_root, arg_name="--output-root") or manifests_root
    output_root.mkdir(parents=True, exist_ok=True)

    output_path = write_traj_diversity_manifest(traj_diversity_manifest, output_root)

    print(
        dumps_json_text(
            {
                "traj_diversity": str(output_path),
                "overall_passed": traj_diversity_manifest["overall_passed"],
                "azimuth_passed": traj_diversity_manifest["azimuth_assertion"].get("passed", False),
                "speed_full_coverage": traj_diversity_manifest["speed_full_coverage"],
                "speed_proportion_passed": traj_diversity_manifest["speed_proportion_assertion"].get("passed", False),
            }
        )
    )

    _rc = 0 if traj_diversity_manifest["overall_passed"] else 2
    print(f"[27_traj_diversity] 完成 | 返回码={_rc}", flush=True)
    return _rc


if __name__ == "__main__":
    raise SystemExit(main())
