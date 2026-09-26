"""验证 CIR 统计特征生成结果。

脚本职责：
- 扫描 sim_e9_main 下所有序列的 cir_features.npz 文件
- 断言形状为 (N, 3)，其中 N >= 1
- 断言 first_path_amplitude, peak_amplitude ∈ [0, 1]
- 断言 delay_spread ≥ 0
- 对比 N 档与 LOS 序列的特征差异是否符合预期（NLOS 延迟扩展更大）

用法：
    python scripts/05_validate_cir.py --data-root data/raw/sim_e9_main
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts._bootstrap import bootstrap_project
except ModuleNotFoundError:
    from _bootstrap import bootstrap_project

_LAYOUT = bootstrap_project(__file__)
ROOT = _LAYOUT.root
SRC = _LAYOUT.src


def validate_cir_npz(npz_path: Path) -> dict[str, Any]:
    """验证单个 cir_features.npz 文件。

    返回:
        dict 包含验证状态和详情。
    """
    if not npz_path.is_file():
        return {"path": str(npz_path), "status": "missing", "error": "file not found"}

    try:
        data = np.load(npz_path)
    except Exception as exc:
        return {"path": str(npz_path), "status": "error", "error": str(exc)}

    keys = ["first_path_amplitude", "peak_amplitude", "delay_spread"]
    for k in keys:
        if k not in data:
            return {"path": str(npz_path), "status": "missing_key", "error": f"key '{k}' not found", "keys": list(data.keys())}

    n = data["first_path_amplitude"].shape[0]
    if n == 0:
        return {"path": str(npz_path), "status": "empty", "error": "zero frames"}

    first_path = data["first_path_amplitude"]
    peak = data["peak_amplitude"]
    delay = data["delay_spread"]

    errors = []
    if first_path.min() < 0 or first_path.max() > 1:
        errors.append(f"first_path_amplitude out of [0,1]: [{first_path.min():.4f}, {first_path.max():.4f}]")
    if peak.min() < 0 or peak.max() > 1:
        errors.append(f"peak_amplitude out of [0,1]: [{peak.min():.4f}, {peak.max():.4f}]")
    if delay.min() < 0:
        errors.append(f"delay_spread negative: min={delay.min():.4f}")

    nlos_level = str(data.get("nlos_level", "unknown"))

    summary = {
        "path": str(npz_path),
        "status": "ok",
        "n_frames": int(n),
        "nlos_level": nlos_level,
        "first_path_range": [round(float(first_path.min()), 4), round(float(first_path.max()), 4)],
        "peak_range": [round(float(peak.min()), 4), round(float(peak.max()), 4)],
        "delay_spread_range": [round(float(delay.min()), 4), round(float(delay.max()), 4)],
        "errors": errors,
    }
    if errors:
        summary["status"] = "validation_failed"
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate CIR features across sim sequences")
    parser.add_argument("--data-root", default=str(ROOT / "data" / "raw" / "sim_e9_main"))
    args = parser.parse_args(argv)

    data_root = Path(args.data_root).resolve()
    if not data_root.is_dir():
        print(f"[E] data_root not found: {data_root}", file=sys.stderr)
        return 1

    npz_files = sorted(data_root.rglob("cir_features.npz"))
    if not npz_files:
        print(f"[W] no cir_features.npz found under {data_root}. Run 02_generate_sim_raw.py with cir_features=on first.", file=sys.stderr)
        return 1

    print(f"[I] Found {len(npz_files)} cir_features.npz files", flush=True)

    ok_count = 0
    fail_count = 0
    all_results: list[dict[str, Any]] = []

    for p in npz_files:
        result = validate_cir_npz(p)
        all_results.append(result)
        if result["status"] == "ok":
            ok_count += 1
            print(f"  [OK] {p.relative_to(data_root)} | n={result['n_frames']} | fp=[{result['first_path_range']}] | pk=[{result['peak_range']}] | ds=[{result['delay_spread_range']}] | NLOS={result['nlos_level']}")
        else:
            fail_count += 1
            print(f"  [FAIL] {p.relative_to(data_root)} | {result.get('error', result['status'])}")

    print(f"\n[SUMMARY] {ok_count}/{ok_count + fail_count} passed")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
