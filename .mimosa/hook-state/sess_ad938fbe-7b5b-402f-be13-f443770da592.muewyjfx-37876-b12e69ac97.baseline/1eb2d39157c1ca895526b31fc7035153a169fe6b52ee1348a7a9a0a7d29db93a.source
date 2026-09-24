"""跨种子聚合并转换 metric_table.csv 为 paired 统计消费的 JSON 长表.

职责:
  - 扫描 `outputs/paper_run/{run_root}/runs/seed{N}/<method>/.../eval/metrics/metric_table.csv`
  - 跨种子拼接为长表, 加 `seed_id` 列
  - 写为 JSON 长表供 12_run_statistics.py / 14_run_paired_statistics.py 消费

概述:
  本脚本解决两个具体缺口:
    (1) `eval_pipeline.py:689` 落 `metric_table.csv` (CSV 格式), 但 `12_run_statistics.py:113`
        用 `_read_json(...)` 期望 JSON 输入 — 格式不匹配.
    (2) §17 6 比较脚手架要求跨种子聚合 (N_seed=30), 但当前 round7 仅 seed=0 单种子.
  本脚本接受单种子或多种子目录, 输出统一 JSON 长表. 单种子情形下 seed_id="seed0".

用法:
  # 多种子目录形态 (推荐):
  python scripts/13_build_multi_seed_metric_table.py \
      --seed-glob "outputs/paper_run/sim_v3_tc_20260723/runs/seed*/<method>/scoring_runs/e9_dual_degradation/eval/metrics/metric_table.csv" \
      --output-path outputs/paper_run/sim_v3_tc_20260723/runs/aggregate/metric_table.json

  # 单种子情形 (退化为 seed0):
  python scripts/13_build_multi_seed_metric_table.py \
      --seed-glob "outputs/paper_run/sim_v3_tc_20260723/final_paper_test_scoring/<method>/scoring_runs/e9_dual_degradation/eval/metrics/metric_table.csv" \
      --output-path outputs/paper_run/sim_v3_tc_20260723/runs/aggregate/metric_table.json \
      --single-seed-alias seed0

输出 JSON 结构:
  {
    "metric_table": [
      {
        "case_ref": "...", "seq_id": "...", "scene_id": "...",
        "method_name": "...", "seed_id": "seed0",
        "task_id": "...", "repeat_id": "...",
        "metric": "rmse", "value": 11.42, "unit": "m",
        "direction": "lower_is_better", "group": "primary",
        "prediction_length": ..., "ground_truth_length": ...,
        "aligned_length": ..., "valid_pair_count": ..., "overlap_ratio": ...,
        "reliability_status": "ok"
      },
      ...
    ],
    "metadata": {
      "n_seeds": 1, "n_methods": 5, "n_seq": 12,
      "seed_ids": ["seed0"], "method_names": [...], "seq_ids": [...],
      "source_glob": "...", "row_count": 8641, "generated_at": "<ISO datetime>"
    }
  }
"""

from __future__ import annotations  # 允许使用前向类型注解。

import argparse  # 解析命令行参数。
import csv  # 读取 CSV 长表。
import json  # 序列化为 JSON。
import sys  # 操作 sys.path 和进程退出码。
from datetime import datetime  # 生成 ISO 时间戳。
from pathlib import Path  # 处理文件路径。
from typing import Any  # 标注任意 JSON 风格结构。

ROOT = Path(__file__).resolve().parents[1]  # 仓库根目录。
SRC = ROOT / "src"  # 源码目录。
if str(SRC) not in sys.path:  # 确保脚本能直接导入项目代码。
    sys.path.insert(0, str(SRC))  # 把源码目录放到导入路径最前面。


# 必须出现在 metric_table.csv 的列名. 来自 eval_pipeline.py:691-700.
_REQUIRED_CSV_COLUMNS: tuple[str, ...] = (
    "case_ref",
    "seq_id",
    "scene_id",
    "method_name",
    "metric",
    "value",
    "unit",
    "direction",
    "group",
)

# 整数列名, 解析时转 int. 空字符串视为 None.
# 注: task_id 在 eval_pipeline.py 是字符串如 "scene_0000"; repeat_id 是 "repeat_0000". 都不在此列.
_INT_COLUMNS: tuple[str, ...] = (
    "prediction_length",
    "ground_truth_length",
    "aligned_length",
    "valid_pair_count",
)

# 浮点列名, 解析时转 float. 空字符串视为 None.
_FLOAT_COLUMNS: tuple[str, ...] = (
    "value",
    "overlap_ratio",
)


def _resolve_path(raw_value: str, *, flag_name: str) -> Path:
    """把命令行传入的路径整理成可直接使用的绝对路径."""
    value = str(raw_value).strip()
    if not value:  # 空字符串非法。
        raise ValueError(f"{flag_name} must be a non-empty path")  # 明确报错。
    path = Path(value)
    if not path.is_absolute():  # 相对路径统一按仓库根目录解释。
        path = (ROOT / path).resolve()
    return path.resolve()


def _glob_metric_csvs(seed_glob: str) -> list[Path]:
    """展开 glob 模式, 返回所有匹配到的 metric_table.csv 路径.

    支持含中间通配符的复合 glob (例如 ``a/*/b/c/d/metric_table.csv``).
    实现策略: 把模式拆成"无通配符的固定前缀 + 含通配符的剩余部分", 用固定前缀
   定位起点目录, 再用 `Path.glob` 递归展开剩余部分. 这样能正确处理跨多段路径
    的 ``*`` 与 ``**`` (pathlib 的 `**` 等价于递归 glob)。
    """
    if not seed_glob:  # 防御空模式。
        raise ValueError("--seed-glob must be a non-empty glob pattern")
    glob_path = Path(seed_glob)
    if not glob_path.is_absolute():
        glob_path = (ROOT / seed_glob).resolve()

    parts = glob_path.parts  # 拆段, 找第一段含 * 的位置。
    fixed_prefix_parts: list[str] = []
    glob_tail_parts: list[str] = []
    found_wildcard = False
    for part in parts:
        if not found_wildcard and "*" not in part and "?" not in part and "[" not in part:
            fixed_prefix_parts.append(part)
        else:
            found_wildcard = True
            glob_tail_parts.append(part)

    if not glob_tail_parts:  # 没有通配符, 直接当文件路径用。
        if glob_path.is_file():
            return [glob_path]
        raise FileNotFoundError(
            f"no files matched --seed-glob pattern (no wildcard, not a file): {seed_glob} "
            f"(resolved: {glob_path})"
        )

    base_dir = Path(*fixed_prefix_parts) if fixed_prefix_parts else Path(".")
    tail_pattern = "/".join(glob_tail_parts)  # pathlib glob 用正斜杠分隔。
    matches = sorted(base_dir.glob(tail_pattern))
    if not matches:  # 没匹配到就报错。
        raise FileNotFoundError(
            f"no files matched --seed-glob pattern: {seed_glob} "
            f"(base={base_dir}, tail={tail_pattern})"
        )
    return matches


def _parse_csv_row(raw_row: dict[str, str], *, seed_id: str, source_path: Path) -> dict[str, Any]:
    """把 CSV 行字典转成 JSON 长表行, 强类型化整数与浮点列, 加 seed_id."""
    row: dict[str, Any] = {}
    for key, raw_value in raw_row.items():
        if key in _INT_COLUMNS:
            stripped = (raw_value or "").strip()
            row[key] = int(stripped) if stripped else None
        elif key in _FLOAT_COLUMNS:
            stripped = (raw_value or "").strip()
            row[key] = float(stripped) if stripped else None
        else:
            row[key] = raw_value
    row["seed_id"] = seed_id
    row["_source_path"] = str(source_path)
    return row


def _read_metric_csv(csv_path: Path, *, seed_id: str) -> list[dict[str, Any]]:
    """读一个 metric_table.csv, 返回 JSON 长表行列表."""
    if not csv_path.is_file():  # 文件不存在就报错。
        raise FileNotFoundError(f"metric_table.csv not found: {csv_path}")
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:  # 支持带 BOM 的 UTF-8。
        reader = csv.DictReader(handle)  # CSV 字典读取器。
        fieldnames = reader.fieldnames or []
        missing = [col for col in _REQUIRED_CSV_COLUMNS if col not in fieldnames]
        if missing:  # 必填列缺失就报错。
            raise ValueError(
                f"metric_table.csv missing required columns {missing}: {csv_path}"
            )
        for raw_row in reader:
            rows.append(_parse_csv_row(raw_row, seed_id=seed_id, source_path=csv_path))
    return rows


def _infer_seed_id_from_path(csv_path: Path, *, single_seed_alias: str | None) -> str:
    """从路径中的 seedN/seed_N/seedN/ 目录推断 seed_id."""
    parts = csv_path.parts
    for part in parts:  # 找第一个 seed 开头或 seed=N 形态的目录名。
        if part.startswith("seed"):
            return part
    if single_seed_alias:  # 单种子情形, 退化用 alias。
        return single_seed_alias
    raise ValueError(
        f"cannot infer seed_id from path {csv_path}; "
        "pass --single-seed-alias for single-seed runs"
    )


def _collect_unique_values(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """收集 method_name/seq_id/scene_id/seed_id 的去重值, 供 metadata."""
    by_key: dict[str, set[str]] = {
        "method_names": set(),
        "seq_ids": set(),
        "scene_ids": set(),
        "seed_ids": set(),
    }
    for row in rows:  # 单次扫描去重。
        by_key["method_names"].add(row.get("method_name") or "")
        by_key["seq_ids"].add(row.get("seq_id") or "")
        by_key["scene_ids"].add(row.get("scene_id") or "")
        by_key["seed_ids"].add(row.get("seed_id") or "")
    return {key: sorted(v - {""}) for key, v in by_key.items()}


def _write_json_long_table(
    output_path: Path,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    """把 JSON 长表写到指定路径."""
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 先建父目录。
    payload = {"metric_table": rows, "metadata": metadata}
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    """脚本主入口: 读 CSV, 聚合, 写 JSON 长表."""
    print("[13_metric_table] 开始 | 跨种子聚合 metric_table CSV -> JSON 长表", flush=True)
    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed metric_table.csv files into a JSON long table."
    )
    parser.add_argument(
        "--seed-glob", required=True, help="Glob pattern matching per-seed metric_table.csv paths"
    )
    parser.add_argument(
        "--output-path", required=True, help="Output JSON long-table path"
    )
    parser.add_argument(
        "--single-seed-alias", default=None,
        help="Seed id to use when paths don't contain a seedN/ directory (e.g. 'seed0')",
    )
    args = parser.parse_args(argv)

    print(f"[13_metric_table] 解析路径与展开 glob", flush=True)
    csv_paths = _glob_metric_csvs(args.seed_glob)  # 展开所有匹配。
    print(f"[13_metric_table] 匹配 {len(csv_paths)} 个 metric_table.csv", flush=True)

    rows: list[dict[str, Any]] = []
    for csv_path in csv_paths:
        seed_id = _infer_seed_id_from_path(csv_path, single_seed_alias=args.single_seed_alias)
        print(f"  - {csv_path.name}  seed_id={seed_id}  ({csv_path.parent.parent.parent.parent.parent.name}/...)", flush=True)
        rows.extend(_read_metric_csv(csv_path, seed_id=seed_id))

    print(f"[13_metric_table] 总行数: {len(rows)}", flush=True)

    unique_values = _collect_unique_values(rows)
    metadata = {
        "n_seeds": len(unique_values["seed_ids"]),
        "n_methods": len(unique_values["method_names"]),
        "n_seq": len(unique_values["seq_ids"]),
        "seed_ids": unique_values["seed_ids"],
        "method_names": unique_values["method_names"],
        "seq_ids": unique_values["seq_ids"],
        "scene_ids": unique_values["scene_ids"],
        "source_glob": args.seed_glob,
        "row_count": len(rows),
        "generated_at": datetime.now().astimezone().isoformat(),
        "script": "scripts/13_build_multi_seed_metric_table.py",
        "protocol_version": 2,
    }

    output_path = _resolve_path(args.output_path, flag_name="--output-path")
    _write_json_long_table(output_path, rows, metadata)

    print(f"[13_metric_table] 完成 | 输出: {output_path}", flush=True)
    print(f"  - n_seeds={metadata['n_seeds']}  n_methods={metadata['n_methods']}  n_seq={metadata['n_seq']}", flush=True)
    print(f"  - row_count={metadata['row_count']}", flush=True)
    return 0


if __name__ == "__main__":  # 直接执行脚本时走这里。
    raise SystemExit(main(sys.argv[1:]))
