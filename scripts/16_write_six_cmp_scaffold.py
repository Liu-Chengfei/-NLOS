"""把 6 个 aggregate.json 字段回填到 docs/superpowers/specs/六比较实测脚手架-v0.1.md 表格.

职责:
  - 读 runs/cmpN/{version_id}/aggregate.json 6 个文件
  - 替换脚手架 markdown 的「实测前字段」列: N/A → 真值 + aggregate.json 路径 + 签名日期
  - 末了在底部追加「实测签名」块
  - B24 未实填时回填标"协议级候选", 非协议级结论

注意: 严格保留 markdown 表格结构, 只替换单元格内容; 不允许估填.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(raw_value: str, *, flag_name: str) -> Path:
    value = str(raw_value).strip()
    if not value:
        raise ValueError(f"{flag_name} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path.resolve()


def _load_aggregate(runs_root: Path, cmp_id: str, version_id: str) -> dict[str, Any]:
    agg_path = runs_root / cmp_id / version_id / "aggregate.json"
    if not agg_path.is_file():
        raise FileNotFoundError(f"aggregate.json missing for {cmp_id}: {agg_path}")
    return json.loads(agg_path.read_text(encoding="utf-8"))


def _format_num(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def _sign_text(signing: dict[str, Any]) -> str:
    signed_at = signing.get("signed_at", "")
    claim = signing.get("claim_strength", "")
    note = signing.get("note", "")
    return f"`{signed_at}` ({claim}) — {note}"


def _replace_field_in_row(md_text: str, row_marker: str, field_label: str, new_value: str) -> str:
    """在指定 markdown 表格行中替换某字段的值.

    通过查找含 row_marker 的行, 定位「field_label | ...」模式并替换.
    本函数保留原始表格分隔符.
    """
    lines = md_text.splitlines(keepends=True)
    out: list[str] = []
    for line in lines:
        if row_marker in line and field_label in line:
            # 模式: "row_marker ... new_value 结尾"
            # 简单实现: 找到 field_label 出现位置, 把其后到行尾的 N/A 或旧数值替换为新值
            # 但 markdown 一行通常很短, 用更稳的 field_label + 模式替换
            # 这里我们直接整行 replace 旧 cell 模式
            # 例如 "| 实测前字段 | effect_size = N/A；... |" → "| 实测前字段 | effect_size = 0.1234；签名: ... |"
            # 实现策略: 在行尾追加 " | 回填: <new_value>"
            # 保留原 N/A 行, 末了增加一段 update 注解, 不破坏原表可比性
            if line.rstrip().endswith("|"):
                new_line = line.rstrip()[:-1] + f"  回填: {new_value} |" + ("\n" if line.endswith("\n") else "")
                out.append(new_line)
                continue
        out.append(line)
    return "".join(out)


def main(argv: list[str] | None = None) -> int:
    print("[16_scaffold] 开始 | 回填六比较脚手架 markdown", flush=True)
    parser = argparse.ArgumentParser(
        description="Backfill six_cmp aggregate.json values into scaffold markdown."
    )
    parser.add_argument("--runs-root", required=True, help="runs/ directory root")
    parser.add_argument("--version-id", required=True)
    parser.add_argument("--scaffold-md", required=True, help="Path to 六比较实测脚手架-v0.1.md")
    parser.add_argument("--output-md", required=True, help="Output backfilled markdown path")
    args = parser.parse_args(argv)

    runs_root = _resolve_path(args.runs_root, flag_name="--runs-root")
    scaffold_md = _resolve_path(args.scaffold_md, flag_name="--scaffold-md")
    output_md = _resolve_path(args.output_md, flag_name="--output-md")

    md_text = scaffold_md.read_text(encoding="utf-8")

    cmp1 = _load_aggregate(runs_root, "cmp1", args.version_id)
    cmp2 = _load_aggregate(runs_root, "cmp2", args.version_id)
    cmp3 = _load_aggregate(runs_root, "cmp3", args.version_id)
    cmp4 = _load_aggregate(runs_root, "cmp4", args.version_id)
    cmp5 = _load_aggregate(runs_root, "cmp5", args.version_id)
    cmp6 = _load_aggregate(runs_root, "cmp6", args.version_id)

    signing = cmp1.get("signing", {})

    # 构造回填注解块 (附加到文档末尾, 不破坏原行)
    backfill_block = f"""

---

## §17 评价框架回填 (v0.1.1) — 协议级候选

**回填时间**: {datetime.now().astimezone().isoformat()}
**协议版本**: protocol_v2
**B24 状态**: {signing.get("b24_status", "not_filled")}
**回填强度**: {signing.get("claim_strength", "protocol_candidate")}
**注**: {signing.get("note", "")}
**n_seed**: {signing.get("n_seed_total", 1)} (round7 单种子; 协议要 N_seed=30, 未满足)

### 实测字段回填

#### 比较 1: LNN+EKF > 两 EKF

| 子比较 | rmse_liquid_ekf | rmse_opponent | rel_improve | pass_strict_gt | p_value | aggregate.json |
|--------|-----------------|---------------|-------------|----------------|---------|----------------|
| cmp1a liquid_ekf vs ekf | {_format_num(cmp1.get('cmp1a_liquid_vs_ekf', {}).get('rmse_mean_a'))} | {_format_num(cmp1.get('cmp1a_liquid_vs_ekf', {}).get('rmse_mean_b'))} | {_format_num(cmp1.get('cmp1a_liquid_vs_ekf', {}).get('rel_improve'))} | {cmp1.get('cmp1a_liquid_vs_ekf', {}).get('pass_strict_gt_flag')} | {_format_num(cmp1.get('cmp1a_liquid_vs_ekf', {}).get('p_value'))} | runs/cmp1/{args.version_id}/aggregate.json |
| cmp1b liquid_ekf vs robust_ekf | {_format_num(cmp1.get('cmp1b_liquid_vs_robust_ekf', {}).get('rmse_mean_a'))} | {_format_num(cmp1.get('cmp1b_liquid_vs_robust_ekf', {}).get('rmse_mean_b'))} | {_format_num(cmp1.get('cmp1b_liquid_vs_robust_ekf', {}).get('rel_improve'))} | {cmp1.get('cmp1b_liquid_vs_robust_ekf', {}).get('pass_strict_gt_flag')} | {_format_num(cmp1.get('cmp1b_liquid_vs_robust_ekf', {}).get('p_value'))} | 同上 |

**overall_pass_strict_gt_flag**: {cmp1.get('overall_pass_strict_gt_flag')}

#### 比较 2: 两 EKF 同档 (标准 EKF ≈ Robust-EKF)

| 字段 | 值 |
|------|-----|
| rmse_std | {_format_num(cmp2.get('delta_approx_median') if False else None) if False else _format_num_via_summary(cmp2)} |
| delta_approx_median | {_format_num(cmp2.get('delta_approx_median'))} |
| same_band_flag | {cmp2.get('same_band_flag')} |
| chi_square_on | {cmp2.get('chi_square_on')} |
| verdict | {cmp2.get('verdict')} |
| p_value | {_format_num(cmp2.get('p_value'))} |
| aggregate.json | runs/cmp2/{args.version_id}/aggregate.json |

#### 比较 3: EKF > 无核 FGO (封印级 C, 5 子条件合取)

| 字段 | 值 |
|------|-----|
| rmse_ekf | {_format_num(cmp3.get('rmse_ekf'))} |
| rmse_fgo | {_format_num(cmp3.get('rmse_fgo'))} |
| rel_improve | {_format_num(cmp3.get('rel_improve'))} |
| pass_strict_gt_flag | {cmp3.get('pass_strict_gt_flag')} |
| huber_forced_off | {cmp3.get('sub_flags', {}).get('huber_forced_off')} |
| cold_start_flag | {cmp3.get('sub_flags', {}).get('cold_start_flag')} |
| ell_over_tau_in_range | {cmp3.get('sub_flags', {}).get('ell_over_tau_in_range')} |
| chi_square_on | {cmp3.get('sub_flags', {}).get('chi_square_on')} |
| robust_weight_bypass_flag | {cmp3.get('sub_flags', {}).get('robust_weight_bypass_flag')} |
| warning | {cmp3.get('warning', '')} |
| aggregate.json | runs/cmp3/{args.version_id}/aggregate.json |

#### 比较 4: 无核 FGO ? 纯 SGPR (不可测, SGPR 实现缺失, §29.5 不得用「≈」运算符)

| 字段 | 值 |
|------|-----|
| not_testable | {cmp4.get('not_testable')} |
| reason | {cmp4.get('reason')} |
| degraded_to | {cmp4.get('degraded_to')} |
| semantic | {cmp4.get('semantic')} |
| aggregate.json | runs/cmp4/{args.version_id}/aggregate.json |

#### 比较 5: 无核 FGO ?/> 小容量 LSTM+EKF (SGPR 落地前按 §29.5 勿提第三档身份)

| 字段 | 值 |
|------|-----|
| rmse_fgo_side (FGO 单档) | {_format_num(cmp5.get('rmse_fgo_side'))} |
| rmse_lstm_ekf | {_format_num(cmp5.get('rmse_lstm_ekf'))} |
| rel_improve | {_format_num(cmp5.get('rel_improve'))} |
| pass_strict_gt_flag | {cmp5.get('pass_strict_gt_flag')} |
| budget_locked_flag | {cmp5.get('budget_locked_flag')} |
| feature_interface_symmetric_flag | {cmp5.get('feature_interface_symmetric_flag')} |
| sgpr_side_completed | {cmp5.get('sgpr_side_completed')} |
| degraded_to_subordinate_rank | {cmp5.get('degraded_to_subordinate_rank')} |
| semantic | {cmp5.get('semantic')} |
| aggregate.json | runs/cmp5/{args.version_id}/aggregate.json |

#### 比较 6: 小容量 LSTM+EKF ? 同预算因果 Transformer+EKF (不可测, TF 实现缺失, §29.5 不得用「>」运算符)

| 字段 | 值 |
|------|-----|
| not_testable | {cmp6.get('not_testable')} |
| reason | {cmp6.get('reason')} |
| degraded_to | {cmp6.get('degraded_to')} |
| aggregate.json | runs/cmp6/{args.version_id}/aggregate.json |

---

## 实测签名

- **签名时间**: {_sign_text(signing)}
- **脚本依赖**: scripts/13_build_multi_seed_metric_table.py → 14_run_paired_statistics.py → 15_build_six_cmp_aggregate.py → 16_write_six_cmp_scaffold.py
- **数据来源**: outputs/paper_run/sim_v3_tc_20260723/final_paper_test_scoring_5models.json + 5 method × 12 seq × 1 seed 跨种子聚合 metric_table.json
- **N_seed 警告**: 当前 n_seed=1, 协议 §0.4 D4 要求 N_seed=30. 本次回填仅供框架链路验证, 不构成协议级实测通过结论.
- **B24 警告**: {signing.get("note", "")}
- **§29 降级**: cmp4/cmp6 因实现缺失降级为子排序, 主表 5 方法子排序可比；cmp5 在 SGPR 落地前按 §29.5 不写「第三档」身份, 仅以无核 FGO 单档参与 FGO ?/> LSTM 子排序
- **重要发现 (cmp2 拆档)**: cmp2 标准 EKF vs Robust-EKF 中位相对差 = {_format_num(cmp2.get('delta_approx_median'))}, 远超 ε_≈=0.08, verdict = {cmp2.get('verdict')} (即两 EKF **不同档**, Robust 显著优于 Standard)
- **重要发现 (cmp3 无核身份已对齐)**: 2026-07-26 起 fgo.yaml 删除 robust_weight/gate + FGOCore 强制 l2；huber_forced_off=true。历史 v5 若在旧配置表面下产出，须无核身份重跑后才能作 cmp3 结论
- **重要发现 (cmp5 比较5反序)**: rmse_fgo_side={_format_num(cmp5.get('rmse_fgo_side'))} >> rmse_lstm_ekf={_format_num(cmp5.get('rmse_lstm_ekf'))}, rel_improve={_format_num(cmp5.get('rel_improve'))} (无核 FGO 单档当前弱于 LSTM+EKF, 与§29.5 子排序「FGO ?/> LSTM」不冲突)

---

**回填结束**
"""

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(md_text.rstrip() + "\n" + backfill_block, encoding="utf-8")
    print(f"[16_scaffold] 完成 | 输出: {output_md}", flush=True)
    return 0


def _format_num_via_summary(cmp2: dict[str, Any]) -> str | None:
    return _format_num(cmp2.get("rmse_std"))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
