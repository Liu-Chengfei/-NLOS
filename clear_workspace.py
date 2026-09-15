# -*- coding: utf-8 -*-
"""0 清场（手册条款 [1]「0 清场」，第 0 阶段第一动作，硬约束）

清空仿真数据集（npz/manifest）+ 权重（checkpoint）→ archive/（移动，不删除）。
第 2 阶段每个实验开跑前再跑一次本脚本并贴输出。
运行输出同时打印到 stdout 并落盘：清场验证.txt 与 logs/clear_workspace_第0阶段.log。
"""
import datetime
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "archive"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DEST_ROOT = ARCHIVE / ("clear_stage0_" + TIMESTAMP)

# 清场范围 = 手册「0 清场」所列三类产物：仿真数据集(npz)、manifest、权重(checkpoint)
SCAN_DIRS = ["data", "checkpoints", "outputs"]
PATTERNS = ("*.npz", "*manifest*", "*.pt", "*.ckpt", "*.pth")


def is_weight_dir(rel: Path) -> bool:
    """checkpoints/ 下全部文件按「权重(checkpoint)」处理，整目录清场。"""
    return rel.parts[0] == "checkpoints"


def collect_targets():
    targets = []
    for d in SCAN_DIRS:
        base = ROOT / d
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(ROOT)
            if is_weight_dir(rel) or any(f.match(p) for p in PATTERNS):
                targets.append(f)
    return targets


def main() -> int:
    lines = []
    lines.append(f"清场时间: {datetime.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"归档目标目录: {DEST_ROOT.relative_to(ROOT)}")
    targets = collect_targets()
    moved = 0
    for f in targets:
        rel = f.relative_to(ROOT)
        dst = DEST_ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), str(dst))
        lines.append(f"移动: {rel.as_posix()} -> {dst.relative_to(ROOT).as_posix()}")
        moved += 1
    lines.append(f"共移动 {moved} 项（0 项 = 工作区本无待清场的仿真数据集/权重）")
    # 清场后复核：确认扫描范围内已无匹配残留
    residual = [str(p.relative_to(ROOT)) for p in collect_targets()]
    lines.append(f"清场后残留复核: {len(residual)} 项" + ("" if not residual else " 残留=" + "; ".join(residual)))
    out = "\n".join(lines)
    print(out)
    (ROOT / "清场验证.txt").write_text(out + "\n", encoding="utf-8")
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    (log_dir / "clear_workspace_第0阶段.log").write_text(out + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
