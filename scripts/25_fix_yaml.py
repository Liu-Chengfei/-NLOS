"""修复 configs/models/*.yaml 的 checkpoint_path 字段。

完全用 yaml.safe_load 解析 + yaml.safe_dump 写回，保留所有结构。
防 path escape：所有 Path 构造走 forward slash 拼接。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(os.getcwd()).as_posix()  # forward slash, 避免 \Q \4 等被 escape
print(f"[25] PROJECT_ROOT (posix): {PROJECT_ROOT}")


MODEL_YAMLS = {
    "lstm_ekf": "configs/models/lstm_ekf.yaml",
    "liquid_ekf": "configs/models/liquid_ekf.yaml",
    "transformer_ekf": "configs/models/transformer_ekf.yaml",
}
CHECKPOINT_FILE = {
    "lstm_ekf": "lstm_ekf_best_checkpoint.pt",
    "liquid_ekf": "liquid_ekf_resume_checkpoint.pt",
    "transformer_ekf": "transformer_ekf_best_checkpoint.pt",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", default="outputs/train_paper_main")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--restore", action="store_true")
    return parser.parse_args()


def _set_ckpt(cfg: dict, path: str | None) -> None:
    cfg["checkpoint_path"] = path or ""


def main():
    args = parse_args()
    # 拼 train_root（用 forward slash 防 escape）
    train_root = Path(PROJECT_ROOT.rstrip("/") + "/" + args.train_root.lstrip("/"))
    for model_name, yaml_rel in MODEL_YAMLS.items():
        yaml_path = Path(PROJECT_ROOT.rstrip("/") + "/" + yaml_rel)
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if args.restore:
            _set_ckpt(cfg, "")
            print(f"restore {yaml_rel}")
        else:
            ckpt = train_root / f"seed{args.seed}" / model_name / "checkpoints" / CHECKPOINT_FILE[model_name]
            if not ckpt.exists():
                print(f"SKIP {yaml_rel}: {ckpt} not found")
                continue
            _set_ckpt(cfg, str(ckpt.resolve()))
            print(f"set {yaml_rel}: checkpoint_path = {ckpt.resolve()}")
        yaml_path.write_text(
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
