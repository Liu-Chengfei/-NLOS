#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clear_workspace.py
Stage 0 清场执行脚本（手动版本，Zcode 环境等效执行）。
目标：每次实验开跑前清空仿真数据集与权重，防止旧产物污染。
严格按手册 §§0 清场 / [272]/[275]-[278] / RZ-0 / C-9 执行。
"""
from __future__ import annotations
import os
import shutil
import json
from datetime import datetime
import pathlib

ROOT = pathlib.Path(".").resolve()
ARCHIVE = ROOT / "archive"
DATA_DIR = ROOT / "data"
WEIGHT_DIR = ROOT / "checkpoints"
RESULT_DIR = ROOT / "outputs"

ALLOWED_KEEP = {".git", ".zcode", ".agents", ".audit", ".venv", "configs", "src"}

def _log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")

def _count_npz(p: pathlib.Path) -> int:
    return len(list(p.rglob("*.npz"))) if p.exists() else 0

def _count_ckpt(p: pathlib.Path) -> int:
    return len(list(p.rglob("*.pt"))) if p.exists() else 0

def _move_or_delete(src: pathlib.Path, dest_root: pathlib.Path, move=True) -> dict:
    record = {"src": str(src), "moved": [], "deleted": []}
    if not src.exists():
        return record
    # 物理隔离：archive/ 下新建时间戳子目录
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = dest_root / f"archive_{stamp}"
    target.mkdir(parents=True, exist_ok=True)
    for f in src.rglob("*"):
        if f.is_file():
            if move:
                rel = f.relative_to(ROOT)
                # 若是 archive 已存在目录，禁止迁移回 archive
                if str(rel).startswith("archive/"):
                    continue
                new_path = target / rel
                new_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(new_path))
                record["moved"].append(str(rel))
            else:
                f.unlink()
                record["deleted"].append(str(f.relative_to(ROOT)))
    return record

def clear_activity():
    _log("Stage 0 清场开始：检查活动目录旧产物")
    before_data_npz = _count_npz(DATA_DIR)
    before_wgt = _count_ckpt(WEIGHT_DIR)
    before_res = sum(1 for _ in RESULT_DIR.rglob("*.json") if RESULT_DIR.exists()) if RESULT_DIR.exists() else 0
    _log(f" 清场前统计：data/*.npz={before_data_npz}, checkpoints/*.pt={before_wgt}, outputs/*json≈{before_res}")

    # 1. 清空仿真数据集：npz + manifest
    data_rec = _move_or_delete(DATA_DIR, ARCHIVE, move=True)
    # 2. 清空权重
    wgt_rec = _move_or_delete(WEIGHT_DIR, ARCHIVE, move=True)
    # 3. 清空结果目录（诊断轨轻量清场/验收轨完整清场，默认完整清场）
    res_rec = _move_or_delete(RESULT_DIR, ARCHIVE, move=True)

    after_data_npz = _count_npz(DATA_DIR)
    after_wgt = _count_ckpt(WEIGHT_DIR)
    _log(f" 清场后统计：data/*.npz={after_data_npz}, checkpoints/*.pt={after_wgt}")

    # 写入清场验证证据
    evidence = {
        "time": datetime.now().isoformat(),
        "before": {"data_npz": before_data_npz, "checkpoints_pt": before_wgt, "outputs_json": before_res},
        "after": {"data_npz": after_data_npz, "checkpoints_pt": after_wgt},
        "moved": {
            "data": data_rec["moved"],
            "weights": wgt_rec["moved"],
            "results": res_rec["moved"]
        },
        "archive_root": str(ARCHIVE)
    }
    with open("清场验证.txt", "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)
        # 同时写入人类可读日志备份
        f.write(f"\n# 清场证据（{datetime.now().isoformat()})\n")
        f.write(f"活动数据目录旧npz数量: {before_data_npz} → {after_data_npz}\n")
        f.write(f"活动权重目录旧pt数量: {before_wgt} → {after_wgt}\n")
        f.write(f"迁移至 archive 下时间戳目录，路径：{ARCHIVE}\n")
    _log(" 清场验证写入 清场验证.txt")
    _log(" Stage 0 清场完成。未清场 = 禁止开跑实验。")
    return evidence

if __name__ == "__main__":
    clear_activity()
