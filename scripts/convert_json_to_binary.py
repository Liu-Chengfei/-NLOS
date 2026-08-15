#!/usr/bin/env python3
"""一键将 prepare 目录下的所有 _events.json 转换为二进制压缩格式（Pickle + Gzip）。

此脚本遍历指定的 prepare_root 目录，寻找所有 `*_events.json`。
对每个文件：
1. 读取 JSON 格式的事件列表。
2. 通过 Pickle + Gzip 写入同名的 `*_events.pkl.gz` 二进制文件。
3. 读取并反序列化 `*_events.pkl.gz`，与原始的事件列表做逐项值校验。
4. 校验完全一致后，就地删除原有的 `*_events.json` 释放空间。
5. 脚本支持断点续传（如果对应的 `.pkl.gz` 已经存在，默认跳过，除非指定 --force）。
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_equality(a: Any, b: Any) -> bool:
    """做精细的值比对（容忍浮点数微小精度差异和结构）。"""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(check_equality(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(check_equality(xi, yi) for xi, yi in zip(a, b))
    if isinstance(a, float):
        # 容忍浮点数转换精度微小差异
        return abs(a - b) < 1e-9
    return a == b


def convert_file(json_path: Path, force: bool = False) -> bool:
    """转换单个 JSON 事件文件为 .pkl.gz。

    转换成功、校验无误后返回 True 并删除原 json。
    如果已经转换且没有 --force，返回 False（表示跳过）。
    """
    seq_id_events = json_path.name
    seq_id = seq_id_events.replace("_events.json", "")
    pkl_gz_path = json_path.parent / f"{seq_id}_events.pkl.gz"

    if pkl_gz_path.is_file() and not force:
        # 断点续传：检测到 pkl.gz 已存在，校验并删除对应的 json
        print(f"[Skip] {pkl_gz_path.name} already exists. Cleaning up JSON...", flush=True)
        if json_path.is_file():
            json_path.unlink()
        return False

    print(f"[Converting] {json_path.name} -> {pkl_gz_path.name}...", end="", flush=True)
    
    # 1) 读取 JSON
    with json_path.open("r", encoding="utf-8") as f:
        events = json.load(f)

    # 2) 写入二进制压缩
    with gzip.open(pkl_gz_path, "wb", compresslevel=3) as f:
        pickle.dump(events, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 3) 校验回读
    with gzip.open(pkl_gz_path, "rb") as f:
        loaded_events = pickle.load(f)

    if not check_equality(events, loaded_events):
        # 校验失败！
        pkl_gz_path.unlink(missing_ok=True)
        print(" [FAIL: Verification error!]", flush=True)
        raise ValueError(f"Verification failed for {json_path}")

    # 4) 校验通过，就地删除原始 JSON
    json_path.unlink()
    
    orig_size = json_path.stat().st_size if json_path.is_file() else 0 # 已经被删除则大小为 0
    new_size = pkl_gz_path.stat().st_size
    ratio = (new_size / orig_size * 100) if orig_size > 0 else 0
    print(f" [OK] (Size: {new_size/1024/1024:.2f}MB)", flush=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert prepare JSON events to binary Gzip+Pickle.")
    parser.add_argument(
        "--prepare-root",
        required=True,
        help="Path to prepare outputs root containing *_events.json",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing .pkl.gz files",
    )
    args = parser.parse_args()

    prepare_root = Path(args.prepare_root).resolve()
    if not prepare_root.is_dir():
        print(f"[Error] Directory not found: {prepare_root}", file=sys.stderr)
        return 1

    # 寻找所有的 *_events.json
    json_files = sorted(list(prepare_root.glob("*_events.json")))
    # 同时也找已经生成的 .pkl.gz，以便在断点续传时清理孤立的 json
    pkl_gz_files = sorted(list(prepare_root.glob("*_events.pkl.gz")))

    total_json = len(json_files)
    print(f"Found {total_json} JSON files to process in {prepare_root}.", flush=True)
    if total_json == 0 and len(pkl_gz_files) > 0:
        print("All JSON files seem to have been already converted to pkl.gz.", flush=True)
        return 0

    converted_count = 0
    skipped_count = 0

    for idx, json_path in enumerate(json_files, 1):
        try:
            print(f"[{idx}/{total_json}] ", end="")
            converted = convert_file(json_path, force=args.force)
            if converted:
                converted_count += 1
            else:
                skipped_count += 1
        except Exception as exc:
            print(f"\n[Error] Failed to convert {json_path}: {exc}", file=sys.stderr)
            return 2

    print(f"\nConversion complete! Converted: {converted_count}, Skipped/Cleaned: {skipped_count}", flush=True)
    
    # 释放空间后的总体积检查
    total_size = sum(p.stat().st_size for p in prepare_root.glob("*_events.pkl.gz"))
    print(f"Total binary events size: {total_size/1024/1024/1024:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
