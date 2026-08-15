"""只重建 prepare_manifest.json，从已有 {seq_id}_events.json 抽取 scene_parameters 和 read_report。

当原始 events 已经产出（来自一次 prepare 运行），但 prepare_manifest.json 因空间清理丢失时，
用此脚本快速重建 manifest，避免重跑整个 prepare pipeline。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
import gzip
import pickle

ROOT = Path(__file__).resolve().parents[1]  # E:/Q4
sys.path.insert(0, str(ROOT))

from liquidloc.dataio.manifests.build_manifests import REQUIRED_STREAMS, build_manifests
from liquidloc.common.io_utils import dumps_json_text


def rebuild_manifest(raw_root: Path, output_root: Path, events_root: Path | None = None) -> dict[str, Any]:
    """重建 prepare_manifest.json，从已有 events 反推场景/参数。

    Args:
        raw_root: 原始数据根目录。
        output_root: prepare 输出根目录（含 {seq_id}_events.json）。
        events_root: 可选，与 output_root 不同时指定事件目录。

    Returns:
        写出的 manifest_payload。
    """
    events_root = events_root or output_root
    dataset_manifest, _ = build_manifests(raw_root, required_streams=REQUIRED_STREAMS)
    manifest_records_by_seq = {r["seq_id"]: r for r in dataset_manifest.get("sequences", [])}

    per_sequence: dict[str, Any] = {}
    scene_to_seq_ids: dict[str, list[str]] = {}
    dataset_sequences: list[dict[str, Any]] = []

    empty_files: list[str] = []
    for events_path in sorted(events_root.glob("*_events.pkl.gz")) + sorted(events_root.glob("*_events.json")):
        seq_id = events_path.name[: -len("_events.pkl.gz")] if events_path.name.endswith("_events.pkl.gz") else events_path.name[: -len("_events.json")]
        if events_path.stat().st_size == 0:
            empty_files.append(events_path.name)
            print(f"[skip-empty] {events_path.name}", flush=True)
            continue
        try:
            if events_path.name.endswith("_events.pkl.gz"):
                import gzip
                import pickle
                with gzip.open(events_path, "rb") as fh:
                    events = pickle.load(fh)
            else:
                with events_path.open("r", encoding="utf-8") as fh:
                    events = json.load(fh)
        except (json.JSONDecodeError, EOFError) as exc:
            print(f"[skip-corrupt] {events_path.name}: {exc}", flush=True)
            continue
        if not events:
            print(f"[skip] empty events list: {events_path}", flush=True)
            continue
        first_meta = events[0].get("meta", {}) or {}
        scene_id = first_meta.get("scene_id", seq_id)
        scene_parameters = first_meta.get("scene_parameters")
        seq_payload: dict[str, Any] = {
            "seq_id": seq_id,
            "scene_id": scene_id,
            "read_report": {"source": "rebuilt_from_events"},
            "check_report": {"status": "skipped_rebuild"},
            "event_count": len(events),
        }
        if scene_parameters is not None:
            seq_payload["scene_parameters"] = scene_parameters
        per_sequence[seq_id] = seq_payload
        scene_to_seq_ids.setdefault(scene_id, []).append(seq_id)

        record = manifest_records_by_seq.get(seq_id)
        if record is None:
            print(f"[warn] seq_id={seq_id!r} not in raw_root manifest; skipping", flush=True)
            continue
        dataset_sequence = {**record, "scene_id": scene_id}
        if scene_parameters is not None:
            dataset_sequence["scene_parameters"] = scene_parameters
        dataset_sequences.append(dataset_sequence)

    dataset_manifest = {
        **dataset_manifest,
        "sequence_count": len(dataset_sequences),
        "sequences": dataset_sequences,
    }
    scene_manifest = {
        "scene_count": len(scene_to_seq_ids),
        "scenes": [
            {"scene_id": sid, "seq_ids": seqs}
            for sid, seqs in sorted(scene_to_seq_ids.items())
        ],
    }
    manifest_payload = {
        "dataset_manifest": dataset_manifest,
        "scene_manifest": scene_manifest,
        "sequences": per_sequence,
    }
    manifest_path = output_root / "prepare_manifest.json"
    manifest_path.write_text(dumps_json_text(manifest_payload), encoding="utf-8")
    print(f"[rebuild] wrote {manifest_path} (seq={len(per_sequence)}, scenes={len(scene_to_seq_ids)})", flush=True)
    return manifest_payload


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Rebuild prepare_manifest.json from existing event files.")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--events-root", default=None)
    args = parser.parse_args()

    raw_root = Path(args.raw_root).resolve()
    output_root = Path(args.output_root).resolve()
    events_root = Path(args.events_root).resolve() if args.events_root else None
    if not raw_root.is_dir():
        print(f"[error] raw_root not a directory: {raw_root}", flush=True)
        return 2
    if not output_root.is_dir():
        print(f"[error] output_root not a directory: {output_root}", flush=True)
        return 2
    rebuild_manifest(raw_root, output_root, events_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
