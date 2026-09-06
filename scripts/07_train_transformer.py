"""Transformer+EKF 训练脚本 — E9 multi-pathway（miluv 或 sim）。

设计目标：
- 与 05_train_lstm.py / 06_train_liquid.py 完全同口径（输入输出、checkpoint 路径、流式分批逻辑）。
- Transformer 默认使用 8 raw feature（dt/ax/ay/gz/range/dx/dy/dyaw）、window=20、单层 causal Transformer。
- 与 LSTM / Liquid 公平比对：相同 160 epochs、相同 batch_size=64、相同 lr、相同 lr scheduler
  (CosineAnnealingLR T_max=160 eta_min=5e-5)、相同 4 输出头 (bias/risk/uwb_scaling/vio_scaling)。

调用示例：
  python scripts/07_train_transformer.py --split-ids mini_seq --epochs 5
  python scripts/07_train_transformer.py --dataset-name sim --events-root outputs/prepare_sim_e9_protocol_20260726
"""

from __future__ import annotations

import os

os.environ.setdefault("LIQUIDLOC_SUPPRESS_PRINT_DICT", "1")
# 位置归一化满量程：避免大跨度 sim 场景 pose_error 持续 >1m 时 risk 标签
# 天然 saturate 到 1.0，导致网络学到常量退化函数。LSTM/Liquid 训练脚本
# 同样设置了该变量（见 05_train_lstm.py:18 / 06_train_liquid.py:20）。
os.environ.setdefault("LIQUIDLOC_ALIGNMENT_POSE_FULL_SCALE_M", "25.0")

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from liquidloc.common.config_utils import load_yaml_config
from liquidloc.pipelines.train_pipeline import TrainPipeline

_MILUV_CONFIG = ROOT / "configs" / "datasets" / "miluv.yaml"
_TRANSFORMER_MODEL_CFG_PATH = ROOT / "configs" / "models" / "transformer_ekf.yaml"


def main(argv=None):
    """Run Transformer training smoke test. E9 multi-pathway: miluv or sim direct-input."""
    print("[07_transformer_train] 开始 | model=transformer_ekf mode=quick", flush=True)
    parser = argparse.ArgumentParser(description="Transformer training smoke — E9 multi-pathway (miluv or sim)")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--split-ids", default="mini_seq", help="Comma-separated seq IDs (default: mini_seq)")
    parser.add_argument("--split-ids-file", default=None, help="Path to file containing comma-separated seq IDs")
    parser.add_argument("--seed", type=int, default=None, help="Training random seed (default: 0 from yaml)")
    parser.add_argument("--dataset-name", default="miluv", help="Dataset name: miluv or sim (default: miluv)")
    parser.add_argument("--events-root", default=None, help="02_prepare_sim_data.py output dir (required for --dataset-name=sim)")
    parser.add_argument("--raw-root", default=None, help="Raw data dir (default: tests/fixtures/datasets/miluv)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override trainer_state['epochs'] (must equal existing train_report.epochs to allow resume)")
    parser.add_argument("--train-split-ids", default=None,
                        help="Comma-separated train seq IDs")
    parser.add_argument("--train-split-ids-file", default=None,
                        help="Path to file containing comma-separated train seq IDs (alternative to --train-split-ids "
                             "to avoid OS argv length limit when train set is large).")
    parser.add_argument("--val-split-ids", default=None,
                        help="Comma-separated val seq IDs (passed through to train_payload['val_split_ids']; "
                             "if omitted, only --split-ids is used as the unified set)")
    parser.add_argument("--val-split-ids-file", default=None,
                        help="Path to file containing comma-separated val seq IDs (alternative to --val-split-ids "
                             "to avoid OS argv length limit).")
    parser.add_argument("--seq-batch-size", type=int, default=None,
                        help="Streaming batch size: split split_ids into sub-batches of N seq each, "
                             "run train_pipeline per batch, share model/optimizer state.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override model_cfg.train.batch_size (default: from transformer_ekf.yaml = 64).")
    parser.add_argument("--disable-scene-leak-check", action="store_true",
                        help="Bypass §0.2 scene_id leakage check. Required for sim_e9 single-scene protocol.")
    args = parser.parse_args(argv)
    from liquidloc.common.tee_logger import print_args, print_dict
    print_args(args, "07_train_transformer")
    print("[07_transformer_train] 加载配置 | model_cfg=transformer_ekf.yaml", flush=True)
    transformer_cfg = load_yaml_config(_TRANSFORMER_MODEL_CFG_PATH)
    transformer_cfg["checkpoint_path"] = None  # quick mode 禁止复用 checkpoint
    if args.seed is not None:
        transformer_cfg.setdefault("train", {})["seed"] = args.seed
    if args.batch_size is not None and args.batch_size > 0:
        transformer_cfg.setdefault("train", {})["batch_size"] = int(args.batch_size)
        print(f"[07_transformer_train] 覆盖 batch_size={int(args.batch_size)}", flush=True)
    print_dict(transformer_cfg, "Transformer 模型配置 (transformer_ekf.yaml)")
    dataset_cfg = load_yaml_config(_MILUV_CONFIG)
    if args.dataset_name == "miluv":
        print_dict(dataset_cfg, "数据集配置 (miluv.yaml)")
    if args.split_ids_file:
        with open(args.split_ids_file) as f:
            raw_ids = f.read().strip()
        split_ids = [s.strip() for s in raw_ids.split(",") if s.strip()]
        print(f"[07_transformer_train] 从文件加载 split_ids: {len(split_ids)} 条", flush=True)
    else:
        split_ids = [s.strip() for s in args.split_ids.split(",") if s.strip()]
    raw_root = str(ROOT / "tests" / "fixtures" / "datasets" / "miluv") if args.raw_root is None else args.raw_root
    train_payload = {
        "model_name": "transformer_ekf",
        "model_cfg": dict(transformer_cfg),
        "mode": "quick",
        "dataset_name": args.dataset_name,
        "raw_root": raw_root,
        "field_mapping": dict(dataset_cfg.get("field_mapping") or {}),
        "split_ids": split_ids,
        "output_root": args.output_root or str(ROOT / "outputs" / "train_transformer_smoke"),
    }
    if getattr(args, "disable_scene_leak_check", False):
        train_payload["disable_scene_leak_check"] = True
    if args.epochs is not None:
        _epochs_int = int(args.epochs)
        train_payload["model_cfg"].setdefault("train", {})["epochs"] = _epochs_int
        _train_cfg = train_payload["model_cfg"]["train"]
        if isinstance(_train_cfg, dict):
            _existing_phase = _train_cfg.get("phase_schedule")
            if isinstance(_existing_phase, dict):
                _existing_warmup = int(_existing_phase.get("warmup_epochs", 0))
                _existing_gate = int(_existing_phase.get("gate_alignment_epochs", 0))
                if _existing_warmup + _existing_gate > _epochs_int:
                    _train_cfg["phase_schedule"] = {
                        "warmup_epochs": 0,
                        "gate_alignment_epochs": 0,
                        "full_tuning_epochs": _epochs_int,
                    }
                    print(f"[07_transformer_train] quick mode: 自动覆盖 phase_schedule 到 warmup=0/gate=0/full={_epochs_int} 以适配 --epochs={_epochs_int}", flush=True)
            _lr_sched = _train_cfg.get("lr_scheduler")
            if isinstance(_lr_sched, dict) and _lr_sched.get("enabled"):
                _existing_t_max = int(_lr_sched.get("T_max", 0))
                if _existing_t_max != _epochs_int:
                    _lr_sched["T_max"] = _epochs_int
                    print(f"[07_transformer_train] quick mode: 自动同步 lr_scheduler.T_max 由 {_existing_t_max} -> {_epochs_int}", flush=True)
    # Handle train/val split IDs from files (avoids OS argv length limit)
    # In streaming batch mode, we don't pass train_split_ids/val_split_ids to avoid
    # normalize_train_request failing with "train_split_ids must be subset of split_ids"
    # when each batch only carries its own 30 seq in split_ids.
    _streaming_mode = bool(args.seq_batch_size and args.seq_batch_size > 0 and len(split_ids) > args.seq_batch_size)
    if not _streaming_mode:
        if args.train_split_ids_file:
            with open(args.train_split_ids_file) as f:
                _train_ids = [s.strip() for s in f.read().split(",") if s.strip()]
            train_payload["train_split_ids"] = _train_ids
            print(f"[07_transformer_train] 显式 train_split_ids (文件 {args.train_split_ids_file}): {len(_train_ids)} 条", flush=True)
        elif args.train_split_ids:
            train_payload["train_split_ids"] = [s.strip() for s in args.train_split_ids.split(",") if s.strip()]
        if args.val_split_ids_file:
            with open(args.val_split_ids_file) as f:
                _val_ids = [s.strip() for s in f.read().split(",") if s.strip()]
            train_payload["val_split_ids"] = _val_ids
            print(f"[07_transformer_train] 显式 val_split_ids (文件 {args.val_split_ids_file}): {len(_val_ids)} 条", flush=True)
        elif args.val_split_ids:
            train_payload["val_split_ids"] = [s.strip() for s in args.val_split_ids.split(",") if s.strip()]
    else:
        print("[07_transformer_train] 流式批处理模式: 跳过显式 train_split_ids/val_split_ids（由 pipeline 从 batch split_ids 推导）", flush=True)
    # === E9 sim pathway ===
    if args.dataset_name == "sim":
        from liquidloc.common.prepared_inputs import load_prepared_events_by_seq_id
        from liquidloc.common.prepared_inputs import load_ground_truth_by_seq_id
        from liquidloc.common.io_utils import read_json
        _prepare_root = Path(args.events_root) if args.events_root else Path("outputs/prepare_sim_e9_protocol_20260726")
        _manifest_candidate = _prepare_root / "prepare_manifest.json"
        if not _manifest_candidate.is_file():
            _manifest_candidate = _prepare_root / "sim" / "prepare_manifest.json"
        if not _manifest_candidate.is_file():
            raise FileNotFoundError(f"prepare_manifest.json not found under {_prepare_root}")
        _prepare_root = _manifest_candidate.parent
        _source_report = read_json(_manifest_candidate)
        _seq_ids = [
            str(r["seq_id"])
            for r in (_source_report.get("dataset_manifest") or {}).get("sequences", [])
        ]
        if not _seq_ids:
            _seq_ids = list(_source_report.get("sequences", {}).keys())
        _user_split_ids = split_ids
        _is_all_request = (
            len(_user_split_ids) == 1 and _user_split_ids[0].upper() == "ALL"
        )
        _is_user_specified = bool(_user_split_ids) and _user_split_ids != ["mini_seq"]
        if _is_all_request:
            split_ids = list(_seq_ids)
            print(f"[07_transformer_train] --split-ids=ALL: 加载 manifest 全部 {len(split_ids)} seq", flush=True)
        elif _is_user_specified:
            _seq_id_set = set(_seq_ids)
            split_ids = [sid for sid in _user_split_ids if sid in _seq_id_set]
            if not split_ids:
                raise ValueError(
                    f"用户传入的 --split-ids={_user_split_ids} 与 manifest "
                    f"中的 {_seq_ids[:5]}... 序列无交集"
                )
        else:
            split_ids = _seq_ids
        train_payload["split_ids"] = split_ids
        _events_by_seq_id = load_prepared_events_by_seq_id(_prepare_root, split_ids)
        _gt_root = Path(args.raw_root) if args.raw_root else ROOT / "data/raw/sim_e9_5seed_25unit_flat"
        _gt_by_seq_id = load_ground_truth_by_seq_id(_gt_root, split_ids)
        train_payload["events_by_seq_id"] = _events_by_seq_id
        train_payload["ground_truth_by_seq_id"] = _gt_by_seq_id
        _seq_manifest = _source_report.get("sequences", {}) if isinstance(_source_report, dict) else {}
        _source_report_by_seq = {}
        for _sid in split_ids:
            _sr = _seq_manifest.get(_sid) or {}
            _source_report_by_seq[_sid] = _sr
        train_payload["source_report_by_seq_id"] = _source_report_by_seq
        for _k in ("dataset_name", "raw_root", "field_mapping"):
            del train_payload[_k]
    # === pathway end ===
    _payload_keys = sorted(train_payload.keys())
    print(f"[07_transformer_train] 训练 payload keys={_payload_keys}", flush=True)
    if "events_by_seq_id" in train_payload:
        _ev = train_payload["events_by_seq_id"]
        print(f"[07_transformer_train] events_by_seq_id: {len(_ev)} 个序列（惰性加载，未展开）", flush=True)
    if "ground_truth_by_seq_id" in train_payload:
        _gt = train_payload["ground_truth_by_seq_id"]
        print(f"[07_transformer_train] ground_truth_by_seq_id: {len(_gt)} 个序列", flush=True)
    print(f"[07_transformer_train] 运行训练流水线 | split_ids={split_ids[:5]}... 共 {len(split_ids)} 个", flush=True)
    if args.seq_batch_size and args.seq_batch_size > 0 and len(split_ids) > args.seq_batch_size:
        _bs = int(args.seq_batch_size)
        _n_batches = (len(split_ids) + _bs - 1) // _bs
        print(f"[07_transformer_train] 流式分批模式: seq_batch_size={_bs}, 共 {_n_batches} 批", flush=True)
        from liquidloc.common.io_utils import dumps_json_text
        import gc
        last_train_report = {}
        last_stage_name = ""
        for _bi in range(_n_batches):
            _batch_start = _bi * _bs
            _batch_end = min(len(split_ids), _batch_start + _bs)
            _batch_ids = split_ids[_batch_start:_batch_end]
            print(f"[07_transformer_train] === 批次 {_bi+1}/{_n_batches} === seq_idx={_batch_start}..{_batch_end-1} ({len(_batch_ids)} seq)", flush=True)
            _batch_payload = dict(train_payload)
            _batch_payload["split_ids"] = _batch_ids
            if "events_by_seq_id" in _batch_payload:
                _ev_src = _batch_payload["events_by_seq_id"]
                if hasattr(_ev_src, "keys"):
                    _keep = set(_batch_ids) & set(_ev_src.keys())
                    from liquidloc.common.prepared_inputs import _LazyEventsDict
                    if isinstance(_ev_src, _LazyEventsDict):
                        _batch_payload["events_by_seq_id"] = _LazyEventsDict(_ev_src._prepare_root, list(_keep))
                    else:
                        _batch_payload["events_by_seq_id"] = {k: _ev_src[k] for k in _keep}
            if "ground_truth_by_seq_id" in _batch_payload:
                _gt_src = _batch_payload["ground_truth_by_seq_id"]
                if hasattr(_gt_src, "keys"):
                    _keep = set(_batch_ids) & set(_gt_src.keys())
                    _batch_payload["ground_truth_by_seq_id"] = {k: _gt_src[k] for k in _keep}
            if "source_report_by_seq_id" in _batch_payload:
                _sr_src = _batch_payload["source_report_by_seq_id"]
                _batch_payload["source_report_by_seq_id"] = {k: _sr_src.get(k) or {} for k in _batch_ids if k in _sr_src}
            result = TrainPipeline().run(_batch_payload)
            last_train_report = result.metadata.get("train_report") or {}
            last_stage_name = result.stage_name
            print(f"[07_transformer_train] 批次 {_bi+1}/{_n_batches} 完成 | best_loss={last_train_report.get('best_loss')}", flush=True)
            del result
            del _batch_payload
            gc.collect()
        print(dumps_json_text({"stage_name": last_stage_name, "model_name": "transformer_ekf", **last_train_report}))
        exit_code = 0
        print(f"[07_transformer_train] 全部 {_n_batches} 批完成 | 返回码={exit_code}", flush=True)
        return exit_code
    result = TrainPipeline().run(train_payload)
    train_report = result.metadata.get("train_report") or {}
    from liquidloc.common.io_utils import dumps_json_text
    print(dumps_json_text({"stage_name": result.stage_name, "model_name": "transformer_ekf", **train_report}))
    exit_code = 0
    print(f"[07_transformer_train] 完成 | 返回码={exit_code}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
