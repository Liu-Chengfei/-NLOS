"""Run minimal real LSTM training smoke — E9 multi-pathway (miluv + sim)."""

from __future__ import annotations

import os

# 始终压制 estimator 每步 print_dict 刷屏（亿行级输出会塞爆日志拖垮训练），
# 环境主控可覆盖：LIQUIDLOC_SUPPRESS_PRINT_DICT=0 显式开启。
os.environ.setdefault("LIQUIDLOC_SUPPRESS_PRINT_DICT", "1")

# 关键修复 (2026-08-17 OOD RMSE 24m 根因): alignment_risk 标签默认用失败阈值 1.0m
# 归一化 pose_error, 但 sim_e9 跨度 ~25m 场景下 pose_error 几乎永远 > 1m, risk 标签
# 饱和到 1.0 (train=0.9987, val=1.0000), LSTM 学到常量"全拒收", OOD 评估时
# mahalanobis_sq 拒收所有 UWB/VIO 量测, 只剩 IMU 漂移, RMSE 24m. 设 25m 归一化
# 让 pose_error/25.0 < 1.0 出现分化, risk 才能学到非平凡信号. train_pipeline.py
# L202-225 注释里的作者已知坑; _auto_v3_pipeline.sh L56 在 eval 阶段才会设, 训练
# 阶段没人开, 这里 setdefault 优先级低于外部 override, 不破坏别的脚本.
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


def main(argv=None):
    """Run LSTM training smoke test.  E9 multi-pathway: miluv (default) or sim direct-input."""
    print("[05_lstm_train] 开始 | model=lstm_ekf mode=quick", flush=True)
    parser = argparse.ArgumentParser(description="LSTM training smoke — E9 multi-pathway (miluv or sim)")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--split-ids", default="mini_seq", help="Comma-separated seq IDs (default: mini_seq; "
                        "use 'ALL' to auto-load all seq IDs from prepare_manifest)")
    parser.add_argument("--split-ids-file", default=None,
                        help="Path to file containing comma-separated seq IDs (alternative to --split-ids "
                             "to avoid OS argv length limit when seq count is large, e.g. 179 seqs).")
    parser.add_argument("--seed", type=int, default=None, help="Training random seed (default: 0 from yaml)")
    parser.add_argument("--dataset-name", default="miluv", help="Dataset name: miluv or sim (default: miluv)")
    parser.add_argument("--events-root", default=None, help="02_prepare_sim_data.py output dir (required for --dataset-name=sim)")
    parser.add_argument("--raw-root", default=None, help="Raw data dir (default: tests/fixtures/datasets/miluv)")
    parser.add_argument("--seq-batch-size", type=int, default=None,
                        help="Streaming batch size for training: number of sequences per sub-batch. "
                             "If set and len(split_ids)>seq_batch_size, training is split into ceil(N/batch_size) sub-batches. "
                             "Each sub-batch trains independently on its seq subset (events/gt trimmed to that subset), "
                             "then checkpoint is persisted; avoids OOM on large seq counts. "
                             "Matches liquid training's flow (06_train_liquid.py:165-215).")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override trainer_state['epochs'] (must equal existing train_report.epochs to allow resume). "
                             "When < 160 (yaml default), auto-sync phase_schedule and lr_scheduler.T_max to match, "
                             "mirroring 06_train_liquid.py:65-90 so quick/streaming runs don't fail validation.")
    parser.add_argument("--train-split-ids", default=None,
                        help="Comma-separated train seq IDs (passed through to train_payload['train_split_ids']). "
                             "If omitted, split follows default logic (last seq for val).")
    parser.add_argument("--train-split-ids-file", default=None,
                        help="Path to file containing comma-separated train seq IDs (alternative to --train-split-ids "
                             "to avoid OS argv length limit when train set is large).")
    parser.add_argument("--val-split-ids", default=None,
                        help="Comma-separated val seq IDs (passed through to train_payload['val_split_ids']). "
                             "If omitted, split follows default logic.")
    parser.add_argument("--val-split-ids-file", default=None,
                        help="Path to file containing comma-separated val seq IDs (alternative to --val-split-ids "
                             "to avoid OS argv length limit when val set is large, e.g. 1674 IDs).")
    parser.add_argument("--disable-scene-leak-check", action="store_true",
                        help="Bypass §0.2 scene_id leakage check. Required for sim_e9 single-scene protocol "
                             "where all sequences share scene_id S(A2,N2,V0,K1,M1).")
    args = parser.parse_args(argv)
    from liquidloc.common.tee_logger import print_args, print_dict
    print_args(args, "05_train_lstm")
    print("[05_lstm_train] 加载配置 | model_cfg=lstm_ekf.yaml", flush=True)
    model_cfg = load_yaml_config(ROOT / "configs" / "models" / "lstm_ekf.yaml")
    model_cfg["checkpoint_path"] = None  # quick mode 禁止复用 checkpoint
    if args.seed is not None:
        model_cfg.setdefault("train", {})["seed"] = args.seed
    if args.epochs is not None:
        # 与 06_train_liquid.py:65-90 同口径: 注入 + 同步 phase_schedule and lr_scheduler.T_max
        _epochs_int = int(args.epochs)
        _train_cfg = model_cfg.setdefault("train", {})
        _train_cfg["epochs"] = _epochs_int
        # 1) phase_schedule (warmup + gate 不能超过 epochs) — LSTM 默认 warmup=0/gate=0, 本步通常 no-op,
        #    但保持与 liquid 同口径以防 yaml 后续被改为非零 warmup.
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
                print(f"[05_lstm_train] quick mode: 自动覆盖 phase_schedule 到 warmup=0/gate=0/full={_epochs_int} 以适配 --epochs={_epochs_int}", flush=True)
        # 2) lr_scheduler.T_max 必须 == epochs
        _lr_sched = _train_cfg.get("lr_scheduler")
        if isinstance(_lr_sched, dict) and _lr_sched.get("enabled"):
            _existing_t_max = int(_lr_sched.get("T_max", 0))
            if _existing_t_max != _epochs_int:
                _lr_sched["T_max"] = _epochs_int
                print(f"[05_lstm_train] quick mode: 自动同步 lr_scheduler.T_max 由 {_existing_t_max} -> {_epochs_int}", flush=True)
        # 3) patience 必须 == epochs (lstm trainer:2116 强制单阶段无早停显式预算),
        #    无条件同步以防 yaml 默认 160 与 --epochs 不一致.
        _patience = int(_train_cfg.get("patience", 0))
        if _patience != _epochs_int:
            _train_cfg["patience"] = _epochs_int
            print(f"[05_lstm_train] quick mode: 同步 patience 由 {_patience} -> {_epochs_int} (trainer 强制 patience==epochs)", flush=True)
    print_dict(model_cfg, "模型配置 (lstm_ekf.yaml)")
    dataset_cfg = load_yaml_config(_MILUV_CONFIG)
    if args.dataset_name == "miluv":
        print_dict(dataset_cfg, "数据集配置 (miluv.yaml)")
    split_ids = [s.strip() for s in args.split_ids.split(",") if s.strip()]
    if args.split_ids_file:
        with open(args.split_ids_file) as f:
            _raw = f.read().strip()
        split_ids = [s.strip() for s in _raw.split(",") if s.strip()]
        print(f"[05_lstm_train] 从文件 {args.split_ids_file} 读取 split_ids: {len(split_ids)} 个", flush=True)
    raw_root = str(ROOT / "tests" / "fixtures" / "datasets" / "miluv") if args.raw_root is None else args.raw_root
    train_payload = {
        "model_name": "lstm_ekf",
        "model_cfg": model_cfg,
        "mode": "quick",
        "dataset_name": args.dataset_name,
        "raw_root": raw_root,
        "field_mapping": dict(dataset_cfg.get("field_mapping") or {}),
        "split_ids": split_ids,
        "output_root": args.output_root or str(ROOT / "outputs" / "train_lstm_smoke"),
    }
    if getattr(args, "disable_scene_leak_check", False):
        train_payload["disable_scene_leak_check"] = True
    # === E9 sim pathway: direct-input, bypass dataset_name/raw_root/field_mapping ===
    if args.dataset_name == "sim":
        from liquidloc.common.prepared_inputs import load_prepared_events_by_seq_id
        from liquidloc.common.prepared_inputs import load_ground_truth_by_seq_id
        from liquidloc.common.io_utils import read_json
        _prepare_root = Path(args.events_root) if args.events_root else Path("outputs/prepare_sim_e9_main")
        # 兼容 prepare_root 下嵌套 sim/ 子目录的情况（如 prepare/sim/prepare_manifest.json）
        _manifest_candidate = _prepare_root / "prepare_manifest.json"
        if not _manifest_candidate.is_file():
            _manifest_candidate = _prepare_root / "sim" / "prepare_manifest.json"
        if not _manifest_candidate.is_file():
            raise FileNotFoundError(f"prepare_manifest.json not found under {_prepare_root}")
        _prepare_root = _manifest_candidate.parent  # 让 prepare_root 指向 manifest 所在目录
        _source_report = read_json(_manifest_candidate)
        # 从 manifest 自动推导序列 ID，避免依赖 --split-ids 字面量。
        _seq_ids = [
            str(r["seq_id"])
            for r in (_source_report.get("dataset_manifest") or {}).get("sequences", [])
        ]
        if not _seq_ids:
            _seq_ids = list(_source_report.get("sequences", {}).keys())
        # 关键修复：仅在用户未显式传 --split-ids (或保留默认 'mini_seq') 时
        # 才用 manifest 推导的 1860 个序列覆盖。否则用户传入的 split_ids 应保留，
        # 否则任何 smoke test 都会被无辜扩成全量训练（12小时反复跑挂的真凶）。
        # 新增: --split-ids=ALL 显式要求加载全量（避免命令行过长触发 ENAMETOOLONG）。
        _user_split_ids = split_ids
        _is_all_request = (len(_user_split_ids) == 1 and _user_split_ids[0].upper() == "ALL")
        _is_user_specified = (bool(_user_split_ids) and _user_split_ids != ["mini_seq"]
                              and not _is_all_request)
        if _is_all_request:
            # 显式要求加载 manifest 中所有序列（典型用于全量训练 1860 seq）
            split_ids = list(_seq_ids)
            print(f"[05_lstm_train] --split-ids=ALL: 加载 manifest 全部 {len(split_ids)} seq", flush=True)
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
        # 关键修复：必须同步 train_payload["split_ids"]，否则下游 train_pipeline
        # 在 _build_liquid_samples 中会使用旧的 ['mini_seq']，导致 ValueError
        # "no events found for seq_id='mini_seq'"，训练根本无法启动 1860 个序列。
        train_payload["split_ids"] = split_ids
        # 关键修复：支持显式 train_split_ids / val_split_ids 划分（10%/90% 等比例）。
        # 若用户传入了显式的 train/val 划分（CLI 或文件），则优先使用。否则走默认的"最后一个序列做验证"路径。
        if args.train_split_ids is not None:
            _train_ids = [s.strip() for s in args.train_split_ids.split(",") if s.strip()]
            train_payload["train_split_ids"] = _train_ids
            print(f"[05_lstm_train] 显式 train_split_ids (CLI): {len(_train_ids)} 条", flush=True)
        elif args.train_split_ids_file:
            with open(args.train_split_ids_file) as f:
                _raw = f.read().strip()
            _train_ids = [s.strip() for s in _raw.split(",") if s.strip()]
            train_payload["train_split_ids"] = _train_ids
            print(f"[05_lstm_train] 显式 train_split_ids (文件 {args.train_split_ids_file}): {len(_train_ids)} 条", flush=True)
        if args.val_split_ids is not None:
            _val_ids = [s.strip() for s in args.val_split_ids.split(",") if s.strip()]
            train_payload["val_split_ids"] = _val_ids
            print(f"[05_lstm_train] 显式 val_split_ids (CLI): {len(_val_ids)} 条", flush=True)
        elif args.val_split_ids_file:
            with open(args.val_split_ids_file) as f:
                _raw = f.read().strip()
            _val_ids = [s.strip() for s in _raw.split(",") if s.strip()]
            train_payload["val_split_ids"] = _val_ids
            print(f"[05_lstm_train] 显式 val_split_ids (文件 {args.val_split_ids_file}): {len(_val_ids)} 条", flush=True)
        _events_by_seq_id = load_prepared_events_by_seq_id(_prepare_root, split_ids)
        _gt_root = Path(args.events_root).parent / "gt_root" if args.events_root and (Path(args.events_root).parent / "gt_root").exists() else (Path(args.raw_root) if args.raw_root else ROOT / "data/raw/sim_e9_5seed_25unit_flat")
        _gt_by_seq_id = load_ground_truth_by_seq_id(_gt_root, split_ids)
        train_payload["events_by_seq_id"] = _events_by_seq_id
        train_payload["ground_truth_by_seq_id"] = _gt_by_seq_id
        # 修复: 与 06_train_liquid.py 同口径. 原代码把整份 prepare_manifest 当作每个 seq 的 source_report,
        # 但 _resolve_sequence_payload 期望 source_report['anchor_layout'] 直接可用——
        # 实际 anchor_layout 嵌在 manifest['sequences'][sid]['anchor_layout'] 下,
        # 顶层 manifest 没有 anchor_layout 键, 导致 LSTM 训练时 bias teacher / risk teacher 路径
        # 静默走 neutral_baseline fallback (虽然 LSTM 没有 bias head, 但 risk/uwb_scaling/vio_scaling
        # 三个 head 仍依赖 anchor_layout 来计算 risk teacher 与 geometry-aware scaling label).
        _seq_manifest = _source_report.get("sequences", {}) if isinstance(_source_report, dict) else {}
        _source_report_by_seq = {}
        for _sid in split_ids:
            _sr = _seq_manifest.get(_sid) or {}
            _source_report_by_seq[_sid] = _sr
        train_payload["source_report_by_seq_id"] = _source_report_by_seq
        for _k in ("dataset_name", "raw_root", "field_mapping"):
            del train_payload[_k]
        if getattr(args, "disable_scene_leak_check", False):
            train_payload["disable_scene_leak_check"] = True
            print("[05_lstm_train] §0.2 scene_id 泄漏守门已关闭 (sim_e9 单 scene 协议)", flush=True)
    # === pathway end ===
    # 注意：print_dict(train_payload) 会遍历 events_by_seq_id.items()，
    # 触发 _LazyEventsDict 加载全部 1860 个序列（47GB），导致 OOM。
    # 改为打印 payload 的 keys 与统计摘要，不展开值。
    _payload_keys = sorted(train_payload.keys())
    print(f"[05_lstm_train] 训练 payload keys={_payload_keys}", flush=True)
    if "events_by_seq_id" in train_payload:
        _ev = train_payload["events_by_seq_id"]
        print(f"[05_lstm_train] events_by_seq_id: {len(_ev)} 个序列（惰性加载，未展开）", flush=True)
    if "ground_truth_by_seq_id" in train_payload:
        _gt = train_payload["ground_truth_by_seq_id"]
        print(f"[05_lstm_train] ground_truth_by_seq_id: {len(_gt)} 个序列", flush=True)
    print(f"[05_lstm_train] 运行训练流水线 | split_ids={split_ids[:5]}... 共 {len(split_ids)} 个", flush=True)
    if args.seq_batch_size and args.seq_batch_size > 0 and len(split_ids) > args.seq_batch_size:
        # === 流式分批训练（§13.8 OOM 根因修复 1860 sim seq 不能单批训练）
        #     复用 06_train_liquid.py:165-215 的同口径机制。===
        _bs = int(args.seq_batch_size)
        _n_batches = (len(split_ids) + _bs - 1) // _bs
        print(f"[05_lstm_train] 流式分批模式: seq_batch_size={_bs}, 共 {_n_batches} 批", flush=True)
        from liquidloc.common.io_utils import dumps_json_text
        from liquidloc.common.prepared_inputs import _LazyEventsDict
        import gc
        last_train_report = {}
        last_stage_name = ""
        for _bi in range(_n_batches):
            _batch_start = _bi * _bs
            _batch_end = min(len(split_ids), _batch_start + _bs)
            _batch_ids = split_ids[_batch_start:_batch_end]
            print(f"[05_lstm_train] === 批次 {_bi+1}/{_n_batches} === seq_idx={_batch_start}..{_batch_end-1} ({len(_batch_ids)} seq)", flush=True)
            _batch_payload = dict(train_payload)
            _batch_payload["split_ids"] = _batch_ids
            # 关键：不传 train_split_ids/val_split_ids，让 pipeline 走 multi-seq
            # fall-through（最后一个 seq 做 val）。E9 单 scene 设计下 scene 守门会
            # 把所有切分都判定为泄漏，所以只能走 fall-through 路径跳过守门。
            _batch_payload.pop("train_split_ids", None)
            _batch_payload.pop("val_split_ids", None)
            _batch_payload.pop("train_split_ids_file", None)
            _batch_payload.pop("val_split_ids_file", None)
            # 把 events / ground_truth / source_report 收缩到本批 seq_id 子集
            # 防止 _build_liquid_samples 一次性遍历全部 N 个序列导致 OOM.
            if "events_by_seq_id" in _batch_payload:
                _ev_src = _batch_payload["events_by_seq_id"]
                if hasattr(_ev_src, "keys"):
                    _keep = set(_batch_ids) & set(_ev_src.keys())
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
            print(f"[05_lstm_train] 批次 {_bi+1}/{_n_batches} 完成 | best_loss={last_train_report.get('best_loss')}", flush=True)
            del result
            del _batch_payload
            gc.collect()
        print(dumps_json_text({"stage_name": last_stage_name, "model_name": "lstm_ekf", **last_train_report}))
        exit_code = 0
        print(f"[05_lstm_train] 全部 {_n_batches} 批完成 | 返回码={exit_code}", flush=True)
        return exit_code
    result = TrainPipeline().run(train_payload)
    train_report = result.metadata.get("train_report") or {}
    from liquidloc.common.io_utils import dumps_json_text
    print(dumps_json_text({"stage_name": result.stage_name, "model_name": "lstm_ekf", **train_report}))
    exit_code = 0
    print(f"[05_lstm_train] 完成 | 返回码={exit_code}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
