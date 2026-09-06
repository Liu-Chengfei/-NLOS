"""Run minimal real Liquid training smoke — E9 multi-pathway (miluv + sim)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# 与 05_train_lstm.py 同口径修复 (2026-08-17): alignment_risk 标签默认失败阈值 1.0m,
# sim_e9 跨度 ~25m 场景下 pose_error 几乎永远 > 1m, risk 标签 saturated 到 1.0,
# Liquid 学到常量"全拒收", OOD 评估时 mahalanobis_sq 拒收所有观测, 只剩 IMU 漂移。
# 用 25.0m 归一化让 risk 标签出现分化, 才能 learn 到非平凡风险信号。
os.environ.setdefault("LIQUIDLOC_ALIGNMENT_POSE_FULL_SCALE_M", "25.0")
os.environ.setdefault("LIQUIDLOC_SUPPRESS_PRINT_DICT", "1")

from liquidloc.common.config_utils import load_yaml_config
from liquidloc.pipelines.train_pipeline import TrainPipeline

_MILUV_CONFIG = ROOT / "configs" / "datasets" / "miluv.yaml"
_LIQUID_MODEL_CFG_PATH = ROOT / "configs" / "models" / "liquid_ekf.yaml"


def main(argv=None):
    """Run Liquid training smoke test.  E9 multi-pathway: miluv (default) or sim direct-input."""
    print("[06_liquid_train] 开始 | model=liquid_ekf mode=quick", flush=True)
    parser = argparse.ArgumentParser(description="Liquid training smoke — E9 multi-pathway (miluv or sim)")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--split-ids", default="mini_seq", help="Comma-separated seq IDs (default: mini_seq)")
    parser.add_argument("--split-ids-file", default=None, help="Path to file containing comma-separated seq IDs")
    parser.add_argument("--seed", type=int, default=None, help="Training random seed (default: 0 from yaml)")
    parser.add_argument("--dataset-name", default="miluv", help="Dataset name: miluv or sim (default: miluv)")
    parser.add_argument("--events-root", default=None, help="02_prepare_sim_data.py output dir (required for --dataset-name=sim)")
    parser.add_argument("--raw-root", default=None, help="Raw data dir (default: tests/fixtures/datasets/miluv)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override trainer_state['epochs'] (must equal existing train_report.epochs to allow resume)")
    parser.add_argument("--val-split-ids", default=None,
                        help="Comma-separated val seq IDs (passed through to train_payload['val_split_ids']; "
                             "if omitted, only --split-ids is used as the unified set)")
    parser.add_argument("--val-split-ids-file", default=None,
                        help="Path to file containing comma-separated val seq IDs (alternative to --val-split-ids "
                             "to avoid OS argv length limit when val set is large, e.g. 1674 IDs)")
    parser.add_argument("--train-split-ids", default=None,
                        help="Comma-separated train seq IDs (passed through to train_payload['train_split_ids']; "
                             "if omitted, only --split-ids is used as the unified set)")
    parser.add_argument("--train-split-ids-file", default=None,
                        help="Path to file containing comma-separated train seq IDs (alternative to --train-split-ids)")
    parser.add_argument("--seq-batch-size", type=int, default=None,
                        help="Streaming batch size: split split_ids into sub-batches of N seq each, "
                             "run train_pipeline per batch, share model/optimizer state. "
                             "Use to avoid OOM when total seq count is large (e.g. 1860 sim seq, 8GB RAM). "
                             "Default: None = full-batch (legacy behavior).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override model_cfg.train.batch_size (default: from liquid_ekf.yaml = 64). "
                             "Reduce to e.g. 16 or 32 to avoid CUDA OOM on 8GB laptop GPUs.")
    parser.add_argument("--disable-scene-leak-check", action="store_true",
                        help="Bypass §0.2 scene_id leakage check. Required for sim_e9 single-scene protocol.")
    args = parser.parse_args(argv)
    from liquidloc.common.tee_logger import print_args, print_dict
    print_args(args, "06_train_liquid")
    print("[06_liquid_train] 加载配置 | model_cfg=liquid_ekf.yaml", flush=True)
    liquid_cfg = load_yaml_config(_LIQUID_MODEL_CFG_PATH)
    liquid_cfg["checkpoint_path"] = None  # quick mode 禁止复用 checkpoint
    if args.seed is not None:
        liquid_cfg.setdefault("train", {})["seed"] = args.seed
    if args.batch_size is not None and args.batch_size > 0:
        liquid_cfg.setdefault("train", {})["batch_size"] = int(args.batch_size)
        print(f"[06_liquid_train] 覆盖 batch_size={int(args.batch_size)}", flush=True)
    print_dict(liquid_cfg, "Liquid 模型配置 (liquid_ekf.yaml)")
    dataset_cfg = load_yaml_config(_MILUV_CONFIG)
    if args.dataset_name == "miluv":
        print_dict(dataset_cfg, "数据集配置 (miluv.yaml)")
    if args.split_ids_file:
        with open(args.split_ids_file) as f:
            raw_ids = f.read().strip()
        split_ids = [s.strip() for s in raw_ids.split(",") if s.strip()]
        print(f"[06_liquid_train] 从文件加载 split_ids: {len(split_ids)} 条", flush=True)
    else:
        split_ids = [s.strip() for s in args.split_ids.split(",") if s.strip()]
    # val_split_ids: 支持文件读取（避免 1674 条 ID 导致 OS argv 长度超限）
    if args.val_split_ids is not None:
        val_split_ids = [s.strip() for s in args.val_split_ids.split(",") if s.strip()]
        print(f"[06_liquid_train] val_split_ids (CLI): {len(val_split_ids)} 条", flush=True)
    elif args.val_split_ids_file:
        with open(args.val_split_ids_file) as f:
            raw = f.read().strip()
        val_split_ids = [s.strip() for s in raw.split(",") if s.strip()]
        print(f"[06_liquid_train] val_split_ids (文件 {args.val_split_ids_file}): {len(val_split_ids)} 条", flush=True)
    else:
        val_split_ids = None
    # train_split_ids: 支持文件读取
    if args.train_split_ids is not None:
        train_split_ids = [s.strip() for s in args.train_split_ids.split(",") if s.strip()]
        print(f"[06_liquid_train] train_split_ids (CLI): {len(train_split_ids)} 条", flush=True)
    elif args.train_split_ids_file:
        with open(args.train_split_ids_file) as f:
            raw = f.read().strip()
        train_split_ids = [s.strip() for s in raw.split(",") if s.strip()]
        print(f"[06_liquid_train] train_split_ids (文件 {args.train_split_ids_file}): {len(train_split_ids)} 条", flush=True)
    else:
        train_split_ids = None
    raw_root = str(ROOT / "tests" / "fixtures" / "datasets" / "miluv") if args.raw_root is None else args.raw_root
    # P2 修复 (2026-09-02): 改用 mode=full，跑满 60 epoch 而不是 quick (20 epoch)
    # Liquid 在 quick 模式下被 _enforce_train_device_constraints 拒绝继续训练
    train_payload = {
        "model_name": "liquid_ekf",
        "model_cfg": dict(liquid_cfg),
        "mode": "full",
        "dataset_name": args.dataset_name,
        "raw_root": raw_root,
        "field_mapping": dict(dataset_cfg.get("field_mapping") or {}),
        "split_ids": split_ids,
        "output_root": args.output_root or str(ROOT / "outputs" / "train_liquid_smoke"),
    }
    if getattr(args, "disable_scene_leak_check", False):
        train_payload["disable_scene_leak_check"] = True
    # 支持显式 train_split_ids / val_split_ids 划分（10%/90% 等比例）。
    if train_split_ids is not None:
        train_payload["train_split_ids"] = train_split_ids
        print(f"[06_liquid_train] 显式 train_split_ids: {len(train_split_ids)} 条", flush=True)
    if val_split_ids is not None:
        train_payload["val_split_ids"] = val_split_ids
        print(f"[06_liquid_train] 显式 val_split_ids: {len(val_split_ids)} 条", flush=True)
    if args.epochs is not None:
        # 注入到 model_cfg.train.epochs，pipeline 会读取它构建 trainer_state
        _epochs_int = int(args.epochs)
        train_payload["model_cfg"].setdefault("train", {})["epochs"] = _epochs_int
        # 关键修复：liquid_ekf.yaml 默认配置全为 160 epochs 设计 (phase_schedule 和=160,
        # lr_scheduler.T_max=160). 当用户用 --epochs < 160 时这些校验都会触发失败。
        # 在 quick 实验模式下自动同步覆盖这些与 epochs 强耦合的配置，让训练能跑通。
        _train_cfg = train_payload["model_cfg"]["train"]
        if isinstance(_train_cfg, dict):
            # 1) phase_schedule: 必须 sync full_tuning_epochs，否则 _resolve_phase_epochs 校验失败
            _existing_phase = _train_cfg.get("phase_schedule")
            if isinstance(_existing_phase, dict):
                _existing_full = int(_existing_phase.get("full_tuning_epochs", _epochs_int))
                if _existing_full != _epochs_int:
                    _train_cfg["phase_schedule"] = {
                        "warmup_epochs": 0,
                        "gate_alignment_epochs": 0,
                        "full_tuning_epochs": _epochs_int,
                    }
                    print(f"[06_liquid_train] quick mode: 自动覆盖 phase_schedule 到 warmup=0/gate=0/full={_epochs_int} 以适配 --epochs={_epochs_int}", flush=True)
            # 2) lr_scheduler.T_max 必须 == epochs
            _lr_sched = _train_cfg.get("lr_scheduler")
            if isinstance(_lr_sched, dict) and _lr_sched.get("enabled"):
                _existing_t_max = int(_lr_sched.get("T_max", 0))
                if _existing_t_max != _epochs_int:
                    _lr_sched["T_max"] = _epochs_int
                    print(f"[06_liquid_train] quick mode: 自动同步 lr_scheduler.T_max={_epochs_int} 以适配 --epochs={_epochs_int}", flush=True)
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
        _user_split_ids = split_ids
        _is_user_specified = bool(_user_split_ids) and _user_split_ids != ["mini_seq"]
        # 关键修复：用户传 --split-ids=ALL 时，等价于"加载 manifest 全部序列"。
        # 否则会因 'ALL' 不在 manifest 序列集合里而误判为无交集。
        _is_all_request = (
            len(_user_split_ids) == 1 and _user_split_ids[0].upper() == "ALL"
        )
        if _is_all_request:
            split_ids = list(_seq_ids)
            print(
                f"[06_liquid_train] --split-ids=ALL: 加载 manifest 全部 {len(split_ids)} seq",
                flush=True,
            )
        elif _is_user_specified:
            # 用户显式指定了序列 ID，只取与 manifest 的交集
            # （shrink 到真实的子集），保证不超出 manifest 也不丢用户意图。
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
        _events_by_seq_id = load_prepared_events_by_seq_id(_prepare_root, split_ids)
        _raw_root = Path(args.raw_root) if args.raw_root else ROOT / "data/raw/sim_e9_main"
        _gt_by_seq_id = load_ground_truth_by_seq_id(_raw_root, split_ids)
        train_payload["events_by_seq_id"] = _events_by_seq_id
        train_payload["ground_truth_by_seq_id"] = _gt_by_seq_id
        # 修复：原代码把整个 manifest 当作每个 seq 的 source_report 传入，
        # 但 _resolve_sequence_payload 期望 source_report['anchor_layout'] 直接可用——
        # 实际 anchor_layout 嵌在 manifest['sequences'][sid]['anchor_layout'] 下，
        # 顶层 manifest 没有 anchor_layout 键，导致训练时 bias teacher 被跳过
        # （bias_source 恒为 neutral_baseline，bias_head 永远学不到非零权重）。
        _seq_manifest = _source_report.get("sequences", {}) if isinstance(_source_report, dict) else {}
        _source_report_by_seq = {}
        for _sid in split_ids:
            _sr = _seq_manifest.get(_sid) or {}
            _source_report_by_seq[_sid] = _sr
        train_payload["source_report_by_seq_id"] = _source_report_by_seq
        for _k in ("dataset_name", "raw_root", "field_mapping"):
            del train_payload[_k]
    # === pathway end ===
    # 注意：print_dict(train_payload) 会遍历 events_by_seq_id.items()，
    # 触发 _LazyEventsDict 加载全部 1860 个序列（47GB），导致 OOM。
    # 改为打印 payload 的 keys 与统计摘要，不展开值。
    _payload_keys = sorted(train_payload.keys())
    print(f"[06_liquid_train] 训练 payload keys={_payload_keys}", flush=True)
    if "events_by_seq_id" in train_payload:
        _ev = train_payload["events_by_seq_id"]
        print(f"[06_liquid_train] events_by_seq_id: {len(_ev)} 个序列（惰性加载，未展开）", flush=True)
    if "ground_truth_by_seq_id" in train_payload:
        _gt = train_payload["ground_truth_by_seq_id"]
        print(f"[06_liquid_train] ground_truth_by_seq_id: {len(_gt)} 个序列", flush=True)
    print(f"[06_liquid_train] 运行训练流水线 | split_ids={split_ids[:5]}... 共 {len(split_ids)} 个", flush=True)
    if args.seq_batch_size and args.seq_batch_size > 0 and len(split_ids) > args.seq_batch_size:
        # === 流式分批训练（§13.8 OOM 根因修复 1860 sim seq 不能单批训练）===
        # 将 N 个 seq_id 拆分为 ceil(N/seq_batch_size) 个子批次
        # 每个子批次独立运行 train_pipeline，但 output_root 共用（保留 checkpoint 跨批续训）
        # 这样每批 50 seq → 样本构建 ≈ 800 MB（远小于全量 ~28 GB）
        _bs = int(args.seq_batch_size)
        _n_batches = (len(split_ids) + _bs - 1) // _bs
        print(f"[06_liquid_train] 流式分批模式: seq_batch_size={_bs}, 共 {_n_batches} 批", flush=True)
        from liquidloc.common.io_utils import dumps_json_text
        import gc
        last_train_report = {}
        last_stage_name = ""
        for _bi in range(_n_batches):
            _batch_start = _bi * _bs
            _batch_end = min(len(split_ids), _batch_start + _bs)
            _batch_ids = split_ids[_batch_start:_batch_end]
            print(f"[06_liquid_train] === 批次 {_bi+1}/{_n_batches} === seq_idx={_batch_start}..{_batch_end-1} ({len(_batch_ids)} seq)", flush=True)
            _batch_payload = dict(train_payload)
            # 流式分批核心：split_ids 窄化为本批 seq（事件/gt 同步收缩）。
            # 不能保留全量 1860（events_by_seq_id 只有本批 30 seq，pipeline 会为缺失的
            # 1830 seq 抛出 _SeqIdNotFound）。全量验证在训练后单独跑推理脚本做。
            #
            # 关键：不传 train_split_ids/val_split_ids，让 pipeline 走 multi-seq
            # fall-through（最后一个 seq 做 val）。E9 单 scene 设计下 scene 守门会
            # 把所有切分都判定为泄漏，所以只能走 fall-through 路径跳过守门。
            _batch_payload["split_ids"] = _batch_ids
            _batch_payload.pop("train_split_ids", None)
            _batch_payload.pop("val_split_ids", None)
            # §13.8 流式分批：把 events / ground_truth / source_report 收缩到本批 seq_id 子集。
            # 这样 train_pipeline 内部持有的引用限定在 ~30 seq 内（~400 MB），而非全部 1860 seq（~28 GB）。
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
            print(f"[06_liquid_train] 批次 {_bi+1}/{_n_batches} 完成 | best_loss={last_train_report.get('best_loss')}", flush=True)
            # §13.8 关键：批次处理完毕强制释放本轮 result/payload 引用，防止跨批累积。
            # 实测未释放时每批泄漏 ~1.3 GB（pipeline 内部持有 events/gt 引用）。
            del result
            del _batch_payload
            gc.collect()
        print(dumps_json_text({"stage_name": last_stage_name, "model_name": "liquid_ekf", **last_train_report}))
        exit_code = 0
        print(f"[06_liquid_train] 全部 {_n_batches} 批完成 | 返回码={exit_code}", flush=True)
        return exit_code
    result = TrainPipeline().run(train_payload)
    train_report = result.metadata.get("train_report") or {}
    from liquidloc.common.io_utils import dumps_json_text
    print(dumps_json_text({"stage_name": result.stage_name, "model_name": "liquid_ekf", **train_report}))
    exit_code = 0
    print(f"[06_liquid_train] 完成 | 返回码={exit_code}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
