"""Paper 数据集训练入口：3 神经网络 (lstm_ekf / liquid_ekf / transformer_ekf)。

对每个 seed 依次训练 3 个模型，写到 --output-root-prefix/seed{N}/{model_name}/checkpoints/。
本质是 scripts/05_train_lstm.py + 06_train_liquid.py + 07_train_transformer.py
的并行编排器，复用它们的子进程入口。

与原 05/06/07 训练脚本完全隔离：不修改它们的 default checkpoint 路径，
通过 CLI 参数把输出写到 paper 专用目录。训练前会临时把 configs/models/*.yaml
的 checkpoint_path 字段清空（quick 模式不允许复用）。
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_YAMLS = {
    "lstm_ekf": "configs/models/lstm_ekf.yaml",
    "liquid_ekf": "configs/models/liquid_ekf.yaml",
    "transformer_ekf": "configs/models/transformer_ekf.yaml",
}


def _run_train(model_name: str, events_root: Path, raw_root: Path, output_root: Path, epochs: int, seed: int, extra_args: list[str]) -> int:
    """调用 05/06/07 之一训练一个模型。"""
    script_map = {
        "lstm_ekf": "scripts/05_train_lstm.py",
        "liquid_ekf": "scripts/06_train_liquid.py",
        "transformer_ekf": "scripts/07_train_transformer.py",
    }
    script = script_map[model_name]
    # 临时清空 model yaml 的 checkpoint_path（quick 模式不允许复用旧 checkpoint）
    yaml_path = PROJECT_ROOT / MODEL_YAMLS[model_name]
    backup = yaml_path.read_text(encoding="utf-8")
    cleared = re.sub(r"^(\s*checkpoint_path\s*:\s*).*$", r"\1", backup, flags=re.MULTILINE)
    yaml_path.write_text(cleared, encoding="utf-8")
    try:
        cmd = [
            sys.executable, script,
            "--dataset-name", "sim",
            "--events-root", str(events_root),
            "--raw-root", str(raw_root),
            "--output-root", str(output_root),
            "--epochs", str(epochs),
            "--seed", str(seed),
        ] + extra_args
        print(f"[23_train_3nn_paper] running: {' '.join(cmd)}")
        return subprocess.call(cmd, cwd=str(PROJECT_ROOT))
    finally:
        yaml_path.write_text(backup, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 3 NN on paper_main dataset")
    parser.add_argument("--events-root", type=str, default="outputs/prepare_paper_main_v1")
    parser.add_argument("--raw-root", type=str, default="data/raw/paper_main_v1",
                        help="raw data root (for ground_truth lookup during training)")
    parser.add_argument("--output-root-prefix", type=str, default="outputs/train_paper_main")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--n-seed", type=int, default=None, help="override n_seed (default from paper_dataset.yaml)")
    parser.add_argument("--models", type=str, nargs="+", default=["lstm_ekf", "liquid_ekf", "transformer_ekf"])
    parser.add_argument("--seq-batch-size", type=int, default=50, help="passed to LSTM/Transformer")
    parser.add_argument("--batch-size", type=int, default=64, help="passed to Liquid/Transformer")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    events_root = Path(args.events_root).resolve()
    raw_root = Path(args.raw_root).resolve()
    output_prefix = Path(args.output_root_prefix).resolve()
    print(f"[23_train_3nn_paper] events_root={events_root}")
    print(f"[23_train_3nn_paper] raw_root={raw_root}")
    print(f"[23_train_3nn_paper] output_prefix={output_prefix}")
    print(f"[23_train_3nn_paper] epochs={args.epochs} models={args.models}")

    # 决定 n_seed
    n_seed = args.n_seed
    if n_seed is None:
        import yaml
        paper_cfg = yaml.safe_load((PROJECT_ROOT / "configs/datasets/paper_dataset.yaml").read_text(encoding="utf-8"))
        n_seed = paper_cfg["paper_spec"]["scale"]["n_seed"]
    print(f"[23_train_3nn_paper] n_seed={n_seed}")

    failures = 0
    for seed_id in range(n_seed):
        seed_events_root = events_root / f"seed{seed_id}" if (events_root / f"seed{seed_id}").exists() else events_root
        for model_name in args.models:
            output_root = output_prefix / f"seed{seed_id}" / model_name
            output_root.mkdir(parents=True, exist_ok=True)
            extra_args = ["--seq-batch-size", str(args.seq_batch_size)]
            if model_name != "lstm_ekf":
                extra_args += ["--batch-size", str(args.batch_size)]
            rc = _run_train(model_name, seed_events_root, raw_root, output_root, args.epochs, seed_id, extra_args)
            if rc != 0:
                print(f"[23_train_3nn_paper] FAILED: seed={seed_id} model={model_name} rc={rc}")
                failures += 1

    if failures:
        print(f"[23_train_3nn_paper] DONE with {failures} failures")
        return 1
    print(f"[23_train_3nn_paper] DONE: {n_seed} seeds × {len(args.models)} models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
