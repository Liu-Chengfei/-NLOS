"""Paper 数据集 prepare 入口。

复用 PreparePipeline.run，把 paper_main raw 数据（每 seed 400 序列）写成
{seq_id}_events.pkl.gz + prepare_manifest.json 到 --output-root/。

与原 sim 数据集 prepare 完全隔离：不同 raw_root、不同 output_root，但 dataset_name
仍用 "sim"（复用 sim_reader 路径），sim_meta.json 字段契约与 sim_e9 一致
（axes_override / class_id / paper_dataset / generator_version）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from liquidloc.common.config_utils import find_project_root, load_yaml_config
from liquidloc.dataio.manifests.dataset_checks import inspect_sim_materialized_contract
from liquidloc.pipelines.prepare_pipeline import PreparePipeline


def _collect_seq_ids_per_seed(raw_root: Path) -> list[str]:
    """按 seed0/seq_id 格式收集所有 seq_id。"""
    seq_ids: list[str] = []
    for seed_dir in sorted(raw_root.iterdir()):
        if not seed_dir.is_dir():
            continue
        if not seed_dir.name.startswith("seed"):
            continue
        for seq_dir in sorted(seed_dir.iterdir()):
            if seq_dir.is_dir():
                seq_ids.append(seq_dir.name)
    return seq_ids


def _build_scene_id_by_seq(raw_root: Path, seq_ids: list[str]) -> dict[str, str]:
    """从 sim_meta.json 构造 S(A,N,V,G,K) 形式的 scene_id。"""
    scene_id_by_seq: dict[str, str] = {}
    for seq_id in seq_ids:
        # 找这个 seq_id 属于哪个 seed
        candidates = list(raw_root.glob(f"seed*/{seq_id}/sim_meta.json"))
        if not candidates:
            scene_id_by_seq[seq_id] = "S(A2,N2,V0,K3)"  # fallback
            continue
        sim_meta = json.loads(candidates[0].read_text(encoding="utf-8"))
        axes = sim_meta.get("axes_override") or {}
        axis_order = ["A", "N", "V", "K"]
        parts = [axes.get(a, "?") for a in axis_order]
        scene_id_by_seq[seq_id] = f"S({','.join(parts)})"
    return scene_id_by_seq


def _build_seq_id_dirs(raw_root: Path, seq_ids: list[str]) -> dict[str, Path]:
    """每个 seq_id 映射到其 seed 目录（PreparePipeline 需要从 raw_root/<seq_id> 读）。"""
    seq_id_dirs: dict[str, Path] = {}
    for seed_dir in sorted(raw_root.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
            continue
        for seq_dir in sorted(seed_dir.iterdir()):
            if seq_dir.is_dir() and seq_dir.name in seq_ids:
                seq_id_dirs[seq_dir.name] = seq_dir
    return seq_id_dirs


def _ensure_sim_e9_symlinks(raw_root: Path, project_root: Path) -> None:
    """为每条 paper 序列在 sim_e9 raw_root 下建 symlink，绕开 05/06/07 训练脚本
    写死的 'data/raw/sim_e9_main' 路径（仅用于 gt.json 回查）。
    不复制文件、不污染 sim_e9，symlink 失效时下游报错而非静默错误。
    """
    sim_e9_root = project_root / "data" / "raw" / "sim_e9_main"
    sim_e9_root.mkdir(parents=True, exist_ok=True)
    for seed_dir in sorted(raw_root.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
            continue
        for seq_dir in sorted(seed_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            link = sim_e9_root / seq_dir.name
            if link.exists() or link.is_symlink():
                link.unlink()
            # relative symlink：便于跨机器迁移
            rel = Path("..") / ".." / "paper_main_v1" / seed_dir.name / seq_dir.name
            try:
                link.symlink_to(rel, target_is_directory=True)
            except OSError:
                # Windows 可能无 symlink 权限；退化用 junction
                import subprocess
                subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(seq_dir.resolve())], check=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare paper_main dataset")
    parser.add_argument("--paper-config", type=str, default="configs/datasets/paper_dataset.yaml")
    parser.add_argument("--raw-root", type=str, default="data/raw/paper_main_v1")
    parser.add_argument("--output-root", type=str, default="outputs/prepare_paper_main_v1")
    parser.add_argument("--n-seed", type=int, default=None, help="override paper_spec.scale.n_seed")
    parser.add_argument("--max-seqs", type=int, default=None, help="quick: cap total seq_ids (across all seeds)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = find_project_root()
    paper_cfg = load_yaml_config(Path(args.paper_config))
    raw_root = Path(args.raw_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    seq_ids = _collect_seq_ids_per_seed(raw_root)
    if args.max_seqs is not None:
        seq_ids = seq_ids[: args.max_seqs]
        print(f"[22_prepare_paper_data] capped to {len(seq_ids)} seq_ids (max-seqs={args.max_seqs})")
    else:
        print(f"[22_prepare_paper_data] total seq_ids={len(seq_ids)}")

    # 构造 PreparePipeline 配置：dataset_name=sim，raw_root 指向临时合并目录
    # 但 PreparePipeline._enforce_sim_materialized_contract 会读 raw_root 根目录的
    # 子目录列表；这里我们做一个 symlink 合并视图。
    merge_root = project_root / "data" / "raw" / f"_paper_merge_{raw_root.name}"
    if merge_root.exists():
        import shutil as _sh
        _sh.rmtree(merge_root)
    merge_root.mkdir(parents=True)
    for sid in seq_ids:
        for seed_dir in raw_root.iterdir():
            if seed_dir.is_dir() and seed_dir.name.startswith("seed"):
                src = seed_dir / sid
                if src.exists():
                    dst = merge_root / sid
                    if not dst.exists():
                        try:
                            dst.symlink_to(src.resolve(), target_is_directory=True)
                        except OSError:
                            import shutil as _sh
                            _sh.copytree(src, dst, dirs_exist_ok=False)
                    break

    contract_report = inspect_sim_materialized_contract(merge_root)
    print(f"[22_prepare_paper_data] sim contract valid: {contract_report.get('is_valid')}")
    if not contract_report.get("is_valid"):
        print(f"[22_prepare_paper_data] contract report: {contract_report}")
        return 1

    scene_id_by_seq = _build_scene_id_by_seq(raw_root, seq_ids)
    print(f"[22_prepare_paper_data] sample scene_id: {next(iter(scene_id_by_seq.values()))}")

    # 为绕开 05/06/07 训练脚本的 sim_e9 hardcoded raw_root，建 symlink
    _ensure_sim_e9_symlinks(raw_root, project_root)

    pipeline_cfg = {
        "dataset_name": "sim",
        "raw_root": str(merge_root),
        "output_root": str(output_root),
        "seq_ids": seq_ids,
        "scene_id_by_seq": scene_id_by_seq,
    }
    t0 = time.time()
    result = PreparePipeline().run(pipeline_cfg)
    elapsed = time.time() - t0
    print(f"[22_prepare_paper_data] DONE: {elapsed:.1f}s")
    print(f"[22_prepare_paper_data] manifest: {output_root}/prepare_manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
