from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC))


def _load_script22_module():
    script_path = ROOT / "scripts" / "22_run_single_neural_candidate.py"
    spec = importlib.util.spec_from_file_location("resume_checkpoint_selection_utils", script_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Cannot load module from {script_path}: spec.loader is None")
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> dict[str, Any]:
    from liquidloc.common.tee_logger import print_dict
    print_dict({"path": str(path)}, "_load_json 入口参数")
    from liquidloc.common.io_utils import read_json

    return read_json(path)


def _resolve_train_report_path(candidate_root: Path, explicit_train_report: str | None) -> Path:
    from liquidloc.common.tee_logger import print_dict
    print_dict({"candidate_root": str(candidate_root), "explicit_train_report": explicit_train_report}, "_resolve_train_report_path 入口参数")
    if explicit_train_report:
        return Path(explicit_train_report).resolve()
    return (candidate_root / "train" / "reports" / "liquid_ekf_train_report.json").resolve()


def _resolve_selection_root(candidate_root: Path, explicit_selection_root: str | None) -> Path:
    from liquidloc.common.tee_logger import print_dict
    print_dict({"candidate_root": str(candidate_root), "explicit_selection_root": explicit_selection_root}, "_resolve_selection_root 入口参数")
    if explicit_selection_root:
        return Path(explicit_selection_root).resolve()
    return (candidate_root / "paper_checkpoint_selection").resolve()


_EXPORT_SCORE_RESCORE_TOP_K_DEFAULT = 8


def _select_export_score_top_k_candidates(
    train_report: dict[str, Any],
    *,
    top_k: int = _EXPORT_SCORE_RESCORE_TOP_K_DEFAULT,
) -> tuple[list[str], list[int], list[dict[str, Any]]]:
    """按 train_report 中的 export_score 排序取 top-K 候选 epoch。

    train_report 中 `epoch_candidate_paths` 与 `epoch_candidate_epochs` 是平行的两个列表。
    每个候选 checkpoint 的 epoch progress snapshot 写入了 `export_score` 字段。
    这里复用 `_safe_load_checkpoint_best_epoch` 的同款 torch.load 方法读取每个候选的
    export_score 并按升序（越低越好）取 top-K。

    返回 (top_k_paths, top_k_epochs, top_k_export_scores)。
    如果某个候选无法读取 export_score（缺失/不可解析），使用 train_report 的 best_export_score 兜底；
    若 best_export_score 也缺失则用 0.0，所有候选按 export_score 升序排序取前 top-K。
    """
    import torch

    candidate_paths = list(train_report.get("epoch_candidate_paths") or [])
    candidate_epochs = list(train_report.get("epoch_candidate_epochs") or [])
    best_checkpoint_path = str(train_report.get("checkpoint_path") or "").strip()
    if best_checkpoint_path and best_checkpoint_path not in candidate_paths:
        candidate_paths.append(best_checkpoint_path)
        best_epoch_value = train_report.get("best_epoch")
        if best_epoch_value is None:
            best_epoch_value = train_report.get("best_export_epoch") or 0
        candidate_epochs.append(int(best_epoch_value))

    padded_epochs: list[int] = []
    for index, path_value in enumerate(candidate_paths):
        if index < len(candidate_epochs):
            padded_epochs.append(int(candidate_epochs[index]))
        else:
            padded_epochs.append(0)
    candidate_epochs = padded_epochs

    scored_candidates: list[dict[str, Any]] = []
    fallback_export_score = float(train_report.get("best_export_score") or 0.0)
    for path_value, epoch_value in zip(candidate_paths, candidate_epochs):
        scored: dict[str, Any] = {
            "checkpoint_path": str(path_value),
            "epoch": int(epoch_value),
            "export_score": fallback_export_score,
            "fallback": True,
        }
        try:
            payload = torch.load(str(path_value), map_location="cpu", weights_only=True)
            if isinstance(payload, dict):
                raw_export = payload.get("export_score")
                if raw_export is None:
                    raw_export = payload.get("selection_score")
                if raw_export is not None and isinstance(raw_export, (int, float)):
                    scored["export_score"] = float(raw_export)
                    scored["fallback"] = False
        except Exception:
            scored["fallback"] = True
        scored_candidates.append(scored)

    scored_candidates.sort(key=lambda item: (item["export_score"], item["epoch"]))
    top_k = max(1, int(top_k))
    top_candidates = scored_candidates[:top_k]
    top_paths = [item["checkpoint_path"] for item in top_candidates]
    top_epochs = [item["epoch"] for item in top_candidates]
    return top_paths, top_epochs, top_candidates


def main(argv: list[str] | None = None) -> int:
    from liquidloc.common.tee_logger import print_dict
    print_dict({"argv": argv}, "main 入口参数")
    parser = argparse.ArgumentParser(description="Resume paper checkpoint selection for an existing neural candidate")
    parser.add_argument("--paper-run-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--model-name", default="liquid_ekf")
    parser.add_argument("--train-report", default=None)
    parser.add_argument("--selection-root", default=None)
    parser.add_argument(
        "--checkpoint-selection-mode",
        default="downstream_val_rescore",
        choices=("downstream_val_rescore", "trainer_validation_best", "export_score_downstream_rescored"),
    )
    args = parser.parse_args(argv)

    from liquidloc.common.tee_logger import print_args, print_dict
    from liquidloc.common.types import StageResult

    print_args(args, "23_resume_neural_checkpoint_selection")

    script22 = _load_script22_module()
    paper = script22._load_paper_module()

    paper_run_root = Path(args.paper_run_root).resolve()
    candidate_root = Path(args.candidate_root).resolve()
    train_report_path = _resolve_train_report_path(candidate_root, args.train_report)
    selection_root = _resolve_selection_root(candidate_root, args.selection_root)

    if not candidate_root.exists():
        raise FileNotFoundError(f"candidate root not found: {candidate_root}")
    if not train_report_path.exists():
        raise FileNotFoundError(f"train report not found: {train_report_path}")

    selection_root.mkdir(parents=True, exist_ok=True)

    prepare_root, raw_root, split_manifest, classical_search_audit = script22._resolve_runtime_context(paper_run_root)
    train_report = _load_json(train_report_path)
    estimator_cfgs = dict(classical_search_audit["selected_estimator_cfgs"])
    train_result = StageResult(
        stage_name="resume_checkpoint_selection",
        artifacts=[str(train_report_path)],
        metadata={"train_report": train_report},
    )

    print_dict(
        {
            "paper_run_root": str(paper_run_root),
            "candidate_root": str(candidate_root),
            "train_report_path": str(train_report_path),
            "selection_root": str(selection_root),
            "candidate_epoch_count": len(list(train_report.get("epoch_candidate_paths") or [])),
            "existing_selection_dirs": len([path for path in selection_root.iterdir() if path.is_dir()]),
            "checkpoint_selection_mode": str(args.checkpoint_selection_mode),
        },
        "checkpoint 复选上下文",
    )

    selection_mode = str(args.checkpoint_selection_mode)

    if selection_mode == "export_score_downstream_rescored":
        top_paths, top_epochs, top_candidates_meta = _select_export_score_top_k_candidates(
            train_report,
            top_k=_EXPORT_SCORE_RESCORE_TOP_K_DEFAULT,
        )
        print_dict(
            {
                "selection_mode": selection_mode,
                "top_k": len(top_paths),
                "top_k_paths": top_paths,
                "top_k_epochs": top_epochs,
                "top_k_export_scores": [
                    item["export_score"] for item in top_candidates_meta
                ],
                "fallback_flags": [item["fallback"] for item in top_candidates_meta],
            },
            "export_score_downstream_rescored top-K 候选",
        )
        # Override train_report.view_of candidates so downstream rescoring only
        # runs on top-K export-score candidates. Keep the original list intact
        # under `epoch_candidate_paths_full` for probe/audit.
        train_report = copy.deepcopy(train_report)
        full_paths = list(train_report.get("epoch_candidate_paths") or [])
        full_epochs = list(train_report.get("epoch_candidate_epochs") or [])
        train_report["epoch_candidate_paths_full"] = full_paths
        train_report["epoch_candidate_epochs_full"] = full_epochs
        train_report["epoch_candidate_paths"] = top_paths
        train_report["epoch_candidate_epochs"] = top_epochs
        train_result = StageResult(
            stage_name="resume_checkpoint_selection",
            artifacts=[str(train_report_path)],
            metadata={"train_report": train_report},
        )
        try:
            result = paper._select_final_neural_checkpoint(
                model_name=str(args.model_name),
                train_result=train_result,
                prepare_root=prepare_root,
                raw_root=raw_root,
                split_manifest=split_manifest,
                estimator_cfgs=estimator_cfgs,
                selection_root=selection_root,
                checkpoint_selection_mode="downstream_val_rescore",
                baseline_scoring_metrics_by_experiment=None,
            )
            selection_report = dict(result.get("selection_report") or {})
            selection_report["selection_mode"] = "export_score_downstream_rescored"
            selection_report["export_score_top_k"] = len(top_paths)
            selection_report["export_score_top_k_candidates"] = top_candidates_meta
            selection_report["fallback"] = False
            result["selection_report"] = selection_report
        except Exception as exc:
            print_dict(
                {
                    "selection_mode": selection_mode,
                    "fallback_reason": str(exc),
                },
                "export_score_downstream_rescored 回退到 trainer_validation_best",
            )
            train_result_fallback = StageResult(
                stage_name="resume_checkpoint_selection",
                artifacts=[str(train_report_path)],
                metadata={"train_report": dict(_load_json(train_report_path))},
            )
            result = paper._select_final_neural_checkpoint(
                model_name=str(args.model_name),
                train_result=train_result_fallback,
                prepare_root=prepare_root,
                raw_root=raw_root,
                split_manifest=split_manifest,
                estimator_cfgs=estimator_cfgs,
                selection_root=selection_root,
                checkpoint_selection_mode="trainer_validation_best",
                baseline_scoring_metrics_by_experiment=None,
            )
            selection_report = dict(result.get("selection_report") or {})
            selection_report["selection_mode"] = "export_score_downstream_rescored"
            selection_report["fallback"] = True
            selection_report["fallback_reason"] = str(exc)
            selection_report["export_score_top_k"] = len(top_paths)
            selection_report["export_score_top_k_candidates"] = top_candidates_meta
            result["selection_report"] = selection_report
    else:
        result = paper._select_final_neural_checkpoint(
            model_name=str(args.model_name),
            train_result=train_result,
            prepare_root=prepare_root,
            raw_root=raw_root,
            split_manifest=split_manifest,
            estimator_cfgs=estimator_cfgs,
            selection_root=selection_root,
            checkpoint_selection_mode=selection_mode,
            baseline_scoring_metrics_by_experiment=None,
        )

    selection_report = dict(result.get("selection_report") or {})
    mutated_train_report = dict(result["train_result"].metadata["train_report"])
    print_dict(
        {
            "selected_checkpoint_path": selection_report.get("selected_checkpoint_path"),
            "selected_best_epoch": selection_report.get("selected_best_epoch"),
            "candidate_count": selection_report.get("candidate_count"),
            "selection_report_path": selection_report.get("selection_report_path"),
            "paper_selected_checkpoint_path": mutated_train_report.get("paper_selected_checkpoint_path"),
            "paper_selected_best_epoch": mutated_train_report.get("paper_selected_best_epoch"),
        },
        "checkpoint 复选结果",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
