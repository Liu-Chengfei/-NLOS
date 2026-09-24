from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC))

_SUPPORTED_MODELS = ("lstm_ekf", "liquid_ekf")
_SUPPORTED_PROFILES = ("paper", "small_eval", "medium_eval")
_EXTRA_LIQUID_ROBUSTNESS_PROFILES = (
    {
        "name": "v13_cellgate_tail_async_bridge",
        "train.weight_decay": 2e-4,
        "network.pooling_logit_scale": 0.38,
        "network.cell_update_scale_floor": 0.66,
        "network.cell_update_scale_span": 0.34,
        "network.reliability_bias_init": 1.85,
        "network.bad_observation_floor": 0.12,
        "network.bad_observation_span": 0.88,
        "network.bad_observation_residual_full_scale": 0.36,
        "network.bad_observation_track_drop_full_scale": 28.0,
        "network.bad_observation_reproj_slope_full_scale": 1.20,
        "network.bad_observation_interaction_coeff": 1.05,
        "bridge_thresholds.robust_supplement_quality_threshold": 0.46,
        "bridge_thresholds.robust_supplement_alignment_threshold": 0.36,
        "bridge_thresholds.observation_risk_blend": 0.50,
        "bridge_thresholds.uwb_geometric_bias_full_scale": 0.34,
        "bridge_thresholds.async_gap_full_scale": 0.22,
    },
    {
        "name": "v13_cellgate_tail_async_hard",
        "train.weight_decay": 2.5e-4,
        "network.pooling_logit_scale": 0.45,
        "network.cell_update_scale_floor": 0.58,
        "network.cell_update_scale_span": 0.42,
        "network.reliability_bias_init": 1.55,
        "network.bad_observation_floor": 0.10,
        "network.bad_observation_span": 0.90,
        "network.bad_observation_residual_full_scale": 0.32,
        "network.bad_observation_track_drop_full_scale": 32.0,
        "network.bad_observation_reproj_slope_full_scale": 1.35,
        "network.bad_observation_interaction_coeff": 1.15,
        "bridge_thresholds.robust_supplement_quality_threshold": 0.43,
        "bridge_thresholds.robust_supplement_alignment_threshold": 0.34,
        "bridge_thresholds.observation_risk_blend": 0.56,
        "bridge_thresholds.uwb_geometric_bias_full_scale": 0.30,
        "bridge_thresholds.async_gap_full_scale": 0.18,
    },
    {
        "name": "v14_cellgate_tail_async_mid",
        "train.weight_decay": 1.5e-4,
        "network.pooling_logit_scale": 0.28,
        "network.cell_update_scale_floor": 0.76,
        "network.cell_update_scale_span": 0.24,
        "network.reliability_bias_init": 2.10,
        "network.bad_observation_floor": 0.14,
        "network.bad_observation_span": 0.86,
        "network.bad_observation_residual_full_scale": 0.40,
        "network.bad_observation_track_drop_full_scale": 24.0,
        "network.bad_observation_reproj_slope_full_scale": 1.00,
        "network.bad_observation_interaction_coeff": 1.00,
        "network.context_modulation_scale": 0.26,
        "network.filter_context_modulation_scale": 0.26,
        "bridge_thresholds.robust_supplement_quality_threshold": 0.50,
        "bridge_thresholds.robust_supplement_alignment_threshold": 0.40,
        "bridge_thresholds.observation_risk_blend": 0.42,
        "bridge_thresholds.uwb_geometric_bias_full_scale": 0.42,
        "bridge_thresholds.async_gap_full_scale": 0.24,
        "bridge_thresholds.async_uwb_scaling_coeff": 0.30,
        "bridge_thresholds.async_vio_scaling_coeff": 0.30,
    },
)


def _load_paper_module():
    script_path = ROOT / "scripts" / "20_run_paper_experiments.py"
    spec = importlib.util.spec_from_file_location("paper_run_single_candidate", script_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Cannot load module from {script_path}: spec.loader is None")
    spec.loader.exec_module(module)
    if hasattr(module, "_bootstrap_impl_namespace"):
        return SimpleNamespace(**module._bootstrap_impl_namespace())
    if hasattr(module, "_load_impl_namespace"):
        return SimpleNamespace(**module._load_impl_namespace())
    return module


def _load_json(path: Path) -> dict[str, Any]:
    from liquidloc.common.io_utils import read_json

    return read_json(path)


def _resolve_report_path(raw_path: str | Path | None, *, report_path: Path) -> Path | None:
    if raw_path is None:
        return None
    normalized = str(raw_path).strip()
    if not normalized:
        return None
    path = Path(normalized)
    if path.is_absolute():
        return path.resolve()
    return (report_path.parent / path).resolve()


def _resolve_runtime_raw_root(paper_run_root: Path, *, default_raw_root: Path) -> Path:
    final_report_path = paper_run_root / "paper_run_report.json"
    readiness_report_path = paper_run_root / "audits" / "data_readiness.json"
    if final_report_path.exists():
        final_report = _load_json(final_report_path)
        if isinstance(final_report, dict):
            resolved_readiness = _resolve_report_path(final_report.get("readiness_report"), report_path=final_report_path)
            if resolved_readiness is not None:
                readiness_report_path = resolved_readiness
    if not readiness_report_path.exists():
        return default_raw_root.resolve()
    readiness_report = _load_json(readiness_report_path)
    if not isinstance(readiness_report, dict):
        return default_raw_root.resolve()
    sim_report = readiness_report.get("sim")
    if not isinstance(sim_report, dict):
        return default_raw_root.resolve()
    raw_root_value = str(sim_report.get("raw_root") or "").strip()
    if not raw_root_value:
        return default_raw_root.resolve()
    resolved_raw_root = Path(raw_root_value)
    if not resolved_raw_root.is_absolute():
        resolved_raw_root = (readiness_report_path.parent / resolved_raw_root).resolve()
    return resolved_raw_root.resolve()


def _resolve_runtime_surface(module, *, paper_run_root: Path, requested_profile: str) -> dict[str, Any]:
    default_surface = module._resolve_neural_search_surface(requested_profile)
    default_profiles = list(default_surface.get("liquid_robustness_profiles") or [])
    default_known_names = {str(profile.get("name") or "") for profile in default_profiles}
    for profile in _EXTRA_LIQUID_ROBUSTNESS_PROFILES:
        profile_name = str(profile.get("name") or "")
        if profile_name and profile_name not in default_known_names:
            default_profiles.append(dict(profile))
            default_known_names.add(profile_name)
    default_surface["liquid_robustness_profiles"] = default_profiles
    final_report_path = paper_run_root / "paper_run_report.json"
    if not final_report_path.exists():
        return default_surface
    final_report = _load_json(final_report_path)
    if not isinstance(final_report, dict):
        return default_surface
    runtime_surface = final_report.get("neural_search_surface")
    if not isinstance(runtime_surface, dict):
        return default_surface
    merged_surface = dict(default_surface)
    merged_surface.update(runtime_surface)
    # Keep the currently loaded code's paper-selection contract authoritative.
    # Older paper_run_report.json files may still carry trainer-best defaults.
    if str(requested_profile).strip().lower() == "paper":
        for key in ("checkpoint_selection_mode", "epoch_candidate_stride"):
            merged_surface[key] = default_surface.get(key)
    default_liquid_profiles = list(default_surface.get("liquid_robustness_profiles") or [])
    runtime_liquid_profiles = runtime_surface.get("liquid_robustness_profiles")
    if isinstance(runtime_liquid_profiles, list):
        if len(runtime_liquid_profiles) >= len(default_liquid_profiles):
            merged_surface["liquid_robustness_profiles"] = list(runtime_liquid_profiles)
        else:
            merged_surface["liquid_robustness_profiles"] = default_liquid_profiles
    profiles = list(merged_surface.get("liquid_robustness_profiles") or [])
    known_names = {str(profile.get("name") or "") for profile in profiles}
    for profile in _EXTRA_LIQUID_ROBUSTNESS_PROFILES:
        profile_name = str(profile.get("name") or "")
        if profile_name and profile_name not in known_names:
            profiles.append(dict(profile))
            known_names.add(profile_name)
    merged_surface["liquid_robustness_profiles"] = profiles
    if str(requested_profile).strip().lower() == "paper":
        merged_surface["checkpoint_selection_mode"] = str(
            merged_surface.get("checkpoint_selection_mode") or "downstream_val_rescore"
        )
        if merged_surface.get("epoch_candidate_stride") is None:
            merged_surface["epoch_candidate_stride"] = 10
    return merged_surface


def _default_output_root(model_name: str, profile_name: str) -> Path:
    return ROOT / "outputs" / f"{model_name}_single_candidate_{profile_name}"


def _resolve_runtime_context(paper_run_root: Path) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    prepare_root = paper_run_root / "prepare" / "sim"
    raw_root = _resolve_runtime_raw_root(paper_run_root, default_raw_root=ROOT / "data" / "raw" / "sim")
    split_manifest_path = paper_run_root / "splits" / "split_manifest.json"
    classical_search_audit_path = paper_run_root / "classical_search" / "search_audit.json"
    if not prepare_root.exists():
        raise FileNotFoundError(f"prepare root not found: {prepare_root}")
    if not raw_root.exists():
        raise FileNotFoundError(f"sim raw root not found: {raw_root}")
    if not split_manifest_path.exists():
        raise FileNotFoundError(f"split manifest not found: {split_manifest_path}")
    if not classical_search_audit_path.exists():
        raise FileNotFoundError(f"classical search audit not found: {classical_search_audit_path}")
    split_manifest = _load_json(split_manifest_path)
    classical_search_audit = _load_json(classical_search_audit_path)
    return prepare_root, raw_root, split_manifest, classical_search_audit


def _resolve_baseline_scoring_metrics_by_experiment(classical_search_audit: dict[str, Any]) -> dict[str, Any]:
    direct_metrics = classical_search_audit.get("selected_ekf_metrics_by_experiment")
    if isinstance(direct_metrics, dict) and direct_metrics:
        return dict(direct_metrics)
    ekf_audit = classical_search_audit.get("ekf")
    if isinstance(ekf_audit, dict):
        selected_metrics = ekf_audit.get("selected_metrics")
        if isinstance(selected_metrics, dict) and selected_metrics:
            return dict(selected_metrics)
    raise ValueError(
        "paper classical search audit is missing EKF baseline scoring metrics; "
        "expected selected_ekf_metrics_by_experiment or ekf.selected_metrics"
    )


def _select_liquid_profile(surface: dict[str, Any], profile_name: str | None) -> dict[str, Any]:
    profiles = list(surface.get("liquid_robustness_profiles") or [])
    if not profiles:
        raise RuntimeError("liquid_ekf single-candidate run requires at least one robustness profile")
    if profile_name is None:
        return dict(profiles[0])
    for profile in profiles:
        if str(profile.get("name") or "") == profile_name:
            return dict(profile)
    available = [str(profile.get("name") or "") for profile in profiles]
    raise ValueError(f"unknown liquid robustness profile '{profile_name}'; available: {available}")


def _resolve_candidate_overrides(
    module,
    *,
    model_name: str,
    surface: dict[str, Any],
    window_size: int | None,
    hidden_dim: int | None,
    lr: float | None,
    weight_decay: float | None,
    liquid_profile_name: str | None,
) -> dict[str, Any]:
    neural_grid = dict(surface["neural_grid"][model_name])
    overrides: dict[str, Any] = {
        "window.size": int(window_size if window_size is not None else neural_grid["window"][0]),
        "network.hidden_dim": int(hidden_dim if hidden_dim is not None else neural_grid["hidden_dim"][0]),
        "train.lr": float(lr if lr is not None else neural_grid["lr"][0]),
    }
    if model_name == "lstm_ekf":
        default_weight_decay = float(weight_decay if weight_decay is not None else neural_grid["weight_decay"][0])
        overrides["train.weight_decay"] = default_weight_decay
        return overrides
    selected_profile = _select_liquid_profile(surface, liquid_profile_name)
    for key, value in selected_profile.items():
        if key == "name":
            continue
        overrides[str(key)] = value
    if weight_decay is not None:
        overrides["train.weight_decay"] = float(weight_decay)
    return overrides


def _resolve_budget(surface: dict[str, Any], model_name: str) -> tuple[int, int, int]:
    budget = dict(surface["training_budget"])
    epoch_key = "liquid_epochs" if model_name == "liquid_ekf" else "lstm_epochs"
    return int(budget[epoch_key]), int(budget["batch_size"]), int(budget["eval_batch_size"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one paper-grade neural candidate end to end")
    parser.add_argument("--paper-run-root", required=True)
    parser.add_argument("--model-name", choices=_SUPPORTED_MODELS, default="liquid_ekf")
    parser.add_argument("--profile", choices=_SUPPORTED_PROFILES, default="paper")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--liquid-robustness-profile", default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)

    from liquidloc.common.tee_logger import print_args, print_dict
    print_args(args, "22_run_single_neural_candidate")

    print("[22_single_candidate] 开始 | model_name=" + str(args.model_name) + " profile=" + str(args.profile) + " seed=" + str(args.seed), flush=True)

    paper_run_root = Path(args.paper_run_root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else _default_output_root(args.model_name, args.profile)
    output_root.mkdir(parents=True, exist_ok=True)
    print("[22_single_candidate] 输出目录就绪 | 正在加载 paper 模块", flush=True)

    module = _load_paper_module()
    surface = _resolve_runtime_surface(module, paper_run_root=paper_run_root, requested_profile=str(args.profile))
    print_dict(surface, "神经搜索网格 (surface)")
    overrides = _resolve_candidate_overrides(
        module,
        model_name=args.model_name,
        surface=surface,
        window_size=args.window_size,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        liquid_profile_name=args.liquid_robustness_profile,
    )
    print_dict(overrides, "候选覆盖参数 (overrides)")
    epochs, batch_size, eval_batch_size = _resolve_budget(surface, args.model_name)
    prepare_root, raw_root, split_manifest, classical_search_audit = _resolve_runtime_context(paper_run_root)
    print_dict({"epochs": epochs, "batch_size": batch_size, "eval_batch_size": eval_batch_size, "prepare_root": str(prepare_root), "raw_root": str(raw_root)}, "训练预算与路径")
    estimator_cfgs = dict(classical_search_audit["selected_estimator_cfgs"])
    baseline_scoring_metrics_by_experiment = _resolve_baseline_scoring_metrics_by_experiment(classical_search_audit)
    print_dict(baseline_scoring_metrics_by_experiment, "paper_ekf_baseline_metrics_by_experiment")
    print_dict(estimator_cfgs, "估计器配置 (estimator_cfgs)")
    print("[22_single_candidate] 正在运行神经搜索候选 | model=" + str(args.model_name) + " epochs=" + str(epochs), flush=True)

    result = module._run_neural_search_candidate(
        model_name=args.model_name,
        overrides=overrides,
        seed=int(args.seed),
        prepare_root=prepare_root,
        raw_root=raw_root,
        split_manifest=split_manifest,
        search_root=output_root / "search",
        estimator_cfgs=estimator_cfgs,
        device=str(args.device),
        epochs=epochs,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        baseline_scoring_metrics_by_experiment=baseline_scoring_metrics_by_experiment,
        epoch_candidate_stride=surface.get("epoch_candidate_stride"),
        checkpoint_selection_mode=str(surface.get("checkpoint_selection_mode") or "downstream_val_rescore"),
        allow_cache=False,
    )

    train_report = result.get("train_report") or result.get("metadata", {}).get("train_report") or {}
    checkpoint_selection = result.get("checkpoint_selection") or {}
    summary = {
        "paper_run_root": str(paper_run_root),
        "raw_root": str(raw_root),
        "profile": str(surface.get("profile_name") or args.profile),
        "requested_profile": str(args.profile),
        "model_name": str(args.model_name),
        "seed": int(args.seed),
        "device": str(args.device),
        "resolved_budget": {"epochs": epochs, "batch_size": batch_size, "eval_batch_size": eval_batch_size},
        "overrides": dict(overrides),
        "baseline_scoring_metrics_by_experiment": dict(baseline_scoring_metrics_by_experiment),
        "candidate_signature": str(result["signature"]),
        "score_vector": list(result["score_vector"]),
        "train_report": dict(train_report),
        "checkpoint_selection": dict(checkpoint_selection),
        "scoring_runs": list(result["scoring_runs"]),
    }
    summary_path = output_root / "single_candidate_summary.json"
    print("[22_single_candidate] 正在写入汇总 | path=" + str(summary_path), flush=True)
    from liquidloc.common.io_utils import dumps_json_text

    summary_path.write_text(dumps_json_text(summary), encoding="utf-8")
    print(dumps_json_text({"status": "ok", "summary_path": str(summary_path), "candidate_signature": result["signature"]}, indent=None))
    print("[22_single_candidate] 完成 | 返回码=0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
