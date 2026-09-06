from __future__ import annotations

import builtins
from contextlib import contextmanager
from functools import wraps
import inspect
from pathlib import Path
import sys
from typing import Any
import copy

_SCRIPT_PATH = Path(__file__).resolve()
_IMPL_SOURCE_PATH = _SCRIPT_PATH.with_suffix(".bak.py")


def _safe_print(*args: Any, **kwargs: Any) -> None:
    # 关键修复 (2026-07-31 paper-run EKF grid 后失败): 之前遇到 print 抛 ValueError
    # (I/O operation on closed file) 时只 catch OSError, 但 ValueError 不在 OSError
    # 子类里. 原因是 6 worker 在 ThreadPoolExecutor 里把 sys.stdout / sys.stderr 重
    # 定向到 per-worker log 文件, 主进程退出时某些句柄已被 ThreadPoolExecutor 或
    # OS 关掉, 此时再 print 会抛 ValueError. 这里把 Exception 都吞掉, 保证 wrapper
    # print 失败不会让 candidate 返回值丢失 (本来 candidate_report.json 已写好).
    try:
        kwargs.setdefault("file", sys.stderr)
        print(*args, **kwargs)
    except Exception:
        try:
            import sys as _sys
            if _sys.stderr is not None and not getattr(_sys.stderr, "closed", False):
                _sys.stderr.write(str(args[0]) if args else "")
                _sys.stderr.write("\n")
        except Exception:
            pass


def _looks_like_json_payload(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    return (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("[") and stripped.endswith("]")
    )


class _ResumeStageResult:
    __slots__ = ("stage_name", "artifacts", "metadata")

    def __init__(self, stage_name: str, artifacts: list[str] | None = None, metadata: dict[str, Any] | None = None):
        self.stage_name = stage_name
        self.artifacts = [] if artifacts is None else artifacts
        self.metadata = {} if metadata is None else metadata


def _strip_broken_docstrings(source: str) -> str:
    lines = source.splitlines(keepends=True)
    cleaned: list[str] = []
    skip_multiline = False
    previous_meaningful = ""

    for index, line in enumerate(lines):
        stripped = line.strip()
        if skip_multiline:
            if '"""' in line:
                skip_multiline = False
            continue

        if index == 0 and stripped.startswith('"""') and stripped.count('"""') == 1:
            # The backup file keeps a corrupted header docstring opener on line 1.
            continue

        if stripped.startswith('"""'):
            quote_count = line.count('"""')
            if quote_count >= 2:
                continue
            if previous_meaningful.endswith(":"):
                skip_multiline = True
                continue
            continue

        cleaned.append(line)
        if stripped:
            previous_meaningful = stripped

    return "".join(cleaned)


def _sanitize_impl_source(source: str) -> str:
    # The backup file is polluted at the front and may contain non-printable
    # control bytes inside comments/docstrings. Keep only the executable body.
    source = source.replace("\ufeff", "")
    source = source.replace("\x00", "")
    source = source.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    source = _strip_broken_docstrings(source)
    marker = "from __future__ import annotations"
    marker_index = source.find(marker)
    if marker_index > 0:
        source = source[marker_index:]
    allowed = {"\n", "\r", "\t"}
    source = "".join(ch for ch in source if ch.isprintable() or ch in allowed)
    return source


def _load_impl_namespace() -> dict[str, Any]:
    if not _IMPL_SOURCE_PATH.exists():
        raise FileNotFoundError(f"paper-run implementation not found: {_IMPL_SOURCE_PATH}")
    source = _IMPL_SOURCE_PATH.read_text(encoding="utf-8-sig", errors="replace")
    code = compile(_sanitize_impl_source(source), str(_IMPL_SOURCE_PATH), "exec")
    namespace: dict[str, Any] = {
        "__name__": "_paper_run_impl",
        "__file__": str(_IMPL_SOURCE_PATH),
        "__package__": None,
        "__cached__": None,
    }
    exec(code, namespace)
    return namespace



@contextmanager
def _temporary_globals(namespace: dict[str, Any], updates: dict[str, Any]):
    sentinel = object()
    previous = {key: namespace.get(key, sentinel) for key in updates}
    namespace.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is sentinel:
                namespace.pop(key, None)
            else:
                namespace[key] = value


def _sync_exported_globals_into_namespace(namespace: dict[str, Any]) -> None:
    wrapper_globals = globals()
    for key in list(namespace.keys()):
        if key.startswith("__") or key in _WRAPPER_RESERVED_NAMES:
            continue
        if key in wrapper_globals:
            value = wrapper_globals[key]
            proxy_key = getattr(value, "__namespace_proxy_key__", None)
            if proxy_key == key:
                namespace[key] = getattr(value, "__namespace_proxy_original__", namespace.get(key))
            else:
                namespace[key] = value


def _build_namespace_callable_proxy(namespace: dict[str, Any], key: str, target: Any):
    @wraps(target)
    def _proxy(*args: Any, **kwargs: Any):
        _sync_exported_globals_into_namespace(namespace)
        current_target = namespace.get(key, target)
        if getattr(current_target, "__namespace_proxy_key__", None) == key:
            current_target = getattr(current_target, "__namespace_proxy_original__", target)
            namespace[key] = current_target
        return current_target(*args, **kwargs)

    _proxy.__namespace_proxy_key__ = key
    _proxy.__namespace_proxy_original__ = target
    return _proxy


def _unwrap_namespace_proxy(key: str, value: Any) -> Any:
    if getattr(value, "__namespace_proxy_key__", None) == key:
        return getattr(value, "__namespace_proxy_original__", value)
    return value


def _normalize_scene_id_set(scene_ids: Any) -> set[str]:
    normalized: set[str] = set()
    for raw_scene_id in scene_ids or ():
        scene_id = str(raw_scene_id or "").strip()
        if scene_id:
            normalized.add(scene_id)
    return normalized


def _extract_scene_ids_from_payload(payload: Any) -> set[str]:
    scene_ids: set[str] = set()
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict):
                scene_id = str(row.get("scene_id") or "").strip()
                if scene_id:
                    scene_ids.add(scene_id)
    elif isinstance(payload, dict):
        for key in ("main_cases", "failure_cases", "boundary_cases", "scene_cases", "prediction_bundles"):
            if key in payload:
                scene_ids.update(_extract_scene_ids_from_payload(payload.get(key)))
    return scene_ids


def _expected_scene_ids_for_config(namespace: dict[str, Any], config_name: str) -> set[str]:
    sample_scenes_cached = namespace.get("_sample_scenes_cached")
    if not callable(sample_scenes_cached):
        return set()
    try:
        scene_tasks = sample_scenes_cached(config_name)
    except Exception:
        return set()
    return _normalize_scene_id_set(
        task.get("scene_id")
        for task in scene_tasks
        if isinstance(task, dict)
    )


def _normalize_checkpoint_selection_report(
    selection_report: dict[str, Any] | None,
    *,
    default_selection_mode: str,
    default_candidate_reports: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(selection_report or {}))
    normalized["selection_mode"] = str(
        normalized.get("selection_mode") or default_selection_mode
    ).strip()

    candidate_reports = normalized.get("candidate_reports")
    if not isinstance(candidate_reports, list):
        candidate_reports = copy.deepcopy(list(default_candidate_reports or []))
    else:
        candidate_reports = copy.deepcopy(candidate_reports)
    normalized["candidate_reports"] = candidate_reports

    candidate_count = normalized.get("candidate_count")
    if candidate_count is None:
        candidate_count = len(candidate_reports)
    normalized["candidate_count"] = int(candidate_count)
    return normalized


def _wrap_candidate_functions(namespace: dict[str, Any]) -> None:
    original_classical = namespace["_run_classical_search_candidate"]
    original_signature = namespace["_search_candidate_signature"]

    def _unique_override_signature(model_name: str, overrides: dict[str, Any], seed: int) -> str:
        # Keep the original readable prefix, but make the full override payload
        # part of the cache key so distinct neural grid points cannot collide.
        import hashlib
        import json

        prefix = str(original_signature(model_name, overrides, seed))
        payload = {
            "model_name": str(model_name),
            "seed": int(seed),
            "overrides": sorted((str(key), value) for key, value in dict(overrides).items()),
        }
        digest = hashlib.sha1(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()[:16]
        return f"{prefix}__o{digest}"

    def _maybe_resume_classical_candidate(
        *,
        method_name: str,
        overrides: dict[str, Any],
        prepare_root: Any,
        raw_root: Any,
        search_root: Any,
        split_manifest: dict[str, Any],
        baseline_ekf_cfg: dict[str, Any] | None,
        baseline_scoring_metrics_by_experiment: dict[str, dict[str, float]] | None,
    ) -> dict[str, Any] | None:
        include_safe = method_name != "ekf"
        scoring_config_names = namespace["_search_scoring_config_names"](method_name, include_safe=include_safe)
        candidate_signature = namespace["_classical_candidate_signature"](method_name, overrides)
        cache_context = {
            "method_name": method_name,
            "overrides": dict(overrides),
            "prepare_root": str(Path(prepare_root).resolve()),
            "raw_root": str(Path(raw_root).resolve()),
            "val_ids": list(split_manifest["val_ids"]),
            "baseline_ekf_cfg": baseline_ekf_cfg,
            "baseline_scoring_metrics_by_experiment": namespace["_normalize_baseline_scoring_metrics_for_cache"](
                baseline_scoring_metrics_by_experiment
            ),
            "search_experiments": list(scoring_config_names),
            "implementation_fingerprint": namespace["_resolve_search_implementation_fingerprint"](
                search_kind="classical",
                implementation_name=method_name,
                experiment_config_names=tuple(scoring_config_names),
            ),
        }
        search_root_path = Path(search_root)
        current_candidate_root = search_root_path / method_name / candidate_signature
        current_report_path = current_candidate_root / "candidate_report.json"
        candidate_roots = [current_candidate_root]

        # Allow resume from the latest completed sibling run when the current
        # output root is a fresh retry of the same paper scope.
        try:
            current_output_root = search_root_path.parents[1]
        except IndexError:
            current_output_root = None
        if current_output_root is not None:
            outputs_root = current_output_root.parent
            scope_prefix = current_output_root.name.split("_full_run", 1)[0]
            if outputs_root.is_dir():
                for sibling_output_root in sorted(outputs_root.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True):
                    if not sibling_output_root.is_dir() or sibling_output_root == current_output_root:
                        continue
                    if scope_prefix and not sibling_output_root.name.startswith(scope_prefix):
                        continue
                    candidate_roots.append(
                        sibling_output_root / "classical_search" / "search" / method_name / candidate_signature
                    )

        seen_candidate_roots: set[Path] = set()

        def _matches_cached_classical_candidate(cached_context: dict[str, Any]) -> bool:
            if cached_context.get("method_name") != cache_context.get("method_name"):
                return False
            if cached_context.get("overrides") != cache_context.get("overrides"):
                return False
            if cached_context.get("raw_root") != cache_context.get("raw_root"):
                return False
            if cached_context.get("val_ids") != cache_context.get("val_ids"):
                return False
            if cached_context.get("baseline_ekf_cfg") != cache_context.get("baseline_ekf_cfg"):
                return False
            if cached_context.get("baseline_scoring_metrics_by_experiment") != cache_context.get("baseline_scoring_metrics_by_experiment"):
                return False
            if cached_context.get("search_experiments") != cache_context.get("search_experiments"):
                return False
            if cached_context.get("implementation_fingerprint") != cache_context.get("implementation_fingerprint"):
                return False
            cached_copy = dict(cached_context)
            current_copy = dict(cache_context)
            cached_copy.pop("prepare_root", None)
            current_copy.pop("prepare_root", None)
            return cached_copy == current_copy

        def _rewrite_cached_classical_candidate(
            cached_report: dict[str, Any],
            *,
            source_report_path: Path,
        ) -> dict[str, Any]:
            rewritten = copy.deepcopy(cached_report)
            rewritten_cache_context = dict(rewritten.get("cache_context") or {})
            rewritten_cache_context["prepare_root"] = cache_context["prepare_root"]
            rewritten["cache_context"] = rewritten_cache_context

            source_candidate_root = source_report_path.parent
            source_search_root = source_candidate_root.parent.parent.parent
            source_search_root_text = str(source_search_root.resolve())
            current_search_root_text = str(search_root_path.resolve())

            for scoring_run in list(rewritten.get("scoring_runs") or []):
                if not isinstance(scoring_run, dict):
                    continue
                output_root_text = str(scoring_run.get("output_root") or "")
                if output_root_text.startswith(source_search_root_text):
                    scoring_run["output_root"] = current_search_root_text + output_root_text[len(source_search_root_text):]

            return rewritten

        for candidate_root in candidate_roots:
            if candidate_root in seen_candidate_roots:
                continue
            seen_candidate_roots.add(candidate_root)
            report_path = candidate_root / "candidate_report.json"
            if not report_path.is_file():
                continue
            cached_report = dict(namespace["_read_json"](report_path) or {})
            cached_context = dict(cached_report.get("cache_context") or {})
            if not _matches_cached_classical_candidate(cached_context):
                continue
            scoring_runs = list(cached_report.get("scoring_runs") or [])
            score_vector = list(cached_report.get("score_vector") or [])
            if not scoring_runs or not score_vector:
                continue
            resumed = _rewrite_cached_classical_candidate(cached_report, source_report_path=report_path)
            namespace["_write_json"](current_report_path, resumed)
            return resumed
        return None

    def _maybe_resume_neural_candidate(
        *,
        model_name: str,
        seed: int,
        search_root: Any,
        overrides: dict[str, Any],
    ) -> dict[str, Any] | None:
        requested_signature = namespace["_search_candidate_signature"](model_name, overrides, seed)
        model_root = Path(search_root) / model_name
        candidate_root = model_root / requested_signature
        if not candidate_root.is_dir():
            return None

        train_root = candidate_root / "train"
        scoring_root = candidate_root / "scoring_runs"
        train_report_path = train_root / "reports" / f"{model_name}_train_report.json"
        selection_report_path = candidate_root / "paper_checkpoint_selection" / "final_checkpoint_selection.json"
        prediction_index_path = scoring_root / "e9_dual_degradation" / "core" / "audits" / "prediction_index.json"
        eval_audit_path = scoring_root / "e9_dual_degradation" / "eval" / "audits" / "eval_audit.json"
        statistics_path = scoring_root / "e9_dual_degradation" / "eval" / "statistics" / "statistics_table.json"
        required_paths = (
            train_report_path,
            selection_report_path,
            prediction_index_path,
            eval_audit_path,
            statistics_path,
        )
        if not all(path.is_file() for path in required_paths):
            return None

        train_report = dict(namespace["_read_json"](train_report_path) or {})
        selection_report = _normalize_checkpoint_selection_report(
            namespace["_read_json"](selection_report_path) or {},
            default_selection_mode="downstream_val_rescore",
        )
        prediction_index_payload = list(namespace["_read_json"](prediction_index_path) or [])
        eval_audit = dict(namespace["_read_json"](eval_audit_path) or {})
        statistics_payload = dict(namespace["_read_json"](statistics_path) or {})
        selection_mode = str(selection_report.get("selection_mode") or "").strip().lower()
        candidate_count = int(selection_report.get("candidate_count") or 0)
        has_legacy_selected_checkpoint = bool(
            str(selection_report.get("selected_checkpoint_path") or "").strip()
        ) and "selection_mode" not in selection_report and "candidate_count" not in selection_report
        if selection_mode != "downstream_val_rescore" or (
            candidate_count <= 0 and not has_legacy_selected_checkpoint
        ):
            return None

        expected_scene_ids = _expected_scene_ids_for_config(namespace, "e9_dual_degradation.yaml")
        actual_scene_ids = _extract_scene_ids_from_payload(prediction_index_payload)
        if expected_scene_ids and actual_scene_ids != expected_scene_ids:
            return None

        if dict(train_report.get("overrides") or {}) != dict(overrides):
            return None

        main_table = list(statistics_payload.get("main_table") or [])
        main_row = next((row for row in main_table if str(row.get("method_name")) == model_name), None)
        if not isinstance(main_row, dict):
            return None

        metrics = {
            "p95": float(main_row["mean_p95"]),
            "hard_p95": float(main_row["mean_p95"]),
            "failure_rate": float(main_row["mean_failure_rate"]),
            "hard_failure_rate": float(main_row["mean_failure_rate"]),
            "rmse": float(main_row["mean_rmse"]),
            "mae": float(main_row["mean_mae"]),
        }
        score_vector = list(
            namespace["_score_candidate_lexicographically"](
                metrics_by_experiment={"e9_dual_degradation": metrics}
            )
        )
        selected_checkpoint_path = str(
            selection_report.get("selected_checkpoint_path")
            or train_report.get("checkpoint_path")
            or ""
        ).strip()
        if not selected_checkpoint_path:
            return None

        resumed_train_report = copy.deepcopy(train_report)
        resumed_train_report["paper_selected_checkpoint_path"] = selected_checkpoint_path
        resumed_train_report["paper_selected_best_epoch"] = int(
            selection_report.get("selected_best_epoch") or resumed_train_report.get("best_epoch") or 0
        )
        resumed_train_report["paper_selected_score_vector"] = list(score_vector)
        resumed_train_report["paper_selected_metrics"] = dict(metrics)
        resumed_train_report["paper_checkpoint_selection_report"] = str(selection_report_path)

        return {
            "model_name": model_name,
            "signature": requested_signature,
            "seed": int(seed),
            "overrides": dict(overrides),
            "cache_context": {},
            "train_report": resumed_train_report,
            "checkpoint_selection": {
                "selected_checkpoint_path": selected_checkpoint_path,
                "selected_best_epoch": int(selection_report.get("selected_best_epoch") or 0),
                "selected_score_vector": list(score_vector),
                "candidate_count": int(selection_report.get("candidate_count") or 0),
                "selection_mode": str(selection_report.get("selection_mode") or ""),
                "candidate_reports": copy.deepcopy(list(selection_report.get("candidate_reports") or [])),
                "selection_report_path": str(selection_report_path),
            },
            "scoring_runs": [
                {
                    "experiment_id": "e9_dual_degradation",
                    "output_root": str(scoring_root / "e9_dual_degradation"),
                    "metrics": metrics,
                    "best_method": str(eval_audit.get("best_method") or model_name),
                    "best_method_by_priority": str(eval_audit.get("best_method_by_priority") or model_name),
                    "prediction_index": list(prediction_index_payload),
                }
            ],
            "score_vector": list(score_vector),
        }

    def patched_classical(*, method_name: str, **kwargs: Any):
        _sync_exported_globals_into_namespace(namespace)
        _safe_print(f"[paper-run] 经典搜索候选开始: {method_name}", flush=True)
        resumed = _maybe_resume_classical_candidate(
            method_name=method_name,
            overrides=dict(kwargs["overrides"]),
            prepare_root=kwargs["prepare_root"],
            raw_root=kwargs["raw_root"],
            search_root=kwargs["search_root"],
            split_manifest=dict(kwargs["split_manifest"]),
            baseline_ekf_cfg=kwargs.get("baseline_ekf_cfg"),
            baseline_scoring_metrics_by_experiment=kwargs.get("baseline_scoring_metrics_by_experiment"),
        )
        if resumed is not None:
            _safe_print(f"[paper-run] 经典搜索候选从缓存恢复: {method_name}", flush=True)
            return resumed
        updates = {
            "method_name": method_name,
            "baseline_method_name": "ekf" if method_name != "ekf" else method_name,
            "include_safe": method_name != "ekf",
        }
        with _temporary_globals(namespace, updates):
            result = original_classical(method_name=method_name, **kwargs)
            _safe_print(f"[paper-run] 经典搜索候选完成: {method_name}", flush=True)
            return result

    def _build_compatible_neural_candidate_report(
        *,
        model_name: str,
        overrides: dict[str, Any],
        seed: int,
        prepare_root: Any,
        raw_root: Any,
        split_manifest: dict[str, Any],
        search_root: Any,
        estimator_cfgs: dict[str, dict[str, Any]],
        device: str,
        epochs: int,
        batch_size: int,
        eval_batch_size: int,
        baseline_scoring_metrics_by_experiment: dict[str, dict[str, float]] | None = None,
        epoch_candidate_stride: int | None = None,
        checkpoint_selection_mode: str = "downstream_val_rescore",
        allow_cache: bool = False,
    ) -> dict[str, Any]:
        search_scoring_config_names = namespace["_search_scoring_config_names"]
        load_cached_search_report = namespace["_load_cached_search_report"]
        normalize_baseline_cache = namespace["_normalize_baseline_scoring_metrics_for_cache"]
        resolve_search_impl_fingerprint = namespace["_resolve_search_implementation_fingerprint"]
        run_frontend_training = namespace["_run_frontend_training"]
        select_final_neural_checkpoint = namespace["_select_final_neural_checkpoint"]
        run_scoring_experiment = namespace["_run_scoring_experiment"]
        score_candidate_lexicographically = namespace["_score_candidate_lexicographically"]
        inject_self_baseline_safe_metrics = namespace["_inject_self_baseline_safe_metrics"]
        write_json = namespace["_write_json"]
        if baseline_scoring_metrics_by_experiment:
            full_scoring_config_names = tuple(
                list(namespace["_SEARCH_SCORING_EXPERIMENTS"]) + [namespace["_SAFE_SCORING_EXPERIMENT"]]
            )
            scoring_config_names = tuple(
                config_name
                for config_name in full_scoring_config_names
                if Path(config_name).stem in baseline_scoring_metrics_by_experiment
            ) or tuple(search_scoring_config_names(model_name))
        else:
            scoring_config_names = tuple(search_scoring_config_names(model_name))
        candidate_signature = namespace["_search_candidate_signature"](model_name, overrides, seed)
        candidate_root = Path(search_root) / model_name / candidate_signature
        report_path = candidate_root / "candidate_report.json"
        cache_context = {
            "model_name": model_name,
            "seed": int(seed),
            "overrides": dict(overrides),
            "prepare_root": str(Path(prepare_root).resolve()),
            "raw_root": str(Path(raw_root).resolve()),
            "train_ids": list(split_manifest["train_ids"]),
            "val_ids": list(split_manifest["val_ids"]),
            "estimator_cfgs": estimator_cfgs,
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "eval_batch_size": int(eval_batch_size),
            "search_experiments": list(scoring_config_names),
            "baseline_scoring_metrics_by_experiment": normalize_baseline_cache(
                baseline_scoring_metrics_by_experiment
            ),
            "checkpoint_selection_mode": str(checkpoint_selection_mode),
            "epoch_candidate_stride": int(epoch_candidate_stride) if epoch_candidate_stride is not None else None,
            "implementation_fingerprint": resolve_search_impl_fingerprint(
                search_kind="neural",
                implementation_name=model_name,
                experiment_config_names=scoring_config_names,
            ),
        }
        cached_report = (
            load_cached_search_report(report_path, cache_context=cache_context)
            if allow_cache
            else None
        )
        if cached_report is not None:
            return cached_report

        train_overrides = dict(overrides)
        train_overrides["train.seed"] = int(seed)
        if checkpoint_selection_mode == "downstream_val_rescore":
            train_overrides["train.save_epoch_candidates"] = True
            if epoch_candidate_stride is not None:
                train_overrides["train.epoch_candidate_stride"] = int(epoch_candidate_stride)

        train_result = run_frontend_training(
            model_name=model_name,
            prepare_root=prepare_root,
            raw_root=raw_root,
            split_manifest=split_manifest,
            output_root=candidate_root / "train",
            estimator_cfg=copy.deepcopy(estimator_cfgs["ekf"]),
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            model_overrides=train_overrides,
        )
        checkpoint_selection = select_final_neural_checkpoint(
            model_name=model_name,
            train_result=train_result,
            prepare_root=prepare_root,
            raw_root=raw_root,
            split_manifest=split_manifest,
            estimator_cfgs=estimator_cfgs,
            selection_root=candidate_root / "paper_checkpoint_selection",
            checkpoint_selection_mode=checkpoint_selection_mode,
            baseline_scoring_metrics_by_experiment=baseline_scoring_metrics_by_experiment,
        )
        train_result = checkpoint_selection["train_result"]
        train_metadata = dict(getattr(train_result, "metadata", {}) or {})
        train_report = copy.deepcopy(dict(train_metadata.get("train_report") or {}))
        sample_report = copy.deepcopy(dict(train_metadata.get("sample_report") or {}))
        training_flow_contract = copy.deepcopy(dict(train_metadata.get("training_flow_contract") or {}))
        model_cfg = copy.deepcopy(dict(checkpoint_selection["selected_model_cfg"]))
        model_cfg.pop("inference_device", None)

        scoring_seq_ids = list(split_manifest["val_ids"])
        if not scoring_seq_ids:
            raise RuntimeError(f"{model_name} search requires non-empty val_ids for fair scoring")

        scoring_runs = []
        for config_name in scoring_config_names:
            experiment_id = Path(config_name).stem
            baseline_metrics = None
            if baseline_scoring_metrics_by_experiment is not None and experiment_id in baseline_scoring_metrics_by_experiment:
                baseline_metrics = dict(baseline_scoring_metrics_by_experiment[experiment_id])
            scoring_runs.append(
                run_scoring_experiment(
                    config_name=config_name,
                    prepare_root=prepare_root,
                    raw_root=raw_root,
                    method_name=model_name,
                    estimator_cfgs=estimator_cfgs,
                    model_cfgs={model_name: model_cfg},
                    output_root=candidate_root / "scoring_runs",
                    seq_ids=scoring_seq_ids,
                    baseline_metrics=baseline_metrics,
                )
            )

        metrics_by_experiment = {run["experiment_id"]: dict(run["metrics"]) for run in scoring_runs}
        safe_metrics = metrics_by_experiment.get("e0_safe_mode")
        if safe_metrics is None and baseline_scoring_metrics_by_experiment is not None:
            safe_metrics = baseline_scoring_metrics_by_experiment.get("e0_safe_mode")
        if safe_metrics is None:
            metrics_by_experiment = inject_self_baseline_safe_metrics(metrics_by_experiment)
            safe_metrics = metrics_by_experiment["e0_safe_mode"]
        elif "e0_safe_mode" not in metrics_by_experiment:
            metrics_by_experiment = dict(metrics_by_experiment)
            metrics_by_experiment["e0_safe_mode"] = dict(safe_metrics)
        safe_p95 = float(safe_metrics.get("p95", 0.0))
        safe_ekf_p95 = float(safe_metrics.get("ekf_p95", safe_p95))

        normalized_checkpoint_selection = _normalize_checkpoint_selection_report(
            dict(checkpoint_selection.get("selection_report") or {}),
            default_selection_mode=str(checkpoint_selection_mode),
        )

        candidate_report = {
            "method_name": model_name,
            "model_name": model_name,
            "seed": int(seed),
            "signature": candidate_signature,
            "overrides": dict(overrides),
            "cache_context": cache_context,
            "estimator_cfgs": estimator_cfgs,
            "baseline_method_name": "ekf",
            "safe_scoring_included": namespace["_SAFE_SCORING_EXPERIMENT"] in scoring_config_names,
            "scoring_seq_ids": scoring_seq_ids,
            "scoring_runs": scoring_runs,
            "score_vector": score_candidate_lexicographically(metrics_by_experiment=metrics_by_experiment),
            "safe_margin_vs_ekf": safe_p95 - safe_ekf_p95,
            "safe_relative_margin": (safe_p95 - safe_ekf_p95) / max(safe_ekf_p95, 1e-9),
            "train_result": train_result,
            "train_report": train_report,
            "sample_report": sample_report,
            "training_flow_contract": training_flow_contract,
            "model_cfg": model_cfg,
            "checkpoint_selection": normalized_checkpoint_selection,
        }
        write_json(
            report_path,
            {
                key: value
                for key, value in candidate_report.items()
                if key not in {"train_result"}
            },
        )
        return candidate_report

    def patched_neural(*, model_name: str, **kwargs: Any):
        _sync_exported_globals_into_namespace(namespace)
        print(
            f"[paper-run] 神经搜索候选开始: {model_name} | seed={kwargs.get('seed')} | "
            f"overrides={kwargs.get('overrides')}",
            flush=True,
        )
        resumed = _maybe_resume_neural_candidate(
            model_name=model_name,
            seed=int(kwargs["seed"]),
            search_root=kwargs["search_root"],
            overrides=dict(kwargs["overrides"]),
        )
        if resumed is not None:
            print(f"[paper-run] 神经搜索候选从缓存恢复: {model_name}", flush=True)
            return resumed
        expected_selection_mode = str(
            (namespace["_resolve_neural_search_surface"]("paper") or {}).get("checkpoint_selection_mode") or ""
        ).strip().lower()
        allow_cache = bool(kwargs.get("allow_cache", False))
        if allow_cache and expected_selection_mode == "downstream_val_rescore":
            candidate_signature = namespace["_search_candidate_signature"](
                model_name,
                dict(kwargs["overrides"]),
                int(kwargs["seed"]),
            )
            candidate_report_path = Path(kwargs["search_root"]) / model_name / candidate_signature / "candidate_report.json"
            if candidate_report_path.is_file():
                cached_report = dict(namespace["_read_json"](candidate_report_path) or {})
                checkpoint_selection = dict(cached_report.get("checkpoint_selection") or {})
                train_report = dict(cached_report.get("train_report") or {})
                selection_report_path = str(
                    checkpoint_selection.get("selection_report_path")
                    or train_report.get("paper_checkpoint_selection_report")
                    or ""
                ).strip()
                selection_mode = ""
                candidate_count = 0
                if selection_report_path and Path(selection_report_path).is_file():
                    selection_report = dict(namespace["_read_json"](selection_report_path) or {})
                    selection_mode = str(selection_report.get("selection_mode") or "").strip().lower()
                    candidate_count = int(selection_report.get("candidate_count") or 0)
                if selection_mode != "downstream_val_rescore" or candidate_count <= 0:
                    kwargs = dict(kwargs)
                    kwargs["allow_cache"] = False
        updates = {
            "method_name": model_name,
            "baseline_method_name": "ekf",
            "include_safe": model_name != "ekf",
        }
        with _temporary_globals(namespace, updates):
            result = _build_compatible_neural_candidate_report(model_name=model_name, **kwargs)
            print(f"[paper-run] 神经搜索候选完成: {model_name}", flush=True)
            return result

    namespace["_run_classical_search_candidate"] = patched_classical
    namespace["_run_neural_search_candidate"] = patched_neural
    namespace["_search_candidate_signature"] = _unique_override_signature


def _wrap_model_cfg_builder(namespace: dict[str, Any]) -> None:
    original_builder = namespace["_build_model_cfgs_from_train_reports"]

    def patched_builder(train_results: dict[str, Any]) -> dict[str, dict[str, Any]]:
        model_cfgs = original_builder(train_results)
        for model_cfg in model_cfgs.values():
            if isinstance(model_cfg, dict):
                model_cfg["inference_device"] = "cpu"  # inference_device 是输入配置键；runtime_device 是 dataclass 状态字段（由 _apply_runtime_device 写入），不得作为输入键注入 cfg。
        return model_cfgs

    namespace["_build_model_cfgs_from_train_reports"] = patched_builder


def _wrap_scope_specific_contract_gate(namespace: dict[str, Any]) -> None:
    def patched_collector(**kwargs: Any) -> list[str]:
        violations: list[str] = []
        # wrapper 修复 (2026-08-08 paper-run e9 趋势跑): 原 contract gate 写死在
        # "full paper run" 假设上, 不管 paper_scope 是什么都触发 blocked. sim_e9_only
        # 趋势跑的设计意图是只看 sim e9 序列 (miluv/ntu_viral 关闭, small_eval 神经
        # profile, cpu device), 这些都违反 full-run contract. 让 contract gate 在
        # sim_e9_only scope 下完全短路 (返回空), gate 真正的作用交给后续 stage 自己
        # 处理 (env_check / readiness_audit).
        active_scope = str(
            namespace.get("_ACTIVE_PAPER_SCOPE") or ""
        ).strip().lower()
        if active_scope == "sim_e9_only":
            return violations
        requested_public_datasets = list(kwargs.get("requested_public_datasets") or [])
        required_public_datasets = list(namespace["_SUPPORTED_PUBLIC_DATASETS"])
        allow_incomplete_paper_run = bool(kwargs.get("allow_incomplete_paper_run"))
        skip_public_benchmarks = bool(
            namespace.get("_WRAPPER_SKIP_PUBLIC_BENCHMARKS", kwargs.get("skip_public_benchmarks"))
        )
        reuse_search_cache = bool(kwargs.get("reuse_search_cache"))
        requested_device = str(kwargs.get("requested_device") or "").strip().lower()
        requested_neural_search_profile = str(kwargs.get("requested_neural_search_profile") or "").strip().lower()
        if not allow_incomplete_paper_run and skip_public_benchmarks:
            violations.append(
                "full paper run requires the complete public benchmark surface and cannot skip public benchmarks"
            )
        if not allow_incomplete_paper_run and set(requested_public_datasets) != set(required_public_datasets):
            violations.append(
                "full paper run requires all supported public datasets: "
                f"required={required_public_datasets}, requested={requested_public_datasets}"
            )
        if requested_device not in ("cuda", "auto"):
            violations.append("full paper run requires requested device 'cuda' or 'auto' for fresh neural training")
        if reuse_search_cache:
            violations.append("full paper run requires fresh search/training and cannot reuse search cache")
        if requested_neural_search_profile != "paper":
            violations.append("full paper run requires neural search profile 'paper'")
        if any(
            kwargs.get(name) is not None
            for name in ("lstm_epochs", "liquid_epochs", "batch_size", "eval_batch_size")
        ):
            violations.append(
                "full paper run requires the default paper-grade training budget from the model YAML and cannot use CLI budget overrides"
            )
        return violations

    namespace["_collect_full_paper_run_contract_violations"] = patched_collector


def _wrap_resume_fingerprint_compat(namespace: dict[str, Any]) -> None:
    original_resolver = namespace["_resolve_search_implementation_fingerprint"]
    build_file_fingerprint = namespace["_build_file_fingerprint"]
    resolve_experiment_paths = namespace["_resolve_experiment_fingerprint_paths"]

    def patched_resolver(
        *,
        search_kind: str,
        implementation_name: str,
        experiment_config_names: tuple[str, ...],
    ) -> str:
        common_paths = tuple(namespace.get("_SEARCH_COMMON_FINGERPRINT_PATHS") or ())
        impl_map_name = (
            "_SEARCH_MODEL_FINGERPRINT_PATHS" if search_kind == "neural" else "_SEARCH_ESTIMATOR_FINGERPRINT_PATHS"
        )
        implementation_paths = dict(namespace.get(impl_map_name) or {}).get(implementation_name)
        if implementation_paths is None:
            return original_resolver(
                search_kind=search_kind,
                implementation_name=implementation_name,
                experiment_config_names=experiment_config_names,
            )

        compat_common_paths: list[str] = []
        for path_text in common_paths:
            normalized = str(path_text).replace("\\", "/")
            if normalized.endswith("scripts/20_run_paper_experiments.py"):
                compat_common_paths.append(str(_IMPL_SOURCE_PATH))
            else:
                compat_common_paths.append(str(path_text))

        fingerprint_paths = tuple(
            dict.fromkeys(
                compat_common_paths
                + list(implementation_paths)
                + list(resolve_experiment_paths(experiment_config_names))
            )
        )
        return build_file_fingerprint(fingerprint_paths)

    namespace["_resolve_search_implementation_fingerprint"] = patched_resolver


def _wrap_neural_final_selected_resume(namespace: dict[str, Any]) -> None:
    original_search = namespace["_search_best_neural_train_result"]
    original_surface_resolver = namespace["_resolve_neural_search_surface"]

    def _slim_train_report(train_report: dict[str, Any]) -> dict[str, Any]:
        allowed_keys = {
            "model_name",
            "status",
            "checkpoint_format",
            "trainer_mode",
            "epochs",
            "best_epoch",
            "best_loss",
            "best_selection_score",
            "checkpoint_path",
            "best_ckpt",
            "report_path",
            "paper_selected_checkpoint_path",
            "paper_selected_best_epoch",
            "paper_selected_score_vector",
            "paper_selected_metrics",
            "paper_checkpoint_selection_report",
            "tail_selection_observation_coeff",
            "best_export_epoch",
            "best_export_score",
            "score_role_summary",
            "checkpoint_role_summary",
            "control_anchor_checkpoint",
            "training_best_checkpoint",
            "export_best_checkpoint",
            "phase_control_summary",
            "current_phase_control_state",
            "global_best_phase_control_state",
            "soft_control_events",
            "soft_control_event_count",
            "phase_shock_streaks",
            "max_phase_shock_streak",
            "global_selection_mode",
            "global_best_phase_name",
            "gate_alignment_readout_lr_scale",
            "soft_control_streak_length",
            "soft_control_streak_lr_decay",
            "soft_control_buffer_epochs",
            "soft_control_cooldown_epochs",
            "soft_control_lr_floor_scale",
            "soft_control_rollback_patience",
            "training_stability_audit",
        }
        return {key: copy.deepcopy(value) for key, value in train_report.items() if key in allowed_keys}

    def _load_stage_result_from_final_selected(*, model_name: str, output_root: Any) -> dict[str, Any] | None:
        train_root = Path(output_root)
        final_root = train_root / "final_selected"
        search_audit_path = train_root / "search_audit.json"
        train_report_path = final_root / "reports" / f"{model_name}_train_report.json"
        selection_report_path = final_root / "paper_checkpoint_selection" / "final_checkpoint_selection.json"
        required_paths = (search_audit_path, train_report_path, selection_report_path)
        if not all(path.is_file() for path in required_paths):
            return None

        expected_selection_mode = str(
            (namespace["_resolve_neural_search_surface"]("paper") or {}).get("checkpoint_selection_mode") or ""
        ).strip().lower()
        if expected_selection_mode and expected_selection_mode != "downstream_val_rescore":
            return None

        train_report = dict(namespace["_read_json"](train_report_path) or {})
        checkpoint_path = str(train_report.get("checkpoint_path") or "").strip()
        if not checkpoint_path or not Path(checkpoint_path).is_file():
            return None

        search_audit = dict(namespace["_read_json"](search_audit_path) or {})
        raw_selection_report = dict(namespace["_read_json"](selection_report_path) or {})
        selection_report = _normalize_checkpoint_selection_report(
            raw_selection_report,
            default_selection_mode="downstream_val_rescore",
        )
        if not search_audit or not selection_report:
            return None

        selection_mode = str(selection_report.get("selection_mode") or "").strip().lower()
        candidate_count = int(selection_report.get("candidate_count") or 0)
        has_legacy_selected_checkpoint = bool(
            str(selection_report.get("selected_checkpoint_path") or "").strip()
        ) and "selection_mode" not in raw_selection_report and "candidate_count" not in raw_selection_report
        if selection_mode != "downstream_val_rescore" or (
            candidate_count <= 0 and not has_legacy_selected_checkpoint
        ):
            return None

        metrics_by_experiment = dict(search_audit.get("selected_metrics") or {})
        score_vector = list(search_audit.get("selected_score_vector") or [])
        selected_checkpoint_path = str(
            selection_report.get("selected_checkpoint_path")
            or train_report.get("paper_selected_checkpoint_path")
            or checkpoint_path
        ).strip()
        if not selected_checkpoint_path or not Path(selected_checkpoint_path).is_file():
            return None

        train_report = _slim_train_report(train_report)
        train_report["paper_selected_checkpoint_path"] = selected_checkpoint_path
        train_report["paper_selected_best_epoch"] = int(
            selection_report.get("selected_best_epoch") or train_report.get("best_epoch") or 0
        )
        train_report["paper_selected_score_vector"] = score_vector
        train_report["paper_selected_metrics"] = copy.deepcopy(metrics_by_experiment)
        train_report["paper_checkpoint_selection_report"] = str(selection_report_path)

        metadata: dict[str, Any] = {"train_report": train_report}
        artifacts: list[str] = [selected_checkpoint_path, str(train_report_path), str(selection_report_path)]
        audit_files = {
            "protocol_gate": final_root / "audits" / f"{model_name}_protocol_gate.json",
            "sample_report": final_root / "audits" / f"{model_name}_sample_report.json",
            "target_contract": final_root / "audits" / f"{model_name}_target_contract.json",
            "checkpoint_smoke": final_root / "audits" / f"{model_name}_checkpoint_smoke.json",
            "training_flow_contract": final_root / "audits" / f"{model_name}_training_flow_contract.json",
        }
        for metadata_key, path in audit_files.items():
            if path.is_file():
                metadata[metadata_key] = dict(namespace["_read_json"](path) or {})
                artifacts.append(str(path))

        resumed_search_audit = copy.deepcopy(search_audit)
        resumed_search_audit["final_checkpoint_selection"] = copy.deepcopy(selection_report)
        resumed_search_audit["selected_score_vector"] = list(score_vector)
        resumed_search_audit["selected_metrics"] = copy.deepcopy(metrics_by_experiment)

        return {
            "train_result": _ResumeStageResult(
                stage_name="train_pipeline",
                artifacts=artifacts,
                metadata=metadata,
            ),
            "search_audit": resumed_search_audit,
        }

    def patched_search(*, allow_cache: bool = False, **kwargs: Any):
        _sync_exported_globals_into_namespace(namespace)
        model_name = str(kwargs.get("model_name") or "")
        print(f"[paper-run] 神经最终选择搜索开始: {model_name} | allow_cache={allow_cache}", flush=True)
        if allow_cache:
            resumed = _load_stage_result_from_final_selected(
                model_name=model_name,
                output_root=kwargs["output_root"],
            )
            if resumed is not None:
                print(f"[paper-run] 神经最终选择搜索恢复: {model_name}", flush=True)
                return resumed
        result = original_search(allow_cache=allow_cache, **kwargs)
        train_result = result.get("train_result")
        metadata = getattr(train_result, "metadata", None)
        if isinstance(metadata, dict) and isinstance(metadata.get("train_report"), dict):
            metadata["train_report"] = _slim_train_report(dict(metadata["train_report"]))
        print(f"[paper-run] 神经最终选择搜索完成: {model_name}", flush=True)
        return result

    namespace["_search_best_neural_train_result"] = patched_search


def _wrap_core_experiment_resume(namespace: dict[str, Any]) -> None:
    original_run_core_experiment = namespace["_run_core_experiment"]

    def _load_resumed_core_experiment(
        *,
        config_name: str,
        prepare_root: Any,
        raw_root: Any,
        estimator_cfgs: dict[str, dict[str, Any]],
        model_cfgs: dict[str, dict[str, Any]],
        output_root: Any,
    ) -> dict[str, Any] | None:
        experiment_id = Path(config_name).stem
        experiment_output_root = Path(output_root) / experiment_id
        eval_root = experiment_output_root / "eval"
        core_prediction_index_path = experiment_output_root / "core" / "audits" / "prediction_index.json"
        required_paths = [
            eval_root / "metrics" / "metric_table.csv",
            eval_root / "statistics" / "statistics_table.json",
            eval_root / "cases" / "selected_cases.json",
            eval_root / "plotting_inputs" / "main_table.json",
            eval_root / "plotting_inputs" / "runtime_table.json",
            core_prediction_index_path,
        ]
        if not all(path.is_file() for path in required_paths):
            return None

        statistics_payload = dict(namespace["_read_json"](eval_root / "statistics" / "statistics_table.json") or {})
        selected_cases = dict(namespace["_read_json"](eval_root / "cases" / "selected_cases.json") or {})
        main_table = list(namespace["_read_json"](eval_root / "plotting_inputs" / "main_table.json") or [])
        runtime_table = list(namespace["_read_json"](eval_root / "plotting_inputs" / "runtime_table.json") or [])
        prediction_index = list(namespace["_read_json"](core_prediction_index_path) or [])
        sweep_path = eval_root / "plotting_inputs" / "sweep_table.json"
        sweep_table = list(namespace["_read_json"](sweep_path) or []) if sweep_path.is_file() else None

        if not main_table or not runtime_table:
            return None
        if not isinstance(main_table[0], dict) or not isinstance(runtime_table[0], dict):
            return None
        if not isinstance(statistics_payload, dict) or not isinstance(selected_cases, dict):
            return None
        expected_scene_ids = _expected_scene_ids_for_config(namespace, config_name)
        actual_scene_ids = (
            _extract_scene_ids_from_payload(prediction_index)
            or _extract_scene_ids_from_payload(selected_cases)
            or _extract_scene_ids_from_payload(runtime_table)
        )
        if expected_scene_ids and actual_scene_ids != expected_scene_ids:
            return None

        main_method = next(
            (row for row in main_table if str(row.get("method_name")) in model_cfgs or str(row.get("method_name")) in estimator_cfgs),
            None,
        )
        if main_method is None:
            return None

        eval_result = _ResumeStageResult(
            stage_name="eval_pipeline",
            artifacts=[
                str((eval_root / "metrics" / "metric_table.csv").resolve()),
                str((eval_root / "statistics" / "statistics_table.json").resolve()),
                str((eval_root / "cases" / "selected_cases.json").resolve()),
                str((eval_root / "plotting_inputs" / "main_table.json").resolve()),
                str((eval_root / "plotting_inputs" / "runtime_table.json").resolve()),
            ],
            metadata={
                "output_root": str(eval_root.resolve()),
                "metrics_path": str((eval_root / "metrics" / "metric_table.csv").resolve()),
                "statistics_path": str((eval_root / "statistics" / "statistics_table.json").resolve()),
                "selected_cases_path": str((eval_root / "cases" / "selected_cases.json").resolve()),
                "main_table_path": str((eval_root / "plotting_inputs" / "main_table.json").resolve()),
                "runtime_table_path": str((eval_root / "plotting_inputs" / "runtime_table.json").resolve()),
                "sweep_table_path": str(sweep_path.resolve()) if sweep_path.is_file() else None,
                "main_table": main_table,
                "runtime_table": runtime_table,
                "selected_cases": selected_cases,
                "statistics": statistics_payload,
                "sweep_table": sweep_table,
                "prediction_bundles": list(main_table),
                "prediction_index": prediction_index,
            },
        )

        core_result = _ResumeStageResult(
            stage_name="core_pipeline",
            artifacts=[str(eval_root.resolve())],
            metadata={
                "prediction_bundles": list(main_table),
                "eval_root": str(eval_root.resolve()),
                "prediction_index": prediction_index,
            },
        )

        return {
            "experiment_id": experiment_id,
            "output_root": str(experiment_output_root.resolve()),
            "core_result": core_result,
            "eval_result": eval_result,
        }

    def patched_run_core_experiment(**kwargs: Any):
        config_name = str(kwargs.get("config_name") or "")
        print(f"[paper-run] 核心实验开始: {Path(config_name).stem}", flush=True)
        if str(namespace.get("_ACTIVE_PAPER_SCOPE") or "").strip().lower() == "sim_e9_only":
            resumed = _load_resumed_core_experiment(
                config_name=config_name,
                prepare_root=kwargs["prepare_root"],
                raw_root=kwargs["raw_root"],
                estimator_cfgs=dict(kwargs["estimator_cfgs"]),
                model_cfgs=dict(kwargs["model_cfgs"]),
                output_root=kwargs["output_root"],
            )
            if resumed is not None:
                print(f"[paper-run] 核心实验从缓存恢复: {Path(config_name).stem}", flush=True)
                return resumed
        result = original_run_core_experiment(**kwargs)
        print(f"[paper-run] 核心实验完成: {Path(config_name).stem}", flush=True)
        return result

    namespace["_run_core_experiment"] = patched_run_core_experiment


def _wrap_sim_root_refresh_contract(namespace: dict[str, Any]) -> None:
    original_sim_root_has_only_placeholder = namespace["_sim_root_has_only_placeholder"]
    original_can_materialize_sim_raw = namespace.get("can_materialize_sim_raw", lambda _output_root: False)
    repo_root = Path(namespace.get("ROOT", _SCRIPT_PATH.parents[1])).resolve()
    repo_default_sim_raw_root = (repo_root / "data" / "raw" / "sim_e9_protocol_20260726").resolve()

    def _get_sim_e9_only_compact_sequence_specs() -> tuple[Any, ...]:
        try:
            from liquidloc.dataio import sim_materializer as _sim_materializer
        except Exception:
            return tuple()
        return tuple(getattr(_sim_materializer, "SIM_E9_ONLY_COMPACT_SEQUENCE_SPECS", ()))

    def _sim_root_matches_compact_profile(raw_root: Path) -> bool:
        if not raw_root.is_dir():
            return False
        compact_specs = _get_sim_e9_only_compact_sequence_specs()
        if not compact_specs:
            return False
        expected_seq_ids = {spec.seq_id for spec in compact_specs}
        # 兼容 02_prepare_sim_data 把单 seq 展开为 30 个 _seed{N} 后缀子目录的 layout：
        # 对每个 expected seq_id，只要 raw_root/<seq_id>/ 下五件套齐备 + gt 非空即视为就绪。
        # 允许 raw_root 含其他条目（其它 seed、其它 scope 的产物），不影响 contract 判定。
        required_files = ("gt.json", "imu.json", "uwb.json", "vio.json", "sim_meta.json")
        for seq_id in sorted(expected_seq_ids):
            seq_dir = raw_root / seq_id
            if not seq_dir.is_dir():
                return False
            for fname in required_files:
                if not (seq_dir / fname).is_file():
                    return False
            try:
                gt_rows = namespace["_read_json"](seq_dir / "gt.json")
            except Exception:
                return False
            if not isinstance(gt_rows, list) or not gt_rows:
                return False
        return True

    def patched_sim_root_has_only_placeholder(raw_root: Path) -> bool:
        active_scope = str(namespace.get("_ACTIVE_PAPER_SCOPE") or "").strip().lower()
        if active_scope == "sim_e9_only":
            return not _sim_root_matches_compact_profile(Path(raw_root))
        return original_sim_root_has_only_placeholder(raw_root)

    namespace["_sim_root_has_only_placeholder"] = patched_sim_root_has_only_placeholder

    def patched_can_materialize_sim_raw(output_root: Path) -> bool:
        active_scope = str(namespace.get("_ACTIVE_PAPER_SCOPE") or "").strip().lower()
        if active_scope == "sim_e9_only":
            raw_root = Path(output_root).resolve()
            if raw_root == repo_default_sim_raw_root and raw_root.is_dir() and not _sim_root_matches_compact_profile(raw_root):
                return True
            return bool(original_can_materialize_sim_raw(output_root))
        return bool(original_can_materialize_sim_raw(output_root))

    namespace["can_materialize_sim_raw"] = patched_can_materialize_sim_raw


def _wrap_paper_checkpoint_selection_surface(namespace: dict[str, Any]) -> None:
    original_resolver = namespace["_resolve_neural_search_surface"]
    paper_selection_mode = "downstream_val_rescore"
    paper_epoch_candidate_stride = 10
    legacy_note = (
        "paper profile uses the trainer-selected validation-best checkpoint "
        "instead of a second downstream rescoring layer"
    )
    selection_note = (
        "paper profile enables downstream validation rescoring over exported epoch candidates"
    )
    stride_note = (
        "paper profile exports epoch candidates every 10 epochs for downstream rescoring"
    )

    def patched_resolver(profile_name: str | None) -> dict[str, Any]:
        surface = copy.deepcopy(dict(original_resolver(profile_name)))
        normalized_profile = str(profile_name or "paper").strip().lower()
        if normalized_profile != "paper":
            return surface

        surface["checkpoint_selection_mode"] = paper_selection_mode
        surface["epoch_candidate_stride"] = paper_epoch_candidate_stride

        notes = [str(note) for note in list(surface.get("notes") or []) if str(note) != legacy_note]
        if selection_note not in notes:
            notes.append(selection_note)
        if stride_note not in notes:
            notes.append(stride_note)
        surface["notes"] = notes
        return surface

    namespace["_resolve_neural_search_surface"] = patched_resolver


def _wrap_tail_selection_search_surface(namespace: dict[str, Any]) -> None:
    original_resolver = namespace["_resolve_neural_search_surface"]
    original_iter = namespace["_iter_neural_override_grid"]

    grid_key = "tail_selection_observation_coeff"
    override_key = "train.tail_selection_observation_coeff"
    liquid_model_name = "liquid_ekf"
    # Keep the default first so equal-score tie breaks preserve the prior
    # baseline behavior while still exploring lighter/heavier tail coverage.
    search_values = (0.10, 0.00, 0.05, 0.15)

    def _clone_and_inject_liquid_tail_grid(
        neural_grid: dict[str, dict[str, Any]] | None,
    ) -> dict[str, dict[str, Any]] | None:
        if neural_grid is None:
            return None
        cloned_grid: dict[str, dict[str, Any]] = {}
        for model_name, model_grid in dict(neural_grid).items():
            cloned_grid[str(model_name)] = dict(model_grid or {})
        liquid_grid = dict(cloned_grid.get(liquid_model_name) or {})
        liquid_grid.setdefault(grid_key, search_values)
        cloned_grid[liquid_model_name] = liquid_grid
        return cloned_grid

    def _resolve_tail_search_values(neural_grid: dict[str, dict[str, Any]] | None) -> tuple[float, ...]:
        resolved_grid = _clone_and_inject_liquid_tail_grid(
            neural_grid if neural_grid is not None else namespace.get("_FAIR_NEURAL_GRID")
        )
        if not isinstance(resolved_grid, dict):
            return search_values
        liquid_grid = dict(resolved_grid.get(liquid_model_name) or {})
        raw_values = liquid_grid.get(grid_key, search_values)
        return tuple(float(value) for value in raw_values)

    def patched_resolver(profile_name: str | None) -> dict[str, Any]:
        surface = dict(original_resolver(profile_name))
        surface["neural_grid"] = _clone_and_inject_liquid_tail_grid(surface.get("neural_grid"))
        return surface

    def patched_iter(
        model_name: str,
        *,
        neural_grid: dict[str, dict[str, Any]] | None = None,
        liquid_robustness_profiles: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        injected_grid = _clone_and_inject_liquid_tail_grid(neural_grid)
        overrides = list(
            original_iter(
                model_name,
                neural_grid=injected_grid,
                liquid_robustness_profiles=liquid_robustness_profiles,
            )
        )
        if str(model_name).strip().lower() != liquid_model_name:
            return overrides

        coeff_values = _resolve_tail_search_values(injected_grid)
        if not coeff_values:
            return overrides

        if any(override_key in dict(candidate or {}) for candidate in overrides):
            normalized_overrides: list[dict[str, Any]] = []
            for candidate in overrides:
                normalized_candidate = dict(candidate or {})
                normalized_candidate.pop(grid_key, None)
                if override_key in normalized_candidate:
                    normalized_candidate[override_key] = float(normalized_candidate[override_key])
                normalized_overrides.append(normalized_candidate)
            return normalized_overrides

        expanded_overrides: list[dict[str, Any]] = []
        for candidate in overrides:
            base_candidate = dict(candidate or {})
            base_candidate.pop(grid_key, None)
            for coeff in coeff_values:
                expanded_candidate = dict(base_candidate)
                expanded_candidate[override_key] = float(coeff)
                expanded_overrides.append(expanded_candidate)
        return expanded_overrides

    namespace["_SMALL_EVAL_NEURAL_GRID"] = _clone_and_inject_liquid_tail_grid(namespace.get("_SMALL_EVAL_NEURAL_GRID"))
    namespace["_MEDIUM_EVAL_NEURAL_GRID"] = _clone_and_inject_liquid_tail_grid(namespace.get("_MEDIUM_EVAL_NEURAL_GRID"))
    namespace["_FAIR_NEURAL_GRID"] = _clone_and_inject_liquid_tail_grid(namespace.get("_FAIR_NEURAL_GRID"))
    namespace["_resolve_neural_search_surface"] = patched_resolver
    namespace["_iter_neural_override_grid"] = patched_iter


def _wrap_classical_search_audit_metrics(namespace: dict[str, Any]) -> None:
    original_search = namespace["_search_best_classical_estimator_cfgs"]

    def _metrics_by_experiment(report: dict[str, Any]) -> dict[str, dict[str, float]]:
        metrics: dict[str, dict[str, float]] = {}
        for run in report.get("scoring_runs") or []:
            if not isinstance(run, dict):
                continue
            experiment_id = str(run.get("experiment_id") or "").strip()
            run_metrics = run.get("metrics")
            if experiment_id and isinstance(run_metrics, dict):
                metrics[experiment_id] = copy.deepcopy(dict(run_metrics))
        return metrics

    def _patch_method_audit(method_audit: Any) -> dict[str, Any]:
        if not isinstance(method_audit, dict):
            return {}
        patched = copy.deepcopy(method_audit)
        selected_signature = str(patched.get("selected_signature") or "").strip()
        if not selected_signature:
            return patched
        selected_report: dict[str, Any] | None = None
        for report in patched.get("grid_reports") or []:
            if isinstance(report, dict) and str(report.get("signature") or "").strip() == selected_signature:
                selected_report = report
                break
        if not selected_report:
            return patched
        selected_metrics = _metrics_by_experiment(selected_report)
        if selected_metrics:
            patched["selected_metrics"] = selected_metrics
        score_vector = selected_report.get("score_vector")
        if isinstance(score_vector, list):
            patched["selected_score_vector"] = copy.deepcopy(score_vector)
        return patched

    def patched_search(*args: Any, **kwargs: Any):
        result = original_search(*args, **kwargs)
        if not isinstance(result, dict):
            return result
        search_audit = result.get("search_audit")
        if not isinstance(search_audit, dict):
            return result
        patched_audit = copy.deepcopy(search_audit)
        for method_name in ("robust_ekf",):
            patched_audit[method_name] = _patch_method_audit(patched_audit.get(method_name))
        result = copy.deepcopy(result)
        result["search_audit"] = patched_audit
        output_root = kwargs.get("output_root")
        if output_root is not None:
            audit_path = Path(output_root) / "search_audit.json"
            try:
                namespace["_write_json"](audit_path, patched_audit)
            except Exception:
                pass
        return result

    namespace["_search_best_classical_estimator_cfgs"] = patched_search


def _install_liquid_failure_lstm_hook(namespace: dict[str, Any]) -> dict[str, Any]:
    search_key = "_search_best_neural_train_result"
    writer_key = "_write_failed_paper_run_report"
    original_search = _unwrap_namespace_proxy(search_key, namespace[search_key])
    original_writer = _unwrap_namespace_proxy(writer_key, namespace[writer_key])
    state: dict[str, Any] = {
        "liquid_search_kwargs": None,
        "lstm_search_kwargs": None,
        "lstm_result": None,
    }

    def _recover_lstm_result(output_root: Path) -> dict[str, Any] | None:
        cached_result = state.get("lstm_result")
        if isinstance(cached_result, dict):
            return cached_result
        # Prefer cached lstm_ekf kwargs (success-path call) over deriving from liquid
        lstm_search_kwargs = state.get("lstm_search_kwargs")
        if isinstance(lstm_search_kwargs, dict):
            try:
                recovered_result = original_search(**lstm_search_kwargs)
            except Exception:
                return None
            state["lstm_result"] = recovered_result
            return recovered_result
        # Fall back to deriving from liquid_ekf kwargs
        liquid_search_kwargs = state.get("liquid_search_kwargs")
        if not isinstance(liquid_search_kwargs, dict):
            return None
        resolved_training_budget = dict(liquid_search_kwargs.get("resolved_training_budget") or {})
        lstm_epochs = resolved_training_budget.get("lstm_epochs")
        if lstm_epochs is None:
            return None
        derived_lstm_kwargs = dict(liquid_search_kwargs)
        derived_lstm_kwargs["model_name"] = "lstm_ekf"
        derived_lstm_kwargs["epochs"] = int(lstm_epochs)
        derived_lstm_kwargs["output_root"] = Path(output_root) / "train" / "lstm_ekf"
        try:
            recovered_result = original_search(**derived_lstm_kwargs)
        except Exception:
            return None
        state["lstm_result"] = recovered_result
        return recovered_result

    def patched_search(*, model_name: str, **kwargs: Any):
        name_lower = str(model_name).strip().lower()
        print(f"[paper-run] 神经训练搜索开始: {model_name}", flush=True)
        if name_lower == "liquid_ekf":
            state["liquid_search_kwargs"] = dict(kwargs)
            state["lstm_result"] = None
        elif name_lower == "lstm_ekf":
            state["lstm_search_kwargs"] = dict(kwargs)
        result = original_search(model_name=model_name, **kwargs)
        if name_lower == "lstm_ekf" and isinstance(result, dict):
            state["lstm_result"] = result
        print(f"[paper-run] 神经训练搜索完成: {model_name}", flush=True)
        return result

    def patched_writer(output_root: Path, payload: dict[str, Any]) -> int:
        patched_payload = dict(payload)
        if str(patched_payload.get("stage") or "").strip().lower() == "neural_search_liquid":
            train_reports = dict(patched_payload.get("train_reports") or {})
            if "lstm_ekf" not in train_reports:
                recovered_result = _recover_lstm_result(Path(output_root))
                if isinstance(recovered_result, dict):
                    completed_stages = list(patched_payload.get("completed_stages") or [])
                    if "neural_search_lstm" not in completed_stages:
                        completed_stages.append("neural_search_lstm")
                    patched_payload["completed_stages"] = completed_stages

                    train_result = recovered_result.get("train_result")
                    train_metadata = getattr(train_result, "metadata", None)
                    train_report = (
                        copy.deepcopy(dict(train_metadata.get("train_report") or {}))
                        if isinstance(train_metadata, dict)
                        else {}
                    )
                    if train_report:
                        train_reports["lstm_ekf"] = train_report
                        patched_payload["train_reports"] = train_reports

                    search_audit = copy.deepcopy(dict(recovered_result.get("search_audit") or {}))
                    neural_search = dict(patched_payload.get("neural_search") or {})
                    neural_search["lstm_ekf"] = {
                        "search_audit_path": str((Path(output_root) / "train" / "lstm_ekf" / "search_audit.json").resolve()),
                        "selected_signature": search_audit.get("selected_signature"),
                        "selected_overrides": copy.deepcopy(search_audit.get("selected_overrides")),
                        "selected_seed": search_audit.get("selected_seed"),
                        "selected_score_vector": copy.deepcopy(search_audit.get("selected_score_vector")),
                    }
                    patched_payload["neural_search"] = neural_search
        return original_writer(output_root, patched_payload)

    namespace[search_key] = patched_search
    namespace[writer_key] = patched_writer
    return {
        search_key: patched_search,
        writer_key: patched_writer,
    }


_IMPL_NAMESPACE: dict[str, Any] | None = None
_WRAPPER_RESERVED_NAMES = {
    "_IMPL_NAMESPACE",
    "_IMPL_SOURCE_PATH",
    "_SCRIPT_PATH",
    "_ResumeStageResult",
    "_bootstrap_impl_namespace",
    "_load_impl_namespace",
    "_temporary_globals",
    "_wrap_candidate_functions",
    "_wrap_model_cfg_builder",
    "_wrap_scope_specific_contract_gate",
    "_wrap_resume_fingerprint_compat",
    "_wrap_neural_final_selected_resume",
    "_wrap_core_experiment_resume",
    "_wrap_sim_root_refresh_contract",
    "_wrap_tail_selection_search_surface",
    "_wrap_classical_search_audit_metrics",
    "_install_liquid_failure_lstm_hook",
    "_unwrap_namespace_proxy",
    "main",
}


def _bootstrap_impl_namespace() -> dict[str, Any]:
    global _IMPL_NAMESPACE
    if _IMPL_NAMESPACE is not None:
        return _IMPL_NAMESPACE
    namespace = _load_impl_namespace()
    _wrap_candidate_functions(namespace)
    _wrap_model_cfg_builder(namespace)
    _wrap_scope_specific_contract_gate(namespace)
    _wrap_resume_fingerprint_compat(namespace)
    _wrap_neural_final_selected_resume(namespace)
    _wrap_core_experiment_resume(namespace)
    _wrap_sim_root_refresh_contract(namespace)
    _wrap_paper_checkpoint_selection_surface(namespace)
    _wrap_tail_selection_search_surface(namespace)
    _wrap_classical_search_audit_metrics(namespace)
    _IMPL_NAMESPACE = namespace
    for key, value in namespace.items():
        if key.startswith("__") or key in _WRAPPER_RESERVED_NAMES:
            continue
        if inspect.isfunction(value):
            globals().setdefault(key, _build_namespace_callable_proxy(namespace, key, value))
        else:
            globals().setdefault(key, value)
    return namespace


def main(argv: list[str] | None = None) -> int:
    # 安装 tee：所有 print 同时写入终端和日志文件
    try:
        from liquidloc.common.tee_logger import install_tee, uninstall_tee
        _log_dir = _SCRIPT_PATH.parent.parent / "outputs" / "paper_run" / "logs"
        _run_name = f"paper_run_{__import__('time').strftime('%Y%m%d_%H%M%S')}"
        _original_print = builtins.print

        def _stderr_print(*args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("file", sys.stderr)
            _original_print(*args, **kwargs)

        builtins.print = _stderr_print
        try:
            install_tee(_log_dir, run_name=_run_name)
        finally:
            builtins.print = _original_print
        _tee_installed = True
    except Exception as _tee_err:
        _safe_print(f"[paper-run] tee 安装失败，仅终端输出 | 原因: {_tee_err}", flush=True)
        _tee_installed = False

    _safe_print("[paper-run] 正在引导实现命名空间 ...", flush=True)
    namespace = _bootstrap_impl_namespace()
    _safe_print("[paper-run] 实现命名空间就绪", flush=True)
    preview_parser = namespace["argparse"].ArgumentParser(add_help=False)
    preview_parser.add_argument("--skip-public-benchmarks", action="store_true")
    preview_parser.add_argument("--allow-incomplete-paper-run", action="store_true")
    preview_parser.add_argument("--paper-scope", choices=("full", "sim_e9_only"), default="full")
    preview_parser.add_argument("--public-datasets", nargs="*", choices=namespace["_SUPPORTED_PUBLIC_DATASETS"], default=None)
    preview_parser.add_argument("--reuse-search-cache", action="store_true")
    # 额外解析关键参数，仅用于打印，不影响 impl main 的解析
    preview_parser.add_argument("--neural-search-profile", choices=("paper", "small_eval", "medium_eval"), default="paper")
    preview_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    preview_parser.add_argument("--lstm-epochs", type=int, default=None)
    preview_parser.add_argument("--liquid-epochs", type=int, default=None)
    preview_parser.add_argument("--batch-size", type=int, default=None)
    preview_parser.add_argument("--eval-batch-size", type=int, default=None)
    preview_parser.add_argument("--seed", type=int, default=0)
    preview_parser.add_argument("--auto-materialize-sim", dest="auto_materialize_sim", action="store_true", default=True)
    preview_parser.add_argument("--no-auto-materialize-sim", dest="auto_materialize_sim", action="store_false")
    preview_args, _ = preview_parser.parse_known_args(argv)
    for key in list(namespace.keys()):
        if key.startswith("__") or key in _WRAPPER_RESERVED_NAMES:
            continue
        if key in globals():
            namespace[key] = globals()[key]
    namespace.setdefault("neural_search_results", {})
    if preview_args.paper_scope == "full":
        requested_public_datasets = namespace["_resolve_requested_public_datasets"](
            list(preview_args.public_datasets) if preview_args.public_datasets is not None else None,
            skip_public_benchmarks=bool(preview_args.skip_public_benchmarks),
        )
    else:
        requested_public_datasets = []
    wrapper_overrides = {
        "_ACTIVE_PAPER_SCOPE": str(preview_args.paper_scope),
        "_WRAPPER_SKIP_PUBLIC_BENCHMARKS": bool(preview_args.skip_public_benchmarks),
        "_WRAPPER_REQUESTED_PUBLIC_DATASETS": requested_public_datasets,
    }
    _safe_print(
        f"[paper-run] 开始主执行 | scope={preview_args.paper_scope} | "
        f"skip_public_benchmarks={preview_args.skip_public_benchmarks} | "
        f"reuse_search_cache={preview_args.reuse_search_cache}",
        flush=True,
    )
    # 打印所有命令行参数值，方便核对
    _safe_print("[paper-run] 命令行参数:", flush=True)
    _safe_print(f"  paper_scope = {preview_args.paper_scope!r}", flush=True)
    _safe_print(f"  skip_public_benchmarks = {preview_args.skip_public_benchmarks!r}", flush=True)
    _safe_print(f"  allow_incomplete_paper_run = {preview_args.allow_incomplete_paper_run!r}", flush=True)
    _safe_print(f"  public_datasets = {preview_args.public_datasets!r}", flush=True)
    _safe_print(f"  reuse_search_cache = {preview_args.reuse_search_cache!r}", flush=True)
    _safe_print(f"  requested_public_datasets = {requested_public_datasets!r}", flush=True)
    _safe_print(f"  neural_search_profile = {preview_args.neural_search_profile!r}", flush=True)
    _safe_print(f"  device = {preview_args.device!r}", flush=True)
    _safe_print(f"  lstm_epochs = {preview_args.lstm_epochs!r}", flush=True)
    _safe_print(f"  liquid_epochs = {preview_args.liquid_epochs!r}", flush=True)
    _safe_print(f"  batch_size = {preview_args.batch_size!r}", flush=True)
    _safe_print(f"  eval_batch_size = {preview_args.eval_batch_size!r}", flush=True)
    _safe_print(f"  seed = {preview_args.seed!r}", flush=True)
    _safe_print(f"  auto_materialize_sim = {preview_args.auto_materialize_sim!r}", flush=True)

    # 预解析并打印 neural_search_surface（搜索网格）和 training_budget
    try:
        _surface = namespace["_resolve_neural_search_surface"](preview_args.neural_search_profile)
        if _surface:
            _safe_print("[paper-run] 神经搜索网格 (neural_search_surface):", flush=True)
            _safe_print(f"  profile_name = {_surface.get('profile_name')!r}", flush=True)
            _grid = _surface.get("neural_grid") or {}
            for _model_name, _grid_cfg in _grid.items():
                _safe_print(f"  [{_model_name}] 搜索网格:", flush=True)
                for _k, _v in _grid_cfg.items():
                    _safe_print(f"    {_k} = {_v!r}", flush=True)
            _profiles = _surface.get("liquid_robustness_profiles") or []
            _safe_print(f"  liquid_robustness_profiles 数量 = {len(_profiles)}", flush=True)
            for _i, _p in enumerate(_profiles):
                _safe_print(f"    [{_i}] name = {_p.get('name')!r}", flush=True)
            _budget = _surface.get("training_budget") or {}
            _safe_print("[paper-run] 训练预算 (training_budget):", flush=True)
            for _k, _v in _budget.items():
                _safe_print(f"  {_k} = {_v!r}", flush=True)
            _safe_print(f"  top_k_multiseed = {_surface.get('top_k_multiseed')!r}", flush=True)
            _safe_print(f"  multiseed_values = {_surface.get('multiseed_values')!r}", flush=True)
            _safe_print(f"  epoch_candidate_stride = {_surface.get('epoch_candidate_stride')!r}", flush=True)
            _safe_print(f"  checkpoint_selection_mode = {_surface.get('checkpoint_selection_mode')!r}", flush=True)
    except Exception as _surface_err:
        _safe_print(f"[paper-run] 神经搜索网格预解析失败: {_surface_err}", flush=True)
    hook_updates = _install_liquid_failure_lstm_hook(namespace)
    wrapper_globals = globals()
    sentinel = object()
    previous = {key: wrapper_globals.get(key, sentinel) for key in hook_updates}
    wrapper_globals.update(hook_updates)
    result = None
    _original_print = builtins.print

    def _impl_print_router(*args: Any, **kwargs: Any) -> None:
        target_stream = kwargs.get("file")
        if target_stream is not None and target_stream is not sys.stdout:
            _original_print(*args, **kwargs)
            return
        sep = kwargs.get("sep", " ")
        text = sep.join(str(arg) for arg in args)
        if _looks_like_json_payload(text):
            _original_print(*args, **kwargs)
            return
        routed_kwargs = dict(kwargs)
        routed_kwargs["file"] = sys.stderr
        _original_print(*args, **routed_kwargs)
    try:
        with _temporary_globals(namespace, wrapper_overrides):
            _safe_print("[paper-run] 进入实现层 main()", flush=True)
            builtins.print = _impl_print_router
            try:
                result = namespace["main"](argv)
            finally:
                builtins.print = _original_print
            _safe_print(f"[paper-run] 实现层 main() 返回: {result}", flush=True)
    finally:
        builtins.print = _original_print
        for key, value in previous.items():
            if value is sentinel:
                wrapper_globals.pop(key, None)
            else:
                wrapper_globals[key] = value
        # 卸载 tee，关闭日志文件
        if _tee_installed:
            try:
                uninstall_tee()
            except Exception:
                pass
    return 0 if result is None else int(result)


if __name__ == "__main__":
    raise SystemExit(main())
