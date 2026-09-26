
"""Paper experiment runner."""


from __future__ import annotations  

import argparse  
import copy  
import csv  
import hashlib  
import itertools  
import json  
import math  
import pickle  
import shutil  
import subprocess  
import sys  
from functools import lru_cache  
from pathlib import Path  
from collections.abc import Mapping, Sequence
from typing import Any  

import torch  

ROOT = Path(__file__).resolve().parents[3]  
from liquidloc.analysis.summary_builder import build_summary  
from liquidloc.common.config_utils import load_yaml_config  
from liquidloc.common.seed_utils import set_global_seed  
from liquidloc.factories.model_factory import create_model  
from liquidloc.common.prepared_inputs import (  
    expand_scene_tasks_across_seq_ids,  
    load_ground_truth_by_seq_id,  
    load_prepare_manifest,  
    load_prepared_events_by_seq_id,  
    resolve_experiment_seq_ids_from_prepare_manifest,  
    load_source_report_by_seq_id,  
    resolve_seq_ids_from_prepare_manifest,  
)  
from liquidloc.dataio.registry.public_dataset_registry import (
    get_dataset_entry,
    load_public_dataset_registry,
    resolve_public_eval_seq_ids,
)
from liquidloc.dataio.manifests.dataset_checks import (  
    inspect_layout_family_coverage,
    inspect_sim_materialized_contract,
)
from liquidloc.dataio.manifests.split_builder import build_splits  
from liquidloc.dataio.sim_materializer import can_materialize_sim_raw  
from liquidloc.dataio.readers.miluv_reader import inspect_miluv_raw_readiness  
from liquidloc.dataio.readers.ntu_viral_reader import inspect_ntu_viral_raw_readiness  
from liquidloc.pipelines.core_pipeline import CorePipeline  
from liquidloc.pipelines.eval_pipeline import (  
    EvalPipeline,
    build_repeat_summary_rows,
    build_scene_summary_rows,
)
from liquidloc.pipelines.prepare_pipeline import PreparePipeline  
from liquidloc.pipelines.public_benchmark_pipeline import PublicBenchmarkPipeline  
from liquidloc.pipelines.train_pipeline import TrainPipeline  
from liquidloc.plotting.plot_calibration import render_calibration_figure  
from liquidloc.plotting.plot_cases import render_case_figures  
from liquidloc.plotting.plot_main_table import render_main_table_figure  
from liquidloc.plotting.plot_runtime import render_runtime_figure  
from liquidloc.plotting.plot_sweeps import render_sweep_figure  
from liquidloc.plotting.plot_trajectories import render_trajectory_figure  
from liquidloc.protocol.experiment_gates import (  
    get_public_benchmark_allowed_datasets,  
    load_experiment_protocol,  
)
from liquidloc.scenarios.scene_sampler import sample_scenes  


_SIM_REQUIRED_FILES = ("imu.json", "uwb.json", "vio.json", "gt.json", "anchor_layout.json")  
_PAPER_MIN_SIM_LAYOUT_FAMILY_COUNT = 3  
_CORE_EXPERIMENT_CONFIGS = (  
    "e0_safe_mode.yaml",  
    "e0_traditional_only.yaml",  
    "e1_main_table.yaml",  
    "e2_async.yaml",  
    "e3_nlos.yaml",  
    "e4_visual.yaml",  
    "e6_geometry.yaml",  
    "e8_runtime.yaml",  
    "e9_dual_degradation.yaml",  
    "e10_modality_recovery.yaml",  
    "e11_phase_switch.yaml",  
)  
_HELD_OUT_CONCLUSION_CONFIGS = (  
    "e0_safe_mode.yaml",  
    "e1_main_table.yaml",  
    "e2_async.yaml",  
    "e3_nlos.yaml",  
    "e9_dual_degradation.yaml",  
)  
_EKF_INIT_SENSITIVITY_CONFIGS = (  
    "e0_safe_mode.yaml",  
    "e2_async.yaml",  
    "e3_nlos.yaml",  
    "e9_dual_degradation.yaml",  
)  
_FAIR_EKF_GRID = {  
    "window_size": (5, 10, 20),
    "max_iters": (5,),
    "uwb_noise": (0.20, 0.25, 0.30),  
    "vio_pos": (0.06, 0.08, 0.10),  
    "vio_yaw": (0.02, 0.03),  
}  
_FAIR_ROBUST_GRID = {  
    "delta": (1.0, 1.5, 2.0),  
    "quality_floor": (0.15, 0.18, 0.22),  
}  
_FAIR_FGO_GRID = {  
    "window_size": (20,),  
    "max_iters": (5, 15),  
}  
_FAIR_NEURAL_GRID = {  
    "lstm_ekf": {  
        "window": (20, 30),  
        "hidden_dim": (64, 96),  
        "lr": (1e-3, 3e-4),  
        "weight_decay": (0.0, 1e-4),  
    },  
    "liquid_ekf": {  
        "window": (20, 30),  
        
        "hidden_dim": (44, 66),  
        "lr": (1e-3, 3e-4),  
    },  
}  
_LIQUID_ROBUSTNESS_PROFILES = (  
    {  
        "name": "v13_cellgate_tail_balanced",  
        "train.weight_decay": 1e-4,  
        "network.pooling_logit_scale": 0.25,  
        "network.cell_update_scale_floor": 0.80,  
        "network.cell_update_scale_span": 0.20,  
        "network.reliability_bias_init": 2.35,  
        "network.bad_observation_floor": 0.16,  
        "network.bad_observation_span": 0.84,  
        "network.bad_observation_residual_full_scale": 0.42,  
        "network.bad_observation_track_drop_full_scale": 18.0,  
        "network.bad_observation_reproj_slope_full_scale": 0.75,  
        "network.bad_observation_interaction_coeff": 0.95,  
        "bridge_thresholds.robust_supplement_quality_threshold": 0.52,  
        "bridge_thresholds.robust_supplement_alignment_threshold": 0.42,  
        "bridge_thresholds.observation_risk_blend": 0.38,  
        "bridge_thresholds.uwb_geometric_bias_full_scale": 0.45,  
        "bridge_thresholds.async_gap_full_scale": 0.28,  
    },  
    {  
        "name": "v13_cellgate_tail_conservative",  
        "train.weight_decay": 3e-4,  
        "network.pooling_logit_scale": 0.30,  
        "network.cell_update_scale_floor": 0.74,  
        "network.cell_update_scale_span": 0.26,  
        "network.reliability_bias_init": 2.10,  
        "network.bad_observation_floor": 0.22,  
        "network.bad_observation_span": 0.78,  
        "network.bad_observation_residual_full_scale": 0.50,  
        "network.bad_observation_track_drop_full_scale": 24.0,  
        "network.bad_observation_reproj_slope_full_scale": 1.0,  
        "network.bad_observation_interaction_coeff": 0.80,  
        "bridge_thresholds.robust_supplement_quality_threshold": 0.55,  
        "bridge_thresholds.robust_supplement_alignment_threshold": 0.45,  
        "bridge_thresholds.observation_risk_blend": 0.38,  
        "bridge_thresholds.uwb_geometric_bias_full_scale": 0.48,  
        "bridge_thresholds.async_gap_full_scale": 0.28,  
    },  
)  
_SUPPORTED_PUBLIC_DATASETS = get_public_benchmark_allowed_datasets()  
_NEURAL_TOP_K_MULTI_SEED = 4  
_NEURAL_MULTI_SEEDS = (0, 1, 2)  
_SMALL_EVAL_NEURAL_GRID = {  
    "lstm_ekf": {  
        "window": (20,),  
        "hidden_dim": (64,),  
        "lr": (1e-3,),  
        "weight_decay": (0.0,),  
    },  
    "liquid_ekf": {  
        "window": (20,),  
        "hidden_dim": (44,),  
        "lr": (1e-3,),  
    },  
}  
_SMALL_EVAL_LIQUID_ROBUSTNESS_PROFILES = tuple(  
    profile  
    for profile in _LIQUID_ROBUSTNESS_PROFILES  
    if str(profile.get("name") or "") == "v13_cellgate_tail_balanced"  
) or (_LIQUID_ROBUSTNESS_PROFILES[0],)  
_SMALL_EVAL_NEURAL_TOP_K_MULTI_SEED = 1  
_SMALL_EVAL_NEURAL_MULTI_SEEDS = (0,)  
_SMALL_EVAL_EPOCH_CANDIDATE_STRIDE = 4  
_MEDIUM_EVAL_NEURAL_GRID = {  
    "lstm_ekf": {  
        "window": (20,),  
        "hidden_dim": (64, 96),  
        "lr": (1e-3, 3e-4),  
        "weight_decay": (0.0, 1e-4),  
    },  
    "liquid_ekf": {  
        "window": (20,),  
        "hidden_dim": (44, 66),  
        "lr": (1e-3, 3e-4),  
    },  
}  
_MEDIUM_EVAL_LIQUID_ROBUSTNESS_PROFILES = copy.deepcopy(_LIQUID_ROBUSTNESS_PROFILES)  
_MEDIUM_EVAL_NEURAL_TOP_K_MULTI_SEED = 2  
_MEDIUM_EVAL_NEURAL_MULTI_SEEDS = (0, 1)  
_MEDIUM_EVAL_EPOCH_CANDIDATE_STRIDE = 4  
_EKF_TOP_K_FOR_CLASSICAL = 3  
_SEARCH_SCORING_EXPERIMENTS = (  
    "e2_async.yaml",  
    "e3_nlos.yaml",  
    "e9_dual_degradation.yaml",  
)  
_SAFE_SCORING_EXPERIMENT = "e0_safe_mode.yaml"  
_SAFE_P95_RELATIVE_TOLERANCE = 0.05  
_EMBEDDED_RUNTIME_PROXY_PROFILES = (  
    {"name": "arm_a78_proxy", "latency_scale": 4.0, "ram_limit_mb": 512.0, "params_limit": 5_000_000.0},  
    {"name": "arm_a55_proxy", "latency_scale": 10.0, "ram_limit_mb": 256.0, "params_limit": 2_000_000.0},  
    {"name": "m7_microcontroller_proxy", "latency_scale": 250.0, "ram_limit_mb": 8.0, "params_limit": 200_000.0},  
)  
_CROSS_FACTORIAL_CONFIG = "e12_cross_factorial_interactions.yaml"  
_SEARCH_COMMON_FINGERPRINT_PATHS = (  
    "scripts/20_run_paper_experiments.py",  
    "src/liquidloc/pipelines/train_pipeline.py",  
    "src/liquidloc/pipelines/core_pipeline.py",  
    "src/liquidloc/pipelines/eval_pipeline.py",  
    "src/liquidloc/metrics/metric_runner.py",  
    "src/liquidloc/metrics/tail_metrics.py",  
    "src/liquidloc/protocol/metric_schema.py",  
    "configs/base/metrics.yaml",  
)  
_SEARCH_MODEL_FINGERPRINT_PATHS = {  
    "lstm_ekf": (  
        "configs/models/lstm_ekf.yaml",  
        "src/liquidloc/models/lstm/network.py",  
        "src/liquidloc/models/lstm/trainer.py",  
        "src/liquidloc/factories/model_factory.py",  
    ),  
    "liquid_ekf": (  
        "configs/models/liquid_ekf.yaml",  
        "src/liquidloc/models/liquid/cell.py",  
        "src/liquidloc/models/liquid/network.py",  
        "src/liquidloc/models/liquid/trainer.py",  
        "src/liquidloc/factories/model_factory.py",  
    ),  
}  
_SEARCH_ESTIMATOR_FINGERPRINT_PATHS = {  
    "ekf": (  
        "src/liquidloc/factories/estimator_factory.py",  
        "src/liquidloc/estimators/ekf_core.py",  
        "src/liquidloc/estimators/predict_step.py",  
        "src/liquidloc/estimators/uwb_update_step.py",  
        "src/liquidloc/estimators/vision_update_step.py",  
    ),  
    "robust_ekf": (  
        "src/liquidloc/factories/estimator_factory.py",  
        "src/liquidloc/estimators/ekf_core.py",  
        "src/liquidloc/estimators/robust_ekf_core.py",  
        "src/liquidloc/estimators/predict_step.py",  
        "src/liquidloc/estimators/uwb_update_step.py",  
        "src/liquidloc/estimators/vision_update_step.py",  
    ),  
    "fgo": (  
        "src/liquidloc/factories/estimator_factory.py",  
        "src/liquidloc/estimators/fgo_core.py",  
    ),  
}  

def _write_json(path: Path, payload: Any) -> str:
    """Write JSON atomically and return the final path string."""
    path.parent.mkdir(parents=True, exist_ok=True)
    from liquidloc.common.io_utils import dumps_json_text
    text = dumps_json_text(payload)
    tmp_path = path.parent / f".{path.name}.tmp"
    try:
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)
    except OSError:
        path.write_text(text, encoding="utf-8")
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
    return str(path)


def _read_json(path: Path) -> Any:
    "Read JSON from disk and return the decoded payload."
    from liquidloc.common.io_utils import read_json
    return read_json(path)


def _resolve_prediction_bundles(stage_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Rebuild prediction bundles from on-disk artifacts when metadata keeps only indexes."""
    bundles = list(stage_metadata.get("prediction_bundles") or [])
    if bundles:
        return bundles
    prediction_index = list(stage_metadata.get("prediction_index") or [])
    if not prediction_index:
        return []
    resolved_bundles: list[dict[str, Any]] = []
    for entry in prediction_index:
        prediction_path = entry.get("prediction_path")
        if not prediction_path:
            continue
        resolved_bundles.append(dict(_read_json(Path(prediction_path)) or {}))
    return resolved_bundles

@lru_cache(maxsize=1)
def _resolve_paper_training_budget() -> dict[str, Any]:
    """Resolve the default paper-grade training budget from model YAML files."""
    lstm_cfg = _load_yaml_config_cached(str((ROOT / "configs" / "models" / "lstm_ekf.yaml").resolve()))
    liquid_cfg = _load_yaml_config_cached(str((ROOT / "configs" / "models" / "liquid_ekf.yaml").resolve()))
    lstm_train_cfg = dict(lstm_cfg.get("train") or {})
    liquid_train_cfg = dict(liquid_cfg.get("train") or {})
    batch_size = int(lstm_train_cfg.get("batch_size") or 0)
    eval_batch_size = int(lstm_train_cfg.get("eval_batch_size") or 0)
    liquid_batch_size = int(liquid_train_cfg.get("batch_size") or 0)
    liquid_eval_batch_size = int(liquid_train_cfg.get("eval_batch_size") or 0)
    if batch_size <= 0 or eval_batch_size <= 0:
        raise ValueError("lstm_ekf.yaml must declare positive paper-grade batch_size and eval_batch_size")
    if liquid_batch_size <= 0 or liquid_eval_batch_size <= 0:
        raise ValueError("liquid_ekf.yaml must declare positive paper-grade batch_size and eval_batch_size")
    if batch_size != liquid_batch_size or eval_batch_size != liquid_eval_batch_size:
        raise ValueError("paper-grade neural defaults must share the same batch_size and eval_batch_size")
    lstm_epochs = int(lstm_train_cfg.get("epochs") or 0)
    liquid_epochs = int(liquid_train_cfg.get("epochs") or 0)
    if lstm_epochs <= 0:
        raise ValueError("lstm_ekf.yaml must declare positive paper-grade epochs")
    if liquid_epochs <= 0:
        raise ValueError("liquid_ekf.yaml must declare positive paper-grade epochs")
    return {
        "lstm_epochs": lstm_epochs,
        "liquid_epochs": liquid_epochs,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "budget_source": "profile_default",
        "paper_grade": True,
        "diagnostic_only": False,
    }


def _resolve_neural_search_surface(profile_name: str | None) -> dict[str, Any]:
    """Resolve the neural-search surface for the requested profile."""
    normalized_profile = str(profile_name or "paper").strip().lower()
  
    if normalized_profile == "paper":  
        paper_training_budget = _resolve_paper_training_budget()
        return {  
            "profile_name": "paper",  
            "neural_grid": copy.deepcopy(_FAIR_NEURAL_GRID),  
            "liquid_robustness_profiles": copy.deepcopy(_LIQUID_ROBUSTNESS_PROFILES),
            "top_k_multiseed": 2,  
            "multiseed_values": [0, 1],  
            "epoch_candidate_stride": None,  
            "checkpoint_selection_mode": "trainer_validation_best",  
            "training_budget": dict(paper_training_budget),
            "notes": [  
                "paper profile uses the full fairness search grid",  
                "paper profile derives the default training budget directly from the model YAML defaults",  
                "paper profile uses the trainer-selected validation-best checkpoint instead of a second downstream rescoring layer",  
            ],  
        }  
    if normalized_profile == "small_eval":  
        return {  
            "profile_name": "small_eval",  
            "neural_grid": copy.deepcopy(_SMALL_EVAL_NEURAL_GRID),  
            "liquid_robustness_profiles": copy.deepcopy(_SMALL_EVAL_LIQUID_ROBUSTNESS_PROFILES),  
            "top_k_multiseed": int(_SMALL_EVAL_NEURAL_TOP_K_MULTI_SEED),  
            "multiseed_values": list(_SMALL_EVAL_NEURAL_MULTI_SEEDS),  
            "epoch_candidate_stride": _SMALL_EVAL_EPOCH_CANDIDATE_STRIDE,  
            "checkpoint_selection_mode": "downstream_val_rescore",  
            "training_budget": {
                "lstm_epochs": 80,
                "liquid_epochs": 80,
                "batch_size": 64,
                "eval_batch_size": 128,
                "budget_source": "profile_default",
                "paper_grade": False,
                "diagnostic_only": True,
            },
            "notes": [  
                "small_eval profile is for incomplete sim-only trend checks and is not valid for final paper claims",  
                "small_eval is diagnostic_only and does not represent the paper-grade training budget",  
                "small_eval narrows the neural grid and reduces multiseed expansion while keeping the same batch-statistics regime",  
            ],  
        }  
    if normalized_profile == "medium_eval":  
        return {  
            "profile_name": "medium_eval",  
            "neural_grid": copy.deepcopy(_MEDIUM_EVAL_NEURAL_GRID),  
            "liquid_robustness_profiles": copy.deepcopy(_MEDIUM_EVAL_LIQUID_ROBUSTNESS_PROFILES),  
            "top_k_multiseed": int(_MEDIUM_EVAL_NEURAL_TOP_K_MULTI_SEED),  
            "multiseed_values": list(_MEDIUM_EVAL_NEURAL_MULTI_SEEDS),  
            "epoch_candidate_stride": _MEDIUM_EVAL_EPOCH_CANDIDATE_STRIDE,  
            "checkpoint_selection_mode": "downstream_val_rescore",  
            "training_budget": {
                "lstm_epochs": 120,
                "liquid_epochs": 120,
                "batch_size": 64,
                "eval_batch_size": 128,
                "budget_source": "profile_default",
                "paper_grade": False,
                "diagnostic_only": True,
            },
            "notes": [  
                "medium_eval profile is for incomplete sim-only trend checks and is not valid for final paper claims",  
                "medium_eval is diagnostic_only and does not represent the paper-grade training budget",  
                "medium_eval restores both neural capacity bands, both liquid robustness profiles, and a two-seed top-k check",  
            ],  
        }  
    raise ValueError(f"unsupported neural search profile '{normalized_profile}'")  


def _resolve_training_budget_overrides(
    neural_search_surface: dict[str, Any],
    *,
    lstm_epochs: int | None,
    liquid_epochs: int | None,
    batch_size: int | None,
    eval_batch_size: int | None,
) -> dict[str, Any]:
    """Merge requested overrides into the resolved training budget."""
    base_budget = dict(neural_search_surface.get("training_budget") or {})
    if not base_budget:
        raise ValueError("neural_search_surface must declare training_budget")
    has_cli_override = any(
        value is not None
        for value in (lstm_epochs, liquid_epochs, batch_size, eval_batch_size)
    )
    resolved = {
        "lstm_epochs": int(base_budget["lstm_epochs"]),
        "liquid_epochs": int(base_budget["liquid_epochs"]),
        "batch_size": int(base_budget["batch_size"]),
        "eval_batch_size": int(base_budget["eval_batch_size"]),
        "paper_grade": bool(base_budget.get("paper_grade", False)) and not has_cli_override,
        "diagnostic_only": bool(base_budget.get("diagnostic_only", False)),
        "budget_source": "profile_default",
    }
    if lstm_epochs is not None:
        resolved["lstm_epochs"] = int(lstm_epochs)
        resolved["budget_source"] = "cli_override"
    if liquid_epochs is not None:
        resolved["liquid_epochs"] = int(liquid_epochs)
        resolved["budget_source"] = "cli_override"
    if batch_size is not None:
        resolved["batch_size"] = int(batch_size)
        resolved["budget_source"] = "cli_override"
    if eval_batch_size is not None:
        resolved["eval_batch_size"] = int(eval_batch_size)
        resolved["budget_source"] = "cli_override"
    return resolved


@lru_cache(maxsize=None)
def _load_yaml_config_cached(config_path: str) -> dict[str, Any]:
    """Load a YAML config file from disk."""
    return load_yaml_config(config_path)


@lru_cache(maxsize=None)
def _load_experiment_cfg_cached(config_name: str) -> dict[str, Any]:
    """Load an experiment config by name and return a deep copy."""
    experiment_cfg = copy.deepcopy(
        _load_yaml_config_cached(str((ROOT / "configs" / "experiments" / config_name).resolve()))
    )
    experiment_cfg["mode"] = "full"
    return experiment_cfg


@lru_cache(maxsize=None)
def _sample_scenes_cached(config_name: str) -> tuple[dict[str, Any], ...]:
    """Return the configured scene sample list for a given experiment."""
    return tuple(sample_scenes(_load_experiment_cfg_cached(config_name)))


@lru_cache(maxsize=None)
def _load_prepare_manifest_cached(prepare_root: str) -> dict[str, Any]:
    """莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳鹿茅聴麓莽莽潞戮氓陆聮忙聼聼茅聴聜氓炉赂莽禄戮忙聝搂忙鹿卤茅聨虏忙聬聛忙聜搂茅聧芦忙驴聢莽聭篓莽录聜盲陆潞氓搂鲁茅聧聴忙聤陆氓录禄茅聬聰氓聟录芒聮聮茅聬聨莽聰碌忙驴隆盲禄聥忙垄潞莽禄聥忙氓聞聧茅聳赂氓漏聝莽鹿聢氓炉氓录麓茅聬聰猫聶鹿茅聬聼忙禄聝莽搂露茅聢搂氓聡鹿茅聧聯莽聠录氓鹿聝茅聧篓茅聬聨忙露聶茅聧聵盲禄聥忙垄潞莽录聛忙篓聫茅聭赂茅聺聸莽路聻茅聳赂忙聸篓氓聨陆茅聧聽忙聞掳忙聼隆忙戮露氓漏聞盲禄垄茅聳潞氓聴聶氓聳聯茅聢?prepare 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳鹿茅聴麓莽莽潞戮氓陆聮忙聼聼茅聴聜氓炉赂茅聳禄茫聜聨氓聶拢茅聫聦莽聠录氓鹿聠茅聬聴氓聽茅聴?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓聙驴茅聳驴忙聸聻氓聙鹿茅聧聨忙聢聽忙聠赂茅聬聴氓聽聺莽卢聮莽录聛芒聜戮露氓卤赂氓聛聯茅聧聫氓盲潞戮茅聴聛茅聙聻氓卤陆氓垄麓茅聳赂芒聰戮氓聛聯氓娄聻茫聜聠氓赂聤茅聬聮茂鹿聜忙聝聝茅聨麓茂赂陆莽聟聲忙驴隆茫聜聞忙碌聡忙戮鹿忙漏聙忙聼聧莽聭聶氓聥聛麓茅聳鹿忙聶聺茅聢搂忙鹿卤忙驴聮庐脣聣氓漏碌氓聽垄莽拢录莽录聜盲陆碌氓聻鹿茅陋聻忙聞掳忙聼聸莽聙拢氓麓聦忙驴漏聞莽虏聨路氓聭麓氓麓篓忙驴聽氓聠篓氓聣碌忙驴聽莽聰碌氓聛聽茅隆聲忙禄聟莽虏禄茅聨潞忙聸聵盲潞鹿忙驴聽氓漏聜莽聣聤茅聢路?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    prepare_root : str
        prepare 氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄忙驴聻忙聢聶氓聻潞茅聧聥茫聜聣忙聼聲忙驴聻脩聝忙聡聨茅聫聠忙楼聟忙垄潞莽禄聥猫路篓莽聛卤茅聳赂氓卢篓聭茅隆聰氓驴聯芒聰聭莽聙拢氓聨碌莽录聜盲陆赂莽聙碌脩聡氓麓碌茅聧楼茫聞娄氓聫聫氓漏碌莽聤虏氓聲炉茅聤聡忙驴聢芒聜忙麓娄茅聧聥氓漏聜莽聲路茅聬聰莽垄卤忙聲聡氓猫路潞莽聣聠茅聬聫莽麓聮莽录聛忙莽庐聻茅聳赂芒聰戮氓聛聯氓娄聻茫聜聠氓赂聮莽聙職忙露碌茅聢搂忙垄潞莽聮聡猫聦卢莽庐鲁莽录聛氓聻掳茅隆陋氓聠漏莽莽聙拢莽楼娄茅聴聤忙麓娄氓篓虏忙聢聺芒聳陆茅氓聽聼氓聛拢氓篓麓茂陆聢氓搂陇茅聬聺忙聼拢茅茫聢垄莽忙驴漏聞莽虏聨路氓聭麓氓麓篓忙驴聽氓聠篓氓聣碌忙驴聽莽聰碌氓聛聻氓篓录忙卢聭芒聜掳录氓聳聳氓聧卤茅聳鹿莽聣聢忙戮聵茅聧聴忙聤陆忙聼聲猫鹿聡氓篓戮氓聛聯茅聧聫氓聭聸聨茅聴聜氓聜聶莽陇聛茅聨虏芒聞聝莽卢聼氓娄陇氓聭颅氓聻聜忙陇陇氓驴聯莽聣聠莽潞戮氓忙聼聼茅聨鹿忙聞碌氓職聶茅聳赂茫聞楼氓聙鹿茅聤聡氓聥鹿聡茅聧聯盲陆鹿氓聛搂茅聴聛氓聭聤氓聼聤莽禄禄忙驴聥氓录禄茅聰聺氓聴聵氓聞聼茅聳碌氓驴聲氓搂碌茅聨炉忙露垄忙垄潞茅聧聸氓漏聝芒聮聬茅聳赂茫聞楼莽聛聺茅聬拢氓鲁掳脙潞茅隆聫氓聥颅氓聤?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    dict[str, Any]
        prepare manifest 茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼忙聲聥莽聭聹氓驴聸氓聼聙茅隆聮盲陆潞芒聳聯茅聳潞氓聭鹿聞茅聧聮忙聨聭芒聳聨茅聨戮氓麓聡氓漏碌茂录聞氓聛聸茅聬篓莽聝陆氓麓聭茅聨戮猫炉虏莽忙陇陇忙聴聜氓聨搂莽禄聬忙露垄忙垄潞茅聧聺氓聥碌忙搂鹿茅聳赂茫聞篓氓茅隆陇氓聜聻忙垄禄茅聧聦忙卢聭莽陋聰茅聳鲁脩聦氓聣聸茅聧聥忙露職芒聜掳录氓聶拢氓庐聲?
    """
    return load_prepare_manifest(prepare_root)


@lru_cache(maxsize=None)
def _load_scoring_inputs_cached(
    prepare_root: str,
    raw_root: str,
    seq_ids: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳鹿茅聴麓莽莽潞戮氓陆聮忙聼聼茅聴聜氓炉赂莽禄戮忙聝搂忙鹿卤茅聨虏忙聬聛忙聜搂茅聧芦忙驴聢莽聭篓莽录聜盲陆潞氓搂鲁茅聧聴忙聤陆氓录禄茅聬聰氓聟录芒聮聮茅聬聨莽聰碌忙驴隆盲禄聥忙垄潞莽禄聥忙氓聞聧茅聳赂氓漏聝莽鹿聢氓炉氓录麓茅聬聰猫聶鹿茅聴聛莽禄聵氓聤娄茅聧聯忙掳露忙聜碌茅聫聝氓聜聹莽拢陆氓篓麓茂陆聠氓陆聝忙碌聹茅鹿聝忙垄潞茅聧聸氓漏聝氓陇聞茅聳赂茫聞漏盲潞露茅聨?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮莽聲聦忙聠赂氓庐楼氓聽垄盲潞卤茅聴聜盲陆赂忙聬芦茅聧聤氓陆聮氓篓聠氓驴聥芒聜聨聯莽鹿聺氓氓聨碌茅聴聛莽禄聵氓聻露猫聯卤忙陇陇茫聝楼忙茅聢鹿忙聢娄茅聮聵氓忙聻聶茅聴聛氓聯聞忙聹卢氓篓虏忙忙聧垄氓漏碌氓漏聞莽聳聙茅聳鹿氓聻庐氓聲聧聽忙漏聙忙垄禄茅聧聦忙掳聜卢忙聬聛氓麓聬茅聨录盲陆鹿忙聶聺茅聳驴忙聸聻氓聙赂氓聸麓茅聴聛氓聹颅猫聶聨茅聧聽忙楼聟忙聜隆茅聫聡茫聢隆氓聙碌茅聳驴忙驴聠茅忙碌聹忙驴聥忙聜聵猫鹿聡氓潞隆氓碌聠茅聳潞氓虏聙氓聳聨茅聬拢茂驴聽忙聜搂茅聧驴聡莽聲禄茅聴聜盲陆陆茅聧聽忙楼聝氓聢聮氓庐聙氓聠潞陆茅聴聜盲陆鹿氓聞鲁莽禄禄忙聨聴氓鹿聤茅聨戮氓颅聵氓聫聫茅聴聛莽搂碌茅聢路忙聨聴忙聼聸莽聛聻氓聣搂忙聡聠茅聳赂氓驴聯忙聹卢茅聤聡氓聥炉茅聬聬茅聺聸芒聟碌茅聴聛猫陆掳莽茅聧聼忙聮庐氓录聧聬莽拢录忙驴庐忙漏聠氓聣聶忙碌聹氓聽聲忙聼拢茅聬聰氓聯楼莽聼聥忙驴隆氓聲麓茅聨录麓莽聮潞茫聞搂忙聺赂茅聴聛氓聯聞氓聙鹿茅忙卢聯氓麓碌茅聤聢氓芒聮聭茅聳赂忙驴聠氓氓娄虏氓陆聮忙聼聸茅聤聤忙陆聬忙陇陇茫聞拢氓聞碌氓庐聲氓聽聲忙碌拢氓聫聣氓芦录茅聴聜氓聜職氓聙赂茅聬聴氓漏聝莽卢聼氓娄陇芒聜卢茅隆漏聟茅聳鲁猫氓聻驴茅聧篓茅鹿聝忙聬鹿茅隆聬茫聢隆莽聛聨茅聴聜盲陆陆茅聧聽忙聮禄氓麓聲茅聬垄莽猫聯聼茅聳赂芒聞聝茅聧職氓卢聼聸茅聢漏氓聠陋忙聡聝忙莽聜虏莽搂鹿氓搂聤氓氓麓聮氓猫聜漏氓聫聲茅聴聛氓聭聤忙聦禄忙路聡忙麓娄碌茅聨掳忙聺驴莽陇聝忙陇陇忙聴聡氓聛聬茅隆娄芒聰聭茅隆聰莽颅聥莽聞戮茅聳赂芒聲聜莽聣聽茅聨聙鲁芒聳聨茅聨戮忙聺聭盲禄聴聛忙聬聻氓聞卤氓篓虏茫聜聣氓鹿聤茅聨掳莽聣聢氓聞聤茅聳赂氓聞虏茅聧聤莽聰赂忙聜路氓篓聠氓驴聯莽录聛芒聜戮露氓卢氓漏垄猫路潞氓篓虏忙聮庐忙聼聸茅聢鹿氓聻聤聹莽聙碌忙聺驴氓芦禄禄忙驴聡忙聝聰茅聤聫莽聤虏氓聨聻茅聴聜氓聜聶莽莽录聧氓漏聡茫聙聜茅聰聲氓聜聸氓聙驴茅聳驴忙聸聻氓聙鹿莽聙職莽聟聨茅聭赂茅聺聸氓搂聢茅聳鲁茅聰聥氓聻驴茅聫聦茫聜聠忙聟篓茅聨掳氓聛聯茅聫聡莽聜碌忙聭楼氓漏碌莽聤碌忙聲赂茅聧聥忙露垄忙聝聡忙碌录忙聺聭莽拢鹿忙驴隆茫聜聟忙聡聨茅聬聮氓聻芦茫聛聞茅聨潞氓露聠莽楼掳氓漏垄猫茂鹿垄茅聫聦忙露垄氓娄赂茅聤聣茫聝垄氓聛聯茅聧搂忙楼聝氓碌聛茅聳潞氓露聨莽聛聰忙聬麓忙聲聠茅聳鲁脩聦氓聣聼茅聨庐忙漏聙茅聧隆忙卢聭茂录聺茅聬聨氓鹿驴氓聞卤茅聨鲁氓驴聸氓麓掳氓芒聜聫聦芒聙鲁莽聲聺茅聢搂氓麓掳茅聫聧氓碌聛茅聳赂芒聞聝氓聡鹿氓娄虏氓漏職忙垄潞茅聧聺氓聥卢氓篓麓忙禄聨氓聣聼茅聧漏芒聜陇陇忙聨聧莽路隆茅聧聼氓驴聸忙聼聸茅聬聵忙聞聛忙庐聲莽录聛莽禄聵莽篓聮莽禄禄忙驴聥氓麓露猫陇聭茅聨陆?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    prepare_root : str
        prepare 氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄忙驴聻忙聢聶氓聻潞茅聧聥茫聜聣忙聼聲忙驴聻脩聝忙聡聨茅聫聠忙楼聟忙垄潞莽禄聥猫路篓莽聛卤茅聳赂氓卢篓聭茅隆聰氓驴聯芒聰聭莽聙拢氓聨碌莽录聜盲陆赂莽聙碌脩聡氓麓碌茅聧楼茫聞娄氓聫聫氓漏碌莽聤虏氓聲炉茅聤聡忙驴聢芒聜忙麓娄茅聧聥氓漏聜莽聲路茅聬聰莽垄卤忙聲聡氓猫路潞莽聣聠茅聬聫莽麓聮莽录聛忙莽庐聻茅聳赂芒聰戮氓聛聯氓娄聻茫聜聠氓赂聮莽聙職忙露碌茅聢搂忙垄潞莽聮聡猫聦卢莽庐鲁莽录聛氓聻掳茅隆陋氓聠漏莽莽聙拢莽楼娄茅聴聤忙麓娄氓篓虏忙聢聺芒聳陆茅氓聽聼氓聛拢氓篓麓茂陆聢氓搂陇茅聬聺忙聼拢茅茫聢垄莽忙驴漏聞莽虏聨路氓聭麓氓麓篓忙驴聽氓聠篓氓聣碌忙驴聽莽聰碌氓聛聻氓篓录忙卢聭芒聜掳录氓聳聳氓聧卤茅聳鹿莽聣聢忙戮聵茅聧聴忙聤陆忙聼聲猫鹿聡氓篓戮氓聛聯茅聧聫氓聭聸聨茅聴聜氓聜聶莽陇聛茅聨虏芒聞聝莽卢聼氓娄陇氓聭颅氓聻聜忙陇陇氓驴聯莽聣聠莽潞戮氓忙聼聼茅聨鹿忙聞碌氓職聶茅聳赂茫聞楼氓聙鹿茅聤聡氓聥鹿聡茅聧聯盲陆鹿氓聛搂茅聴聛氓聭聤氓聼聤莽禄禄忙驴聥氓录禄茅聰聺氓聴聵氓聞聼茅聳碌氓驴聲氓搂碌茅聨炉忙露垄忙垄潞茅聧聸氓漏聝芒聮聬茅聳赂茫聞楼莽聛聺茅聬拢氓鲁掳脙潞茅隆聫氓聥颅氓聤?
    raw_root : str
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏氓庐聲氓聸搂氓鹿聢?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬氓庐聞忙聡聤氓陆聮氓麓露猫陇聹氓篓麓忙聽聼聲忙驴聻莽聜卢忙芦聠茅聳赂茫聜聟氓茅聫聦茫聜聡莽虏聶莽聮潞茫聞陇莽虏碌氓漏碌莽聜虏忙聡聬忙驴庐氓聻漏卤忙驴聽氓聸搂脦搂茅聳赂氓卤戮莽聼戮鹿忙聸垄忙垄禄忙碌拢茅盲赂虏莽录聛猫鹿虏聶茅聨篓潞忙聺聢芒聲聝莽聲露茅聴聛莽禄聵忙垄禄茅聧聵脩聡忙聜聻茅聧篓盲潞聹茅聳鹿氓聯聞忙拢聴忙碌聹忙聝搂茅聤聢氓聴聵莽漏驴莽录聛忙聫聮氓垄陆莽禄庐氓卢鹿聮茅聫聜氓聯聞盲潞戮茅聳驴忙驴聠莽掳录莽禄篓猫聜漏芒聜忙聦赂莽禄聣忙聮聙聰茅聧聬脦搂茅聳赂芒聞聝脙陋茅聫聠忙忙垄潞茅聧聺茫聝楼脨拢茅聳鹿莽聭掳忙麓颅茅陋聻氓聠漏氓炉录茅聫聣忙聡鲁茫聞漏忙聼聧茅聧聺氓聥聜卢氓聭聤莽茅隆聯莽聜漏忙聰陆茅聳禄忙篓驴氓茫聝漏忙聼聼茅聧聬忙聼楼氓聫聞茅聳碌氓聴聴氓聙驴氓庐聲忙楼聟忙聡聳茅聢潞氓聠戮氓聲聨茅聴聜盲陆赂忙聠隆茅聬聼茫聝漏脣聡忙聺聢芒聲聝盲赂聧茅聳禄忙聞聞陇茅聬聯忙卢聬茫聛聞茅聫聧茫聞娄莽職聭茅聳禄忙聨聭忙聠隆茅隆篓茫聝漏忙聜聶茅聧聶氓忙拢楼氓茫聛卤氓聜聹莽陋聰茅聳潞氓卤戮莽聸炉忙驴隆氓聽聲氓麓卤茅聢潞氓聜聹忙聺鹿茅聴聜盲陆赂忙聬聫聢忙聝聞茅忙掳录氓職聲茅聳潞氓氓職娄茅聫聝芒聜聨路氓聭聠麓茅聧聲忙職聵茅聴?
    seq_ids : tuple[str, ...]
        茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聟忙職娄茅聳禄茫聢垄莽禄聬忙楼聟忙聼聼茅聳鹿聙氓聤聮忙驴聻氓路潞聹莽禄聽茂陆聣忙聼拢氓娄炉氓聟录忙職漏茅聳禄忙聨聭氓聬聥莽禄禄忙露聶忙陆聫茅聧聯脩聟脨職氓茫聜聨氓聛聞茅隆娄茅聺聸莽聲路茅聰聺氓聽垄莽聳聙忙驴聻忙聢聻莽聭搂茅聧聵盲禄聥忙垄潞猫陇掳忙聨聭盲潞掳茅聳赂忙聮麓莽聸炉茅陋聻忙忙聜漏氓篓麓氓卢聳聙氓娄聻茫聜聠忙垄禄茅聫聟忙聝聦茅聨潞忙楼聛莽聟聸莽聙鹿芒聜卢莽聙拢脨聹茅聴聛猫陆掳莽聞娄茅聧聰忙卢聫莽聲路茅聤聤芒聲聤忙聲聤茅聳录忙聛聮氓聞卤茅盲录麓忙垄禄茅聧聦忙卢聬莽录聛茅鲁聳氓聳職莽禄卤莽聜麓忙聥聟茅聧娄忙麓赂茅聴聛氓聣虏氓聛聟氓篓虏忙禄聝莽聲碌忙赂職芒聜卢茅聫聦芒聙鲁忙聬麓芒聙鲁莽莽录聜盲陆潞氓芦聫茅聧楼茫聞搂氓聨碌茅聳禄氓潞拢忙聲赂茅隆颅氓聽聺忙職漏忙驴隆莽聜陋氓聙聳茅聫聦茫聞楼脙潞茅莽聝聠忙陆聳?ID 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛氓露聣氓麓聼茅盲陆鹿忙驴聽氓戮路忙聺驴芒聜聸垄氓鹿聤氓庐聙氓聥聞碌忙陇陇忙聨聠氓聛聯茅聳赂莽聡聛氓聻掳茅隆聲茫聝漏氓麓聼茅忙掳卤莽聳聙氓娄聻茫聜聠氓聻聼茅聫聜氓聜聻氓麓聲茅聬垄莽猫聯聼忙驴聻忙聢聶氓聻鹿莽禄聽忙露聶茫聛聞茅聨戮猫路潞忙聻聰莽聭聹猫聢卢氓聤聭氓氓漏職氓麓聮氓篓聠忙聢颅忙搂聴聛莽隆聟氓搂陇莽禄潞拢氓聢聮忙驴聽忙掳莽录聜盲陆鹿脙陋忙碌聽氓聯聞茅聤聢氓聴聶氓垄篓聯忙掳聢禄忙聮鲁脦聼茅聭虏芒聲聝茂录聹茅聳潞氓莽聣聞茅聧聰氓露聡盲潞赂忙碌录氓颅聵莽聟聶莽聙鹿氓聥颅氓垄聴茅聧拢莽聲聦莽麓聮茅隆聰莽聲聦忙聡聟茅聳鹿莽聭掳氓忙驴隆忙颅聦茅聳驴忙露聶莽虏聯茅聫聦芒聞聝氓戮聞忙驴聽氓聥芦茅聴聛氓聭聤忙聼篓氓篓麓茅拢聨氓聢聮忙驴聽忙掳颅氓聙路莽禄戮莽聣聢脙碌聹?茅聴聜氓聜職氓聙赂茅聧聤莽聝陆忙聡聴茅聧聸聵芒聜陆鹿莽鹿聺茅聧聬氓陆戮茅聴聛氓聣虏氓聛聠茅聧聽忙聮禄茫聙聟茅聧聶氓陇聢盲潞聹茅聳鹿莽聝聵氓聻碌茅芒聲聟忙聼隆莽聙拢芒聲聝茅聴聡氓忙聼拢氓篓聠氓驴聰氓聶陆茅聳禄茂陆聟莽聛職茅聨卤茫聞娄莽鹿聺茅聧聸忙聝聻茅聴聛茅聙聻氓卤陆氓垄聳禄氓驴聰氓虏赂茅陋聻氓聠庐氓麓聮茅聬聬忙聺聭芒聜卢氓聻鹿聸氓庐赂莽掳录茅聳鲁茅聰聥氓赂聮茅聹聣茅聳驴忙驴聠忙聥聥氓莽聠录忙聼聧猫陇聹茅聧聫忙漏聙氓麓聭茅聨戮莽禄聵芒聮聭?茅聴聜氓聜職氓聙路氓篓麓氓聸茅隆篓茫聝漏氓鹿聳忙碌拢莽聭掳芒聜卢猫聢碌忙聟篓氓搂聴氓聴聴氓垄禄忙驴聻氓聟录莽聣聽茅聫聦茫聜聠忙聟篓茅聨掳氓聛聯茅聫聧莽露聳茅聳潞氓聸楼氓聻潞茅聬聯忙卢聯忙聼聼莽聭聶氓聥搂娄茅聳赂茫聜聠莽聭搂莽禄卤忙聨聴忙聜漏茅聧聺氓聥芦忙聝聻茅聴聛茅聙聻氓卤陆氓垄虏忙陇陇莽聟陇茅聳潞氓露聨莽聛聬氓篓虏忙聶聺茅聳赂氓卤戮忙掳篓茅聬陇氓聭鹿聺茅聴聢忙露聵芒聜聬聛氓鹿聣茅聰聼莽聤虏莽聟聲茅聳鹿猫戮戮茅聶聞茅聬聯忙卢聭莽麓聮莽聙拢茅聳鹿莽聶赂莽露聛茅聫聦氓露聣忙聦聤忙戮露忙篓禄脨搂茅聴聛茅聺聸忙麓陇莽聙職盲录麓氓录聧聬莽聞碌忙陇陇忙聨聭氓茅聧?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]
        氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄茅聳禄忙聞聬芦莽禄聽盲录麓忙聼陇忙驴庐忙聞聜卢忙聸垄氓录赂茅聧聭漏氓聛聯茅陋聻氓聻聤職忙驴聻猫莽庐聭茅聬拢?氓漏?prepare_manifest茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮莽聲聦忙聠赂氓庐楼氓聽垄盲潞卤茅聴聜盲陆赂忙聬芦茅聧聼忙聜搂茅聧隆忙露聵氓聻聜氓篓拢氓聸露忙芦聽ared_events茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮莽聲聦忙聠赂氓庐楼氓聽垄盲潞卤茅聴聜盲陆赂忙聬芦茅聧聼氓麓聬茅聬垄莽莽拢禄茅聳鹿忙聧聬茅聨鹿猫聢碌茫聛聞茅聨戮猫路潞盲禄搂忙陇陇忙聳驴氓碌聙und_truth茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮莽聲聦忙聠赂氓庐楼氓聽垄盲潞卤茅聴聜盲陆赂忙聬芦茅聧聼莽聤禄氓麓聭茅聨戮氓聻聸氓聛聯茅聧篓莽路虏茅聬聨忙掳芦氓聧鲁茅陋聻氓聠聜聶茅聨聨聠氓聛聯茅聢搂莽麓聮茅茫聜聡氓陆聭e_report茅聴?
    """
    prepare_manifest = _load_prepare_manifest_cached(prepare_root)
    seq_id_list = list(seq_ids)
    return (
        prepare_manifest,
        load_prepared_events_by_seq_id(prepare_root, seq_id_list),
        load_ground_truth_by_seq_id(raw_root, seq_id_list),
        load_source_report_by_seq_id(
            raw_root,
            prepare_manifest,
            seq_id_list,
            default_source="sim_prepare_bridge",
        ),
    )


def _resolve_experiment_fingerprint_paths(config_names: tuple[str, ...]) -> tuple[str, ...]:
    """茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聫茅聢搂莽聞娄莽禄禄忙驴聥氓麓聮氓娄陇盲陆潞盲潞聶忙驴聽莽聰碌氓聛聻茅聧篓氓麓鹿氓篓虏莽聳戮忙驴聽茅聺聸茅聬聯氓驴聸忙聼聸茅茂陆聡莽忙驴隆忙聫聮莽露聤茅聫聦忙露職芒聜卢茂陆聟氓聬聙茅聳聼隆莽聛聻氓聣搂忙麓麓莽聙碌忙聦鲁茅聨聜聠忙聥聟茅聧聸氓漏聝莽聫芦茅聴聜氓聜職氓聙路氓篓麓氓聸茅聨鹿茫聢陇氓鹿聮氓娄陇芒聜潞戮氓漏職忙聼拢氓娄聦聨莽聫聳赂忙卢聫莽鹿聵茅聫聦忙聸垄氓麓录氓漏碌忙聞娄莽录聛莽聜聞鲁莽录聧氓漏聡氓录禄茅聬聰莽聟聨莽庐職茅聳禄忙莽聣聹氓娄芦氓聥麓聯?茅聴聜氓聜聶忙碌聡茅脩聝茅聰聲氓聜娄莽露聳氓猫路潞芒聤聲茅聧聺氓聥碌忙陆篓茅聳鲁脩聦氓聣聼茅陋聻氓聠聲聯茅聬聵猫聶芦忙芦垄茅聴聛莽禄聵茂鹿垄氓篓录脩聟莽虏聤茅聰聲氓聜聸莽聟聼茅聨录忙聬聻鲁莽禄聣莽聰赂忙聼聸茅聬聵氓麓聡氓垄卤莽录?茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙碌拢茅聺漏猫聙驴茅聴聛莽搂麓茅聧聦茫聞搂忙聠赂茅聫聝氓聽聺莽庐聳忙驴聻氓路潞聹忙聝聼茅聴聛茅聺聸茅聧聽忙楼聝莽麓聻忙聬麓茫聞娄莽聟聼茅聳禄忙篓潞氓录露忙戮露氓聥颅莽麓聮忙碌拢猫搂聞氓掳娄莽聙碌忙聜聜氓炉麓录茅聬聰氓聯聞氓鹿聧茅聴聛氓聯聞茅聬聴忙聮露莽虏禄茅聫聧脩聦莽聙露忙陇陇忙聸聴氓聛聯茅聳潞氓虏聥莽露聛茅陋聻忙漏聙氓搂聵茅聴聜氓聜聶忙碌聡茅隆聲脩聞茅聧聺茫聝搂盲赂聧莽聙鹿茫聝楼莽潞戮氓聺聴忙聼聸氓娄陇氓聠搂莽麓鲁忙驴聻氓路庐盲潞聹忙聝聼茅聴聛氓庐聽忙隆篓莽禄聙盲戮聙氓麓碌茅聨麓茂赂陆芒聮聭莽录聛氓忙鲁聲茅聬聫茅聛聧忙聼聸莽聙拢盲禄卤茅聳鹿芒聜卢忙聳楼忙茅聳碌氓驴楼莽麓聲茅聧聵茅聛聧忙聼拢猫鹿聡忙聸聻盲禄聹氓脩聜莽虏聶茅聤聠氓聜聻氓录禄茅聤聣茂赂戮氓聨陆氓茫聜聠忙聦戮茅聧拢茫聝楼芒聳聯茅聫聡莽聤碌莽拢录莽录聜盲陆鹿氓篓虏氓芒聜忙麓聼忙碌聹氓聽聲莽聧庐茅聧楼忙聲聠茅聳赂氓卤戮莽聭搂莽禄聙氓聥垄禄茅聧聦忙卢聬莽楼掳忙陇陇忙聸聼莽聣聽氓庐聲氓陆聮氓鹿聧茅莽聭掳盲潞戮忙驴氓聨搂茅聫聥茫聜聣忙聼聸?茅聴?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓麓茂陆聢氓篓聟莽聤禄氓鹿聳氓篓拢氓莽禄戮忙聝搂茅聫聦忙聸垄氓麓录氓漏碌氓聸露忙莽录?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    config_names : tuple[str, ...]
        茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼忙聲聥莽聭聹氓潞篓茫聙聝茅聤聣脩聟莽聼聶茅聳赂忙聼聜聜氓聛聯茅聧搂忙楼聛芒聮聭忙陇陇忙聨聠氓聛聯莽录聛氓陇聤忙戮鹿忙聸聼莽鹿聺氓搂聵氓聨陋茅聴聛氓聣虏氓聛聟莽禄禄氓露聟脦拢猫陇掳忙聨聯莽聟聲茅聬聨茂陆聨氓聙聳茅聨麓莽聤鲁氓麓虏忙驴聻忙聢聶氓聻鹿茅陋聻茫聢隆盲驴聤茅隆聳忙驴聬氓職聶茅聴聜氓陇聤莽搂鹿氓搂聤猫聶鹿莽麓聮氓娄炉猫炉虏莽職聭茅聴聛莽茅聨鹿氓聸卢氓录禄茅聰聺氓聴聵氓聞聼茅聳驴忙露聵氓芦聫聠忙露垄忙垄潞莽禄聥氓聺聴莽聙鹿忙聸篓氓聣聸茅聧聶氓聭麓忙垄潞茅聴聢忙碌楼氓聽楼氓录虏氓篓聭茫聢聽莽陇聞茅聳禄忙篓录莽虏炉茅聬聯忙聸聽芒聜聟聣氓聛聯茅聳鲁脩聦氓聣聼氓庐聲忙聢娄忙聜聶茅聬聭忙篓潞盲潞聴氓漏碌莽聜麓氓聻炉茅聧篓忙麓陋忙聜隆茅聫聡茫聢隆氓聙露茅聳禄忙聞陆聝茅聢路忙聴聙茫聛聞茅聰聺氓聴聶氓聫聙茅聳潞氓虏聥莽露聛茅隆垄忙漏聙忙聜陇忙碌拢氓聹職娄莽录聜氓聜聶猫聝露莽禄庐忙聝聞茅忙掳卤忙職娄茅聴聛莽搂麓莽禄聽忙露聶茫聛聞茅聨潞忙聢聺芒聜卢茅拢聨忙氓篓聠忙聮麓莽聟聼茅聨录忙路聙陇氓忙聞鹿聸茅聨禄忙聨聰莽庐禄莽聙鹿忙聸聼忙麓聼茅聫聦氓聴聴茅聧隆忙卢聫氓鹿聢忙驴聽莽聰碌氓聛聻茅聧篓猫路潞茅聤聤芒聲聛莽庐聽茅聳潞氓露聨氓聛聠莽潞戮盲禄聥忙聼聸莽聛聻忙聧聬氓拢聵茅聳鲁脩聦氓聣職茅聨庐氓聜職莽聲路茅聨掳忙搂鹿茅聨录猫戮戮莽碌驴茅聬聮氓聟录莽鹿聸忙聺聢氓聣搂莽搂聳碌氓聴聫氓聛聯茅聴聛莽茅聨赂忙聬聛茂陆聻氓漏碌氓卢篓茅聳碌氓陇聢氓聞潞氓篓录氓潞篓忙垄禄忙碌拢猫聶鹿茫聙聤茅聳潞氓聜聶莽路職氓庐?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    tuple[str, ...]
        茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼忙聲聥莽聭聹氓潞篓茫聙聝茅聤聣脩聟莽聼聶茅聳赂忙聼聜聜氓聛聯茅聧搂忙楼聛芒聮聭忙陇陇忙聨聠氓聛聯莽录聛氓陇聤忙戮鹿忙聸聼莽鹿聺氓搂聵氓聨陋茅聴聛氓聣虏氓聛聟莽禄禄氓露聟脦拢猫陇掳忙聨聯莽聟聲茅聬聨茂陆聨氓聙聳茅聨麓莽聤鲁氓麓虏忙驴聻忙聢聶氓聻鹿茅陋聻茫聢隆盲驴聤茅隆聳忙驴聬氓職聶茅聴聜氓陇聤莽搂鹿氓搂聤猫聶鹿莽麓聮氓娄炉猫炉虏莽職聭茅聴聛莽茅聨鹿氓聸卢氓录禄茅聰聺氓聴聵氓聞聼茅聳驴忙露聵氓芦聫聠忙露垄忙垄潞莽禄聥氓聺聴莽聙鹿忙聸篓氓聣聸茅聧聶氓聭麓忙垄潞茅聴聢忙碌楼氓聽楼氓录虏氓篓聭茫聢聽莽陇聞茅聳禄忙篓录莽虏炉茅聬聯忙聸聽芒聜聟聣氓聛聯茅聳鲁脩聦氓聣聼氓庐聲忙聢娄忙聜聶茅聬聭忙篓潞盲潞聴茅聴聛莽禄聵忙聼篓茅聧職氓忙聜隆猫鹿聡忙聮聰陆茅聳禄忙聞炉氓掳聶忙碌聽茫聢陇忙聼聸茅聫聝芒聜卢莽潞掳氓露聡忙聧垄氓漏碌氓卢聛篓茅聴聜氓聽聼莽篓聬氓篓聦芒聰聭茅聬聬忙聫聮忙聜聭茅聳禄莽聠虏忙聶聽氓猫炉虏茅聧芦脨陋茅聴聜盲陆赂忙聬娄芦氓虏聞氓录虏氓漏聤氓聽聺脦娄茅聳赂忙聸篓氓聻掳茅隆芦氓聭麓忙聼拢氓篓聠氓驴聰氓職聙茅聳赂忙掳录氓聛聤氓猫聶鹿忙聥聽茅聢楼氓麓聡盲禄漏茅聴聛忙麓禄氓聨录茅聧聤氓聺聴忙聜掳茅芒聜庐聲氓聧聻茅聧聫莽垄卤芒聜卢氓聸垄禄忙碌拢猫氓聛聽茅聳赂氓漏聝莽聣聽茅聤聠氓聠漏莽鹿聺茅聧楼脩聠忙聭聲氓漏碌莽聜麓氓聻炉茅聧篓氓陆聮氓麓隆茅聨露猫聤楼莽聟聫氓漏碌莽聜虏莽聛聺茅聧聰忙卢聭莽麓聮茅聤聤茫聝庐莽聟录忙驴庐氓漏聟忙茅聨路莽聤禄忙聡聽茅氓聜陇氓聰鹿茅聴聜盲陆陆氓麓碌茅聧聼氓聺聴氓录虏茅聬聵猫聧陇氓碌聛茅聳赂忙聞碌盲潞聺氓庐聲氓忙聜露氓篓聭忙聨聭氓垄虏忙陇陇莽搂鹿氓搂聤忙麓陋氓麓聧聲忙陋篓茅聴聛莽篓驴莽鹿聵茅聫聛氓录虏茅聬聵氓路庐忙陆?
    """
    return tuple(f"configs/experiments/{config_name}" for config_name in config_names)


@lru_cache(maxsize=None)
def _build_file_fingerprint(relative_paths: tuple[str, ...]) -> str:
    """茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聫茅聢搂莽聞娄莽禄禄忙驴聥氓麓聮氓娄陇盲陆潞盲潞聶忙驴聽莽聰碌氓聛聻茅聧篓氓麓鹿氓篓虏莽聳戮忙驴聽茅聺聸茅聬聯茫聝娄莽鹿聺茅聴聢忙露聶莽聙鹿氓聭聤莽鹿聢茅聫聦忙露聵脦聰忙碌拢莽篓驴忙聭碌茅聴聛猫氓聳聴茅隆篓氓漏聜脦聸茅聧聬茫聞垄忙忙驴聻氓聴聵氓聤聲茅聫聥忙聢聽莽麓聯忙碌拢氓聯聞盲潞戮茅聳赂氓潞垄氓篓聤忙驴隆氓聽聲氓鹿聳忙碌拢忙颅聦忙聭聲氓漏碌莽聜麓氓聻聧篓氓陆聮氓麓隆茅聨露猫聤楼莽聟聫氓漏碌莽聜虏莽聛聺茅聧聰忙掳颅茫聛聞茅茅聜驴庐氓漏聟忙茅聨路莽聤禄忙聡聽茅氓聜聸茅聳禄莽聠赂莽虏聨忙戮搂忙忙聲娄陇氓聭拢禄茅聳鹿忙聣庐氓垄聴茅聬聯莽聠录忙聼聼莽聙碌莽氓聣卤茅聳禄忙聨聧莽虏聶茅聫聦忙聸垄氓麓卤氓娄炉氓聯聞氓漏碌茂鹿楼氓猫聴聣莽聲路茅隆聬茂鹿聙脦篓茅聳碌氓氓聙聴茅聨鹿忙聸垄忙垄禄茅聧聦忙卢聭莽陋聰茅聳鲁脩聦氓聣聸茅聧聥忙露聵脙鹿茅聧聲莽陇聤茅聳潞氓碌虏聙氓聰碌茅聴聛氓聟录忙聜聜氓篓录脩聞忙聟篓茅聧聝娄莽聟聲茅聨录忙路聻鹿茅聬聫芒聲聟忙聼聼茅聧聶氓陇聥氓聙聬茅聳鹿氓聸搂茅聠聡忙驴聽茅聺聸莽聣聫茅聨鲁氓聴聶莽麓聯茅聧聦忙卢聮氓聟聦氓漏碌忙聲禄氓庐聲氓陆聮氓麓路茅隆聯莽聡聛忙陆聼茅聴聛猫搂聞氓聞鲁茅聬隆茫聞娄氓聢聤茅聫聣忙聢聰忙聰陆忙陇陇忙聴聡忙拢聞茅聬聮忙聝鹿氓氓潞聺茅聜炉忙楼聽莽聜虏氓聲隆芦忙禄聢氓聼聙茅隆聮芒聜卢茅聬拢莽聝陆氓麓录茅聫聡莽聜碌氓聨赂茅聴聛氓聭聤氓聤娄忙碌聹茅拢聨忙茅聫聦茫聝娄芒聮聭茅聬聽氓聸篓氓陆聝茅聧陇猫戮漏莽麓聮茅聧聶氓陇聤氓聤陇莽聬職忙卢聯忙聼聼茅聨碌茅聳鲁茅聰聥氓聻碌茅聹聣茅聳赂氓驴職氓聛聬茅聨鲁氓聭麓聮忙戮露忙篓潞忙聻陇莽录聛猫戮篓氓赂隆茅聨聲聛莽聲路茅聧楼脩聙忙聥掳茅聴聜盲陆潞莽虏炉忙赂職忙氓麓鲁茅聰聺氓陇聦氓碌聛茅聳赂茫聝庐氓聣聶莽禄戮脩聡忙聜鹿茅聨颅忙聨聭氓茅聳鲁脩聦氓聣聸茅聧聥茫聜聟氓篓虏莽茅聳赂忙卢聫茅聧聨忙聮禄忙垄潞莽禄聥氓聦隆茅戮職忙陇陇忙聸聠氓聸拢芒聜卢茂鹿聜氓麓露茅隆聮芒聜卢茅聧聬芒聜卢氓娄聻茫聜聠忙聦戮茅聧聼氓聺聴氓鹿聫?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    relative_paths : tuple[str, ...]
        氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄忙驴聻忙聢聶氓聼聞茅聫聛氓茫聛聞茅聨戮猫路潞忙聻聭茅聳赂忙卢聫莽鹿聵茅聫聦茂陆聣氓茅聤聫芒聲聝氓聤赂莽录聜盲陆潞氓搂鲁茅聧聴忙聤陆氓录禄氓篓聭忙篓路忙聰陆茅聳赂忙聸篓氓聛聞忙驴庐茫聜聠莽鹿聺氓篓聢氓聻庐忙聻聼氓漏碌莽聜虏茂鹿陇茅隆驴聯脙潞茅隆聫氓露聠忙聶垄茅聴聛茅聙聻氓卤陆氓垄掳莽录聛忙垄禄茅聢搂忙陆搂茅聨陆忙禄聢忙聝聦茅聧隆忙露聶莽聟聲茅聳鹿盲录麓忙陆搂茅聫聥莽聠录忙聼拢茅聨戮氓颅聵莽莽录聛莽禄聵莽聸陋聻氓卢聰聭茅聧隆忙掳卢莽聟聵茅聴聜盲陆陆忙氓驴聲忙聧垄氓漏碌氓猫聯聼茅聳驴忙驴聠茅聫聟忙聫聮莽聮潞盲戮搂鹿茅聳赂莽聤鲁氓虏赂氓忙麓麓芦茅聧聲氓聤鹿莽录聜氓聜職莽搂鲁莽禄露忙掳聲路氓篓虏忙聶赂茅聳禄忙氓掳聶茅隆娄茫聞漏忙垄潞茅聧聸氓漏聝氓炉聶茅聳赂忙聞漏氓聣卤?茅聴?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓麓茂陆聢氓篓聟莽聤禄氓鹿聳氓篓拢氓莽禄戮忙聝搂茅聫聦忙聸垄氓麓录氓漏碌氓聸搂忙茅聴聛氓聯聞忙拢麓茅聴聞氓聥漏卤忙驴聽氓聸搂脦搂茅聳赂忙露卤氓聳聳氓篓聟茂陆聡莽麓聯忙碌拢忙聫聮盲潞戮茅聴聛氓聭聤氓聤聫茅聫聜氓聜聸莽娄聧氓氓聽垄氓聫聯茅聳赂茫聝拢氓聤聧莽聛聫莽聣聢茫聛聞茅聤聣脩聝芒聳聢茅聳潞氓虏聥莽露聛茅陋聻忙漏聙忙聡聽茅氓聜聹氓麓虏忙驴聽忙氓聳聳茅聰聲茫聝楼忙碌聽氓卢碌聛茅聳鹿茅聜娄氓聨陆茅聧聨莽聠录忙聼篓氓漏碌氓聴聴芒聜陆聮茂陆聻氓氓陇聥芒聮聮氓篓麓茂陆聟忙聠隆茅聨炉氓聽楼忙聼拢氓娄陇芒聜卢茅聧聼忙聺聭莽聲路茅聨麓茂鹿聙莽庐禄茅聬聽氓聸篓氓陆聝茅聬聨茅聤聢氓聴聴氓茅聳赂氓漏聝茅聨驴聽氓麓聮氓氓聭录聺忙驴聻忙聮麓猫聣戮氓篓虏茫聜聣氓录赂茅聬聰氓聟录莽聟聼茅聳禄忙聳驴氓录露氓篓聟氓聭麓忙聼拢茅聰聺氓聭搂禄忙陇陇莽虏聯茅聧漏芒聜卢忙陇陇忙聨聭氓芦禄聽莽聤禄忙聼拢氓娄聧陇忙職聶茅聳鹿莽聝聵忙聦禄茅聧聨莽聠录忙聼聲忙驴聽氓驴職氓聬聜茅聳鹿茅聰聥忙陇陆氓忙麓茅聢潞盲陆潞忙搂拢茅聴聛氓聭聤脙聤聢莽篓驴莽聲路氓漏碌氓聽職氓聛聯茅聰聺氓潞隆忙聻聼茅聳赂氓氓聤聧茅聤聡氓聥录陆茅聤聤脩聙忙搂聢莽录聜盲陆鹿茫聛聠茅聧楼茫聞搂氓聨戮氓漏碌莽聜虏忙篓聤忙碌聹忙卢聯氓鹿聣茅聬聬盲潞聹茅聳碌氓驴聤忙茅聭掳氓聻庐芒聰聭茅莽聭掳盲潞戮茅聴聜盲陆潞莽虏隆颅氓聸潞茅聧聸忙聺聭氓聧鲁茅聳碌氓驴楼莽麓聲茅聧聵莽聰赂忙垄潞莽禄聥猫路篓莽庐潞茅聳赂氓卢聤聶氓炉聠虏茅聢搂氓录禄氓篓聭茫聢聽忙拢聵忙驴聻氓聴聶氓聛聯茅聢禄氓聽楼忙聜聴莽聭聶氓聥茅聳赂茫聞楼忙陆隆茅聬聸脦聼茅聧聫莽聟聨莽聫掳茅聴聛猫聜漏芒聮聢茅聧聯忙露垄氓录露茅聧聸氓漏聛莽鹿聠茅聳禄忙聞碌盲潞聹茅聢搂忙麓聹茅聨鹿茫聢陇氓鹿聡茅聬聴氓聽聺盲潞卤茅聴聛莽鲁聲氓聣聺茅聤聡忙露垄氓录赂茅聫聝芒聜禄禄忙驴聥忙氓篓聭忙掳卢芒聟卤莽录聛莽聜麓聵氓娄聥炉忙鹿聛茅聴聛忙聦聨莽鹿聜氓篓虏茂鹿聜氓麓碌茅聢搂忙垄潞茅聧聸氓漏聝莽聟聼茅聳赂氓漏聝莽鹿聢氓猫炉虏芒聵聣茅聤聫芒聙鲁盲潞聹茅聴聜盲陆赂莽聛聺茅氓聭茅聨戮氓聙碌茅聬聼忙卢聫氓莽禄聦猫路篓莽麓聯氓氓聥聳禄茂陆聡氓聟聵氓庐聲茂驴聽忙聜聶茅聢潞氓聜陇忙陆聰忙驴聽莽忙搂聟茅聧篓忙聤陆氓聼聸茅聧聝楼猫聢鹿茅聳赂忙麓聵茅聢路忙聨聭脙鹿茅聰聺氓聽陆聬茅聳赂氓氓篓聙茅聫聦忙露聶忙聝聰茅氓聟录莽聫聺茅聬聨忙氓聳聴猫陇掳氓聠聙娄茅聤聤芒聲聛茅聠聡茅聳禄忙聳驴氓聵虏忙戮鹿氓露聣忙垄禄忙驴聽氓潞隆盲潞聹忙驴聻猫炉搂氓莽禄卢聯氓鹿聮氓娄陇盲陆赂盲潞聴?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    str
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聶氓聜娄莽庐聯茅聫聦茂陆聣氓鹿聡茅猫聤楼忙炉聞茅聴聛忙麓禄氓聨录氓娄芦忙楼聟氓娄麓茅聨潞忙聢颅忙陆漏茅聳驴忙驴聠忙聡聧忙戮鹿忙聸垄忙聼拢忙聬麓茫聜聨忙陆聬忙驴聻忙碌聥忙聶聺氓氓漏聝莽莽聙拢氓聣聛莽篓聫氓漏碌莽聤禄忙陆搂氓篓虏茫聞漏茫聙聟茅聫聦茫聝娄莽聟聫茅聳赂莽鹿聝茅隆楼莽聤碌莽麓聮茅聬聵猫聧陇莽聲潞茅聳潞氓虏聙氓聳聯茅聢搂忙聲禄茅聫聟猫聦茅聯聫盲潞聹茅聼聸篓忙聦戮忙戮搂忙露垄忙聼聸莽聙拢芒聜卢氓聺聴氓录禄茅聬聰氓聭聤莽露聡氓娄陇茂陆聟氓聲炉茅隆聨氓露聣忙垄潞莽录聛忙篓禄氓掳掳茅聳禄莽聠聺氓聭聳鹿莽聝聵忙聼隆茅聧聥忙聮禄忙聲聬忙聬麓芒聙虏莽掳禄氓娄聻茫聜聠忙麓聵莽禄庐氓潞篓氓聼聙茅隆聮氓聜聸氓職聙茅聳鹿猫莽麓聲茅聨鹿茫聢陇氓麓聵茅隆聰猫聜漏莽聳聞茅聴聛茅聺聸脛聸茅聧聲莽聙聴聜盲陆赂忙聠隆氓篓虏茂鹿聜氓麓鹿忙碌录氓颅聵氓聙驴茅聳禄茫聢垄忙聲赂莽潞戮氓楼赂忙聜聴茅聰聺氓潞隆盲潞聹忙驴聻忙聬颅氓聶拢茅聫聦?氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄茅聳禄忙聞聬芦茅聧篓氓聜聻忙聼聸茅茂陆聠莽莽录聛莽聤卤氓聧鲁茅聫聦莽聠录忙垄禄莽聙碌莽氓娄陆茅聴聛莽篓驴氓茅聧篓莽聰赂忙鹿聛茅聴聛莽篓驴莽鹿聬茅聧職氓莽麓聻茅聨麓茂赂陆莽聟聼?

    茅聴聜氓聜職氓聙路氓篓麓氓聸茅聢搂莽篓聢茅聫聟莽聤禄茫聙聛氓篓聠忙聮鲁氓聠聙忙陇陇忙聮露氓聛聼茅聬聸茫聝漏忙垄潞茅聧聺氓聥颅芒聳聣茅聳赂忙篓驴氓聛聬忙戮鹿忙聸隆茫聛聬茅聨潞忙聮露氓聨碌茅聴聜氓聜職氓聙赂茅氓麓聺氓氓陇聥莽聟聲?
    ------
    FileNotFoundError
        茅聴聜氓聜職氓聙路氓篓麓氓聸搂莽聲路茅聬垄莽莽陋聴茅聳鹿茅聜娄氓聳聳茅聧篓忙驴聥氓鹿聳氓篓拢莽聛驴聻氓聭搂碌忙路聡氓漏聡氓娄露茅聧聸忙芦拢莽录聜盲陆潞氓搂碌氓漏垄忙漏聙氓聼聻茅聨麓茂赂陆氓聛聬莽聙鹿忙聸聻忙碌聜茅聫聠茂驴聽忙垄潞莽禄聥忙氓聞聫茅聳赂氓漏聜忙陆隆氓炉芒聵聣茅聤聫莽聤聳茫聞漏忙聼篓氓漏碌氓聴聵氓聶鹿忙陇陇氓?茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼忙聲聥莽聭聹忙聳驴莽聲路氓漏碌氓脦搂茅聳赂芒聞聝氓聤聮莽禄篓氓漏職忙垄潞茅聧聺氓聥卢氓陇聞茅聳驴忙掳露忙聼聸茅聫聝忙聮鲁茅聳?茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙碌拢茅聺漏猫聙驴茅聴聛莽搂麓茅聧聦茫聞搂忙聠赂茅聫聝氓聽聺莽庐聳忙驴聻氓路潞聹忙聝聼茅聴聛茅聺聸茅聧聽忙楼聝莽麓聻忙聬麓茫聞娄莽聟聼茅聳禄忙篓潞氓录露忙戮露氓聥颅莽麓聮忙碌拢猫搂聞氓掳娄莽聙碌忙聜聜氓炉麓录茅聬聰氓聯聞氓鹿聧茅聴聛氓聯聞茅聬聴氓聴聵忙聶露忙碌聽氓氓聛漏茅聫聣氓聽聸氓聤聥莽禄禄氓聴聵茫聛聞茅聨掳氓聣聛莽搂露茅聳录忙聺驴忙陆隆茅聫聦茂录聞盲禄娄茅聧聯脩聞氓聢聸忙驴聽莽莽麓聲茅聧聫忙漏聙茫聙聛氓篓聭茫聢隆氓聙路茅聳鹿莽垄卤氓聨赂茅聧聥忙聮鲁茅聧聹盲陆鹿氓炉聦茅聴聛茅聺聸莽聣聠氓娄忙聟篓莽聛聻氓聫楼莽鹿聠忙陇陇忙聞漏氓聻鹿茅忙聼聸茅聢鹿氓聻聤聹莽聙碌忙聺驴氓芦掳茅陋聻氓聸聧聦茫聜聟忙聶陋茅聴聛猫忙掳娄茅隆芦氓陇聤脙潞茅聫聧脩聙忙聻聳茅聳驴忙聸聻氓聙赂茅隆芦氓聭麓忙聼聲茅聧驴聡茅聳赂忙篓鹿莽虏聶茅聫聦莽聠录忙聜聹氓娄隆隆忙聻聶茅聬聨猫搂聞忙麓聵莽禄禄氓聽楼莽聧聙拢忙聰陆茅聳赂氓聳聬茅隆聰忙聸垄忙垄禄忙碌拢猫聶鹿氓赂聸茅聳赂茫聞搂氓聨搂猫聻聻茅聳赂忙聸篓氓聣聶茅聧聰忙聴聢忙垄禄茅聧聦忙卢聭氓聤聧茅聬聨莽卢聸氓聠戮忙陆聳氓漏碌莽聤鲁莽茅聧聥茫聜聣忙聼隆茅聧聦忙驴聬氓職娄茅聳驴忙驴聠茅聫聛氓露聣忙聼拢茅聨掳脣聣茅聳鹿莽聳炉忙芦拢莽禄卤忙聨聰莽麓聮茅聤聫莽聤虏莽庐鹿茅聴聛莽聭掳氓聲驴茅聳掳茂陆聣氓鹿聝茅聰聼莽聤虏莽庐聸忙陇陇忙聳驴氓垄陆茅聰聸忙禄聢忙聼拢忙聬麓莽搂碌茅聳赂氓氓芦掳茅聨庐忙篓录芒聜卢莽聰碌莽隆露茅聧聥忙聮露忙聠麓茅聧聲茅聴聛猫陆掳莽陇聛茅隆颅莽聝陆氓娄麓忙碌拢氓聣虏忙陆篓茅聳鲁脩聦氓聣職茅聨卤茫聞漏氓聼聙茅隆聮氓聥卢莽聟?
    """
    digest = hashlib.sha1()
    for relative_path in sorted(relative_paths):
        normalized_relative_path = str(relative_path).replace("\\", "/")
        file_path = (ROOT / normalized_relative_path).resolve()
        if not file_path.is_file():
            raise FileNotFoundError(f"fingerprint source file not found: {file_path}")
        digest.update(normalized_relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@lru_cache(maxsize=None)
def _resolve_search_implementation_fingerprint(
    *,
    search_kind: str,
    implementation_name: str,
    experiment_config_names: tuple[str, ...],
) -> str:
    """茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聫茅聢搂莽聞娄莽禄禄忙驴聥氓麓聮氓娄陇盲陆潞盲潞聶忙驴聽莽聰碌氓聛聻茅聧篓氓麓鹿氓篓虏莽聳戮忙驴聽茅聺聸茅聬聯氓驴聯莽聮潞莽聝聵忙驴聻氓聭颅忙聡聬莽禄卤忙聨聴忙聜漏氓庐聲氓聸搂芒聮聢茅聴聛茅聙聻氓卤陆氓垄虏忙陇陇莽聟陇茅聳潞氓露聨莽聛聬氓篓虏忙聶聺茅聳赂氓卤戮莽篓聢茅聨路?茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聟忙職娄茅聳赂氓聛聟茅聧聶氓驴聸忙聼拢茅聨麓茂陆聠莽禄戮忙聝搂茅聫聦莽聠录忙聜搂茅聧芦莽聠潞氓聡隆茅聴聛莽禄聵氓赂聮茅聬聫忙掳卤忙聧垄氓篓聭茫聢聽氓路聺忙陇陇忙聴聜忙聶芦茅隆娄忙垄潞茅聳聜虏猫聥戮搂莽聴聫聣猫路篓脨搂茅聳鹿莽鹿聺茅聫聛忙聮禄忙聜漏茅聭录猫聢碌忙聠掳忙驴聽莽聰碌氓聛聻茅聧篓氓聣聛氓聢聮忙聬麓茫聞娄氓聛陋茅聳鲁脩聦氓聣聶茅聢鹿忙聢娄茅聫聜茫聜聞莽路碌茅聴聛氓聭聤茂鹿垄莽禄聽忙聽篓盲驴聤茅聨戮莽庐聸茅聳禄忙楼聙莽聣聝茅聨路忙麓垄潞茅聧聸氓漏聞芒聵聣忙陇陇氓聣職莽聙碌氓楼赂氓录麓茅聤聫茂录聞氓聨卤茅聴聛茅聺聸莽禄虏猫聤楼盲驴聤茅聧搂忙楼聛莽聟聲?茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅茅聰聰莽禄聵忙聶聻茅聴聛氓聭聤莽篓聮氓篓录忙卢聯忙聜驴茅氓搂聬盲潞聹茅聳潞氓露聝氓聤聨茅聬颅氓虏聞忙聼拢茅隆聯莽聝聠忙篓聙茅聳潞氓虏聙氓聳聳茅聨庐忙禄聝茅聧隆忙篓录忙職娄茅聴聜盲陆鹿忙聜露茅聧聤忙聸垄忙戮露忙聞聯聼茅聳驴忙驴聠茅隆聭麓忙聼拢氓娄炉氓聯聞忙聜聛莽录聛忙聲禄氓搂聤忙麓陋忙聜隆忙聬麓茂陆聞猫聶聨氓漏碌莽聜虏忙篓聤茅聧聵氓楼赂茂陆聻猫鹿聡忙露聶忙聝聻茅聬聼忙卢聫氓茅聬聺盲录麓忙垄潞茅聬聬猫聴聣氓聤職茅聳赂莽聠路忙陆隆茅聰聺莽聜潞茅聤聢氓聸漏茂录聺忙驴聻忙聮麓猫聣戮氓篓虏茫聜聣氓录赂氓篓聭茫聢隆莽聟聲茅聳碌氓篓驴氓聞卤茅聢搂莽露聤茅陋聻氓聽楼氓娄赂茅聤聣氓潞拢氓聳聳茅陋聻忙聞颅忙聝聰茅聰聺氓聠拢氓聛聯茅聨掳莽拢陆氓篓麓茂陆聟忙聬掳聫茅聴聛氓聭聤忙驴聻氓漏聜莽聮聡忙碌聥忙搂聢茅聳碌氓驴聲氓茅聤聤芒聲聜莽聟聫茅聼芦氓聸搂茂鹿陇忙戮搂忙聼楼忙聼拢氓漏碌氓聸篓莽忙驴漏聜忙聬聟陇茅聬聽氓聸陋氓录聯莽禄篓猫聧陇莽麓聯忙碌拢猫聶鹿氓赂聸茅聳驴忙聸聵氓聻鹿茅隆?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    search_kind : str
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聳莽禄聽莽聤禄忙聼聼茅聨鹿忙聞碌氓職聶茅聬聼忙卢聶莽庐聯茅聨忙聲聬忙聬麓芒聙鲁茅聴聛忙聬聻氓聤聧莽禄禄氓聽楼莽聧聫聧氓潞篓茅聢搂盲驴聤茅聭虏芒聲聝忙拢炉茅聳鹿盲陆赂盲禄聴聛氓聯聞忙聹卢氓篓虏忙忙聧垄氓篓聭忙禄聨忙搂禄茅聴聛莽聭掳氓聲驴茅聳掳茂陆聟茅聧聬氓聫聜忙聴聢氓麓篓茅忙聝搂氓鹿聴茅聴聜盲陆潞莽虏聰職莽聙碌氓莽禄掳露氓麓赂氓娄陇盲陆赂芒聜卢氓聽聲莽聟芦茅聧楼茫聞娄芒聮聮茅聳鹿氓聠虏忙麓颅茅聫聦忙露垄氓聼聻茅聨炉脩聜氓拢聲茅聴聜氓聜聶猫聝露茅隆聽垄忙聧垄茅聨麓茂赂陆忙聶聺茅聳碌氓驴聲氓陋聧陇忙聸聼茫聛聞茅聨掳芒聲聛莽聞潞茅聳赂忙掳聫聦忙露垄氓娄路茅隆聳忙禄聝氓聼聦忙驴聻氓颅聵莽虏聯莽禄聽忙聽颅氓录禄茅聰聺氓陇聥氓聛聞莽录聛氓氓娄聴聸忙垄?neural"`` 茅聴?``"classical"``茅聴?
    implementation_name : str
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聧茅聢搂莽庐聯氓庐聲氓楼赂氓茅聢楼氓聠虏莽掳楼忙驴聽氓戮路忙聺驴芒聜聞掳氓麓聭茅聨戮莽禄聵莽聟聫茅聳赂芒聞聝脙陋氓篓麓芒聜卢猫搂聞忙麓聹茅聧聫忙漏聙茫聙聛氓芒聜聨庐芒聲聛芒聳聯茅聨赂氓鹿庐茂鹿聜忙垄禄忙碌拢莽颅聥氓職聶茅聳赂忙聢聽忙聶芦氓炉掳氓漏聡忙聡聻茅聤聣脩聦忙麓赂忙驴隆茫聜聫莽碌陆茅聧聼忙聜隆茅聫聡茫聢隆莽聟楼茅聳潞氓聠拢氓聛聯茅聬聨莽卢聸氓聸搂忙聵楼忙驴聻忙聢聻莽聭陇莽禄禄氓聴聸忙聼聲茅聧驴聡忙聬聬茅聧聼忙聬聛忙垄潞莽禄聥忙氓聞聬莽录聛氓卢聻鹿茅聧聶氓聭颅茅聤聢氓聴聵氓掳碌茅聳赂氓卢聳聳氓炉聤陆忙聲聝茅聧聦忙露職氓聨陋忙驴聽茫聢拢忙鲁聸茅聬聴氓聴聸氓麓聺茅聤聣脩聟莽拢录茅聳禄忙篓潞莽拢篓虏氓炉赂芒聜卢忙庐驴氓聳聴茅聨赂忙聤陆氓鹿聝茅聢潞盲陆潞忙職娄茅聳赂茫聝垄忙聶聺氓娄虏?``"lstm_ekf"``茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮莽聲聦忙聠赂氓庐楼氓聽垄盲潞卤茅聴聜盲陆赂忙聬芦茅聧聤氓陆聮氓篓聠氓驴拢莽聞陆茅聳潞氓露聝氓聤聧茅聧?ekf"`` 莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳禄忙聢拢氓搂陇茅聧聤氓聺聴忙聼篓茅聫聜氓聻聮聰茅聳禄莽聭漏茅聫聦莽聠路芒聳聨茅聶聡茅隆聲脩聝脙潞茅莽聜碌忙職娄忙驴聻氓路庐盲潞聹茅聬聮?
    experiment_config_names : tuple[str, ...]
        茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聟忙職娄茅聳禄茫聢垄莽禄聬忙楼聟忙聼聼茅聳鹿聙氓聤聮忙驴聻氓路潞聹莽禄聽茂陆聠莽鹿聺茅聧篓氓搂聢茅聬聨茅聺聸莽聼聜氓搂聤忙麓陋氓麓路茅莽聜虏氓氓漏碌茂鹿陇茅隆鲁掳莽聲路茅聢楼忙聞聛忙陆漏茅聳录氓聯楼忙聲赂茅聧聵莽聰赂忙垄潞莽聮聡芒聙鲁莽聙禄茅聳赂忙露卤氓聛聞忙碌聹盲录麓忙垄禄茅聧聦忙掳聜卢茅拢聨莽虏聢忙赂職芒聜卢茅陋聻氓忙聲聯茅聬聵猫聦聴聛莽禄聵氓聻录忙驴聜聣氓麓聬茅聧搂忙楼聛莽聟楼茅聳潞氓聸漏氓聛聸茅聢搂忙聭聹莽禄庐氓漏職氓录禄茅聤聣茂赂戮氓聨卤氓茫聜聠氓聤聴忙驴聮庐氓麓聺氓氓聽聼莽聟聼茅聳鹿忙聝搂莽聯篓莽禄聙氓卢聼拢茅聨驴氓聠篓盲潞掳莽聙鹿忙聸聼氓聞录茅隆搂盲戮聙忙聼聸忙聬麓茂录聞莽茅聳潞氓卤戮莽聸氓聼聙茅隆聮氓聜職氓聻聺茅聳鹿忙聧聬茅聮聝莽聠录忙聼篓氓漏碌氓聴漏忙搂鹿茅聳潞氓聭虏聯茅聨陆聮氓麓露猫莽禄戮脩聡忙聼拢猫鹿聡忙聴聙氓麓聦忙驴庐氓漏聝忙陇陆茅聫聦氓聭麓忙聜聶茅聨禄忙聨聺盲禄戮茅聴聜盲陆鹿茅聧聵氓楼赂氓聼聻茅聨麓茂赂陆氓聛聬忙陇陇忙聞碌忙戮聙忙戮鹿忙聸垄忙垄禄忙碌拢氓聭聤氓聲聻忙陇陇茫聞楼莽聣聯氓庐聲忙聢娄氓鹿聡茅聫聣氓聻庐氓職聭茅聴聛氓聯聞氓聲芦茅聬聴氓漏聡忙聜隆氓篓聠忙聮麓氓聙碌茅聳禄茫聢隆茅聬聮氓聣搂忙聜庐茅聤聣芒聲聜芒聮聭莽录聛氓卢聧聨忙聞掳忙聼聼茅聬聼莽聣聢忙聬聬茅聳禄茂陆聡氓聟聵茅聫聦茫聞搂芒聜陆聨氓聣娄氓篓麓氓聥垄禄忙赂職氓聸搂氓垄录聛氓陇聦忙聲禄茅聨驴聽忙聜垄茅聧聺茫聝搂茅聴聛莽聭掳茅聧聥莽聠录氓鹿聤茅聧聸氓颅聵莽聟聶茅聳赂忙聞录露茅聧聽忙漏聙忙聼隆氓庐聙氓卢拢驴氓篓麓莽聥聟莽庐聯氓氓娄露茅聧聸莽聼聴茅聴聜氓聜職氓聙路茅聬聮茂鹿聜氓鹿聝茅聧聸聻莽拢驴忙陇陇忙聸聴氓聛聯莽聙碌氓陆聝茅隆聞漏氓麓录茅聬聰氓掳聛氓聛聯茅聧篓盲潞聹茅聳鹿忙聝搂氓麓聬聫忙聼聧茅聳驴氓卢聛麓茅聳潞氓卤戮莽篓聭茅聢鹿忙聢娄氓麓卤氓娄陇氓漏聛忙聜拢茅聴聜盲戮聙芒聜聜虏猫聥戮搂莽聣聢茫聛聞茅隆聯莽聜碌氓垄垄茅聳赂忙卢聫忙聡聵氓搂聤忙麓陋氓鹿聬忙聬麓茂陆聡莽聲碌茅聴聛莽聭掳氓聲驴茅隆娄茅聺聸茅聧聸氓漏聝氓聞聼茅聳碌氓氓娄聡茅聧聵忙拢聟茅茫聢隆忙聬聬忙陇陇忙聢聺忙聡聵茅聧漏芒聜陇陇忙聨聧莽聞娄莽禄聙氓聽楼忙聼聼茅聧聶氓氓掳娄茅聳潞盲陆赂氓聙驴茅聫聦茫聞漏氓忙驴隆茫聜聡氓搂路茅聧聥忙露垄忙聜聵氓漏碌氓莽陇聣忙驴庐忙篓驴氓聼聳茅聬聯忙聸垄忙聼拢茅聫聜忙聦聨忙聝聢茅聳鲁猫莽聣聫茅聢搂?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    str
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬氓聫聣莽聳聞氓漏碌脗掳氓聙搂莽禄虏盲禄聥氓麓炉茅隆聳莽聜虏莽拢禄茅聬聨茂陆聨氓聳聴茅聬聯忙聸垄忙聼聧茅聢潞盲陆赂忙職聼茅聳鲁忙麓聼茅聫聦忙露垄氓娄露茅聧隆忙楼聙茅聴聛氓聯聞莽聼聣莽录聧盲陆赂茅芒聜庐聲忙聨聭茅猫路篓氓赂聸茅聴聜盲陆陆氓麓碌茅聧聽忙聞掳忙聜路忙聺聢芒聲聝盲赂聧茅聳鹿忙聧聬氓麓碌氓庐聯盲戮聙忙聼隆氓氓聠篓氓莽录聜氓露聟忙麓聫聦忙露垄氓鹿聵氓娄陇芒聜卢茅聨卢聯氓鹿聥茅聰聰猫搂拢忙聥潞忙驴聻忙聺聭氓聬聥茅聬聼脩聟莽虏聰茅忙碌聨莽聭掳茅聧聬莽聟聼茅聬聨忙氓聳聸茅脩聡氓聼楼忙戮露氓漏聨忙陆篓茅聳赂芒聞聝脙陋茅聬娄忙禄聢忙垄禄忙碌拢氓聭聤氓聲聻茅聳鹿莽忙拢聺氓庐聲氓聽聲氓娄赂猫陇聧茅陋聻氓聽聼莽鹿聺茅聬垄茅聺聸盲禄聹忙驴隆莽聭漏氓庐聲忙驴聠脦聰茅聧聸氓聛聯茅聧聦莽聜虏莽聼聴氓漏垄猫路潞莽聻录茅忙聸垄忙垄潞茅聬聯氓潞拢莽搂潞茅聳赂氓卢鹿陋氓庐聲茅聺聸芒聳聨茅聨戮氓炉赂氓聨聯茅聴聛猫聤楼脙陋茅隆娄猫聴聣脦拢茅聨录盲陆鹿忙聡職茅聳驴忙驴聠茅聬聯茂陆聣莽聟芦茅聧楼茫聞娄莽陇聝茅聳赂茫聜聜氓聛聯忙驴隆茫聜聡氓搂路茅聧聥忙聞掳氓麓聭茅聨戮莽禄聵芒聮聮茅聳赂氓卤戮猫聣戮茅聢搂莽禄庐氓聽聼莽卢聼茅聢搂莽聧庐茅聫聧忙聲聝茅聳驴忙聸聴氓聛聯莽禄戮忙聝搂忙鹿卤莽禄卤忙聨聰芒聜陆聨氓聛聯莽禄卤芒聲聜莽鹿聸氓庐赂莽掳录茅聳潞氓聭庐莽聟隆茅聫聦忙露聵芒聵聣茅聧聶忙氓聡鹿茅聴聛猫炉虏茅茅隆鲁掳氓篓虏忙驴聻忙聢聻忙掳篓茅聬陇氓聥炉忙垄潞莽禄聥氓聦隆忙聰禄忙驴聻氓聫聣莽虏聨茅聧聮忙聨聰莽麓聮氓娄戮炉氓職陇茅聳禄氓潞垄莽篓聮茅隆聸卢氓麓垄茅聳卢氓露聠芒聮聭茅聴聜氓聽聼盲戮聴氓娄虏氓聽聲忙聼聸茅聤聤茫聝娄忙聝聢茅聳鲁猫炉虏莽搂鹿氓庐聲氓聧聻芒聵聣氓篓聠忙聢聺氓鹿聴忙驴隆莽聜陋氓聙聳茅聧聰忙聢娄氓麓鹿茅聭陆茫聜聡盲赂聧氓漏碌莽聤鲁莽茅聬?

    茅聴聜氓聜職氓聙路氓篓麓氓聸茅聢搂莽篓聢茅聫聟莽聤禄茫聙聛氓篓聠忙聮鲁氓聠聙忙陇陇忙聮露氓聛聼茅聬聸茫聝漏忙垄潞茅聧聺氓聥颅芒聳聣茅聳赂忙篓驴氓聛聬忙戮鹿忙聸隆茫聛聬茅聨潞忙聮露氓聨碌茅聴聜氓聜職氓聙赂茅氓麓聺氓氓陇聥莽聟聲?
    ------
    ValueError
        茅聴?``search_kind`` 氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄茅聳禄忙聞聬芦茅聧篓氓聜聻忙聼聸茅茂陆聠莽莽录聛莽聤卤氓聧鲁茅聫聦忙露聵芒聰聭茅聧聲氓搂垄茅聴聛忙麓禄氓聨陆茅聨鹿氓聸录禄茅聢漏氓聠篓莽路聻茅聬聨茂陆聨盲潞聻茅聧聵忙聞掳忙垄?``"neural"``/``"classical"`` 氓漏?氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆茫聙聜茅莽聜聳潞氓聸楼氓聻鹿莽潞戮莽禄聵莽鹿聸茅聨麓忙卢聫莽聛漏莽禄戮忙聝搂莽露聤茅聫聦莽聠潞莽虏聶莽聮潞茫聞楼氓職录聜?
    KeyError
        茅聴?``implementation_name`` 氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄茅聳禄忙聞聬芦茅聧篓氓聜聻忙聼聸茅茂陆聠莽莽录聛莽聤卤氓聧鲁茅聫聦忙露聵芒聰聭茅聧聲氓搂垄茅聴聛忙麓禄氓聨陆茅聨鹿氓聸录禄茅聢漏氓聠篓莽路聻茅聬聨茂陆聨盲潞聻茅聧聵忙聞掳忙垄潞茅聭虏芒聲聛茅聬聨莽卢聸氓聸搂脦娄茅聳赂忙聸篓氓聳職茅聬陇芒聜聴聛猫搂聞氓麓聵氓篓聣忙露垄茫聙聜茅聯聫氓聹拢陆氓篓麓茂陆聟忙聬掳聫茅聴聛氓聭聤忙驴聻氓漏聜莽聮聡忙碌聥忙搂聢茅聳碌氓驴聲氓茅聤聤芒聲聜莽聟聫茅聼芦氓聸搂茂鹿陇忙戮搂忙聼楼忙聼拢氓漏碌氓聸篓莽忙驴漏聜忙聬聟陇茅聬聽氓聸陋氓录聯莽禄篓猫聧陇莽麓聯忙碌拢猫聶鹿氓赂聸茅聳驴忙聸聵氓聻鹿茅隆聲茂陆聠莽鹿聺氓忙芦垄茅聴聛莽禄聵莽聛聫忙卢聯脣聡忙碌录氓颅聵氓聙碌忙驴聻忙聢聻莽聭搂莽禄聽莽聜麓忙聟篓忙驴聽氓聜聸忙聝聢茅聫聛氓聽楼忙聼聸茅聤聣忙聢聺氓聳職茅聬聨氓鲁掳芒聰聭茅聬聵忙聞聛氓聲聫茅聳赂莽聤鲁氓虏赂氓庐聲忙楼聙茅聢搂忙聜掳茅隆聲芒聜卢氓庐聲忙漏聙氓聼隆忙赂職氓聸搂忙茅聴聜盲陆赂忙聬隆娄盲录麓氓麓碌茅聰聲氓聜職莽聟聸茅聳赂忙露卤氓聳職茅聧聵忙垄潞茅聧聯忙聫聮盲潞聺莽录聛猫炉虏氓茅聫聟氓聽聲氓录麓茅聬聲盲陆鹿茅聴聛莽禄聵氓聤聫茅聫聟忙露聶莽娄聧忙楼拢氓聠漏芒聮聮茅聳赂氓卤戮猫聣戮茅聢搂莽禄聽聼莽卢聼茅聢搂莽聧聫聧忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聨氓聳職茅聢搂氓篓虏忙聽搂脙潞茅聤聤脩聟莽聼聠茅聳赂忙聸篓氓聙聥猫陇掳忙聨聭莽茅聳赂氓驴職氓聛聬莽禄聥氓聸垄潞?
    """
    if search_kind == "neural":
        implementation_paths = _SEARCH_MODEL_FINGERPRINT_PATHS.get(implementation_name)
    elif search_kind == "classical":
        implementation_paths = _SEARCH_ESTIMATOR_FINGERPRINT_PATHS.get(implementation_name)
    else:
        raise ValueError(f"unsupported search_kind: {search_kind}")
    if implementation_paths is None:
        raise KeyError(f"unsupported implementation fingerprint target: {implementation_name}")
    fingerprint_paths = tuple(
        dict.fromkeys(
            _SEARCH_COMMON_FINGERPRINT_PATHS
            + implementation_paths
            + _resolve_experiment_fingerprint_paths(experiment_config_names)
        )
    )
    return _build_file_fingerprint(fingerprint_paths)


def _load_cached_search_report(
    report_path: Path,
    *,
    cache_context: dict[str, Any],
) -> dict[str, Any] | None:
    """茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼忙聲聥莽聭聹氓驴聸氓鹿聭茅聤聫茫聜聟莽庐聸忙陇陇忙聴聡忙忙聬麓氓漏聡忙垄潞茅聬聼茅聜娄氓職聼氓漏碌忙陆搂茅聬拢莽聝陆氓录禄茅聤聣茂赂戮氓聨卤茅聴聛忙聳楼忙聶聸茅聧聼盲录麓氓聼聤茅聫聡茫聢隆莽聟聲茅聨录莽虏鹿忙茅聫聙氓漏職忙聼聲茅聧楼脩聝莽聙碌莽虏聶忙驴隆忙颅聦茅聳鲁脩聦氓聣職莽聰聙?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聳莽潞戮氓聺聴忙聼拢茅聤聫茫聢漏莽聞戮茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲陆鹿氓炉搂氓篓虏忙聽麓聬猫陇掳忙聨聭莽拢禄茅聬聨茂陆聨氓聛聜莽禄禄氓聴聸忙聼聲茅聧芦忙驴聡氓聶潞茅聳赂茂陆聟忙鹿卤莽禄卤忙聨聴忙聜漏氓庐聲氓聸搂芒聮聢茅聴聛茅聙聻氓卤陆氓垄虏忙陇陇莽聟陇茅聳潞氓露聨莽聛聬氓篓虏忙聶聺茅聳赂氓卤戮莽篓聢茅聨路忙麓垄禄茅聧聦忙掳卤氓垄聸莽录聛氓卢聦戮莽禄忙聜聶茅聬聭忙篓录氓聨卤茅聴聛茅聺聸脛聸茅聧聸茅聳禄氓潞隆莽聛職氓漏垄忙芒聜掳颅忙戮聵茅聬拢莽聝陆忙聜垄莽潞掳猫戮篓忙聶漏碌莽聜麓氓聻露猫聫陇茅聳赂氓卢聧聡氓庐聲氓陆聮茅聤聢氓聸露氓聲聨茅聴聜盲陆潞忙聡聯茅隆聲氓麓聭茅聬聰莽聟聨莽庐聽茅聳潞氓聸楼氓聻潞茅聬聯忙聸垄莽聟聧楼茅拢聨氓聢掳氓漏垄脩聞氓莽禄卤忙聨聯忙陆聫茅聤聤茫聝搂莽聭虏氓漏碌莽聜碌氓聨搂莽禄禄忙篓禄盲驴聤氓搂聺氓虏聤忙搂戮茅聴聛盲录聶莽露聙氓篓聟茂陆聡忙搂聬茅聨戮氓颅聵氓莽录聛氓莽掳卤茅聧聥忙聮禄氓麓路茅隆聯莽聠赂忙庐掳茅聴聛莽聜陆氓聞卤莽潞戮氓录鲁茅聰聲氓聜聸莽聟聫氓漏碌莽聜碌氓聛聯氓篓聟忙聮禄忙聼隆忙碌拢氓聭聤茅聳潞氓卤戮莽篓聯忙碌聽茂鹿聜氓麓聹猫茅聬聯盲戮聙忙垄潞茅聧聺茫聝楼忙聻聭忙驴聽茫聢隆茂鹿垄茅聧漏莽聜潞氓鹿聥茅聰聰猫聴聣芒聜聝陆忙聼隆忙戮露氓卢聛漏氓篓麓莽聤鲁忙聜聜氓忙麓聜路茅聬聴氓聽聺忙職聢茅聴聛猫炉隆氓聳聳茅聧聤氓聺聴氓娄麓忙赂職芒聜盲禄聥茅聬聬猫聦露芒聜卢氓聻聰聭茅聬聬忙聺聭莽聛娄茅聳驴忙掳露忙聼聸茅聤聢氓聴聴莽聛麓忙驴庐氓漏聝忙陇陆氓庐聲茫聞娄莽鹿聺茅聧聦忙掳录忙陆聯氓漏碌莽聜虏莽聯篓莽禄庐忙聮露莽虏禄茅聫聧猫聯聼茅聳驴忙驴聠茅聧聥氓聥颅莽麓聮莽聙拢氓聞楼茅聳鲁脩聦氓聣職莽禄聮禄氓鹿聝氓娄陇芒聜隆芦忙驴聥茅聤聫莽聤碌茫聜庐茅聴聜盲陆赂忙聬聫聢莽虏聶茅聨麓茂鹿聙莽聟聺茅聨录忙路聙聴茅聬陇芒聜聴聛氓聯聞忙麓篓忙驴麓卢茅聧娄莽拢陆茅聳赂氓卤戮莽聭搂茅隆娄茂赂陆茫聛聞茅聨麓忙驴聡氓職聡茅聮聵忙驴聥忙聼拢茅聤聫茫聢漏莽聞戮茅聳潞氓聥漏氓聞碌茅聫聦氓露聢茅聳赂忙聮麓氓虏赂忙驴隆莽聰赂氓麓聼茅隆聳忙掳卤茅聴聛莽聭掳莽聯篓莽禄禄氓露聠莽忙楼拢氓聠漏芒聮聭茅聳鹿猫聜漏氓聛聸茅聢搂忙麓聼茅聫聛氓聠庐茅聧驴聢莽聲潞氓漏碌脗掳氓聙聬茅聧聥忙聮禄忙聼聧茅聮聵氓忙搂赂茅聳鲁猫炉虏莽搂鹿茅聨录氓陆聮茅聤聫莽麓聺茅聴聛猫聤楼脡聭莽禄禄氓聥炉氓鹿聣莽禄聥忙卢戮氓聬聥氓漏垄猫路潞忙碌聽盲驴聤茅聤聢氓驴聸忙陋聬聨莽卢聸氓聸露忙聲聜茅聳禄忙篓录莽虏炉茅聧陇氓聠庐忙聼隆氓氓潞隆氓鹿聴茅聳鲁茅聰聥氓聻驴氓氓聽聲忙聜聶茅聫聣氓聻庐莽楼漏碌芒聙鲁氓垄陆茅聧聫忙漏聙氓录禄氓篓聭忙卢聭莽炉聙茅聳赂忙聞碌盲潞聻茅聬拢莽聝聵莽鹿聸?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    report_path : Path
        莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳鹿茅聴麓莽莽潞戮氓陆聮忙聼聼茅聴聜氓炉赂莽禄戮忙聝搂忙鹿卤茅聨虏忙聬聛忙聜搂茅聧芦忙驴聢莽聭篓莽录聜盲陆潞氓搂鲁茅聧聴忙聤陆氓录禄茅聬聰氓聟录芒聮聮茅聬聨莽聰碌忙驴隆盲禄聥忙垄潞莽禄聥忙氓聞聧茅聳赂氓漏聝莽鹿聢氓炉氓录麓茅聬聰猫聶鹿茅聴聛莽禄聵氓聤娄茅聧聯忙掳露忙聜碌茅聫聝氓聜聹莽拢陆氓篓麓茂陆聟忙聬氓虏聞忙聼聸茅聰聺氓驴聲莽搂聙碌氓忙聮聜垄忙陇聥氓潞拢莽聧聴聜盲陆赂氓茅聧聥氓聭炉忙聧垄茅聧聦忙露聶氓聫聫茅聳潞氓聸楼氓聻潞茅聢路忙聢娄忙聼聸氓漏碌氓聥芦氓聤職茅聳潞氓卢茅聫聦忙露職芒聜陆聝氓聤聦茅聢搂莽鹿聜茅茂陆聠莽鹿聺氓搂聵忙茅聴聛莽禄聵莽聛聫忙卢聯忙聲聧氓氓聠拢忙聰陆忙陇陇忙聴聙忙聻禄忙赂職忙露垄忙聼聸氓搂聴芒聜卢莽禄聽忙聽颅氓鹿聝茅聰聼莽聤虏莽忙驴聽茅聺聸氓聲聫聜芒聜聴聜盲陆潞莽虏炉茅聧聰忙聸聻茅聧聸聙氓陇聬氓鹿聵莽录聛忙氓聨卤茅聴聛茅聺聸茅聧聨忙麓聜隆茅聧聫莽垄卤忙庐?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓麓茂陆聢氓篓聟莽聤禄氓鹿聳氓篓拢氓莽禄戮忙聝搂茅聫聦忙聸垄氓麓录氓漏碌氓聸露忙莽录?
    cache_context : dict[str, Any]
        茅聴聜氓聜職氓聙路氓篓麓氓聸搂莽聲路茅聬垄莽莽陋聴茅聳鹿茅聜娄氓聳聳茅聧篓忙驴聥氓鹿聳氓篓拢莽聛驴聻氓聭搂碌忙路聡氓漏聡氓娄露茅聧聸忙芦拢莽录聜盲陆潞氓搂鲁茅聧聴忙聤陆氓录禄氓篓聭忙篓路忙忙驴莽聣聤茅聫聛忙聝搂芒聰聭茅聬聬氓聫聣氓聶陆氓漏碌莽聜漏氓聛聯茅聴聛氓聯聞莽聼聣莽禄虏氓聙聼莽聛聮茅聴聛氓聹颅氓篓麓莽聳炉茫聛聬猫莽禄禄忙露職芒聜露聶脙陋氓篓聢莽聤禄忙聼聸忙驴聽氓聠庐聻莽聙?茅聴聜氓聜職氓聙赂茅聧聤莽聝陆忙聜聻茅聰聲氓聜聸氓聞聭莽聭聹莽聣聢氓赂聮茅聫聞茫聝漏忙聼聸茅聫聡茫聞楼莽聛聽茅聳赂茫聞楼氓聙赂茅聹聣茅聳赂氓驴聯氓聬聥莽录聧忙聢娄忙聜聴氓搂聵忙篓聙茅聳潞氓卤禄氓聤聭氓忙聺聭脦聼茅聯聫茅聙聸忙聜拢忙驴隆莽聜陋氓聙陇茅聧聵氓陆聮氓麓聺茅聨麓忙驴聢忙職娄茅聳禄忙聮鲁莽掳露茅聫聙盲禄聥忙聼聸茅隆聬盲潞聺氓娄陇忙聴聢忙垄禄忙碌拢猫聴聣氓聬鹿茅聳赂莽聤鲁氓聤聴茅聧聮忙聨聯忙聝聰茅聤聫芒聞聝莽聫路氓漏碌脗掳氓聙聲茅聧聼氓录鲁氓氓聽聼莽聟聶茅聳禄忙聢聻脡聭茅聢路忙聞掳忙聼隆茅聧隆忙篓录莽聼聮茅聳潞氓虏聙氓聳聳茅陋聻氓聴聸氓录露茅聧聼氓聠漏忙聧聡茅聫聦茂陆聣氓鹿聵莽聮潞莽聝聵莽聛聢茅聴聛氓聯聞忙聹卢氓篓虏忙忙聧垄氓篓聭茫聢聽氓聻聜忙陇陇忙聴聜氓聨搂茅隆芦忙聮露莽麓聯茅聧聦忙掳卢芒聜聬聛氓麓聬茅聬聭猫聤楼氓聙驴茅聳驴忙聸聻氓聙赂茅聭赂氓聹麓聯忙碌拢氓猫聜漏忙聧垄茅聧隆忙露聶莽聟聲茅聳碌氓陇聵氓聳聳忙戮搂忙聼拢茅聨潞忙聴聙忙職漏莽录聛猫戮篓氓赂聻茅聢搂莽露聠茅聧聥氓聥颅莽虏莽禄禄忙驴聥氓聼聙茅隆聮盲陆鹿氓陋麓茅聳戮氓驴聯忙鹿卤茅聰聸忙驴聡莽鹿聸忙聺聢氓聣搂莽搂卢忙陇陇忙聸聼莽聣聲茅聢禄忙聮禄氓录庐茅聧聠潞氓聨卤忙驴聽莽聰碌氓搂麓茅聧聤氓聹颅忙聥鹿茅聰聸氓聥拢录忙陇陇忙聴聜忙聶聨鲁氓聥聜卢氓聻戮聵莽聙職莽茅聰聰氓聣搂莽聼聶氓氓聣聛茅隆楼忙掳露忙垄禄忙碌拢莽聭掳莽录聛猫氓聙鹿茅隆篓茫聝漏氓麓篓忙驴聽氓聥碌忙麓赂氓漏碌莽聤禄忙陆搂茅隆聭氓聴聸忙聜隆茅聬聰莽聟聨莽忙驴聻茂陆聨氓聣聶茅聬聫忙聼聸氓篓聠氓驴聰茅聳鲁猫搂聞氓聻驴茅聨卢聯氓录露茅聨麓莽聤卤忙忙驴隆莽聜陋氓聙陇茅聧聵莽聟聨脙潞茅茫聢聽莽庐聳茅聳禄忙聞卢忙聬禄聽莽聭掳脙鹿茅聰聺氓聭麓隆茅聳赂忙篓禄忙聧聡茅聨庐氓鲁掳茅聧聲莽聟聣茅聬聨猫搂聞忙麓聵莽禄虏聞茂陆聻氓漏碌氓卢職聥茅聴聜氓聽聼莽篓聮茅聬陇氓聥聰聭忙聨鲁茅聧聤脩聟氓戮聤忙碌聽氓卢聳庐忙陇陇忙聽芦氓聛聸莽潞戮猫炉虏茅聭赂茅聺聸氓茅聳禄忙聮麓忙麓聵茅聤聡氓聥鹿聡茅聧聯盲陆潞脡聭莽录聜盲陆虏氓聛聯茅聳鲁脩聦氓聣聸莽禄卤忙聮麓忙聥聟茅聨禄忙聨聧氓拢聲茅聳禄氓潞隆氓聨聹茅聧聥忙聮禄忙聼聸茅聫聡茫聞楼氓茅聳赂忙篓潞氓麓聢鹿忙聢娄忙聜聶茅聫聣忙聢聽盲禄麓茅聬聨猫搂聞忙麓娄茅聧聯莽聠录忙聜掳茅氓聵聣忙聴聢氓麓篓茅忙聝搂氓鹿聢茅聴聛莽聭掳氓聬聥茅聬拢氓麓鹿茅聯聫茅聙聸莽聞聤茅聳潞氓茂录聯忙聬麓茫聞楼莽聜聤莽聭聹忙驴聠茅聤聣芒聲聜忙聡職茅聳潞氓露聨氓聨陆茅聧聤莽聰赂忙聼篓氓漏碌氓聴聸氓脩聦氓聳聴茅聤聡氓聥录麓茅聢楼忙聫聮忙聝聢莽录聜盲陆陆茅虏赂茅聨鹿氓聸搂莽聲路茅聨潞忙聢陆漏忙驴陆聠莽聭拢茅聧篓盲禄聥氓录禄茅聬聰氓聟录忙聝聻忙陇陇忙聞茅聧篓氓聣聛莽虏職忙聺聢茫聞搂莽炉聙茅聳赂忙聸聵氓聤聦忙碌聹莽聜鹿聸茅聨麓莽聝聠盲录聠茅聳鹿氓炉赂氓聛聸茅聧?氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聧聸氓漏聝氓聫聫氓篓麓氓聟录忙陆聶茅聧篓氓聜聻忙聼拢茅聨戮氓麓聡氓虏赂茅聳潞氓莽聣聴莽禄禄忙露垄氓鹿聥茅聬聬芒聲聜茂录聢氓漏碌莽聜虏忙篓聤忙碌聹忙禄聝忙麓驴氓漏碌莽聤聻录氓篓聣忙露垄忙聲聝茅聰聲氓聜陇忙聲聤茅聳赂芒聞聝莽禄聽茅鹿聝忙聼隆忙戮露氓卢聛楼茅聳鹿氓聻露莽禄禄忙露垄氓麓聴茅聭虏芒聲聜氓聶庐莽录聜盲戮聡氓聳聳茅聬聴忙聮鲁莽聲路茅聨潞忙氓聙路茅聬聨莽聰碌茅陋聻氓露聣忙垄禄忙碌拢莽颅聥氓職聝茅聳赂茫聞楼氓麓卢茅隆聻庐脦聼氓篓聠莽聜卢忙炉聬茅聴聜氓聜職氓聙路莽禄聙盲戮聙氓鹿聳茅芒聭陆忙露聵芒聳聨茅聨麓莽聤碌莽聟聯茅聴聛猫氓麓聵茅脩聡忙聜聻茅聧篓盲潞聹茅聳鹿氓聯聞莽搂露莽聮聬盲录麓忙聼聸茅聬聰茅拢聨莽茅聳潞氓卤戮莽聸聨庐芒聲聟氓麓聵茅聧聶氓莽聶聛茅聳禄氓潞陇氓篓虏忙漏聙氓麓鹿茅聧搂忙楼聝氓碌聛忙驴隆氓聛聬莽潞戮氓聟录盲驴聤茅忙禄聝氓赂露忙楼聽莽聜虏莽聣聯氓莽禄聵莽鹿聺忙聬麓芒聙虏莽掳禄氓漏碌莽聜露莽禄聽忙聢聽氓聫聟茅聴聛氓聯聞莽搂虏茅聧聰氓潞篓茫聙聟茅聳驴氓卢莽聙鹿氓聣虏氓聳聲莽禄篓氓楼赂忙聼拢?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    dict[str, Any] | None
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿陆聲茅聴聜盲陆赂忙聬篓虏茫聞娄忙聟篓忙陇聥氓潞拢莽聼聠氓漏碌莽聤虏氓聙碌茅聫聙盲禄聥忙聼拢氓娄聯聞莽潞搂茅聳鹿氓聟录氓聤聶茅聫聦忙露職盲禄娄莽聮聡忙聫聮茅聴聛氓聯聞忙聹卢氓篓虏莽聠路莽聲路茅聬聯氓潞聵忙聰陆茅聳赂茫聝篓氓聼聳茅聰聼茫聝漏忙垄禄忙碌拢氓聯楼茂录聧莽录聛氓聻露氓氓漏職氓娄赂茅聢潞盲陆潞氓聨麓茅聴聛莽隆聟忙聫陋茅聴聞氓聥炉氓麓聭茅聨掳莽聣聢莽聟聶莽录聜盲陆鹿脙陋忙路聡脙鹿茅聬聵猫聧陇莽聲潞忙驴漏聞氓聞陇莽聙碌莽聟聨芒聳聨茅聨麓莽聤聵忙掳颅莽鹿聸莽聙聸忙莽聼陇茅聳赂忙聮麓莽篓聭氓聧卤茅聳鹿莽聣聢忙戮聵莽禄聽氓忙聜鹿茅聧楼莽職潞茅聳驴忙露聶莽虏聦茅聢鹿忙聢娄忙聜聶茅聫聣忙聢聽盲禄赂茅聴聛猫聧陇氓聳聠茅聧聨忙聤陆氓麓掳忙驴聽氓聜陇氓聬聥氓猫路潞茅聨路忙麓聼拢忙聬麓茂陆聠茅聴聜氓聥芦氓芦隆颅氓聻聼莽录聛忙篓潞氓录驴茅聴聛莽忙聹赂職忙楼聙芒聳聯忙碌聽氓卢聟聼茅聳鹿茅聜娄氓聨戮茅聧聶芒聜聴聛氓聭聤莽聭楼莽聙職氓聼聻茅聨麓茂赂陆忙聶聳赂忙聸篓氓聛聞茅陋聻氓莽鹿聺莽潞掳氓陇聦莽拢驴茅聳赂氓驴聰茂鹿垄氓氓麓聵茅猫聜漏氓聤娄氓娄聻茫聜聠氓赂聮莽聙職脨搂茅聫聦茫聝娄莽聟聲忙陇陇忙聞庐氓搂麓茅聧聰忙禄聢忙聼聧茅聳驴氓莽聛麓茅聳潞氓虏聙氓聳聯莽禄庐忙卢聯氓鹿聬忙聬麓茫聢聽茅聴聴聜盲陆鹿氓聣職茅聧篓氓聧聻脙潞氓漏聤氓聭庐茅聳鹿莽聝聵忙聼隆茅聧聥忙聮禄忙聲聬忙聬麓芒聙虏莽掳禄忙驴聽忙氓聳聳茅隆娄猫戮漏忙茅聨潞忙忙聲聥茅聳赂忙露卤氓聨录莽禄聽忙麓陋忙聜聴茅聧篓莽路虏茅聬聨忙掳录氓聳聴莽禄聜忙聨聴忙聲聜茅聧聯脩聟莽聟聯忙驴聽茫聢聽氓茅隆聽楼脣聣氓搂聵莽炉聙茅聳禄茫聢陇忙陆搂忙碌聽茂鹿聜忙聼聸茅聤聤茫聝搂氓聼拢忙陇陇茫聢隆忙茅隆垄忙聴聢氓麓录茅聬聰氓聭聤莽聫聳茅聴聜盲戮聙忙陆搂莽禄禄氓聽楼氓麓聬茅聤聢氓忙聻聛茅聳赂氓虏聙氓聛聻茅聬聯忙聸聼茫聛聞茅聨掳氓聣聛忙聜录莽录聛莽聤鲁脡聭茅聤聡氓聥鹿聵莽聙聸忙篓赂盲禄垄氓漏碌茂鹿楼氓猫聴聣莽聲路茅隆聬茂鹿聙莽陇聥忙陇陇忙聮麓莽篓聮茅聬聨莽芒聰聭茅聬聵忙庐驴忙職炉茅聳鲁脩聦氓聣聺氓篓聟忙聸垄氓麓掳氓芒聜聫聦莽聠录茅聬聯氓潞垄脨漏茅聴聛氓庐聽茅聧篓氓聻聲路忙聺聢莽聳聙茅聳潞氓聠漏氓聻露莽录聧氓虏聞忙垄禄茅聧聦忙卢聮氓聟聦茅聳鹿猫聶聧聛茅聤聡忙聞掳忙聜聶莽聙碌氓聯楼氓陆戮茅聴聛氓聯聞忙麓垄茅聧篓忙麓陋氓聼聸茅聨潞忙聡篓氓陆聮忙聲聬茅聧聸氓聞聯茅聳禄茫聢漏氓聻掳茅聬聳氓楼赂氓录禄氓篓聭茫聢聽茅聳驴忙露聵氓氓炉聞漏忙垄潞莽聮聡芒聙鲁忙聻聼忙驴隆莽聜露氓搂陇忙路聡氓漏聡氓鹿聳忙碌拢猫聜聺茫聞漏忙聼聧忙聺聻忙聥聦氓聟聦茅聳驴忙露聶莽虏聦茅聢鹿忙聢娄忙聜聶茅聫聣忙聢聽盲禄赂茅聴聛猫聧陇氓聳聠茅聧聤忙聸陆聻氓漏碌氓忙聶聺茅聳赂氓卤戮莽篓聢茅聨路忙麓陋忙垄潞茅聧聸氓漏聞莽庐聯茅聬聨忙掳录氓聣聼茅聧漏芒聜陇陇忙聨聠氓聛聯忙驴聽芒聜卢茅聳露聟莽聞碌忙陇陇忙聨聭氓聙禄茅聨鲁忙楼聟忙聼聸氓篓聭氓聧聻莽聛拢茅聳禄忙聨聭茫聜赂氓搂?``None``茅聴?
    """
    def _path_exists(value: Any) -> bool:
        path_text = str(value or "").strip()
        return bool(path_text) and Path(path_text).is_file()

    def _matches_with_classical_fingerprint_compat(
        cached_context: dict[str, Any],
        current_context: dict[str, Any],
    ) -> bool:
        if cached_context == current_context:
            return True
        if "method_name" not in current_context or "model_name" in current_context:
            return False
        if cached_context.get("method_name") != current_context.get("method_name"):
            return False
        cached_impl = cached_context.get("implementation_fingerprint")
        current_impl = current_context.get("implementation_fingerprint")
        if not cached_impl or not current_impl or cached_impl == current_impl:
            return False
        cached_copy = dict(cached_context)
        current_copy = dict(current_context)
        cached_copy.pop("implementation_fingerprint", None)
        current_copy.pop("implementation_fingerprint", None)
        return cached_copy == current_copy

    if not report_path.is_file():
        return None
    cached_report = _read_json(report_path)
    cached_cache_context = dict(cached_report.get("cache_context") or {})
    if not _matches_with_classical_fingerprint_compat(cached_cache_context, dict(cache_context)):
        return None
    if report_path.name == "candidate_report.json":
        scoring_runs = list(cached_report.get("scoring_runs") or [])
        score_vector = list(cached_report.get("score_vector") or [])
        if not scoring_runs or not score_vector:
            return None
        train_report = dict(cached_report.get("train_report") or {})
        checkpoint_selection = dict(cached_report.get("checkpoint_selection") or {})
        if train_report and (
            not _path_exists(train_report.get("report_path"))
            or not _path_exists(train_report.get("checkpoint_path"))
            or not _path_exists(checkpoint_selection.get("selected_checkpoint_path"))
            or not _path_exists(train_report.get("training_flow_contract_path"))
            or not _path_exists(train_report.get("loss_diagnostics_path"))
            or not _path_exists(train_report.get("epoch_predictions_vs_targets_path"))
        ):
            return None
    return cached_report


def _normalize_baseline_scoring_metrics_for_cache(
    baseline_scoring_metrics_by_experiment: dict[str, dict[str, float]] | None,
) -> dict[str, dict[str, float]] | None:
    """Resolve baseline summaries for report consumers."""
    if baseline_scoring_metrics_by_experiment is None:
        return None
    normalized: dict[str, dict[str, float]] = {}
    for experiment_id, metrics in baseline_scoring_metrics_by_experiment.items():
        normalized[str(experiment_id)] = {
            str(metric_name): float(metric_value)
            for metric_name, metric_value in dict(metrics or {}).items()
        }
    return normalized


def _resolve_requested_public_datasets(
    requested_public_datasets: list[str] | None,
    *,
    skip_public_benchmarks: bool,
) -> list[str]:
    """茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聫茅聢搂莽聞娄莽禄禄忙驴聥氓麓聮氓娄陇盲陆潞盲潞聶忙驴聽莽聰碌氓聛聻茅聧篓氓麓鹿氓篓虏莽聳戮忙驴聽茅聺聸茅聬聯茫聝漏忙聼隆茅聧楼芒聲聛盲禄聹茅聳鲁脩聦氓聣職茅聨聜職茅聧聸氓漏聞莽聭鹿茅聳鲁脩聦氓聣聼氓氓漏職氓录麓茅聤聫莽聤禄氓聰露氓漏碌莽聤虏莽聛職茅聧聰忙聽芦莽聭搂莽录聜氓聜職氓聙路茅聭鲁猫聢碌氓聢聤莽聙碌氓聯楼莽陇聤氓篓聯忙掳拢氓聛聯莽聙?忙驴聽莽聰碌氓搂路茅聫聛氓聭聤忙聟篓茅聨戮氓聙露忙驴庐忙篓驴忙鲁聸茅聧職氓陆聮忙聼聧猫茅聧聯莽聠路茅聯聫氓聹颅莽聼聶茅聳赂忙聽碌莽鹿聺莽禄露忙聺聭茅聤聢氓聴聵猫聜聽茅聳赂茫聝聭氓聭颅氓鹿聮茅聴聜盲陆陆氓庐聲忙漏聙猫陇聫莽禄庐氓漏職忙聲聬忙戮露氓氓聫聠茅聴聛氓聯聞忙麓娄茅隆篓氓聭庐莽娄聧茅聧聯脩聟莽拢陆氓篓麓茂陆聠氓陆聝忙碌?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬氓庐聞忙聡聤氓陆聮氓麓露茅聤聤茂陆聞莽虏聯茅聴聛氓陆聮忙拢驴莽禄聙盲戮聙氓麓鹿茅聧聦莽聜麓莽聟聲忙陇陇氓聻碌氓篓聟忙漏聙忙聼聸茅聤聢氓聴聵莽茅聳潞氓虏聙氓聳聳茅隆垄忙卢聯氓娄赂茅聤聣茂赂陆忙聜聶茅聴聜盲陆赂氓麓卢氓篓虏茂赂戮忙聧垄氓漏碌氓卢驴聻忙聢聻猫聣戮茅隆拢氓聽聲忙垄潞莽禄聥忙卢戮氓聞聫茅聳驴忙聸篓盲潞聹茅隆芦氓驴聰莽鹿聺氓搂聵茂录聹氓漏碌茂鹿漏茅聧聫忙漏聙氓麓聭茅聨戮氓聣聛莽聼聶茅聬聽氓聸漏氓聛聯莽禄聥忙聢拢莽聛聝茅聴聛茅聙聻氓卤陆氓垄虏茅聳禄氓驴聯茅聢鹿忙聢娄忙聜聶茅聧聶氓忙拢隆茅聴聛莽聭掳忙聭聹茅隆聽聼忙聶楼茅聴聛氓聯聞猫茅聧聨忙聼楼忙聜隆茅聤聣茂赂戮莽炉聙茅聴聜氓聽聼莽篓聮茅聰聸氓露聠盲驴聤茅聨禄忙聨聰莽搂潞茅聳潞氓卤戮莽篓聯茅聢搂莽露聠茅聧聯忙卢聯氓聼垄茅聧芦茫聝娄莽聟聸茅聬聫莽聜虏氓垄陆氓篓虏忙聮聼聸茅聢潞盲陆赂莽聙鹿忙聸聼茅聨聼聙茅隆聮盲陆潞忙陇陇氓驴聯莽聣聠莽禄聽忙聽篓忙聟篓氓娄聻猫盲潞戮茅聬聨猫搂聞忙麓聳茅聬聳氓楼赂茫聙聛氓娄陇盲陆鹿氓茅聳赂忙驴聠氓芦禄聽盲录麓忙垄禄茅聧聦忙卢聭氓漏垄莽聭掳茂鹿庐聲忙聢娄氓麓篓莽聙聸忙篓潞忙庐聻茅聴聛莽禄聵氓聤娄茅聧聯忙露垄茫聙聟茅聫聦茫聝娄莽聟聲茅聳鹿莽聜卢茅聨鲁忙聞掳忙聲聧氓漏碌氓卢聮聭莽录聛氓氓炉搂氓漏聙盲录麓忙聼拢茅聬聰忙驴聠忙鲁職茅聧楼忙聶聺茅聳赂氓卤戮莽篓聯茅聧聵猫聴聣芒聰聭忙聨鲁茅聧聤忙聞掳氓麓聭茅聨戮氓聣聛莽拢录莽录聜盲陆鹿脙陋忙驴露聣忙聼聸茅聢鹿氓聻聛漏忙陇陇忙聮鲁茫聜赂氓庐聲氓楼赂茅聰聺氓聜聸莽聙赂茅聴聜盲陆陆氓茂鹿聜茂陆聻忙戮露氓篓聮茅赂驴莽庐聸氓篓聭忙茂录聼茅聴聜盲戮聡氓聣聶莽禄聣莽聰赂忙聜隆茅聫聡莽聠潞莽茅聳鹿氓颅聵莽聭楼茅聬聫忙聼聧茅聳驴氓卢聛麓茅聳潞氓虏聙氓聳聯莽禄庐忙卢聯氓鹿聬忙聬麓茫聢聽茅聴聴聜盲陆赂莽聳聡氓娄聥炉脣聡茅聬垄莽猫聯?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬氓庐聞忙聡聤氓陆聮氓麓露猫陇聫茅聫聠茂鹿聜忙聼拢茅聤聫茫聢漏莽聞戮莽禄戮氓聣搂莽虏炉莽禄禄忙露垄氓鹿聥茅聬聭氓聴聶氓聤炉氓漏碌莽聜麓氓聻露茅聰聲芒聲聟忙聜隆茅聤聣芒聲聜莽聟聲忙陇陇忙聞漏氓聙聲茅聫聥氓潞篓忙聼聧猫陇聹茅聧聯忙卢聯芒聙娄茅聬聭猫聴聣猫聯聼茅聳碌氓篓戮忙聶聸莽禄芦氓露聣忙聼聸茅聤聣盲禄垄茅聳鹿忙聧聬莽搂聬聼忙聣鲁忙聰娄茅聳赂氓聳聮茅聧聥忙聮鲁芒聰聭莽聙拢忙聭聲氓漏碌莽聜麓氓聻聧篓莽聭掳忙聲庐茅聴聜盲陆潞忙聡聯茅聬隆茫聞搂忙聧垄茅聧聦莽聝聠忙聲聜茅聳鲁氓聯聞忙聡聫茅聢路忙聨聴忙聼聸莽聛聻氓聣搂忙聡聟忙陇陇茂赂录莽庐聯茅聫聦莽聠潞忙聥聽猫陇聫莽禄隆芒聜聴聛猫陆掳莽茅隆聭氓聭麓芒聙聰茅聧聬脦搂茅聨录莽虏鹿芒聜麓聡茅聴聜盲陆潞莽禄禄忙聢娄忙聲聥茅聴聛盲录聶莽碌驴茅聧聫忙漏聙莽聧聬麓茫聞娄氓炉聽氓漏垄猫路潞莽聼聢茅聧聼氓聸搂芒聰聭茅聬聬猫聢碌氓陆搂莽录聜氓露聛氓聛聯茅聴?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    requested_public_datasets : list[str] | None
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓聙驴茅聳驴忙聸聴氓聛聯忙陇陇氓聲麓氓庐聲氓陆聮茅聧聸氓搂潞茅聴聜盲陆潞茅聧聲氓聻芦莽聲露莽录聜盲陆虏氓聛聯氓漏垄猫路潞氓陇聡茅聧聥忙聮禄莽聧庐茅聧篓氓氓娄聻茫聜聫氓炉聦茅聧篓猫路潞忙碌录忙聺聭脦篓茅聳驴忙聳驴氓垄陆茅隆聰忙聸垄忙垄潞茅聬聼忙聣聭聽茅聳禄莽聠麓氓聞碌茅聨庐忙漏聙氓聼隆茅聧聸氓聨卤氓漏碌莽聜虏忙忙楼聽氓驴聸莽聟氓聥聟聸莽聙鹿芒聜卢茅聢搂氓麓掳茅聫聧脩聟忙職娄茅聳碌氓篓戮忙聶漏茅聫聛氓露聣忙聼拢茅聨掳脣聣茅聳鹿茅聰聥忙陇陆氓搂聤忙麓陋忙拢聝茅聢潞盲陆潞忙搂拢茅聴聛氓聭聤茅聧聥忙聞掳氓麓拢茅聧聸氓颅聵芒聮聭茅聴聜氓聽聼莽篓聯忙戮搂忙聸聼茫聛聞茅聰聰氓聻庐氓碌聠莽聙鹿忙聸聼氓聻驴茅聧漏芒聞聝忙聥聟茅聧聲茅聳潞盲陆鹿氓聤聶氓庐聲茫聞漏茅聬聰氓聣虏氓聰鹿茅聴聜氓聜聶猫聝露茅隆聽聼茅聧聺茫聝搂莽聼聯莽聭聹莽聣聢氓赂聮茅聮聝莽聠赂莽鹿聸茅聨麓忙卢聫莽聛漏茅聳赂忙氓篓聤茅聫聦忙聸聼莽鹿聸茅聧篓氓莽录聜盲陆碌氓聤聦茅隆颅莽聝陆氓鹿聝氓庐聞茅聜娄莽聟楼茅聳赂忙聸篓氓聣聸茅聧聭芒聙鲁芒聰聭茅聢陆氓聴聴盲潞聺茅聳禄芒聲聤氓聻露氓庐聲忙麓陋氓聼聙茅隆聮盲陆碌盲潞聹茅聳鹿氓聯聞莽搂露莽聮聬盲录麓忙聼聸茅聬聰茅拢聨莽茅聳潞氓卤戮莽聸聨庐芒聲聟氓麓聵茅聧聶氓莽聶聛茅聳禄氓潞陇氓篓虏忙漏聙氓麓鹿茅聧搂忙楼聝氓碌聛忙驴隆氓聛聬莽潞戮氓聟录盲驴聤茅忙禄聝氓赂露忙楼聽莽聜麓氓聤聲茅聢鹿忙聢娄忙聜聶茅聭赂茅聺聸忙露搂莽录聜盲陆陆氓录聯莽禄卢聯莽聧聫聧忙聲聝茅聳驴忙聴聡忙茅聢搂忙聲禄茅聨鲁掳芒聳聨猫鹿聡忙聯聝盲禄戮茅聳禄氓驴聯忙碌職茅聧聥氓聴聶忙茅聨潞忙氓職聥氓篓麓茂陆聟氓聲芦茅隆楼忙驴聠茅聤聢氓潞隆盲潞聹莽录聛莽禄聵脙聬拢茅拢聨莽虏聶莽聮聡莽聜卢忙拢聝茅聧漏芒聜陇陇忙聨聭氓氓庐聯盲戮聙忙聼聲猫鹿聡氓篓戮氓聛聯茅聭录莽聲戮茅聴聜盲戮聙忙陆搂茅聬聴氓聴聸氓麓聬茅聧聸氓漏聞氓娄陇茂陆聟氓聲聢路忙聢娄忙聜鹿茅聧楼莽職潞莽录聛莽聤鲁莽聟隆茅聫聦茂陆聣忙聜垄茅聧聶氓莽聫職氓娄陇莽聤碌氓聛聸茅聧聼盲录麓氓鹿聭茅聧聲氓聛聵茅聳鲁芒聲聛氓聳職氓篓录忙聴聢忙垄禄忙碌拢氓聭聤忙聝聢茅聬聽氓聸漏氓聛陇氓庐聲忙楼聟忙聼篓茅隆聳忙露聶莽聫職氓漏碌茂鹿楼氓盲禄聥氓录禄茅聧聸氓颅聵氓聙路茅聳录猫聣掳忙鲁聸茅隆聫忙垄潞茅聧聸氓篓聮氓鹿聴氓氓聭聳鹿莽聝聵忙垄禄莽潞戮氓聟录盲驴聤茅忙驴聠氓聬鹿茅聳禄茫聞搂莽隆碌one`` 茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聟忙職娄茅聳禄茫聢垄莽潞戮氓漏職忙聼拢茅聨掳忙職漏茅聳禄莽聭漏茅聬聯氓麓露茅聤聤脩聙忙搂聢莽录聜盲陆虏氓聛聯氓漏垄猫聢碌氓聤聲莽禄聽猫搂聞莽鹿聸茅聰聺氓潞隆氓垄漏碌茂录聞氓聨搂茅隆漏氓楼赂氓麓篓茅忙露聶忙聳聙氓娄聻茫聜聠忙垄禄茅聧聴忙聤陆氓鹿聥茅聬聬忙忙碌聞氓漏碌芒聵聠氓聻碌茅聤聠芒聜卢茅聳潞芒聜卢?茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聟忙職娄茅聳赂氓聛聟茅聧聶氓驴聸忙聼拢茅聨麓茂陆聠莽禄戮忙聝搂忙陆隆茅聫聦忙聸垄氓麓录氓漏碌忙聞颅脨娄茅聴聛忙聬聻氓聤聦茅聧聤氓聺聴氓录禄茅聰聺氓茅聳碌氓驴聤忙聶聺茅聧聰忙聬聛忙垄潞氓篓虏猫氓鹿聴忙陇陇茫聞楼茂鹿陋茅聨驴聽茅聰聲芒聜卢茅聬聬氓漏聡忙聼聲忙驴聻脩聙氓聛聯氓漏碌忙麓猫聶鹿莽麓聮氓娄聝聵猫炉聻莽录聜盲陆潞氓搂碌茅聬聴莽聤虏忙驴聽忙掳卢氓聧鲁茅聳禄忙聴聡忙茅聬娄氓聽楼忙拢聟茅茫聢隆氓忙驴聻氓聫聣莽聣聽茅聨炉氓卤陆芒聰聭莽聙拢忙聥潞?
    skip_public_benchmarks : bool
        茅聴?茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙碌拢茅聺漏猫聙驴茅聫聣氓聽職忙聶聺茅聬聹氓忙聼隆茅聧聬茫聞搂氓茅聳禄忙聮麓莽聭娄茅聤聡氓聥录陆茅聤聤茫聞楼氓娄聻茫聜聟忙碌職茅聧聥氓漏聡氓录禄氓庐楼氓聽芦忙陋篓茅聴聛氓聭聤忙聦禄氓搂聵茫聞搂氓聢聮氓篓聠忙聮鲁莽忙驴隆茫聜聞莽聞娄氓篓聟忙聴聙莽鹿聺茅聬垄茅聺聸脨聻茅聳赂茫聞搂氓聨搂茅聢禄忙聳驴芒聵聣氓猫戮漏莽聟聥茅聬聼忙禄聟氓垄聴莽聰炉忙聨聰莽虏禄茅聫聧氓聳聬茅聳禄忙莽聣聠莽禄聴忙聨聭脙鹿茅聭虏芒聲聛芒聳聢茅聳潞氓卤戮莽聸忙聮娄赂茅聤聣茫聝搂茅聴聜盲陆陆莽聙碌忙聨聴氓麓掳氓猫戮篓茅聳鹿忙聝搂莽聯篓忙驴莽麓聮氓篓聭忙漏聠氓聞聫忙驴聻氓聽芦氓聫聝氓忙麓麓芦茅聤聣茫聝娄莽聙戮茅聴聛莽聭掳忙聠隆忙驴聻氓漏聡氓鹿聝忙碌录忙聬聲聤莽禄戮忙聺驴氓麓聨庐忙聢聻莽鹿聺茅聤聫茂陆聠莽隆戮茅聳驴忙聸庐聯氓庐聲忙驴聥忙聲聝茅聢搂芒聙聰茅聧聬脦搂茅聳赂芒聞聝茅聬聼茫聞漏忙聼拢忙聬麓茫聢聽氓職卢茅聳赂忙聮麓莽聸聧漏芒聜陇陇忙聨聧忙碌聡忙戮鹿忙篓禄茫聛聞茅聰聰猫炉虏莽聛娄茅聳鲁茫聝炉氓娄聡茅聨鹿氓聥娄赂茅聰聲芒聜卢莽潞戮忙垄潞莽录聛忙忙驴聻氓聟录莽聭娄莽聙碌氓楼赂氓鹿聡茅聬聴氓聽聺盲禄颅氓漏碌莽聤虏茂鹿陇莽聙職忙茅聧聶氓莽鹿聠忙陇陇忙聞露莽碌驴氓篓虏忙聮庐忙聼聸茅聢鹿忙聝搂盲潞戮忙驴隆莽聜陋氓聙聳莽聰炉氓漏聞忙茅聰聼莽聤虏莽聳聼芦氓聸篓莽篓聮茅聬聯忙聸垄忙聼拢氓娄炉氓聯聞忙職卤忙驴聻忙聬颅氓聳職茅聢搂?benchmark茅聴?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    list[str]
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜麓莽聣聽茅隆垄忙聸垄忙聲聜茅聮聵氓莽陆聲茅聴聜盲陆赂忙聠隆茅聧聰茂鹿聜氓麓掳茅聫聧氓漏碌莽聤鲁莽垄聧茅聬聯忙卢聯忙聼聼茅隆聳氓聴聴氓聳職茅聫聠茫聢垄莽麓聮茅聬聬莽忙驴隆氓氓漏聙盲戮聙忙垄潞茅聨赂氓潞拢莽庐聯茅聳鹿氓聠虏莽鹿聵茅陋聻氓陇聦莽虏聶氓篓聠氓聣搂氓聰碌茅聬聼忙禄聝氓庐聲氓陆聮忙聬鹿莽聭聶氓聥芦氓潞聤氓漏碌莽聤禄忙陆搂茅隆聭氓聭麓忙聜聻茅聧篓盲潞聹茅聳鹿莽聝聵氓聻碌茅隆聫芒聲聟忙聼拢猫鹿聡忙聳驴芒聳聲茅聳潞氓虏聙氓聳聬茅聰聸忙聞聥聟茅聧聺氓聥芦忙驴聽莽氓聣聺茅聤聡忙禄聢氓麓聺茅聨麓忙驴聬忙陆?茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聟忙職娄茅聳赂氓聛聟茅聧聶氓驴聸忙聼拢茅聨麓茂陆聠莽禄戮忙聝搂忙陆隆茅聫聦忙聸垄氓麓录氓漏碌忙聞颅脨娄茅聴聛忙聬聻氓聤聦茅聧聤氓聺聴氓录禄茅聰聺氓氓聛聞莽录聛氓卢娄芦氓潞篓忙垄潞茅聧聺氓聥碌氓聞聬忙驴隆氓聲麓氓氓漏職忙聜垄茅聧聸氓漏聞莽搂露茅聴聛氓聭聤忙聦聠茅聧聸莽录聜氓聜職氓聙路莽聮聛忙聼楼氓麓聭?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬氓庐聞忙聡聤氓陆聮氓麓露茅聤聤茂陆聞莽虏聯茅聴聛氓陆聮忙拢驴莽禄聙盲戮聙氓麓鹿茅聧聦莽聜麓莽聟聲忙陇陇氓聻碌氓篓聟忙漏聙忙聼聸茅聤聢氓聴聵莽茅聳潞氓虏聙氓聳聳茅隆垄忙卢聯氓娄赂茅聤聣茂赂陆忙聜聶茅聴聜盲陆赂氓麓卢氓篓虏茂赂戮忙聧垄氓漏碌氓卢驴聻忙聢聻猫聣戮茅隆拢氓聽聲忙垄潞莽禄聥忙卢戮氓聞聫茅聳驴忙聸篓盲潞聹茅隆芦氓驴聰莽鹿聺氓搂聵茂录聹氓漏碌茂鹿漏茅聧聫忙漏聙氓麓聭茅聨戮氓聣聛莽聼聶茅聬聽氓聸漏氓聛聯莽禄聥忙聢拢莽聛聝茅聴聛茅聙聻氓卤陆氓垄虏茅聳禄氓驴聯茅聢鹿忙聢聻氓鹿聤茅聳赂氓漏聝忙麓聼茅陋聻氓漏聝忙聝聰茅聰聺氓聸娄莽卢聣茅聴聛氓聯聞莽篓聛茅聧聵盲禄聥忙聜隆氓篓聭茫聢隆莽聟聲茅聳碌氓莽碌陆茅聧聰忙掳露忙聼拢猫鹿聡氓漏聝忙芦聡茅聳禄茫聝潞陋茅隆垄忙篓潞芒聵聣氓娄炉莽聭掳茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅茅聢路忙掳颅忙陇陆茅聫聛氓聠漏忙聥聟茅聧聺茫聝搂芒聳聲茅聴聜?

    茅聴聜氓聜職氓聙路氓篓麓氓聸茅聢搂莽篓聢茅聫聟莽聤禄茫聙聛氓篓聠忙聮鲁氓聠聙忙陇陇忙聮露氓聛聼茅聬聸茫聝漏忙垄潞茅聧聺氓聥颅芒聳聣茅聳赂忙篓驴氓聛聬忙戮鹿忙聸隆茫聛聬茅聨潞忙聮露氓聨碌茅聴聜氓聜職氓聙赂茅氓麓聺氓氓陇聥莽聟聲?
    ------
    ValueError
        茅聴聜氓聜職氓聙路氓篓麓氓聸搂莽聲路茅聬垄莽莽陋聴茅聳鹿茅聜娄氓聳聳茅聧篓忙驴聥氓鹿聳氓篓拢莽聛驴聻氓聭搂碌忙路聡氓漏聡氓娄露茅聧聸忙芦拢莽录聜?忙驴聽莽聰碌氓搂路茅聫聛氓聭聤忙聟篓茅聨戮氓聙露忙驴庐忙篓驴忙鲁聸茅聧職氓陆聮忙聼聧猫茅聧聯莽聠路茅聯聫氓聹颅莽聼聶茅聳赂忙聼聜聟忙聝聢忙驴隆莽聜聙聭茅聧聨忙聼楼氓鹿聭茅聧楼莽庐聽茅聳禄忙聞庐氓聞陇茅聬聮猫聜漏芒聜卢氓鹿驴氓聞卤茅聨鲁忙聞掳氓麓聺茅聨戮芒聮聭茅聳赂忙露聵茂鹿陇忙驴鹿聙脙鹿氓漏聤氓聥颅氓垄碌莽聙碌忙聜聜氓炉禄聥茅聬聰氓掳聛氓聛聯茅聧篓忙庐陆茅聳禄忙聞掳聶忙碌聽茫聢陇忙聼聸茅聫聝芒聜庐聙忙聬聛氓录禄茅聬聰莽聟聨莽陇聝茅聳录莽垄卤氓聣聸茅隆聰忙聨聴忙垄潞莽录聛忙篓禄忙聝聞茅聳赂忙聮露氓聳聨茅茂陆聠莽鹿聺氓搂聵忙芦聹茅聴聛氓聣虏氓聛聮茅聧聥氓聭麓氓聼垄忙戮露氓莽炉聙莽禄戮莽聣聢脡聭茅隆漏聡忙聼聸莽聙拢茅聳鲁脩聦氓聣職茅聬颅氓麓隆茅聨露猫聤楼氓聛聵忙陇陇忙聸聴氓聛聯莽聙鹿忙聸聼茂鹿垄茅隆垄忙卢聯忙聜隆忙聬麓茂录聞茫聜聴聜氓聜聶莽路職氓篓录脩聟忙聧垄茅聨潞忙聮麓氓聫聫茅聳禄忙聢聺茅聢搂莽路職氓庐聲茫聞搂芒聜卢忙露聶脟職茅聨路忙麓垄潞茅聧聸氓漏聞莽庐聯茅聬聨忙掳录氓聨陆氓炉掳氓聥炉氓麓聼茅氓聥聟拢氓庐聲氓聽聲氓娄赂茅聰聰猫陆掳猫聶聨茅聳禄氓潞陇氓篓虏氓驴聸氓麓鹿茅聮聵氓陇聵氓職聲氓漏碌莽聤鲁莽垄聧茅聫聟忙聫聮莽聮潞氓楼陆茂驴聽氓聼隆茅聧聸氓聨驴聽茫聢拢忙鲁聸茅聬聴氓聴聸氓麓聺茅聰聲氓聜聸莽聟聸莽聙鹿芒聜卢莽聙拢莽聵聢茅聬聨猫忙麓聳氓庐聲氓聼垄忙聬麓茫聜聡莽聳聙茅聳潞氓聠篓氓聙路茅聳颅芒聰?氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄忙驴聻忙聢聰忙聦聥茅聤聤芒聲聛莽聞碌忙陇陇忙聨聭氓聙禄莽潞戮氓楼赂忙聼拢氓娄聦聨莽聫聬聫氓驴拢氓聨陆茅聤聡氓聥炉氓录麓茅氓聴聶脨聹氓娄陇莽聤碌氓聛聸茅聰聲茫聜聣芒聙娄茅聤聤芒聲聛茅聠聡茅聳禄忙聸職莽聞娄茅隆楼氓聽聼莽鹿聺茅聬垄茅聺聸盲禄娄茅聳赂茫聞楼氓聬聥茅聤聡忙聴聢氓麓聤聣脩聟茅聴聛氓聯聞忙麓垄茅聧篓忙麓聼聸茅聨潞忙聡篓氓陆聮忙聲聬茅聧聸忙職聢茅聴聛氓聯楼氓聳聯茅聧聥茫聜聣氓录禄氓篓聭茫聢聽脦漏茅聳碌氓氓聞鲁茅聧聵莽麓聯忙碌拢莽篓驴芒聜卢氓聯楼莽虏聰茅聬垄莽聰碌莽聼聣茅聳鹿莽聝聵忙聼隆茅聧聥忙聮禄忙聲聬忙聬麓忙驴聢忙聲聳茅聴?
    """
    if skip_public_benchmarks:
        return []
    if requested_public_datasets is None:
        return list(_SUPPORTED_PUBLIC_DATASETS)

    normalized: list[str] = []
    for dataset_name in requested_public_datasets:
        normalized_name = str(dataset_name).strip()
        if not normalized_name:
            continue
        if normalized_name not in _SUPPORTED_PUBLIC_DATASETS:
            raise ValueError(
                f"unsupported public dataset '{normalized_name}', expected one of {_SUPPORTED_PUBLIC_DATASETS}"
            )
        if normalized_name not in normalized:
            normalized.append(normalized_name)
    return normalized


def _collect_full_paper_run_contract_violations(
    *,
    requested_public_datasets: list[str],
    skip_public_benchmarks: bool,
    reuse_search_cache: bool,
    allow_incomplete_paper_run: bool,
    requested_device: str,
    requested_neural_search_profile: str,
    lstm_epochs: int | None,
    liquid_epochs: int | None,
    batch_size: int | None,
    eval_batch_size: int | None,
) -> list[str]:
    """忙驴聽莽聰碌氓搂路茅聫聛氓聭聤忙聟篓茅聬聭氓芒聜聜聻忙聦聤忙戮鹿芒聲聥莽露聛氓庐聲茫聞漏氓录露茅聨麓莽聥聟莽麓卤茅聴聜盲戮聙芒聜聜虏猫聥戮搂忙聮聼隆莽聛聻氓聣搂茫聙聮茅聳鲁脩聦氓聣篓莽录聧氓聴聸氓麓聭茅聧聸忙職娄莽聙鹿芒聜卢茅聧聲氓聨赂?茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聟忙職娄茅聳赂氓聛聟茅聧聶氓驴聸忙聼聼茅聨聺驴脨聯茅聳赂茫聜聟氓芦掳茅聫聦忙露垄茅聰聺氓聴聶莽录聛莽聜聞鲁氓篓录茂驴聽氓录禄茅聫聡茫聢聽茅聠聡忙驴聽氓聻芦氓聤聳莽卢聭茅聴聜盲陆赂忙聠隆茅聧篓猫搂聞莽鹿聸茅聢搂忙聼隆莽聛聻氓聣搂忙麓麓忙莽聜虏忙麓颅茅隆垄忙漏聙忙聜垄莽禄庐莽聭掳盲潞戮茅聳赂忙聞碌氓聳聴茅聢路忙聨聭脙鹿茅聰聺氓聸篓氓職聙忙陇陇忙聸聼莽虏聨莽禄卤忙聨聰忙聥聽茅聨禄忙聨聠氓聛聯茅聧搂忙楼聝氓碌聛氓漏碌莽聤虏忙麓陇莽禄聽忙露聶茫聛聞茅聫聝氓聸篓茅聳赂忙篓潞莽露聤氓忙麓聼聟茅聬聬猫聦露氓芦掳氓脩聞忙聦鲁茅聫聦?茅聴?茅聴聜氓聜職氓聙赂茅聧聤莽聝陆忙聡聴茅聧聸聵芒聜陆鹿莽鹿聺茅聧楼脩聟茅聬聨氓聟聣氓聛聯茅聳赂忙聸篓氓聣聸茅聧?茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聦忙碌聳茅聳碌氓篓戮忙聶聹茅聢路忙聴聜芒聜鹿驴氓聞卤茅隆娄氓拢聲莽聭聶氓聥炉聙茅聴聜氓聽聼盲戮聴茅聧聨忙聢聽芒聜卢猫搂聞忙聦路莽禄露忙掳露氓鹿聝氓娄陇氓聭聶虏茅聨录莽虏鹿氓聣聺茅聬聫氓露聣忙垄潞莽禄聥氓聦隆氓路楼忙陇陇氓聟聵氓氓漏職忙聲聝茅聢搂莽聛聮忙驴聻忙聮麓氓聡鹿茅聧篓猫戮漏莽聭搂茅聴聜盲陆陆氓麓碌忙驴庐氓露聣氓麓聵茅聨聝氓莽录?茅聴聜氓聜職氓聙赂茅聧聤莽聝陆忙聡聴茅聧聯盲陆赂氓聻聺忙陇陇忙聽聛聭茅聢搂氓聲鲁莽禄聽忙露聵芒聵聣氓娄炉莽垄卤忙搂露茅聴聜盲陆潞莽虏娄芦氓聠聣茅聨戮猫炉虏忙卤虏茅聳驴忙聴聡氓搂陇茅聧聶氓驴聰莽鹿聺茅聴聢忙露聶茅聳禄氓驴聸盲潞露茅聫聦茂陆聣氓娄露茅聧聦忙掳芦氓拢聲茅聴聜氓聜職氓聙路莽禄露忙掳茅聫聛氓露聢莽庐聸氓篓聭忙篓潞莽陋聸茅聳禄氓潞垄莽篓聮茅隆颅氓聸麓垄茅聧聸氓漏聞莽炉聙茅聳碌氓驴楼莽碌戮莽潞颅莽聜驴聤茅隆聬茂陆聡忙聡聝茅聳鲁猫炉虏莽搂鹿氓庐聲氓聧聻芒聵聣氓篓聠忙聢聺氓鹿聬茅聴聛猫莽鹿聮茅聧聥莽聤禄猫忙路聡氓漏聝莽赂陆茅聧聸氓漏聜盲潞聴茅聴聛茅聺聸莽聣聠茅隆娄盲录麓忙聜隆氓篓聭忙掳拢氓聛聯茅陋聻氓聻聤職茅聳赂莽聡聛氓聛聟忙路聡氓漏聡忙聦聤忙戮鹿忙驴聢忙聡聯茅聢鹿氓聠麓鹿茅隆聰莽聭掳盲潞戮茅聳潞氓露聨氓聛聟氓庐?氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆茫聙聜茅莽聜聳潞氓聸楼氓聻鹿莽潞戮氓忙聼聼忙聺聢氓聣搂莽聲卤莽禄戮忙聧聬忙路聡氓漏聡氓娄露茅聧聸忙芦拢莽录聜盲陆潞氓搂路忙驴聻芒聜卢茅聳潞氓虏聙氓聳聳茅陋聻忙聢娄氓鹿聡茅聴聢忙露聶茅聴聜盲陆鹿忙聜露茅聧聰氓虏聞氓麓聬猫陇掳忙聨聭脦娄茅聳赂忙聸篓氓聳職茅聬陇芒聜聴聛氓聹聛聯茅聨卤陆脦聸茅聰聼莽聤鲁芒聮聭莽录聛氓卢聧聨忙聞掳忙聼篓茅聫聡忙篓潞莽聛虏莽聙碌氓忙聮聜垄茅聯聫氓聹颅忙搂聡茅聴聜氓漏聨氓聛聞氓篓虏茂鹿聙脙潞茅聫聧氓聞聴忙驴聽茅聺聸茅聢?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    requested_public_datasets : list[str]
        茅聴??茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聟忙職娄茅聳禄茫聢垄莽禄聬忙楼聟忙聼聼茅聳鹿聙氓聤聮忙驴聻氓路潞聹莽禄聽茂陆聣忙聼篓茅聫聝氓聸篓茅聳鲁脩聦氓聣聶氓茂驴聽氓录禄茅聤聤芒聲聛氓聧鲁茅聳禄忙聞楼麓茅聧聥忙聮鹿茫聛聬茅聨潞忙聢聽茂录聼茅聴聛莽聭掳莽聯篓莽禄露莽聳聛聬茅聨潞忙聢聽茅聴聛莽莽鹿聴茅聧聥忙聞虏聶茅聧楼芒聲聝莽拢陆氓篓麓茂陆聟忙聬芦猫垄篓茅聴聛莽篓驴氓忙驴聻氓漏聡莽聧?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓聙驴茅聳驴忙聸聻氓聙鹿茅聧聨忙聢聽忙聠赂茅聬聴氓聽聺莽卢聮莽录聛芒聜戮露氓卤赂氓聛聯茅聧聫氓盲潞戮茅聴聛茅聙聻氓卤陆氓垄麓茅聳赂芒聰戮氓聛聯氓娄聻茫聜聠氓赂聤氓篓麓氓聸篓莽芦聼氓搂聴芒聜卢茅聫聦茂陆聣茅聬聲盲陆鹿莽聳聛茅聴聛氓聯聞莽聼聣莽禄禄忙驴聠莽聲路茅聧芦忙聸聻莽聟聸氓篓麓茂陆聡茅隆楼氓潞篓忙垄禄忙碌拢猫搂聞氓聛聽茅聳赂氓漏聝莽聣聲莽聟陇茅聳禄忙聴聜氓聨搂茅聮聝莽聠潞芒聜卢氓鹿驴氓聞卤茅聬聴忙禄聝茅聤聤芒聲聜芒聮聭茅聳赂忙露聵茂鹿楼莽聛聫莽聰赂忙聼聸茅聬聵忙聼楼氓聫聞忙陇陇茫聢隆氓虏赂茅聫聛忙聞戮聞忙驴聽氓聥颅忙職聳禄莽聠赂莽虏聧茅聫聦茫聞漏忙聜戮茅聬聭猫聤楼氓聛聜茅聨录氓颅聵脡聭茅聫聜芒聜卢茅聴聜盲陆赂氓拢聤茅聧聥氓聭虏聶茅聨麓莽聜碌氓篓麓莽聟聨莽聯篓茅聢路忙聨聴忙聼聸莽聛聻忙聧聬氓拢聵茅聳鲁脩聦氓聣職茅聨庐氓聜職莽聲路茅聨掳莽聼聶茅聬聽氓聸漏氓聛聯莽禄聥忙聢拢莽聛聝茅聴聛茅聙聻氓卤陆氓垄虏茅聳禄氓驴聯茅聢鹿忙聢聻氓鹿聤茅聳赂氓漏聝忙麓聼茅陋聻氓漏聝忙聝聰茅聤聤茫聝楼莽聙聴聛莽篓驴忙聹卢莽禄聥氓驴聰莽娄聧氓氓聽垄氓聫聯茅聳赂茫聝拢氓聤聧莽聛聫莽聣聢茫聛聞茅聤聣脩聝盲禄卤茅聳潞氓卤戮莽篓聯茅聢搂莽露聠茅聧聥忙聮禄忙聡聯茅聧搂忙莽聟聸茅聬聫莽聜碌氓聛聯莽禄聙氓聠麓聮莽录聛忙莽庐職莽录聛莽聤鲁莽聸炉氓炉聮庐忙聜聶茅聬垄茅聺聸氓鹿聤茅聴?
    skip_public_benchmarks : bool
        茅聴?茅聴聜氓聜職氓聙赂茅聧聤莽聝陆忙聡聴茅聧聸聵芒聜陆鹿莽鹿聺茅聧楼脩聟茅聬聨氓聟聣氓聛聯茅聳赂忙聸篓氓聣聸茅聧聵莽聰赂忙垄潞茅聭陆茫聜聟忙聻聭忙驴聽茫聢隆茂鹿垄茅隆垄忙聴聢茅聰聰莽聲聦氓聨聯茅聬聼忙禄聞莽虏聯氓庐聲忙禄聝芒聳聨猫鹿聡茫聝篓芒聜聺聴氓录露茅聧芦忙露聵氓茅聳赂忙卢聫莽鹿聬茅隆陋氓聠聜卢茂陆聨盲潞聺茅聨鹿茂陆聟芒聵聣茅聨戮氓麓聡脨搂茅聳潞氓卤赂芒聜卢忙聞聛氓聬聥茅聨录莽虏鹿芒聜麓聡茅聴?莽录?benchmark茅聴?
    reuse_search_cache : bool
        茅聴?茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙碌拢茅聺漏猫聙驴茅聴聛莽搂麓莽禄聙氓陇聦芒聜卢氓鹿驴氓聞卤氓篓虏莽聤禄氓麓聭茅聨戮猫炉虏氓聻聜忙陇陇忙聞露莽碌驴茅聧聭茂驴聽忙聼拢忙聬麓茫聢聽氓職聳赂忙聮露氓聳聳氓炉芒聵聣茅聤聫莽聤聳茫聞漏忙聼篓氓漏碌氓聴聵氓聶鹿忙陇陇茅聢鹿忙聢娄忙聜聶茅聭录茅聴聛忙聬聻氓聤聫氓娄聥炉茂陆聻猫鹿聡忙聮聛忙麓陋茅聧聸莽聰禄茅聴聜盲陆赂忙聬隆娄忙聣聜卢莽卢聸氓聸搂莽拢鹿忙陇陇忙聽篓莽虏聨莽潞戮猫聴聣茫聙聮氓忙聬聬茅聳潞氓卢聸炉氓搂聤猫聶鹿忙聲忙聝搂氓聼聦茅聴聛盲录聶莽碌驴茅聧聫忙漏聙莽聧聨潞忙聡鲁忙聴聢氓聼聙茅隆聮盲陆赂氓聤聰茅聴聛猫聧陇氓聳聬莽禄虏聙忙聧垄茅聨潞忙聮露莽聫露茅聳潞氓聸漏氓聳聯茅隆漏莽聝陆忙聜聴茅聰聺氓潞隆盲潞聻茅聳赂忙篓录忙聲禄氓搂聤猫聶鹿忙聥聽茅聢楼氓麓聡芒聜卢氓漏職忙聼聸氓漏聤氓聠篓忙職聼莽录聛芒聜驴聽氓聸篓芒聮聮氓篓麓茂陆聟忙聠隆莽聮聬茂驴聽忙聼隆茅聧聹盲陆赂盲潞聺莽录聛莽禄聵莽聸炉茅聧漏芒聜卢忙陇陇忙聨聭氓芦聬聯?
    allow_incomplete_paper_run : bool
        茅聴?茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙碌拢茅聺漏猫聙驴茅聫聣氓聽職忙聶聺茅聬聹氓忙聼隆茅聧聬茫聞楼芒聜卢莽聰赂氓麓聭茅聨戮猫聢碌氓聳聠茅聳赂忙聸篓氓聣聸茅隆娄茫聞楼茅聤聢氓聠篓氓娄录忙驴隆莽鹿聢茅聬聸?氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄茅聳禄忙聞聬芦茅聧篓氓聜聻忙聼聸茅茂陆聠莽莽录聛莽聤卤氓聧鲁茅聫聦忙露聵芒聰聭茅聧聲氓搂垄茅聴聛忙麓禄氓聨陆茅聨鹿氓聸录禄茅聬聰猫聶鹿莽拢录茅聳碌氓驴聲氓搂碌茅聬聫氓露聣忙垄潞莽禄聥忙卢戮氓聞聦茅聳潞氓聭庐莽虏聯忙驴隆莽聰赂氓麓聼茅隆聳忙掳卤茅聴聛莽聭掳莽聯篓莽禄禄氓露聠莽忙驴聜氓聭聤芒聮聭茅聳赂忙驴聠氓茅聧聨氓潞篓忙聼陇茅聧聬猫聦聼拢茅聳鹿氓聛聯氓娄聻茫聜聟莽鹿聬茅聬聴氓聴聶莽虏禄氓搂聵猫戮篓氓聳聬茅聼芦氓聸篓忙麓聳茅聧篓忙驴聠芒聰聭茅聬聵氓庐聽氓茅聧聻忙垄潞茅聧聬氓聫聣忙聝聢茅聳赂氓漏聜忙聠隆莽禄聜忙聨聴氓聼隆忙碌拢氓聫聣忙聳聙茅聴聛莽禄聵氓聤聳氓忙卢聯忙聜聵茅聰聰氓盲潞聹忙陇陇忙聮露氓聛聼忙戮搂茂赂陆忙聟?
    requested_device : str
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓聙驴茅聳驴忙聸聴氓聛聯忙陇陇氓聲麓氓庐聲氓陆聮茅聧聸氓搂潞茅聴聜盲陆潞茅聧聲氓聻芦莽聲露莽录聜盲陆虏氓聛聯氓漏垄猫路潞氓陇聡茅聧聥忙聮禄莽聧庐茅聧篓氓氓娄聻茫聜聫氓炉聦茅聧篓猫路潞忙碌录忙聺聭脦篓茅聳驴忙聳驴氓垄陆茅隆聰忙聸垄忙垄潞茅聬聼茅聜娄氓職聼茅聳赂氓氓聳聳茅聤聠氓聜聻氓录禄茅聤聣茂赂戮氓聨卤茅聴聛莽禄聵忙茅聫聙茅聛聧氓鹿聠茅聧聝娄氓漏垄猫聢碌氓聤聳茅聬聯茫聝娄莽鹿聺茅聴聢忙露聶茅聳赂茂陆聠忙陇陆茅聨庐氓陆聮氓鹿聡茅聤聤茫聝楼芒聮聣莽录聜盲陆潞莽虏炉莽禄禄氓聽楼氓麓潞茅聢搂茫聛聞茅聨潞忙聢聺莽录聛莽聥聶氓聸篓莽聟聶茅聳禄忙聞忙驴隆猫聜漏莽麓聯氓氓聮聝猫潞卢莽聙碌氓聛聤茅陋聻氓聸卢氓录露茅聧搂忙楼聙忙聲鹿茅聴聜盲陆潞莽虏炉氓搂聤猫莽娄聧忙碌聽氓卢聞碌忙陇陇忙聨聠氓聛聯茅聳驴忙聸篓盲潞聹茅隆驴聸忙聜聹氓娄炉猫炉虏忙驴聻氓炉赂氓聨搂茅聬隆茫聞漏茅聫聧芒聮聭莽录聛氓氓掳聟茅聫聙猫聧陇莽麓聮莽聮聡猫聦卢氓垄娄茅聳禄忙露卤氓聶拢氓庐聲氓聧聻芒聵聣氓娄炉莽垄卤忙搂掳忙驴隆莽聜陋氓聙聳氓娄芦盲戮聙氓麓聭茅聨掳氓聟?氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聧聸氓漏聝氓聫聫氓篓麓氓聟录忙陆聶茅聧篓氓聜聻忙聼拢茅聨戮氓麓聡氓虏赂茅聳潞氓莽聣聴莽禄禄忙露垄氓鹿聥茅聬聬芒聲聜茂录聢氓漏碌莽聜虏忙篓聤忙碌聹忙禄聢茂陆聻茅聧聸氓颅聵莽聟聫茅聳赂莽鹿聝茅聧拢盲禄聥忙聼?
    requested_neural_search_profile : str
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓聙驴茅聳驴忙聸聴氓聛聯忙陇陇氓聲麓氓庐聲氓陆聮茅聧聸氓搂潞茅聴聜盲陆潞茅聧聲氓聻芦莽聲露莽录聜盲陆虏氓聛聯氓漏垄猫路潞氓陇聡茅聧聥忙聮禄莽聧庐茅聧篓氓氓娄聻茫聜聫氓炉聦茅聧篓猫路潞忙碌录忙聺聭脦篓茅聳驴忙聳驴氓垄陆茅隆聰忙聸垄忙垄潞茅聬聼茅聜娄氓職聼茅聳赂氓氓聳聳茅聤聠氓聜聻氓录禄茅聤聣茂赂戮氓聨卤茅聴聛莽禄聵忙茅聫聙茅聛聧氓鹿聠茅聧聝娄氓漏垄猫聢碌氓聤聳茅聬聯茫聝娄莽鹿聺茅聴聢忙露聶茅聳赂茂陆聠忙陇陆茅聨庐氓陆聮氓鹿聡茅聤聤茫聝楼芒聮聣莽录聜盲陆潞莽虏炉莽禄禄氓聽楼氓麓潞茅聢搂茫聛聞茅聨潞忙聢聺莽录聛莽聤碌莽虏炉茅聤聡氓聥炉氓录聧聦忙露聶莽聫陋茅聴聛氓聭聤脙录莽禄卢聫氓篓虏忙驴聻忙聢拢盲潞卢茅聧聰忙聬聛忙垄?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳鹿茅聴麓莽莽潞戮莽聭掳莽聙拢忙聧拢茅聳禄忙茅聹聣茅聳驴忙驴聠忙麓陇茅聧聰氓芒聜忙麓聵茅聬聯茂驴聽氓录禄茅聬聰氓聭聤莽露聡茅聳赂忙聮聴氓聭庐忙庐赂莽录聜氓聜聶猫聝露忙驴聰赂忙聜搂氓氓聽聺莽聞碌忙陇陇忙聨聭氓聳職氓忙聞鹿聸茅聧聶氓氓垄聬聯茫聜聣忙聼篓茅聫聡莽聜虏芒聜陆聮氓聼聸茅聨麓茂赂陆芒聮聮茅聳赂氓聳聯茅聤聠忙聨聴氓录露茅聢潞氓聜聸茅聳潞氓虏聙氓聳聳氓娄芦氓聠篓芒聵聣氓娄禄聵莽聵拢氓漏碌莽聜虏莽聯篓莽禄聮露莽虏禄茅聫聧猫聯聼忙驴聻忙聢聻莽聼庐氓篓聦忙聼聸茅聨戮忙陇禄氓聛聯氓漏碌忙麓娄莽禄禄忙驴聠芒聳聯茅聧篓盲禄漏氓娄聻茫聜聠忙麓娄茅聧聶氓聠麓潞茅聢搂茫聛聞茅聨潞氓聴聴盲潞戮莽录聜盲陆潞氓搂麓莽禄聣莽聭掳莽聲路茅聫聡茫聢隆莽聞录莽聙鹿茫聝楼忙聡聬茅聫聦氓聽楼忙垄潞莽录聛忙漏聠莽聞戮忙陇陇忙聸聠氓聭颅?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    list[str]
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓麓茂陆聣忙陆禄茅聭掳氓聺聴忙垄潞茅芒聜禄聣莽聰赂忙聜隆茅聫聡茫聢隆氓聙碌茅聳驴忙驴聠茅忙碌聹忙驴聢芒聜忙麓聳茅聬颅氓聜聻氓录禄茅聢楼氓麓聡忙職聠茅聬聨茂陆聨氓聣聸茅聬娄氓聽楼忙聜聴莽聭聶氓聥茅聳赂忙氓聸搂莽聟聵茅聳鹿猫戮戮莽庐聭莽禄聽忙露聶茫聛聞茅聨戮忙聞聛忙篓聢氓篓麓忙禄聟氓聛聼氓娄虏忙聞掳氓鹿聵莽聙聸忙篓潞茅聴聛忙聝搂莽鹿聮茅聨鲁忙聮露莽虏颅茅聰聼莽聤鲁氓聙碌茅聳赂氓驴聯忙碌聹氓聣搂莽麓聯忙碌拢氓聣虏氓聞聯忙驴聻氓陇聥忙麓聼氓炉聤陆忙聲聝茅聧聦忙露職氓聫聠氓漏碌莽聤禄忙陆搂氓娄芦忙卢聯氓麓碌茅聧楼芒聰戮氓聛聯莽聭聶氓聥拢赂茅聳赂忙聳驴莽聼聜茅聧漏茫聢陇氓鹿聵氓篓虏莽禄聵忙聶聴聛茅聙聻氓卤陆氓垄陇陇莽鹿聵茅聨庐忙禄聝脦聼莽聮聡氓路陆炉茅聴聜盲陆潞忙聡聯茅聧垄忙聼楼忙聡聼茅隆聳莽聜虏氓聜篓茅聳鲁茫聝娄氓聣職氓篓?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜麓莽聣聽茅隆垄忙聸垄忙聲聝茅聢搂氓茅聧娄莽拢录茅聬聨茂陆聨氓聛聯莽禄卤芒聲聜莽鹿聸氓庐赂莽掳禄茅聳禄忙聮麓莽聸聫聦忙露垄氓鹿聡茅聤聤茂赂陆莽聫聰茅聳鹿氓聟录忙陆聶茅聰聲氓鹿聝氓娄陇氓聭炉氓碌聛茅聳赂氓聳聳忙驴庐氓潞隆芒聰聭茅聬聬茅氓職聥莽录聛莽聤鲁忙聦赂茅聬拢莽聝陆氓鹿聥氓漏碌忙聞禄潞茅聴聛茅聙聻氓卤陆氓垄麓忙莽聜虏氓聲芦茅隆禄聢氓聼聙茅隆聮氓聥芦莽庐聳忙驴聻氓聴聵氓聻露氓庐聲氓忙聼拢茅聢潞氓聜職莽卢職茅聳鹿莽聭掳忙麓猫炉虏芒聵聣氓篓聠忙聢聽茅聴聛氓聣虏莽聟聢茅聧聽忙聨聴氓麓聭茅聨戮忙聳楼忙聰陆茅聬聨茂陆聨盲潞拢茅聨陆忙聸聻芒聰聭茅聬聬忙聺聭莽聛娄茅聳驴忙聸聴忙聶聸茅隆颅氓聸聼隆茅聧聸氓聨陆茅聴聤忙麓娄氓庐聙忙聬聛氓麓芦茅聯聫氓聹颅莽拢录莽录聜盲陆鹿氓篓虏氓炉赂芒聜卢猫搂聞忙麓聳茅聤聢忙聬聛氓鹿聝茅聤聫芒聵聠忙搂鹿茅聨录氓颅聵莽鹿聞莽禄聙氓陇聬忙垄潞莽聮聡忙聫聮忙陇陇忙聴聙莽聣聽氓庐聲盲录麓氓录陆茅隆聯莽聠路盲潞驴聽茅聺聸氓聙禄茅隆颅氓聽楼氓录赂忙碌拢猫隆隆氓聛聯茅陋聻氓聻聤職忙陇陇茂赂禄氓聻鲁莽禄庐氓聽垄芒聜卢茂陆聨氓聳聴茅聬聯莽聠赂莽鹿聸茅聨麓莽聜碌氓垄陋茅聳赂忙聴聯茅聶聞茅聨卤茫聞娄莽鹿聺茅聧聬芒聰驴盲禄垄氓漏碌芒聵聠氓聛聞茅聧聼氓聼楼忙戮露氓篓戮忙聻聨忙驴隆忙陇驴氓娄芦脩聡忙垄禄茅聧聦忙卢聮氓聟聦忙陇陇莽聣聯茅聫聛氓聠庐氓娄路茅聢潞盲陆潞莽聲潞茅聴聛莽忙聹茫聞漏氓录鲁茅聰聲氓聜聸莽聟聫茅聼芦氓聸搂茅聢搂忙麓聹莽禄庐氓漏職忙聲聬忙戮露氓氓聨陆氓漏碌芒聵聠氓聻掳茅聨录氓录鲁忙聺聢茫聞拢盲潞聹茅聨录忙路聻禄氓篓虏氓颅聵忙聟篓忙驴聽氓聜陇莽聟录莽聙鹿忙聸聼氓赂聮猫聻聳茅聳鲁脩聦氓聣職莽禄聜氓露聠茫聜拢茅聧聬芒聰戮忙聥潞茅聴聛茅聺漏氓炉聦茅聧聵氓楼赂茫聙聜忙驴聜氓聭聤莽聟聲茅聳禄忙篓潞氓聲驴氓篓麓忙聟篓忙驴聽氓聭聳鹿茅拢聨氓篓聙氓庐聲茂陆聠氓戮聞莽聙拢忙拢路茅聴聜氓聜聶猫聝露茅隆垄氓漏聜芒聳聧茅聫聡茫聢聽莽庐虏茅聳赂茫聝聭氓卤陆忙庐篓茅聴聜氓聜職莽聣聤忙赂職忙氓麓碌茅聫聡茫聢隆莽聟聶茅聳鹿忙戮聵猫垄職茅聴聛氓聯聞忙聡聫茅聬聯茂驴聽氓录禄氓篓聭茫聢聽莽聞潞茅聳赂氓驴聲氓搂路茅聧聵脩聡茂陆聻氓漏碌氓卢聶聺茅聳赂氓卤戮莽篓聢茅聨路?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜麓莽聣聽茅隆垄忙聸垄忙聲聝茅聢搂忙聜驴茅隆聲芒聜卢茅聫聦莽聠录氓麓聹忙碌拢莽聝聵忙聠聵茅聴聛猫陆掳莽陇聛茅聰聲氓录禄茅聬聰氓聟录莽聞陆茅聳驴忙聸聴氓聛聯茅聳潞氓卢聳聬茅聤聡氓聥炉茅聰聺氓聴聶莽录聜盲陆潞莽虏禄禄氓聽聺脦聸茅聧聬茫聞垄忙茅聳赂忙氓聛聸忙戮鹿氓聽楼忙垄禄?茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅茅聰聲芒聜卢莽潞戮氓陆聮忙聼拢茅聨麓茂陆聟脨聯茅聳赂茫聜聟氓茅聫聦忙露垄茅聰聺氓聴聶茅聴聛莽禄聵氓赂隆氓脩聡氓聼聻茅聨麓茂鹿聙莽聟隆茅聳赂芒聞聝茫聛聻莽禄篓氓忙聼拢忙聬麓茫聢垄忙驴聽茫聢隆茂鹿垄茅聨鹿聝忙聜聹茅聭潞茫聝娄忙芦聹忙驴聽茫聢拢忙鲁聸茅聰聲茫聜聟莽聲碌茅聧隆忙露聶芒聮聭莽录聜盲陆鹿脙聬聫茫聝娄盲驴聤茅隆聬茂陆聨忙庐聰茅聳鲁莽禄聵忙聧聡忙驴隆忙聬聛氓聼隆茅聧聦忙楼聛氓?benchmark 茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼忙聲聥莽聭聹氓露聣忙鹿聛茅聴聛莽禄聵氓聻录氓娄芦氓聥虏禄茅聬聵莽禄聵莽聟聶茅聳鹿猫搂聞氓聤娄茅聧陇忙卢聯忙聜聴氓忙搂潞茅聳鲁脩聦氓聣職茅忙聽麓掳氓娄陇盲陆潞莽庐戮氓漏碌莽聤虏忙麓陇茅聫聥盲戮聙忙聼聼茅聳颅茂鹿聙氓鹿聳莽录聛氓聣聛氓聛陇茅聨忙聲聬忙聬麓芒聞聝莽聛聧茅聴?忙驴聽莽聰碌氓搂路茅聫聛氓聭聤忙聟篓茅聨戮氓聙露忙驴庐忙篓驴忙鲁聸茅聧職氓陆聮忙聼聧猫茅聧聯莽聠路茅聯聫氓聹颅莽聼聶忙驴聻氓聴聵氓聞鲁茅聧聯盲录麓忙垄潞茅聨赂氓潞垄莽漏驴莽录聛莽聤鲁氓聻碌氓庐聯?
        氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄茅聳禄忙聞聬芦茅聧篓氓聜聻忙聼聸茅茂陆聠莽莽录聛莽聤卤氓聧鲁茅聫聦忙露聵芒聰聭茅聧聲氓搂垄茅聴聛忙麓禄氓聨陆茅聨鹿氓聸录禄茅聰聺氓茅聳碌氓驴聤忙聶聺茅聧聰莽篓驴芒聰聭茅聢陆氓聴聴盲潞聺茅聳驴忙掳露忙忙戮露忙聞颅莽聟楼忙驴聽茅聺聸忙忙戮搂忙聢娄忙聼聧猫茅聧聫忙漏聙氓麓聭茅聨戮莽禄聵芒聮聭?fresh search/training茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮莽聲聦忙聠赂氓庐楼氓聽垄盲潞卤茅聴聜盲陆赂忙聬芦茅聧聼忙聜搂茅聧隆忙露聵氓聻聜氓篓拢氓聸漏忙職麓r profile茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮莽聲聦忙聠赂氓庐楼氓聽垄盲潞卤茅聴聜盲陆赂忙聬芦茅聧聤氓陆聮忙聲?氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聧聸氓漏聝氓聫聫氓篓麓氓聟录忙陆聶茅聧篓氓聜聻忙聼拢茅聨戮氓麓聡氓虏赂茅聳潞氓莽聣聴莽禄禄忙露垄氓鹿聥茅聬聬芒聲聜茂录聢氓漏碌莽聜虏忙篓聤忙碌聹忙驴聥氓录虏氓漏碌氓卢聟聲茅聬聫莽聜碌猫聶聨茅聧陇茫聜聣忙聼聲忙戮露氓聴聵忙芦聠茅聳禄忙聮麓忙麓聼茅聨漏聙忙聜聶茅聨禄忙聨聠忙聦聝茅聴聛忙聞聛氓聻卤氓篓虏忙聽聼聻茅聨麓茂赂禄忙聲聧忙驴庐忙篓录氓聨搂氓篓聢氓聽聲忙垄潞茅聳芦莽聜虏猫聥炉忙戮搂莽麓聮茅聬聵猫聦卢脨搂莽聙鹿忙聸聼忙麓聼氓庐聲茂驴聽忙聜聶氓庐楼氓聴聶芒聵聣茅聯禄忙聽聼聸氓篓聭氓聧聻忙聻聸茅聳鲁脩聦氓聣聺茅聧漏忙聽篓莽漏卤忙驴聽氓聸搂脦搂茅聳赂氓卤戮莽聼戮鹿忙聸聻芒聰聭茅聬聵氓聻芦忙職漏茅聳赂氓卢聛陇氓庐聲氓陆聮氓麓录茅聫聡莽聠路盲禄颅氓娄聻茫聜聠莽聣聹茅聧聥莽聜潞氓麓聬?茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聟忙職娄茅聳赂氓聛聟茅聧聶氓驴聸忙聼聼茅聨聺驴脨聯茅聳赂茫聜聟氓芦掳茅聫聦忙露垄茅聰聺氓聴聶莽录聛莽聜聞鲁氓篓录茂驴聽氓录禄茅聫聡茫聢聽茅聠聡忙驴聽氓聻芦氓聤聳莽卢聭茅聴聜盲陆赂忙聠隆茅聧篓猫搂聞莽鹿聸茅聢搂忙聼隆莽聛聻氓聣搂忙麓麓忙莽聜虏忙麓颅茅隆垄忙漏聙氓娄赂猫茅隆垄氓聭麓忙垄?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳禄忙聢拢氓搂陇茅聧陇氓聥炉忙聼陇莽禄聥猫路篓盲禄聸茅聳赂忙卢聫莽鹿聵茅聫聦茂陆聣氓茅聢漏氓聠篓盲禄漏茅聴聛?忙驴聽莽聰碌氓搂路茅聫聛氓聭聤忙聟篓茅聨戮氓聙露忙驴庐忙篓驴忙鲁聸茅聧職氓陆聮忙聼聧猫茅聧聯莽聠路茅聯聫氓聹颅莽聼聶茅聳赂忙聽陇芒聜卢氓聟聟莽麓拢茅聴聜盲陆潞莽虏聬聴忙聸垄氓娄露莽禄聥猫路篓忙職娄忙驴聻氓路潞聹茅聬?
    """
    violations: list[str] = []
    required_public_datasets = list(_SUPPORTED_PUBLIC_DATASETS)
    if not allow_incomplete_paper_run and skip_public_benchmarks:
        violations.append(
            'full paper run requires the complete public benchmark surface and cannot skip public benchmarks'
        )
    if not allow_incomplete_paper_run and set(requested_public_datasets) != set(required_public_datasets):
        violations.append(
            'full paper run requires all supported public datasets: '
            f'required={required_public_datasets}, requested={list(requested_public_datasets)}'
        )
    if str(requested_device).strip().lower() not in ('cuda', 'auto'):
        violations.append("full paper run requires requested device 'cuda' or 'auto' for fresh neural training")
    if not allow_incomplete_paper_run and reuse_search_cache:
        violations.append('full paper run requires fresh search/training and cannot reuse search cache')
    if str(requested_neural_search_profile).strip().lower() != "paper":
        violations.append("full paper run requires neural search profile 'paper'")
    if any(value is not None for value in (lstm_epochs, liquid_epochs, batch_size, eval_batch_size)):
        violations.append(
            "full paper run requires the default paper-grade training budget from the model YAML and cannot use CLI budget overrides"
        )
    return violations


def _write_blocked_paper_run_report(output_root: Path, payload: dict[str, Any]) -> int:
    """茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聫茅聢搂莽聞娄莽禄禄忙驴聥氓麓聮氓娄陇盲陆潞盲潞聶忙驴聽莽聰碌氓聛聻茅聧篓氓麓鹿氓篓虏莽聳戮?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳鹿茅聴麓莽莽潞戮氓陆聮忙聼聼茅聴聜氓炉赂茅聳禄茫聜聨氓聶拢茅聫聦莽聠路芒聳聯茅聧篓莽聛聢茅聴聛忙聬聻氓聤聧莽禄禄氓聥炉氓聼聙茅隆聮盲戮聡氓聲聫茅聳赂莽聤鲁莽聣聽茅聢楼茂鹿聙莽聲隆茅聨碌忙庐戮茅聴聛氓聣虏氓聛聟氓篓虏忙聽芦氓拢聲莽聮潞茫聞垄氓聛聯茅陋聻氓聻庐氓聤職忙陇陇茂赂驴氓聧鲁茅聨?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜麓莽聣聽茅隆垄忙聸垄忙聲聝茅聢搂氓茅聧娄莽拢录茅聬聨茂陆聨氓聛聯莽禄卤芒聲聜莽鹿聸氓庐赂莽掳禄茅聳禄忙聮麓莽聸聫聦忙露垄氓鹿聡茅聤聤茂赂陆莽聫聰茅聳鹿氓聟录忙陆聶茅聰聲氓鹿聝氓娄陇氓聭炉氓碌聛茅聳赂氓聳聳忙驴庐氓潞隆芒聰聭茅聬聬茅氓職聥莽录聛莽聤鲁忙聦赂茅聬拢莽聝陆氓鹿聥氓漏碌忙聞禄潞茅聴聛茅聙聻氓卤陆氓垄麓忙莽聜虏氓聲芦茅隆禄聢氓聼聙茅隆聮氓聥芦莽庐聳忙驴聻氓聴聵忙聦禄茅聧陇氓莽鹿聸茅聧芦莽聠路忙陇陇茫聝楼氓莽禄聵忙聥聟茅聧聸氓漏聝氓聞聯茅聴聛莽篓驴茂鹿楼茅聧篓氓聻庐莽聲路茅聧芦忙聸聻脦漏莽聭聹氓露聣莽聧庐茅聧聝娄芒聮聮氓篓聯忙掳拢氓聛聯忙驴聻盲陆鲁氓聸篓忙聶卢茅聼聸篓莽篓聭莽禄聙氓聠聼聸茅聨掳氓聣聛莽聲卤茅聳禄氓驴聰莽聟隆茅聫聦茂录聞盲禄娄茅聧聯脩聞氓聢聸茅聴聛猫炉隆氓聳聯茅聧篓猫聴聣莽聲路茅聤聤茂赂戮莽炉聙茅聳赂氓卤戮莽虏拢莽禄卤茂鹿聙芒聰聭忙聨鲁茅聧聤忙忙聟篓茅聬聭猫聴聣莽拢禄忙驴聻忙聢聶氓聼聞茅聫聛氓聥聭赂茅聺聸氓莽聙鹿忙聮虏莽路職茅聫聦忙露垄氓娄路茅隆聰莽聟聨芒聮聮茅聴聛猫陆掳莽氓篓虏氓录禄茅聰聺氓忙拢聵茅聳赂氓聶聫聠茫聢陇忙聼拢茅聤聫芒聲聛莽聛隆莽录聛氓卢聦戮氓娄虏忙聞掳氓鹿聵莽聮聡猫聦录聹氓漏碌茂鹿漏茅聧聫忙漏聙氓麓聭茅聨戮忙聬搂鹿茅聨录猫戮戮莽碌驴茅聬聮氓聟录莽鹿聸茅聨戮忙聺聭莽聞鹿茅聳赂氓卢聧聡茅聫聦莽聠录茅聤聤茫聝楼莽录聛芒聶聙氓聛聯茅聴聜盲陆赂忙聠隆茅聧聰氓驴聸氓录虏氓篓聭忙漏聠氓職聲茅聳赂脩聞茅聢路忙聢娄忙垄禄茅聧聠潞茫聙聮氓脩聞忙麓聵茅聤聡氓聥聲聜莽聮聡猫聦卢盲禄漏氓茫聜聠忙聼篓莽禄禄忙聞掳氓聼聻茅聨麓茂鹿聙氓鹿垄茅聼芦氓聸篓忙垄鹿茅聫聣忙聬聛忙垄禄忙碌拢猫聶鹿氓赂聸忙陇陇莽聣聫忙碌聹莽芒聳聨茅聨麓莽聤卤莽卢聣茅聬聨猫忙麓聳氓篓虏茫聞搂氓忙驴聜氓聭聤莽聟聝莽聭聹忙禄聢氓麓聹氓搂聵氓職聴茅聳赂忙聸篓氓聨赂茅聧聥忙聮禄忙聲聬忙聬麓芒聞聝莽聛聧茅聴聛莽篓驴茂鹿娄茅聧聫忙漏聙氓录禄茅聢漏氓聠篓莽路聻茅聬聨茂陆聨盲潞聻忙碌聽氓颅聵莽鹿聺氓篓聢氓聻庐忙聻聸茅聳鹿猫聤楼莽虏聨氓娄虏忙聞掳氓鹿聵莽聙聸忙篓潞茅聴聛忙聝搂莽鹿聮茅聨鲁忙聮露莽虏颅茅聰聼莽聤鲁芒聮聭茅聳鹿猫聜漏氓聛聸忙驴隆氓楼赂忙聼拢猫鹿聡忙聴聜莽庐聻茅聳鹿莽庐聳茅聨禄聢忙聦聤忙戮露忙聞聲鹿茅聴聜盲陆赂忙聬篓虏茫聞漏氓麓掳忙聬麓茫聞楼莽聞碌忙陇陇忙聨聭氓聙赂茅聧聵忙聮聼隆氓庐聙氓卢搂庐忙楼聽莽聜虏茅聨篓潞脙潞莽聮聛忙聤陆氓聛聴氓漏碌茂鹿陇茅隆颅氓鲁掳莽聲路茅聨潞忙聢颅忙陆漏忙陇陇忙聢拢忙忙碌聹氓聣搂莽麓聮莽聙拢氓職娄忙驴聻忙聢聻茅聫聝氓聽聺氓路聺忙陇陇忙聴聢忙拢驴茅聧聭猫聴聣芒聰聭茅聬聵莽禄聵忙露搂茅聳赂氓漏聜茅聢楼茂鹿聜氓麓录茅聤聣茂鹿聙芒聜聺聴忙聼聸茅隆颅忙聢聺盲潞聳氓篓麓忙禄聞莽虏聯茅聫聦莽聠赂忙陆聫茅聧聯脩聟脡聭氓茫聜聝氓聞卤茅隆娄忙鹿聛氓漏碌莽聤虏茂鹿陇茅聬聴氓驴聸忙聜戮茅聬聭忙篓禄氓聫聫氓漏碌莽聤虏氓聲炉茅聤聡忙驴聢芒聜忙麓聫茅聧聰氓露聡忙聧垄氓漏碌氓莽拢录忙驴聽氓篓戮氓聛聯莽录聛猫戮篓脡聭莽禄禄忙驴聥忙聜陆茅聴聢忙露聴莽虏露茅聴聛氓陆聮忙聰卤茅聧篓氓聹茅聬聬氓莽聭鹿茅聳鲁脩聦氓聣聼茅聬聸氓麓聵茅氓聵聣茫聞漏忙聼聲忙戮露忙篓潞忙聻陇茅聳赂忙卢聫忙拢聺氓忙麓麓聹茅聨禄忙聨聧莽掳录茅聴聛?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    output_root : Path
        茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聧茅聳禄忙聞陆聶忙碌聹猫戮篓莽鹿聺茅聬垄茅聺聸脨垄忙驴聻氓卤聜氓聙聴莽禄庐莽禄聵氓莽聙聸忙氓聨碌茅聴聜氓聜職氓聙赂茅脣聡茅聰聲氓聜聸莽聟聲忙驴庐忙篓禄氓聣職氓篓录忙聞掳忙聼聲茅聧楼脩聝忙楼聽莽聜麓茂鹿垄氓庐聲茂驴聽忙聜聶茅聧聫氓聭聤氓聛聟茅聴聜氓聜聶莽茅聨虏茂驴聽氓麓聺猫鹿聡忙露聵莽拢禄茅聳鹿氓聣搂莽虏聢路忙聨聭茫聙聮氓茂陆聟盲潞戮氓漏碌莽聜露氓拢聢氓庐聲莽聰赂氓聼聙茅隆聮氓聥卢忙露搂茅聳禄氓聙赂茅聬拢莽聝陆氓茅聤聢氓聴聴盲禄录茅聬聨氓聟聣氓聛聯茅聳鲁脩聦氓聣聼茅聨麓聡莽麓聮氓娄炉氓聹庐職茅聴聛茅聺聸莽聣聠茅聨鲁氓潞篓氓录鲁茅聬聬莽聝聵莽聟聲茅聳禄忙篓潞氓聲驴忙驴露聣忙聼隆氓庐聙氓卢聢掳茅聯禄忙聽颅忙聼聧猫茅聧聯莽聠录氓鹿聝猫莽禄禄忙驴聥氓麓聼茅茅聴聜盲陆赂莽露聤氓娄芦氓聺聴忙聜聻茅聰聲氓聜聸氓聛?
    payload : dict[str, Any]
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳鹿茅聴麓莽莽潞戮氓陆聮忙聼聼茅聴聜氓炉赂茅聳禄茫聜聨氓聶拢茅聫聦莽聠路芒聳聯茅聧篓莽聛聢茅聴聛忙聬聻氓聤聧莽禄禄氓聥炉氓聼聙茅隆聮盲戮聡氓聲聫茅聳赂莽聤鲁莽聣聽茅聢楼茂鹿聙莽聲隆茅聨碌忙庐戮茅聴聛氓聭聤忙聦路茅聭鲁猫聢碌莽篓聣氓氓聴聬盲潞聹茅聳潞氓聸篓忙碌聹茅鹿聝忙垄潞茅聧聺氓聥鹿掳忙驴聻氓聫聣脙陋茅隆聲茫聝漏氓麓聼茅忙掳露氓聰露氓漏碌莽聤禄忙陆搂氓娄芦忙楼聟茂陆聻氓搂聵猫聢碌芒聮聮氓篓麓茂陆聟忙聠隆茅聨虏忙聬聛忙聼聸茅聬聵猫聶麓聦莽聙鹿忙聸聼氓聻驴忙驴隆猫聢碌氓戮聞忙驴聽氓聭庐忙聠掳忙驴聽莽聰碌氓聛聻茅聧篓氓麓鹿茅聧娄盲赂聧忙驴聻忙聢聶氓聻潞茅聬聯?茅聴?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    int
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿氓氓卢茅隆聬盲禄聥忙聼篓茅聬聰氓聯聞脨垄莽禄戮忙聝搂茅聫聦氓露聢茅聳赂忙聮露氓聳聳氓炉芒聵聣氓茂陆聟盲潞戮茅聳驴忙驴聠茅陋赂忙碌聹忙驴聢芒聜卢猫搂聞忙麓聳茅聬聜聻氓录禄茅聰聸氓聥虏陋茅聳赂忙聮聴氓聥聻卤忙驴隆茫聜聡氓搂路茅聧聥忙露垄忙戮露忙聞聻聛茅聳鹿莽聣聢忙戮聵茅陋聻茫聢隆莽鹿聸茅聨麓莽聜碌忙聡聟茅聳赂?``2``茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓聙驴茅聳驴忙聴聢忙聲庐茅聧聥忙聮露莽虏篓聭忙篓禄氓聶陆茅聳禄莽聭漏茅聫聦莽聠赂忙陆聫忙楼拢氓聠漏茅聴聛莽篓驴氓茅隆篓氓聭麓氓娄麓茅聨潞忙聢陆漏茅聳驴忙驴聠忙聡聧忙戮鹿忙聸聻芒聰聭茅聬聵忙忙職聳赂忙聮麓莽鹿聢茅陋聻忙聞掳氓麓聵茅聭录忙庐戮氓漏碌茫聛聸茅聧聥忙聮禄忙聼聼茅隆聳忙掳卤莽聞赂莽聙鹿忙聸聻氓麓聢陆氓陇聤芒聳聨茅聨麓忙驴聠莽庐潞茅聴聜氓聜聶猫聝露茅聨鲁忙聮禄茅隆聯茫聢聽莽拢禄茅聳碌氓聽聺忙莽禄禄氓聴聸忙聼聸茅茂陆聠莽茅聳潞氓聭庐莽虏聯茅聫聦忙露垄氓鹿聵氓娄陇芒聜卢茅聨路忙聢聽忙搂聬茅聫聦茫聝娄芒聮聭茅聬聽氓聸篓氓陆聝茅聧陇猫戮漏莽麓聮茅聼聼聶莽聙?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬氓聫聣莽聳聞氓漏碌脗掳氓聙搂莽禄虏盲禄聥氓麓炉茅隆聳莽聜虏莽拢禄茅聬聨茂陆聨氓聳聴茅聬聯忙聸垄忙聼聧茅聢潞盲陆赂忙職聼茅聳鲁猫聴聣莽禄卤忙聨聴忙聜漏氓庐聲氓聸搂芒聮聦茅聴聛氓聯聞莽聻录茅聧聽忙聽鹿聝氓篓聠忙聯聝忙聲聠茅聳赂氓卤戮莽虏聨氓庐聯盲陆鹿莽鹿聺茅聬垄茅聺聸盲禄娄茅聳赂氓聻露氓庐聲莽茫聛聬茅聨潞忙聮鲁芒聜卢氓聺聴忙聼聸茅隆聢聺盲潞聳氓篓麓忙禄聞莽虏聯茅聫聦莽聠录忙聜聹氓娄炉猫聶鹿盲禄麓氓娄聻茫聜聟莽鹿聠茅聫聟莽聤禄氓录禄茅聰聲氓聜職莽陇聝忙陇陇氓驴聯氓茅聬聫氓聽楼忙垄潞茅聧聺氓聥鹿掳忙驴聻氓聫聣脙陋茅隆聲茫聝漏氓麓聼茅莽聠赂氓漏碌忙陇麓莽芒聳聯茅聧聸氓漏聛莽鹿聠茅聳禄忙聞碌盲潞聹茅聢搂忙麓聹茅聨鹿茫聢陇氓麓卤氓篓聭忙氓聤娄氓茫聜聠氓赂聮茅聧聼忙聥聦芒聟碌茅聳禄莽聠赂莽虏聨忙戮搂忙忙聲娄陇氓聭聳戮茅聳潞氓卤禄氓聞虏茅聬聯忙聸聼莽鹿聸茅聨麓忙驴聠猫聢鹿茅聳禄氓驴楼莽露聛茅聫聦茂陆聣氓娄露茅聧聸盲录聝氓漏碌?
    """
    normalized_payload = dict(payload)
    normalized_payload["completed_stages"] = list(normalized_payload.get("completed_stages") or [])
    _write_final_report(output_root, normalized_payload)
    try:
        from liquidloc.common.io_utils import dumps_json_text
        print(dumps_json_text(normalized_payload))
    except Exception:
        
        pass
    return 2


def _write_failed_paper_run_report(output_root: Path, payload: dict[str, Any]) -> int:
    """Write the failed paper-run report and mirror it to stdout when possible."""
    _write_final_report(output_root, payload)
    try:
        from liquidloc.common.io_utils import dumps_json_text
        print(dumps_json_text(payload))
    except Exception:
        
        pass
    return 1


def _loads_json_from_stdout(stdout_text: str, *, source_name: str) -> Any:
    """Parse a structured JSON payload even when stdout contains progress logs."""
    from liquidloc.common.io_utils import loads_json_text

    text = str(stdout_text or "").strip()
    if not text:
        raise RuntimeError(f"{source_name} returned empty stdout")
    try:
        return loads_json_text(text)
    except json.JSONDecodeError:
        pass

    start_index = text.find("{")
    if start_index == -1:
        start_index = text.find("[")
    if start_index == -1:
        raise RuntimeError(f"{source_name} returned non-JSON stdout: {stdout_text}")

    opener = text[start_index]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for end_index in range(start_index, len(text)):
        char = text[end_index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == opener:
            depth += 1
            continue
        if char == closer:
            depth -= 1
            if depth == 0:
                candidate = text[start_index : end_index + 1]
                return loads_json_text(candidate)
    raise RuntimeError(f"{source_name} returned non-JSON stdout: {stdout_text}")


def _build_paper_run_runtime_context(
    *,
    seed_manifest_path: str,
    env_report_path: str | None,
    readiness_report_path: str | None,
    sim_materialization_report_path: str | None,
    paper_scope: str,
    requested_public_datasets: list[str],
    reuse_search_cache: bool,
    allow_incomplete_paper_run: bool,
    neural_search_profile: str,
    neural_search_surface: Mapping[str, Any],
    public_roots: Mapping[str, Any],
    resolved_training_budget: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the paper_run_report I/O block."""
    return {
        "seed_manifest": seed_manifest_path,
        "env_report": env_report_path,
        "readiness_report": readiness_report_path,
        "sim_materialization_report": sim_materialization_report_path,
        "paper_scope": str(paper_scope),
        "requested_public_datasets": list(requested_public_datasets),
        "reuse_search_cache": bool(reuse_search_cache),
        "allow_incomplete_paper_run": bool(allow_incomplete_paper_run),
        "neural_search_profile": str(neural_search_profile),
        "neural_search_surface": copy.deepcopy(dict(neural_search_surface)),
        "public_roots": copy.deepcopy(dict(public_roots)),
        "resolved_training_budget": dict(resolved_training_budget),
    }


_ACTIVE_PAPER_SCOPE = "sim_e9_only"
_SIM_E9_ONLY_CORE_EXPERIMENT_CONFIGS = ("e9_dual_degradation.yaml",)
_SIM_E9_ONLY_METHODS = ("ekf", "robust_ekf", "fgo", "lstm_ekf", "liquid_ekf")


def _resolve_paper_scope(paper_scope: str | None = None) -> str:
    """Resolve and validate the requested paper scope."""
    resolved_scope = str(paper_scope or "full").strip().lower()
    if resolved_scope not in {"full", "sim_e9_only"}:
        raise ValueError(f"unsupported paper_scope: {paper_scope}")
    return resolved_scope


def _set_active_paper_scope(paper_scope: str | None = None) -> str:
    """Persist the active paper scope for helper functions used by the run."""
    global _ACTIVE_PAPER_SCOPE
    _ACTIVE_PAPER_SCOPE = _resolve_paper_scope(paper_scope)
    return _ACTIVE_PAPER_SCOPE


def _core_experiment_config_names_for_scope(paper_scope: str | None = None) -> tuple[str, ...]:
    """Return the core experiment surface for the active paper scope."""
    resolved_scope = _resolve_paper_scope(paper_scope or _ACTIVE_PAPER_SCOPE)
    if resolved_scope == "sim_e9_only":
        return _SIM_E9_ONLY_CORE_EXPERIMENT_CONFIGS
    return _CORE_EXPERIMENT_CONFIGS


def _build_paper_run_contract_block_reason(contract_violations: Sequence[str]) -> dict[str, str]:
    """Build the blocked/failure reason block for report contracts."""
    violations = [str(item).strip() for item in contract_violations if str(item).strip()]
    if not violations:
        return {
            "reason": "paper_run_contract_violations",
            "message": "paper-grade full run prerequisites are not satisfied",
        }
    return {
        "reason": "paper_run_contract_violations",
        "message": "paper-grade full run prerequisites are not satisfied; see contract_violations for exact blockers",
    }


def _resolve_output_root(raw_output_root: str | None) -> Path:
    """茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聫茅聢搂莽聞娄莽禄禄忙驴聥氓麓聮氓娄陇盲陆潞盲潞聶忙驴聽莽聰碌氓聛聻茅聧篓氓麓鹿氓篓虏莽聳戮忙驴聽茅聺聸茅聬聯氓驴聯芒聰聭茅聬聬忙聢聺氓聲芦茅隆聫忙聮庐忙聼聸莽聙拢茫聞楼氓娄录忙陇陇氓聲麓忙驴隆氓聽聲氓麓卤氓娄赂驴忙聲驴茅聴聜盲陆鹿忙聜露茅聧聰氓卤陆茂鹿垄忙聺聢芒聲聛氓聻聺茅聳赂氓聞卤莽禄聙氓卢聛聞茅聫聧茫聞搂茫聙聮茅聳赂忙卢聫氓氓娄聭麓茅聤聫芒聙鲁氓漏碌脟聹茅聧聰氓潞篓氓麓職茅聨潞忙楼聛忙聶虏茅聳赂忙露聵氓聙鹿氓娄芦氓聠篓莽聲路氓搂聴芒聜卢茅聧漏芒聞聝忙聥聟茅聧娄氓聛聯忙楼拢氓聠漏芒聮聮氓篓麓茂陆聢忙芦拢莽聰炉忙露聶茫聛聞茅聤聤茫聝娄氓職聡茅聳鹿氓聥炉聙茅聳赂忙聸篓氓聨戮茅聬聮氓楼赂忙垄潞茅聳聜虏猫聥戮搂忙聮聼聼茅隆聰莽颅聥忙庐聹忙驴隆氓聲聢陆氓陇聤脦聼茅聳驴氓卢聴聜盲陆陆莽聙聸芒聲聟茅聫聧莽陋聴茅聳潞氓露聨氓聛聠忙露聯氓露聣忙聼聲忙戮露忙露聹莽陇聥氓篓麓忙禄聞莽虏聯茅聫聦茂驴聽氓麓聵茅聰聺氓聭茅聴聜盲陆潞莽聙聸芒聲聜莽鹿聸忙驴聽氓聸搂莽庐聳茅聳鲁茫聝娄氓聣職氓芒聲聜茫聛聞茅聨戮猫路潞脨楼忙陇陇忙聴聙忙麓聼氓忙麓聜路茅聳鹿聙忙庐露茅聴聛莽篓驴茅聧聥莽聠赂莽茅聨录盲陆赂脦漏茅聳鲁氓聯聞氓聙禄茅聧聵茅聛聧忙聼拢忙聬麓莽搂碌茅聳赂氓忙聡聨茅聬拢氓鲁掳莽聲聺茅聧聲氓聨碌氓娄聻茫聜聠忙垄禄茅聧聵脩聝脦拢忙驴聽忙掳颅莽聟聝茅聬聽氓聸篓莽聟隆茅隆聨忙忙聼聧莽聮聡忙聼楼氓聫聞忙楼聽莽聜虏茅聨聲聟氓鹿聡茅莽聜碌盲录聦氓漏碌茂鹿篓氓篓聟茂陆聣氓鹿聫茅聬聵莽禄聵氓職聭忙陇陇忙聨聠氓聛聯茅聳赂茫聜聨氓聣聸莽禄卤忙聮麓莽卢聼茅聧楼脩聞莽聛聧茅聴聛猫聧陇氓聲驴茅聫聟莽聤禄氓鹿聝忙碌录忙聬聲聤茅聴聛忙氓聻掳忙碌聹茅鹿聝忙聼篓氓漏碌氓聴聸茅聳潞氓莽篓聯莽禄卤忙聨聴忙聜漏氓庐聲氓聸拢氓搂聡茅聴聛茅聺聸忙麓陇莽聙職忙聺驴茅隆聲芒聜卢茅聧漏芒聜卢忙陇陇忙聨聭氓芦潞戮氓聺聴忙聼聸茅聨掳莽職潞茅聳潞氓聴颅茅聶聞茅聤聡氓聥庐聸茅聨戮猫路潞茅聴聛莽禄聵忙聦戮茅聧聽忙聽聧庐茅聫聧茫聞娄莽露聡茅聳赂忙聮聴氓聬路忙聢娄忙垄潞茅聧聺氓聥卢茅聳潞氓聭庐莽虏聯忙驴隆莽聰赂氓麓聼茅隆聳忙掳录忙陆聤茅聴聛莽禄聵莽聭楼茅聨鲁忙聮禄氓麓楼茅茂赂陆芒聮聭茅聬聽氓聸拢芒聜麓聡盲禄漏茅聴聛忙麓禄氓聨录茅聧聤氓聺聴忙聜掳茅芒聜庐聲氓聧聻茅聧聫莽垄卤芒聜卢氓聸垄禄忙碌拢猫氓聛聽茅聳赂氓漏聝莽聣聽茅聤聠氓聠漏莽鹿聺茅聧楼脩聠忙聭聲氓漏碌莽聜麓氓聻炉茅聧篓氓陆聮氓麓隆茅聯聫莽鹿聠茅聳碌氓聽聺氓茅聧聰氓露聠莽鹿聺茅聤聫芒聙鲁忙碌庐忙驴?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    raw_output_root : str | None
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓聙驴茅聳驴忙聸聴氓聛聯忙陇陇氓聲麓氓庐聲氓陆聮茅聧聸氓搂潞茅聴聜盲陆潞茅聧聲氓聻芦莽聲露莽录聜盲陆虏氓聛聯氓漏垄猫路潞氓陇聡茅聧聥忙聮禄莽聧庐茅聧篓氓氓娄聻茫聜聫氓炉聦茅聧篓猫路潞忙碌录忙聺聭脦篓茅聳驴忙聳驴氓垄陆茅隆聰忙聸垄忙垄潞茅聬聼茅聜娄氓職聼茅聳赂氓氓聳聳茅陋聻氓漏聡氓麓聵茅隆聰猫搂拢忙聥禄忙驴聻猫戮戮莽露聙氓篓聟茂陆聡忙聲篓聭茫聢隆氓芦聟茅聫聣氓聽職忙聶聹茅聧聽忙聞掳忙聼隆忙戮露氓漏聞盲禄垄茅聳潞氓聴聵莽聭娄茅聤聡氓聥炉氓录麓茅氓聴聶脨聹氓娄陇莽聤碌氓聛聸茅聳掳茂陆聣氓鹿聤茅聬聬氓聯聞脦篓莽聭聹氓露聣莽聧庐氓娄陇氓聭庐芒聮聮氓篓聯忙掳拢氓聛聯忙驴聻盲陆鲁氓聸拢芒聜卢茂鹿聜氓麓录茅聤聣茫聝搂茂录聼茅聳潞氓陇聤莽聣聝茅聫聟忙驴聢氓拢聲茅聮聵氓陇聢芒聜卢忙聨聭茅聰聺氓聭忙驴隆莽聜陋氓聙聭茅聧聤忙聸垄氓麓聬茅聧篓忙職娄茅聳驴忙驴聠莽禄聙氓聠聼聲忙驴聻脩聝氓茅隆聰忙聸聼莽鹿聺茅聬垄茅聺聸盲禄聹忙驴隆莽聭漏茅陋聻忙聞颅莽鹿聺氓莽聲戮茅聴聛莽禄聵氓聤聴茅聧聨茫聜聣氓麓聭茅聢漏氓聠茅聳赂芒聞聝莽禄聽莽聣聢茫聛聞猫鹿聡茂陆聟盲潞掳茅聳潞氓虏聥莽碌陆茅聢陆氓陇聬氓录陆猫陇聧忙驴庐茂鹿聜忙垄潞莽聮聡芒聙鲁忙聻聸茅聳赂茫聢隆莽聟隆茅聢楼茂鹿聜氓娄赂茅聢潞盲陆赂莽陋聸氓茫聜聠忙聦聠茅聧聲忙忙驴聽莽聰碌氓搂路茅聫聛忙聬聛氓麓聭氓篓聭忙漏聠氓職聲茅聳赂忙麓聵茅聧聥氓卢聼拢氓娄氓職聶莽录聛莽聤鲁莽聣聲猫聻聳茅聳驴忙聴聜莽聯聰茅聧聨茫聝娄盲驴聤茅聨禄忙聨聭芒聮聰莽录聛猫戮篓忙聦禄莽禄聴莽聠录氓麓露猫陇聨茅聬聫氓露聣忙垄潞茅聧聺氓聥卢氓聶潞莽录聛忙聧聡氓氓漏職氓娄露忙戮露氓氓聞茫聜聠氓聤聭茅聧聨茅聛聧忙聜隆氓芒聜氓漏職氓麓聮氓搂聵氓聛聯茅聬聭猫聤楼氓聙驴茅聳驴忙聴聢忙聲庐茅聧聥忙聮露莽虏篓聭忙篓禄氓聶陆茅聳禄莽聭漏茅聫聦莽聠赂忙陆聫忙楼拢氓聠漏茅聴聛莽篓驴茅聫聟莽聤禄氓录禄茅聰聺氓氓聛聭茅聳鲁脩聦氓聣聶茅隆卢氓漏聞忙忙驴隆氓庐聰`` 氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄茅聳禄忙篓驴莽禄聽氓聻聛聞猫鹿聡忙聮麓莽录聛芒聜卢忙驴聽氓聸篓莽聟聝莽聭聹忙禄聢氓麓聹氓搂聵猫戮篓氓麓虏忙驴聻忙聢聻莽聭楼莽禄露莽聜潞忙聜聴茅聰聺氓潞隆盲潞聻忙陇陇茂赂驴氓聧鲁茅聨庐忙楼聙芒聳聯茅聧篓莽聫聴聛莽篓驴茅聰聲茫聝娄莽漏卤忙驴聽氓聸搂茅聠聡茅聳潞氓聸漏氓聛聸茅聭掳氓聺聴忙垄潞茅聧聬氓聫聣忙聝聢茅聳赂氓漏聡茅聧聮忙聨聴氓篓聭忙掳卤莽聵?茅聴?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓麓茂陆聢氓篓聟莽聤禄氓鹿聳氓篓拢氓莽禄戮忙聝搂茅聫聦忙聸垄氓麓录氓漏碌氓聸露忙莽录?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    Path
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜麓莽聣聽茅隆垄忙聸垄氓娄露茅聧楼芒聲聝氓聨聽茅聴聜盲陆赂忙聬芦茅隆娄盲录麓氓麓碌氓忙麓聧聲氓鹿聠戮莽聠聠茅聬聽猫聶鹿氓掳篓茅聼驴聤茅隆聬茫聢聽忙搂禄忙陇陇氓聲麓忙驴隆氓聽聲氓麓卤氓篓聠氓驴聤忙聥隆茅聴聜盲陆潞茅氓聴聶莽虏聶茅聨戮猫路潞氓聻聺茅聳赂莽鹿聞茅聫聠氓聴聸忙聼聸茅聫聡茫聞楼氓垄掳茅聳赂忙篓禄忙聜聜茅聨忙聠麓茅聧聲氓聞聨茅聴聛氓聯楼氓搂碌茅聬聴忙禄聢氓聼聙茅隆聮盲陆赂氓職聳赂忙聮露氓聳聳氓庐聲忙麓陋氓聼聙茅隆聮盲陆碌盲潞聹茅聳鹿莽聝聵氓聻碌茅聢搂莽露聤氓炉聤陆茅聧聲氓聨赂茅聬聨氓聟聣氓聛聯茅聬聨茂陆聨氓聣聸猫垄娄氓漏碌莽聤碌茅聧聯忙驴聥氓鹿聝茅聧聦忙露聴莽鹿聠茅聳禄忙聢拢忙氓聰聬聼忙禄聝莽鹿聮忙戮鹿忙聸聼莽赂陆茅聧聸氓篓戮忙聥潞茅聴聜盲戮聡氓聛聠茅聧聥忙露職芒聜卢忙掳录氓聨录猫垄聶茅聳赂忙聸篓氓聛聮氓篓聯氓忙聼聲茅聧芦忙驴聠忙職聶茅聳禄氓驴聤忙聶聽茅聫聦茫聞漏氓录麓茅聤聤茫聜聣氓聛聴氓娄陇莽聤碌氓聛聸氓娄忙聜戮氓漏碌氓卢忙陇陇忙聞漏氓聙聴茅聫聠盲录麓忙聼拢茅聬聰氓聯楼莽聼聥莽录聛忙聦赂茅聬拢莽聝陆忙氓篓麓猫聶芦忙拢聝氓庐聲茫聞漏茅聨炉脩聝莽庐聻茅聴聜氓聜聶莽茅聨录脙潞茅聰聲氓聜聵芒聜卢氓聠麓卤氓篓聭忙茅陋聡茅聬聼忙禄聟氓莽聰忙聜?

    茅聴聜氓聜職氓聙路氓篓麓氓聸茅聢搂莽篓聢茅聫聟莽聤禄茫聙聛氓篓聠忙聮鲁氓聠聙忙陇陇忙聮露氓聛聼茅聬聸茫聝漏忙垄潞茅聧聺氓聥颅芒聳聣茅聳赂忙篓驴氓聛聬忙戮鹿忙聸隆茫聛聬茅聨潞忙聮露氓聨碌茅聴聜氓聜職氓聙赂茅氓麓聺氓氓陇聥莽聟聲?
    ------
    ValueError
        茅聴聜氓聜職氓聙路氓篓麓氓聸搂莽聲路茅聬垄莽莽陋聴茅聳鹿茅聜娄氓聳聳茅聧篓忙驴聥氓鹿聳氓篓拢莽聛驴聻氓聭搂碌忙路聡氓漏聡氓娄露茅聧聸忙芦拢莽录聜盲陆潞氓搂碌氓漏垄忙漏聙芒聙聰茅聧聬氓聛聯茅聰聺氓潞聺莽掳禄忙陇陇忙聨聥氓聻潞茅聤聡氓聥聰聺氓聴聶茅聴聛氓聯聞忙聹卢茅聬漏茫聙聛茅聫聡茫聢隆忙聶虏茅聳赂芒聞聝莽聭娄虏莽麓聯茅聧聦忙卢聮茅聳赂氓卢聧聡茅聫聦忙露聵莽聲聺茅聢搂氓麓聭茅聬聬莽聝聵氓聛聜茅聼聸篓忙聬漏茅聬聰氓聴聸忙聜鹿茅聧楼茂陆聜氓聛聯氓庐聲氓聸娄莽虏麓茅聴聜盲陆潞茅隆聭氓聴聸氓鹿聭茅聧楼猫聯聼茅聳鲁芒聲聛氓聲炉忙驴莽麓聮莽聙拢氓聞楼氓篓麓莽聟聨氓聲麓氓猫聶鹿忙聥聽茅聢楼猫聶芦莽聛聧氓茫聜聝氓聤聦茅聨鲁忙漏聙氓麓潞茅聤聣茂鹿聙莽路聻氓漏碌莽聜碌氓聛聯茅聬聨猫芒聰聭茅聢陆氓聴聴莽聛楼莽聙鹿忙聸聽忙聲禄氓?氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄茅聳禄忙聞聬芦茅聧篓氓聜聻忙聼聸茅茂陆聠莽莽录聛莽聤鲁氓拢聠莽禄卤忙聨聰芒聜卢茂陆聨氓聛聮茅聧聨茫聝漏忙聼聸茅聤聢氓聴聵莽茅聳潞氓卤戮莽聸隆垄忙聸垄忙聲聬茅聧隆忙卢聵忙搂聴聜盲陆鹿忙聜露茅聧聰氓虏聞氓麓聬茅聧搂忙楼聙猫聯聼茅聳潞氓聸漏氓聳聯莽禄隆芒聜卢茅聴聤忙麓娄茅聨录芒聲聛脙潞茅隆聯茫聢隆芒聮聮茅聳赂氓卤戮莽聭搂茅隆娄茂鹿聜忙聼聼氓篓麓氓卢垄虏莽录聛忙氓聳聨猫聻聳茅聳赂忙露卤氓聨录茅聧聥氓露聠莽鹿聸莽聙碌莽脨垄忙陇陇忙聢聺忙聡聬忙戮鹿忙聸垄氓麓赂氓娄陇盲陆潞氓聨碌茅聴聛猫炉隆氓聻聨茅聧聬莽聜卢忙庐潞莽录聜盲陆潞氓聛聯氓庐聲忙漏聙氓麓聭茅聨掳忙聺驴莽聞碌忙陇陇忙聨聠氓聛聯茅聳赂忙篓禄莽虏聯氓庐聲忙聢娄氓鹿聵莽录聛忙篓录氓聨卤茅聴聛莽氓聼聡茅聧聰氓露聟芒聳聧茅聧隆忙卢聭莽拢录莽录聜盲陆鹿氓篓虏氓炉赂芒聜卢猫搂聞忙麓聫茅聧聰忙聢聺莽聲路茅聨炉茅聜娄忙搂录忙驴聽茫聢拢忙鲁聸莽聙職盲录麓氓录禄氓篓聭忙卢聬氓聞聼茅聳驴忙露聵氓芦聟猫茅聫聦茂录聞盲禄娄茅聧聯脩聞氓聢聸茅聬聨忙庐驴氓聳聴茅聨赂猫聶芦盲驴聤茅聨录盲陆赂脦篓茅聳驴忙聳驴氓陆聻忙戮鹿忙聸垄忙垄禄茅聧聦忙卢聮氓聟聦莽录聛氓聻露氓炉氓娄赂茅聢潞盲陆潞莽聳聡茅聴聛氓聹颅氓垄聴氓忙漏聙忙聜聻茅聧篓盲潞聹茅聳鹿氓聯聞忙忙碌聹茅鹿聝忙垄潞莽聮聡猫聦茅聳录氓陆聮莽庐聳茅聢楼茂鹿聜氓麓鹿茅隆聰茫聝搂氓聳聯忙碌聹忙聬聛氓录陆猫陇聦忙戮鹿忙聸聻芒聰聭茅隆聰莽颅聥莽聞戮忙驴聻氓陇聥莽篓聯忙戮鹿忙聸聼莽赂陆茅聧聸氓漏聞氓聨陆茅聴聛氓陆聮氓聛聤茅聧篓盲录麓忙聝聝茅聬聭忙篓赂盲潞聹茅聳潞氓聜聸氓茅聧拢氓陆聮忙聼聧莽聭聶氓聥聛麓茅聳鹿莽聭漏氓炉聽聲氓鹿聥茅聬聵猫戮篓茅聴聜氓聜聶莽聞娄莽聙碌脩聟氓聳聤氓庐楼氓莽庐聻茅聳碌氓篓戮忙聶聹茅聫聟忙聼楼忙聼聸氓芒聜聬聮茂鹿聜氓聼垄茅聧聝娄芒聮?
    """
    if raw_output_root is None:
        return (ROOT / "outputs" / "paper_run").resolve()
    if not raw_output_root.strip():
        raise ValueError("--output-root must be a non-empty path")
    output_root = Path(raw_output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    return output_root.resolve()


def _resolve_optional_path(raw_value: str | None, *, flag_name: str) -> str | None:
    """茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聫茅聢搂莽聞娄莽禄禄忙驴聥氓麓聮氓娄陇盲陆潞盲潞聶忙驴聽莽聰碌氓聛聻茅聧篓氓麓鹿氓篓虏莽聳戮忙驴聽茅聺聸茅聬聯氓驴聸忙聼聸茅茂陆聡莽庐聵氓漏垄脩聞氓聻碌茅聹聣忙驴聽氓漏聜氓聲掳莽禄聣忙麓陋忙聼隆氓庐聙氓卢拢驴氓篓麓莽聥聟氓娄聡茅聨路莽聤禄氓鹿聯茅聨潞忙聢颅忙陆楼茅聴聜氓聜聶猫聝露莽禄聲庐茅聬聰莽垄卤忙聲聥莽聭聹忙聴聢氓聻戮茅聰聰莽聜卢氓麓聣茅聳碌氓漏聫莽庐聭莽潞戮忙垄潞莽录聛忙篓潞莽聛娄茅聮聝猫戮漏芒聜卢猫搂聞忙麓聵忙驴聻氓漏聜茅聯聫氓聹颅莽拢录忙驴隆氓麓碌茅聧聶氓聴聸忙垄潞莽潞聭庐莽聫聳赂忙聴聙莽聭漏氓庐聲忙麓聼聙茅隆聮盲陆碌盲潞聹茅聳鹿氓聯聞忙忙碌聹茅鹿聝忙垄潞莽聮聡猫聦茅聳录氓陆聮莽庐聳茅聢楼茂鹿聜氓麓鹿茅隆聰茫聝搂氓聳聯忙碌聹忙聬聛氓录陆猫陇聦忙戮鹿忙聸聻芒聰聭茅隆聰忙聳陇猫聙驴忙陇陇茫聜聟氓聙驴氓氓麓卤氓篓聭忙卢聭氓聨卤茅聴聛氓聯聞忙麓垄茅聧聰氓卤戮莽氓漏碌氓莽聟聼茅聨录忙聬聳聳忙戮搂氓颅聵忙聟篓忙驴聽氓聭聳鹿氓聫聣忙聦鲁氓庐聲莽聠录忙拢聝氓篓聭忙卢聬莽聫卤茅聴聜氓聜聶莽陇聛茅聨陆忙禄聞忙聟篓茅聴聢忙露職忙職聠莽录聛氓卢鲁聲茅聫聠茂赂陆莽鹿聸氓莽掳录茅聳赂氓卢聤聳莽禄禄忙露垄氓录露茅聨麓氓聣卤氓茫聜聨氓聛聞莽禄聣莽聭掳氓篓虏氓聛隆茅聳潞氓忙鲁聲莽禄聥氓驴聰莽鹿聺茅聤聫茂陆聨忙陆聬忙驴聻猫聦聳聨茅隆?茅聴聜氓聜職氓聙赂茅聧聤莽聝陆忙聜聻茅聰聲氓聜職莽庐聳茅聳赂忙麓聳莽禄聙氓陇聦芒聜卢氓鹿驴氓聞卤茅聨路氓茅聢潞氓聭庐莽聟聫氓漏碌莽聜聭氓聠篓忙驴聻猫聜漏氓聳聳氓忙麓麓路茅隆聯莽聜虏氓娄潞茅聴聛忙聬聻氓聤聨茅聧聽忙聽搂?茅聴?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓麓茂陆聢氓篓聟莽聤禄氓鹿聳氓篓拢氓莽禄戮忙聝搂茅聫聦茂陆聣氓录聧聦忙掳禄聵茅聴聛莽禄聵氓赂聻茅聫聛氓聯楼忙茅聨潞忙聳潞氓聛聯茅聰聺氓潞隆盲潞聹忙驴聻忙聬颅氓聶拢茅聫聦?氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄茅聳禄忙聞聬芦茅聧篓氓聜聻忙聼聸茅茂陆聠莽莽录聛莽聤鲁氓陆聝茅聤聠忙聨聭茅聰聺氓聭忙驴隆莽聜陋氓聙聭茅聧篓忙麓聡聺忙楼拢氓聠拢芒聜鹿聜氓娄赂茅聢潞盲陆赂莽陋聸氓娄聻茫聜聠忙麓聳茅聨鲁氓驴聲莽麓聻氓娄陇盲陆潞莽炉聙茅聳禄茫聢陇忙陆搂茅聢禄氓聽聼莽鹿聸忙碌拢氓聠虏忙麓娄茅聧聥氓卢聟芦茅聧楼茫聞搂芒聤聲茅聳赂氓潞垄茅聨卤茫聞楼脦聼茅聧篓氓聨聴聛莽禄聵忙聦禄氓篓虏忙忙聧垄氓漏碌氓卢驴聻忙聢聻莽聭炉氓娄聜虏茅聤聢氓聭炉莽庐聻茅聳赂氓漏聝莽鹿聢氓炉芒聵聣氓茂陆聟盲潞戮茅聳驴忙驴聠茅陋赂忙碌聹忙驴聢芒聜卢猫搂聞忙麓聳茅聬聜聻氓录禄茅聬聰莽垄卤忙聲聤茅聳禄莽聠赂忙戮聵茅聢路氓陇聬忙垄潞莽聮聡芒聙鲁莽聛隆忙驴隆氓聲麓氓炉戮露氓漏聞氓茫聜聟莽鹿聬茅聨鲁氓潞篓忙聜隆茅聫聡莽聜漏忙聰陆茅聳禄忙篓潞莽聛職茅聫聠芒聲聟忙聼聸忙聬麓茫聢聽莽聛漏莽聙碌忙聺驴茂鹿聨赂氓漏聝莽芦聼氓漏碌氓聸篓茫聛聺茅聳潞盲陆赂氓聙驴茅聨庐忙卢聯茅聢搂忙聝聰忙驴聽氓聸篓芒聮聭茅聳赂忙露聵茂鹿陇忙驴庐茂鹿聙脙鹿氓漏聤氓聥颅氓垄碌莽聙鹿忙聸聼氓聻驴茅陋聻忙篓录忙聥聽茅聨禄忙聨聰芒聜聤聢氓聴聴氓搂搂莽录聛氓聫聣忙陇陆茅陋聻氓驴聸忙聡隆茅聤聢氓聸漏茂录聺忙驴聻忙聮麓猫聣戮氓篓虏茫聜聣氓录赂茅聬聰猫隆隆氓聛聯茅聧聫氓盲潞戮茅聴聛氓陆聮忙拢驴茅聭鲁猫路潞莽聲碌忙碌拢忙聬聻芒聜卢忙聨聭茅聰聺氓聭忙驴隆茫聜聡氓搂路茅聧聥莽聜碌忙聲庐茅聢楼氓麓聡忙職娄茅聳赂忙楼聝氓聞聬氓篓聯氓忙聼聼茅氓聴聴忙聹卢茅隆?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    raw_value : str | None
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓聙驴茅聳驴忙聸聴氓聛聯忙陇陇氓聲麓氓庐聲氓陆聮茅聧聸氓搂潞茅聴聜盲陆潞茅聧聲氓聻芦莽聲露莽录聜盲陆虏氓聛聯氓漏垄猫路潞氓陇聡茅聧聥忙聮禄莽聧庐茅聧篓氓氓娄聻茫聜聫氓炉聦茅聧篓猫路潞忙碌录忙聺聭脦篓茅聳驴忙聳驴氓垄陆茅隆聰忙聸垄忙垄潞茅聬聼茅聜娄氓職聼茅聳赂氓氓聳聳茅陋聻氓漏聡氓麓聵茅隆聰猫搂拢忙聥禄忙驴聻猫戮戮莽露聙氓篓聟茂陆聡忙聲篓聭茫聢隆氓芦聟茅聫聣氓聽職忙聶聹茅聧聽忙聞掳忙聼隆忙戮露氓漏聞盲禄垄茅聳潞氓聴聵莽聭娄茅聤聡氓聥炉氓录麓茅氓聴聶脨聹氓娄陇莽聤碌氓聛聸茅聳掳茂陆聣氓鹿聤茅聬聬氓聯聞脦篓莽聭聹氓露聣莽聧庐氓娄陇氓聭庐芒聮聮氓篓聯忙掳拢氓聛聯忙驴聻盲陆鲁氓聸拢芒聜卢茂鹿聜氓麓录茅聤聣茫聝搂茂录聼茅聴聜盲戮聡氓聣聶莽禄聣莽聰赂氓麓聭茅聢漏氓聠炉聙氓漏碌莽聤虏氓聙鹿茅聧拢猫搂聞茫聛聞茅聰聺氓聭芦氓聛垄茅聳潞氓虏聥莽露聛茅陋聻忙卢聯氓麓聼茅莽聤碌忙茅聳禄氓潞垄茅聧聽忙聽颅芒聙娄氓庐聞忙聣職娄茅聳碌氓篓戮忙聶漏茅聫聛氓漏聡忙聼陇氓篓麓氓忙芦聡茅聳禄忙露卤氓聶拢氓搂聤猫聶鹿忙露碌茅聧聸忙拢聢茅聴聛猫搂聞忙陇驴忙碌聹莽聜虏忙驴聽氓聸搂茅聰陇忙驴隆茫聜聜氓聛聯茅聳录莽莽露聤茅聫聦莽聠潞莽麓聮茅聤聫莽聤鲁莽聛聧茅聴聛莽篓驴茂鹿娄茅聫聛氓氓录禄氓篓聭芒聭漏芒聜卢茂鹿聜氓鹿聥氓漏碌氓聭炉氓聡鹿莽录聜氓聜聶忙碌聡茅聧漏忙聽颅忙聜隆茅聰聼莽聤虏莽庐聳忙驴隆茫聜聜氓聛聯忙陇陇莽鹿聢茅隆垄忙漏聙忙聜聶茅聬聴氓聽陆聼茅聴聜盲陆陆氓麓碌茅聧聽忙聞聢聮氓庐聙芒聜聨鹿茫聢陇氓鹿聮氓娄陇芒聜卢莽禄聙氓聴聸忙聼聼茅聬聭忙漏聠忙庐聰茅聳潞氓卢漏莽路職茅聫聦忙露聵莽聲聺茅聢搂氓麓聭茅聬聬氓聯聞氓聬鹿茅聳赂忙聸篓氓聣聸莽禄聽氓聣搂芒聜卢莽聭掳氓茅聧聽忙聽篓忙聶露莽聙聸忙篓禄氓聫?
    flag_name : str
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛氓露聣氓麓聼茅莽聤碌氓聨炉茅聴聜盲陆赂忙鹿卤茅聧聨茫聜聣茫聙聝茅聧聸忙聺聭氓聻聜茅聳赂氓卤戮莽篓聯莽禄隆芒聜聴聜氓聜職莽聣聤忙赂職忙氓麓聲猫鹿聡忙聸聽莽拢录茅聳禄忙卢聦氓聛聯茅聳赂茂陆聫莽露聛氓炉氓录麓茅聤聫莽聤禄氓聰露氓漏碌莽聤禄忙陆搂氓篓虏茫聜聣脣聡茅聢潞氓聭聮聭莽录聛氓茅聧聨忙聞掳忙聼篓茅聫聡忙篓潞莽聛虏忙楼聽莽聜虏氓聲隆芦忙禄聢氓聼聙茅隆聮氓聥芦莽庐聳忙驴聻氓聴聵忙聦赂莽禄聽莽聰赂忙聼聼茅聧聺氓聥聬聨氓聻聣聽氓搂聤莽禄聵忙聥聟莽聭聶氓聥禄漏茅聴聛氓聭聤忙聼篓茅聬颅氓聜聸盲驴聤茅聧聸聻莽庐鹿氓篓麓茂陆聡忙聡聯忙碌聹忙茅聤聢氓聴聴氓搂搂莽录聛莽聤鲁氓聻驴茅聨卢聬莽鹿聺茅聧聬茅聴聛氓聣虏氓聛聠茅聧聽忙聞掳脣聣茅聧芦茫聝娄莽聟聸莽聙鹿芒聜卢茅聢搂氓麓掳忙聬麓茂陆聟氓录虏茅聴聜盲陆潞莽虏炉氓搂聤氓氓麓垄猫氓炉聭聤忙聥聟莽聬聸芒聙鲁盲潞戮忙楼聽莽聜虏莽聛聺茅聧聰忙掳露忙聼聼茅聯聫氓莽聙碌氓篓聤氓庐聲氓聧聻芒聵聣氓篓聠忙聢聺氓赂聴茅聳禄莽聠赂莽虏聦莽禄聣莽聭掳茅聧聸氓漏聞莽聭鹿茅聳鲁脩聦氓聣聼氓炉氓录麓茅聤聫莽聤禄氓聰露氓漏碌莽聤禄忙陆搂氓篓麓氓聜職脦聸茅聤聢氓陇聥芒聮聮氓篓麓茂陆聡氓聞陇茅聧陇芒聜聴聛氓聭聤猫聣戮茅隆聭氓氓鹿聝忙楼拢氓聠戮忙聲聜茅聳赂莽鹿聞茅隆娄忙聜聴茅陋聻氓聻庐氓聤職忙陇陇茂赂禄氓聻露茅聨聦聨茫聜拢茅聧聬芒聲聝氓聨聯茅聴聛氓聭聤莽鹿聜莽聙職氓麓鲁氓篓虏莽聟聲?茅聴聜氓聜聶忙碌聡氓庐聲莽聰赂氓麓掳茅聨掳氓聻聺茅聳鹿莽聜卢莽聹鹿茅聢搂氓聙驴茅聫聛氓聠麓聮氓篓聭忙篓禄莽聲卤茅聴聜盲陆赂忙聠隆茅聨赂茅莽虏聢忙赂職芒聜庐聲莽聭掳芒聰聭莽聙拢氓聨卤氓娄聻茫聜聠氓聤聴忙驴庐忙聮麓聺氓搂聵猫戮漏莽拢录茅聳禄忙篓潞氓聲驴氓篓麓忙聼隆氓庐聙芒聜聧聽忙聽搂莽聲路茅隆聬茂鹿聙莽陇聥忙陇陇忙聨聭茅隆楼茂陆聟芒聰聭?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    str | None
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜麓莽聣聽茅隆垄忙聸垄氓娄露茅聧楼芒聲聝氓聨聽茅聴聜盲陆赂忙聬芦茅隆娄盲录麓氓麓碌氓忙麓聧聲氓鹿聠戮莽聠聠茅聬聽猫聶鹿氓掳篓茅聼驴聤茅隆聬茫聢聽忙搂禄忙陇陇氓聲麓忙驴隆氓聽聲氓麓卤氓篓聠氓驴聤忙聥隆茅聴聜盲陆潞茅氓聴聶莽虏聶茅聨戮猫路潞氓聻聺茅聳赂莽鹿聞茅聫聠氓聴聸忙聼聸茅聫聡茫聞楼氓垄掳茅聳赂忙篓禄忙聜聜茅聨忙聠麓茅聧聲氓聞聨茅聴聛氓聯楼氓搂碌茅聬聴忙禄聢氓聼聙茅隆聮盲陆赂氓職聳赂忙聮露氓聳聳氓庐聲忙麓陋氓聼聙茅隆聮盲陆碌盲潞聹茅聳鹿莽聝聵氓聻碌茅聢搂莽露聤氓炉聤陆茅聧聲氓聨赂茅聬聨氓聟聣氓聛聯茅聬聨茂陆聨氓聣聸猫垄娄氓漏碌莽聤碌茅聧聯忙驴聥氓鹿聝茅聧聦忙露聴莽鹿聠茅聳禄忙聢拢忙氓聰聬聼忙禄聝莽鹿聮忙戮鹿忙聸聼莽赂陆茅聧聸氓篓戮忙聥潞茅聴聜盲戮聡氓聛聠茅聧聥忙露職芒聜卢忙掳录氓聨录猫垄聶茅聳赂忙聸篓氓聛聮氓篓聯氓忙聼聲茅聧芦忙驴聠忙職聶茅聳禄氓驴聤忙聶聽茅聫聦茫聞漏氓录麓茅聤聤茫聜聣氓聛聴氓娄陇莽聤碌氓聛聸氓娄忙聜戮氓漏碌氓卢忙陇陇忙聞漏氓聙聴茅聫聠盲录麓忙聼拢茅聬聰氓聯楼莽聼聥莽录聛忙聦赂茅聬拢莽聝陆忙氓篓麓猫聶芦忙拢聝氓庐聲茫聞漏茅聨炉脩聝莽庐聻茅聴聜氓聜聶莽茅聨录脙潞茅聤聫茫聜聟莽茅聬聫莽聜虏氓垄陆茅隆漏忙聼楼忙聼拢茅聨掳氓聤聥茅聳禄忙聮麓莽聭漏茅聫聦茂陆聣忙聲聬茅聧聸忙聺驴茅聴聛氓聯聞氓搂聤猫聧陇忙搂聬茅聨潞忙聮麓莽露聡茅聳碌氓氓聞鲁茅聬聼茫聞漏忙聼搂莽录聛忙氓垄碌茅聳鹿茅聫聛忙聞颅忙聝聰茅聢楼忙聞聛忙陆戮莽录聜氓聜聶莽聞娄茅聧聰忙聽聙聝忙驴聽氓聸搂猫聯聼茅聳驴忙驴聠茅聧聥氓聥炉忙聼聸氓篓聭忙漏聢氓聤聼忙陇陇茂赂潞氓聞卤茅聢鹿忙聢娄氓聼楼茅聧隆忙陇戮莽掳录莽录聜盲陆陆茅聬聴氓聴聸茅聰聺氓陇聵忙聲聝茅聳碌氓驴聤忙聶聸茅聨庐忙聢娄忙垄潞茅聨录莽聜虏氓聤碌茅聴聜氓聥氓庐聲莽聠录氓麓聵茅猫搂拢忙聥潞茅聳禄莽聤虏忙麓聽茅聢搂莽拢聢搂茅聫聦忙露聵芒聵聣茅聧聴莽聜虏茅聴聛?``None``茅聴?

    茅聴聜氓聜職氓聙路氓篓麓氓聸茅聢搂莽篓聢茅聫聟莽聤禄茫聙聛氓篓聠忙聮鲁氓聠聙忙陇陇忙聮露氓聛聼茅聬聸茫聝漏忙垄潞茅聧聺氓聥颅芒聳聣茅聳赂忙篓驴氓聛聬忙戮鹿忙聸隆茫聛聬茅聨潞忙聮露氓聨碌茅聴聜氓聜職氓聙赂茅氓麓聺氓氓陇聥莽聟聲?
    ------
    ValueError
        茅聴聜氓聜職氓聙路氓篓麓氓聸搂莽聲路茅聬垄莽莽陋聴茅聳鹿茅聜娄氓聳聳茅聧篓忙驴聥氓鹿聳氓篓拢莽聛驴聻氓聭搂碌忙路聡氓漏聡氓娄露茅聧聸忙芦拢莽录聜盲陆潞氓搂碌氓漏垄忙漏聙芒聙聰茅聧聬氓聛聯茅聰聺氓潞聺莽掳禄忙陇陇忙聨聥氓聻潞茅聤聡氓聥聰聺氓聴聶茅聴聛氓聯聞忙聹卢茅聬漏茫聙聛茅聫聡茫聢隆忙聶虏茅聳赂芒聞聝莽聭娄虏莽麓聯茅聧聦忙卢聮茅聳赂氓卢聧聡茅聫聦忙露聵莽聲聺茅聢搂氓麓聭茅聬聬莽聝聵氓聛聜茅聼聸篓忙聬漏茅聬聰氓聴聸忙聜鹿茅聯聫莽聭掳氓聤聥忙陇陇茫聝搂氓聛聻莽禄禄忙露垄氓麓篓茅莽聡聛忙茅聴聛氓庐聽茅聧篓氓聺聴茫聙聛氓篓聠忙聮麓氓聛漏莽聙鹿芒聜卢茅聢搂忙聜隆茅聧聦忙禄聟莽拢陆氓篓麓茂陆聞莽虏聶茅聧聺忙麓陋忙聜陆茅忙露聵忙碌庐茅聳鹿氓聞鲁茅聢鹿忙聢聽芒聜陆聨盲潞聻茅隆娄茫聞漏忙垄潞氓庐聲氓聸篓氓職聙莽录聛茫聞漏盲潞露氓庐聲忙聢娄氓鹿聵莽聮聡猫聦卢氓娄聻茫聜聟莽鹿聬氓娄芦忙露垄忙聲聧氓漏聤氓聽聼芒聮聭茅聳录忙碌聥忙聳聛茅聨路茂驴聽忙聼聸茅聨戮氓炉赂忙聡聞莽录聛氓聜職莽搂鹿氓庐聲忙禄聠莽禄戮忙聝搂茫聜赂茅聫聦忙露職盲禄娄茅聧聯脩聙忙搂聢氓漏碌莽聜麓忙聝聦茅聧聽忙掳露氓聼聙茅隆聮氓聠戮忙陆聬忙驴聻忙聫聮莽鹿聵氓庐聲忙驴聥氓鹿聥茅聰聰猫隆隆氓聛聯忙碌拢氓聣虏忙陆篓茅聳鲁脩聦氓聣聼茅陋聻氓聠茅聬聵氓聻芦忙庐聺茅聴聛氓聯聞茅忙卢聫脦聸茅聨掳莽拢陆茅聳赂氓卤戮猫聣戮茅聢搂忙聜聜氓庐聲忙聞颅忙聬麓茂陆聡莽聞录忙驴聻忙聮麓氓聼聝茅聧聥忙聮禄忙聼聼茅忙掳鹿掳茅聳鹿氓鲁掳忙聡聬茅聧聳忙聽录麓茅聬聰猫隆隆氓聛聯氓篓虏芒聮聭茅聳鹿莽篓驴忙碌路莽禄聽忙聮禄忙聼聼茅聧聬氓路篓聙莽聙碌茅氓楼赂氓麓露茅聤聤茫聝娄莽聙戮茅聴聜盲陆赂忙聬芦茅聧聼忙聜搂茅聧隆忙卢聭盲赂聧忙驴庐忙篓驴氓聼聳茅聬聯忙聸垄忙聼聼忙碌录氓颅聵忙露搂茅聳潞氓莽篓聯莽禄卤忙聨聴氓聼聙茅隆聮氓聜聴氓聛聯茅聰聺氓潞隆盲潞聳氓篓麓忙禄聞莽虏聯茅聫聦茂驴聽氓麓聮氓搂聵氓聫聵猫聶聨茅聴聛忙聤陆忙聰卤氓娄聠庐氓录禄氓篓聭茫聢隆氓聙路茅聬聨忙露聶脙聧聻氓陇聬忙垄潞莽聮聡芒聙鲁忙聻聼忙陇陇茫聞楼茂鹿陇茅聬拢莽聳聛聬茅聨潞忙驴聡忙聶聺茅聴聛忙聦聨莽鹿聜氓篓虏茫聜聣氓麓碌茅聨潞忙楼聛芒聮聭茅聬聽氓聸篓氓陆聝茅茫聢垄芒聜卢氓聟聣氓聛聯莽录聛氓忙鲁聲茅聫聠茂鹿聜忙聼聼茅聬聭忙漏聠忙庐聲茅聳赂氓漏聞氓聟聵茅聫聦茫聜聡氓聳聤茅聧聴莽聜虏芒聮聮茅聴聛氓聯楼氓聛聸茅聬隆茫聞搂忙聧垄氓漏碌氓氓聛篓茅聴聜氓聽聼莽篓聬氓篓聦忙垄潞茅聨录莽聜虏氓娄录茅聳禄忙聽搂茫聜赂氓炉聜卢茂陆聨盲潞聳茅聫聙猫聤楼茫聛聞茅聫聡莽聜麓茅聳赂忙聨聰氓垄聴茅聤聡氓聥聲聜茅聬聬忙炉聢茅聬?
    """
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    if not value:
        raise ValueError(f"{flag_name} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return str(path.resolve())


def _resolve_dataset_raw_root(dataset_name: str, raw_root_override: str | None) -> Path:
    """茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聟忙職娄茅聳禄茫聢垄莽禄聬忙楼聟忙聼聼茅聨碌茅聳赂氓卢露莽鹿聺氓炉聽聲氓麓录氓莽聫聰莽录聜盲陆陆莽驴掳茅聧聤忙聸隆莽聛聝茅聴聛忙聦聨莽鹿聜茅聨鲁氓潞篓氓录鲁茅聬聬莽聝聵莽聟聲氓漏碌莽聤虏忙麓娄氓篓聭脩聡忙拢聛忙戮露忙聞聟聼忙驴隆氓聞陇茅聢禄忙聸垄忙聼聸忙聬麓茂录聞莽茅聳潞氓虏聥莽庐聯氓庐聲忙漏聙茅聧聲盲潞陋茅聴聜盲陆赂忙聬聬卢氓楼赂氓麓掳茅聫聧猫聯聼茅聳赂茫聝庐氓聣職茅聫聟忙禄聢忙聼聸猫陇聨茅隆篓氓聭茅聨掳莽聣聢芒聮聮氓篓麓茂陆聢茅聧聥莽聤禄忙聼聸忙驴聽氓聠漏莽陇聥忙陇陇茫聢隆莽聣聴氓炉掳氓聥炉氓鹿聡茅氓聠漏忙聠掳茅聴聜盲陆鹿忙聻录茅聫聟忙露垄氓麓炉茅隆聳忙露聵氓聻聰茅聳鹿莽禄垄氓聳聴茅聬聯忙聸聼莽鹿聺茅聴聢忙露聶忙陇陇氓驴拢氓聛聤茅聫聦氓露聢茅聳赂忙聮麓莽鹿聢茅聰聺莽聜麓氓戮聞茅聨掳忙驴隆莽聜虏莽聙聸芒聲聟茅聭潞茫聝娄氓聫聫忙驴庐忙楼聙忙拢聴忙戮搂莽禄聵莽鹿聸茅聧聫猫聜漏氓垄碌茅聳潞氓卤戮莽茅隆娄猫聶芦茫聛聞茅聤聤茫聝娄氓職聡茅聳禄忙露卤氓聶拢忙驴庐芒聜聳驴忙露聵氓茅隆聰忙聸垄忙垄潞莽禄炉忙聸聻氓垄虏茅聳驴忙掳卢芒聰聭茅莽垄卤忙篓聙茅聳潞氓虏聥莽鹿聺氓庐聲氓聧聻芒聳聨猫鹿聡氓漏聝莽聙掳忙驴聽忙庐驴氓聳聳茅聰聲茫聝楼茅聮聵氓忙職娄茅聳碌氓漏聫氓莽禄隆芒聜卢茅聴聛忙聬聻氓聞聹茅聧聸莽聜?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬氓庐聞忙聡聤氓陆聮氓麓露猫陇聹氓篓麓忙聽聼聲忙驴聻莽聜卢忙芦聠茅聳赂茫聜聟氓茅聫聦茫聜聡莽虏聶莽聮潞茫聞陇莽虏碌氓漏碌莽聜虏忙聡聬忙驴庐氓聻漏卤忙驴聽氓聸搂脦搂茅聳赂氓卤戮莽聼戮鹿忙聸垄忙垄禄忙碌拢茅盲赂虏莽录聛猫鹿虏聶茅聨篓潞忙聺聢芒聲聝莽聲露茅聴聛莽禄聵忙垄禄茅聧聵脩聡忙聜聻茅聧篓盲潞聹茅聳鹿氓聯聞忙拢聴忙碌聹忙聝搂茅聤聢氓聴聵莽漏驴莽录聛忙聫聮氓垄陆莽禄庐氓卢鹿聮茅聫聜氓聯聞盲潞戮茅聳驴忙驴聠莽掳录莽禄篓猫聜漏芒聜忙聦赂莽禄聣忙聮聙聰茅聧聬脦搂茅聳赂芒聞聝脙陋茅聫聠忙忙垄潞茅聧聺茫聝楼脨拢茅聳鹿莽聭掳忙麓颅茅陋聻氓聠漏氓炉录茅聫聣忙聡鲁茫聞漏忙聼聧茅聧聺氓聥聜卢氓聭聤莽茅隆聯莽聜漏忙聰陆茅聳禄忙篓驴氓茫聝漏忙聼聼茅聧聬忙聼楼氓聫聞茅聳碌氓聴聴氓聙驴氓庐聲忙楼聟忙聡聳茅聢潞氓聠戮氓聲聨忙驴聽莽聰碌氓聛聻氓篓录忙卢聯茅聧隆忙陇禄忙聻聧茅聳赂茫聝拢氓聤聥莽禄禄氓聴聸忙聼聸氓篓聠氓驴拢忙搂赂氓漏碌忙麓颅茅聫聦氓露聡忙聠麓茅聧聲盲录聦茅聬聨猫搂聞忙麓聵茅聧聨氓楼赂脣聡茅聧聬莽聺卢氓潞篓氓麓露猫陇聺茅聼卢茅聬聬氓氓垄碌忙陇陇茫聢隆忙麓聼忙驴隆氓聽聲氓麓卤氓娄炉忙麓禄氓聰露茅聴聜氓聜聶莽路職氓脩聡脣聡茅隆聬茂鹿聙莽聳戮忙驴聽氓漏聜莽聣聤茅聧聥氓聜聻忙聼拢茅聫聝芒聜聨庐猫聢碌莽氓漏聤氓聥漏盲潞聹茅聳鹿忙聣鲁忙聶聸茅聬聮忙麓陋忙聼拢忙聬麓茫聢聽芒聳聲忙驴漏聞莽虏聨路氓聭虏碌莽聙拢莽搂路茅聴聜盲陆赂忙聬氓聭麓氓娄露莽禄聥猫路篓忙職娄忙驴聽氓漏聜氓聲聽茅聫聙盲禄聥忙聼聸茅隆聬莽聤虏莽聛聳禄忙聨聭忙陆隆氓搂聤忙麓陋氓麓路茅莽聜虏氓氓娄聻茫聜聝氓聤聦茅聨鲁氓潞隆脙?茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聟忙職娄茅聳禄茫聢垄莽禄聬忙楼聟忙聼聼茅聳鹿聙氓聤聮忙驴聻氓路潞聹茅聧聬芒聜卢氓娄聻茫聜聨氓聣娄茅聧聯忙掳卤莽虏聶茅聨麓茂鹿聙莽聟隆氓猫路潞脙隆茅聬聰猫聧陇莽聳聙茅聳潞氓聸漏氓聛聸忙驴隆猫聶鹿莽麓聯忙碌拢猫聶鹿氓赂聸莽录聛猫炉虏莽聣聠茅聬拢莽聝陆氓麓录茅聫聡莽聜聧芒聲聟忙聼篓茅聫聝氓聜聸忙芦聲莽聭聹猫聢卢氓聤聭氓莽禄聵忙聥聟莽禄聸氓忙庐录聜盲陆潞莽虏炉莽聰聙鲁莽聲路茅聨麓茂赂陆氓聛聬茅聳戮氓驴拢氓聛聞氓炉聺聴忙垄潞茅聴聢忙露職氓忙陇陇茂赂禄氓聻鲁忙戮鹿忙聸垄氓麓鹿茅隆聰猫忙聳聙茅聴聛莽禄聵脙陋氓炉聢碌氓戮聞茅聧聲莽聟聼茅聳鹿忙聝搂莽聯篓莽禄聙氓卢聼隆氓庐聙芒聜聧聽忙聽鹿聝氓漏聤氓聟戮莽聠潞忙聥聽茅聧聶氓茅聴聜氓聜聶猫聝露茅聨鲁忙聮禄氓麓炉茅聧搂忙莽聳陇陇忙聽篓莽虏聧氓庐聲氓聫聣莽鹿聸茅聨麓忙卢聫莽聛漏茅聳禄茅隆颅猫路篓忙聧聞忙楼娄氓聻庐茅聴聛氓聠虏氓聵虏茅聬聴忙聮鲁忙楼娄猫拢陇盲潞鹿茅聳鹿莽聝聵莽鹿聝茅隆楼忙聽篓莽鹿聺氓篓聢氓聻庐忙芦聧忙陇陇茫聝娄忙聦赂氓聧卤茅聳鹿莽聣聢忙戮聵莽禄聽忙聤陆忙聼聼茅聧聬猫聦录聧茅聳赂莽聤鲁忙聦戮氓娄虏忙聞掳氓鹿聮茅聨麓氓禄潞茅聴聛茅聙聻氓卤陆氓垄陇陇?氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄忙驴聻忙聢聰忙聦聥茅聤聤芒聲聛莽聞碌忙陇陇忙聨聭氓聙禄莽潞戮氓楼赂忙聼拢氓娄聦聨莽聫聙鹿忙聸聻莽聻录茅聢搂茅聧聽忙掳卤氓聢聮茅忙禄聞氓麓虏忙驴聽茅聺聸莽潞戮氓楼赂忙聼拢茅聨掳莽職潞茅聳赂氓漏聞氓聛陇氓搂聤忙麓陋氓麓聵茅聧聶氓陇聥氓聞聯茅聴聛莽茂鹿陇茅隆颅莽聝陆氓麓潞茅聢搂茫聛聞茅聨潞忙聢颅忙搂赂莽聙碌氓聳聴茅聨?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    dataset_name : str
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬氓庐聞忙聡聤氓陆聮氓麓露猫陇聹氓篓麓忙聽聼聲忙驴聻莽聜卢忙芦聠茅聳赂茫聜聟氓茅聫聦茫聜聡莽虏聶莽聮潞茫聞陇莽虏碌氓漏碌莽聜虏忙聡聬忙驴庐氓聻漏卤忙驴聽氓聸搂脦搂茅聳赂氓卤戮莽聼戮鹿忙聸垄忙垄禄忙碌拢茅盲赂虏莽录聛猫鹿虏聶茅聨篓潞忙聺聢芒聲聝莽聲露茅聴聛莽禄聵忙垄禄茅聧聵脩聡忙聜聻茅聧篓盲潞聹茅聳鹿氓聯聞忙拢聴忙碌聹忙聝搂茅聤聢氓潞隆盲潞聹莽录聛莽禄聵脙陋茅聬拢茅莽虏聶莽聮聡莽聜茅聧漏芒聜卢忙陇陇忙聨聭氓芦庐聯盲戮聙忙聼聲猫鹿聡氓篓戮氓聛聯茅聭录莽聲戮茅聴聜盲戮聙忙陆搂茅聬聴氓聴聶芒聜卢忙掳录氓聶拢猫聴聺茅聳驴忙聸聻氓聙鹿茅聬聯氓聠聠赂氓漏聤氓聽聺莽陇聢忙驴聻氓聴聵氓聻碌茅隆楼忙掳颅茫聛聞茅聨戮猫路潞脨拢忙陇陇忙麓聼茅聫聦莽聠录忙聜聹氓娄莽垄聧莽录聜盲陆赂氓茅聧聥忙露垄氓聼聻茅聨麓茂赂陆氓聛聬茅聳鹿莽禄聵氓路禄茅聧聥忙聮麓莽鹿聺氓搂聵氓聣庐茅聳鹿猫聣掳忙聫禄虏猫路篓氓茅聮聵氓氓陆聮忙聲聬茅聧聲莽聟聯茅聴聛氓聭聤莽鹿聠氓篓聟茂陆聡忙茅聨潞忙聴聙氓芦職茅聳鹿莽禄聵氓赂聴氓篓聢猫聦卢莽麓?``"miluv"``茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮莽聲聦忙聠赂氓庐楼氓聽垄盲潞卤茅聴聜盲陆赂忙聬芦茅聧聤氓陆聮氓篓聠氓驴拢莽聞陆茅聳潞氓露聝氓聤聧茅聧?ntu_viral"``茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮莽聲聦忙聠赂氓庐楼氓聽垄盲潞卤茅聴聜盲陆赂忙聬芦茅聧聤氓陆聮氓篓聠氓驴拢莽聞陆茅聳潞氓露聝氓聤聧茅聧?sim"``茅聴?
    raw_root_override : str | None
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛氓露聣氓麓聼茅莽聤碌氓聨炉茅聴聜盲陆赂忙鹿卤茅聧聨茫聜聣茫聙聝茅聧聸忙聺聭氓聻聜茅聳赂氓卤戮莽篓聯莽禄隆芒聜聴聜氓聜職莽聣聤忙赂職忙氓麓聲猫鹿聡忙聸聽莽拢录茅聳禄忙卢聦氓聛聯茅聳赂茂陆聫莽露聛氓炉氓录麓茅聤聫莽聤禄氓聰露氓漏碌莽聤禄忙陆搂氓篓虏茫聜聣脣聡茅聢潞氓聭聮聭莽录聛氓茅聧聨忙聞掳忙聼篓茅聫聡忙篓潞莽聛虏忙楼聽莽聜虏氓聲隆芦忙禄聢氓聼聙茅隆聮氓聥芦莽庐聳忙驴聻氓聴聵忙聦禄茅隆楼氓聽聼莽鹿聸茅聨麓莽聝聠忙庐聰忙陇陇茫聝娄氓虏赂氓莽禄聵忙聥聟茅聧聺茫聝炉莽麓聻莽录聜盲戮聡氓聶庐茅聧篓盲录麓茂陆聻猫鹿聡忙聴聜莽炉聙忙驴漏聠氓聣聸茅聧聰茂鹿聙茅聤聢氓聴聴莽卢聜莽录聛猫庐鹿莽鹿聺氓庐聲莽庐聸氓篓聭忙卢聭氓聨卤茅聴聛莽禄聵脙潞戮忙聶露茅聳卢氓露聠莽聟聶?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聳莽潞戮氓聺聴忙聼拢茅聤聫茫聢漏莽聞戮茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛莽篓驴茅氓聴聶忙聧垄氓漏碌氓聸漏莽炉聙茅聳赂氓卤戮莽垄聧氓篓聢忙禄聢忙垄潞茅聧聺氓聥碌氓聞聬忙驴隆氓聲麓氓氓漏職忙聜垄莽聬聸芒聙鲁盲潞戮茅聳禄茫聢隆茅聬聮茅鹿聝氓录陆茅聰聼莽聤鲁氓聙?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓麓茂陆聢氓篓聟莽聤禄氓鹿聳氓篓拢氓莽禄戮忙聝搂茅聫聦忙聸垄氓麓录氓漏碌忙聞娄茅聴聛氓聯聞茅聭鲁氓聽聲氓聼聙茅隆聮芒聜卢莽禄聽氓露聣氓麓聲茅聧聴莽聛禄忙麓陋忙聜漏莽聮聡猫聦录聸茅聴聛莽聭掳氓垄陆莽禄庐氓麓聭茅聰聼莽聤鲁莽聟聶忙驴聻氓聽聺莽聛聺茅聫聥忙聴聢忙聼聧猫陇聹茅聧聯忙卢聯氓麓聺茅聢潞莽聨颅e`` 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛氓露聣氓麓聼茅盲陆鹿忙驴聽氓戮路忙聺驴芒聜聸垄氓鹿聤猫鹿聡忙露聵莽拢禄茅聳驴莽聠潞氓搂碌茅聬聯氓驴聸忙聼聸茅隆聬茂陆聡莽庐聳茅聳鹿氓聟录氓聤聶茅聫聦忙露垄氓娄赂茅聰聰氓聣搂莽聳聞茅聴聛氓聯聞莽聼聣莽禄虏氓聙聼莽聛聮茅聴聛氓聟录莽楼麓茅聫聟忙露職莽虏颅茅聰聼莽聤鲁芒聮聭莽录聜盲陆鹿脡聭莽聛聫茫聞漏忙聼聸莽聙拢氓麓聦茅聳潞氓虏聥莽碌戮茅聨炉忙聴聢忙聲聧茅聧聲忙茅聴聜盲陆潞忙楼聽莽莽聲路茅氓聜職氓录虏茅聴聜盲戮聙忙陆搂猫聣聦茅聳潞氓聭庐莽聸庐聲忙卢聯忙聜漏莽录聛忙氓聨卤茅聬聨氓聟聣氓聛聯茅聳鲁脩聦氓聣聼氓庐聲忙聢娄忙聜聶茅聧聺氓聥录聸茅聴聛忙聦聨莽鹿聜茅隆娄盲录麓忙聜隆茅聫聡茫聢隆莽聟聫氓猫路潞莽聣聠茅聧聰忙禄聞茫聛聞茅聫聜氓聻聳潞氓虏聥莽露聛茅聨聢陇氓麓聤聣脩聟氓聨聭忙驴聽莽聰碌氓搂掳茅聧篓忙麓陋忙聲聥茅聳禄忙聮卤氓聙聳茅聤聡氓聥鹿聵莽聮潞莽聝聵莽聙潞莽录聜?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    Path
        茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聟忙職娄茅聳禄茫聢垄莽禄聬忙楼聟忙聼聼茅聨碌茅聳赂氓卢露莽鹿聺氓炉聽聲氓麓录氓莽聫聰莽录聜盲陆陆莽驴掳茅聧聤忙聸隆莽聛聝茅聴聛忙聦聨莽鹿聜茅聨鲁氓潞篓氓录鲁茅聬聬莽聝聵莽聟聲氓漏碌莽聤虏忙麓娄氓篓聭脩聡忙拢聛忙戮露忙聞聟聼忙驴隆氓聞陇茅聢禄忙聸垄忙聼聸忙聬麓茂录聞莽茅聳鹿氓聵虏茅聢禄氓潞隆芒聳聨茅聨麓莽聤鲁忙忙驴隆茫聜聡氓搂路茅聧聥莽聜潞忙聜搂氓娄陇芒聜卢茅隆聸陋莽庐聸氓篓聭忙篓录茅聬聼忙禄聝莽鹿聴茅聨陆忙垄禄茅聧聦忙卢聬莽录聛茅鲁聳氓聳職莽禄卤莽聜潞氓聼聙茅隆聮芒聜卢茅聹聣忙驴聽氓漏聜茅忙戮搂莽聣聢盲驴聤茅聧聶氓陇聤氓忙楼聽莽聜麓茅聢搂莽篓聮茅聰職忙陇陇氓潞垄忙聧聡氓搂聤忙麓陋忙拢聝茅聢潞盲陆潞忙搂拢茅聴聛氓聭聤茂鹿垄莽禄聽忙聽鹿聝?茅聴?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓麓茂陆聢氓篓聟莽聤禄氓鹿聳氓篓拢氓莽禄戮忙聝搂茅聫聦忙聸垄氓麓录氓漏碌氓聸露忙莽录?

    茅聴聜氓聜職氓聙路氓篓麓氓聸茅聢搂莽篓聢茅聫聟莽聤禄茫聙聛氓篓聠忙聮鲁氓聠聙忙陇陇忙聮露氓聛聼茅聬聸茫聝漏忙垄潞茅聧聺氓聥颅芒聳聣茅聳赂忙篓驴氓聛聬忙戮鹿忙聸隆茫聛聬茅聨潞忙聮露氓聨碌茅聴聜氓聜職氓聙赂茅氓麓聺氓氓陇聥莽聟聲?
    ------
    ValueError
        茅聴聜氓聜職氓聙路氓篓麓氓聸搂莽聲路茅聬垄莽莽陋聴茅聳鹿茅聜娄氓聳聳茅聧篓忙驴聥氓鹿聳氓篓拢莽聛驴聻氓聭搂碌忙路聡氓漏聡氓娄露茅聧聸忙芦拢莽录聜?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聳莽潞戮氓聺聴忙聼拢茅聤聫茫聢漏莽聞戮茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛莽篓驴茅氓陇聬氓录禄茅聰聺氓莽庐聸忙陇陇忙聴聜氓聨搂忙驴隆忙麓陋忙垄潞莽禄聥氓聻陇陇茫聞楼茂鹿陋氓炉忙聜垄莽潞掳猫戮篓忙聶卢茅聴聛莽鲁聲氓聣聺茅隆篓氓陇聦芒聜卢忙掳卤氓莽禄禄忙露垄氓麓录茅氓聳聤氓漏碌茂鹿楼氓猫聴聣莽聲路茅聤聤茂赂戮莽炉聙茅聳赂忙聞碌莽聟聢氓娄虏氓聧聻芒聰聭茅聬聬猫聦卢忙聭聽莽录聛莽碌驴氓氓漏聞忙聧聞茅聤聤茫聝楼莽聛聤茅聴聛氓聣虏氓聛聙茅聨鲁氓聸卢氓麓聭茅聨戮莽禄聵忙聶虏茅聨录莽虏鹿氓聣聺茅聬聫氓露聣忙垄潞茅聨露猫聤楼忙聲卤茅聬垄莽鹿聺氓猫炉虏芒聵聣茅聤聫芒聲聜忙聶聺茅聴聛莽禄聵莽聛聫忙卢聬莽茅聧聫猫聜漏莽庐戮茅聬聨莽聰碌茅隆芦忙聨聴忙聼聸茅聤聣茂陆聟氓娄驴茅聳赂忙篓禄忙聧聡茅聫聦茂陆聣忙聜垄茅聧聺茫聝娄氓聶搂茅聳禄氓潞垄氓聡鹿茅聧聽忙楼聟氓录虏茅聭赂氓聻芦莽露聡茅聳鲁氓聯聞氓聲掳茅聰聸忙禄聝茅聤聢氓聴聵忙驴隆氓氓庐聲脙潞茅氓聭聨碌茅聴聛莽禄聵忙聦赂莽聙聸芒聲聝忙聥鹿茅聰聼莽聤鲁莽聟聶茅聳赂忙卢聫莽聛聫茂赂戮芒聜忙麓聹茅聧聽忙聽搂莽聲路氓娄陇氓聭颅氓路聺氓漏碌莽聤虏氓莽录聧忙忙垄禄茅聧聦忙掳聜卢氓鲁掳脙鹿茅聧楼忙聲聥莽聭聹氓驴聸氓聼聙茅隆聮盲陆潞芒聳聯茅聳潞氓聭鹿聞茅聧聮忙聨聭芒聳聨茅聨戮氓麓聡氓漏碌茂录聞氓聛聸茅聬篓莽聝陆氓麓?氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄茅聳禄忙聞聬芦茅聧篓氓聜聻忙聼聸茅茂陆聠莽莽录聛莽聤鲁氓拢聠莽禄卤忙聨聰芒聜卢茂陆聨氓聛聮茅聧聨茫聝漏忙聼聸茅聤聢氓聴聵莽茅聳鹿莽露聤茅聨禄聝脦聼氓搂鹿氓聸娄猫聶聨茅聳禄氓潞陇氓篓虏忙篓录忙聲庐茅聰聼莽聤虏莽庐聳忙驴聻氓聴聵忙聦禄茅聧陇忙聢聻茫聛聞茅聨潞忙聢搂赂茅聳潞盲陆陆忙鲁聸茅聢鹿忙聢娄忙聜漏茅聧篓忙炉聞茅聴聛莽茅聧聥茫聜聟莽聲路猫陇掳忙聨聭茅聠聡茅聳潞氓聸漏氓聳聬氓篓聟忙聳驴芒聰聭茅聬聵忙聞聛氓聲聫茅聳赂茂陆聨氓聻鲁氓炉掳氓漏聝莽娄聮莽聙拢氓潞隆忙麓聨庐莽聲聦莽虏聶茅聭鲁忙聠掳?
    """
    if raw_root_override is not None:
        if not raw_root_override.strip():
            raise ValueError(f"--{dataset_name}-raw-root must be a non-empty path")
        raw_root = Path(raw_root_override)
        if not raw_root.is_absolute():
            raw_root = ROOT / raw_root
        return raw_root.resolve()

    dataset_cfg = copy.deepcopy(
        _load_yaml_config_cached(str((ROOT / "configs" / "datasets" / f"{dataset_name}.yaml").resolve()))
    )
    raw_root = Path(dataset_cfg["raw_root"])
    if not raw_root.is_absolute():
        raw_root = ROOT / raw_root
    return raw_root.resolve()


def _list_sequence_ids(raw_root: Path) -> list[str]:
    """茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛氓露聣氓麓聼茅盲陆鹿忙驴聽氓戮路忙聺驴芒聜聸垄氓鹿聤氓庐聙氓聥聞碌忙陇陇忙聨聠氓聛聯茅聳赂莽聠赂忙陆聶茅聬拢莽聝陆氓娄赂茅聢潞盲陆潞茅陋聡茅聴聛莽聭掳莽聯篓莽禄禄氓聠麓聲茅隆聫氓聭卢莽鹿聠茅聳禄忙聞碌盲潞聹茅聢搂莽聣聽茅陋聻忙聞掳氓麓录茅聫聡莽聜虏莽聙聬聼忙禄聟氓莽聰聨聰氓拢聲忙驴聜氓聭聤莽聟聫氓漏碌莽聜碌氓聛聯氓篓聟氓聴聸忙聼拢?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬氓庐聞忙聡聤氓陆聮氓麓露猫陇聹氓篓麓忙聽聼聲忙驴聻莽聜卢忙芦聠茅聳赂茫聜聟氓茅聫聦茫聜聡莽虏聶莽聮潞茫聞陇莽虏碌氓漏碌莽聜虏忙聡聬忙驴庐氓聻漏卤忙驴聽氓聸搂脦搂茅聳赂氓卤戮莽聼戮鹿忙聸垄忙垄禄忙碌拢茅盲赂虏莽录聛猫鹿虏聶茅聨篓潞忙聺聢芒聲聝莽聲露茅聴聛莽禄聵忙垄禄茅聧聵脩聡忙聜聻茅聧篓盲潞聹茅聳鹿氓聯聞忙拢聴忙碌聹忙聝搂茅聤聢氓聴聵莽漏驴莽录聛忙聫聮氓垄陆莽禄庐氓卢鹿聮茅聫聜氓聯聞盲潞戮茅聳驴忙驴聠莽掳录莽禄篓猫聜漏芒聜忙聦赂莽禄聣忙聮聙聰茅聧聬脦搂茅聳赂芒聞聝脙陋茅聫聠忙忙垄潞茅聧聺茫聝楼脨拢茅聳鹿莽聭掳忙麓颅茅陋聻氓聠漏氓炉录茅聫聣忙聡鲁茫聞漏忙聼聧茅聧聺氓聥聜卢氓聭聤莽茅隆聯莽聜漏忙聰陆茅聳禄忙篓驴氓茫聝漏忙聼聼茅聧聬忙聼楼氓聫聞茅聳碌氓聴聴氓聙驴氓庐聲忙楼聟忙聡聳茅聢潞氓聠戮氓聲聨茅聴聜盲陆赂忙聠隆茅聬聼茫聝漏脣聡茅隆聳莽聜麓氓聙露茅聳鲁氓聯聞氓聲掳莽潞戮氓聟录茫聛聞茅聫聡莽聜麓氓聶鹿莽聙碌氓聳職茅聢搂氓篓虏氓驴聸氓麓聺茅聨麓忙驴聬氓職聲茅聳赂忙麓聳莽禄聽茂陆聣忙聼拢氓篓聠氓卤陆盲潞戮茅聴聛氓聭聤猫聣戮茅隆聭氓莽录聛忙莽聭鹿茅聳鲁脩聦氓聣聼茅聧漏芒聜陇陇忙聨聭氓聙赂忙碌聽忙禄聢忙聼陇氓篓虏猫莽聛隆茅聳潞氓聭茅聢搂忙聲赂莽潞聽聲氓麓聭茅聨戮忙聳楼忙聻聰茅聳赂氓聳聴茅聬聫氓聽楼忙垄潞莽聮聡猫聦茅聳录氓陆聮莽庐聳茅隆垄忙掳露氓娄路茅聢潞盲陆潞氓娄聻茫聜聠忙聦禄莽禄聥忙聢聻莽忙楼拢氓聠漏莽聟楼忙驴聽茅聺聸忙忙戮露氓聥卢莽鹿聸茅聧芦莽聠赂茫聛聺忙驴漏聜氓庐聲忙聨聭芒聳聨茅聨掳氓聛聵忙驴聽莽垄聵氓聣聺茅聤聡忙禄聢氓麓聺忙聬麓茂陆聟氓聻聺茅聳赂氓聳聯茅聬聼氓陆聮忙聼聧猫茅聧聯莽聠录莽聧聧聬忙篓聞茅聬聨猫搂聞忙麓聳氓庐聲氓聼垄忙聬麓茫聞楼莽庐鲁茅聳鹿氓聠虏莽聛聢茅聢搂氓聣職氓篓聯氓忙聼聲茅聧芦忙驴聡茅聬聫氓驴拢盲潞陋茅聫聦忙露垄氓娄赂茅聤聣茂鹿聛莽莽录聜盲戮聡氓聳聳茅聬聴忙聮鲁莽聲路茅聧聴莽聜聧莽聠录氓麓路茅隆聯莽聤碌忙職掳茅聴聜氓聜聶莽路職氓篓录脩聡忙聜搂茅聧隆忙露聵莽庐聽茅聨录忙路聤聦氓搂鹿忙聼楼忙聼拢茅聨掳忙職漏莽禄戮忙聝搂茫聜赂茅聫聦莽聠录忙聼聡茅聰聲芒聜莽聜鹿聸茅聧聡聜猫聙驴茅聳潞氓虏聙氓聳聳氓庐聲茂陆聠氓戮聞莽聙拢氓聛聯茅聨掳莽聣聢莽聟聶忙陇陇忙聳禄氓聤聦氓篓虏茫聜聣忙聼聥茅聧楼莽聟聼茅聳潞氓聠篓氓聙赂茅聬篓忙聢聽莽麓聮忙搂聦忙聽搂氓录驴庐?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    raw_root : Path
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏氓庐聲氓聸搂氓鹿聢?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬氓庐聞忙聡聤氓陆聮氓麓露猫陇聹氓篓麓忙聽聼聲忙驴聻莽聜卢忙芦聠茅聳赂茫聜聟氓茅聫聦茫聜聡莽虏聶莽聮潞茫聞陇莽虏碌氓漏碌莽聜虏忙聡聬忙驴庐氓聻漏卤忙驴聽氓聸搂脦搂茅聳赂氓卤戮莽聼戮鹿忙聸垄忙垄禄忙碌拢茅盲赂虏莽录聛猫鹿虏聶茅聨篓潞忙聺聢芒聲聝莽聲露茅聴聛莽禄聵忙垄禄茅聧聵脩聡忙聜聻茅聧篓盲潞聹茅聳鹿氓聯聞忙拢聴忙碌聹忙聝搂茅聤聢氓聴聵莽漏驴莽录聛忙聫聮氓垄陆莽禄庐氓卢鹿聮茅聫聜氓聯聞盲潞戮茅聳驴忙驴聠莽掳录莽禄篓猫聜漏芒聜忙聦赂莽禄聣忙聮聙聰茅聧聬脦搂茅聳赂芒聞聝脙陋茅聫聠忙忙垄潞茅聧聺茫聝楼脨拢茅聳鹿莽聭掳忙麓颅茅陋聻氓聠漏氓炉录茅聫聣忙聡鲁茫聞漏忙聼聧茅聧聺氓聥聜卢氓聭聤莽茅隆聯莽聜漏忙聰陆茅聳禄忙篓驴氓茫聝漏忙聼聼茅聧聬忙聼楼氓聫聞茅聳碌氓聴聴氓聙驴氓庐聲忙楼聟忙聡聳茅聢潞氓聠戮氓聲聨茅聴聜盲陆赂忙鹿卤莽禄聲庐茅聬聬莽聰碌盲赂聧氓漏碌莽聤鲁莽茅聬聯忙聸垄忙聼聧莽聭聶氓聥聤陇氓篓?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    list[str]
        茅聴聜氓聜職氓聙赂茅聧聤莽聝陆忙聡聴茅聧聸聻莽庐聽茅聳鹿忙聧聬莽聭聶茂鹿聜忙聜聴茅聰聺氓潞聺忙聠戮茅聳禄忙聨聲氓搂陇莽禄禄忙露垄氓麓卤氓娄炉猫聶鹿盲禄赂茅聬聨猫忙麓聵茅聬聯茂驴聽氓录禄茅聬聰氓聟录氓聛聥茅聳赂氓聳聯茅聧聭茂驴聽忙垄潞?ID 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛氓露聣氓麓聼茅盲陆鹿忙驴聽氓戮路忙聺驴芒聜聸垄氓鹿聤氓庐聙氓聥聞碌忙陇陇忙聨聠氓聛聯茅聳赂莽聡聛氓聻掳茅隆聲茫聝漏氓麓聼茅忙掳卤莽聳聙氓娄聻茫聜聠氓聻聼茅聫聜氓聜聻氓麓聲茅聬垄莽猫聯聼忙驴聻忙聢聶氓聻鹿莽禄聽忙露聶茫聛聞茅聨戮猫路潞忙聻聰莽聭聹猫聢卢氓聤聭氓莽禄聵忙聥聟莽禄聸氓忙庐录聜盲陆潞莽虏炉莽聰聙鲁莽聲路茅聨麓茂赂陆氓聙路茅聳赂忙驴聠氓茅聰聸莽聠录忙聜聴莽聭聶氓聥忙驴聻氓卤聜氓聤聨莽禄庐氓漏職氓录庐茅聧聠潞氓聨碌茅聴聛氓聭聤忙聦聠茅聧聸茅聴聜?``config`` 茅聴?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳赂忙驴聞氓聞鲁茅聬聮禄忙聜聴茅聴聢忙露聶忙聠赂茅聳禄忙聧聡茅聫聦茂陆聣忙聜垄莽禄聥忙卢戮忙茅聴聛氓聯聞忙拢麓莽禄聽忙聮禄氓录禄茅聰聺氓忙聶虏茅聳赂芒聞聝莽聭搂茅聬陇氓聥炉忙聜聴莽聭聶氓聥茅聳赂莽聤鲁莽聣聲茅聬拢莽聝陆氓娄路茅聢潞盲陆赂莽陋聸茅聳禄氓潞垄莽篓聮猫聫陇茅聳鹿茅聰聥忙陇陆氓搂聤忙麓陋氓麓聴茅聴聜莽拢聥氓聼聟茅聴聛莽茅陋赂莽禄聣氓陆聮茫聙聛茅聫聝氓聽聺氓鹿垄忙驴隆忙聬芦莽禄篓茫聝漏忙垄禄忙碌拢氓聭聤猫麓隆茅聳赂氓潞聸氓聙聺氓庐聲茅聺聸茅莽聜麓忙陇陆茅隆垄忙聴聢氓麓聼茅脩聝忙聲赂茅聴聜盲陆陆莽禄禄忙聨聴氓麓聭茅聬聵忙麓禄忙聲聤氓漏碌莽聤虏氓聙聬猫猫聻聳茅聳赂忙露卤氓聳聮茅聨路忙麓陋忙垄潞茅聧聸氓漏聞莽庐聯茅聬聨忙掳录氓聨陆氓氓聥麓聼茅隆聳忙露聵忙庐篓茅聴聛茅聺聸莽聣聠茅隆娄盲录麓忙聜?
    """
    if not raw_root.is_dir():
        return []
    return [
        path.name
        for path in sorted(raw_root.iterdir())
        if path.is_dir() and path.name != "config" and not path.name.startswith(".")
    ]


def _inspect_sim_raw_readiness(raw_root: Path) -> dict[str, Any]:
    """忙驴聽莽聰碌氓搂路茅聫聛氓聭聤忙聟篓茅聬聭氓芒聜聜聻忙聦聤忙戮鹿芒聲聥莽露聛氓庐聲茫聞漏氓录露茅聨麓莽聥聟莽麓卤茅聴聜盲戮聙芒聜聜虏猫聥戮搂忙聮聼隆莽聛聻氓聣搂茫聙聮茅聳鲁脩聦氓聣篓莽录聧氓聴聸氓麓聭茅聧聸忙職娄莽聙鹿芒聜卢茅聧聲氓聨赂?SIM 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏氓庐聲氓聸搂氓鹿聢?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬氓庐聞忙聡聤氓陆聮氓麓露猫陇聹氓篓麓忙聽聼聲忙驴聻莽聜卢忙芦聠茅聳赂茫聜聟氓茅聫聦茫聜聡莽虏聶莽聮潞茫聞陇莽虏碌氓漏碌莽聜虏忙聡聬忙驴庐氓聻漏卤忙驴聽氓聸搂脦搂茅聳赂氓卤戮莽聼戮鹿忙聸垄忙垄禄忙碌拢茅盲赂虏莽录聛猫鹿虏聶茅聨篓潞忙聺聢芒聲聝莽聲露茅聴聛莽禄聵忙垄禄茅聧聵脩聡忙聜聻茅聧篓盲潞聹茅聳鹿莽聝聵氓聻碌茅芒聲聝莽麓聮?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳赂忙驴聞氓聞鲁茅聬聮禄忙聜聴茅聴聢忙露聶忙聠赂茅聳禄忙聧聡茅聫聦茂陆聣忙聜垄茅聬聯氓潞聺氓聳聤茅聴?茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙碌拢茅聺漏猫聙驴茅聴聛莽搂麓莽禄聙氓陇聦芒聜卢氓鹿驴氓聞卤氓娄潞垄莽娄聧氓氓聥漏盲潞聹茅聳鹿氓聯聞忙拢聴忙碌聹忙聝搂芒聰聭茅聬聬氓聫聣芒聳聲茅聳赂忙卢聫氓聲麓茅聨?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓聙驴茅聳驴忙聴聢忙聲庐茅聧聥忙聮露莽虏篓聭忙篓禄氓聶陆茅聳禄莽聭漏茅聫聦莽聠赂忙陆聫忙楼拢氓聠漏茅聴聛忙聬聻氓聙聳茅聧聰忙卢聯氓录禄茅聬聰茅忙職娄茅聳赂茫聝聭氓聥颅盲潞聴聜盲陆陆茅聧聽忙聮禄氓麓鹿茅聮聵氓忙職娄茅聳碌氓漏聫氓莽禄隆芒聜卢忙驴聻猫戮戮莽碌驴茅聨鲁忙聮禄氓录篓茅隆聯茫聢隆芒聮聮茅聳赂氓卤戮莽聭搂茅隆娄茂鹿聜忙聼聼莽潞掳氓聧聻盲潞掳莽聙鹿忙聸聼氓聤聶氓庐聲莽聝陆氓篓聭忙篓鹿氓莽录聜盲陆鹿莽禄聥忙聢娄氓鹿聤茅聨掳莽聣聢氓聞聴茅聳赂芒聞聝莽篓聮茅聬聯氓聠聠赂氓漏聤氓聽聺莽陇聢茅聳禄忙聳驴氓忙聽搂芒聰聭茅聬聵氓庐聽氓拢聵莽禄戮忙聝搂氓聬聥茅聨卤茫聞楼茅聧隆忙氓鹿聥茅聴聛莽篓驴茅聨鹿氓聸搂莽聲路忙驴聜莽氓聧鲁茅聳禄忙聞娄虏?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聫茅聢搂莽聞娄莽禄禄忙驴聥氓麓聼茅隆聯莽聥聙忙搂卤氓漏碌莽聜麓忙陆聶茅聧職氓氓篓聠忙聢聺茅聳赂脩聞茅聢路忙聢聻茫聛聞茅聨潞氓聴聴盲潞戮莽录聜盲陆赂茅氓聢聮忙驴聽忙掳聛聞茅聳赂氓驴職氓聛聬茅聧聶氓聴聸忙垄潞莽录聛忙篓禄莽聟楼茅聳鹿莽垄卤氓聛聞茅聬隆忙聼隆?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    raw_root : Path
        SIM 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏氓庐聲氓聸搂氓鹿聢?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬氓庐聞忙聡聤氓陆聮氓麓露猫陇聹氓篓麓忙聽聼聲忙驴聻莽聜卢忙芦聠茅聳赂茫聜聟氓茅聫聦茫聜聡莽虏聶莽聮潞茫聞陇莽虏碌氓漏碌莽聜虏忙聡聬忙驴庐氓聻漏卤忙驴聽氓聸搂脦搂茅聳赂氓卤戮莽聼戮鹿忙聸垄忙垄禄忙碌拢茅盲赂虏莽录聛猫鹿虏聶茅聨篓潞忙聺聢芒聲聝莽聲露茅聴聛莽禄聵忙垄禄茅聧聵脩聡忙聜聻茅聧篓盲潞聹茅聳鹿氓聯聞忙拢聴忙碌聹忙聝搂茅聤聢氓聴聵莽漏驴莽录聛忙聫聮氓垄陆莽禄庐氓卢鹿聮茅聫聜氓聯聞盲潞戮茅聳驴忙驴聠莽掳录莽禄篓猫聜漏芒聜忙聦赂莽禄聣忙聮聙聰茅聧聬脦搂茅聳赂芒聞聝脙陋茅聫聠忙忙垄潞茅聧聺茫聝楼脨拢茅聳鹿莽聭掳忙麓颅茅陋聻氓聠漏氓炉录茅聫聣忙聡鲁茫聞漏忙聼聧茅聧聺氓聥聜卢氓聭聤莽茅隆聯莽聜漏忙聰陆茅聳禄忙篓驴氓茫聝漏忙聼聼茅聧聬忙聼楼氓聫聞茅聳碌氓聴聴氓聙驴氓庐聲忙楼聟忙聡聳茅聢潞氓聠戮氓聲聨茅聴聜盲陆赂忙鹿卤莽禄聲庐茅聬聬莽聰碌盲赂聧氓漏碌莽聤鲁莽茅聬聯忙聸垄忙聼聧莽聭聶氓聥聤陇氓篓?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    dict[str, Any]
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳鹿茅聴麓莽莽潞戮氓陆聮忙聼聼茅聴聜氓炉赂莽禄戮忙聝搂莽露聤茅聫聦忙露垄茅聰聺氓聴聶莽录聜盲陆虏氓聛?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聫茅聢搂莽聞娄莽禄禄忙驴聥氓麓聼茅隆聯莽聥聙忙搂卤氓漏碌莽聜麓忙陆聶茅聧職氓氓篓聠忙聢聺茅聳赂脩聞茅聢路忙聢聻茫聛聞茅聨潞氓聴聴盲潞戮莽录聜盲陆赂茅氓聢聮忙驴聽忙掳聛聞茅聳赂氓驴職氓聛聬茅聧聶氓聴聸忙垄潞莽录聛忙篓禄莽聟楼茅聳鹿莽垄卤氓聛聟茅聬聴氓潞篓忙垄禄忙碌拢氓聭聤忙聝聢茅聳鹿氓鲁掳氓庐聲忙禄聢忙聜垄茅聬聯氓潞拢莽聲潞氓漏碌茫聛聸茅聧聥忙聮禄忙聼聼茅莽聲聦忙聡聡茅聳鹿猫聶芦莽虏聦茅聢禄忙聮禄氓鹿聬忙聬麓茅聴聜氓聜職氓聙路氓忙聨聰忙茅隆聰氓聣搂盲赂聧茅聳鹿猫戮戮茅聶聞茅聧聥氓芒聰聭茅聧聦忙禄聨莽聞戮茅聳?``dataset_name``茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮莽聲聦忙聠赂氓庐楼氓聽垄盲潞卤茅聴聜盲陆赂忙聬芦茅聧聤氓陆聮氓篓聠氓驴拢莽聞陆茅聳潞氓露聝氓聤聧茅聧聶氓驴聯芒聰聭茅聬聵忙聫聮芒聜卢氓漏職忙聜掳氓篓聯tus``茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮莽聲聦忙聠赂氓庐楼氓聽垄盲潞卤茅聴聜盲陆赂忙聬芦茅聧聤氓陆聮氓篓聠氓驴拢莽聞陆茅聳潞氓露聝氓聤聧茅聧聶氓驴聯芒聰聭茅聬聵忙聫聮芒聜卢氓漏職氓聼聻茅聨卤莽e_action``茅聴?
        ``reasons``茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮莽聲聦忙聠赂氓庐楼氓聽垄盲潞卤茅聴聜盲陆赂忙聬芦茅聧聤氓陆聮氓篓聠氓驴拢莽聞陆茅聳潞氓露聝氓聤聧茅聧聶氓驴聯芒聰聭茅聬聵忙聫聮芒聜卢氓漏職忙聜掳氓篓聯氓聰聯uence_ids``茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮莽聲聦忙聠赂氓庐楼氓聽垄盲潞卤茅聴聜盲陆赂忙聬芦茅聧聤氓陆聮氓篓聠氓驴拢莽聞陆茅聳潞氓露聝氓聤聧茅聧聶氓驴聯芒聰聭茅聬聵忙聫聮芒聜卢氓漏職忙聜掳氓篓聯氓聰聯uences``茅聴?
        ``required_layout_family_count``茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮莽聲聦忙聠赂氓庐楼氓聽垄盲潞卤茅聴聜盲陆赂忙聬芦茅聧聤氓陆聮氓篓聠氓驴拢莽聞陆茅聳潞氓露聝氓聤聧茅聧聶氓驴聯芒聰聭茅聬聵忙聫聮芒聜卢氓漏職忙聜掳氓篓聯ports_clean_train_val_test_family_split``
        莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳禄忙聢拢氓搂陇茅聧聤氓聺聴忙聼篓茅聫聜氓聻聮聰茅聳禄莽聭漏茅聫聦莽聠路芒聳聨茅聶聡茅隆聲脩聝脙潞茅茫聢垄氓碌聛茅聳赂忙聞篓忙聳聜茅聧聥氓漏聝莽鹿聸茅聧隆忙卢聫盲潞戮忙碌聹忙露垄忙垄禄茅聧聦忙卢聭莽禄戮莽禄垄氓聨戮茅聧聮忙聨聴茅聬聬猫聦聴聛忙聬聻氓聞潞茅聧聯忙卢聭莽虏禄茅聫聧茫聞搂莽茅聳赂芒聞聝脡聭莽聛聫莽麓聮茅聢搂氓麓聮氓搂聵猫聜漏氓聰碌茅聳禄莽聤潞忙莽聛聫茫聞楼氓娄陇芒聜卢茅隆?
    """
    sequence_ids = _list_sequence_ids(raw_root)  
    sequence_reports: dict[str, Any] = {}  
    ready_sequence_count = 0  
    for seq_id in sequence_ids:  
        seq_dir = raw_root / seq_id  
        missing_files = [name for name in _SIM_REQUIRED_FILES if not (seq_dir / name).is_file()]  
        ready = not missing_files  
        if ready:  
            ready_sequence_count += 1  
        sequence_reports[seq_id] = {  
            "seq_id": seq_id,  
            "seq_dir": str(seq_dir),  
            "status": "ready" if ready else "not_ready",  
            "missing_files": missing_files,  
        }  
    reasons: list[str] = []  
    if not raw_root.is_dir():  
        reasons.append("missing_raw_root")  
    if not sequence_ids:  
        reasons.append("missing_raw_sequence")  
    if ready_sequence_count != len(sequence_ids):  
        reasons.append("missing_required_streams")  
        if ready_sequence_count == 0 and sequence_ids:  
            reasons.append("incomplete_existing_raw_contract")  
    layout_family_report: dict[str, Any] | None = None  
    sim_materialized_contract_report: dict[str, Any] | None = None  
    supports_clean_train_val_test_family_split = None  
    if raw_root.is_dir() and sequence_ids:  
        layout_family_report = inspect_layout_family_coverage(raw_root)  
        if not layout_family_report["is_valid"]:  
            reasons.append("invalid_anchor_layout_family_metadata")  
        supports_clean_train_val_test_family_split = (  
            bool(layout_family_report["is_valid"])  
            and int(layout_family_report["family_count"]) >= _PAPER_MIN_SIM_LAYOUT_FAMILY_COUNT  
        )  
        if not supports_clean_train_val_test_family_split:  
            reasons.append("insufficient_independent_layout_families")  
        sim_materialized_contract_report = inspect_sim_materialized_contract(raw_root)  
        if not sim_materialized_contract_report["is_valid"]:  
            reasons.append("stale_sim_materialized_contract")  
    overall_ready = (  
        bool(sequence_ids)  
        and ready_sequence_count == len(sequence_ids)  
        and "invalid_anchor_layout_family_metadata" not in reasons  
        and "insufficient_independent_layout_families" not in reasons  
        and "stale_sim_materialized_contract" not in reasons  
    )  
    return {  
        "dataset_name": "sim",  
        "raw_root": str(raw_root),  
        "status": "ready" if overall_ready else "not_ready",  
        "gate_action": "pass" if overall_ready else "blocked",  
        "sequence_count": len(sequence_ids),  
        "ready_sequence_count": ready_sequence_count,  
        "sequence_ids": sequence_ids,  
        "required_streams": list(_SIM_REQUIRED_FILES),  
        "required_layout_family_count": _PAPER_MIN_SIM_LAYOUT_FAMILY_COUNT,  
        "supports_clean_train_val_test_family_split": supports_clean_train_val_test_family_split,  
        "layout_family_report": layout_family_report,  
        "sim_materialized_contract_report": sim_materialized_contract_report,  
        "reasons": reasons,  
        "sequences": sequence_reports,  
    }  


def _sim_root_has_only_placeholder(raw_root: Path) -> bool:
    """茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛氓露聣氓麓聼茅盲陆鹿茅聴聜盲陆鹿忙聜露茅聧聨忙麓陋氓麓聹氓篓聠忙聢聺莽聭聹莽聣聢氓赂聴茅聬聯忙露職芒聜鹿驴氓聞卤忙忙聬聛莽聧庐茅聫聧莽聟垄?SIM 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏氓庐聲氓聸搂氓鹿聢?茅聴?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳赂忙驴聞氓聞鲁茅聬聮禄忙聜聴茅聴聢忙露聶忙聠赂茅聳禄忙聧聡茅聫聦茂陆聣忙聜垄茅聬聯氓潞聺氓聳聤茅聴?茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙碌拢茅聺漏猫聙驴茅聫聣氓聽職忙聶聺茅聬聹氓忙聼隆茅聧聬茫聞楼芒聜卢莽聰赂氓麓聭?氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聧聸氓漏聝氓聫聫忙陇陇忙聮卤氓聻鹿莽潞戮氓忙聼聼茅聧聯脩聝盲禄潞茅聳鲁脩聦氓聣職莽聰炉忙聨聴茂陆聻氓漏碌氓忙聶虏茅聳赂忙露卤氓聣聶茅隆楼忙掳露忙垄潞莽聮聡氓聫聣氓聰聣忙陇陇莽聟陇茅聳鲁氓聯聞氓聲庐聲氓聫聣莽鹿聺茅聴聢忙露卤氓聞聫莽录聛莽聤鲁莽聣聽茅聫聦氓露聣氓麓聧聲氓聛聯茅聧娄莽陇聤茅聳赂茫聝聣職茅聬聯氓聠庐忙聼聸氓漏碌氓聴聴茅聳潞盲戮聡盲潞陋茅聫聦忙聸隆莽庐聸莽聙拢氓聛聯茅聫聧莽聟聲茅聳鹿猫戮戮茅聶聞茅聬聯忙聸聼莽鹿聸茅聨麓莽聝聡氓聟聙鹿忙聬聻氓聞卤茅聢鹿忙聢娄茅聧聫莽聟聨忙聻聽茅聴?茅聴?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    raw_root : Path
        SIM 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏氓庐聲氓聸搂氓鹿聢?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬氓庐聞忙聡聤氓陆聮氓麓露猫陇聹氓篓麓忙聽聼聲忙驴聻莽聜卢忙芦聠茅聳赂茫聜聟氓茅聫聦茫聜聡莽虏聶莽聮潞茫聞陇莽虏碌氓漏碌莽聜虏忙聡聬忙驴庐氓聻漏卤忙驴聽氓聸搂脦搂茅聳赂氓卤戮莽聼戮鹿忙聸垄忙垄禄忙碌拢茅盲赂虏莽录聛猫鹿虏聶茅聨篓潞忙聺聢芒聲聝莽聲露茅聴聛莽禄聵忙垄禄茅聧聵脩聡忙聜聻茅聧篓盲潞聹茅聳鹿氓聯聞忙拢聴忙碌聹忙聝搂茅聤聢氓聴聵莽漏驴莽录聛忙聫聮氓垄陆莽禄庐氓卢鹿聮茅聫聜氓聯聞盲潞戮茅聳驴忙驴聠莽掳录莽禄篓猫聜漏芒聜忙聦赂莽禄聣忙聮聙聰茅聧聬脦搂茅聳赂芒聞聝脙陋茅聫聠忙忙垄潞茅聧聺茫聝楼脨拢茅聳鹿莽聭掳忙麓颅茅陋聻氓聠漏氓炉录茅聫聣忙聡鲁茫聞漏忙聼聧茅聧聺氓聥聜卢氓聭聤莽茅隆聯莽聜漏忙聰陆茅聳禄忙篓驴氓茫聝漏忙聼聼茅聧聬忙聼楼氓聫聞茅聳碌氓聴聴氓聙驴氓庐聲忙楼聟忙聡聳茅聢潞氓聠戮氓聲聨茅聴聜盲陆赂忙鹿卤莽禄聲庐茅聬聬莽聰碌盲赂聧氓漏碌莽聤鲁莽茅聬聯忙聸垄忙聼聧莽聭聶氓聥聤陇氓篓?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    bool
        茅聴?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳赂忙驴聞氓聞鲁茅聬聮禄忙聜聴茅聴聢忙露聶忙聠赂茅聳禄忙聧聡茅聫聦茂陆聣氓搂聙茅聤聫芒聞聝忙炉聞茅聴聛忙麓禄氓聨陆茅聬聼茫聝漏芒聙娄莽聮潞茫聞娄氓麓聣茅聳禄忙掳卤茅聧職忙聜聴莽聭聶氓聥卢莽拢赂茅聳赂忙聴聙氓聻驴茅聬聸氓戮聞莽聙拢忙茅聴聛氓聭聤莽篓聭猫聣聦茅聳鹿茅聰聥忙陇陆氓氓漏職氓麓聮氓搂聵氓聧聻茅聴聛氓聯聞忙聡聬茅聧聥莽聠潞氓聢聮茅聧聭漏氓聛聯茅聰聺氓潞隆忙聻聼茅聳禄忙聮麓忙麓聼茅聫聦茂驴聽氓麓聮氓漏碌氓聸漏茫聙聤茅聳潞盲陆赂莽聣聯茅聨忙聠麓茅聧聲莽录聜氓聜職莽搂麓茅聰聲茫聝娄莽漏卤忙驴聽氓聸搂莽庐鹿氓篓麓茂陆聢氓篓聟氓聴聵盲驴聤茅聤聢氓驴聲氓聢掳茅聳赂莽聡聛氓聻露茅聨驴聽氓录麓茅聬聰猫聶鹿茅聴聛莽聭掳莽聯篓莽禄禄氓聜聻忙聡聹莽聭聶氓聥碌忙茅聳禄忙聞颅忙聝聦氓篓聢忙禄聢忙聼隆莽聛聻氓聣搂芒聤聲茅聳鹿忙拢聝忙驴庐芒聜聳鲁茫聝楼莽拢颅茅聰聸忙聮麓莽鹿聺氓篓聢氓聻聻聸茅聳驴忙聸聵氓聙聺茅聢楼茂鹿聙莽聲隆茅聨碌忙庐戮茅聴聛茅聺聸茅聴聞氓聥麓拢猫鹿聡忙露聶氓聙露茅聳禄忙聞陆聝莽录聧忙聽芦莽麓聮莽聮聡氓聫聣茅聳潞氓虏聥莽鹿聺氓庐聲氓楼赂茅聤聫莽聤芦忙聥漏碌莽聤碌莽禄卤忙聸垄氓录芦莽聮聡猫聦卢忙職娄茅聳禄忙聴聜芒聮聢茅聫聛氓聸卢忙聼拢茅聫聝氓聽聼忙聲卤茅聳禄氓潞庐氓聫聝氓莽禄聵忙聥聟茅聧聸氓漏聜忙職聢氓漏碌莽聜虏氓录露茅隆聽楼氓聼聟茅聧芦忙聸聼芒聮聮茅聳赂氓卤戮莽聭搂茅隆娄茂鹿聜忙聼聼莽潞掳氓聧聻盲潞掳茅聮聵忙驴聡茫聛聞茅聨掳氓聣聛莽聲卤茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽颅氓录禄茅聫聡莽聠潞莽茅聳赂氓聳聳忙戮?``True``茅聴?
    """
    if not raw_root.exists():
        return True
    entries = [path for path in raw_root.iterdir() if not path.name.startswith(".")]
    return not entries


def _materialize_sim_raw(raw_root: Path, output_root: Path) -> dict[str, Any]:
    """茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮盲禄聥忙聼陇忙驴聞卢芒聜聽芦芒聜掳卢茅聫聦茂陆聣氓录聧聦忙掳禄聵莽录聜盲陆虏氓聛聯氓猫聢碌氓聤聳茅聬聯忙卢聯忙聜聴氓篓聭忙卢聵莽聞陆莽录聜氓聜聸氓虏赂茅聫聦氓露聢茅聳赂忙聮麓忙掳篓茅聫聛氓聻戮氓庐聲氓聸篓忙庐戮氓漏碌莽聤虏茂鹿陇莽聙職氓聢聤茅聨戮莽聟聼茅聳禄忙聳驴忙聬隆拢氓聟录忙聟篓茅聰聺氓聭庐忙聬聬茅聳鲁猫搂聞氓聻驴茅聨卢聯忙聡聹茅聴聢忙露聶茅聰颅莽录聜氓聜職氓聙赂莽禄聣氓麓聭茅聨戮莽驴聽莽鹿聠茅聳禄忙聞碌盲潞聶莽禄卤忙聽鹿茅聰聺氓聭聲聻氓漏垄猫路潞氓芦掳茅聫聦茂陆聣氓鹿聵莽聮潞莽聝聵莽聛聢茅聴聛氓聯聞忙聹卢氓篓虏忙驴聢莽虏禄氓篓聭忙掳拢氓聛聯茅聰聺氓潞聺莽掳陆氓篓麓莽聤虏茫聜赂氓搂聤忙麓陋氓麓聧聲茅聺聸氓聣卤茅聴聛忙聬聻氓聤聦氓芒聙?SIM 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏氓庐聲氓聸搂氓鹿聢?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬氓庐聞忙聡聤氓陆聮氓麓露猫陇聹氓篓麓忙聽聼聲忙驴聻莽聜卢忙芦聠茅聳赂茫聜聟氓茅聫聦茫聜聡莽虏聶莽聮潞茫聞陇莽虏碌氓漏碌莽聜虏忙聡聬忙驴庐氓聻漏卤忙驴聽氓聸搂脦搂茅聳赂氓卤戮莽聼戮鹿忙聸垄忙垄禄忙碌拢茅盲赂虏莽录聛猫鹿虏聶茅聨篓潞忙聺聢芒聲聝莽聲露茅聴聛莽禄聵忙垄禄茅聧聵脩聡忙聜聻茅聧篓盲潞聹茅聳鹿莽聝聵氓聻碌茅忙聼聸莽聙拢忙聢娄茫聙?茅聴聜氓聜職氓聙路氓篓麓氓聸搂莽聲路茅聬垄莽莽陋聴茅聳鹿茅聜娄氓聳聳茅聧篓忙驴聥氓鹿聳氓篓拢莽聛驴聻氓聭搂碌忙路聡氓漏聡氓娄露茅聧聦忙掳茅聴?茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聟忙職娄茅聳赂氓聛聟茅聧聶氓驴聲芒聜卢猫搂聞忙麓聳氓篓虏茂鹿聜忙碌聽氓莽聟聫茅聳赂莽鹿聝忙戮露氓聥聛聞茅隆聬茂陆聠莽陇聥茅聳潞氓虏聥莽鹿聺氓庐聲氓聽聲氓聼隆忙碌拢氓聸楼氓聻陆茅聫聦忙露垄氓娄赂茅聰聰氓聣搂莽聳聞茅聴聛氓聯聞忙聹篓虏忙篓录忙聧垄氓篓聭茫聢隆氓聙路忙陇陇忙聨聧猫聝露猫陇掳氓聭麓忙垄禄忙碌拢氓聭聤氓聲聻茅聳赂茫聞篓茅聨卤茫聞漏忙聜陆茅聧篓氓潞聤氓漏碌莽聤禄忙陆搂茅隆拢茅聭陆茫聜聡莽拢陆氓篓麓茂陆聣忙陆搂忙驴庐氓聜職脙鹿氓氓聴聵莽隆戮茅聳?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    raw_root : Path
        SIM 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏氓庐聲氓聸搂氓鹿聢?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬氓庐聞忙聡聤氓陆聮氓麓露猫陇聹氓篓麓忙聽聼聲忙驴聻莽聜卢忙芦聠茅聳赂茫聜聟氓茅聫聦茫聜聡莽虏聶莽聮潞茫聞陇莽虏碌氓漏碌莽聜虏忙聡聬忙驴庐氓聻漏卤忙驴聽氓聸搂脦搂茅聳赂氓卤戮莽聼戮鹿忙聸垄忙垄禄忙碌拢茅盲赂虏莽录聛猫鹿虏聶茅聨篓潞忙聺聢芒聲聝莽聲露茅聴聛莽禄聵忙垄禄茅聧聵脩聡忙聜聻茅聧篓盲潞聹茅聳鹿莽聝聵氓聻碌茅芒聲聝莽麓聮?茅聴聜氓聜職氓聙赂茅聧聤莽聝陆忙聡聴茅聧聯盲陆赂氓聻聺忙陇陇忙聽聛聻茅聧聥氓茫聛聞茅聨戮猫路潞脨楼莽录聛芒聜驴聽氓聜聵芒聜卢忙聨聭茅聰聺氓聭忙驴?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳赂忙驴聞氓聞鲁茅聬聮禄忙聜聴茅聴聢忙露聶忙聠赂茅聳禄忙聧聡茅聨鲁掳芒聳聨猫鹿聡忙聯聝盲禄鹿莽录?
    output_root : Path
        茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聧茅聳禄忙聞陆聶忙碌聹猫戮篓莽鹿聺茅聬垄茅聺聸脨垄忙驴聻氓卤聜氓聙聴莽禄庐莽禄聵氓莽聙聸忙氓聨碌茅聴聜氓聜職氓聙赂茅脣聡茅聰聲氓聜聸莽聟聲忙驴庐忙篓禄氓聣職氓篓录忙聞掳忙聼聲茅聧楼脩聝忙楼聽莽聜麓茂鹿垄氓庐聲茂驴聽忙聜聶茅聧聫氓聭聤氓聛聟茅聴聜氓聜聶莽茅聨虏茂驴聽氓麓聺猫鹿聡忙露聵莽拢禄茅聳鹿氓聣搂莽虏聢路忙聨聭茫聙聮氓茂陆聟盲潞戮氓漏碌莽聜露氓拢聢氓庐聲莽聰赂氓聼聙茅隆聮氓聥卢忙露搂茅聳禄氓聙赂茅聬拢莽聝陆氓茅聤聢氓聴聴盲禄录茅聬聨氓聟聣氓聛聯茅聳鲁脩聦氓聣聼茅聨麓聡莽麓聮氓娄炉氓聹庐職茅聴聛茅聺聸莽聣聠茅聨鲁氓潞篓氓录鲁茅聬聬莽聝聵莽聟聲茅聳禄忙篓潞氓聲驴忙驴露聣忙聼隆氓庐聙氓卢聢掳茅聯禄忙聽颅忙聼聧猫茅聧聯莽聠录氓鹿聝猫莽禄禄忙驴聥氓麓聼茅莽聤碌氓聨掳茅聴聛猫聧陇氓搂碌忙碌聹盲禄聥氓麓聺忙聬麓茂陆聟氓漏碌氓聜聹莽禄戮脩聡忙聼聸茅隆聬茂陆聡芒聵聣氓漏碌脗陇忙鲁聲莽禄卤忙聨聴忙聜漏莽禄聸莽芒聜聺聴忙聼?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜麓莽聣聽茅隆垄忙聸垄忙聲聝茅聢搂氓茅聧娄莽拢录茅聬聨茂陆聨氓聛聯莽禄卤芒聲聜莽鹿聸氓庐赂莽掳录茅聳潞氓聭庐莽聟隆茅聫聦忙露聵芒聵聣茅聧聶忙氓聡鹿茅聴聛猫炉虏茅茅隆鲁掳氓篓虏忙驴聻忙聢聻忙掳篓茅聬拢茅鹿聝忙垄潞茅聳芦莽聜虏猫聥炉忙戮搂氓聫聣莽鹿聸茅聧聸莽陇聥忙莽聜虏莽聣聯忙驴隆忙聬聛氓娄路茅猫聴聣莽录聯茅聴聜盲陆潞莽隆露茅聧聯忙驴聥茫聙聝茅聧聸猫戮戮莽露聳茅聳鲁氓聯聞忙聡聫茅聢路忙聢娄忙聼聸氓漏碌氓聴聴茅聳赂茂陆聠忙陇陆茅聫聦莽聠录茅聬聯氓潞聵氓聛聯茅聧篓氓職聲茅聬聽氓聸篓莽楼麓氓娄虏氓聽楼忙聼聲猫鹿聡忙聸聻脨楼忙陇陇忙聳驴茂鹿猫聶鹿忙聠麓茅聧聲氓搂垄氓娄聻茫聜聠忙麓娄茅聧聶氓聠篓莽聲路氓篓聠忙聮麓忙聶赂茅聳禄忙篓禄忙聻录茅聨?``audits/`` 茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼忙聲聥莽聭聹氓驴聸氓聼聙茅隆聮盲陆潞芒聳聯茅聳潞氓聭鹿聞茅聧聮忙聨聭芒聳聨茅聨戮氓麓聡氓漏碌脗掳氓聙聬茅聧聥忙聮露莽麓聮茅聢搂氓戮聞莽聙拢芒聲聝莽庐職茅聴聛茅聺聸莽聣聠茅聧聤氓聭聤莽娄聧忙楼拢氓聠漏莽聟聼茅聳鹿忙聝搂氓麓聢搂莽鹿聢氓氓漏職忙聜垄忙陇聥氓潞垄忙庐聺茅聴聛莽聭掳氓聵虏茅聬聽聺茅聭赂莽聰碌莽庐戮茅聬聨忙露聶茅聬颅氓漏聞莽麓聯氓庐楼氓聮聝猫潞聙碌茅聫聛忙聮禄忙聜漏茅聭录芒聜卢莽聜潞忙垄潞茅聴聢忙露卤莽聟漏碌芒聙鲁忙麓陇茅聢禄忙聮禄氓娄露忙戮露氓卢搂忙聥潞茅聴聛莽聟聨茅聧聤忙聸垄氓录赂茅聨麓茂赂陆莽聟聶茅聳禄莽聠潞氓垄聴茅聧聽忙篓录芒聜卢?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    dict[str, Any]
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓聙驴茅聳驴忙聸聻氓聙鹿茅聧聨忙聢聽忙聠赂茅聬聴氓聽聺莽卢聮茅聳赂茫聞楼氓聙赂茅聢鹿忙聢娄忙聜漏莽聙鹿莽聤虏莽录聜盲陆鹿氓聤聳茅隆篓氓漏聡氓录禄茅聬聰氓聟录氓聛聥茅聳赂氓聳聯茅聧聭茂驴聽忙垄潞茅聨录莽聜虏氓娄录茅聳赂氓漏聜莽露聤茅陋聻氓聽楼氓娄赂茅聤聣氓潞拢氓聳聳氓庐聲氓陆聮茅聨聝莽录職茅聴聜氓聜聶莽聞娄茅聨庐氓聽聲氓麓聬茅聫聡莽聛禄忙麓聲聝茅聧聦忙掳卤莽聳聞茅聴聛茅聺聸茅聧聨氓聯聞茅聤聤芒聲聜莽聟聲茅聳鲁芒聲聛氓聳職氓篓聢忙聴聜莽麓聮茅聬聵猫聶芦茫聛聺忙驴庐氓漏聝忙陇陆氓庐聲莽聝陆茅聬聬忙聫聮莽庐拢氓漏碌莽聜虏莽聯篓莽禄虏聞忙聜楼茅聬聯氓潞職聲氓漏碌莽聤鲁莽垄聧茅聫聟忙聫聮莽聮潞氓楼陆茂驴聽氓聼隆茅聧聸氓聨驴聽茫聢拢忙鲁聸茅聬聴氓聴聸氓麓聺茅聰聲氓聜聸莽聟聸莽聙鹿芒聜卢莽聙拢脨聹茅聬聨忙庐驴氓聳聲莽禄卢聯氓鹿聝忙碌聽氓卢聛篓茅聳潞盲陆赂忙聹禄庐忙聮禄氓鹿聝氓娄陇氓聠漏氓聳聠茅聳赂忙聸篓氓聣聸茅隆娄茫聞漏忙垄潞茅聭陆茫聜聟忙職卤茅聳潞氓聭庐莽聸聨庐茅鹿聝忙聜聹茅聮聵氓茅聴聛莽禄聵氓聻碌氓娄芦忙卢聯氓麓聯茅聬垄氓聭聤莽聟聸氓猫路潞茂鹿娄忙戮搂忙聞掳忙聼隆茅聧聸莽庐聻莽聙鹿忙聸聼氓聤聶茅陋聻氓潞篓忙聡聻茅聤聣茫聢隆氓芦陆氓漏碌莽聜露忙聫陋莽禄虏盲禄聥氓鹿聣茅聰聼莽聤虏莽聳庐茅聳禄忙聞庐氓聞陇茅聬聯忙聸垄忙聼聲忙驴聻氓聻庐氓娄陆莽禄戮氓垄聨莽禄卤忙聨聯忙陆聫茅聤聤茂鹿聙莽垄聧氓茫聜聨氓聛聟莽禄禄氓聽楼莽聧聬麓茫聞娄氓職聧茅聳碌氓氓聻潞莽聙職氓楼赂忙垄禄忙碌拢氓聭聤氓聲聻莽录聛猫氓聙禄茅聢搂莽垄聧氓忙漏聙忙聜戮氓庐聞忙聣鲁忙聰陆茅聳赂莽聥聙氓聳聴氓娄芦氓聠篓莽聲路氓搂聴芒聜卢忙驴隆忙聬聛氓娄路猫茅聧聽氓漏聡忙垄禄茅聧聦忙卢聭莽陋聰忙驴聻盲陆鲁氓聴聴茅聴聜盲陆赂忙聬芦茅聨鲁氓驴聸忙聝聞茅忙掳卤忙職娄忙驴聻氓聴聵忙聦禄茅聧聥忙聞聜卢猫搂聞忙麓聳氓篓虏茫聞漏忙聝聛茅聧芦茫聝娄氓聛隆忙驴聽氓漏聜氓聲掳猫聜聳茅聴聛氓聟录忙陇陆忙碌聹氓聽聲氓录禄茅聧隆忙氓聬聥茅聳赂忙露卤氓聻陆莽禄?

    茅聴聜氓聜職氓聙路氓篓麓氓聸茅聢搂莽篓聢茅聫聟莽聤禄茫聙聛氓篓聠忙聮鲁氓聠聙忙陇陇忙聮露氓聛聼茅聬聸茫聝漏忙垄潞茅聧聺氓聥颅芒聳聣茅聳赂忙篓驴氓聛聬忙戮鹿忙聸隆茫聛聬茅聨潞忙聮露氓聨碌茅聴聜氓聜職氓聙赂茅氓麓聺氓氓陇聥莽聟聲?
    ------
    RuntimeError
        茅聴聜氓聜職氓聙路氓篓麓氓聸搂莽聲路茅聬垄莽莽陋聴茅聳鹿茅聜娄氓聳聳茅聧篓忙驴聥氓鹿聳氓篓拢莽聛驴聻氓聭搂碌忙路聡氓漏聡氓娄露茅聧聸忙芦拢莽录聜盲陆潞氓搂鲁茅聧聴忙聤陆氓录禄茅聬聰氓聟录芒聮聮茅聬聨莽聰碌忙驴隆盲禄聥忙垄潞莽禄聥忙卢戮氓聞聧茅聳赂氓漏聝莽聣聹茅聨鹿茫聢聽芒聵聣茅聤聫莽聤虏猫聙驴氓漏碌芒聵聠氓聻碌茅隆聲脩聡忙氓庐楼氓芒聮聭茅聳鹿猫聜漏氓聛聸茅聢搂忙麓聳莽聟陇茅聳驴忙聸聻氓聙赂莽禄聽莽聜潞忙聼聲忙驴聻氓聻庐氓聤麓忙驴聻莽卢潞氓聤聭茅聫聦忙露垄氓聼聞茅聧聬莽聜聲露莽录聜盲陆潞氓搂碌氓猫聴聣氓篓虏忙聧露忙陇陇忙聮露氓聛聸茅陋聻氓莽鹿聸莽聙聸忙莽聼聨忙驴聻氓陇聥莽聸炉茅隆垄忙掳露忙聲聬忙戮露氓漏聡氓聰露茅聴聛氓聯聞忙麓篓茅聧聥莽聠录氓麓垄茅聨录盲陆鹿芒聮聭莽录聛氓卢氓漏聙忙聣麓聮莽聙拢氓麓聦莽聙鹿忙聸陋莽露聛氓庐聲氓聧聻莽录聛忙篓禄忙聺赂茅聴聜盲陆潞莽虏聧聰忙聽聛聬茅聧聬忙拢炉莽聭聹忙聴聢氓录禄茅聰聺氓聭聜忙聴聢氓麓聵茅聤聤忙聞漏氓聻陆茅聨陆氓聽聲忙聜聶茅聬聭忙篓录氓聨卤茅聴聛茅聙聸氓聧鲁氓篓录脩聡氓录赂茅聬聰氓聟录莽聟聲氓漏碌莽聤虏氓茅聧聵忙聮庐忙聼隆氓庐聙氓卢搂庐忙楼聽莽聜虏茅聨禄聢氓麓卤氓篓聠氓驴聯脨篓茅聴聜氓聜聶莽路職氓篓录脩聡忙聜搂茅聧聬莽聳戮茅聳禄忙篓潞忙篓聤氓忙聽麓聯忙碌拢猫莽聞陆茅聳禄莽聠路莽露聤茅聫聦忙露垄氓娄路茅聨麓忙驴聠芒聜卢氓漏職忙聜楼茅聳露聠芒聮聮氓篓麓茂陆聞猫茅聬聮莽聜卢茫聛聞茅聨麓忙驴聡忙莽聜麓氓聻驴氓庐聲氓聽聲茅聢搂忙聥聭茅聬聰氓聯楼莽庐戮茅聳鹿氓颅聵莽聭楼茅聬聫芒聲聝莽麓聮茅聬聶氓聭搂庐茅聳潞氓卤禄氓聤聦茅聢鹿忙聢娄氓麓卤氓搂聴氓聴聵茂录聠茅聴聜盲陆赂忙聬芦茅聬莽麓聻忙碌拢氓聣虏氓職聲氓篓麓莽聤虏忙聝聼茅聬聼忙禄聝莽鹿聵茅聨聞聝氓戮聞忙驴聻茫聢隆忙茅聨聲聟忙拢聝氓篓聭忙楼聝忙聧聲忙驴隆莽聜陋氓聙聳茅聧篓莽聰赂脣聡茅莽聝聠忙聲聥茅聳驴忙驴聠忙拢聸茅隆漏莽聝陆忙聜聴茅聰聺氓聭炉盲禄聸茅聳潞氓聭聻陆氓搂聤猫聶鹿莽虏聶茅聨赂氓潞垄忙聥卤莽录聜盲戮聡氓聳聳茅聳掳茂陆聟莽聲路茅聨掳忙聺驴莽庐聳录忙聛聮氓聞庐茅聨路莽禄聵莽鹿聸忙聺聢氓聣搂氓聢掳茅聳鹿猫莽鹿聺氓庐聲忙聜聶茅聬聭忙篓录氓聨卤茅聴聛茅聺聸脛聸茅聧聬茫聞陇莽虏驴隆莽聜聙聬茅聫聟忙露垄氓娄露茅聧聸氓漏聛莽鹿聠忙驴庐忙驴聠莽聼聢氓娄虏氓聣搂莽麓聯忙碌拢氓聯聞脨拢茅聳潞氓卤禄氓聤聭茅聢楼茂鹿聜氓麓赂氓娄陇盲陆赂忙庐聺茅聴聜氓聜職莽聣聤莽禄聥忙聢娄茂陆聻氓庐聙氓聥聮聭?茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙碌拢猫聶鹿莽陇聤氓漏碌莽聤虏忙碌聜茅聫聛盲陆鹿盲驴聤茅聤聢氓聭聶潞茅聳?JSON 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜麓莽聣聽茅隆垄忙聸垄忙聲聝茅聢搂忙聜驴茅隆聲芒聜卢茅聫聦莽聠录氓鹿聠茅聬聴氓聽茅聴聛氓聯聞氓芦篓茅聧聨莽聰赂氓录禄茅聢楼忙聞聛氓聬聥茅聳赂忙聞漏氓聤聦茅隆忙垄潞茅聧聸氓漏聞忙聡聝莽录聛莽禄聵茂鹿垄茅聨庐茂驴聽茅聰聲芒聜聬聬氓漏聡忙聼聲忙驴聻脩聙氓聛聯忙驴隆忙聫聮莽聣聫莽禄卤忙聮麓忙聥聟茅聨禄忙聨聧氓茅聴聜盲陆赂忙聠隆氓篓虏茂鹿聜氓麓鹿茅聳露聠氓聛聜忙驴聻氓聴聵氓聻聼茅聧聥忙聮禄氓麓路茅脩聟氓聤赂氓茫聜聨氓聨录茅聧垄忙聼楼忙聜?
    """
    sequence_profile = "sim_e9_only_compact" if _ACTIVE_PAPER_SCOPE == "sim_e9_only" else "default"
    completed = subprocess.run(  
        [  
            sys.executable,  
            str(ROOT / "scripts" / "02_generate_sim_raw.py"),  
            "--sequence-profile",
            sequence_profile,
            "--output-root",  
            str(raw_root),  
        ],  
        cwd=str(ROOT),  
        capture_output=True,  
        text=True,  
        check=False,  
    )  
    try:  
        stdout_payload = _loads_json_from_stdout(completed.stdout, source_name="02_generate_sim_raw.py")
    except RuntimeError:
        raise
    _write_json(output_root / "audits" / "sim_materialization.json", stdout_payload)  
    if completed.returncode != 0:  
        raise RuntimeError(f"02_generate_sim_raw.py failed: {stdout_payload}")  
    return stdout_payload  


def _write_final_report(output_root: Path, payload: dict[str, Any]) -> str:
    """茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聫茅聢搂莽聞娄莽禄禄忙驴聥氓麓聮氓娄陇盲陆潞盲潞聶忙驴聽莽聰碌氓聛聻茅聧篓氓麓鹿氓篓虏莽聳戮忙驴聽茅聺聸茅聬聯氓驴聯莽聮潞猫聶鹿氓垄聲茅聳赂忙聴聙茂鹿垄茅聫聦氓露聢茅聳赂忙聮麓忙掳篓茅聫聛氓聻戮氓庐聲氓聸篓忙庐戮氓漏碌莽聤虏茂鹿陇莽聙職氓聢聤茅聨戮莽聟拢茅聼聸篓忙麓聵茅莽聠潞莽麓聮忙聺聢茫聞搂氓聫聙忙驴漏聞莽虏聨路氓聭卢聼茅隆聳忙禄聝氓搂录茅聴聜盲陆赂忙聬芦氓娄驴聡莽娄聧茅聧芦忙聸聻猫聯聼忙驴聻忙聢聻氓聼聝茅聧聥忙聮禄忙聲聬忙聬麓芒聙虏莽掳录茅聳禄氓驴聯莽鹿聮茅聫聛氓聯楼忙茅聨潞忙聮麓莽露聡茅聳碌氓篓驴氓聞鲁茅聬聼茫聞漏忙聼搂忙碌录忙卢聭莽搂庐茅聳潞氓虏聥莽露聛茅聧漏茂驴聽氓麓聮氓搂聵猫戮漏忙聡聳茅聴聜盲陆赂忙鹿卤茅聨鲁忙聮鲁脙潞茅隆聯莽聝聠氓職聲茅聬聽氓聸篓莽楼麓氓娄虏氓聽聼盲驴聤茅莽聜麓忙聲卤忙陇陇莽搂麓茅聢鹿忙聢娄莽禄聸氓莽聲露茅聴聛莽禄聵氓娄芦忙卢聭芒聜卢茅聺聸茫聜赂茅陋聻氓聸卢忙聜搂茅聧芦忙掳茅聴聜盲陆赂忙聠隆莽禄潞氓聺聴氓麓聲茅聳露聟氓聡聽茅聳鹿猫戮戮茅聶聞茅聬聯忙卢聯忙聼拢茅聬聰氓聯聞茅聳鹿氓聟录氓聤聶茅聫聦茂陆聣忙聲聬茅聧聸脨漏氓娄聻茫聜聨氓聛聟莽禄禄氓聥鹿聭茅聧聲忙聝聻莽聭聹莽聣聢氓赂聻氓庐聲忙禄聢忙垄禄茅聧聦忙卢聭忙驴聽芒聜卢茅聳露聛芒聜鹿聜氓麓录氓忙驴聠莽露聤茅聨禄聢忙聦聤忙戮鹿芒聰聠拢忙聰陆茅聳赂氓卤戮莽茅聧聼莽聜潞忙聼拢茅聨戮猫路潞忙聻聸忙楼聽莽聜麓莽聣聲猫聫聬忙陇陇忙聨聠氓聛聯茅聳鲁脩聦氓聣職茅隆篓氓漏聜莽聲路茅聬聴氓聽垄莽炉聙茅聳赂氓卤禄氓聙聲茅聨庐忙聢聻莽鹿聺茅聤聫茂陆聠莽隆戮茅聳驴忙聸聵氓聙聺氓庐聲莽聰赂氓鹿聥茅聰聰猫搂拢忙聥潞茅聴聛氓聭聤莽篓聭茅聰聲茫聜聠忙聟篓茅聢搂忙垄潞茅聧聺氓聥聶潞莽录聛猫莽聣聯氓氓漏職忙聲聬忙戮露氓盲潞鹿茅聴聛莽禄聵氓聻露茅聰聲芒聲聜莽娄聮茅聢潞氓聭聮聭莽录聛氓忙鲁聲茅聫聞氓聜聻忙聼聸忙驴聽氓聠陋莽庐聯茅聳禄茂陆聡氓聟聵茅陋聻氓聸卢氓录露莽聮潞氓聰麓茅聴聜盲陆赂氓聬聥忙碌聹盲禄聥氓麓聲茅聨露猫聴聣氓聜篓?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    output_root : Path
        茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聧茅聳禄忙聞陆聶忙碌聹猫戮篓莽鹿聺茅聬垄茅聺聸脨垄忙驴聻氓卤聜氓聙聴莽禄庐莽禄聵氓莽聙聸忙氓聨碌茅聴聜氓聜職氓聙赂茅脣聡茅聰聲氓聜聸莽聟聲忙驴庐忙篓禄氓聣職氓篓录忙聞掳忙聼聲茅聧楼脩聝忙楼聽莽聜麓茂鹿垄氓庐聲茂驴聽忙聜聶茅聧聫氓聭聤氓聛聟茅聴聜氓聜聶莽茅聨虏茂驴聽氓麓聺猫鹿聡忙露聵莽拢禄茅聳鹿氓聣搂莽虏聢路忙聨聭茫聙聮氓茂陆聟盲潞戮氓漏碌莽聜露氓拢聢氓庐聲莽聰赂氓聼聙茅隆聮氓聥卢忙露搂茅聳禄氓聙赂茅聬拢莽聝陆氓茅聤聢氓聴聴盲禄录茅聬聨氓聟聣氓聛聯茅聳鲁脩聦氓聣聼茅聨麓聡莽麓聮氓娄炉氓聹庐職茅聴聛茅聺聸莽聣聠茅聨鲁氓潞篓氓录鲁茅聬聬莽聝聵莽聟聲茅聳禄忙篓潞氓聲驴忙驴露聣忙聼隆氓庐聙氓卢聢掳茅聯禄忙聽颅忙聼聧猫茅聧聯莽聠录氓鹿聝猫莽禄禄忙驴聥氓麓聼茅茅聴聜盲陆赂莽露聤氓娄芦氓聺聴忙聜聻茅聰聲氓聜聸氓聛?
    payload : dict[str, Any]
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聫茅聢搂莽聞娄莽禄禄忙驴聥氓麓聼茅隆聯莽聥聙忙搂卤氓漏碌莽聜麓忙陆聶茅聧職氓氓篓聠忙聢聺茅聳赂脩聞茅聢路忙聢聻茫聛聞茅聨潞氓聴聴盲潞戮莽录聜盲陆赂茅氓聢聮忙驴聽忙掳聛聞茅聳赂氓驴職氓聛聬茅聧聶氓聴聸忙垄潞莽录聛忙篓禄莽聟楼茅聳鹿莽垄卤氓聛聞茅聬隆莽麓聯茅聧聦忙卢聮茅聳赂氓忙聧聡茅聫聦忙露垄氓鹿聡茅聴聢忙露聶莽聛聧茅聴聛?茅聴?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    str
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聫茅聢搂莽聞娄莽禄禄忙驴聥氓麓聼茅隆聯莽聥聙忙搂卤氓漏碌莽聜麓忙陆聶茅聧職氓氓篓聠忙聢聺茅聳赂脩聞茅聢路忙聢聻茫聛聞茅聨潞氓聴聴盲潞戮莽录聜盲陆赂茅氓聢聮忙驴聽忙掳聛聞茅聳赂氓驴職氓聛聬茅聧聶氓聴聸忙垄潞莽录聛忙篓禄莽聟楼茅聳鹿莽垄卤氓聛聞茅聬隆莽鹿聺莽潞掳忙篓禄茅聳赂氓忙聡聵茅聫聛氓聠茅聤聢氓聭掳忙聭聲氓漏碌莽聜麓氓聻炉茅聧篓莽聭掳忙聲驴隆莽聜聙聳茅聬聴忙禄聢氓麓聭茅聬聬盲戮炉忙聲聛莽聙鹿茫聝楼忙聥聥氓篓聯氓漏職忙聼聲茅聧驴聡氓漏碌氓聲掳莽禄卤忙聨聴茅聬拢氓卤戮氓聶掳氓娄陇?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓麓茂陆聢氓篓聟莽聤禄氓鹿聳氓篓拢氓莽禄戮忙聝搂茅聫聦茂陆聣氓录聧聦忙掳禄聵茅聴聛莽禄聵氓赂聻茅聫聛氓聯楼忙茅聨潞忙聳潞氓聛聯茅聰聺氓潞隆盲潞聹忙驴聻忙聬颅氓聶拢茅聫聦?氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄茅聳禄忙聞聬芦茅聧篓氓聜聻忙聼聸茅茂陆聠莽莽录聛莽聤卤氓聧鲁茅聫聦莽聠录忙垄禄莽聙碌莽氓娄陆茅聴聛莽篓驴氓茅聧篓莽聰赂忙鹿聛茅聴聛莽篓驴莽鹿聬茅聧職氓莽麓聻茅聨麓茂赂陆莽聟聼?
    """
    return _write_json(output_root / "paper_run_report.json", payload)


def _lock_seed(seed: int) -> dict[str, Any]:
    """茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聫茅聢搂莽聞娄莽禄禄忙驴聥氓麓聮氓娄陇盲陆潞盲潞聶忙驴聽莽聰碌氓聛聻茅聧篓氓麓鹿氓篓虏莽聳戮忙驴聽茅聺聸茅聬聯氓驴聸忙聼聸茅茂陆聡莽忙驴隆忙聫聮莽露聤茅聫聦忙露職芒聜卢茂陆聨盲潞聹茅隆聫忙聼隆莽聛聻莽聲聦脨楼忙陇陇茂鹿聙莽聯篓忙路聡氓漏聡茅茫聢禄颅茅聴聛茅聺聸忙麓陇莽聙職盲录麓莽聧娄炉氓掳聳氓聛聯茅聬垄莽聰碌茫聙聤忙驴隆忙聫聮莽聣聤莽禄禄忙露職芒聜露聶茅聬聨聰莽麓聮茅聬聵氓麓聡氓垄陋忙陇陇莽鹿聭莽禄禄忙驴聠茅隆聮氓聜娄莽麓虏忙驴聽忙庐驴氓聳聴茅聰聲芒聲聟氓麓垄茅聬聯氓潞聵忙聻聧茅聳驴忙驴聠忙麓篓莽潞戮猫聴聣脙鹿茅聰聺氓聽聼茅聬垄莽聸聫聦忙露垄氓娄赂猫陇聨茅聫聠忙聼聸茅聢潞氓聠篓盲禄卤莽聙鹿忙聸聼茂鹿垄茅隆垄忙卢聯忙聡聳茅聢潞盲陆赂脨聲莽录聜氓聜職氓聙路氓篓麓氓聸篓茫聛聰茅聧聺茫聝娄忙職聹忙驴隆茫聜聜氓聛聯忙陇陇忙聸聵氓聞碌氓庐聲莽聠录茅聧聦忙禄聟忙茅聴聜盲陆赂忙聠隆氓篓虏忙漏聙氓麓鹿茅莽聜虏氓聻聜氓篓麓莽聤虏茅聮聝莽聠录忙聼篓氓漏碌氓聴漏忙搂赂茅聳禄忙聨聯忙陇陆茅聫聦忙露垄氓鹿聡茅聧聫氓聯楼氓聡鹿茅聴聛茅聺漏莽碌录茅聧聴忙聤陆氓鹿聝氓娄陇氓聠漏氓聳聠茅聳赂忙聸篓氓聣聸茅隆娄茫聞漏忙垄潞茅聨赂氓潞隆氓聯潞茅聳潞氓卤陆莽搂露茅聨路莽聤禄氓麓拢忙戮露氓漏聝忙聲陋莽录聜氓聜聶猫聝露忙驴庐莽聰赂忙聝聞茅隆聳莽聜碌氓碌聛忙驴庐忙陇禄氓聛聯茅聳鹿莽聟聨莽露聤茅隆垄忙聸垄氓娄路猫陇聧猫陇掳氓聜聻忙垄禄茅聧聦忙掳卢芒聜聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聧茅聢搂莽庐聯氓庐聲氓楼赂氓搂聙茅聢楼氓聠虏莽掳楼茅聴聜盲陆赂氓茅聧聥盲戮聙氓麓聲忙聺聢芒聲聝莽聼聠氓漏垄猫路潞氓陇聡茅聧聥忙聮禄氓麓聴忙戮露氓漏聛氓拢聲茅聴聜盲陆赂忙聠隆氓篓虏茂鹿聜氓麓聹氓篓聠忙聮鲁莽茅聳赂忙聞卢忙篓聤氓篓聯氓漏職忙聼聲茅聧驴聡莽录聛忙氓赂隆茅聨庐忙楼聙茅聬聯氓潞隆莽卤聧茅聬聨猫搂聞忙麓聵茅聧篓氓聺聴莽聧聧楼氓聛聥茅聳赂氓聻鹿茅陋聻忙漏聙忙垄禄忙碌拢氓聭聤氓聲聻莽录聛氓聯聞忙陆隆氓庐聲忙聸垄忙氓篓聭忙掳颅忙庐戮茅聴聛莽隆聟忙聫陋莽禄聽忙聢聽莽虏禄忙驴聽忙掳聙碌茅聴聜氓聽聼莽篓聮茅聨虏忙聬聛忙聼聼茅聯聫莽庐聳莽录聛莽禄聵莽鹿聢茅聨禄聥忙拢聝氓篓麓猫潞虏氓聻陆茅聨茅聬聯氓潞垄莽聛聯莽录聜?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙聮虏莽茅聧篓忙驴聡茫聛聞茅聫聧茫聞娄莽職聭茅聳潞氓聴聴忙拢聸莽禄卤忙聨聰芒聜卢茂陆聨氓聨陆莽潞聽聲忙聼隆茅聧隆莽聲碌茅聳潞氓卤戮忙麓聵氓炉掳氓聥芦芒聵聣茅聤聫芒聵聠芒聜卢氓露聣忙垄潞莽禄聥忙聦聨忙聧拢茅聳赂莽聤鲁莽聣聽氓炉氓娄赂茅聤聣茫聢隆忙聳聙茅聴聛莽鲁聲氓聣聺茅聰職忙陇陇茂赂鹿忙聶聽氓忙麓麓芦茅聧聲忙茅聴聛氓聭聤茅聧聼氓聺聴茂陆聻猫鹿聡忙聸聽莽拢录忙驴隆茅忙聮禄忙垄潞茅聭陆茫聜聟忙聻聭氓莽聭掳忙莽禄漏聜莽聲隆茅聨碌猫聶聳聙茅聴聛莽禄聵氓聤聶茅隆陇氓聜聻氓麓碌莽聙拢莽拢录茅聳禄忙忙芦聲氓庐聲氓卤戮茅聬聬忙聫聮忙職垄忙陇陇茂鹿聙莽露聤茅聫聦茂录聞盲禄娄茅聭潞茫聝庐忙拢聻氓茫聜聠忙聦赂茅聤聢莽莽聲路茅聤聤芒聲聛莽拢录忙驴隆忙聬芦忙驴潞篓忙垄?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳鹿茅聴麓莽莽潞戮莽聭掳莽聙拢忙聧拢茅聳禄忙茅聤聠忙聨聭茅聰聺氓聭卢氓拢聲茅聳禄氓潞陇氓篓虏茂鹿聜氓麓鹿茅聧聸聼氓聛聵忙陇陇忙聴聡忙鲁聲茅聬陇芒聜卢茅聴聛氓聯聞氓聙鹿茅忙卢聯氓麓碌茅聤聢氓芒聮聭茅聳赂忙驴聠氓氓娄虏氓陆聮忙聼聸茅聤聤茫聝搂莽搂潞忙陇陇茫聢隆莽聭漏茅陋聻氓聸聭潞茫聝娄忙聺赂茅聴聜盲陆潞莽虏隆颅氓聽楼氓氓聠戮氓聲聟茅聳碌氓忙聶聹茅聬聯忙卢聯忙聼聸茅聨掳氓聫聙茅聳赂忙卢聫氓芦聢搂氓篓虏忙漏聙氓麓鹿茅聬垄氓聣虏忙聲聡茅聳赂氓驴聲氓聨搂莽禄露莽聜潞忙聜聴茅聰聸氓聧聻氓芦莽聭掳芒聰聭茅聰聸氓聥芦盲潞录茅聳赂氓漏聝莽聣聲茅隆驴聸氓麓路茅忙露聵盲潞戮莽录聛忙莽篓聭氓篓虏茂鹿聜氓麓聭茅聢漏氓聠陋莽炉聙氓漏碌莽聤虏氓聙赂茅莽麓聮茅聨炉忙聺驴莽職潞茅聳鲁脩聦氓聣職茅隆聰忙聽颅氓麓掳氓篓聭忙篓路忙聻聳忙驴聻忙聢聶氓聼聞茅聫聛氓卢聼聲忙戮露氓聴聵忙芦聟莽录聛芒聜聙拢盲潞聹茅聳鹿忙聺驴氓垄聨茅聬拢茅聛聧忙聼拢茅聨掳氓聤聼茅聳鹿氓聫聣莽聯聲莽禄聽忙露聵芒聵聣氓篓聠氓驴聨莽聴聴聜盲戮聙芒聜聜虏猫聥戮搂氓颅聵忙聟篓?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    seed : int
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙聮虏莽聞娄茅聧聫忙禄聢忙聼聸茅茂陆聠莽茅聳禄茫聜聣莽庐聳茅聫聦茂陆聣氓鹿聡茅隆聯莽聤庐氓聛聻茅聴聛?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳鹿茅聴麓莽莽潞戮氓陆聮忙聼聼茅聴聜氓炉赂莽录聛莽聤碌氓篓聙茅陋聻忙聽篓忙陆聫茅聧聯脩聟脡聭茅聬聨猫忙麓聵茅聬聯茂驴聽氓录禄茅聬聰猫隆隆氓聛聯氓篓聭忙卢聵莽聞陆氓漏碌芒聙鲁忙麓颅茅聫聦茂驴聽氓聼聙茅隆聮氓聥卢忙聶聺茅聳赂氓卤戮莽篓聯茅聧聵忙聮禄忙聜路氓氓聠氓漏垄猫忙麓聼茅聨庐芒聞聝忙聝聰茅隆聰忙聝搂忙聲赂茅聴聛猫聧陇氓搂戮氓篓聟茂陆聡盲潞赂茅聤聤芒聲聛莽茅聳禄忙聴聡茫聙聙茅聧聥忙聮鲁芒聵聣氓篓聠忙聢聽莽聳聞氓漏碌茂鹿楼氓娄聻盲禄聥氓鹿聝氓氓聽聺莽聟聸氓篓麓茂陆聡莽隆露茅聨路茫聜聣忙垄禄忙碌拢猫聴聣氓聬鹿茅聳赂莽聠路忙陆隆氓炉聫聮茅聧聲忙聰鲁?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    dict[str, Any]
        莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿氓氓卢碌路茅聬聴聴脙鹿茅聰聺氓聽聼莽禄戮氓聣搂氓聨搂茅隆颅猫路篓忙聧聞忙赂職氓聣卤茅聳禄氓驴聯茅陋聫莽禄聽忙聮禄氓录禄茅聬聰莽垄卤忙聲聤茅聳赂氓驴聯氓聬聥茅聧聨忙聮禄忙垄潞莽聙鹿莽聥聙氓職聶茅聴聛氓赂露聛茅聬聸芒聜卢茂陆聨氓聳聴茅聧聤茅芒聜聭掳氓拢聤茅聧聽忙聨聴氓麓聭茅聨戮忙聳楼忙搂聢茅聳碌氓驴聤忙茅聨路忙麓陋忙垄潞茅聧聸氓漏聞芒聵聣茅聳驴忙聸聵氓聞鲁茅聢搂莽聛職茅聬聼芒聲聟氓录禄氓篓聭忙篓路忙搂聢茅聳赂忙楼聝氓聛聻茅聬聫忙聮禄忙垄潞茅聳芦莽聜虏猫聥炉忙戮搂氓忙聼聸氓篓聠氓驴聯忙職聶忙陇陇莽鹿聵茅聨录氓聹颅忙聠麓茅聧聲氓鹿聭茅聴聜盲陆赂忙聠隆忙赂職忙氓麓鹿忙碌录氓颅聵氓聛聥忙驴氓聼聳茅聢路忙聢聽莽麓聮莽聙拢氓聛楼茅聳赂忙卢聫氓芦聹聣忙驴聽氓漏聜氓聲掳茅聧漏茂陆聣忙聼拢氓篓聭氓聧聻忙聻聼莽聙碌忙聺驴氓芦掳茅陋聻氓聸聬聰氓聣虏盲禄搂茅聴聜氓聜聶猫聝露莽禄庐忙聲聥茅聬聨?
    """
    return set_global_seed(int(seed))


def _run_env_check(output_root: Path) -> dict[str, Any]:
    """茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝莽录聜盲陆虏氓聛?茅聴??忙驴聽莽聰碌氓搂路茅聫聛氓聭聤忙聟篓茅聬聭氓芒聜聜聻忙聦聤忙戮鹿芒聲聥莽露聛氓庐聲茫聞漏氓录露茅聨麓莽聥聟莽麓卤茅聴聜盲戮聙芒聜聜虏猫聥戮搂忙聮聼隆莽聛聻氓聣搂茫聙聮茅聳鲁脩聦氓聣篓莽录聧氓聴聸氓麓聭茅聧聸忙職娄莽聙鹿芒聜卢茅聧聲氓聨赂茅聬聨氓聟聣氓聛聯茅聬聨茂陆聨氓聣聸茅聰聸莽聠赂莽鹿聸莽聙碌莽氓漏碌芒聙鲁忙麓颅茅陋聻氓驴聸忙聜漏莽聮聡猫聦聴聛氓聹聞陇茅聧陇芒聲聟氓鹿聵莽录聛忙篓路忙聥禄忙驴聻猫戮戮莽碌陆茅聨虏茂驴聽氓麓炉茅聬聬莽聝聵莽聟聶茅聳赂忙露聵茂鹿楼茅聧聤忙聞卢芒聰聭茅聢楼氓聠虏莽录聛莽聭楼茅聢禄氓潞篓氓鹿聠猫陇聨茅?茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼茫聞搂莽聫露茅聳赂莽鹿聞茅聫聠茂鹿聜忙聼拢茅聤聫茫聢漏莽聞戮莽录聛忙聞颅茅聫聦莽聠录氓鹿聭茅聨掳茅聺聸忙莽录聜盲陆潞氓陋聧篓氓聽聲氓录禄茅聰聺氓聭聜莽聠录忙聬鹿茅聳露職茅漏鲁茅聴聛猫炉虏氓莽潞掳氓露聟莽聲聺茅聨录盲陆赂猫聯聼茅聳禄忙聴聡忙芦拢莽潞戮氓聟录盲驴聤茅隆聳忙驴聡忙陇陇忙聢聺氓聙驴氓猫聶鹿忙聥聽茅聢楼猫聶芦莽聛聲茅聴聛莽篓驴茅聧聰忙楼聙脦拢茅聨掳忙聺驴莽庐禄茅聬聽氓聸掳聶茅聰聸忙禄聝芒聰聭茅聬聬氓聫聣茅聳赂茫聞楼莽聛職忙路聡氓漏聝莽娄聮莽聙拢忙聥潞茅聴聛茅聺漏氓炉聦茅聧聶氓潞篓忙聜聻茅聬聬盲潞聹忙陇陇忙聮露氓搂麓茅聧聵氓颅聵茅聬聬忙聫聮忙職聶茅聯禄忙聽颅忙聼聸氓芒聜聬聮茂鹿聜氓录虏茅隆聮氓聥卢芒聮聭莽录聜盲陆鹿脙驴庐氓虏聞忙聜聵猫鹿聡忙聴聜忙聡聡茅聮聵忙驴聥忙垄潞茅隆聮芒聜卢莽禄聣莽聰赂氓聼聸茅聨麓茂赂陆莽聟聲忙驴聽茅聺聸忙拢聴茅忙聜氓聽聼芒聮聭茅聳赂忙露聵茂鹿楼茅聬聯茫聝漏忙聼拢茅隆聯忙驴聢莽陋聰茅聳碌氓聴聴氓聙驴氓庐聲莽篓驴脦聰忙碌拢氓聫聣氓氓漏碌莽聜露忙聫禄虏盲禄聥氓鹿聣茅聰聼莽聤虏莽聳聳禄忙聞聲赂莽潞戮氓楼赂忙聜聴茅聰聺氓潞隆盲潞聹茅聳禄氓驴聰忙聦鲁茅聫聦氓露聣氓娄路茅隆聳忙禄聟忙職陇茅聬聨猫搂聞忙麓聳茅聤聢氓聭聤盲驴聤茅聬聭猫聴聣脦篓莽聭聹氓露聠忙聟篓?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    output_root : Path
        茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聧茅聳禄忙聞陆聶忙碌聹猫戮篓莽鹿聺茅聬垄茅聺聸脨垄忙驴聻氓卤聜氓聙聴莽禄庐莽禄聵氓莽聙聸忙氓聨碌茅聴聜氓聜職氓聙赂茅脣聡茅聰聲氓聜聸莽聟聲忙驴庐忙篓禄氓聣職氓篓录忙聞掳忙聼聲茅聧楼脩聝忙楼聽莽聜麓茂鹿垄氓庐聲茂驴聽忙聜聶茅聧聫氓聭聤氓聛聟茅聴聜氓聜聶莽茅聨虏茂驴聽氓麓聺猫鹿聡忙露聵莽拢禄茅聳鹿氓聣搂莽虏聢路忙聨聭茫聙聮氓茂陆聟盲潞戮氓漏碌莽聜露氓拢聢氓庐聲莽聰赂氓聼聙茅隆聮氓聥卢忙露搂茅聳禄氓聙赂茅聬拢莽聝陆氓茅聤聢氓聴聴盲禄录茅聬聨氓聟聣氓聛聯茅聳鲁脩聦氓聣聼茅聨麓聡莽麓聮氓娄炉氓聹庐職茅聴聛茅聺聸莽聣聠茅聨鲁氓潞篓氓录鲁茅聬聬莽聝聵莽聟聲茅聳禄忙篓潞氓聲驴忙驴露聣忙聼隆氓庐聙氓卢聢掳茅聯禄忙聽颅忙聼聧猫茅聧聯莽聠录氓鹿聝猫莽禄禄忙驴聥氓麓聼茅莽聤碌氓聨掳茅聴聛猫聧陇氓搂碌忙碌聹盲禄聥氓麓聺忙聬麓茂陆聟氓漏碌氓聜聹莽禄戮脩聡忙聼聸茅隆聬茂陆聡芒聵聣氓漏碌脗陇忙鲁聲莽禄卤忙聨聴忙聜漏茅聴聢忙露聴忙搂聬茅聴聛??茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聫茅聢搂莽聞娄莽禄禄忙驴聥氓麓聼茅隆聯莽聥聙忙搂卤氓漏碌莽聜麓忙陆聶茅聧職氓氓篓聠忙聢聺茅聳赂脩聞茅聢路忙聢聻茫聛聞茅聨潞氓聴聴盲潞戮莽录聜盲陆赂茅氓聢聮忙驴聽忙掳聛聞茅聳赂氓驴職氓聛聬茅聧聶氓聴聸忙垄潞莽录聛忙篓禄莽聟楼茅聳鹿莽垄卤氓聛聞茅聬隆莽麓聯茅聧聦忙卢聮茅聳赂氓忙聧聡茅聫聦忙露垄氓鹿聡茅聤聤茂赂陆莽聫聳莽录聜盲陆潞氓篓聙莽禄聽忙聽颅氓录禄氓篓聭茫聢聽莽聞潞茅聳赂忙聞庐氓聛聯茅聬拢猫聜漏莽麓聯忙碌拢氓聯聞脨隆茅聳赂忙氓篓聤氓?``audits/env_report.json``茅聴?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    dict[str, Any]
        茅聴??忙驴聽莽聰碌氓搂路茅聫聛氓聭聤忙聟篓茅聬聭氓芒聜聜聻忙聦聤忙戮鹿芒聲聥莽露聛氓庐聲茫聞漏氓录露茅聨麓莽聥聟莽麓卤茅聴聜盲戮聙芒聜聜虏猫聥戮搂忙聮聼隆莽聛聻氓聣搂茫聙聮茅聳鲁脩聦氓聣篓莽录聧氓聴聸氓麓聭茅聧聸忙職娄莽聙鹿芒聜卢茅聧聲氓聨赂茅聬聨氓聟聣氓聛聯茅聬聨茂陆聨氓聣聸茅聰聸莽聠赂莽鹿聸莽聙碌莽氓漏碌芒聙鲁忙麓颅茅陋聻氓驴聸忙聜漏莽聮聡猫聦聴聛氓聹聞陇茅聧漏氓聽聺茅聤聣茂赂戮莽炉聙茅聳禄茫聢陇忙陆搂茅聢禄氓聽聼莽鹿聸忙碌拢氓聠虏忙麓陇茅聬聮氓聻聛聞茅聨潞氓露聢氓聟聦莽录聛莽聤鲁忙聧聡茅聫聦茂陆聣忙聲聬茅聧楼脩聝氓鹿聥忙驴聽莽垄聵氓聣聺茅聨聜聻氓录聧聦莽聜虏莽莽聭聹氓驴聯脦拢茅聧楼芒聮聮氓篓麓茂陆聞猫颅娄茅聬聮茅鹿聝忙聼隆茅聧茅聳赂忙聨聯氓赂聮茅氓聟录氓戮聞忙驴聽氓聥颅忙聻聙氓漏碌莽聤碌忙聲赂忙驴庐莽聰碌氓聳聤氓庐楼氓莽聟聲茅聳鹿氓炉赂氓搂路莽潞戮氓楼赂忙聜聴茅聰聺氓潞隆氓鹿聴莽禄戮氓莽禄卤忙聨聴忙聜漏氓庐聲氓聸搂氓录篓茅聴聛氓聯聞莽聻录茅聧聽忙聽聙聛氓篓聠忙聮麓氓職聝茅聳鲁猫陆掳猫聝露茅聧聵盲禄聥忙垄禄忙碌拢猫聶鹿茫聙聤茅聳潞氓聜聶莽路職氓庐?

    茅聴聜氓聜職氓聙路氓篓麓氓聸茅聢搂莽篓聢茅聫聟莽聤禄茫聙聛氓篓聠忙聮鲁氓聠聙忙陇陇忙聮露氓聛聼茅聬聸茫聝漏忙垄潞茅聧聺氓聥颅芒聳聣茅聳赂忙篓驴氓聛聬忙戮鹿忙聸隆茫聛聬茅聨潞忙聮露氓聨碌茅聴聜氓聜職氓聙赂茅氓麓聺氓氓陇聥莽聟聲?
    ------
    RuntimeError
        茅聴聜氓聜職氓聙路氓篓麓氓聸搂莽聲路茅聬垄莽莽陋聴茅聳鹿茅聜娄氓聳聳茅聧篓忙驴聥氓鹿聳氓篓拢莽聛驴聻氓聭搂碌忙路聡氓漏聡氓娄露茅聧聸忙芦拢莽录聜盲陆潞氓搂碌氓漏垄忙莽聛聝茅聴聛忙聦聨莽鹿聜茅聨鲁氓潞篓氓录鲁茅聬聰猫隆隆氓聛聯莽聭聶氓聥卢氓掳录聛猫聢碌氓虏赂氓炉芒聵聣茅聤聫莽聤碌氓聤麓茅聴聛莽莽鹿聠茅聫聟忙禄聢氓麓鹿茅聬垄莽莽聟聺莽聙鹿茫聝楼茅聧聥忙聞庐莽麓聯忙碌拢猫炉隆莽聞陆茅聳?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜麓莽聣聽茅隆垄忙聸垄氓娄露茅聧隆忙陇戮莽虏隆忙驴隆莽聜聙聳茅聧聰脩聟莽虏禄茅聨麓莽聤虏茅聳赂脩聞茅聬聯忙卢聭莽麓聯忙碌拢茅聺聸莽聛漏茅聳潞氓莽篓聯莽禄卤忙聨聴氓聼聙茅隆聮氓聥芦茅聠聡忙驴聽茅聺聸氓聲炉茅聫聜芒聜卢茅聴聜盲陆潞茅聫聢氓聢聮茅聬聵莽禄聵氓聙驴氓娄陇茂陆聟氓聲炉茅聧?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙聮虏莽茅聧篓忙驴聡茫聛聞茅聫聧盲禄聹茅聳赂茫聞楼氓聙驴茅聫聦忙露垄芒聙娄茅聨麓忙驴聡莽聛聯茅聴聛氓聯聞忙拢麓莽禄聽忙聮禄氓录禄氓篓聭茫聢漏忙聲聝茅聳驴忙驴聠忙茅隆娄茫聞漏忙垄潞莽禄聥氓聦隆氓聤聼茅聳赂氓驴聰茂鹿垄氓氓漏職忙聲聬忙戮露氓漏聞茅聴聛猫莽鹿聭氓忙聦聨莽茅隆聫氓聥碌忙陆聳氓漏碌莽聤鲁猫聣戮莽潞戮氓聟录忙聟篓氓氓聴聴氓聨麓茅聳赂氓忙聦戮莽禄庐忙卢聭忙聥聽茅聬聬芒聵聣茅聯禄忙聽颅忙聼聸氓篓聭氓聧聻忙聻聸氓篓麓忙禄聨氓聻卤莽禄禄忙驴聥忙氓篓聭忙篓路氓聛聯忙驴聽忙掳拢芒聜卢茂鹿聜氓麓录茅聤聣茂鹿聙芒聜聺聴忙聼聸茅隆颅忙聢聺盲潞聳氓篓麓忙禄聞莽虏聯茅聫聦莽聠赂忙陆聫茅聧聯脩聟脡聭氓茫聜聝氓聞卤茅隆娄忙鹿聛氓漏碌莽聤虏茂鹿陇茅聬聴氓驴聸忙聜戮茅聬聭忙篓禄氓聫聫氓漏碌莽聤虏氓聲炉茅聤聡忙驴聢芒聜忙麓聫茅聧聰氓露聡忙聧垄氓漏碌氓莽拢录忙驴聽氓篓戮氓聛聯莽录聛猫戮篓脡聭莽禄禄忙驴聥忙聜陆茅聴聢忙露聴莽虏露茅聴聛氓陆聮忙聰卤茅聧篓氓聹茅聬聬氓莽聭鹿茅聳鲁脩聦氓聣聼茅聬聸氓麓聵茅氓聵聣茫聞漏忙聼聲忙戮露忙篓潞忙聻陇茅聳赂忙卢聫忙拢聺氓忙麓麓聹茅聨禄忙聨聧莽掳录莽录聜盲陆陆莽聙職盲录麓茫聙聛氓娄炉氓楼陆忙麓陋茅聧聸氓茅聴聜盲陆赂忙聠隆莽禄禄氓聜聹芒聜卢忙掳录氓聶拢茅聧漏茫聢隆氓戮聞茅聨掳氓聨卤茅聴聛猫搂聞氓聞鲁茅隆聲氓娄麓茅聨潞忙聴聜莽拢录茅聫聣氓聽聸氓聤聨茅聨鲁氓聥颅莽麓聮莽录聛忙篓录莽庐聯忙陇陇莽鹿聢茅隆垄忙漏聙忙聜漏茅聨禄忙聨聰莽聳聞茅聴聜氓聜職氓聙路茅聭鲁氓聽聲茫聙聛忙驴聽氓聸搂莽忙陇陇氓驴聯忙拢聴莽禄露茫聜聟脦聰茅聰聺氓聭職聶忙聺漏忙驴聥忙垄潞茅聧聸氓漏聜氓搂娄氓篓麓忙禄聟氓聛聼忙戮鹿忙聸垄氓麓聴茅聴聜氓莽庐職茅聴聛茅聺聸莽聣聠茅聧聤氓聭聤莽娄聧忙楼拢氓聠漏莽聟聼?茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙碌拢猫聶鹿莽陇聤氓漏碌莽聤虏忙碌聜茅聫聛盲陆鹿盲驴聤茅聤聢氓聭聶潞茅聳?JSON 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜麓莽聣聽茅隆垄忙聸垄忙聲聝茅聢搂忙聜驴茅隆聲芒聜卢茅聫聦莽聠录氓鹿聠茅聬聴氓聽茅聴聛氓聯聞氓芦篓茅聧聨莽聰赂氓录禄茅聢楼忙聞聛氓聬聥茅聳赂忙聞漏氓聤聦茅隆忙垄潞茅聧聸氓漏聞忙聡聝莽录聛莽禄聵茂鹿垄茅聨庐茂驴聽茅聰聲芒聜聬聬氓漏聡忙聼聲忙驴聻脩聙氓聛聯忙驴隆忙聫聮莽聣聫莽禄卤忙聮麓忙聥聟茅聨禄忙聨聧氓茅聴聜盲陆赂忙聠隆氓篓虏茂鹿聜氓麓鹿茅聳露聠氓聛聜忙驴聻氓聴聵氓聻聼茅聧聥忙聮禄氓麓路茅脩聟氓聤赂氓茫聜聨氓聨录茅聧垄忙聼楼忙聜?
    """
    report_path = output_root / "audits" / "env_report.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "00_check_env.py"),
            "--project-root",
            str(ROOT),
            "--report-path",
            str(report_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        stdout_payload = _loads_json_from_stdout(completed.stdout, source_name="00_check_env.py")
    except RuntimeError:
        raise
    if completed.returncode != 0:
        raise RuntimeError(f"00_check_env.py failed: {stdout_payload}")
    return stdout_payload


def _run_prepare(dataset_name: str, raw_root: Path, output_root: Path):
    """莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳鹿茅聴麓莽莽潞戮莽聭掳莽聙拢忙聧拢茅聳禄忙茅聹聣茅聳驴忙驴聠莽聣聞茅聢搂氓聛聯茅聴? prepare 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳鹿茅聴麓莽莽潞戮氓陆聮忙聼聼茅聴聜氓炉赂茅聳禄茫聜聨氓聶拢茅聫聦莽聠录氓鹿聠茅聬聴氓聽茅聴?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫聫茅聬聼忙卢聫氓茅聧聶芒聜卢茅聴聛氓聯聞莽聼聣莽录聧盲陆赂茅芒聜庐聲忙聝掳忙驴庐猫搂聞茅聨庐忙楼聝忙聠麓茅聧聲莽聼庐莽录聜盲陆陆氓聼聳氓庐聙莽篓驴忙驴聽忙掳聛聞茅聳赂忙驴聞氓聞鲁猫陇掳茅鹿聝忙垄潞莽禄聸聻氓垄虏茅聳碌氓聸篓莽聸炉氓炉聬聛氓录聧芦莽聠潞氓聨卤氓娄聻茫聜聠氓聤搂莽禄虏氓聣搂莽虏聢茅聧聬莽聟聼茅聳鹿忙聧聬忙鲁聸茅聫聥忙聢娄忙聼聲茅聧楼脩聝忙陇陇茫聢隆忙麓聼忙驴聜卢茅聳鲁氓聯聞氓聲聽茅聧聥忙聮禄氓鹿聨茅聭潞茫聝垄忙聥禄忙驴聻忙聮麓氓聼聝茅聧聥忙聮麓莽鹿聸茅聭鹿忙聺驴氓聬鹿莽录聛猫戮漏氓聛陇氓庐聲氓聧聻芒聵聣氓娄聶鹿芒聜垄潞茅聧聸氓忙庐聵茅聳赂氓潞垄氓聤聶氓庐聲茅聺聸脦聰茅聧聸氓聨卤茅聴聛氓聯聞忙麓垄茅聧聰氓虏聞忙聜聵茅聬聵莽禄聵莽聟聼茅聳鹿茅聜娄氓聣篓茅聼聼隆莽聛聻莽聜卢莽陇聝莽录聛莽禄聵莽茅隆娄猫戮漏莽麓聮茅聢搂忙聥聟茅聧娄忙麓赂茅聴聛莽禄聵茅聬聴氓驴聲氓录搂茅聢搂忙聜聴茅聬垄茅聺聸忙陇陇茫聞楼莽鹿聭莽禄聜氓露聣茅聧芦忙露聮氓聛聯忙碌拢氓聣虏忙陆漏忙陇陇忙聨聭氓聙鹿氓篓聢茅鹿聝忙垄潞茅聧聯猫聦卢氓聨聸茅聳赂氓卢聡聵氓庐聲忙聸垄忙聜垄茅聧聫猫聜漏氓聨陋忙驴聽莽聰碌氓聛聸茅聬聫忙驴聥氓麓聹?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    dataset_name : str
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬氓庐聞忙聡聤氓陆聮氓麓露猫陇聹氓篓麓忙聽聼聲忙驴聻莽聜卢忙芦聠茅聳赂茫聜聟氓茅聫聦茫聜聡莽虏聶莽聮潞茫聞陇莽虏碌氓漏碌莽聜虏忙聡聬忙驴庐氓聻漏卤忙驴聽氓聸搂脦搂茅聳赂氓卤戮莽聼戮鹿忙聸垄忙垄禄忙碌拢茅盲赂虏莽录聛猫鹿虏聶茅聨篓潞忙聺聢芒聲聝莽聲露茅聴聛莽禄聵忙垄禄茅聧聵脩聡忙聜聻茅聧篓盲潞聹茅聳鹿氓聯聞忙拢聴忙碌聹忙聝搂茅聤聢氓潞隆盲潞聹莽录聛莽禄聵脙陋茅聬拢茅莽虏聶莽聮聡莽聜茅聧漏芒聜卢忙陇陇忙聨聭氓芦庐聯盲戮聙忙聼聲猫鹿聡氓篓戮氓聛聯茅聭录莽聲戮茅聴聜盲戮聙忙陆搂茅聬聴氓聴聶芒聜卢忙掳录氓聶拢猫聴聺茅聳驴忙聸聻氓聙鹿茅聬聯氓聠聠赂氓漏聤氓聽聺莽陇聢忙驴聻氓聴聵氓聻碌茅隆楼忙掳颅茫聛聞茅聨戮猫路潞脨拢忙陇陇忙麓聼茅聫聦莽聠录忙聜聹氓娄莽垄聧莽录聜盲陆赂氓茅聧聥茫聜聣莽聧聫聧茫聞娄莽露聡茅聳赂忙聮聴氓聮聝莽麓碌茅聴聜盲陆赂忙聠隆氓氓漏職氓麓聴氓娄戮炉忙陆?
    raw_root : Path
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏氓庐聲氓聸搂氓鹿聢?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬氓庐聞忙聡聤氓陆聮氓麓露猫陇聹氓篓麓忙聽聼聲忙驴聻莽聜卢忙芦聠茅聳赂茫聜聟氓茅聫聦茫聜聡莽虏聶莽聮潞茫聞陇莽虏碌氓漏碌莽聜虏忙聡聬忙驴庐氓聻漏卤忙驴聽氓聸搂脦搂茅聳赂氓卤戮莽聼戮鹿忙聸垄忙垄禄忙碌拢茅盲赂虏莽录聛猫鹿虏聶茅聨篓潞忙聺聢芒聲聝莽聲露茅聴聛莽禄聵忙垄禄茅聧聵脩聡忙聜聻茅聧篓盲潞聹茅聳鹿氓聯聞忙拢聴忙碌聹忙聝搂茅聤聢氓聴聵莽漏驴莽录聛忙聫聮氓垄陆莽禄庐氓卢鹿聮茅聫聜氓聯聞盲潞戮茅聳驴忙驴聠莽掳录莽禄篓猫聜漏芒聜忙聦赂莽禄聣忙聮聙聰茅聧聬脦搂茅聳赂芒聞聝脙陋茅聫聠忙忙垄潞茅聧聺茫聝楼脨拢茅聳鹿莽聭掳忙麓颅茅陋聻氓聠漏氓炉录茅聫聣忙聡鲁茫聞漏忙聼聧茅聧聺氓聥聜卢氓聭聤莽茅隆聯莽聜漏忙聰陆茅聳禄忙篓驴氓茫聝漏忙聼聼茅聧聬忙聼楼氓聫聞茅聳碌氓聴聴氓聙驴氓庐聲忙楼聟忙聡聳茅聢潞氓聠戮氓聲聨茅聴聜盲陆赂忙鹿卤莽禄聲庐茅聬聬莽聰碌盲赂聧氓漏碌莽聤鲁莽茅聬聯忙聸垄忙聼聧莽聭聶氓聥聤陇氓篓?
    output_root : Path
        茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聧茅聳禄忙聞陆聶忙碌聹猫戮篓莽鹿聺茅聬垄茅聺聸脨垄忙驴聻氓卤聜氓聙聴莽禄庐莽禄聵氓莽聙聸忙氓聨碌茅聴聜氓聜職氓聙赂茅脣聡茅聰聲氓聜聸莽聟聲忙驴庐忙篓禄氓聣職氓篓录忙聞掳忙聼聲茅聧楼脩聝忙楼聽莽聜麓茂鹿垄氓庐聲茂驴聽忙聜聶茅聧聫氓聭聤氓聛聟茅聴聜氓聜聶莽茅聨虏茂驴聽氓麓聺猫鹿聡忙露聵莽拢禄茅聳鹿氓聣搂莽虏聢路忙聨聭茫聙聮氓茂陆聟盲潞戮氓漏碌莽聜露氓拢聢氓庐聲莽聰赂氓聼聙茅隆聮氓聥卢忙露搂茅聳禄氓聙赂茅聬拢莽聝陆氓茅聤聢氓聴聴盲禄录茅聬聨氓聟聣氓聛聯茅聳鲁脩聦氓聣聼茅聨麓聡莽麓聮氓娄炉氓聹庐職茅聴聛茅聺聸莽聣聠茅聨鲁氓潞篓氓录鲁茅聬聬莽聝聵莽聟聲茅聳禄忙篓潞氓聲驴忙驴露聣忙聼隆氓庐聙氓卢聢掳茅聯禄忙聽颅忙聼聧猫茅聧聯莽聠录氓鹿聝猫莽禄禄忙驴聥氓麓聼茅茅聴聜盲陆赂莽露聤氓娄芦氓聺聴忙聜聻茅聰聲氓聜聸氓聛?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    Any
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聨禄忙聨聰芒聜卢忙垄禄茅聧聦忙掳卤氓茅聳?忙驴聽莽聰碌氓搂路茅聫聛氓聭聤忙聟篓茅聬聭猫聴聣忙聻聛茅聳鹿茅聴麓莽莽潞戮氓忙聼拢茅聫聝氓聜職氓聤聥莽聙鹿忙聸聻氓录露莽禄禄忙驴聥忙氓篓聭氓聧聻氓聣卤茅聴聛莽篓驴茂鹿陇氓篓录茂驴聽氓录禄茅聤聤芒聲聛氓聧鲁茅聳禄忙聞颅莽楼麓茅聧聥忙聮禄氓麓芦茅聤聣脩聜莽卢聣茅聴聛氓聯聞茅隆聭忙卢聯忙聜垄茅聧隆忙露聶氓聛拢氓娄陇茂赂陆氓炉搂茅莽聤禄忙聼拢猫鹿聡忙掳露氓聯潞氓篓拢氓聸搂茂鹿陋茅隆垄忙聸垄茅聤聫忙茅聴聜盲陆陆茅聧聽忙聽芦莽麓聻忙驴聽氓聸漏氓碌聛氓篓聯忙掳拢氓聛聯忙莽聜麓氓赂隆茅陋聻氓芒聰聭茅聨掳氓聛聯茅聨戮芒聮聮氓篓麓茂陆聣忙露聯猫聦麓聯氓庐楼盲陆赂盲潞戮氓漏碌莽聜虏莽聯篓莽禄聤禄氓麓聹氓篓聭忙篓聹茂陆聣氓鹿聡茅聫聣猫路篓茅聳禄氓潞垄莽篓聮茅聰職忙陇陇氓潞垄忙聧聡茅聫聦茂陆聣忙聜垄茅聧聺茫聝娄氓聶搂茅聳禄氓潞垄氓聡鹿茅聧聯莽聠路脦聸茅聧聲氓聬聥氓猫路潞茅聨路忙麓聼拢茅聬聵氓聫聣莽漏驴茅聬聫氓驴聰莽聭搂莽禄禄聵氓炉录莽聙聸忙篓路忙聥潞茅聴聜盲戮聡氓聛聠茅聧聥忙露職芒聜掳录氓聨录猫茅聳赂忙聸篓氓聛聮氓篓聯氓漏職忙聼聲茅聧驴聠忙職聶茅聳禄氓驴聤忙聶聽茅聫聦茫聞漏氓录麓茅聤聤茫聜聣氓聛聴氓娄陇莽聤碌氓聛聻茅聧聰忙卢聫莽聲路茅聤聤芒聲聛莽潞搂茅聳鹿氓炉赂氓氓娄虏氓陆聮忙聼?
    """
    dataset_cfg = copy.deepcopy(  
        _load_yaml_config_cached(str((ROOT / "configs" / "datasets" / f"{dataset_name}.yaml").resolve()))  
    )  
    seq_ids = _list_sequence_ids(raw_root)  
    cfg: dict[str, Any] = {  
        "dataset_name": dataset_name,  
        "raw_root": str(raw_root),  
        "seq_ids": seq_ids,  
        "output_root": str(output_root),  
        "project_root": ROOT,  
    }  
    if dataset_name == "sim":  
        cfg["scene_id_by_seq"] = {seq_id: "S(A0,N0,V0,G0,K6)" for seq_id in seq_ids}  
    else:  
        cfg["field_mapping"] = dataset_cfg["field_mapping"]  
    return PreparePipeline().run(cfg)  


def _build_sim_paper_split_rules(raw_root: Path, dataset_manifest: dict[str, Any]) -> dict[str, Any]:
    """氓漏?SIM 茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聟忙職娄茅聳赂氓聛聟茅聧聶氓驴聸忙聼聼茅聨聺驴脨聯茅聳赂茫聜聟氓芦掳茅聫聦忙露垄茅聰聺氓聴聶莽录聛莽聜聞鲁氓篓录茂驴聽氓录禄茅聫聡茫聢聽茅聠聡忙驴聽氓聻芦氓聤聳莽卢聭茅聴聜盲陆赂忙聠隆茅聧篓猫搂聞莽鹿聸茅聢搂忙聼隆莽聛聻氓聣搂忙麓麓忙莽聜虏忙麓颅茅隆垄忙氓麓聮茅聰聰氓聣?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓麓茂陆聢氓篓聟莽聤禄氓鹿聳氓篓拢氓莽禄戮忙聝搂茅聫聦忙聸垄氓麓录氓漏碌忙聞娄莽录聜盲陆潞氓搂碌茅聬聴忙聸垄忙鹿聛茅聴聛莽禄聵脙娄炉氓漏聡氓麓聲茅聨掳莽聣聢莽聟聲茅聬聨茂陆聨盲潞聹茅隆聫莽麓聮忙聺聢茫聞楼莽卢搂茅聳鲁脩聦氓聣篓莽录聧氓聴聸氓麓垄茅聬拢氓卤戮氓聟聹茅聬聽氓聯聞茫聜陆茅聬聼茅聜娄莽聟楼茅聳鹿茅聜娄氓聳職猫垄娄忙驴聽忙庐驴氓聳聳茅聰聲茫聝楼茅聮聵氓忙職娄茅聳禄忙聢聽莽聯聰茅聫聛氓聸卢忙聼聧莽聮潞莽聝聵忙聝聢忙驴聻氓卤聜氓聤聭氓庐聲忙掳露氓鹿聮氓娄陇芒聜卢茅聬聯忙露垄忙聼聸莽聛聻猫炉虏芒聜卢忙聽篓忙聶楼茅聴聜盲陆赂忙鹿卤茅聧聵脩聡忙聜潞茅聤聤脩聞氓麓虏茅聳赂忙聞碌氓聳聳莽禄聙氓卢聼聸茅聨掳茅聺聸忙聻聼茅聳鲁茅聰聥氓聻驴氓搂聤忙麓陋茅聤聢氓聭卢莽虏露茅聴聛氓聟录忙陇驴茅聧篓茅聛聧氓录虏茅聧聯盲陆虏氓聛聯茅聬垄莽聰碌茫聙聤茅聳赂氓卢聦赂茅聢禄忙聮禄氓麓鹿茅莽聲聦盲潞戮茅聴聜盲陆鹿莽聹鹿茅聧聤忙聞掳氓麓聭茅聨戮莽禄聵芒聮聮氓篓麓茂陆聡莽聯聰氓篓录忙聞掳忙聼聸忙聬麓茂录聞茂录聧氓漏垄猫忙麓聼茅隆垄忙漏聙氓搂聙茅聬聵茅聺聸氓搂娄忙驴隆莽聜聙聳氓氓漏職氓麓聭茅聨戮莽禄聵莽聟聫茅聳赂氓聳聬茅聧聤忙聞聜卢忙庐驴氓聳聸茅隆聲脩聡氓聼楼忙戮露忙聞卢茅聳禄忙篓禄茅聧聻忙麓陋忙垄禄忙碌拢莽聝聵氓職聙忙陇陇忙聸篓氓聫聝氓庐聲忙聢娄忙聜聶茅聧篓氓潞聤氓漏碌莽聜麓氓聻聧篓氓陆聮忙聼聥茅聧楼莽聟聼茅聳潞氓聠篓莽聣聹氓娄芦忙聢聺脙鹿茅聬聯氓潞拢莽聞赂茅聳鹿茅聢禄忙聮禄氓麓鹿茅隆聰莽聲聦盲潞聻茅聴聜盲陆赂忙聠隆茅隆篓氓聴聵莽鹿聸忙驴聽氓聸篓氓聛聵忙陇陇忙聸聴氓聛聯莽聙鹿忙聸聻氓麓隆芦忙驴聥氓麓聴忙戮露猫氓聣聸莽禄卤忙聮麓忙聥聟忙楼娄忙聦聨茅聴聛氓聭聤脡聭茅聨聜職莽聲路茅聨麓茂鹿聙莽庐禄莽录聜盲陆鹿脟職茅聨路忙麓垄潞茅聬聯氓潞庐忙陆聼茅聳赂氓漏聝忙麓聼氓炉聬聛氓鹿聥茅聰聰莽聲聦氓聨陆茅聴聛莽禄聵忙垄鹿氓忙卢聯氓麓聺茅聤聢氓莽聟聝莽聭聹忙禄聢氓麓聹氓搂聵猫聢碌氓職聦氓娄陇茂陆聟氓聲芦茅聧篓氓聜聻忙聼聸莽聛聻氓聣搂芒聴陇氓篓麓忙禄聞莽虏聯茅聫聦芒聙鲁忙聬麓芒聙虏莽掳禄忙驴聻氓氓搂碌莽禄庐氓露?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    raw_root : Path
        SIM 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏氓庐聲氓聸搂氓鹿聢?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬氓庐聞忙聡聤氓陆聮氓麓露猫陇聹氓篓麓忙聽聼聲忙驴聻莽聜卢忙芦聠茅聳赂茫聜聟氓茅聫聦茫聜聡莽虏聶莽聮潞茫聞陇莽虏碌氓漏碌莽聜虏忙聡聬忙驴庐氓聻漏卤忙驴聽氓聸搂脦搂茅聳赂氓卤戮莽聼戮鹿忙聸垄忙垄禄忙碌拢茅盲赂虏莽录聛猫鹿虏聶茅聨篓潞忙聺聢芒聲聝莽聲露茅聴聛莽禄聵忙垄禄茅聧聵脩聡忙聜聻茅聧篓盲潞聹茅聳鹿氓聯聞忙拢聴忙碌聹忙聝搂茅聤聢氓聴聵莽漏驴莽录聛忙聫聮氓垄陆莽禄庐氓卢鹿聮茅聫聜氓聯聞盲潞戮茅聳驴忙驴聠莽掳录莽禄篓猫聜漏芒聜忙聦赂莽禄聣忙聮聙聰茅聧聬脦搂茅聳赂芒聞聝脙陋茅聫聠忙忙垄潞茅聧聺茫聝楼脨拢茅聳鹿莽聭掳忙麓颅茅陋聻氓聠漏氓炉录茅聫聣忙聡鲁茫聞漏忙聼聧茅聧聺氓聥聜卢氓聭聤莽茅隆聯莽聜漏忙聰陆茅聳禄忙篓驴氓茫聝漏忙聼聼茅聧聬忙聼楼氓聫聞茅聳碌氓聴聴氓聙驴氓庐聲忙楼聟忙聡聳茅聢潞氓聠戮氓聲聨茅聴聜盲陆赂忙鹿卤莽禄聲庐茅聬聬莽聰碌盲赂聧氓漏碌莽聤鲁莽茅聬聯忙聸垄忙聼聧莽聭聶氓聥聤陇氓篓?
    dataset_manifest : dict[str, Any]
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬氓庐聞忙聡聤氓陆聮氓麓露猫陇聹氓篓麓忙聽聼聲忙驴聻莽聜卢忙芦聠茅聳赂茫聜聟氓茅聫聦茫聜聡莽虏聶莽聮潞茫聞陇莽虏碌氓漏碌莽聜虏忙聡聬忙驴庐氓聻漏卤忙驴聽氓聸搂脦搂茅聳赂氓卤戮莽聼戮鹿忙聸垄忙垄禄忙碌拢茅盲赂虏莽录聛猫鹿虏聶茅聨篓潞忙聺聢芒聲聝莽聲露茅聴聛莽禄聵忙垄禄茅聧聵脩聡忙聜聻茅聧篓盲潞聹茅聳鹿氓聯聞忙拢聴忙碌聹忙聝搂茅聤聢氓潞隆盲潞聹莽录聛莽禄聵脙陋茅聬拢茅莽虏聶莽聮聡莽聜茅聧漏芒聜卢忙陇陇忙聨聭氓芦庐聯盲戮聙忙聼聲猫鹿聡氓篓戮氓聛聯茅聭录莽聲戮忙驴隆莽聜陋氓聙聳茅聬聴忙楼聛莽芦聼茅聫聡茫聢聽莽庐聳鲁茫聝篓莽搂庐猫陇掳忙聨聭莽陇聜茅聳赂氓驴職氓聛聬茅隆娄忙聺驴芒聰聭茅莽隆路忙聰禄忙驴隆莽聜露忙拢聞茅聬拢莽聝陆氓娄赂茅聰聰氓聣搂莽聵聢茅聴聛莽篓驴茅聧篓氓聧聻忙碌聽氓卢聵楼茅聳驴莽聠潞氓搂麓莽禄聙氓聠聼聧茅聧聺氓聥露忙聰禄莽录聜氓露聠莽禄禄忙驴聥忙聜陆茅聴聢忙露聮忙聻聟氓漏碌莽聜掳氓聦聽茅聧楼猫聢碌忙聶聻茅聴聛莽鲁聲氓聣聺莽禄聥忙聨聴氓聼聸茅聨麓茂赂陆芒聮聭忙陇陇忙聞漏氓聙聲忙碌聽忙禄聢忙聼陇氓篓虏猫莽聛隆茅聳潞氓聭茅聢搂忙聲赂莽潞聽聲氓麓聭茅聨戮猫聢碌氓聳聠茅聳赂忙聸篓氓聣聸茅隆娄莽鹿聺茅聬垄茅聺聸盲潞鹿茅聳赂氓忙聧聡茅聨庐忙楼聝忙聠麓茅聧聲茂录聵茅聴聛氓聭聤忙垄鹿茅聧篓氓聺聴氓娄麓忙碌拢氓聣虏忙陆篓茅聳鲁脩聦氓聣聼茅陋聻氓聠茅聬聵氓聻芦忙庐聺茅聴聛氓聯聞茅忙卢聫脦?``sequences`` 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛氓露聣氓麓聼茅盲陆鹿忙驴聽氓戮路忙聺驴芒聜聸垄氓鹿聤氓庐聙氓聥聞碌忙陇陇忙聨聠氓聛聯茅聳赂莽聡聛氓聻掳茅隆聲茫聝漏氓麓聼茅忙掳卤莽聳聙氓娄聻茫聜聠氓聻聼茅聫聜氓聜聻氓麓聲茅聬垄莽猫聯聼忙驴聻忙聢聶氓聻鹿莽禄?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    dict[str, Any]
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜麓莽聣聽茅隆垄忙聸垄氓娄露茅聧聦忙掳芦氓拢聲氓漏碌茅聧聵脩聝茅茅聰聛聯莽聭聶氓聥卢莽陇聝忙驴聻氓聫聣莽聣聽茅聨鹿聝忙聜聹茅聮聵氓陇聦氓聙聻氓茫聜聨氓聨陆茅聧篓氓聣聛莽麓聻氓娄陇氓聭聮聮氓篓麓茂陆聢氓搂陇莽潞颅氓聽聲忙聼聸茅聬聵猫聶芦氓掳掳茅聳鹿盲戮驴氓聤聨茅聢搂忙陆搂茅聨陆忙禄聢忙聝聦茅聧隆忙露聶莽聟聲茅聳鲁芒聲聛氓聲掳茅聢炉忙聸垄忙聼拢茅聨戮氓颅聵莽陇聥茅聳潞氓卤戮忙麓聵氓炉掳氓聥颅芒聜陆聨盲潞聻忙碌聽忙聮垄潞茅聧聺茫聝楼忙聻聧茅聳赂忙楼聛氓篓聤氓庐聲忙麓陋氓聼聙茅隆聮盲陆碌盲潞聹茅聳鹿氓聯聞忙忙碌聹忙聝搂茅聤聢氓潞隆氓鹿聵茅聳赂氓驴聰茂鹿垄茅聬聸氓麓聵茅聤聤脩聞忙庐聺茅聴聛茅聙聸莽碌禄氓篓聟忙聸垄忙聜聴忙驴聽忙掳陇陇忙聞漏氓聻潞忙戮露氓聥炉忙聼聼茅聬聼莽聣聢氓聳聵莽聙碌茅聢陆氓陇聬氓茅聬聵猫聦卢芒聜垄潞茅聧聯忙聫聮盲潞聺忙驴聻氓聫聣莽聣聽氓炉聭聤忙聥聟莽聬聸芒聙鲁盲潞戮忙楼聽莽聜虏莽聛聺茅聧聰忙掳鹿聸莽聮聡脩聟氓聢掳茅聳鲁猫莽虏聶茅隆垄忙聴聢氓麓录茅聬聰氓聯聞氓录驴聽莽垄聵忙茅聧篓忙聥聝茅聰聲芒聜卢氓聧卤茅聳禄忙聢聻脙录莽禄禄氓聴聵茫聛聞茅聫聧茫聞搂忙聡聟莽录聛忙聞垄鹿茅聨?``explicit_ids`` 茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼忙聲聥莽聭聹氓驴聸氓聼聙茅隆聮盲陆潞芒聳聯茅聳潞氓聭鹿聞茅聧聮忙聨聭芒聳聨茅聨戮氓麓聡氓漏碌茂录聞氓聛聸茅聬篓莽聝陆氓麓?
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓聙驴茅聳驴忙聴聢忙聲庐茅聧聥忙聮露莽虏篓聭忙篓禄氓聶陆茅聳禄莽聭漏茅聫聦莽聠潞芒聜露聶莽禄聽茫聝漏忙聼拢茅聨潞忙聸聼茅聳鲁猫搂聞氓聻露氓炉掳氓聥芦脦聼茅聭陆茫聜聫莽麓卤rain``/``val``/``test`` 茅聴聜氓聜職氓聙赂茅聧聤莽聝陆忙聡聴茅聧聸聻莽庐聽茅聳鹿忙聧聬莽聭聶茂鹿聜忙聜聴茅聰聺氓潞聺忙聠戮茅聳禄忙聨聲氓搂陇莽禄禄忙露垄氓麓卤氓娄炉猫聶鹿盲禄赂茅聬聨猫忙麓聵茅聬聯茂驴聽氓录禄茅聬聰氓聟录氓聛聥茅聳赂氓聳聯茅聧聭茂驴聽忙垄潞茅聨录莽聜虏氓娄录茅聳赂氓漏聜莽露聤忙驴隆莽聰赂氓麓聼茅氓聣搂忙聺赂茅聴聛猫氓麓聵氓篓聣忙露聙聜茅聯聫氓聹拢陆氓篓麓茂陆聠氓陆聝忙碌聹茅鹿聝忙垄潞莽禄聸聻氓垄虏茅聬陋氓陇聦莽麓聮莽聮聡氓聫聣茅聳潞氓卤戮忙麓聼氓庐聲莽聟聨芒聰聭茅聧楼脩聞忙驴隆莽聜聙聲莽禄禄氓聽楼氓麓聲茅聬垄莽猫聯聼莽聙鹿茫聝楼莽聣聹氓娄虏氓漏職忙垄潞茅聧聺氓聥卢茂录聰莽录聜氓露聞莽路職氓炉氓聼聞茅聧聬芒聲聛茅聴聛氓聭聤莽赂驴茅聧聨忙露垄忙聲聬茅聧楼茂陆聟盲潞戮茅聳赂氓聛聟莽禄露忙聼楼忙聜聴氓莽路虏茅聳禄茂陆聡氓聟聵茅聨聞聝忙聝聰茅猫聴聣氓聟聤忙驴隆莽聜陋氓聙聳茅聧篓莽聰赂忙聜聵氓漏碌氓卢聨?

    茅聴聜氓聜職氓聙路氓篓麓氓聸茅聢搂莽篓聢茅聫聟莽聤禄茫聙聛氓篓聠忙聮鲁氓聠聙忙陇陇忙聮露氓聛聼茅聬聸茫聝漏忙垄潞茅聧聺氓聥颅芒聳聣茅聳赂忙篓驴氓聛聬忙戮鹿忙聸隆茫聛聬茅聨潞忙聮露氓聨碌茅聴聜氓聜職氓聙赂茅氓麓聺氓氓陇聥莽聟聲?
    ------
    RuntimeError
        茅聴聜氓聜職氓聙路氓篓麓氓聸搂莽聲路茅聬垄莽莽陋聴茅聳鹿茅聜娄氓聳聳茅聧篓忙驴聥氓鹿聳氓篓拢莽聛驴聻氓聭搂碌忙路聡氓漏聡氓娄露茅聧聸忙芦拢莽录聜盲陆潞氓搂鲁茅聧聴忙聤陆氓录禄氓篓聭忙篓路忙聰陆茅聳赂忙聸篓氓聛聞忙驴庐茫聢陇忙垄潞茅聧聸氓忙庐聲茅聳录氓陆聮莽庐聳茅聢楼忙聴聢氓麓聵茅盲陆赂盲潞戮茅聳驴忙驴聠莽掳录莽禄篓盲禄聥忙聼聸茅聨麓茂陆聨忙陆聬莽录聛莽禄聵莽篓聭茅聬拢忙驴聡莽鹿聺忙赂職芒聜聧聥忙楼聟忙垄潞茅聧聺氓聥庐氓芦掳莽录聛氓陇聦氓聟聵莽炉聯氓篓聯忙掳拢氓聛聯茅聳潞氓卤戮莽聸陋聻忙篓录忙聧聞茅聬聭忙芒聜茅聤聢氓聴聵莽漏驴莽录聛猫聦露忙鲁聲茅聧聮忙聨聰忙聥聽氓篓麓氓卢聴聛茅聺聸莽鹿聮忙驴聝陆忙聝聢茅聧聲芒聮聭茅聴聛氓聛聸茅聭禄忙聶露茅芒聜聫聦氓露聡莽麓聮氓娄莽拢鲁茅聬聨猫搂聞忙麓聳茅聤聢莽茅聨戮氓聛聞茅聳赂忙卢聫茅隆聫氓陆聮忙垄禄茅聧聦忙卢聭忙驴聽芒聜聺聢芒聲聛氓聻聺忙陇陇忙聽篓莽虏聧茅聫聠忙禄聢忙聼拢茅聨掳忙職漏氓篓聭忙聮鲁莽搂鹿茅聫聦茫聞楼芒聰聭茅聧隆忙篓潞莽陋聸莽录聛莽聜聞鲁茅聧職氓莽漏卤忙驴聽氓聸搂茅聳赂氓卤戮莽聼庐忙戮鹿忙聸垄忙垄?3 氓漏?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬氓庐聞忙聡聤氓陆聮氓麓露茅聤聤茂陆聞莽虏聯茅聴聛氓聭聤莽赂驴茅聧聨忙聫聮莽聲路茅聫聦茫聝娄莽聟聲忙陇陇忙聞庐氓搂麓茅聧聰茫聜聡芒聜卢猫搂聞忙聦路莽禄露忙掳露氓鹿聝氓娄陇氓聭聶虏茅聨录莽虏鹿氓聣聺茅聬聫氓聽楼忙垄潞茅聧聸氓漏聜氓搂聙茅聳赂氓忙聧聡茅聫聦茂陆聣忙聜垄茅聧聺茫聝搂莽聟聙莽录聜盲陆潞莽虏聰聲茫聢陇忙聜掳茅隆聰氓聵聣忙聴聢氓麓篓茅隆聰茅聴麓忙卤聴莽录聜氓聜職氓聙路茅聬聮茂鹿聜芒聙娄茅聧楼氓聜篓茅聴聛莽搂碌茅聢路忙聢娄忙聼聼莽禄聥氓聻庐茅聳禄忙聮卤氓聞碌茅聫聦莽聠潞氓聳聤茅聧聴莽聜掳莽掳录忙驴聻茂陆聟忙麓陇茅聰聲氓鹿聤茅聬聬氓聯聞脦篓莽聭聹氓驴聸忙聝聛茅聧芦茫聝娄芒聮聭茅聴聛氓聛聸茅聭禄忙聶露莽聙聸忙篓赂盲潞聹茅聳碌氓驴聤忙茅聭掳氓聺聴忙聼聼茅猫搂聞忙芦聤莽聙鹿忙聸聼茂鹿垄茅聬聯茅聬聵氓聻芦芒聮?茅聴聜氓聜職氓聙赂茅聧聤莽聝陆忙聜聻茅聰聲氓聜職莽庐聳茅聳赂忙麓聳莽禄聙氓陇聦芒聜卢氓聟聣氓聛聯茅聳赂忙聸篓氓聣聸茅聧聵忙聞掳忙垄潞茅聨赂氓潞拢莽庐聯忙陇陇茂赂禄氓聻露茅聨路忙聢娄忙聜垄莽聛聫盲陆鹿忙聳聙茅聴聛莽禄聵脡聭猫陇掳氓聠庐氓聼聙茅隆聮芒聜卢茅聬聫忙驴聥忙聜戮茅聬聭猫聴聣脦漏茅聳鲁猫陆掳莽茅聧聥氓露聣忙垄潞茅聧娄氓聤聥茅聳鹿莽篓驴氓垄陆莽禄禄聵茫聜拢茅聧聬芒聲聛芒聜卢莽聰赂忙聼拢茅聤聫芒聵聠氓聲聫茅聳禄忙聨聴莽庐聳氓娄聥颅芒聜陆聨盲潞露氓篓聯氓漏職忙聼聲茅聧驴聡茅聳禄忙聮卤氓聞碌茅聫聦茫聞漏氓录麓茅隆聫氓聥碌忙麓漏碌芒聙鲁氓录露茅聧聰忙卢聯氓鹿聝氓篓聠氓驴拢氓聲聟忙陇陇忙聳驴氓聬聥茅聧聤芒聰聭茅聬聵莽聛卤忙驴聻氓陇聥莽聸炉茅聫聛氓聠庐忙聲聝茅聧聥莽聭掳忙麓颅茅隆垄忙聸垄忙聲聜莽聭聶茂陆聠氓茅聴聜盲陆潞茅聧聥忙聞掳氓麓聭氓篓聭茫聢聽莽茅聳鲁脩聦氓聣聼氓搂聤忙麓陋氓麓篓忙驴聽氓聜職脨聲忙驴聻氓颅聵莽虏聧茅聬聴莽聤虏莽聲路茅聨麓茂鹿聙莽庐禄茅聬聨茅聺聸忙聭聹茅聬娄氓聽聺芒聰聭茅忙聳陇盲戮聸茅聳赂忙漏聙莽庐聳氓庐聲茫聢隆莽莽聙拢忙聥潞茅聴聜氓聜職莽聣聤忙赂職忙楼聟忙聜聻忙楼拢氓聠漏莽聟聲茅聨录忙路聙聯茅聧職莽麓聮忙聺聢茫聞娄氓掳掳茅聳鹿氓鲁掳忙聡聬莽禄庐忙卢聭忙聧聞茅聤聤脩聞莽露聛茅聴聜盲陆陆氓搂聵茂鹿聜茂陆聻忙戮露氓篓聮氓聯聞茅聢搂氓鹿聤氓漏碌氓莽庐虏茅聳鹿茅聜娄氓聳聲莽禄篓茅聰聥莽鹿聺茅聬垄茅聺聸盲禄漏茅聳鹿忙麓禄盲潞聻莽禄卤莽聜潞氓聼聙茅隆聮盲陆碌盲潞聹茅聳潞氓聸漏氓聳聯忙戮搂氓麓隆忙聺聢茫聞拢盲潞聹茅聳潞氓露聨氓聛聞忙碌聽茂鹿聜忙聼聧茅聳驴氓卢聛漏茅聳禄茫聝庐盲潞隆垄忙漏聙忙聜聶茅聫聣忙聢聻茅聴聜盲陆潞莽虏炉茅聨赂氓氓麓聺氓娄陇氓聠漏茅聳鹿莽聝聵忙聼隆茅聧聥忙聮禄忙聲聬忙聬麓芒聙虏莽掳禄茅聴聛猫莽鹿聬茅氓聼聙茅隆聮氓聠戮忙陆聬忙驴聻氓聫聣莽虏聯氓炉鹿聺氓搂聵茫聝垄氓聛聯忙碌拢氓聣虏忙陆篓茅聳鲁脩聦氓聣職茅聨卤茫聞漏氓聼聙茅隆聮氓聥卢莽聟?
    """
    layout_report = inspect_layout_family_coverage(raw_root)  
    family_to_sequences = dict(layout_report.get("family_to_sequences") or {})  
    if int(layout_report.get("family_count") or 0) < _PAPER_MIN_SIM_LAYOUT_FAMILY_COUNT:  
        raise RuntimeError(  
            "sim paper path requires at least "  
            f"{_PAPER_MIN_SIM_LAYOUT_FAMILY_COUNT} independent layout families"  
        )  

    ordered_families = sorted(  
        family_to_sequences,  
        key=lambda family_id: (-len(family_to_sequences[family_id]), family_id),  
    )  
    if len(ordered_families) < 3:  
        raise RuntimeError("sim paper split requires at least three layout families")  

    train_family_ids = ordered_families[:-2]  
    val_family_id = ordered_families[-2]  
    test_family_id = ordered_families[-1]  
    explicit_ids = {  
        "train": [seq_id for family_id in train_family_ids for seq_id in family_to_sequences[family_id]],  
        "val": list(family_to_sequences[val_family_id]),  
        "test": list(family_to_sequences[test_family_id]),  
    }  
    covered_seq_ids = {seq_id for seq_ids in explicit_ids.values() for seq_id in seq_ids}  
    manifest_seq_ids = {str(record["seq_id"]) for record in dataset_manifest.get("sequences", [])}  
    missing_seq_ids = sorted(manifest_seq_ids - covered_seq_ids)  
    if missing_seq_ids:  
        raise RuntimeError(  
            "sim paper split must assign every sequence to train/val/test; missing: "  
            + ", ".join(missing_seq_ids)  
        )  
    return {"explicit_ids": explicit_ids}  


def _run_split(prepare_result, output_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聟忙職娄茅聳赂氓聛聟茅聧聶氓驴聸忙聼聼莽录聛茫聢聽忙聻聼茅聳禄忙聮麓忙麓聼茅聫聦茂陆聣氓录聧楼茫聞搂茫聙聬茅聬聨猫搂聞忙聦赂氓娄露垄氓聼聙茅隆聮盲戮聡氓聲聫茅聳赂茂陆聨氓聶拢氓庐聲忙聢聻莽鹿聺茅聧聬脩聟莽庐職茅聴聛氓聟录忙聜聜氓篓录脩聟忙茅聬聬忙碌聥忙聰陆茅聳禄忙篓禄氓陆聝茅隆聫氓颅聵盲驴聤茅聧聶氓陇聥氓掳娄忙驴漏聞莽虏聨路氓聭聠麓茅聧聲氓聯聴茅聴聜盲陆潞莽聙聸芒聲聝氓聢聮茅聨戮氓麓聡忙職娄忙驴陇禄氓聛聯忙陇陇茫聢隆忙垄鹿茅聨路氓聭麓忙垄鹿茅聨掳忙拢聹忙驴聽莽聰碌氓搂路茅聫聛氓聭聤忙聟篓茅聬垄茅聺聸忙聻聢莽聙鹿茫聝楼茅聧聥氓聜聸莽鹿聸茅聨麓忙卢聫莽聛陋茅聳赂氓漏聞忙聲禄茅聫聦茫聞楼芒聰聭茅聧隆忙篓潞莽陋聸莽录聛忙聝搂氓陆聻莽禄卢聯氓录禄氓篓聭忙掳虏陋茅聳赂忙聮聴芒聜驴庐氓聸聜聴莽聭聶氓聥卢莽陇聙茅聳禄茫聜赂茅聧漏芒聜卢忙陇陇忙聨聭氓聳職氓忙聞鹿聸茅聧聶氓拢拢忙聢聽忙聧垄氓篓聭忙卢聭莽炉聙茅聳赂忙聞聦氓聭聲戮茅聴聜盲陆赂莽露聤氓娄芦猫路篓莽虏聢忙碌拢氓聭聤忙職聹茅聴聜氓聜聶莽路職氓篓录脩聡芒聙娄茅聧芦茫聝楼莽拢驴茅聬聨氓聹颅氓搂掳忙碌聹氓陆聮忙聼隆莽聛聻猫氓聤聹茅聳鲁?split/leak 莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳鹿茅聴麓莽莽潞戮氓陆聮忙聼拢茅聨麓茂陆聟脨聯茅聳赂氓漏聜忙陆隆茅聫聦茫聞漏氓录麓茅聬聬忙碌聥氓聛聯茅聧聺茫聝搂盲赂聧茅聳潞氓陇聤莽掳卤茅聫聙盲禄聥忙聼拢氓娄聶鹿茂录聧忙陇陇茂陆聠莽聟隆茅聫聦忙露聶莽鹿聺茅聧聸茫聙聬氓茫聜聨氓聨录氓篓录茂驴聽氓鹿聤氓氓聽聼忙驴聻忙聢聻茂鹿漏氓篓录忙聮露莽麓聯茅聧聦忙卢聮茅聳赂氓卢聧聡茅聫聦忙露垄氓鹿聡茅聤聤茂赂陆莽聫聳莽录聜盲陆潞氓篓聙莽禄聽忙聽颅氓录禄氓篓聭茫聢聽莽聞潞茅聳赂忙聞庐氓聛聯茅聬拢猫聜漏莽麓聯忙碌拢氓聯聞脨隆茅聳赂忙氓篓聤氓氓漏職忙聲聬忙戮露氓漏聞茅聴聛氓聭聤氓聤聳莽禄聛忙聮麓聭茅聨戮猫氓聻聜忙陇陇忙聸聻忙聡聨氓篓聢茫聞漏忙垄潞茅聧聯忙聫聮盲潞聺忙驴聻氓聫聣莽聣聽茅聫聦氓卢麓露茅隆聮盲陆潞氓聨陆茅聴聛氓聯聞氓聲芦茅聧聥氓聴聸忙聜聻茅聧聯脩聟忙拢聳潞氓陇聨忙鲁聲忙聲聢氓漏碌茂鹿楼氓盲禄聥莽聧庐茅聫聧氓聙路忙驴聻忙聺聭忙碌聹茅鹿聝忙聼聲茅聧驴聬忙搂赂茅聳禄茅聫聦忙露垄氓鹿聡茅聴聢忙露聶莽聛聢莽录?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳赂忙驴聞氓聞鲁茅聬聮禄忙聜聴茅聴聢忙露聶忙聠赂茅聳禄忙聧聡茅聨鲁掳芒聳聨猫鹿聡忙聯聝盲禄鹿莽录?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    prepare_result : Any
        prepare 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳鹿茅聴麓莽莽潞戮氓陆聮忙聼聼茅聴聜氓炉赂茅聳禄茫聜聨氓聶拢茅聫聦莽聠录氓鹿聠茅聬聴氓聽茅聴?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳鹿茅聴麓莽莽潞戮氓陆聮忙聼拢茅聨麓茂陆聟脨聯茅聳赂氓漏聜忙陆隆茅聫聦茫聞漏氓录麓茅聬聬忙碌聥氓聛聯茅聧聺茫聝搂盲赂聧茅聳潞氓陇聤莽掳卤茅聫聙盲禄聥忙聼拢氓娄聶鹿茂录聧忙陇陇茂陆聠莽聟隆茅聫聦忙露聶莽鹿聺茅聧聸茫聙聬氓茫聜聨氓聫聙氓篓聣忙聸垄氓鹿聭茅聧聲盲录聽茅聳赂忙聸聻莽聛職忙聽隆茅聴聜氓聜聶猫聝露茅隆聽楼茅聧隆忙露聵莽庐虏茅聳赂茫聝聣聶莽禄聽忙聽颅忙聼娄氓娄炉盲戮搂麓茅聳潞氓忙拢聝茅聫聦莽聠潞莽虏聶茅聧篓氓聤聣茅聴聛氓聯聞忙拢聸茅聧聥茫聜聟莽录聛忙篓禄氓茅聳陆忙篓潞茂录聧茅聨戮莽禄聵莽聟聼忙驴隆茫聜聞莽聞娄莽禄聙氓聽楼忙聼聧猫陇聹茅聧聯忙掳露忙聝聦茅隆聲芒聜卢氓庐聲茂驴聽氓麓聵氓忙庐篓氓茫聜聠氓聤聭茅聧漏氓聥炉氓麓楼莽聙拢莽聟聲忙驴聻忙聢聺氓麓聬聫芒聞聝莽鹿聸茅聧聠赂莽陇聥忙驴庐氓漏聝忙陇陆氓庐聲茫聞娄莽鹿聺茅聧聬忙搂露茅聴聛忙卤聡氓聼聞茅聧篓茅聛聧忙聝聞茅忙掳录忙陆聳莽录聜盲陆鹿脡聭忙驴聮庐忙聼聸氓篓聭忙漏聢茅戮職莽禄戮氓聛聯茅聴聜氓聜聶莽陇聛茅聨虏茂驴聽芒聙娄茅聧隆忙露聵莽聜聤莽聭聹氓驴聸茅聨掳莽聣聢芒聮?``metadata["dataset_manifest"]`` 茅聴?
        ``metadata["scene_manifest"]``茅聴?
    output_root : Path
        茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聧茅聳禄忙聞陆聶忙碌聹猫戮篓莽鹿聺茅聬垄茅聺聸脨垄忙驴聻氓卤聜氓聙聴莽禄庐莽禄聵氓莽聙聸忙氓聨碌茅聴聜氓聜職氓聙赂茅脣聡茅聰聲氓聜聸莽聟聲忙驴庐忙篓禄氓聣職氓篓录忙聞掳忙聼聲茅聧楼脩聝忙楼聽莽聜麓茂鹿垄氓庐聲茂驴聽忙聜聶茅聧聫氓聭聤氓聛聟茅聴聜氓聜聶莽茅聨虏茂驴聽氓麓聺猫鹿聡忙露聵莽拢禄茅聳鹿氓聣搂莽虏聢路忙聨聭茫聙聮氓茂陆聟盲潞戮氓漏碌莽聜露氓拢聢氓庐聲莽聰赂氓聼聙茅隆聮氓聥卢忙露搂茅聳禄氓聙赂茅聬拢莽聝陆氓茅聤聢氓聴聴盲禄录茅聬聨氓聟聣氓聛聯茅聳鲁脩聦氓聣聼茅聨麓聡莽麓聮氓娄炉氓聹庐職茅聴聛茅聺聸莽聣聠茅聨鲁氓潞篓氓录鲁茅聬聬莽聝聵莽聟聲茅聳禄忙篓潞氓聲驴忙驴露聣忙聼隆氓庐聙氓卢聢掳茅聯禄忙聽颅忙聼聧猫茅聧聯莽聠录氓鹿聝猫莽禄禄忙驴聥氓麓聼茅茅聴聜盲陆赂莽露聤氓娄芦氓聺聴忙聜聻茅聰聲氓聜聸氓聛?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    tuple[dict[str, Any], dict[str, Any]]
        ``(split_manifest, leak_report)`` 氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄忙驴聻忙聢聶氓聻潞茅聧聥茫聜聣忙聼聲忙驴聻脩聝氓聲聬聫氓漏聡忙聜聴茅聧聫氓盲潞戮茅聴聛氓聭聤忙麓娄茅聧聯忙卢聫芒聳聯茅聰聺氓陇聥氓聙碌茅聬聼忙卢聫氓莽潞戮忙聬聛忙聼聸茅聤聤茫聜聡莽聼聶茅聳禄忙露卤氓聳聳茅隆芦忙禄聢氓聼聙茅隆聮芒聜卢茅聬拢氓鲁掳茅聧聲茅聴聛莽禄聵猫聣戮茅隆聲忙聝聰忙驴聽氓聸篓芒聮聮茅聳赂氓卤戮莽聭搂茅隆娄茂鹿聜忙聼聼氓篓麓氓卢搂氓聛聯莽聯聮盲陆赂忙聬氓聟录氓戮聞忙驴聽氓聠戮忙拢聼氓漏碌莽聜麓忙聦禄茅聧漏氓聠庐氓麓聭茅聨戮氓聻聸氓聛聯?

    茅聴聜氓聜職氓聙路氓篓麓氓聸茅聢搂莽篓聢茅聫聟莽聤禄茫聙聛氓篓聠忙聮鲁氓聠聙忙陇陇忙聮露氓聛聼茅聬聸茫聝漏忙垄潞茅聧聺氓聥颅芒聳聣茅聳赂忙篓驴氓聛聬忙戮鹿忙聸隆茫聛聬茅聨潞忙聮露氓聨碌茅聴聜氓聜職氓聙赂茅氓麓聺氓氓陇聥莽聟聲?
    ------
    RuntimeError
        茅聴聜氓聜職氓聙路氓篓麓氓聸搂莽聲路茅聬垄莽莽陋聴茅聳鹿茅聜娄氓聳聳茅聧篓忙驴聥氓鹿聳氓篓拢莽聛驴聻氓聭搂碌忙路聡氓漏聡氓娄露茅聧聸忙芦拢莽录聜盲陆潞氓搂鲁茅聧聴忙聤陆氓录禄茅聤聢氓聴聵氓聫聫茅聳鹿茅聜娄氓聟聵茅聧聥忙聮禄氓录陆茅隆聯莽聜碌氓漏碌莽聤虏茂鹿陇茅聬聴氓漏聡忙聜隆茅聬聰茅聲聬莽鹿聠忙陇陇忙聽篓莽虏聦茅聧聰氓忙聼聧莽录聛忙篓禄莽茅聳潞氓虏聥莽露聛茅陋聻忙篓潞莽聲路茅聧楼脩聟氓聨陆茅聴聜盲陆潞莽虏聨录芒聲聟氓麓聭忙驴聽氓聜聹忙職娄茅聳赂忙麓聳忙聝聼茅聴聛忙聦聨忙垄鹿茅聧聨氓聠虏茅聨露猫聤楼莽聟聼茅聳禄忙聞卢茅隆拢猫聦卢芒聜聟聣氓聛聯莽录聛氓卢忙聽篓莽鹿聸氓庐赂莽掳禄莽聙鹿氓聭聤莽鹿聜茅聢鹿忙聢娄忙聜漏茅聨禄忙聨聺盲禄卤氓漏碌芒聞聝氓陆聝茅聬聴忙聮鲁氓篓虏莽茅聨录忙路卢氓聻禄茅聢鹿忙露職莽麓聯忙碌拢猫聴聣猫聰录忙驴聻氓陇聥忙麓聼氓庐聲忙掳露氓鹿聮氓娄陇氓漏聝忙聶拢茅聴聛莽禄聵忙拢聝茅隆楼忙聮露莽虏拢茅聬聬氓莽炉聙茅聳禄茫聢陇忙陆搂忙碌聽茂鹿聜忙聜陆茅隆聳忙禄聝盲禄卤莽聙鹿忙聸聼忙麓聼忙驴隆猫聢碌氓戮聞忙驴聽氓聥碌氓聰碌茅聴聜盲陆赂忙聠隆忙赂職忙氓麓鹿茅聨露猫聤楼氓聞聴忙驴聻氓聴聵忙聦禄茅聬聯忙卢聭莽麓聮莽聙拢茅聳鹿莽聶赂莽碌驴茅聨虏忙聬聛氓录露茅聧搂忙楼聙氓录掳氓漏碌茂鹿娄莽禄庐氓鹿聫茅聧聸氓颅聵氓聛隆茅聴聛莽庐聭氓篓聢忙驴聢莽麓聯茅聧聦忙卢聭忙陇陇忙聢聺忙氓庐聲氓陆聮氓麓鹿茅氓聭茅聴聜氓聜職莽聣聤莽禄聥忙聮禄氓录鲁?train/val 氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄茅聳禄忙聞聬芦茅聧篓氓聜聻忙聼聸茅茂陆聠莽莽录聛莽聤鲁氓陆聝茅聹聣茅聳驴忙驴聠忙麓篓茅聤聠氓漏聡忙聼隆莽聙拢脨搂茅聳潞氓卤禄氓聤聦茅聢鹿忙聢娄氓麓卤茅聢潞氓聜聜茂鹿聜忙垄潞莽录聛忙篓禄氓掳掳茅聳驴忙聸聵氓聻陆氓氓漏職氓聻戮茅聨掳盲陆赂忙聜聭茅聴聛忙聬聻氓聞庐茅聫聟忙禄聟莽麓聻忙赂職芒聜氓漏職氓麓聮氓篓聠忙聢颅忙搂聴聛氓聭聤猫聣戮茅隆聭氓莽聧聧聬莽拢录忙驴庐忙篓驴氓聛聬茅聧聰莽聝陆忙聜路氓氓聥聳潞盲陆鹿氓聤聶茅聨卢聯氓麓聧聲忙陆聣茅聴聜盲陆赂氓茅聧聥氓聴聸氓麓掳茅聨戮猫炉虏氓聜篓茅聴聛莽搂碌茅聢路忙聢娄忙聼拢茅聬聰莽聟聨盲潞掳茅聳赂茫聜聡氓聳聳茅聫聦忙露職芒聜卢茂陆聨氓聛聯茅聭掳氓聺聴忙聼聼茅聯聫莽聼聮忙陇陇茫聢隆忙氓庐聲氓楼赂忙聜垄茅聧聺氓聥卢職?
    """
    split_rules = copy.deepcopy(_load_yaml_config_cached(str((ROOT / "scripts" / "default_split_rules.yaml").resolve())))  
    dataset_manifest = prepare_result.metadata["dataset_manifest"]  
    raw_root = Path(dataset_manifest["data_root"])  
    if prepare_result.metadata["dataset_manifest"].get("sequences") and any(  
        str(record.get("scene_id") or "").startswith("S(") for record in dataset_manifest.get("sequences", [])  
    ):  
        split_rules = _build_sim_paper_split_rules(raw_root, dataset_manifest)  
    split_manifest, leak_report = build_splits(  
        dataset_manifest,  
        split_rules,  
    )  
    if not leak_report["is_clean"]:  
        leak_items = list(leak_report.get("leak_items") or [])
        sim_e9_scope = _resolve_paper_scope(_ACTIVE_PAPER_SCOPE) == "sim_e9_only"
        only_scene_id_overlap = bool(leak_items) and all(
            str(item.get("kind") or "") == "shared_record_scope"
            and str(item.get("scope_field") or "") == "scene_id"
            for item in leak_items
        )
        if sim_e9_scope and only_scene_id_overlap:
            leak_report = dict(leak_report)
            leak_report["accepted_by_scope"] = "sim_e9_only"
            leak_report["accepted_reason"] = (
                "sim_e9_only accepts scene_id overlap because the synthetic e9 variants share one scene_id namespace "
                "while this narrowed run skips held-out/public comparison claims"
            )
        else:
            _write_json(output_root / "split_manifest.json", split_manifest)  
            _write_json(output_root / "leak_report.json", leak_report)  
            raise RuntimeError(f"split leak_report is not clean: {leak_report}")  
    _write_json(output_root / "split_manifest.json", split_manifest)  
    _write_json(output_root / "leak_report.json", leak_report)  
    if len(split_manifest["train_ids"]) < 1 or len(split_manifest["val_ids"]) < 1:  
        raise RuntimeError(f"split_manifest must expose non-empty train_ids and val_ids: {split_manifest}")  
    return split_manifest, leak_report  


def _build_run_seed_manifest(base_seed: int) -> dict[str, Any]:
    """茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聫茅聢搂莽聞娄莽禄禄忙驴聥氓麓聮氓娄陇盲陆潞盲潞聶忙驴聽莽聰碌氓聛聻茅聧篓氓麓鹿氓篓虏莽聳戮忙驴聽茅聺聸茅聬聯茫聝娄莽鹿聺茅聴聢忙露聶莽聙鹿氓聭聤莽鹿聢茅聫聦忙露聵脦聰忙碌拢莽篓驴忙聭碌茅聴聛氓聯聞氓陇聡茅聧聯忙驴聠茅聧聲氓聛聯茅聰聺氓潞聺莽掳陆氓篓麓忙禄聝莽聢录氓忙麓麓芦茅聧聲茅聺聸氓聣掳茅聴聜盲陆赂氓潞聹莽禄虏盲禄聥茂陆?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜麓莽聣聽茅隆垄忙聸垄忙聲聝茅聢搂氓茅聧娄莽拢录茅聬聨茂陆聨氓聛聯莽禄卤芒聲聜莽鹿聸氓庐赂莽掳禄茅聳禄忙聮麓莽聸聫聦忙露垄氓鹿聡茅聤聤茂赂陆莽聫聰茅聳鹿氓聟录忙陆聶茅聰聲氓鹿聝氓娄陇氓聭炉氓碌聛茅聳赂氓聳聳忙驴庐氓潞隆芒聰聭茅聬聬茅氓職聥莽录聛莽聤鲁忙聦赂茅聬拢莽聝陆氓鹿聥氓氓聳聯忙碌聹忙聬聛氓录陆猫陇聦忙戮鹿忙聸垄忙垄潞莽禄聥猫路篓莽庐掳忙陇陇茂赂禄氓聻卤莽禄聜氓露聠茫聜拢茅聧聬芒聰戮忙聥潞茅聴聛氓聭聤莽篓聭茅聰聲茫聝搂莽虏聢莽聙拢氓聫聫氓漏碌莽聤虏氓聲聤聡忙驴聡茅聬聬氓炉赂氓垄碌茅聳鹿莽聭漏茅陋聻忙聮禄氓鹿聮茅聨麓莽露聯茅聴聜氓聜職氓聙路茅聭鲁氓聽聲氓鹿聤茅聨戮猫猫搂娄茅聬聨茂陆聨氓聳聳莽潞戮氓忙聼聧茅聧聺氓聥聶潞茅聳禄忙聮露氓聳聳茅聫聦莽聠录忙聡聹茅隆聮芒聜卢忙驴隆盲禄聥忙聼拢猫鹿聡忙聮鲁忙職拢忙驴漏聟忙芦拢茅聨路莽聤禄氓鹿聯茅聨潞忙聢聻莽聵拢莽录聜氓聜職氓聙赂莽禄聣忙聺聭脦聸氓漏碌氓聴聶忙職娄茅聳鹿猫戮戮莽庐聭氓忙驴聥忙聜聴氓篓聭忙忙掳露氓鹿聬茅聧聬氓聙碌忙莽聜虏莽聛聺茅聧聰忙掳颅茫聛聞茅聰聰氓聻碌聠茅聳鹿莽鹿聺茅聨聢陇氓麓聴茅聭虏芒聲聛氓赂戮氓漏碌莽聤碌忙聲赂茅聧聥忙露垄忙聜聵氓漏碌氓莽陇?茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼忙聲聥莽聭聹氓驴聸氓聼聙茅隆聮盲陆潞芒聳聯茅聳潞氓聭鹿聞茅聧聮忙聨聭芒聳聨茅聨戮氓麓聡氓漏碌脗掳氓聙聬茅聧聥忙聮露莽麓聮茅聢搂氓戮聞茅聨掳氓聨赂茅聴聛忙聬聻氓聞聧聨茅聛聧忙聜聵茅聢潞氓聭庐莽聟聼茅聳碌氓聽聺茅氓篓聟氓露聣忙聼隆氓庐聙芒聜聧聽忙聽聧庐茅聨麓茂鹿聙莽庐聸茅聴聜氓聽聼莽篓聮茅隆聰氓聥颅莽麓聯茅聧聦忙卢聮氓聟聦氓漏碌莽禄卤莽聜麓莽卢聼茅聢搂忙驴聽忙掳聛聞茅聳赂氓驴職氓聛聬茅聧聰莽聝陆忙垄潞茅聨赂氓潞垄莽拢碌茅聳赂氓卢聦禄茅隆篓茫聝漏忙聜聶茅聬隆茅聰聥氓聻驴忙驴隆莽聰赂氓麓聼茅氓聣搂忙聺赂茅聴聛氓聯聞氓聲聽茅聧聥忙聮禄忙聼聸忙驴聽氓聠篓氓搂聧茅聳潞氓虏聥莽麓聲忙碌聽茂鹿聜氓麓聹猫氓娄虏氓聽楼忙聼搂莽录聛忙篓录氓垄陋茅聴聡氓忙聼篓氓漏碌氓聴聵氓聶潞茅聬聽忙聞露莽鹿聺茅聫聦氓露聡忙聠麓茅聧聲芒聜卢忙聨聰莽麓聮忙聺聢茫聞娄氓掳掳茅聳鹿氓鲁掳忙聡聫莽禄聜忙聨聰芒聜陆聨盲潞聺茅聨鲁忙卢聭莽麓聯茅聧聦忙卢聬莽楼掳忙陇陇忙聸聠氓聸篓氓聛聥茅聳赂芒聞聝盲戮聴茅聫聛氓忙聼篓氓漏碌氓聴漏忙搂赂茅聳陆氓聠陋氓聳聳茅聫聦忙露聵芒聵聣茅聧聴莽聜虏莽庐掳氓漏碌芒聞聝氓陆聝茅聬聴氓漏聡氓鹿聢茅聤聤茫聝陆篓茅聳赂氓聻禄猫聯卤茅聴聜盲陆赂忙聬芦茅聨鲁氓驴聸忙聜路茅聤聣芒聲聛莽聟聵茅聳鹿猫戮戮茅聶聞茅聧聥忙聞聛聞茅聫聝氓聸搂盲潞鹿氓篓麓莽聟聨忙麓陇茅聢鹿忙聢娄忙聜聶氓脨搂莽录聜盲陆赂茅聨赂茅鹿聝氓麓職茅聨潞忙楼聙莽卤聧茅聳赂莽鹿聢茅聧聻氓聽聲忙垄潞茅聧聺氓聥卢氓路聺氓漏碌莽禄漏聡忙聲聝茅聧聦忙露聮忙聥潞茅聴聛莽氓聛聬茅聧聥忙露垄氓麓聺氓漏聤氓聽聼氓聛篓忙陇陇忙聽篓莽虏聦茅聫聥忙聴聜莽麓聮茅忙卢戮氓職聶茅聳鲁猫聴聣茅聢禄氓潞篓氓鹿聡茅芒聜驴隆忙聤陆忙垄禄忙赂職芒聜卢氓篓录脩聡芒聙娄茅聧芦茫聝楼莽拢驴茅聳潞氓卤禄氓聞虏茅聧聥忙聼聸莽聛聻氓聣搂茫聙聮莽录聛芒聶聙氓聛聯茅聴聜盲陆鹿莽聹鹿茅聧篓氓氓录聧聺茫聝娄氓篓麓莽聟聨莽聯篓茅聢?茅聴?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    base_seed : int
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜麓莽聣聽茅隆垄忙聸垄忙聲聜茅聮聵氓莽陆聲茅聴聜盲陆赂忙聠隆茅聧聰茂鹿聜氓麓掳茅聫聧氓漏碌莽聤鲁莽垄聧茅聬聯忙卢聯忙聼聸茅聨戮忙忙聡聨莽禄聴茂驴聽忙垄潞莽录聛忙篓禄氓掳掳忙驴聻氓聫聣脡聭莽禄聦猫戮篓莽鹿聺茅聧楼猫聢碌忙聶漏碌莽聤虏茂鹿陇茅聧聼莽氓搂聴芒聜卢氓搂?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓聙驴茅聳驴忙聸聻氓聙鹿茅聧聨忙聢聽忙聠赂茅聬聴氓聽聺莽卢聮莽录聛芒聜戮露氓卤赂氓聛聯茅聧聫氓盲潞戮茅聴聛茅聙聻氓卤陆氓垄麓茅聳赂芒聰戮氓聛聯氓娄聻茫聜聠氓赂聤茅聭鲁氓聽聲茫聙聜茅聫聦茫聝娄莽聟聲茅聬聨茂陆聨盲潞聹茅忙聼隆莽聛聻莽聲聦脨楼忙陇陇茂鹿聙莽聯篓忙路聡氓漏聡茅茫聢禄颅茅聴聛茅聺聸忙麓陇莽聙職盲录麓莽聧娄炉氓掳聳氓聛聯茅聬垄莽聰碌茫聙聤忙驴隆忙聫聮莽聣聤莽禄禄忙露職芒聜露聶茅聬漏聜猫陇聫茅聫聟脦拢茅聨掳忙聺驴莽庐禄茅聬聨忙露聶脙忙聮禄忙垄潞茅聧聸氓漏聞莽录職茅聳赂氓潞垄氓篓聤茅聨虏聞氓鹿聵莽录聛忙篓录氓聨碌莽录聜氓聜聶茅聶聧茅聧篓氓陆聮氓录赂茅聬聰氓聟录莽聟聲茅聬聨茂陆聝氓聤聦茅聢搂莽聣聹茅聨鹿茫聢聽芒聵聣茅聤聫莽聤虏猫聙驴氓漏碌芒聵聠氓聻碌茅隆聲脩聟氓聳聤氓庐楼氓聸漏莽拢录茅聳禄忙聞碌氓聣聶茅聧聰茫聜聠莽鹿聺茅聢搂忙陆聫茅聢潞氓聜陇忙陆聼茅聴聛氓聹颅氓聞陇茅聧陇茂鹿聜氓录庐茅聧驴聢茅聴聛氓庐聽茅聧聨猫聶芦莽娄聧忙楼拢氓聠漏莽聟聝莽聭聹忙禄聢氓麓聹茅聬聰莽聟聨猫聯聼?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    dict[str, Any]
        莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿氓氓卢碌路茅聬聴聴脙鹿茅聰聺氓聽聼莽禄戮氓聣搂氓聨搂茅隆颅猫路篓忙聧聞忙赂職氓聣卤茅聳禄氓驴聯茅陋聫莽禄聽忙聮禄氓录禄茅聬聰莽垄卤忙聲聤茅聳赂氓驴聯氓聬聥茅聧聨忙聮禄忙垄潞莽聙鹿莽聥聙氓職聶忙驴庐忙漏聢氓莽禄芒聵聣氓茂陆聟盲潞戮茅聳驴忙露聵氓麓聧聼芒聲聜莽芦聼茅聫聡茫聢隆芒聮聭茅聳赂茂赂禄氓聨录茅聧聰氓茫聛聞茅聤聤茫聝娄氓聶鹿忙陇陇忙聳驴莽鹿聬茅聢鹿忙聢娄氓麓露茅聤聣茫聜聡莽聸聨茅聴聜盲陆陆氓庐聲忙篓潞芒聳聰氓篓聭忙卢聭莽聭聹忙聳驴忙驴聽忙掳卢莽拢录忙驴聻氓聴聵氓聻碌忙驴隆盲禄聥忙聼拢忙聬麓茂陆聡忙聡聴茅聳赂忙露卤氓聻鲁茅聬聯忙聮禄忙垄潞莽潞聭庐氓陆搂茅聴聜氓聥氓庐聲莽聰赂氓鹿聡茅聬聴氓聽垄氓聨聯茅聬聼忙禄聞莽虏聯氓庐聲忙禄聝芒聳聨茅聨戮忙聺聭芒聜聢碌忙聟篓氓娄炉氓聻聟聬茅聳潞氓聭庐莽鹿聢茅聫聦忙聸隆莽庐聸茅聫聡莽聜聬茫聝漏忙聼隆?``base_seed``茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮莽聲聦忙聠赂氓庐楼氓聽垄盲潞卤茅聴聜盲陆赂忙聬芦茅聧聤氓陆聮氓篓聠氓驴拢莽聞陆茅聳潞氓露聝氓聤聧茅聧聶氓驴聯芒聰聭茅聬聵忙聫聮芒聜卢氓漏職氓聼聻茅聨虏芒聜卢obal_seed_report`` 茅聴?
        ``locked_steps``茅聴?
    """
    return {  
        "base_seed": int(base_seed),  
        "global_seed_report": _lock_seed(base_seed),  
        "locked_steps": {  
            "scene_sampling": {"seed": int(base_seed)},  
            "search": {"seed": int(base_seed)},  
            "training": {"seed": int(base_seed)},  
        },  
    }  


def _load_training_payload(prepare_root: Path, raw_root: Path) -> tuple[list[str], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聨禄忙聨聰芒聜卢忙垄禄茅聧聦忙掳卤氓茅聳?茅聴?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聵茅聧聤氓聺聴忙聼聸氓篓聭忙漏聢忙聰禄茅聳赂忙卢聫莽鹿聬茅聢鹿忙聢娄忙聜漏茅聨禄忙聨聠氓聛聯茅聧聸聻莽拢禄茅聳碌氓驴聥氓聙聳茅聬聯?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳鹿茅聴麓莽莽潞戮氓聹颅莽麓聮莽聙拢莽麓漏忙驴聻氓路庐盲潞聺茅聧聥忙聞掳忙聜鹿茅聧楼莽職潞茅聳驴忙露聶莽聸忙麓聜路茅聫聣氓潞聯忙聦聝莽录聜盲戮聡氓聶庐茅聧篓猫路潞莽聲路茅聨麓茂赂陆忙聶赂茅聳禄忙篓潞莽拢聧聵忙聬聛忙垄潞茅聨录莽聜虏氓聤聵茅聳赂忙聳驴莽禄聜氓露聣氓茅聤聢氓聴聶氓聨卤茅聴聤忙麓娄茅聨赂猫炉虏茂鹿垄茅聨碌莽拢录莽录聜盲陆鹿氓篓虏氓芒聜忙麓聳茅聤聢忙聬聛氓鹿聝茅聤聫芒聵聠忙搂鹿茅聨录氓颅聵莽鹿聝茅隆垄忙漏聙忙垄禄茅聧聦忙卢聬莽楼掳忙陇陇忙聸聼莽聣聽氓庐聲氓陆聮氓麓隆茅聬聬氓露聠莽篓聮茅聨路氓聭麓忙聬鹿茅聳露職莽聭聺忙驴聽莽聰碌氓聛聻茅聧篓氓麓鹿茅聧娄盲赂聧茅聳录氓氓聙聬茅聧聥忙聮露忙聠麓茅聧聲氓漏碌莽聜露莽禄虏氓聺聴氓麓職茅聨潞忙莽庐禄忙陇陇忙聴聡脨聮茅聨路莽禄聵莽鹿聸茅聨戮忙聺聭莽聞鹿茅聳赂氓卢聦禄莽禄禄忙露聶忙聥聟茅聧聬忙聫聮忙聜漏忙驴聻忙聢聶氓聻鹿氓娄芦忙篓禄莽鹿聸茅聧隆忙卢聫盲潞戮茅聬聨茅聺聸莽聼聜氓搂聤忙麓陋氓麓路茅莽聜虏氓氓漏碌芒聵聠氓聛聜茅聬聮茂鹿聜氓鹿聢茅聤聤芒聲聛莽庐聳录忙聛聮氓聞鲁茅聧聵莽鹿聺茅聬垄茅聺聸脨垄莽录聛莽禄聵氓聤聶茅聤聠氓聭麓茅聧聲忙聥禄茅聴聛莽篓驴忙聹卢茅聬聰氓麓聺茅聬聰氓聟录莽聟聫茅聳赂氓聳聬莽聛聫茂鹿聜忙聼隆莽聛聻氓聣搂茫聙聮茅聳鲁脩聦氓聣篓莽录聧氓聴聸氓麓聭茅聧聸莽聞聤氓篓麓莽聟聨莽聯篓茅聬聯忙卢聭芒聜卢莽聭掳氓忙戮搂忙氓麓碌茅聧楼莽聟聶忙陇陇忙聴聜忙聶芦茅聬漏聡忙聭露茅聰聺氓陇聥莽聟拢茅聼聸篓忙麓聵茅聧陇芒聜驴聽忙庐驴氓聶拢忙碌聹猫路潞忙驴聽忙掳卢莽拢录忙驴聻氓聴聵氓赂聮茅聧聵芒聙鲁茅聤聢氓潞隆氓鹿聳茅聳禄忙忙聧聡氓氓麓聵茅猫聤楼忙芦聡茅聴聛莽忙聹聭鹿茅鹿聝氓麓聭氓庐楼氓氓聛隆忙驴聽氓漏聜氓茅隆楼氓聽楼忙聼拢氓篓聭氓聧聻忙聻娄莽录聛莽聤鲁莽聸芒聳聨茅聨炉脩聞茅聴聜氓聜聶莽陇聛茅聨录脣聡忙碌录忙聺聭氓聻聜茅聬聼忙卢聫氓氓篓聢盲禄聥忙聼聸茅聤聣氓垄聳鲁茅聰聥氓聻驴氓搂聤氓氓麓录茅聬聰氓聣聛莽鹿聛氓漏碌茂录聞氓芦聫茅聧聬茂陆聠忙聳聙茅聴聛莽聜陆氓聞麓氓篓聟忙聸垄氓麓掳氓搂聴芒聜卢茅聫聦莽聠录茅聧聺氓聥芦氓搂娄茅聴聛氓聯聞猫聥炉氓娄忙驴聜莽氓鹿垄忙驴聻氓聴聴莽鹿聬莽禄聽茫聝漏忙垄禄茅聧聦忙掳聜卢忙聬聛芒聙娄茅氓聥忙驴氓聛篓忙碌聹氓陆聮忙聼聸茅聨掳茅聺聸忙聻聸莽录聛芒聜戮露氓卢聫聫忙驴隆莽聛聺茅聬聯忙聞掳忙聼拢茅聨戮氓炉赂忙聡聡茅聳鹿莽露聤氓庐聲氓聽聲氓麓露猫茅聧拢盲禄聥忙聼拢茅聰聲芒聜聬聴忙聮禄氓鹿聝氓娄陇氓聠漏氓聳聠茅聳赂忙聸篓氓聣聸茅隆娄芒聰聭茅聬聬氓路庐忙搂鹿茅聳禄芒聲聤氓聻露氓庐聲忙麓陋氓娄路茅聰聲芒聜禄露莽聜虏芒聰聭茅聬聵忙聫聮茅聳赂氓露聡忙聲禄氓茅赂驴茫聛聬茅聨潞忙驴聡忙庐楼茅聴聛莽禄聵氓赂陋莽禄聽忙聮禄氓鹿聫茅聬聭忙篓潞莽路聻茅聳鹿茅聜娄氓聨赂茅聨路忙麓垄潞茅聧聸氓漏聞莽庐聯茅聬聨忙掳聵虏氓聧卤茅聳鹿茅聴麓莽聞娄茅聬聯莽聠赂盲驴聤茅聤聢氓聜職盲潞戮茅聴聛忙聦聨忙麓娄忙碌聹忙聜掳茅芒聜陋聻氓聸录露莽聮潞氓聲聡氓漏碌莽聜露忙聫禄虏盲禄聥氓鹿聴氓漏聤氓聽聼氓聛聜茅聳鹿猫戮戮茅聶聞茅聢路忙聢聽莽麓聮茅忙露聶莽氓漏垄脩聠氓掳聶莽禄卤忙聨聰芒聜陆聨氓聛聞茅聬聫猫戮漏莽麓聮?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    prepare_root : Path
        prepare 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳鹿茅聴麓莽莽潞戮氓陆聮忙聼聼茅聴聜氓炉赂茅聳禄茫聜聨氓聶拢茅聫聦莽聠录氓鹿聠茅聬聴氓聽茅聴?氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄忙驴聻忙聢聶氓聻潞茅聧聥茫聜聣忙聼聲忙驴聻脩聝忙聡聨茅聫聠忙楼聟忙垄潞莽禄聥猫路篓莽聛卤茅聳赂氓卢篓聭茅隆聰氓驴聯芒聰聭莽聙拢氓聨碌莽录聜盲陆赂莽聙碌脩聡氓麓碌茅聧楼茫聞娄氓聫?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳赂忙驴聞氓聞鲁茅聬聮禄忙聜聴茅聴聢忙露聶忙聠赂茅聳禄忙聧聡茅聨鲁掳芒聳聨猫鹿聡忙聯聝盲禄鹿莽录?
    raw_root : Path
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏氓庐聲氓聸搂氓鹿聢?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬氓庐聞忙聡聤氓陆聮氓麓露猫陇聹氓篓麓忙聽聼聲忙驴聻莽聜卢忙芦聠茅聳赂茫聜聟氓茅聫聦茫聜聡莽虏聶莽聮潞茫聞陇莽虏碌氓漏碌莽聜虏忙聡聬忙驴庐氓聻漏卤忙驴聽氓聸搂脦搂茅聳赂氓卤戮莽聼戮鹿忙聸垄忙垄禄忙碌拢茅盲赂虏莽录聛猫鹿虏聶茅聨篓潞忙聺聢芒聲聝莽聲露茅聴聛莽禄聵忙垄禄茅聧聵脩聡忙聜聻茅聧篓盲潞聹茅聳鹿莽聝聵氓聻碌茅芒聲聝莽麓聮?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳赂忙驴聞氓聞鲁茅聬聮禄忙聜聴茅聴聢忙露聶忙聠赂茅聳禄忙聧聡茅聨鲁掳芒聳聨猫鹿聡忙聯聝盲禄鹿莽录?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    tuple[list[str], dict[str, Any], dict[str, Any], dict[str, Any]]
        ``(seq_ids, events_by_seq_id, ground_truth_by_seq_id, source_report_by_seq_id)``茅聴?
    """
    prepare_manifest = _load_prepare_manifest_cached(str(prepare_root.resolve()))  
    seq_ids = resolve_seq_ids_from_prepare_manifest(prepare_manifest)  
    _, events_by_seq_id, ground_truth_by_seq_id, source_report_by_seq_id = _load_scoring_inputs_cached(  
        str(prepare_root.resolve()),  
        str(raw_root.resolve()),  
        tuple(seq_ids),  
    )  
    return (  
        seq_ids,  
        copy.deepcopy(events_by_seq_id),  
        copy.deepcopy(ground_truth_by_seq_id),  
        copy.deepcopy(source_report_by_seq_id),  
    )  


def _resolve_train_cfg(model_name: str, *, epochs: int, batch_size: int, eval_batch_size: int) -> dict[str, Any]:
    """茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聟忙職娄茅聳赂氓聛聟茅聧聶氓驴聸忙聼隆忙戮露氓忙庐驴聻氓路潞聹茅聬聯忙露垄忙聼聸氓芒聜聬聮茂鹿聜氓录虏茅隆聫氓聽聼氓聛隆忙驴聽氓漏聜氓茅聬聫忙聺聭芒聰聭茅聰聸氓聥職卢忙楼聽莽聜虏忙麓氓聣聛忙聧聞茅聤聤茫聝垄氓聛聯茅聬聬莽聝聵芒聮聭莽录聜盲陆鹿脡聭茅聢路忙聨聴忙聼聸氓娄炉莽聤虏忙麓娄茅聧聤氓聣聛芒聜忙麓聳氓篓虏莽聤禄氓麓聭茅聨戮猫聢碌氓聳聠茅聳赂忙聸篓氓聣聸茅隆娄茫聝楼芒聰聭茅聬聬猫路潞莽職潞氓漏碌莽聜漏氓聛聯茅聬聨猫搂聞忙麓聵茅聧篓氓聺聴莽聧娄炉猫聜漏莽拢录忙驴隆茫聙聙茅聧聥忙聮麓莽鹿聺氓氓聨戮茅聴聛猫炉隆氓聨陆莽聰聨聴氓麓聺茅聤聢氓陇聥莽聟聲忙驴隆忙驴庐氓露聣忙聼聼茅隆聰莽颅聥忙庐聹茅聳禄忙露卤氓聶拢氓庐聲氓陆聮茅聬聯氓潞庐忙陆聸氓漏碌莽聤碌忙聲赂茅聧聥忙露垄忙聝聡忙碌录忙聺聭莽茅聳鹿氓颅聵莽鹿聧茅聧陇忙聸垄忙聼聼茅聴聜氓莽禄戮忙聝搂氓聬聥莽禄禄忙露垄氓鹿聥氓篓聠氓驴聲盲禄戮茅聴聛莽篓驴茂鹿娄茅聧聫忙漏聙氓鹿聝莽聭聶氓聥漏氓聛聤茅聳鹿莽氓炉聮忙碌聽芒聲聟忙垄潞茅聧聸氓盲戮聸茅聳赂氓卢聡聬茅聨鹿茫聢聽芒聰聭茅聧芦忙驴聡莽聙鲁氓漏碌芒聵聠氓聻掳茅聧垄氓聫聣莽娄聧忙楼拢氓聠漏莽聟拢茅聼聸路莽碌陆忙碌聹忙驴聥忙聼聸茅聬聵氓聠虏氓搂聧忙驴庐?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聵茅聧聤氓聽聲忙垄潞茅隆聮芒聜卢莽禄聣莽聰赂忙聜隆氓篓聠氓驴聤忙聰陆茅聳赂氓卤戮莽垄聧茅聨虏忙聬聛忙聼拢茅聰聺氓聽聼茅聳鲁脩聦氓聣聺茅隆脙潞猫鹿聡忙露聵莽陇聣茅聳鹿猫戮戮猫麓聼茅聢搂莽茅隆禄聢氓聼聙茅隆聮氓聥芦莽庐聳忙驴聻氓聴聵忙聦赂莽禄聽莽禄聵茫聛聞茅聫聝氓聸聠忙聨聴氓鹿聫氓篓虏莽聟聸氓猫路潞猫聥炉忙碌聽茂鹿聙茅聬聼莽聣聢莽聙鹿忙聸聼莽鹿聜茅氓楼赂氓麓篓茅隆聫氓聭庐忙聺?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聳莽潞戮氓聺聴忙聼拢茅聤聫茫聢漏莽聞戮茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛莽篓驴茅氓聴聵莽漏卤忙驴聽氓聸搂脦搂茅聳赂忙露卤氓聳聳氓篓聟茂陆聣忙垄潞茅聨录莽聜虏氓娄录茅聳赂氓漏聜莽露聤茅陋聻氓聽楼氓娄赂茅聤聣氓潞拢氓聳聳茅陋聻忙聞颅忙聝聰茅聰聺氓聠拢氓聛聯茅聨掳莽聣聢莽聟??茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝莽录聜盲陆虏氓聛?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳鹿茅聴麓莽莽潞戮氓聹颅莽麓聮莽聙拢莽麓漏忙驴聻氓路庐盲潞聺茅聧聥忙聞掳忙聜鹿茅聧楼莽職潞茅聳驴忙露聶莽聸忙麓聜路茅聫聣氓潞聯忙聦聝莽录聜盲戮聡氓聶庐茅聧篓猫路潞莽聲路茅聨麓茂赂陆忙聶赂茅聳禄忙篓潞莽拢聧聵忙聬聛忙垄潞茅聨录莽聜虏氓聤聵茅聳赂忙聳驴莽禄聜氓露聣氓茅聤聢氓聴聶氓聨卤茅聴聤忙麓娄茅聨赂猫炉虏茂鹿垄茅聨碌莽拢录莽录聜盲陆鹿氓篓虏氓芒聜忙麓聳茅聤聢忙聬聛氓鹿聝茅聤聫芒聵聠忙搂鹿茅聨录氓颅聵莽鹿聞莽禄聙氓陇聬忙垄潞莽聮聡氓聫聣氓聰聣忙陇陇莽聟陇忙驴庐忙陇禄氓聛聯莽聙鹿忙聸聼莽鹿聜猫聻聳茅聳赂忙聞漏氓聤聦茅聬聫氓漏聜茅聤聢氓聴聵氓掳碌氓漏碌忙聲禄氓庐聲忙聢聺茅聢搂氓录禄氓庐楼氓脦篓茅聳碌氓氓聞卤莽禄聴莽鹿?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    model_name : str
        忙驴聽莽聰碌氓搂路茅聫聛氓聭聤忙聟篓茅聬聭氓芒聜聜聻忙聦聤忙戮鹿芒聲聥莽露聛氓庐聲茫聞漏氓录露茅聨麓忙驴聢芒聜卢茅聤聢氓聭炉莽庐掳茅聳禄忙楼聙忙拢聺茅聨庐莽聜潞忙聜陆茅聬垄氓聻芦茂录聫莽录聜氓聜聶茅聶聧茅聧篓莽聭掳茅隆聲芒聜卢茅聫聦忙露垄氓聼聻茅聨炉脩聜氓拢聲茅聴聜氓聜聶猫聝露茅隆聽垄忙聧垄茅聨麓茂赂陆忙聶陋茅聴聜盲陆路莽麓禄虏氓聺聴氓麓聴氓娄莽鹿聠茅聳禄忙聢拢忙氓聰聴聛氓庐聽忙隆篓茅聭鲁氓聽聲氓麓聬茅聬聬莽聝聵芒聮聭莽录聛忙麓聳忙戮搂忙聼楼忙聼拢茅聬聰忙聺聭氓聤聹莽录聛氓聜聸氓赂隆氓庐聲茂驴聽忙聜聶茅聢潞氓聜聸忙聺?
    epochs : int
        茅聴?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓聙驴茅聳驴忙聸聴氓聛聯茅聳驴忙聸聵氓聻陆茅聨庐脩聡氓麓录茅聤聫茫聢漏氓聲聨?茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅茅聢露猫聤楼莽篓聭茅聢陆氓陇聬氓茅聬聵莽聰碌氓麓露茅聴聜盲陆赂忙聬芦莽禄聣忙聼楼氓麓聺莽聙拢氓麓虏?
    batch_size : int
        茅聴?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聵茅聧聤氓聺聴忙聼聸氓篓聭忙漏聠莽聟驴聻氓聽聹莽虏聶茅聫聦茂陆聣氓鹿聡茅忙掳卤忙茅聳潞盲陆赂莽聣聯氓猫聶鹿忙聥聽茅聢楼猫聶芦莽聛聯茅聴聛莽篓驴莽鹿聭茅聰聲茫聢陇莽聧驴聽氓聜陇莽聟楼茅聳赂氓驴聲氓搂麓莽录聧氓漏聡氓鹿聝氓漏聤氓聽聼氓職聧茅聳碌氓拢聠篓莽庐潞茅聴聜氓聜職茅聧聥氓聥潞茅聰聲氓聜職莽茅聳赂忙聮聴氓聭卢聣茅聴聛猫茅聧聽忙掳卤氓拢聲忙驴聜氓聭聤氓聛拢茅聳赂茫聝庐盲潞聹茅聬篓氓聯聞猫莽禄聽忙聮禄氓录禄氓篓聭茫聢聽莽聟聵茅聳鹿氓聜職?
    eval_batch_size : int
        茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聟忙職娄茅聳赂氓聛聟茅聧聶氓驴聸忙聼隆忙戮露氓忙庐驴聻氓路潞聹氓聸麓茅聴聛忙聬聻氓聞禄莽禄虏猫聤楼莽娄聧茅聬聬氓聫聫忙驴庐忙楼聙忙拢聴忙戮搂盲录麓忙聼拢茅聨戮忙聺聭莽聛娄莽录聛莽禄聵莽鹿聢茅聨禄聥忙拢聝氓篓麓猫潞虏氓聻潞莽禄禄忙露職盲禄娄茅聧聦忙掳拢氓聛聯氓娄陇氓聠篓氓聻聺氓漏碌莽聤鲁猫聣戮氓聰聴聛氓聠虏忙聬芦茅聧聤氓氓麓垄忙碌聽氓卢聮聭莽录聜盲陆鹿脡聭茅隆楼氓聴聵茫聛聞茅聫聡氓聴聴忙麓娄茅聧聤氓聺聴忙聼娄氓娄炉盲戮拢娄茅聳禄莽聠录氓聛聤茅聬聯茫聝漏忙聼聼茅聳鹿聙忙聻陇莽聬職茂鹿聜忙垄禄忙碌拢莽颅聥氓職聝茅聳赂莽聤鲁茅聢楼茂鹿聜氓麓录茅聤聫莽卢聣氓漏碌莽聜麓氓聻聧篓氓聹虏禄忙驴聽忙掳颅莽聟聲茅聳鹿忙聧聬氓掳陋茅聧聫氓虏聞忙聼隆忙戮露氓陇聦莽聼聶忙驴?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    dict[str, Any]
        茅聴?氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄茅聳禄忙聞驴掳忙碌聹氓陆聮忙聼聸茅聨掳莽聰碌氓聨聸茅聳禄忙聨聧莽虏聶茅聫聦茫聞漏忙聜垄茅聧聺氓聥聭茫聝搂芒聜卢猫搂聞忙聦路莽禄露忙掳露氓录禄茅聬聰氓聟录氓聙禄忙驴庐忙楼聙忙拢聶茅聬拢氓聽聲忙聼聸茅聬聭氓聴聴莽聛娄忙驴庐氓漏聟忙茅聢搂忙鹿卤忙驴聨掳莽聣聢莽聟聲茅聨录忙路卢氓聙聳茅聬聺莽麓聮茅聧聦忙掳卤脨搂忙莽聜虏忙茅聢楼忙聽颅氓聻戮氓庐聲氓聸漏脨篓茅聴聛猫陆掳莽聞娄茅聧聰忙卢聫莽聲路茅聧芦忙聸聻脦漏茅聳驴忙聴聙氓聞鲁氓篓聭脩聟莽麓聯茅聧聦忙掳聜卢莽聝陆忙聡聴茅聭赂氓聻芦氓聫聫茅聳鹿莽禄垄氓聳聮茅聢搂莽庐聯茅聨禄聢忙聦聤忙戮露氓虏聙忙職?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬氓庐聞忙聡聤氓陆聮氓麓露茅聢鹿忙聝搂盲潞戮忙驴庐忙漏聠氓聛聞氓庐聯氓聴聵盲驴聤茅茂录聞氓聤聧莽聙碌忙聺驴氓芦禄禄忙露垄氓鹿聮茅聨麓莽聤碌脨篓茅聴聛猫陆掳莽聞娄茅聧聰忙卢聫莽聲路茅聬聯氓潞聵忙搂聢茅聳鹿莽聝聵氓聻鲁忙戮鹿忙聸垄忙垄潞茅聳芦莽聜虏猫聥炉忙戮搂忙聮庐忙聼隆氓庐聙氓卢拢驴氓篓麓莽聥聟氓娄聡茅聨路莽聤虏脙鹿莽聙拢氓氓漏碌莽聤禄忙陆搂茅氓聭麓忙聥聭茅聬聰氓聟录莽聟楼忙驴聽茅聺聸忙拢颅氓娄虏忙聬聛氓鹿聠茅聬聰氓聟录芒聮聭茅聴聜氓聽聼盲戮聴氓娄虏氓聽聲忙聼聸忙聬麓茂陆聠忙職聙碌茅聫聛忙聞戮聞莽聙拢莽聫鲁茅聴聛莽氓聛聬莽聬聸茫聝漏氓聼聙茅隆聮盲陆鹿莽聛卤茅聳鹿氓楼聣氓聛聤茅聫聦茂陆聠忙聝聰茅聤聫芒聲聛忙聶卤茅聴聛氓聯楼氓搂碌莽聰炉芒聙鲁莽聲路茅聨麓茂鹿聙莽庐禄茅聳潞氓聜聵氓聞虏茅隆露聟芒聰聭茅聬聬氓聫聣茅聳潞氓聽聣聼茅聫聛氓露聠氓戮聞莽聙拢茅聴聛氓聹聛聯氓篓录忙卢聯忙聜聻猫陇掳忙聨聯莽聟聲茅聬聨茂陆聨氓聛聟莽聛聫莽聰赂忙聼聧猫陇聹茅聧聯忙驴聥茂陆聻忙戮露氓篓聮忙麓录陆猫陇聺茅聭掳氓聺聴忙垄潞茅隆聮芒聜卢莽禄聣忙聮庐忙聜聻茅聧篓盲潞聹茅聳鹿莽聝聵氓聼聤茅聧聰莽聠潞莽麓聮忙戮露氓卤戮忙職卢茅聳潞氓卤戮莽聸聨庐芒聲聛莽聲路茅聧楼脩聙忙聥隆茅聴聜盲陆陆忙隆篓莽禄露茂驴聽氓麓掳茅聫聧氓聛漏茅聳禄忙聢拢忙茅聬聬氓漏聡忙聼聧茅聧聺氓聥聻娄莽录?茅聴聜氓聜職氓聙赂茅聧聤莽聝陆忙聡聴茅聧聸聻莽拢驴茅聳赂氓聛聟茅聧聶氓驴聲芒聜卢氓鹿驴氓聞卤茅隆娄忙聜聶忙驴聽氓聥卢氓聳聬茅聼芦氓聸篓猫聸聥茅聧楼忙篓聞氓漏碌茂鹿篓氓篓聟茂陆聟芒聵聣茅聬垄莽篓驴氓路聺忙陇陇忙聮麓莽鹿聝茅聬芦氓驴聸忙垄禄忙碌拢盲戮炉莽聞戮忙陇陇忙聢聺忙氓庐聲忙驴聠芒聳聨猫鹿聡忙聸聼忙?
    """
    model_cfg = copy.deepcopy(  
        _load_yaml_config_cached(str((ROOT / "configs" / "models" / f"{model_name}.yaml").resolve()))  
    )  
    train_cfg = dict(model_cfg.get("train") or {})  
    train_cfg["epochs"] = int(epochs)  
    train_cfg["batch_size"] = int(batch_size)  
    train_cfg["eval_batch_size"] = int(eval_batch_size)  
    phase_schedule = dict(train_cfg.get("phase_schedule") or {})
    if phase_schedule:
        warmup_epochs = int(phase_schedule.get("warmup_epochs") or 0)
        gate_alignment_epochs = int(phase_schedule.get("gate_alignment_epochs") or 0)
        remaining_full_tuning_epochs = int(epochs) - warmup_epochs - gate_alignment_epochs
        if remaining_full_tuning_epochs < 0:
            raise ValueError(
                "requested epochs cannot be smaller than warmup_epochs + gate_alignment_epochs "
                f"for model {model_name!r}"
            )
        phase_schedule["full_tuning_epochs"] = remaining_full_tuning_epochs
        train_cfg["phase_schedule"] = phase_schedule
    patience_value = train_cfg.get("patience")
    if patience_value is None:
        train_cfg["patience"] = int(epochs)
    else:
        train_cfg["patience"] = min(int(patience_value), int(epochs))
    model_cfg["train"] = train_cfg  
    return model_cfg  


def _set_nested_cfg_value(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    """茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓麓茂陆聣忙陆禄茅聭掳氓聺聴氓录露茅聧芦忙露聵氓娄赂氓篓麓忙禄聞莽虏聯茅聫聦茫聞漏氓录麓茅聬聬忙碌聥氓聛聯茅聬聭猫聴聣莽庐聼茅聳赂忙聮聴芒聜聧聥忙聮露忙聠麓茅聧聲茅聴聛忙聬聻氓聤聦氓篓录芒聙鲁茅茂鹿聙莽庐聸忙陇陇忙聮露氓聛聼莽禄聬莽聜麓莽鹿聺茅聬垄茅聺聸脨垄茅聬聨忙掳录氓聶拢氓庐聲茫聢陇茅聧聲忙聥禄忙驴聻猫戮戮莽碌陆氓漏聙氓聧聻茂鹿垄忙碌聽氓莽聟聲氓漏碌莽聜虏莽聛聺茅聢搂莽鹿聜茅聬拢莽聳炉茫聛聬茅聨潞忙聢聽茅聴聛氓聟录盲潞陆茅聧聨猫戮篓莽娄聧氓氓聽聼芒聮聭茅聳赂忙露聵茂鹿拢莽禄露茅聛聧忙聼聸茅隆聢聹茅聜聙鹿忙聸聼氓聻驴茅陋聻忙篓录忙聥聽茅聨禄忙聨聰芒聜聤聢氓聴聴氓搂搂莽录聛忙聫聮忙陆隆氓庐聲忙卢聯忙聲聯茅聬聵氓麓聡氓聨陆茅聳鹿氓聟录莽聲陋茅聧篓氓聟录莽芦聼氓娄炉猫聜漏莽拢录茅聳禄忙篓潞莽拢戮搂莽麓聮莽录聛忙篓录莽庐聳莽录聛莽禄聵莽鹿聢氓庐聲忙聨聭茅莽聡聛氓聲聙茅聴聜氓聜聶莽陇聛茅聨录脦聸忙聬麓茫聞楼莽拢鹿忙驴聽茅聺聸莽禄聽?茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼忙聲聥莽聭聹氓驴聸氓聼聙茅隆聮盲陆潞芒聳聯茅聳潞氓聭鹿聞茅聧聮忙聨聭芒聳聨茅聨戮氓麓聡氓漏碌茂录聞氓聛聸茅聬篓莽聝陆氓麓聭茅聨戮猫炉虏莽忙陇陇忙聴聜氓聨搂莽禄聬忙露垄忙垄潞茅聧聺氓聥碌忙搂鹿茅聳赂茫聞篓氓茅隆陇氓聜聻忙垄禄茅聧聦忙卢聭莽陋聰茅聳鲁脩聦氓聣聸茅聧聥忙露職芒聜掳录氓聶拢氓庐聲茫聢陇忙聼聠氓庐楼氓聴聫忙聥潞茅聴聛茅聺漏氓炉聦茅聧聵氓楼赂茫聙聜忙驴聜氓聭聤莽聟聲茅聳驴忙露聵氓麓篓虏茂鹿聜氓聼聸茅聨麓茂赂陆芒聮聭忙陇陇忙聞漏氓聙聲忙碌聽忙禄聢忙聼陇氓篓虏猫莽聛隆茅聳潞氓聭掳氓聼聳莽聙碌猫聜漏芒聜露聶茅聧聵忙垄潞莽禄聸聻氓垄虏忙陇陇茫聞楼莽鹿聵氓庐聲忙聸垄氓鹿聥茅聬聬忙聣庐職茅聴聛氓聭聤莽聭楼茅隆娄忙聟篓茅聧楼莽聟聝茅聬聼忙卢聫氓茅聬聫芒聲聟忙聼聧猫茅聧聯茫聞搂莽麓聻茅聧隆忙露聵莽拢禄茅聳赂忙聸篓氓聣聸茅隆漏茅聰聥莽鹿聸氓庐赂莽掳录茅聳鲁茅聰聥氓聻驴氓氓漏職氓麓录茅聬聰氓聣聛莽鹿聛茅聴聛莽禄聵忙聬漏茅聧聶氓聠篓忙驴聽忙掳卢莽陇聥忙陇陇忙聞漏氓聼聞忙碌录氓聠篓芒聰聭茅莽隆路茅戮職忙驴隆氓茅隆垄忙漏聙氓麓聴茅聬聯氓潞拢莽陋聻茅聳禄氓潞炉莽露聠忙碌聹忙卢聯忙聜陆忙驴聽忙掳颅芒聮聮氓篓麓茂陆聟忙聠隆茅聨聽聼莽鹿聸莽聛聻氓聜職氓莽聙鹿忙聸聼氓聻驴茅聨庐忙卢聫脙鹿莽聙拢忙聺赂茅聴聜盲陆潞莽虏聧聰忙聽搂氓篓聠忙聢聺莽路陇氓漏碌莽聤鲁莽垄聧茅聬聯茫聝漏忙聼隆茅聧楼氓聼聙茅隆聮盲陆潞茅聳碌氓聸篓氓聤聳莽禄聴莽聠录氓聼聙茅隆聮芒聜卢茅隆驴聯脙潞茅隆聫氓聥芦莽陋聸忙驴聽莽聰碌氓搂麓莽聙聸芒聲聟忙聜戮忙驴聜氓聭聤芒聮聭莽录聛氓茅聧聨忙聢娄忙聼聸茅聬聵忙聼楼氓聫聞茅聳潞盲陆赂氓聲麓忙驴庐芒聜聳碌氓聽聺忙聡聬茅隆聰忙聨聭茅聤聢氓聴聵莽聬職忙卢聯忙聼聼茅聳驴氓卢聛聺忙驴漏聝忙陇陆氓庐聲氓麓聼茅聧篓芒聰聭茅聬聬氓路搂鹿忙驴聻氓聫聣莽鹿聢氓庐聲忙聞掳氓鹿聵茅隆聰猫搂拢忙聥禄忙驴聻猫戮戮莽碌陆氓漏聙氓聧聻茂鹿垄忙碌聽氓莽聟聲氓漏碌莽聤虏氓聲炉莽禄聙芒聜聳戮猫聧陇氓聛聻莽禄禄忙露垄氓鹿聥氓篓聠氓驴聲盲禄录茅聴聛莽禄聵氓赂聻茅聫聛氓聯楼忙茅聨潞忙聳潞氓聛聯茅聰聺氓潞隆盲潞聹忙驴聻忙聬颅氓聶拢茅聫聦?茅聴?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    cfg : dict[str, Any]
        茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聟忙職娄茅聳禄茫聢垄莽禄聬忙楼聟忙聼聼茅聳鹿聙氓聤聮忙驴聻氓路潞聹莽禄聽茂陆聠莽鹿?氓漏?茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聦忙碌聳茅聳碌氓篓搂氓聭炉莽聞录忙驴聻忙聮麓氓聼聝茅聧聥忙聮鲁芒聰聭茅聢楼氓麓聡氓聼拢忙楼聽莽聜麓莽聣聴茅聨路氓聭麓氓麓芦茅聤聣脩聞忙炉聨茅聴聜氓聜聶莽陇聛茅聨虏茂驴聽氓麓聺茅聰聲氓聜職莽陋聴茅聳潞氓露聨氓聛聠茅聬聴聸忙聜聴茅聰聺氓潞隆盲潞聳氓篓麓忙禄聞莽虏聯茅聫聦莽聠录氓鹿聧茅聯聫莽聫聰茅聴聛茅聙聻氓卤陆氓垄聳赂茫聞楼莽聛聺茅聬拢莽聝陆氓鹿聥茅聰聰猫聴聣盲潞聹茅聴聛氓聭聤莽赂驴茅聧聨忙聤陆忙聝聛茅聧聝娄芒聮聮茅聳赂氓卤戮忙掳篓忙戮搂忙露職莽麓聮莽聙拢莽聛麓茅聳赂芒聰戮氓聛聯氓茫聜聠氓路禄茅聧聥忙聮露莽麓聯氓氓聮聝猫潞卢氓漏碌芒聞聝忙聦鲁氓庐聲忙漏聙氓聼隆茅聧聬忙搂聡茅聴聜盲陆鹿忙聜露茅聧聨忙禄聢氓麓聵茅聤聤茫聜聠氓聳職茅聴聜氓聜職氓聙路莽禄露忙掳茅聨聡氓麓卤氓娄聻忙聸聼忙陇陆忙驴隆忙颅聦忙陇陇?茅聴聜氓聜職氓聙赂茅聧聤莽聝陆忙聡聴茅聧聸聻莽拢驴茅聳赂氓聛聟茅聧聶氓驴聲芒聜卢氓鹿驴氓聞卤茅隆娄忙聜聶忙驴聽氓聥卢氓聳聬茅聼芦氓聸篓猫聸聥茅聧楼忙篓聞氓漏碌茂鹿陇茅隆鲁掳莽聲路茅聨潞忙聢陆漏茅聳赂忙氓聞聬茅聫聠茫聜聣忙垄禄忙碌拢氓聭聤忙聝聢茅聳鹿氓聠虏莽鹿聮茅聨鹿茫聢陇氓麓聼茅莽聜虏莽聛聤茅聴聛氓聣虏氓聛聛茅聧聨猫戮漏莽虏聢莽聙拢盲潞聹茅聳潞氓露聝氓聤聨茅聢陆氓陇聬忙聼聧猫陇聹茅聧聯忙卢聯忙聜楼忙驴聜氓聭聤氓聛聜忙陇陇忙聞露莽庐聭茅聬聬氓漏聡忙聼聲忙驴聽氓驴聲莽聲卤莽禄戮忙聺驴莽搂鹿氓忙麓麓聴茅聭虏芒聲聜忙聠聶氓茫聜聨氓聛聞茅隆娄茅聛聧氓鹿聭茅聤聫莽聤聰陆茅聬聨茂陆聨盲潞聻茅聧聰茂鹿聙茅聤聢氓聴聴莽卢聮茅聳赂氓漏聞忙聲禄氓庐聲忙聢娄氓鹿聵茅聫聜茫聝娄氓漏碌莽聜虏忙茅隆聸卢氓麓聺茅聬聭猫聤楼芒聮聭?茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聦忙碌聳茅聳碌氓篓搂氓聭炉莽聞录忙驴聻忙聮麓氓聼聝茅聧聥忙聮鲁芒聰聭茅聢楼氓麓聡氓聼拢忙楼聽莽聜麓莽聣聴茅聨路氓聭麓氓麓篓忙驴聽氓聥篓氓聛聯茅氓聥聙碌忙莽聜虏莽聛聺茅聧聰忙掳颅盲驴聤茅茂陆聨忙庐聵忙驴聻氓聴聬氓聻陆茅聨庐忙卢聯忙聜聹氓娄聟聨氓鹿聢忙驴隆莽聜聙聳茅聧聰脩聟莽麓聻茅聢搂莽聭聶氓聤陆忙鹿?
    dotted_key : str
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙聮虏莽茅聧篓忙驴聡茫聛聞茅聫聧盲禄聹莽录聛芒聜戮露氓卢漏莽鹿聠茅聳碌氓聽聺忙聡聫茅聧拢氓聹麓聮茅聬聵茅聺聸芒聳聢茅聳潞氓虏聙氓聳聳茅陋聻氓聴職茅聳禄茂陆聨氓聨录茅聹聣忙驴聽氓漏聜氓聲掳莽禄聣忙麓陋忙聼隆莽聛聻氓聣搂忙陇陇茂鹿聛忙芦聲茅聤聡氓聥炉氓娄赂茅聤聣茂鹿聙忙芦搂莽录聜盲戮聡氓聳聳茅聬聴忙聮鲁莽聲路茅聧聴莽聜漏忙忙驴聻氓聴聵氓聻碌茅陋聻氓聽楼忙垄禄忙碌拢猫莽聛卤茅聳潞氓聭聸炉氓庐聲氓茅聫聣忙聢聽氓聨聬茅聴聛氓聯聞忙麓篓茅聧聽氓聴聵莽娄聧氓氓聽聼莽聟聶茅聳鹿猫氓聤聲茅聧職氓陆聮忙聼拢茅聰聺氓聴聵茫聛聺茅聳潞氓卤赂芒聜卢氓麓聡忙職聠茅聬聨茂陆聨氓聣聸猫垄娄氓漏碌莽聤碌茅聧聯忙驴聥氓鹿聝茅聧聦忙露聴莽鹿聠茅聳禄忙聢拢忙拢聞氓聰炉茅聴聛氓聠虏忙聬聧聤氓漏職氓麓垄忙碌聽氓芒聮聭莽录聛氓卢聛聺茅隆聮茫聢聽芒聰聭茅聬聬猫炉搂猫潞卢茅聳碌氓聴聲莽陇聛茅隆芦忙驴聢忙聧聞忙碌拢忙聸聵氓聤聭茅聫聦氓露聠忙陆陋茅聨路氓卤陆氓聨隆氓茫聜聫莽录職茅聧聴氓聠虏氓篓虏氓路聺氓漏碌莽聤虏氓聲聬颅忙垄潞茅聬聯氓潞拢忙聻聝忙驴聻氓莽聸聧漏?``"network.hidden_dim"``茅聴?
    value : Any
        茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聟忙職娄茅聳禄茫聢垄莽禄聬忙楼聟忙聼聼茅聳鹿聙氓聤聮忙驴聻氓路潞聹莽禄聽茂陆聣忙聼拢氓娄炉氓聟录忙職漏茅聳禄忙聨聭氓聬聥莽禄禄忙露垄氓鹿聵茅聫聣忙聢聽氓聤掳茅聴聛莽篓驴茅聨赂茅鹿聝氓聼聙茅隆聮氓聠戮忙陆聬忙驴聻猫炉搂莽庐聯氓庐聲氓陆聮氓麓录茅聫聡莽聤虏莽楼娄氓漏碌芒聵聠氓聻掳茅聧職氓卢聜卢忙掳虏赂茅聫聦茫聜聡氓聳聤茅聧聴莽聜虏芒聮聮氓漏碌芒聙鲁忙篓聤茅聧篓猫路潞莽录聛忙莽聭鹿茅聳鲁脩聦氓聣聶茅隆颅氓聸聳潞盲陆赂莽搂路莽禄聽忙露聵芒聵聣氓娄炉莽垄卤忙搂鹿忙驴隆莽聜陋氓聙聳茅聨赂茅鹿聝氓麓掳茅聨掳茅聜娄忙聲聤茅聼芦氓聸篓莽篓聮茅聢?茅聴聜氓聜職氓聙赂茅聧聤莽聝陆忙聡聴茅聧聯盲陆赂氓聻聺忙陇陇忙聽聛聻茅聧聥氓茫聛聞茅聨戮猫路潞脨楼莽录聛芒聜驴聽氓聛陆聮忙聲聬茅聧聸氓聳聬茅聴聛氓聯聞忙拢麓莽禄聽忙聮禄氓录禄茅聬聰氓聟录莽聞陆茅聳驴忙聸聴氓聛聯茅聳潞氓卢篓氓聙聴莽禄卤忙聨聴忙聜漏茅聯聫忙茅聴聛猫氓聳聸氓篓聣忙露垄氓聼楼忙戮露忙聞卢莽庐鲁茅聳赂芒聞聝茫聛聻忙戮鹿忙聸垄忙垄潞莽禄聥猫路篓莽茅聳鹿氓聠虏茅聟拢猫聴聼?

    茅聴聜氓聜職氓聙路氓篓麓氓聸茅聢搂莽篓聢茅聫聟莽聤禄茫聙聛氓篓聠忙聮鲁氓聠聙忙陇陇忙聮露氓聛聼茅聬聸茫聝漏忙垄潞茅聧聺氓聥颅芒聳聣茅聳赂忙篓驴氓聛聬忙戮鹿忙聸隆茫聛聬茅聨潞忙聮露氓聨碌茅聴聜氓聜職氓聙赂茅氓麓聺氓氓陇聥莽聟聲?
    ------
    ValueError
        茅聴?``dotted_key`` 氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄茅聳禄忙聞聬芦茅聧篓氓聜聻忙聼聸茅茂陆聠莽莽录聛莽聤鲁氓陆聝茅聹聣茅聳驴忙驴聠忙麓篓茅聤聠氓漏聡忙聼隆莽聙拢脨搂茅聳潞氓卤禄氓聤聦茅聢鹿忙聢娄氓麓卤茅聢潞氓聜聜茂鹿聜忙垄潞莽录聛忙篓禄氓掳掳茅聳驴忙聸聵氓聻陆氓氓漏職氓聻戮茅聨掳盲陆赂忙聜聭茅聴聛氓聭聤盲戮聴茅聧聯忙卢聯脣聣氓搂聵莽炉聙茅聳禄茫聢陇忙陆搂忙碌聽茂鹿聜忙聼聸茅聤聤茫聝搂氓聼拢茅聳鹿氓聟聵茅隆垄忙露聵茅聰聺氓聴聶盲潞聳?氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄茅聳禄忙聞聬芦茅聧篓氓聜聻忙聼聸茅茂陆聠莽莽录聛莽聤鲁氓拢聠莽禄卤忙聨聰芒聜卢茂陆聨氓聛聮茅聧聨茫聝漏忙聼聸茅聤聢氓聴聵莽茅聳鹿莽露聤茅聨禄聝脦聼氓搂鹿氓聸娄猫聶聨茅聳禄氓潞陇氓篓虏忙篓录忙聲庐茅聰聼莽聤虏莽庐聳忙驴聻氓聴聵忙聦禄茅聧陇忙聢聻茫聛聞茅聨潞忙聢搂赂茅聳潞盲陆陆忙鲁聸茅聢鹿忙聢娄忙聜漏茅聧篓忙炉聞茅聴聛莽茅聧聥茫聜聟莽聲路猫陇掳忙聨聭茅聠聡茅聳潞氓聸漏氓聳聬氓篓聟忙聳驴芒聰聭茅聬聵忙聞聛氓聲聫茅聳赂茂陆聨氓聻鲁氓炉掳氓漏聝莽娄聮莽聙拢氓潞隆忙麓聨庐莽聲聦莽虏聶茅聭鲁忙聠掳?
    """
    parts = [part for part in str(dotted_key).split(".") if part]  
    if not parts:  
        raise ValueError("dotted_key must be non-empty")  
    cursor: dict[str, Any] = cfg  
    for part in parts[:-1]:  
        next_value = cursor.get(part)  
        if not isinstance(next_value, dict):  
            next_value = {}  
            cursor[part] = next_value  
        cursor = next_value  
    cursor[parts[-1]] = value  


def _apply_model_overrides(model_cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聫茅聢搂莽聞娄莽禄禄忙驴聥氓麓聮氓娄陇盲陆潞盲潞聶忙驴聽莽聰碌氓聛聻茅聧篓氓麓鹿氓篓虏莽聳戮忙驴聽茅聺聸茅聬聯氓驴聯莽聮潞猫聶鹿氓垄聲氓漏碌芒聙鲁茫聜赂茅聫聦茂驴聽氓麓卤氓篓聠氓驴聨忙氓漏碌茂鹿娄莽禄鹿聫茅聧聸氓聳聠茅聳赂忙聸篓氓聛聜茅聧聺忙楼聟忙垄禄忙碌拢氓聭聤氓聲聻茅聳潞氓卤禄氓聤聭茅聨虏聞氓麓聮茅聬聬氓聛?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聳莽潞戮氓聺聴忙聼拢茅聤聫茫聢漏莽聞戮茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛莽篓驴茅氓陇聬氓录禄茅聰聺氓聭聜莽聠路芒聳聨茅聨聞茅聳潞氓陇聠忙聶聹莽禄禄氓聽楼氓鹿聝氓娄陇氓聭碌聛茅聳赂氓聳聳忙驴潞隆茅聤聢氓驴聯莽聵聹茅聳赂茫聞楼莽拢颅茅聧聮忙聨聴氓麓录茅聤聣茫聝搂茅聴聛氓聠虏忙聬芦茅聧聼盲录麓茅聫聡茫聢隆芒聮聭莽录聛忙麓聳忙戮搂氓聫聣茫聛聞茅聤聤茫聝搂氓垄垄莽录聛茅聨鲁莽聤禄茅聧聦忙禄聬莽麓虏茅聴聜盲戮聙芒聜聜虏猫聥戮搂忙聮聼聸茅聢鹿忙聝搂盲潞戮忙驴隆莽聜聙聳莽聰炉忙聨聰芒聜卢忙掳录氓聫聝茅聧漏忙露垄氓鹿聥茅聰聰莽聲聦氓聨卤氓漏碌莽聜麓氓聻露茅聰聲茅聤聣脩聟莽拢录忙驴隆莽聝聡莽庐聭茅聧聽忙掳露忙聜垄茅聧隆忙卢聬氓聳聬茅聨录忙路聤聤茅聢搂氓聲炉氓炉掳氓聥炉氓鹿聡茅氓聜聸莽聞戮忙驴隆莽聜陋氓聙聳茅聬聴忙禄聢氓麓聭茅聬聬氓聯聞莽聟聲茅聳鹿猫戮戮茅聶聞茅聬聯忙卢聯忙聼陇氓篓麓氓卢芦聟氓篓麓莽聤鲁莽虏聦茅聢鹿忙聢娄氓聻戮茅聧聬氓聫聣氓聙聴聛茅聺聸忙麓陇莽聙職盲录麓莽聧聙拢忙聰陆茅聳鹿茅聜娄氓聨赂茅聫聥氓聭麓忙聼拢忙聬麓茫聜聨忙陆聬忙驴聻忙聫聮莽鹿聵氓庐聲氓陆聮忙聡聻茅聤聣茫聜聡莽庐職茅聴聛氓聣虏氓聛聟氓篓虏忙聽聜聶忙驴聽氓聸篓莽聟聝茅聳赂忙驴聠氓茅聫聥茂赂戮莽麓?茅聴聜氓聜職氓聙赂茅聧聤茅拢聨忙氓搂聵莽聞陆茅聳录氓氓聙聲莽禄露茫聜聣忙聼聸猫陇聨茅隆篓氓聭虏聢茅聧聦忙露聶氓聫聫忙驴隆莽聜露忙碌聹茅鹿聝忙聜聴氓篓聢氓聻庐忙芦聵茅聳赂氓卢茅隆垄忙篓禄莽赂陆莽禄聥猫路篓氓聙聻茅聴聛忙聦聨忙氓娄聥卢氓聢聤忙碌录氓颅聵芒聮聮氓篓麓茂陆聞猫颅娄茅聬聮茅鹿聝忙聼篓茅聫聡莽聲碌莽聙鹿忙聸聽氓垄聴茅聨炉忙聴聢氓娄赂茅聰聲芒聜卢忙碌聽脩聝茅聤聢氓聴聶氓垄卤茅聳赂氓氓聛陇茅聨庐芒聙鲁芒聵聣氓搂聴氓聴聶氓聰碌茅聳禄莽聤潞忙茅聧聨茫聜聣脣聣茅聬聬氓莽庐戮茅聳赂忙露卤氓聨陆茅隆楼茫聢陇忙聼隆氓庐聙氓卢拢驴氓篓麓莽聥聟氓娄聡茅聨路莽聤禄氓鹿聯茅聨潞忙驴聢芒聜聤陆忙垄潞莽录聛忙篓禄氓掳陋茅聳赂氓漏聝莽鹿聢氓氓漏職氓聻戮茅聨掳盲陆赂忙聜聭茅聴聛莽鲁聲忙茅聫聟忙禄聢脣聣茅聤聢氓聴聶莽炉聙茅聳禄茫聢陇忙陆搂忙碌聽茂赂陆莽鹿聸茅聧聫猫聜漏忙聡聡莽聙鹿忙聸聵氓聞鲁茅聢搂莽露聠忙碌聹氓聽聲忙聜聻茅聭潞茫聝搂莽庐戮茅聳赂芒聞聝脙聬聫茂鹿聜忙聼隆茅聧隆忙莽聼聦茅聳鲁脩聦氓聣聺茅隆芦氓陇聤脙潞茅聫聧忙麓娄赂猫陇聧茅隆楼?茅聴?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    model_cfg : dict[str, Any]
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏氓庐聲氓聸搂氓鹿聢?忙驴聽莽聰碌氓搂路茅聫聛氓聭聤忙聟篓茅聬聭氓芒聜聜聻忙聦聤忙戮鹿芒聲聥莽露聛氓庐聲茫聞漏氓录露茅聨麓忙驴聢芒聜卢茅聤聢氓聭炉莽庐掳茅聳禄忙楼聙忙拢聺茅聨庐莽聜潞忙聜陆茅聬垄氓聻芦茂录聫莽录聜氓聜聶茅聶聧茅聧篓莽聭掳茅隆聲芒聜卢茅聫聦忙露垄氓聼聻茅聨炉脩聜氓拢聲茅聴聜氓聜聶猫聝露茅隆聽垄忙聧垄茅聨麓茂赂陆忙聶聺茅聳碌氓驴聲氓陋聧陇忙聸聽莽麓聯忙碌拢猫莽聞陆莽录聛芒聶聙氓聛聯茅聴聜盲陆潞茅聬卢氓楼赂氓麓聭氓篓聭茫聢隆氓聙路氓漏碌莽聤虏忙麓娄茅聬聯氓聠庐忙聼聸氓漏碌氓聴聴茅聳鲁脩聦氓聣聶茅隆颅莽聝陆氓麓潞茅聢搂茫聛聞茅聨潞忙聢颅忙搂赂莽聙碌氓聳聴茅聨?
    overrides : dict[str, Any]
        茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聟忙職娄茅聳禄茫聢垄莽禄聬忙楼聟忙聼聼茅聳鹿聙氓聤聮忙驴聻氓路潞聹茅聧聬芒聜卢氓娄聻茫聜聨氓聣娄茅聧聯忙掳卤莽虏聶茅聨麓茂鹿聙莽聟隆氓猫路潞脙隆茅聬聰猫聧陇莽聳聙茅聳潞氓聸漏氓聛聸忙驴隆猫聶鹿莽麓聯忙碌拢猫聶鹿氓赂聸莽录聛猫炉虏莽聣聠茅聬拢氓鲁掳茅聢搂氓录芦茅聧聦忙禄聞氓麓聣茅聳碌氓篓搂氓聥卢忙芦聮茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聟忙職娄茅聳禄茫聢垄莽禄聬忙楼聟忙聼聼茅聳鹿聙氓鹿聵茅聳禄忙驴聠莽聢录茅聫聦忙露垄氓聼聞茅聧聬忙搂聢莽录聜盲陆潞氓搂碌氓娄聻盲禄聥氓录禄茅聰聺氓陇聤忙茅聳赂氓聳聴茅聧聤忙垄潞莽录聛忙篓禄氓掳掳忙驴聻猫聦聳聳茅聨驴聽茅聰聲芒聜卢茅聬聬氓漏聡忙聼聲忙驴聻脩聙氓聛聯氓漏碌忙麓娄莽禄禄忙露職芒聜卢忙露聶脙篓聢莽聤禄忙聼聸忙驴聽氓聠戮忙庐聵忙驴隆氓聫聣氓聤聶茅陋聻忙聨聴氓鹿聵莽聙碌氓聯楼脨陇茅聴聜盲戮聙忙陆搂猫聡聨茅聳赂忙露卤氓聻卤氓忙聽垄禄?茅聴聜氓聜職氓聙路氓篓麓氓聸搂莽聲路茅聧篓氓聫聫茅聳禄茫聢垄莽禄聙氓陇聥盲驴聤茅聤聢氓聭庐氓聶鹿莽录聛忙聞聫聦芒聙鲁忙聬麓芒聙鲁莽莽录聜盲陆潞氓搂碌茅聬聴忙楼聝忙聧垄氓篓聭茫聢聽氓鹿垄忙驴隆茅聴聣氓露聣忙垄潞莽禄聥忙卢戮氓聞聹莽录聛莽禄聵莽鹿聢氓氓漏職氓聼聞茅聧聬茫聞搂莽陋聻忙驴聽莽聰碌氓搂麓莽聙職忙聬鹿忙聬麓茫聞娄芒聮聭茅聬聨氓聹茅聧陇忙卢聭莽麓聮莽录聛忙漏聢猫聙驴莽聙?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓麓茂陆聢氓篓聟莽聤禄氓鹿聳氓篓拢氓莽禄戮忙聝搂茅聫聦忙聸垄氓麓录氓漏碌氓聸露忙莽录?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    dict[str, Any]
        茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聟忙職娄茅聳禄茫聢垄莽禄聬忙楼聟忙聼聼茅聳鹿聙氓聤聮忙驴聻氓路潞聹茅聧聬芒聜卢氓娄聻茫聜聨氓聣娄茅聧聯忙掳卤莽虏聶茅聨麓茂鹿聙莽聟隆氓猫路潞脙隆茅聬聰猫聧陇莽聳聙茅聳潞氓聸漏氓聛聸忙驴隆猫聶鹿莽麓聯忙碌拢猫聶鹿氓赂聸莽录聛猫炉虏莽聣聠茅聬拢莽聝陆忙聜垄茅聬聯氓鹿聲聟忙聼聲忙戮露氓聽聺芒聜卢忙聸垄脣聣氓猫聢碌芒聮聮氓篓麓茂陆聢忙拢聞茅聧職氓陆聮忙聼聸茅聬聵猫聶芦氓掳掳茅聳潞氓聭掳忙鲁聸猫聻聳茅聳鲁脩聦氓聣聼茅聫聛氓聠茅聢楼忙聞聛氓職陇茅聳禄氓潞垄莽篓聮茅隆聸卢氓麓垄茅聳卢氓露聠芒聮聭茅聴聜氓聽聼盲戮聴茅聬聯莽麓聮茅聬聵氓聠虏莽聛麓莽聙鹿忙聸篓氓聛聤茅聨录氓陆聮氓麓聴忙戮露氓漏聛氓拢聲茅聳禄莽聠赂莽聭楼莽聙職氓茅聨录盲陆鹿莽聟聲茅聨录忙路聻禄茅聧聶芒聜卢氓娄陇莽聤碌氓聛聸氓娄芦忙驴聠茅芒聜庐聲氓楼赂忙聜垄茅聧聸氓漏聞氓陆赂茅聴聜氓聜聶猫聝露莽潞聽聲氓麓聹氓漏碌氓卢聛隆茅聳驴忙聸聴氓聛聯茅聳鲁莽禄聵忙聧聡茅陋聻忙聽篓忙聥聟茅聧聫氓陇聥氓氓漏碌莽聜露忙聫禄虏盲禄聥氓鹿聣茅聰聼莽聤虏莽庐職茅聳赂氓聞虏茅聬聯忙聸聻芒聰聭茅聬聵忙聫聮茅聬垄莽聸炉茅聫聦茂驴聽氓麓篓茅聬聴氓聽聙聬茅聳戮莽禄聵莽聣聽茅聫聦忙露垄氓录麓茅聤聤茫聝陇莽掳录氓漏碌莽聜虏莽聣聤茅聬聯芒聙鲁忙驴聽忙掳拢录忙驴聻氓聴聵氓聻碌忙驴隆盲禄聥忙垄潞莽聮聡芒聙鲁忙聻聸茅聳禄忙聽芦氓聨搂茅聬拢氓鲁掳芒聰聭茅聧隆忙卢聬忙庐聲茅聴聛氓聭聤忙麓聳茅聬聫忙掳露氓录虏茅聬聬氓炉赂莽炉聙茅聳碌氓驴楼莽碌陆茅聬聯忙聞掳忙聼聸茅聫聝芒聜卢氓篓虏莽聠路莽聲路茅聨潞忙楼聙脦漏茅聳鲁氓聯聞氓聙禄茅聧聵忙垄潞茅聧聯忙聫聮盲潞聺莽录聛猫氓聲氓驴聯芒聰聭茅聧聢隆茂录?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬氓庐聞忙聡聤氓陆聮氓麓露茅聤聤茂陆聞莽虏聯茅聴聛氓陆聮忙拢驴茅聬聮茂鹿聜氓麓聭茅聰聼莽聤鲁莽聟聫氓漏垄猫路潞莽聣聠忙驴隆猫聜漏莽麓聮茅聨禄忙聨聰莽搂潞忙驴庐氓漏聝忙陇陆氓庐聲氓麓聼茅聧鹿聜忙垄潞莽录聛忙漏聠忙庐聰茅聬聨忙掳颅忙戮聵茅聬拢氓鲁掳芒聰聭茅聧隆忙卢聬忙庐聲茅聴聛氓聭聤忙麓聳茅聬聫忙掳露氓录虏氓氓聽聼芒聮聭茅聳赂忙聮麓氓陆聝忙碌聹忙驴聡莽鹿聸茅聧聶氓猫聙驴莽聙鹿忙聸聼氓聻驴茅聫聛忙聞戮聞莽聙拢氓鹿聧茅聴聜盲陆陆茅隆聰忙聽颅氓麓鲁茅芒聜庐聲氓麓聹忙碌拢猫聶鹿茂录聺茅聴聛莽忙聹聬聼茂赂戮莽虏聤茅聫聣忙忙赂禄茅聳潞氓陇聥氓聻露茅聨虏忙聬聛忙聼聼茅陋聻氓聻聛虏莽聙鹿忙聸聼茂鹿垄茅隆垄忙聴聢氓麓卤氓篓聠忙聢聺氓鹿聠忙驴聽莽聰碌氓搂掳茅聧篓莽聟聨芒聳聰氓篓聭茫聢隆忙聶聺忙陇陇氓驴聯氓芦聫聠茂鹿聜忙聼拢茅聫聜忙忙庐聽茅聳赂茫聜聠忙麓聼茅聫聦忙露聹忙陇陇茫聞楼忙拢聺茅聧漏忙露垄氓鹿聥茅聰聰猫聴聣盲禄漏氓漏碌莽聜麓氓聻露氓庐赂氓氓麓聭茅聨戮猫炉虏莽茅聳赂忙露卤氓聫聨氓娄芦忙禄聢忙垄禄茅聧聦忙掳卢芒聜鲁掳脙鹿茅聧楼脩聟忙職娄茅聳禄茫聢垄莽禄聬忙忙聼聼茅聳颅茂鹿聙氓垄禄茅聳禄忙聳驴忙茅聨卤茫聞楼莽聮聡猫鹿颅氓拢聲茅聳禄氓潞垄茅聧拢氓麓聹氓漏碌氓卢聨氓聸陋茫聛聬茅聨潞忙聮露氓聨碌氓娄聻茫聜聠忙垄禄茅聬隆忙聳驴芒聳聯氓漏聤氓聽聼莽聟聶忙陇陇忙聴聙氓炉搂茅隆楼氓陇聬忙聼聧莽录聛忙篓禄氓麓聦忙莽聜麓氓聤聳茅聨聴聢氓搂聙茅聢漏氓聠戮氓聰鹿茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼茫聞娄氓聫聫茅聳碌氓聽聺茅聧聥氓忙聼隆茅聧楼芒聲聛脨聸茅聳赂氓卢聦赂茅隆驴聥茅聤聫莽聤碌茫聜聳禄氓潞垄茅聧拢氓麓聭茅聧聲忙聲聡氓漏碌氓聜聹茅陋聻茫聢隆盲驴聤茅聤聢氓聭颅忙職聼茅聳禄忙露卤氓聶拢氓搂聤莽禄聵忙聥聟莽禄聥忙卢戮忙茅聴聛莽茅聨鲁忙漏聙氓鹿聤氓漏碌氓卢聙路忙陇陇忙聨聭氓聛聠氓篓?
    """
    resolved = copy.deepcopy(model_cfg)  
    for key, value in overrides.items():  
        _set_nested_cfg_value(resolved, key, value)  
    return resolved  


def _apply_estimator_overrides(estimator_cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聫茅聢搂莽聞娄莽禄禄忙驴聥氓麓聮氓娄陇盲陆潞盲潞聶忙驴聽莽聰碌氓聛聻茅聧篓氓麓鹿氓篓虏莽聳戮忙驴聽茅聺聸茅聬聯茫聝娄莽鹿聺茅聴聢忙露聶茂录聧忙驴隆氓聫聣忙聜聜茅聫聦茂陆聣茅聬聲盲陆鹿莽聲露茅聴聛氓庐聽茅聧篓氓聺聴氓鹿聝茅聢潞氓聭茅聳鹿莽聝聵莽鹿聙茅聬楼芒聰聭茅聬聵忙聞聛氓聲聫茅聳赂茫聞陇氓聧鲁忙驴隆氓聽聲氓鹿聳忙碌拢忙颅聦忙聭聲茅聬聨氓鹿驴氓聞卤茅隆娄忙聲聴聜盲戮聙忙陆搂茅隆颅盲驴聤茅聧楼盲潞鹿莽聭聹忙聳驴茅聯聫氓聹拢录忙驴隆氓聛聬莽禄卤盲录麓忙垄潞莽禄聥氓聦隆忙聰禄莽录聛猫莽聣聯茅聬聸氓麓聵茅猫聴聣芒聜卢氓漏職忙聼娄氓娄戮炉忙搂赂莽聙鹿忙聮鲁忙拢聺氓猫聶鹿莽麓聮氓娄聯聞茅聴聛氓忙聺驴氓茅聳鹿氓鹿驴氓聻鹿茅聢陆氓陇聬氓茅聬聵忙氓聰陆茅聴聜盲陆潞氓篓聟茂陆聣氓麓聭茅聬聬氓聯聞忙聻聛氓娄陇茂陆聟氓聲炉茅聢路忙聢娄忙垄禄茅聧芦莽聠潞芒聤聲茅聳鹿氓聟录氓聤聶茅聨庐忙楼聙茅隆聬茫聢聽莽楼娄陇莽聤碌氓聛聸茅聧聼忙聮庐氓鹿聝氓氓聽聼氓炉聽氓漏垄猫路潞莽聼聜茅聬聸忙聮禄忙垄禄忙碌拢氓聭聤猫麓隆忙陇陇莽聣聫茅聢搂莽篓聢茅聫聟莽聤禄茫聙聛茅聫聝氓聽聼氓聛聞茅聳赂氓驴聯莽職路茅聨路莽禄聵莽鹿聸忙聺聢氓聣搂莽搂陇陇氓庐聲忙聢娄忙聜漏莽录聛忙篓录氓聨卤茅聳鹿氓聟录莽聲聧篓氓陆聮氓聼垄茅聫聡莽聤芦氓聛聯莽聭聶氓聥茅聳赂茫聞搂忙聲禄茅聧漏茫聝娄氓戮聞茅聨掳忙陆聤茅聴聛莽禄聵氓篓聟忙聸隆芒聟虏茅聴聜氓聜聶忙碌聡茅隆聲脩聟忙聧垄茅聨潞忙莽拢禄茅聳禄忙聞聬芦莽潞戮氓聺聴忙聼聼茅聨炉忙聺驴脨聯茅聳鲁?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮盲禄聥忙聼陇忙驴聞卢芒聜聸聽忙卢垄茅聤聢氓聭聰陆茅聳禄忙篓潞莽聳聡忙戮鹿忙篓录莽麓聮茅聢搂氓戮聞茅聨掳氓聨碌茅聳潞氓莽聣聠忙戮搂盲禄聥忙聜戮茅聳露聠莽聟聲茅聬聫莽聜露忙驴庐氓聜聻忙聼隆莽聛聻莽聜卢莽陇聥莽聙鹿忙聸聽氓聨搂茅聢鹿忙聢娄氓麓露猫陇聫茅聬聸芒聲聝莽麓聯?茅聴聜氓聜職氓聙赂茅聧聤茅拢聨忙氓搂聵莽聞陆茅聳录氓氓聙聲莽禄露茫聜聣忙聼聸猫陇聨茅隆篓氓聭虏聢茅聧聦忙露聶氓聫聫忙驴隆莽聜露忙碌聹茅鹿聝忙聜聴氓篓聢氓聻庐忙芦聵茅聳赂氓卢茅隆垄忙篓禄莽赂陆莽禄聥猫路篓氓聙聻茅聴聛忙聦聨忙氓娄聥卢氓聢聤忙碌录氓颅聵芒聮聮氓篓麓茂陆聞猫颅娄茅聬聮茅鹿聝忙聼篓茅聫聡莽聲碌莽聙鹿忙聸聽氓垄聴茅聨炉忙聴聢氓娄赂茅聰聲芒聜卢忙碌聽脩聝茅聤聢氓聴聶氓垄卤茅聳赂氓氓聛陇茅聨庐芒聙鲁芒聵聣氓搂聴氓聴聶氓聰碌茅聳禄莽聤潞忙茅聧聨茫聜聣脣聣茅聬聬氓莽庐戮茅聳赂忙露卤氓聨陆茅隆楼茫聢陇忙聼隆氓庐聙氓卢拢驴氓篓麓莽聥聟氓娄聡茅聨路莽聤禄氓鹿聯茅聨潞忙驴聢芒聜聤陆忙垄潞莽录聛忙篓禄氓掳陋茅聳赂氓漏聝莽鹿聢氓氓漏職氓聻戮茅聨掳盲陆赂忙聜聭茅聴聛莽鲁聲忙茅聫聟忙禄聢脣聣茅聤聢氓聴聶莽炉聙茅聳禄茫聢陇忙陆搂忙碌聽茂赂陆莽鹿聸茅聧聫猫聜漏忙聡聡莽聙鹿忙聸聵氓聞鲁茅聢搂莽露聠忙碌聹氓聽聲忙聜聻茅聭潞茫聝搂莽庐戮茅聳赂芒聞聝脙聬聫茂鹿聜忙聼隆茅聧隆忙莽聼聦茅聳鲁脩聦氓聣聺茅隆芦氓陇聤脙潞茅聫聧忙麓娄赂猫陇聧茅隆楼?茅聴?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    estimator_cfg : dict[str, Any]
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏氓庐聲氓聸搂氓鹿聢?氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄茅聳禄忙聞篓聤茅聬聮猫聤楼莽鹿聸茅聧隆忙篓禄氓掳颅茅聬聼忙卢聶莽庐聯茅聨庐忙楼聟忙聲聬忙聬麓芒聙虏莽掳禄茅聴?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮盲禄聥忙聼陇忙驴聞卢芒聜聸聽忙卢垄茅聤聢氓聭聰陆茅聳禄忙篓潞莽聳聡忙戮鹿忙篓录莽麓聮茅聢搂氓戮聞茅聨掳氓聨碌茅聳潞氓莽聣聠忙戮搂盲禄聥忙聜戮茅聳露聠莽聟聲茅聬聫莽聜露忙驴庐氓聜聻忙聼隆莽聛聻莽聜卢莽陇聥莽聙鹿忙聸聽氓聨搂茅聢鹿忙聢娄氓麓露猫陇聫茅聬聸芒聲聝莽麓聯?茅聴?
    overrides : dict[str, Any]
        茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聟忙職娄茅聳禄茫聢垄莽禄聬忙楼聟忙聼聼茅聳鹿聙氓聤聮忙驴聻氓路潞聹茅聧聬芒聜卢氓娄聻茫聜聨氓聣娄茅聧聯忙掳卤莽虏聶茅聨麓茂鹿聙莽聟隆氓猫路潞脙隆茅聬聰猫聧陇莽聳聙茅聳潞氓聸漏氓聛聸忙驴隆猫聶鹿莽麓聯忙碌拢猫聶鹿氓赂聸莽录聛猫炉虏莽聣聠茅聬拢氓鲁掳茅聢搂氓录芦茅聧聦忙禄聞氓麓聣茅聳碌氓篓搂氓聥卢忙芦聮茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聟忙職娄茅聳禄茫聢垄莽禄聬忙楼聟忙聼聼茅聳鹿聙氓鹿聵茅聳禄忙驴聠莽聢录茅聫聦忙露垄氓聼聞茅聧聬忙搂聢莽录聜盲陆潞氓搂碌氓娄聻盲禄聥氓录禄茅聰聺氓陇聤忙茅聳赂氓聳聴茅聧聤忙垄潞莽录聛忙篓禄氓掳掳忙驴聻猫聦聳聳茅聨驴聽茅聰聲芒聜卢茅聬聬氓漏聡忙聼聲忙驴聻脩聙氓聛聯氓漏碌忙麓娄莽禄禄忙露職芒聜卢忙露聶脙篓聢莽聤禄忙聼聸忙驴聽氓聠戮忙庐聵忙驴隆氓聫聣氓聤聶茅陋聻忙聨聴氓鹿聵莽聙碌氓聯楼脨陇茅聴聜盲戮聙忙陆搂猫聡聨茅聳赂忙露卤氓聻卤氓忙聽垄禄?茅聴聜氓聜職氓聙路氓篓麓氓聸搂莽聲路茅聧篓氓聫聫茅聳禄茫聢垄莽禄聙氓陇聥盲驴聤茅聤聢氓聭庐氓聶鹿莽录聛忙聞聫聦芒聙鲁忙聬麓芒聙鲁莽莽录聜盲陆潞氓搂碌茅聬聴忙楼聝忙聧垄氓篓聭茫聢聽氓鹿垄忙驴隆茅聴聣氓露聣忙垄潞莽禄聥忙卢戮氓聞聹莽录聛莽禄聵莽鹿聢氓氓漏職氓聼聞茅聧聬茫聞搂莽陋聻忙驴聽莽聰碌氓搂麓莽聙職忙聬鹿忙聬麓茫聞娄芒聮聭茅聬聨氓聹茅聧陇忙卢聭莽麓聮莽录聛忙漏聢猫聙驴莽聙?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓麓茂陆聢氓篓聟莽聤禄氓鹿聳氓篓拢氓莽禄戮忙聝搂茅聫聦忙聸垄氓麓录氓漏碌氓聸露忙莽录?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    dict[str, Any]
        茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聟忙職娄茅聳禄茫聢垄莽禄聬忙楼聟忙聼聼茅聳鹿聙氓聤聮忙驴聻氓路潞聹茅聧聬芒聜卢氓娄聻茫聜聨氓聣娄茅聧聯忙掳卤莽虏聶茅聨麓茂鹿聙莽聟隆氓猫路潞脙隆茅聬聰猫聧陇莽聳聙茅聳潞氓聸漏氓聛聸忙驴隆猫聶鹿莽麓聯忙碌拢猫聶鹿氓赂聸莽录聛猫炉虏莽聣聠茅聬拢莽聝陆忙聜垄茅聬聯氓鹿聲聟忙聼聲忙戮露氓聽聺芒聜卢忙聸垄脣聣氓猫聢碌芒聮聮氓篓麓茂陆聢忙拢聞茅聧職氓陆聮忙聼聸茅聬聵猫聶芦氓掳掳茅聳潞氓聭掳忙鲁聸猫聻聳茅聳鲁脩聦氓聣聼茅聫聛氓聠茅聢楼忙聞聛氓職陇茅聳禄氓潞垄莽篓聮茅隆聸卢氓麓垄茅聳卢氓露聠芒聮聭茅聴聜氓聽聼盲戮聴茅聬聯莽麓聮茅聬聵氓聠虏莽聛麓莽聙鹿忙聸篓氓聛聤茅聨录氓陆聮氓麓聴忙戮露氓漏聛氓拢聲茅聳禄莽聠赂莽聭楼莽聙職氓茅聨录盲陆鹿莽聟聲茅聨录忙路聻禄茅聧聶芒聜卢氓娄陇莽聤碌氓聛聸氓娄芦忙驴聠茅芒聜庐聲氓楼赂忙聜垄茅聧聸氓漏聞氓陆赂茅聴聜氓聜聶猫聝露莽潞聽聲氓麓聹氓漏碌氓卢聛隆茅聳驴忙聸聴氓聛聯茅聳鲁莽禄聵忙聧聡茅陋聻忙聽篓忙聥聟茅聧聫氓陇聥氓氓漏碌莽聜露忙聫禄虏盲禄聥氓鹿聣茅聰聼莽聤虏莽庐職茅聳赂氓聞虏茅聬聯忙聸聻芒聰聭茅聬聵忙聫聮芒聜陆聮氓麓聭茅聤聣芒聲聜莽聟聫茅聳赂芒聞聝茅聢搂忙陆隆氓炉麓聵茅隆聰猫聜漏茂录聹氓漏碌忙碌聹氓麓垄茅隆聮芒聜卢茅聢鹿忙聢娄忙聜漏茅聧篓忙炉聞忙驴聽忙庐驴氓聳聳茅隆聲氓聼聙茅隆聮盲陆赂氓職聳赂忙篓鹿莽虏聶氓庐聲忙掳颅莽鹿聺茅聧聬忙陋聴聛氓聭聤莽篓聭茅聰聲茂鹿聜氓麓垄茅聨戮莽禄聵芒聮聭茅聴聜氓聽聼盲戮聴氓娄虏氓聽聲忙聼聸茅聫聜猫路篓脨搂茅聳鹿忙聲禄茅隆垄忙聴聢氓麓录茅聬聰氓聯聞氓鹿聴忙驴聽莽聰碌氓篓聙氓篓录脩聟芒聜卢忙掳录氓聣聼茅聧漏芒聜卢忙陇陇忙聨聧猫聝露莽禄聽忙聮麓茫聛聞茅隆聬茫聢垄莽庐掳茅聬聯茫聜聠茫聛聞猫鹿聡忙聮鲁氓娄陇莽聤虏忙麓聫聦茂陆聣忙聜漏茅聧聫氓聭掳氓聼聟茅聴聛氓聭聤忙聧聡忙碌聹猫聢碌盲驴聤茅聭赂茅聺聸莽路聻茅聬聨茂陆聨氓聣聶茅陋聻忙楼聟忙垄禄忙赂職芒聜卢氓篓录脩聡忙聜搂茅聧隆忙露聵莽庐聽茅聼芦氓聸篓猫聸聥忙戮露忙聞卢茅聳碌氓聽聺忙拢聸茅聧聵氓聭聤莽鹿聸忙聺聢氓聣搂氓聢掳茅聳鹿猫聤楼莽虏聧茅聫聟茫聝漏忙聼拢忙聬麓茫聜聨忙陆聬忙驴聻氓聫聣莽聣聽忙驴隆氓聽聲氓鹿聳忙碌拢氓聯楼莽聲潞茅聴聛茅聺聸莽鹿聢茅聧聤忙聽颅氓鹿聠茅聬聬莽聝聵莽聟聲茅聳驴忙聴聡茅陋聻忙莽鹿聸茅聧芦莽聠赂茫聛聺忙驴漏聜氓庐聲忙聨聭芒聳聨茅聨麓脨聹莽录聜氓聜職氓聙赂莽禄聣忙聮聲聝茅隆聫氓聽聺猫聯聼茅聳驴忙露聵氓氓篓聦莽麓聮氓篓聭忙漏聠氓聞聬氓娄陇忙聴聜莽麓聯茅聧聦忙卢聭猫麓掳茅聳赂莽聤鲁莽聣聽茅聨聞莽鹿聺茅聧聦茫聜聟忙庐聴聛氓聭聤忙麓娄茅聧篓忙聣庐莽虏禄氓篓聭忙卢聧盲潞聹茅聳鹿忙聧聬忙鲁聸茅聫聥忙聢娄忙聼拢氓漏碌忙聞卢盲潞掳忙驴庐氓漏聟忙茅聨路莽聤禄忙聡聽茅氓聜陇氓聰潞茅聴聜盲陆潞茅隆聭氓聸篓莽鹿聸茅聢搂芒聜忙麓聵茅聧篓氓聺聴莽聧庐茅聧楼氓聛聥茅聳赂莽鹿聙茅聬垄莽禄聵莽鹿聺茅聬垄茅聺聸芒聳聢忙驴聻盲陆鲁氓聸搂氓聡聽?茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聦忙碌聳茅聳碌氓篓搂氓聭炉莽聞录忙驴聻忙聮麓氓聼聝茅聧聥忙聮鲁芒聰聭茅聢楼氓聠虏莽录聛莽聭搂茅聨鹿氓聥娄赂茅聰聰莽颅聥氓聲聨茅聴聜氓聜聶莽路職氓篓录脩聟忙聧垄茅聧聝楼莽拢鹿茅聳潞氓露聨氓聨录莽潞?茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聟忙職娄茅聳赂氓聛聟茅聧聶氓驴聸忙聼拢茅聨麓茂陆聠莽禄戮忙聝搂茅聫聦莽聠录忙聼聡茅聰聲芒聜聧聰茂鹿聜忙聼篓氓漏碌氓聴漏忙搂赂莽录聛芒聜聙拢盲潞聹茅聳鹿忙聧聬忙鲁聸氓篓聟氓驴聰莽鹿聸茅聭虏芒聲聛忙聻聨茅聳鲁猫搂聞氓聻驴茅隆垄忙卢聭莽虏碌莽聙拢氓搂录忙驴聽莽聰碌氓聛聸茅隆娄忙聣虏聶茅聫聧氓麓虏?
    """
    resolved = copy.deepcopy(estimator_cfg)  
    for key, value in overrides.items():  
        _set_nested_cfg_value(resolved, key, value)  
    return resolved  


def _run_frontend_training(
    *,
    model_name: str,
    prepare_root: Path,
    raw_root: Path,
    split_manifest: dict[str, Any],
    output_root: Path,
    estimator_cfg: dict[str, Any] | None = None,
    device: str,
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    model_overrides: dict[str, Any] | None = None,
):
    """莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳鹿茅聴麓莽莽潞戮莽聭掳莽聙拢忙聧拢茅聳禄忙茅聹聣茅聳驴忙驴聠莽聣聞茅聢搂氓聛聯茅聴?茅聴?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬氓庐聞忙聡聤氓陆聮氓麓露茅聢鹿忙聝搂盲潞戮忙驴庐忙漏聠氓聛聞氓庐聯氓聴聵盲驴聤茅茂录聞氓聤聧莽聙碌忙聺驴氓芦禄禄忙露垄氓鹿聮茅聨麓莽聤碌脨篓茅聴聛猫陆掳莽聞娄茅聧聰忙卢聫莽聲路忙驴聜莽莽聟聭茅聳鲁猫陆掳忙忙戮鹿忙聸垄忙垄潞茅聧聯猫聦聨聸茅聳赂氓忙拢聺茅聨炉氓卤陆莽聲聺茅聢搂忙茅聨潞忙聮麓莽露聡茅聳碌氓篓驴氓聞虏莽聮聡莽聜虏脦聰茅聬聵茅聺聸盲禄聹忙驴隆莽虏聯猫聴聺茅聳鹿忙聝掳氓搂陇茅聬聯忙聸垄忙聜聴茅聰聺氓聠篓氓娄录茅聳赂忙聴聙莽虏聨莽禄卤忙聨聰莽麓聮氓娄聟录氓聢聸氓娄陇莽聤碌氓聛聻茅聰聲茫聢聽莽聲路氓娄炉盲戮聲聟忙陇陇忙聴聡氓搂陇茅忙聢娄忙垄禄茅聧聦忙卢聮氓聟聦忙陇陇茫聢聽茂鹿娄忙碌聹莽芒聳聨茅聨掳莽聫路氓漏碌脗掳氓聙聲茅聨鲁氓潞拢莽虏禄茅隆聳莽聜麓莽聟聲忙驴聻忙聢聻茅聨陆莽聤禄忙聼聸忙驴聽氓聥颅氓赂聸氓篓拢氓聸搂茂鹿陋茅隆垄忙露聵茅隆聮盲陆赂芒聜聰聭茅聬聬猫聦卢氓垄聸茅聳鹿氓聙赂茅隆芦氓驴聲莽麓聮氓娄茅聴聛氓聭聤莽篓聭茅聰聲茫聞搂忙露碌茅聢搂忙垄禄茅聧聦忙卢聮氓聟聦忙陇陇莽聣聯茅隆垄忙聽麓卤氓篓聭忙茅聴聛忙聬聻氓聞潞茅聧聯忙卢聯忙聥聭茅聬聰氓聟录莽聟楼茅聳禄忙聳驴忙聬芦氓颅聺茅聴聜盲陆赂氓麓卢氓篓虏茂赂戮忙聧垄氓漏碌氓卢聻聜忙陇陇忙聞漏氓聻鹿茅隆芦氓露聠忙聟篓氓漏碌氓聴聵氓路聺莽录聛猫戮篓忙聧聡氓庐聲忙聨聭芒聳聨茅聨麓莽聤碌氓麓虏忙驴隆莽聜陋氓聙聭茅聧聤忙聸垄氓麓聬茅聧篓忙職娄茅聳驴忙驴聠莽禄聙氓聠聼聲忙驴聻脩聝氓茅隆聰忙聸聼莽鹿聺氓茅聰聥忙戮聵茅聢搂忙麓聼茅陋聻氓漏聝忙聝聰茅聢漏氓聠漏忙忙驴聽莽莽麓聲茅聧聥忙聢娄氓麓聬茅聫聧氓聛聥忙驴隆茫聜聟氓聲聧篓忙驴聥氓鹿聳茅聨录莽路職莽录聛猫忙拢聺茅聫聦忙聸垄氓麓录氓漏碌氓聸篓忙聜聶茅聳赂莽聠赂忙聜聜氓搂聤猫聶鹿莽虏聶氓篓聠忙聝搂氓聣鲁忙驴聽莽聰碌氓聛聬茅聧聥忙聮禄忙聜聴莽聭聶氓聥卢莽陇聝忙驴隆莽聜露忙茅隆聲茂陆聠莽娄聮莽聙拢芒聲聟忙聼聲茅聧芦忙驴聡氓路聺茅聳禄茂陆聣莽庐聳氓莽禄聵忙聥聟茅聧聺茫聝搂脨篓氓漏碌芒聵聠氓聛聣氓篓聣忙聸聽氓聢聮茅聧聠赂氓陋麓茅聬聻忙露聮芒聜卢氓聟聟氓氓漏碌茂鹿漏茅聧聯茂鹿聙脦聼茅聢搂忙垄潞茅聧聺氓聥庐氓芦掳莽录聛氓陇聦氓聟聵莽炉聯氓篓聯忙掳拢氓聛聯茅聳潞氓卤戮莽聸陋聻忙篓录芒聜卢氓聻聜露茅聢搂氓赂聴忙路聡氓漏聡氓麓拢忙戮露氓漏聜氓娄陇莽聤碌氓聛聻莽聰聨聲莽聛聝茅聴聛茅聙聻氓卤陆氓垄麓茅聳潞?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳鹿茅聴麓莽莽潞戮莽聭掳莽聙拢忙聧拢茅聳禄忙茅聤聠忙聨聭茅聰聺氓聭卢氓拢聲茅聳禄氓潞陇氓篓虏茂鹿聜氓麓鹿茅聧聸聽氓碌聛莽聙鹿茫聝楼茅聫聛氓漏聡忙聼拢茅聨戮猫聶聧拢氓篓麓忙禄聙氓聧鲁氓莽禄聵莽卢聼茅聢搂猫忙戮搂忙驴聥忙垄潞茅聧聺氓聥聶潞茅聳禄芒聲聤氓聻掳茅聬拢氓鲁掳芒聳聨茅聨戮忙聺聭盲潞鹿茅聬聨猫搂聞忙麓聳氓篓虏茫聞漏忙聝聛茅聧芦莽聠潞莽炉聙茅聳禄茫聢陇忙陆搂忙碌聽茂鹿聜忙聼聸忙聬麓茫聜聠莽聫录聛氓聜職莽搂鹿茅陋聻氓聴職茅聳潞芒聜卢茅聭潞茫聝拢盲潞聹茅聳潞氓聠拢氓聛聯茅聳潞氓氓庐聲忙茅聢搂氓聼聻茅聨麓茂鹿聙莽聛鲁茅聳戮氓驴拢氓聛聞莽禄聬忙露聵芒聰聭茅聢楼氓聠虏莽潞搂茅聳赂忙聴聙莽聭漏茅聬聸氓鹿聮氓娄陇芒聜禄聽茂陆聣忙聼隆茅聧聬茂陆聟盲潞戮茅聴聛猫炉虏茅茅隆聝陆氓录禄茅聬聰氓聟录氓職聦茅聳禄忙莽聣聠氓篓聭脩聟莽麓聯忙碌拢莽聭掳莽碌禄忙驴聻莽隆路莽鹿聺茅聬聸氓戮聞茅聨掳氓聻聶聹茅聴聛氓聣虏氓聛聠茅聧聽忙聮禄氓麓垄忙碌聽氓忙陇陇忙聞漏氓聻潞忙戮露氓聥鹿聸茅聧聶氓氓垄录聛氓聜聸氓赂聮茅隆聞漏氓麓录茅聬聰氓聯聞氓鹿聢氓漏碌莽聤碌忙聲赂忙驴庐忙聮麓聬忙驴聽忙掳卢忙聵楼茅聳驴忙驴聠茅聬聯忙聸垄忙聼聧莽聭聶氓聥芦氓聤陇氓篓麓?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    model_name : str
        茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聟忙職娄茅聳禄茫聢垄莽禄聬忙楼聟忙聼聼茅聳鹿聙氓聤聮忙驴聻氓路潞聹莽禄聽茂陆聠莽鹿?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳鹿茅聴麓莽莽潞戮莽聭掳莽聙拢忙聧拢茅聳禄忙茅聤聠忙聨聭茅聰聺氓聭卢氓拢聲茅聳禄氓潞陇氓篓虏茂鹿聜氓麓鹿莽聮潞盲戮炉氓職聲茅聳录莽篓驴莽聛職茅聧聨莽聠录忙聼聼茅聨炉脩聦氓職聼氓篓麓忙禄聞莽聣聽氓搂聤莽禄聵忙聥聟茅聬聭忙篓潞忙庐聦茅聴聛氓忙聺驴氓忙楼聽莽聜虏忙拢聶莽禄禄忙驴聥氓麓露茅聤聤茫聝陇莽掳禄茅聴聜盲陆潞莽虏炉茅聨赂莽脙鹿茅聳驴氓茅聬聨茂陆聨氓聳聴茅聧聤莽聰赂忙聜路氓篓聠氓驴聯莽录聛芒聜卢茅聢搂芒聰聭茅聬聬猫路潞莽職潞氓漏碌莽聜漏氓聛聯茅聬聨猫忙麓聵茅聧篓氓聺聴莽聧庐氓娄聜漏莽拢录忙驴隆茫聙聙茅聧聥忙聮麓莽鹿聺氓氓聫聝氓漏碌芒聵聠氓聻炉莽聮聙茂鹿聙脦拢茅聭虏芒聲聜氓聛聳忙驴聽茅聺聸茅聬聯莽聠录氓鹿聳忙聺聢氓聣搂莽拢驴茅聳禄茂陆聨氓聫聝茅聫聦忙露聶忙聝聰茅聤聤茫聜聠氓聳職茅聬聨猫忙麓娄茅聧篓氓聽聲氓鹿聝茅聢潞氓聠漏氓芦職茅聳录莽垄卤氓聣篓莽禄?
    prepare_root : Path
        prepare 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳鹿茅聴麓莽莽潞戮氓陆聮忙聼聼茅聴聜氓炉赂茅聳禄茫聜聨氓聶拢茅聫聦莽聠录氓鹿聠茅聬聴氓聽茅聴?氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄忙驴聻忙聢聶氓聻潞茅聧聥茫聜聣忙聼聲忙驴聻脩聝忙聡聨茅聫聠忙楼聟忙垄潞莽禄聥猫路篓莽聛卤茅聳赂氓卢篓聭茅隆聰氓驴聯芒聰聭莽聙拢氓聨碌莽录聜盲陆赂莽聙碌脩聡氓麓碌茅聧楼茫聞娄氓聫?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳赂忙驴聞氓聞鲁茅聬聮禄忙聜聴茅聴聢忙露聶忙聠赂茅聳禄忙聧聡茅聨鲁掳芒聳聨猫鹿聡忙聯聝盲禄鹿莽录?
    raw_root : Path
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏氓庐聲氓聸搂氓鹿聢?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬氓庐聞忙聡聤氓陆聮氓麓露猫陇聹氓篓麓忙聽聼聲忙驴聻莽聜卢忙芦聠茅聳赂茫聜聟氓茅聫聦茫聜聡莽虏聶莽聮潞茫聞陇莽虏碌氓漏碌莽聜虏忙聡聬忙驴庐氓聻漏卤忙驴聽氓聸搂脦搂茅聳赂氓卤戮莽聼戮鹿忙聸垄忙垄禄忙碌拢茅盲赂虏莽录聛猫鹿虏聶茅聨篓潞忙聺聢芒聲聝莽聲露茅聴聛莽禄聵忙垄禄茅聧聵脩聡忙聜聻茅聧篓盲潞聹茅聳鹿莽聝聵氓聻碌茅芒聲聝莽麓聮?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳赂忙驴聞氓聞鲁茅聬聮禄忙聜聴茅聴聢忙露聶忙聠赂茅聳禄忙聧聡茅聨鲁掳芒聳聨猫鹿聡忙聯聝盲禄鹿莽录?
    split_manifest : dict[str, Any]
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬氓庐聞忙聡聤氓陆聮氓麓露猫陇聹氓篓麓忙聽聼聲忙驴聻莽聜卢忙芦聠茅聳赂茫聜聟氓茅聫聦茫聜聡莽虏聶莽聮潞茫聞陇莽虏碌氓漏碌莽聜虏忙聡聬忙驴庐氓聻漏卤忙驴聽氓聸搂脦搂茅聳赂氓卤戮莽聼戮鹿忙聸垄忙垄禄忙碌拢茅盲赂虏莽录聛猫鹿虏聶茅聨篓潞忙聺聢芒聲聝莽聲露茅聴聛莽禄聵忙垄禄茅聧聵脩聡忙聜聻茅聧篓盲潞聹茅聳鹿氓聯聞忙拢聴忙碌聹茅鹿聝忙垄潞忙碌录忙卢聭氓聢掳茅聳赂茫聢隆莽聟隆茅聰聺莽聜虏芒聰聭茅聧芦莽聠路莽拢炉茅聴聛忙聝搂莽鹿聮茅聨鲁忙聮禄氓聼聙茅隆聮氓聜職盲禄卤茅聳鹿氓聵虏茅隆颅茫聞漏氓麓聵茅隆聨氓聸篓莽聟聶忙陇陇忙聴聜莽聯聰氓篓聢忙禄聢忙聼聼莽禄聸氓聹炉聙氓漏碌芒聙鲁莽聢录氓庐聲氓聽聲氓聼隆莽聙拢忙拢路氓漏碌莽聤碌忙聲赂茅聧聥莽聤禄氓鹿聯茅隆聫氓聴聶莽陇聤茅聳鲁脩聦氓聣職茅聤聡氓聥聤聫芒聲聜脨篓忙驴聻氓颅聵莽虏聦茅聨虏茂录聞芒聜陆聝氓聛聬茅聢搂莽篓聮茅隆颅氓聸麓垄忙碌聽氓忙陇陇忙聞漏氓聻潞忙戮露氓聥聼聼茅聬聼莽聣聢忙聬聬忙陇陇忙聸露聛茅隆垄忙聸垄氓搂聙茅聢潞盲陆鹿氓聙聴聜盲陆赂忙聠隆忙赂職忙氓麓鹿茅聧聴莽聜漏忙聻聰茅聳潞氓聸楼氓聻潞茅聢路忙聢娄忙聼聸氓篓聭忙漏聢忙聰禄莽录聛芒聜聧芦茫聝娄氓聛篓忙陇陇忙聽篓莽篓聯氓篓虏氓颅聵忙聟篓忙驴聽氓聥颅氓赂聸茅聳鹿氓鲁掳忙聡聵氓庐聲茫聞娄莽鹿聺茅聧聦忙露聶莽聲录茅聴聜氓聜聶猫聝露莽禄庐茫聙聝茅聧隆忙陇聲忙麓陋忙聜垄茅聭虏芒聲聛莽楼娄忙驴聽莽聰碌氓搂麓茅聧聼氓麓聲茅聬聰氓聟录莽聟聫氓漏碌莽聜虏莽聛聺茅聧聰茫聜聣忙聼拢?``train_ids`` 茅聴?``val_ids``茅聴?
    output_root : Path
        茅聴?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聵茅聧聤氓聺聴忙聼聸氓篓聭忙漏聢忙聰禄茅聳赂忙卢聫莽鹿聬茅聢鹿忙聢娄忙聜漏茅聨禄忙聨聠氓聛聯茅聧聸聻莽拢禄茅聳碌氓驴聥氓聙聳茅聬聯?氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄忙驴聻忙聢聶氓聻潞茅聧聥茫聜聣忙聼聲忙驴聻脩聝忙聡聨茅聫聠忙楼聟忙垄潞莽禄聥猫路篓莽聛卤茅聳赂氓卢篓聭茅隆聰氓驴聯芒聰聭莽聙拢氓聨碌莽录聜盲陆赂莽聙碌脩聡氓麓碌茅聧楼莽聟聸茅聬聫莽聜碌忙戮搂猫聦卢芒聜聻芦忙戮聵莽聙職忙聧垄氓漏碌氓莽陇聥忙陇陇忙聞碌氓聛聟氓漏垄忙聢娄忙垄禄茅聧聦忙卢聮氓聟聦茅聳鹿猫聶芦忙聧聡茅陋聻氓陇聬忙聲聯茅聬聵猫聦卢莽聲潞茅聴聛莽篓驴忙聹卢氓搂聵茫聞漏氓录鲁茅聰聲氓聜聸莽聟聼茅聳碌氓驴聲氓搂麓茅隆楼氓驴聸忙聼隆忙碌拢氓聭聤氓聳聵茅聳潞氓卤戮忙麓聺莽禄聽忙露聶莽鹿聸茅聨炉脩聜氓拢聲茅聴聛氓聭聤氓聤聳猫陇掳氓聠庐脣聣?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳赂忙驴聞氓聞鲁茅聬聮禄忙聜聴茅聴聢忙露聶忙聠赂茅聳禄忙聧聡茅聨鲁掳芒聳聨猫鹿聡忙聯聝盲禄鹿莽录?
    estimator_cfg : dict[str, Any] | None, optional
        茅聴?茅聴聜氓聜職氓聙赂茅聧聤茅拢聨忙氓搂聵猫路潞莽聞碌忙陇陇忙聨聭氓聙赂忙碌聽忙禄聢忙聼陇氓篓虏猫莽聛隆茅聳潞氓聭聢录茅隆垄忙卢聬忙聟篓茅聨掳莽聸聨茅聴聜盲陆鹿氓氓篓虏氓露聣氓麓聭茅聨戮氓聻聸氓聛聯莽聭聶氓聥卢氓陇聞茅聳赂忙聮露氓聳聳氓庐聲忙麓陋氓搂聙茅聤聢氓聭炉茅聳禄莽聤虏忙麓漏莽聛聫忙卢聯氓篓虏芒聮聭莽录聜盲陆鹿脙驴庐氓聽楼忙聼拢茅聨驴氓聥颅莽炉聙莽聙?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮盲禄聥忙聼陇忙驴聞卢芒聜聸聽忙卢垄茅聤聢氓聭聰陆茅聳禄忙篓潞莽聳聡忙戮鹿忙篓录莽麓聮茅聢搂氓戮聞茅聨掳氓聨碌茅聳潞氓莽聣聠忙戮搂盲禄聥忙聜戮茅聳露聠莽聟聲茅聬聫莽聜露忙驴庐氓聜聻忙聼隆莽聛聻莽聜卢莽陇聥莽聙鹿忙聸聽氓聨搂茅聢鹿忙聢娄氓麓露猫陇聫茅聬聸芒聲聝莽麓聯?茅聴?
    device : str
        茅聴?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聵茅聧聤氓聽聲忙垄潞茅隆聮芒聜卢莽禄聣莽聰赂忙聜隆氓篓聠忙聢聰氓聛聯茅聬聼忙聺驴茅聳赂氓茂鹿庐聲忙聸垄氓录庐茅聢搂茫聙?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓聙驴茅聳驴忙聴聢忙聲庐茅聧聥忙聮露莽虏篓聭忙篓禄氓聶陆茅聳禄莽聭漏茅聫聦莽聠赂忙陆聫忙楼拢氓聠漏茅聴聛? ``"cpu"`` 茅聴?``"cuda"``茅聴?
    epochs : int
        茅聴?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓聙驴茅聳驴忙聸聴氓聛聯茅聳驴忙聸聵氓聻陆茅聨庐脩聡氓麓录茅聤聫茫聢漏氓聲聨?茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅茅聢露猫聤楼莽篓聭茅聢陆氓陇聬氓茅聬聵莽聰碌氓麓露茅聴聜盲陆赂忙聬芦莽禄聣忙聼楼氓麓聺莽聙拢氓麓虏?
    batch_size : int
        茅聴?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聵茅聧聤氓聺聴忙聼聸氓篓聭忙漏聠莽聟驴聻氓聽聹莽虏聶茅聫聦茂陆聣氓鹿聡茅忙掳卤忙茅聳潞盲陆赂莽聣聯氓猫聶鹿忙聥聽茅聢楼猫聶芦莽聛聯茅聴聛莽篓驴莽鹿聭茅聰聲茫聢陇莽聧驴聽氓聜陇莽聟楼茅聳赂氓驴聲氓搂麓莽录聧氓漏聡氓鹿聝氓漏聤氓聽聼氓職聧茅聳碌氓拢聠篓莽庐潞茅聴聜氓聜職茅聧聥氓聥潞茅聰聲氓聜職莽茅聳赂忙聮聴氓聭卢聣茅聴聛猫茅聧聽忙掳卤氓拢聲忙驴聜氓聭聤氓聛拢茅聳赂茫聝庐盲潞聹茅聬篓氓聯聞猫莽禄聽忙聮禄氓录禄氓篓聭茫聢聽莽聟聵茅聳鹿氓聜職?
    eval_batch_size : int
        茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聟忙職娄茅聳赂氓聛聟茅聧聶氓驴聸忙聼隆忙戮露氓忙庐驴聻氓路潞聹氓聸麓茅聴聛忙聬聻氓聞禄莽禄虏猫聤楼莽娄聧茅聬聬氓聫聫忙驴庐忙楼聙忙拢聴忙戮搂盲录麓忙聼拢茅聨戮忙聺聭莽聛娄莽录聛莽禄聵莽鹿聢茅聨禄聥忙拢聝氓篓麓猫潞虏氓聻潞莽禄禄忙露職盲禄娄茅聧聦忙掳拢氓聛聯氓娄陇氓聠篓氓聻聺氓漏碌莽聤鲁猫聣戮氓聰聴聛氓聠虏忙聬芦茅聧聤氓氓麓垄忙碌聽氓卢聮聭莽录聜盲陆鹿脡聭茅隆楼氓聴聵茫聛聞茅聫聡氓聴聴忙麓娄茅聧聤氓聺聴忙聼娄氓娄炉盲戮拢娄茅聳禄莽聠录氓聛聤茅聬聯茫聝漏忙聼聼茅聳鹿聙忙聻陇莽聬職茂鹿聜忙垄禄忙碌拢莽颅聥氓職聝茅聳赂莽聤鲁茅聢楼茂鹿聜氓麓录茅聤聫莽卢聣氓漏碌莽聜麓氓聻聧篓氓聹虏禄忙驴聽忙掳颅莽聟聲茅聳鹿忙聧聬氓掳陋茅聧聫氓虏聞忙聼隆忙戮露氓陇聦莽聼聶忙驴?
    model_overrides : dict[str, Any] | None, optional
        忙驴聽莽聰碌氓搂路茅聫聛氓聭聤忙聟篓茅聬聭氓芒聜聜聻忙聦聤忙戮鹿芒聲聥莽露聛氓庐聲茫聞漏氓录露茅聨麓忙驴聢芒聜卢茅聤聢氓聭炉莽庐掳茅聳禄忙楼聙忙拢聺茅聨庐莽聜潞忙聜陆茅聬垄氓聻芦茂录聫莽录聜氓聜聶茅聶聧茅聧篓莽聭掳茅隆聲芒聜卢茅聫聦忙露垄氓聼聻茅聨炉脩聜氓拢聲茅聴聜氓聜聶猫聝露茅隆聽垄忙聧垄茅聨麓茂赂陆忙聶聺茅聳碌氓驴聲氓陋聧陇忙聸聽莽麓聯忙碌拢猫莽聞陆莽录聛芒聶聙氓聛聯茅聴聜盲陆潞茅聬卢氓楼赂氓麓聭氓篓聭茫聢隆氓聙路氓漏碌莽聤虏忙麓娄茅聬聯氓聠庐忙聼聸氓漏碌氓聴聴茅聳鲁脩聦氓聣聶茅隆颅莽聝陆氓麓潞茅聢搂茫聛聞茅聨潞忙聢颅忙搂赂莽聙碌氓聳聴茅聨卤茫聞娄莽鹿聺茅聧聦忙露聶氓聞聯茅聳潞盲陆赂莽聣聯茅聫聦忙露垄氓娄路茅聨麓忙驴聠忙聻陇茅聳禄氓潞垄氓篓聤氓搂?氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聧聸氓漏聝氓聫聫氓篓麓氓聟录忙陆聶茅聧篓氓聜職芒聰聭茅聧聦忙禄聨莽聞戮茅聳潞氓聥漏氓聞碌茅聫聦芒聙鲁忙聬麓芒聙鲁莽茅聴?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聳莽潞戮氓聺聴忙聼拢茅聤聫茫聢漏莽聞戮茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛莽篓驴茅氓陇聬氓录禄茅聰聺氓聭聜莽聠路芒聳聨茅聨聞茅聳潞氓陇聠忙聶聹莽禄禄氓聽楼氓鹿聝氓娄陇氓聭碌聛茅聳赂氓聳聳忙驴潞隆茅聤聢氓驴聯莽聵聹茅聳赂茫聞搂忙聲赂茅聨?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    Any
        茅聴?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬氓庐聞忙聡聤氓陆聮氓麓露茅聢鹿忙聝搂盲潞戮忙驴庐忙漏聠氓聛聞氓庐聯氓聴聵盲驴聤茅茂录聞氓聤聧莽聙碌忙聺驴氓芦禄禄忙露垄氓鹿聮茅聨麓莽聤碌脨篓茅聴聛猫陆掳莽聞娄茅聧聰忙卢聫莽聲路忙驴聜莽莽聟聭茅聳鲁猫陆掳忙忙戮鹿忙聸垄忙垄潞茅聧聯猫聦聨聸茅聳赂氓忙拢聺茅聨炉氓卤陆莽聲聺茅聢搂忙茅聨潞忙聮麓莽露聡茅聳碌氓篓驴氓聞虏莽聮聡莽聜虏脦聰茅聬聵茅聺聸盲禄娄茅聳赂茫聞楼忙陆隆茅聬聸氓麓露茅隆聮氓忙聶垄茅聴聛茅聙聻氓卤陆氓垄聳鹿盲戮驴氓聤聧茅隆娄莽聣聢忙聝聰茅聰聺氓聸路莽麓鲁氓漏碌莽聜露忙聫录聧芒聜驴聻氓聮聝莽碌驴茅聫聛芒聵聣茅聤聫茂录聞氓聨卤茅聴聛茅聺聸茅聧聨猫聶茅聤聫茫聜聠莽聟聼茅聳驴忙驴聠忙麓陇茅聧聵氓芒聜忙麓聳茅聤聢莽篓驴茅聨戮氓聛聞茅聬聫氓驴聨莽忙碌聹茅鹿聝忙聼拢茅聨麓茂陆聟脨聯茅聳禄忙聮麓忙麓聼茅聫聦忙聸聼莽鹿聺猫鹿聡忙露聶氓茫聜聠忙聼篓莽禄禄忙聨聰忙搂聬茅聨戮猫炉虏莽拢录忙驴漏聠氓聟聵茅聧聥忙聮禄氓鹿聳忙碌拢氓聯楼茂录聶茅聴聛莽禄聵氓聻录忙驴聜聣氓麓聬茅聧搂忙楼聛莽聟楼茅聳潞氓聸漏氓聛聸茅聢搂忙聭聹莽禄庐氓漏職氓录露忙聬麓忙聮麓忙聳聙茅聴聛莽禄聵脙陋莽潞戮茫聙聜茅聫聦茫聝娄莽聟聲氓漏碌莽聤虏忙聡聫莽潞颅莽聜聛聞茅隆聯芒聞聝莽隆露茅聳鹿莽聭掳氓氓庐聲氓麓聼茅聧陆聣忙垄禄忙碌拢盲戮聞戮忙陇陇忙聢聺忙拢聺茅陋聻忙聢娄氓麓露茅隆聮芒聜卢莽禄聽忙聽颅忙聼娄氓娄炉盲戮搂麓茅聳潞氓忙拢聝茅聫聦莽聠潞莽虏聶茅聧篓氓聤聣茅聴聛氓聯聞忙拢聸茅聧聥茫聜聟莽录聛忙篓禄氓茅聳陆忙篓潞茂录聧茅聨戮莽禄聵莽聟聼忙驴隆茫聜聞莽聞娄莽禄聙氓聽楼忙聼聧猫陇聹茅聧聯忙掳聤聤茂赂陆忙芦聽忙驴隆茫聜聞猫聝露茅聢鹿?

    茅聴聜氓聜職氓聙路氓篓麓氓聸茅聢搂莽篓聢茅聫聟莽聤禄茫聙聛氓篓聠忙聮鲁氓聠聙忙陇陇忙聮露氓聛聼茅聬聸茫聝漏忙垄潞茅聧聺氓聥颅芒聳聣茅聳赂忙篓驴氓聛聬忙戮鹿忙聸隆茫聛聬茅聨潞忙聮露氓聨碌茅聴聜氓聜職氓聙赂茅氓麓聺氓氓陇聥莽聟聲?
    ------
    RuntimeError
        茅聴聜氓聜職氓聙路氓篓麓氓聸搂莽聲路茅聬垄莽莽陋聴茅聳鹿茅聜娄氓聳聳茅聧篓忙驴聥氓鹿聳氓篓拢莽聛驴聻氓聭搂碌忙路聡氓漏聡氓娄露茅聧聸忙芦拢莽录聜盲陆潞氓搂鲁茅聧聴忙聤陆氓录禄茅聬聰氓聟录芒聮聮茅聬聨茅聺聸氓拢聤氓娄虏忙垄潞茅聨录莽聜虏氓茅聳赂氓漏聜忙陆隆茅聨庐茂驴聽茅聰聲芒聜聬聬氓漏聡忙聼聲忙驴聻脩聙氓聛聯忙驴隆忙聫聮莽聣聫莽禄卤忙聮麓忙聥聟茅聨禄忙聨聧氓茅聴聜盲陆潞莽隆露茅聧聯忙驴聢莽虏聶茅聨潞忙氓聛聜茅聳潞氓聸楼氓聻潞茅聬聯氓聠庐忙聼聧忙聺聻忙聣聳潞氓聴碌戮茅聤聡氓聥炉氓搂聙茅聢漏氓聠戮忙拢聝氓娄陇莽聤碌氓聛聻茅聬聴忙氓鹿聫茅聧聸忙聺聭忙聧露忙陇陇忙聮露氓聛聸莽禄聽莽聜虏芒聰聭茅聬聵猫聦卢忙拢聞茅聳潞氓陇聤莽掳卤茅聧聥忙聮禄氓鹿聵莽录聛忙篓潞芒聜卢氓聺聴忙聼篓茅聫聜氓聻庐芒聮聰茅聳禄忙拢聺茅聫聦氓露聢茅聳赂忙聮麓莽篓聯氓娄虏忙聞掳氓鹿聵莽聙聸忙篓潞茅聴聛猫聧陇氓聳聳茅聧聥氓漏聡氓录鲁茅聤聫茫聜聠芒聮聭茅聳赂忙驴聠氓茅聧拢猫聴聣芒聰聭茅聬聬芒聲聛盲潞戮忙驴隆茫聜聡氓搂路茅聧聥忙露垄氓麓炉茅隆聳忙禄聟莽聼聣茅聳鹿莽聝聵忙聼隆茅聧聥忙聮禄忙聲聬忙聬麓忙篓潞盲潞戮忙陇陇忙聮麓莽篓聮氓篓聟氓漏聡忙聼隆莽聛聻莽聜卢莽陇聥莽聙鹿忙聸聽氓聨搂茅聢鹿忙聢娄氓麓露茅聤聤茂鹿聙芒聜聝陆忙垄?prepare 忙驴聽莽聰碌氓搂路茅聫聛氓聭聤忙聟篓茅聬聭猫聴聣忙聻聛茅聳鹿茅聴麓莽莽潞戮氓忙聼拢茅聨掳忙聝聢莽录聛莽聤卤氓聧鲁茅聫聦莽聠录忙垄禄莽聙碌氓聣虏忙搂录聜盲陆潞氓搂路忙驴聻芒聜卢忙楼聽莽聜麓莽聣聴氓氓驴聸氓麓聹猫陇聫茅聰聸莽聠录忙垄潞茅聧聺氓聥卢茂录聰莽录聛莽聤鲁忙聧聡氓炉忙聲聬忙戮露氓漏聝茅聴聛氓聹聙聸芒聲聟氓麓掳茅聨掳莽聣聢芒聮聭?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓聙驴茅聳驴忙聸聻氓聙露茅聢搂氓聲麓氓庐聲茫聞搂芒聜卢忙露聶茅聧聵忙聮禄忙聜路氓漏聤氓聠陋莽庐禄茅聳赂芒聲聛茂鹿陋氓炉聮庐氓搂聙茅聰聸氓聥芦氓录鲁忙驴聽忙庐驴氓聳聴茅聰聲芒聲聟氓麓垄茅聧聯脩聝氓搂聳鲁脩聦氓聣聼氓忙麓聼聟茅聬聬猫聦露氓芦掳氓脩聝莽聯篓茅聤聡氓聥茅聢漏氓聠戮忙拢聝茅聬聨猫搂聞忙麓聳茅聤聢忙聬聛氓鹿聝茅聤聫莽聤碌忙職聥茅聳潞氓氓娄聥炉忙垄禄茅聧聦忙掳聜卢茅拢聨莽虏聢忙赂職芒聜卢茅陋聻氓氓聻戮莽聯聮盲陆鹿忙陇陆茅聫聛氓聠庐氓麓聮氓忙聠聥莽麓卤茅聴聜盲陆潞莽隆露茅聧聯忙驴聥忙聜路茅聤聣芒聲聛氓聡聽茅聳赂忙聡聡茅聧聥忙聮禄莽聧庐茅聧篓氓氓娄聻茫聜聫莽录職茅聧聴忙聤陆氓鹿聝茅聰聼莽聤虏氓聧鲁茅聳碌氓驴楼莽麓聲茅聧聵莽聰赂忙垄潞莽录聛忙篓潞莽聛娄茅聮聝茅聛聧忙聼聼茅忙掳录聛莽禄聵莽聸炉氓庐聲氓楼赂忙聜垄茅聧聺茫聝搂掳氓漏碌莽聝聡莽碌陆氓篓虏茫聜聣忙聲聝茅隆聫氓聽聺莽庐聳忙驴聻氓聴聵忙聦赂莽禄聽莽聰赂忙聼聼茅聧聺氓聥篓麓忙禄聢莽庐聳茅聫聦忙露聹忙陇陇茫聞楼莽聣聠茅聬拢氓聻卢聼茅聢搂莽聧庐茅聫聧氓聻聺茅聬聼忙卢聫氓芦路茅聢搂忙拢聶忙路聡氓漏聡氓鹿聯茅聨潞忙聳驴忙驴聻茂陆聟忙麓陇茅聰聲茫聙聛氓篓聭忙漏聠莽聟楼茅聳赂忙聞漏氓聤聦茅隆卢氓聴聸忙垄禄忙碌拢茅盲赂虏茅聴聜氓聥猫聻聻茅聳赂忙聞碌莽聟聢氓篓?
    """
    split_ids, events_by_seq_id, ground_truth_by_seq_id, source_report_by_seq_id = _load_training_payload(  
        prepare_root,  
        raw_root,  
    )  
    if set(split_manifest["train_ids"] + split_manifest["val_ids"]) - set(split_ids):  
        raise RuntimeError("split_manifest references seq_ids missing from prepare manifest")  
    model_cfg = _resolve_train_cfg(  
        model_name,  
        epochs=epochs,  
        batch_size=batch_size,  
        eval_batch_size=eval_batch_size,  
    )  
    if model_overrides:  
        model_cfg = _apply_model_overrides(model_cfg, dict(model_overrides))  
    return TrainPipeline().run(  
        {  
            "project_root": ROOT,  
            "model_name": model_name,  
            "model_cfg": model_cfg,  
            "estimator_cfg": copy.deepcopy(estimator_cfg) if estimator_cfg is not None else None,  
            "dataset_name": "sim",  
            "split_ids": split_ids,  
            "train_split_ids": list(split_manifest["train_ids"]),  
            "val_split_ids": list(split_manifest["val_ids"]),  
            "forbidden_training_split_ids": list(split_manifest.get("test_ids") or []),  
            "events_by_seq_id": events_by_seq_id,  
            "ground_truth_by_seq_id": ground_truth_by_seq_id,  
            "source_report_by_seq_id": source_report_by_seq_id,  
            "output_root": str(output_root),  
            "mode": "full",  
            "device": device,  
        }  
    )  


def _build_model_cfgs_from_train_reports(train_results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄忙驴聻忙聢聶氓聼聞茅聫聛氓茫聛聞茅聨戮猫路潞忙聻聭茅聳赂忙卢聫莽鹿聵茅聫聦芒聞聝氓戮聞莽聙拢氓聤赂氓漏?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳鹿茅聴麓莽莽潞戮莽聭掳莽聙拢忙聧拢茅聳禄忙茅聤聠忙聨聭茅聰聺氓聭卢氓拢聲茅聳禄氓潞陇氓篓虏茂鹿聜氓麓鹿茅聧聸聽氓碌聛莽聙鹿茫聝楼茅聫聛氓漏聡忙聼拢茅聨戮猫聶聧拢氓篓麓忙禄聙氓聧鲁氓莽禄聵莽卢聼茅聢搂猫忙戮搂忙驴聥忙垄潞茅聧聺氓聥聶潞茅聳禄芒聲聤氓聻掳茅聬拢氓鲁掳芒聳聨茅聨戮忙聺聭盲潞鹿茅聬聨猫搂聞忙麓聳氓篓虏茫聞漏忙聝聛茅聧芦莽聠潞莽炉聙茅聳禄茫聢陇忙陆搂氓颅聺茅聴聛莽茂鹿娄莽禄庐氓录虏茅聧芦忙聸聻氓聧鲁茅聳禄忙聴聡忙芦拢茅忙聸垄忙垄潞茅聧聸氓漏聞莽聼聤茅聳赂莽聠赂氓氓炉聡氓聼聙茅隆聮氓聥卢芒聮聭茅聳赂忙驴聠氓茅聬聫茅聛聧忙聼聸茅聬聵氓麓聡氓垄碌茅聳碌氓聴聲莽陇聛猫聻拢氓篓聯忙掳录氓聻鲁茅聧聶氓聴聸忙垄潞茅聧聸氓漏聜氓搂娄氓篓麓忙禄聞莽聭漏茅陋?忙驴聽莽聰碌氓搂路茅聫聛氓聭聤忙聟篓茅聬聭猫聴聣忙聻聛茅聳鹿茅聴麓莽莽禄聙氓聴聸忙聼聼氓篓聢氓聻庐忙聻陇莽禄戮忙聝搂茫聜赂茅聨陆聮氓麓露猫茅聢禄忙聸垄忙聼聸忙聬麓茫聢隆茅聳潞氓虏聥莽麓聲忙碌聽茂鹿聜氓麓聬茅聬聰氓楼聣盲录聝忙驴隆莽聜陋氓聙陇茅聧聵氓陆聮氓麓聺茅聨麓忙驴聢忙職娄忙驴隆茫聜聟忙聡聨莽禄聙氓聽垄忙聠赂氓庐楼氓氓聡聽忙驴聻忙聢聶氓聻潞茅聬聯忙聸垄忙聼聲忙驴聽氓驴聲莽聲卤茅聳赂忙聳禄氓聙聺茅聫聦茂陆聣忙聲聬茅聧楼脩聝氓鹿聥茅聬聨猫忙麓聳茅聬聳氓楼赂氓录芦茅聨掳忙聺驴氓路聺忙陇陇忙聨聠氓聛聯忙陇陇茫聝娄氓虏赂氓莽禄聵忙聥聟茅聯聫莽聭掳莽卢隆茅聴聛忙聬聻氓聤聦茅聧職氓氓鹿聢茅聤聤芒聲聛脦搂氓猫路潞茂鹿娄茅聰聸莽聠录忙聜聴茅陋聻氓聻聤職忙陇陇茂赂驴氓聧鲁茅聨庐茂驴聽氓录麓茅聤聫茂录聞氓聨赂茅聴聛忙聬聻氓聞炉茅聧聨猫戮篓盲驴聤茅聬聯氓潞聠聠莽聭聹忙禄聢氓麓鹿忙碌录氓颅聵莽庐聮茅聴聜盲陆潞莽虏聯卤茅聳赂忙聮麓氓虏赂茅聨庐莽聜潞忙聜戮氓庐聙芒聜潞戮氓楼赂氓录露茅聧露聵氓娄陆茅聬聫氓驴聨氓聧鲁茅聫聦莽聠路莽聲隆茅聬聴氓聽聺忙庐聴茅聬聨猫忙麓娄茅聧聥氓漏聝盲驴聤茅聬聭猫聴聣脦篓茅聳碌氓驴聲氓搂麓莽禄露脩聡忙垄禄茅聧聦忙掳聜卢茅拢聨莽虏聢忙赂職芒聜卢茅陋聻氓忙陋卤莽聭聹忙聣篓聙氓搂聤猫聶鹿莽虏聶茅聧聳芒聞聝忙聲戮氓娄聻茫聜聝氓聤聦茅聬聴氓聴聴氓碌聞?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬氓庐聞忙聡聤氓陆聮氓麓露茅聤聤茂陆聞莽虏聯茅聴聛氓聭聤莽赂驴茅聧聨忙聫聮莽聲路茅聫聦茫聝娄莽聟聲茅聳鹿忙聺驴氓聬聙莽禄篓莽聲聦莽麓聮茅聬聵猫聶聳潞氓虏聥莽露聛氓炉聧聞茅聤聤茂赂禄氓聰赂忙驴聽莽垄聵氓聣聺猫陇掳氓聠庐忙聜搂忙驴聽氓聠聰聧氓漏碌莽聤虏忙碌聜茅聫聛氓露聠莽鹿聺茅聧篓氓忙驴聻氓聽芦氓聫聝氓忙麓聼聟茅聬聬猫聦露氓芦掳氓脩聜氓聧鲁茅聫聦芒聞聝忙聥聟茅聧聸氓漏聝氓聳聬氓漏碌?茅聴聜氓聜職氓聙路茅聬聮茂鹿聜忙聝聡猫陇掳忙聨聭氓聻聜茅聳录氓搂聬氓聙聴茅聬颅忙聮麓莽鹿聸氓庐赂莽掳录茅聳禄忙聮麓莽聭娄茅聤聡氓聥录庐茅聧楼脩聜忙卤聴茅聴聛莽鲁聲忙聶禄虏氓聺聴氓聼聙茅隆聮氓聠戮忙陆聬忙驴聻氓聫聣莽聣聲茅聬拢莽聝陆茅聧聬莽掳路忙驴聽莽聰碌氓聛聽茅聨禄氓戮聤忙楼拢氓聠篓莽庐聳茅聳赂氓虏聙氓聛聸茅聮聝莽聠录忙聼篓氓漏碌氓聴漏忙搂鹿茅聳赂氓卢聤聶茅聫聦忙露聵芒聳聨猫鹿聡茂陆聠脨搂茅聴聛猫炉隆氓聳聴茅聬聼芒聲聟氓录禄茅聰聺氓聴聵氓聞聼茅聳碌氓驴楼莽碌陆茅隆楼忙驴聥忙垄潞氓篓虏猫氓鹿聴忙陇陇茫聞楼忙聡聵茅聧漏莽聜潞氓鹿聥茅聰聰猫聴聣盲潞鹿茅聴聛氓聹颅莽虏聰炉忙楼聙芒聳聯茅莽聤碌莽拢陆氓篓麓茂陆聡茅隆拢忙聤陆忙聼聸莽聙拢忙聻聸茅聳赂芒聰戮氓聛聯氓茫聜聠氓路禄茅聧聥忙聮露莽麓聮茅聬聵猫聦搂莽聙鹿忙聸聼忙麓聼氓庐聲茂驴聽忙聜聵莽录聛忙篓禄茫聛聺茅聳潞盲陆赂氓聲麓氓庐聲忙聨聭茅隆聯茂录聞氓聫聭茅聴聜氓聜職茅聧聥氓聥潞茅聰聰氓聣搂莽聼聶茅聳鹿莽聝聵茅聧?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    train_results : dict[str, Any]
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓麓茂陆聣忙陆禄茅聭掳氓聺聴氓录露茅聧芦忙露聵氓娄赂氓篓麓忙禄聞莽虏聯茅聫聦茫聞漏氓录庐茅聧楼忙氓娄聻茫聜聟氓颅漏莽禄庐忙聮禄氓录禄氓篓聭茫聢隆氓聛聠氓篓麓茂陆聟忙聭聹忙碌聽莽芒聰聭茅莽隆路忙聰禄忙驴隆莽聜掳莽虏聶茅聬聸氓录陆茅莽聠路芒聜卢莽聝陆忙聼聼莽录聛忙篓潞盲驴聳禄氓潞陇忙芦聲忙路聡氓漏聡忙聜聶茅氓聥篓氓聛聯茅聫聧氓聻聜茅聳禄茫聢垄莽禄聽莽聤鲁忙聟篓氓猫炉虏盲潞戮茅聴聛莽禄聵盲戮聴茅聧拢茂陆聟莽聲路氓搂聴芒聜卢茅隆垄忙卢聯忙聜隆忙聬麓茂录聞茫聜庐茅聴聜氓聜聶猫聝露莽禄庐忙聭驴聽莽聰碌氓聛聸茅聰聲忙驴聽忙掳聧鲁茅聳驴忙露聵氓芦聫聣忙聬聛忙垄潞茅聧聺氓聥卢氓路聺茅聳赂氓漏聨氓聛聤忙驴隆莽聳炉莽禄戮忙聝搂茫聜赂茅聨庐忙漏聙忙聜聶莽聮潞莽聸聮忙聮禄忙聼聸莽聙拢茫聞楼氓聤娄茅聳潞氓卤赂芒聜卢氓麓聡忙職聠茅聬聨茂陆聨氓聣聸猫垄娄氓漏碌莽聤碌茅聧聯忙驴聥氓鹿聝茅聧聦忙露聴莽鹿聠?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳鹿茅聴麓莽莽潞戮莽聭掳莽聙拢忙聧拢茅聳禄忙茅聤聠忙聨聭茅聰聺氓聭卢氓拢聲茅聳禄氓潞陇氓篓虏茂鹿聜氓麓鹿茅聧聸聽氓碌聛莽聙鹿茫聝楼茅聫聛氓漏聡忙聼拢茅聨戮猫聶聧拢氓篓麓忙禄聙氓聧鲁氓莽禄聵莽卢聼茅聢搂猫忙戮搂忙驴聥忙垄潞茅聧聺氓聥聶潞茅聳禄芒聲聤氓聻掳茅聬拢氓鲁掳芒聳聨茅聨戮忙聺聭盲潞鹿茅聬聨猫搂聞忙麓聳氓篓虏茫聞漏忙聝聛茅聧芦莽聠潞莽炉聙茅聳禄茫聢陇忙陆搂氓颅聺茅聴聛莽茂鹿楼茅聨庐氓聜職茅聨露猫聤楼氓聛聞茅聳赂氓驴職氓聛聬茅聧聵莽聰赂忙垄潞忙碌录忙卢聭氓聢掳茅聳录忙麓禄氓聻露氓炉芒聵聣茅聤聫茂录聞氓聨赂茅聳禄氓驴聲氓聛聸忙戮搂盲禄聥芒聙聳茅聭虏芒聲聝莽拢录茅聳禄忙篓潞莽拢篓虏氓炉赂芒聜卢忙庐驴氓聶庐茅聧拢茂陆聣氓麓潞茅聢搂茫聛聞茅聨潞忙聢聺莽禄戮忙聝搂茅聫聦茂陆聟脦聼茅聭虏芒聲聛莽茅聴?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    dict[str, dict[str, Any]]
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓麓茂陆聣忙陆禄茅聭掳氓聺聴氓录露茅聧芦忙露聵氓娄赂氓篓麓忙禄聞莽虏聯茅聫聦茫聞漏氓录庐茅聧楼忙氓娄聻茫聜聟氓颅漏莽禄庐忙聮禄氓录禄氓篓聭茫聢隆氓聛聠氓篓麓茂陆聟忙聭聹忙碌聽莽芒聰聭茅莽隆路忙聰禄忙驴隆莽聜掳莽虏聶茅聬聸氓录陆茅莽聠路芒聜卢莽聝陆忙聼聼莽录聛忙篓潞盲驴聳禄氓潞陇忙芦聲忙路聡氓漏聡忙聜聶茅氓聥篓氓聛聯茅聫聧氓聻聜茅聳禄茫聢垄莽禄聽莽聤鲁忙聟篓氓猫炉虏盲潞戮茅聴聛莽禄聵盲戮聴茅聧拢茂陆聟莽聲路氓搂聴芒聜卢茅隆垄忙卢聯忙聜隆忙聬麓茂录聞茫聜庐茅聴聜氓聜聶猫聝露莽禄庐忙聭驴聽莽聰碌氓聛聸茅聰聲忙驴聽忙掳聧鲁茅聳驴忙露聵氓芦聫聣忙聬聛忙垄潞茅聧聺氓聥卢氓路聺茅聳赂氓漏聨氓聛聤忙驴隆莽聳炉莽禄戮忙聝搂茫聜赂茅聨庐忙漏聙忙聜聶莽聮潞莽聸聮忙聮禄忙聼聸莽聙拢茫聞楼氓聤娄茅聳潞氓卤赂芒聜卢氓麓聡忙職聠茅聬聨茂陆聨氓聣聸猫垄娄氓漏碌莽聤碌茅聧聯忙驴聥氓鹿聝茅聧聦忙露聴莽鹿聠茅聳禄忙聢拢忙拢聞氓聰炉茅聴聛忙聦聨忙聼篓忙戮搂盲禄聥茅聫聡茫聢隆芒聮聭茅聬聼忙卢聫氓芦聧聰茫聜聣忙聼聸茅聬聰茅聰聥氓聛楼茅聳碌氓聴聴氓聙赂茅聬拢莽聝陆氓麓露茅聢潞氓聠驴莽麓虏茅聴聛氓聯聞茅聬聴氓聥麓聺莽聛聻莽聼芦氓陆聮茅聳驴盲潞戮茅聬聼忙卢聫氓莽禄聙芒聜聴聛茅聺聸莽聣聤茅聨聜聻莽聧聧聬忙陇陇氓聨陆茅聧聲氓聠庐忙垄潞茅聧聫猫聜漏茫聜麓茅聳潞氓聭庐莽鹿聢茅聨庐盲录麓氓茅聤聢氓聴聴芒聜聰赂忙聜路氓篓聠氓驴聯莽录聛芒聜聢搂芒聰聭茅聬聵盲潞戮茅聴聜盲戮聡氓聣聶莽禄聣忙聮庐忙聜聻茅聧篓盲潞聹茅聳鹿莽聝聵氓聼聤茅聧聰莽聠潞莽麓聮忙戮露氓卤戮忙職卢茅聳潞氓卤戮莽聸聨庐芒聲聛莽聲路茅聧楼脩聙忙聥隆茅聴聜盲陆陆忙隆篓莽禄露茂驴聽氓麓掳茅聫聧氓聛漏茅聳禄忙聢拢忙茅聬聬氓漏聡忙聼聧茅聧聺氓聥聻娄莽录?茅聴聜氓聜職氓聙赂茅聧聤莽聝陆忙聡聴茅聧聸聻莽拢驴茅聳赂氓聛聟茅聧聶氓驴聲芒聜卢氓鹿驴氓聞卤茅隆娄忙聜聶忙驴聽氓聥卢氓聳聬茅聼芦氓聸篓猫聸聥茅聧楼忙篓聞氓漏碌茂鹿篓氓篓聟茂陆聟芒聵聣茅聬垄莽篓驴氓路聺忙陇陇忙聮麓莽鹿聝茅聬芦氓驴聸忙垄禄忙碌拢盲戮炉莽聞戮忙陇陇忙聢聺忙氓庐聲忙驴聠芒聳聨猫鹿聡忙聸聼忙?
    """
    model_cfgs: dict[str, dict[str, Any]] = {}  
    for model_name, train_result in train_results.items():  
        train_report = dict(train_result.metadata["train_report"])  
        try:
            checkpoint_payload = torch.load(train_report["checkpoint_path"], map_location="cpu", weights_only=True)  
        except pickle.UnpicklingError as exc:
            raise RuntimeError(f"unsafe checkpoint: {train_report['checkpoint_path']}") from exc
        model_cfg = copy.deepcopy(  
            checkpoint_payload.get("model_cfg")  
            or _load_yaml_config_cached(str((ROOT / "configs" / "models" / f"{model_name}.yaml").resolve()))  
        )  
        model_cfg["checkpoint_path"] = train_report["checkpoint_path"]  
        model_cfg["project_root"] = str(ROOT)  
        inference_device = _resolve_cuda_inference_device_from_train_report(train_report)  
        if inference_device is not None:  
            model_cfg["inference_device"] = inference_device  
        model_cfgs[model_name] = model_cfg  
    return model_cfgs  


def _resolve_cuda_inference_device_from_train_report(train_report: dict[str, Any]) -> str | None:  
    
    
    
    return None  


def _candidate_model_cfg_from_checkpoint(  
    model_name: str,  
    checkpoint_path: str,  
    *,  
    inference_device: str | None = None,  
) -> dict[str, Any]:  
    try:
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)  
    except pickle.UnpicklingError as exc:
        raise RuntimeError(f"unsafe checkpoint: {checkpoint_path}") from exc
    model_cfg = copy.deepcopy(  
        checkpoint_payload.get("model_cfg")  
        or _load_yaml_config_cached(str((ROOT / "configs" / "models" / f"{model_name}.yaml").resolve()))  
    )  
    model_cfg["checkpoint_path"] = checkpoint_path  
    model_cfg["project_root"] = str(ROOT)  
    if inference_device is not None:  
        model_cfg["inference_device"] = str(inference_device)  
    return model_cfg  


def _read_metric_rows(metric_table_path: Path) -> list[dict[str, Any]]:  
    with metric_table_path.open("r", encoding="utf-8", newline="") as handle:  
        return list(csv.DictReader(handle))  


def _read_metric_table_views(metric_table_path: Path) -> dict[str, list[dict[str, Any]]]:  
    single_run_rows = _read_metric_rows(metric_table_path)  
    repeat_summary_rows = build_repeat_summary_rows(single_run_rows)  
    scene_summary_rows = build_scene_summary_rows(repeat_summary_rows)  
    return {
        "single_run_rows": single_run_rows,
        "repeat_summary_rows": repeat_summary_rows,
        "scene_summary_rows": scene_summary_rows,
    }


def _mean_metric_for_method(  
    metric_rows: list[dict[str, Any]],  
    *,  
    method_name: str,  
    metric_name: str,  
    scene_filters: tuple[str, ...] = (),  
) -> float:  
    values: list[float] = []  
    for row in metric_rows:  
        if row.get("method_name") != method_name or row.get("metric") != metric_name:  
            continue  
        scene_id = str(row.get("scene_id") or "")  
        if any(filter_token not in scene_id for filter_token in scene_filters):  
            continue  
        values.append(float(row["value"]))  
    if not values:  
        raise RuntimeError(  
            f"missing metric rows for method={method_name} metric={metric_name} filters={scene_filters}"  
        )  
    return float(sum(values) / len(values))  


def _parse_scene_axes(scene_id: str) -> dict[str, list[str]]:
    """茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聫茅聢搂莽聞娄莽禄禄忙驴聥氓麓聮氓娄陇盲陆潞盲潞聶忙驴聽莽聰碌氓聛聻茅聧篓氓麓鹿氓篓虏莽聳戮忙驴聽茅聺聸茅聬聯氓驴聸忙聼聸茅茂陆聡莽庐聳茅聳赂茂陆聠氓篓聤茅聫聦茂驴聽氓聼聙茅隆聮氓聥卢忙聶聺茅聳赂氓卤戮莽篓聯茅聧聵忙聮禄忙聜路氓氓聠聫聞茅聳?ID 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳禄忙聢拢氓搂陇茅聧聤氓聺聴忙聼篓茅聫聡忙掳露忙氓篓麓忙禄聝莽露聤茅聫聦忙露垄氓鹿聡茅聴聢忙露聶莽聛聧茅聴聛莽篓驴氓氓娄聠庐氓录禄茅聰聺氓陇聤莽莽聙鹿忙聸聻忙麓陇茅聧聺忙麓垄潞茅聧聺氓聥碌氓聞聬忙驴隆氓聲麓氓氓漏職忙聜垄莽聬聸芒聙鲁盲潞戮茅聳禄茫聢隆茅聬聮忙聝搂脙鹿茅聬聶氓聭炉莽聲卤茅聳鲁猫搂聞氓聻驴茅隆垄忙卢聯忙聜戮氓庐聙芒聜聬聯氓聜聹莽麓聯忙碌拢盲禄聥氓聯潞茅聳鹿莽聭掳忙麓陇茅聬拢茅聰聥莽赂陆莽禄聥忙卢戮忙陆聤茅聴聛莽禄聵脡聭茅聰聲忙茅聫聡莽聤鲁茅聳鹿莽聝聵忙垄鹿莽聙職忙掳露忙聼聸氓篓聠氓驴拢忙篓聢忙驴隆莽庐聯氓搂?忙驴聽莽聰碌氓搂路茅聫聛氓聭聤忙聟篓茅聬聭猫聴聣忙聻聛茅聳鹿茅聴麓莽茅聫聞茫聝漏忙聼篓氓漏碌氓聴聵莽茅聳赂氓忙聦戮茅聨虏忙聮麓聼茅隆聮氓聜陇忙搂掳茅聴聜盲陆陆氓麓碌茅聧聼氓聺聴氓录虏茅聬聵猫聧陇氓碌聛茅聳赂忙聞碌莽聟聢茅聬聯茫聝漏忙聼聸猫鹿聡忙聮鲁芒聮聰茅聳赂莽聤鲁氓聤聴茅聨鹿茫聢聽芒聰聭茅聧隆芒聲聜氓聝碌氓娄聻茫聜聠忙聦聠茅聢搂氓麓聭茅聨戮猫炉虏氓聠聙忙陇陇忙聮露氓聲驴茅聨路莽禄聵莽鹿聸忙聺聢氓聣搂忙聜录茅聳鹿猫聶聧聡茅隆垄忙掳卢茅聧聸茂录聹茅聳禄氓潞炉莽露聠茅聧聥氓聥聜聵莽聙碌莽鲁聲氓聛聯氓篓聢氓聻芦聧忙驴聻盲陆鲁氓聸露忙聲聥茅聳驴忙驴聠忙拢聸莽禄隆芒聜漏碌茂鹿漏茅聧聵氓聟录莽氓漏聤氓聽聼芒聮聭茅聳赂忙露聵茂鹿娄莽录聜忙掳露忙聼聸忙聬麓茫聢聽芒聳聲忙楼聽莽聜虏茂鹿陇茅聢鹿忙聢聽芒聜卢茂陆聨氓聛聙茅聨路猫聶鹿莽麓聯忙碌拢氓聣虏氓聞聬茅聬聨莽卢聸氓聸篓氓聙露莽聙鹿茫聝楼茅聢路忙聢聽莽麓聮莽聙拢莽職隆茅聳赂忙聴聜氓聳聳茅聫聦芒聞聝忙聥聟莽禄聸氓忙庐茫聜聨氓聛聞莽禄禄忙聢聽忙聧垄氓漏碌氓聴漏忙陆漏忙陇陇忙聨聭氓芦聫聛猫搂聞莽鹿聺茅聬垄茅聺聸盲禄娄茅聳赂茫聞楼莽聢录茅聢楼茂鹿聜氓麓露茅聤聤脩聦忙碌聞忙驴隆茫聜聫莽碌陆茅聧聼忙聜隆?A/N/G/K 莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳禄忙聢拢氓搂陇茅聧聤氓聺聴忙聼篓茅聫聜氓聻聮聰茅聳禄莽聭漏茅聫聦莽聠路芒聳聨茅聶聡茅隆聲脩聝脙潞茅莽聜碌忙職娄茅聳驴莽聠潞氓搂碌茅聧聤莽聝陆忙聼聸茅聨掳芒聲聛氓搂聜茅聳赂忙卢聫氓茅聤聡氓聥炉氓搂聙茅聰聸氓聥卢氓聶掳茅聴聛莽聭掳莽拢颅茅聧聥茫聜聠盲驴聤茅聬聭猫聴聣脦篓茅聳碌氓陇聢氓聞聶茅聤聢氓芒聮聮茅聳赂氓卤戮莽聭搂茅聧聰氓露聣忙聜聴莽禄聴忙露聵氓聨戮茅聧聺氓聥碌氓聬聥氓猫路潞茂鹿娄茅聰聸忙露垄忙垄潞茅聬聼忙聺驴芒聮聰莽录聛氓聻露茅聨驴聽氓录麓茅聤聫茂录聞氓聨陋忙驴聽莽聰碌氓聛聸茅聬聫忙驴聥氓麓聺茅聰聲氓聜聸莽聟聲忙驴聻氓聴聴莽鹿聬茅隆聫忙聼聧猫陇聹茅聧聯忙掳露茅聫聧莽陋聴忙驴隆茫聜聟忙聡聨莽禄聽忙聽颅忙聼聸氓庐聙芒聜聧聥莽聠路莽聲碌忙赂職芒聜卢茅聫聦忙露垄氓娄路茅莽聟聨茅聴聛莽禄聵忙聦禄茅聰聲茫聢陇氓录禄茅聬聰莽垄卤忙聲聧茅聳赂芒聞聝茅聢搂忙聜聜猫聴?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    scene_id : str
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮莽聲聦莽麓聯忙碌拢氓猫聜漏忙聧垄茅聧隆忙露聶莽聟聲茅聳碌氓陇聵氓聳聳忙戮搂莽聴陋茅聨聞氓聛楼茅聳潞氓卤戮忙麓聼氓庐聲莽聟聨芒聰聭茅聧楼脩聞莽聫麓茅聴聜盲陆赂忙聬氓聭虏聰茅聬垄氓聻芦氓麓虏忙驴聻忙聢聶氓聻鹿茅陋聻茫聢隆盲驴聤茅忙驴聬氓職聶莽禄戮忙聺驴莽搂鹿氓忙麓麓篓忙驴聽氓聜職茅聴聛莽茅聨鹿氓聸搂莽录聛忙篓禄氓陋麓茅聬聼忙卢聫氓忙碌聽芒聲聟忙垄潞莽禄聥忙聦聨忙聧拢茅聳赂氓潞聸氓聙聴莽禄庐氓卢戮露氓卢聜卢莽聝陆忙聼聸氓篓聠氓驴聯盲潞掳莽录聜氓露聞莽茅隆聲茂陆聣氓鹿聵茅莽垄卤忙陆聤茅聴聛莽聜陆氓聞虏茅聫聥氓楼赂氓麓聭茅聨戮猫氓聻聜忙陇陇忙聞漏氓聙聴茅忙聸垄忙垄潞茅聬聼茅聜娄氓職聼茅聳赂氓卢聳聳茅陋聻氓漏聡氓鹿聡茅莽聤?氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄茅聳禄忙聞聬芦茅聧篓氓聜聻忙聼聸茅茂陆聠莽莽录聛莽聤鲁氓陆聝茅聤聠忙聨聭茅聰聺氓聭忙驴隆莽聜陋氓聙聭茅聧篓忙麓聡聺忙楼拢氓聠拢芒聜鹿聜氓娄赂茅聢潞盲陆赂莽陋聸氓娄聻茫聜聠忙麓聳茅聨鲁氓驴聲莽麓聻氓娄陇盲陆潞莽炉聙茅聳禄茫聢陇忙陆搂茅聢禄氓聽聼莽鹿聸忙碌拢氓聠虏忙碌聜茅聫聛氓聥聼聸氓篓聭忙卢聬莽露聭莽录聛芒聜聧聬莽聟聝莽聭聹忙禄聢氓麓聹茅聬聰氓忙陆聳茅聳戮氓驴聯忙鹿卤莽潞戮氓聟录盲驴聤茅忙驴聬氓職聶莽禄戮忙聺驴莽搂麓茅聢鹿忙聢娄忙聜聶莽聙聸忙篓潞氓聰聣氓茫聜聝氓聞虏茅聨赂忙聬聛氓聼楼忙戮露忙聞聛篓莽录聛氓忙鲁聲茅聧聰茂鹿聙茅聤聢氓聴聴莽卢聜茅聳录氓聠虏氓莽禄?``S(A1,A2,N3)`` 莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳禄忙聢拢氓搂陇茅聧聤氓聺聴忙聼篓茅聫聜氓聻聮聰茅聳禄莽聭漏茅聫聦莽聠路芒聳聨茅聶聡茅隆聲脩聝脙潞茅莽聜碌忙職娄忙驴聻氓路庐盲潞聹茅聬聮?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    dict[str, list[str]]
        茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼莽聞录莽聙鹿茫聝楼忙聡聬茅聰聸忙露垄忙垄潞莽录聛忙篓潞氓搂聡茅聳鹿莽垄卤氓聛聠莽禄?茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼忙聲聥莽聭聹氓驴聸氓聼聙茅隆聮盲陆潞芒聳聯茅聳潞氓聭鹿聞茅聧聮忙聨聭芒聳聨茅聨戮氓麓聡氓漏碌茂录聞氓聛聸茅聬篓莽聝陆氓麓聭茅聨戮猫炉虏莽忙陇陇忙聴聜氓聨搂茅聭掳氓聻聰聭茅聬聬忙聺聭莽聛娄茅聳禄莽聠麓氓聞碌氓庐聲忙禄聢氓娄赂茅聤聣茫聢隆忙聳聙茅聴聛莽禄聵莽禄聵聣猫陇聨茅聤聡氓聥鹿聥氓漏碌氓聸搂忙芦搂莽录聜盲戮聡氓聳聳茅聬聴忙聮鲁莽聲路茅聧聴莽聜漏忙忙驴聻氓聴聵氓聻碌茅陋聻?茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼莽聞录莽聙鹿茫聝楼忙聡聬茅聰聸忙露聵芒聰聭茅聬聬忙聺聭莽聛聼茅聳赂茫聞娄忙陇陆氓庐聲忙聢娄氓娄露忙戮露氓卢聨陋忙驴聽莽聰碌氓搂麓莽禄禄忙篓潞茅莽聲聦莽茅聳赂忙露卤氓聻掳氓聲赂茅聴聛茅聙聻氓卤陆氓垄录聛莽聤虏莽搂鹿氓庐聲忙聸垄忙聼聠氓庐聯茫聝楼忙麓颅茅隆垄忙露聵芒聰聝茅聨炉脩聜氓拢?token 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛氓露聣氓麓聼茅盲陆鹿忙驴聽氓戮路忙聺驴芒聜聸垄氓鹿聤氓庐聙氓聥聞碌忙陇陇忙聨聠氓聛聯茅聳赂莽聡聛氓聻掳茅隆聲茫聝漏氓麓聼茅忙掳卤莽聳聙氓娄聻茫聜聠氓聻聼茅聫聜氓聜聻氓麓聲茅聬垄莽猫聯聼忙驴聻忙聢聶氓聻鹿莽禄聽忙露聶茫聛聞茅聨潞氓聴聴盲潞戮忙驴聽茫聢垄忙聡聯莽聙職忙聺驴莽录聛忙篓禄氓陋麓茅聳赂忙露聵茂鹿陇茅聫聠氓聽聺芒聰聭茅聬聬茅垄聺氓職聥莽录聛莽聤鲁忙聦赂茅聬拢氓鲁掳茅聧聬莽聜聝碌茅聴聛莽禄聵氓聤聵猫茅聳赂氓忙聧聡氓盲禄聥茅聬聰氓聫聣氓茅聴聛猫聧陇氓搂麓氓篓虏芒聲聝盲潞赂氓篓聠忙聢聽盲赂聧茅聳鹿忙庐驴氓聳職莽潞戮猫聜漏莽麓聯忙碌拢氓搂聭忙聟篓忙戮露忙聞颅莽聟聝莽聭聹忙禄聢氓麓聹氓搂聵猫路潞盲录聽茅聨录忙路聤聦莽禄聙氓陇聬忙聼聼忙聺聢氓聣搂莽聲卤茅聳潞氓聥漏氓聞碌茅聫聦氓露聢茅聳赂忙聮麓氓虏赂忙驴隆莽聰赂氓麓聼茅隆聳忙掳卤茅聴聛莽鲁聲氓聣聺茅聤聡莽聜麓莽芦聼茅聫聡茫聢隆芒聮聮氓篓麓茂陆聞猫茅聬聮茅鹿聝忙聼篓茅聫聡茫聞陇茅聜炉茅聮聵莽聠录忙聼聼茅聬聬莽聛卤茅聳潞氓卢聟拢茅聫聦忙聸隆莽庐聸莽聙拢氓聛聯茅聫聡莽聜麓忙職聹茅聴聜氓聜聶莽聞娄莽聙碌脩聝脦聸忙碌拢莽鲁聲忙聻聳茅聳潞氓聸漏氓搂路忙露聯氓露聠忙聟篓氓娄聻庐莽聟聬茅聳鲁茅聰聥氓聻鹿莽禄卤忙聨聰芒聜陆聨氓聛聞茅忙禄聞莽鹿聺茅聢搂氓鹿聧茅隆聰氓聣搂茂录聹茅聳禄莽聤虏忙麓娄猫陇掳氓聠庐茅隆聬茫聜聡莽拢录茅聫聣氓聽聸氓聤聨茅聨鲁氓聥颅莽麓聮莽录聛忙篓录莽庐聯忙陇陇莽鹿聢茅隆垄忙漏聙忙聜漏茅聨禄忙聨聰莽聳聞?``S(A1,A2,N3)``
        茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?``{"A": ["A1", "A2"], "N": ["N3"]}``茅聴?

    忙驴聽莽聰碌氓搂路茅聫聛氓聭聤忙聟篓茅聬聭猫聴聣忙聻聛茅聳鹿茅聴麓莽莽潞戮莽禄聵莽鹿聸茅聨麓忙卢聫莽聛陋茅聳禄忙聮麓忙麓聼茅聫聦茂驴聽氓麓聮氓漏碌氓聸漏茫聙聤茅聳潞盲陆赂莽聣聯茅聨?
    -----
    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜麓莽聣聽茅隆垄忙聸垄忙聲聝茅聢搂忙聡聹猫陇掳忙聨聯莽聟聸茅聬聫莽聜露茅聧聰忙掳卤莽麓聮茅聢搂氓麓聵茅聢潞氓聠陋莽聵聢茅聴聛茅聺聸茅陋聫莽禄卤忙聸聼忙聶露茅聧篓氓聫聫氓漏碌莽聤虏忙聡聫茅隆聫莽聤碌莽麓聮忙聺聢茫聞楼莽卢聮茅聯禄茂陆聣忙聼陇氓篓麓氓卢芦聡茅聳赂忙篓禄氓聻碌茅聢鹿忙聢娄忙聜漏茅隆聰猫聜漏盲录聞茅聴聛莽隆聟忙芦聲茅聨赂忙颅聦莽聛聥氓漏碌莽聤禄忙陆搂氓娄麓碌?``axes[氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆茫聙聜茅莽聜聳潞氓聸漏氓搂路茅聬卢聫芒聰聭茅聬聵氓麓聡茅聳禄忙聮露氓聳聬忙路聡氓漏聡氓娄露茅聧聲氓娄陆莽录聜盲陆虏氓聛聯茅聬聨茂陆聨氓聳聴茅聬聯忙卢聬茫聛聞茅聫聡忙盲潞戮莽录聜氓聜職氓聙鹿氓庐聙忙颅聦忙茅聴聛猫聜漏芒聮聢茅聧聯忙掳露氓麓聭茅聫聧茫聞搂莽庐戮茅聳赂芒聞聝脙聬聫氓露聣忙聼聸?= token`` 氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄茅聳禄忙聞篓聤茅聬聮猫聤楼莽鹿聸茅聧隆忙篓禄氓掳颅茅聬聼忙卢聶莽庐聯茅聨庐忙楼聟忙聲聬忙聬麓芒聞聝莽聛聧茅聴聛?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯氓漏碌芒聙鲁莽聢录茅陋聻忙聽篓忙聥聟茅聧聺茫聝楼忙聭聥茅聴聜盲陆赂忙聠隆茅隆篓氓聽楼氓麓聲茅聨掳莽聣聢氓聙驴茅聳录忙聛聮氓聞鲁莽禄聽茅鹿聝忙聼聸茅聢漏氓聟漏忙聢聽盲潞赂茅莽聤碌莽拢录茅聳禄忙篓潞莽拢戮搂莽聰赂忙聼隆氓庐聙芒聜聧聽忙聽聧庐茅聨麓茂鹿聙莽庐聸忙陇陇忙聮露氓聛聻茅聳驴莽麓聯茅聧聦忙卢聬莽楼掳氓氓炉赂猫聻聻茅聳赂忙聞碌氓聳聳茅聮聝莽聠录忙聼聧茅聤聣茫聝搂茂录聧茅聳潞氓聴聬氓聙聳茅聤聡氓聥录卢茅聧篓氓聙聴聛茅聺漏莽碌戮氓娄聻盲禄聥氓录禄茅聬聰氓聟录氓職聭氓娄聣聢莽鹿聜氓篓聢忙垄潞莽聙鹿莽聥聙忙陆聬茅聳赂茫聞搂忙聡聯茅隆聲茫聝娄莽鹿聺茅聧楼茫聞楼盲禄录茅聬聨氓聟聣氓聛聯茅聳鲁脩聦氓聣聼茅聨炉氓虏聞氓鹿聵莽录聛忙篓潞盲禄驴隆茫聜聫莽庐聮茅聰聼茫聝漏氓聼聙茅隆聭猫聴聣盲潞戮茅聴聜盲陆赂忙聬芦茅隆聭氓聴聴脦聸茅聧聬猫聯聼茅聳禄忙聴聡茫聙聙茅聧聥忙聮禄忙聜陆氓篓聢猫路篓氓聤赂忙驴聻氓氓搂碌莽禄庐氓潞拢忙茅聨潞忙莽陇聢氓篓麓茂陆聣忙聶虏氓篓聦忙垄潞茅聳露職芒聰聮茅聳赂忙聳驴莽聼聜茅聰聺莽聜虏茅聧驴聠猫聙驴氓漏碌茂录聞氓聛聻茅聧聨忙聤陆氓录赂茅聧芦茫聝娄芒聮聮?token
    茅聴??茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聳莽潞戮氓聺聴忙聼拢茅聤聫茫聢漏莽聞戮茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛莽篓驴茅氓聴聵莽漏卤忙驴聽氓聸搂莽聼聴氓猫路潞茂鹿陇茅隆虏聞忙垄潞莽聙鹿莽聥聙氓職聶茅聴聛氓赂露聛茅聬聸氓麓露茅隆聮氓忙聶聹茅聴聛莽鲁聲氓聣聺茅聬聼脩聡氓录鲁?``A1`` 茅聴?``A2`` 茅聴?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙聮虏莽茅聧篓忙驴聡茫聛聞茅聫聧茫聞娄莽職聭茅聳潞氓聴聴忙拢聺茅聫聦莽聠录忙垄禄莽聙碌氓聣虏忙搂录聜?``A2``茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓聙驴茅聳驴忙聴聢忙聲庐茅聧聥忙聮露莽虏篓聭忙篓禄氓聶陆茅聳禄莽聭漏茅聫聦莽聠路芒聳聨茅聢楼氓麓聡忙鹿麓茅聳赂忙聴聙氓聻驴氓庐聲忙麓聼聙茅隆聮盲陆碌盲潞聹茅聳鹿莽聝聵氓聻碌茅聢搂氓麓碌忙戮鹿忙聸聼莽赂陆莽禄聥猫路篓茅陋聡茅聴聛氓聣虏氓聛聟莽禄聥忙聢聻盲驴聤莽聮潞茫聞搂莽拢录茅聳禄忙漏聙忙陆搂忙碌聽茅聛聧忙聼隆氓庐聙芒聜聧聽忙聽聙聛氓篓聠忙聮麓氓炉聽氓漏垄猫路潞忙拢聶茅聬聫氓聴聸忙聼拢忙聬麓茫聢漏氓聲聫茅聳赂莽聤鲁莽聣聽茅聨聝漏忙聜隆忙聬麓茂陆聠忙茅聴聛猫聧陇氓聳聬忙戮鹿氓卢录赂茅聫聝氓聽聼莽聟聫氓猫路潞忙氓篓聟氓聴聸忙聼聧茅聳驴氓卢聛麓茅聳鹿忙鲁聸茅聢陆氓氓鹿聮茅聧隆忙陇戮氓茅聴聛莽莽鹿聬茅聨鲁氓潞篓脣聣氓猫聢碌芒聮聮茅聳赂氓卤戮猫聣戮茅聢搂氓聟聵茅聨鲁忙楼聟氓麓录茅聫聡忙陇禄氓聛聯茅聫聧氓職聝茅聳鲁氓聯聞氓聲掳茅聰聸忙禄聢忙垄潞茅聳芦莽聜虏猫聥炉忙戮搂盲录麓忙聼聧猫茅聧聯茫聞搂莽麓聻茅聧隆忙露聵莽陋聴忙驴隆茫聜聟忙聡聫茅聧聥氓聜聸莽鹿聸茅聧隆忙篓潞莽聛漏莽禄戮忙聝搂茫聜赂茅聨忙聲聬忙聬麓芒聙虏猫聢陋茅聴聛莽莽掳潞茅聧聨氓聫聣莽漏卤忙驴聽氓聸搂莽聼聴氓猫路潞茅聢搂氓聤聶茅聫聦茂录聞盲禄娄茅聧聯脩聙氓搂聡莽录聜盲陆赂氓聙赂莽禄聣氓陆聮氓鹿聝氓篓聠忙聮鲁盲录聽茅聳赂忙聸篓氓聨戮茅聫聠芒聰聭茅聬聬氓颅聵莽露聭茅聳赂忙掳隆颅氓聸搂芒聳聨茅聨掳氓茅聴聜氓聜職氓聙路茅聬聮茂赂戮芒聜卢莽卢聸氓聟戮忙露聶忙聥聟莽聭聶氓聥芦氓录驴茅聴聛茅聺聸莽聣聠茅聨虏忙露垄氓鹿聮氓娄陇芒聜陋聻茫聢隆莽鹿聸茅聨麓莽聜碌忙聠驴茅聳鹿氓鲁掳氓猫聶鹿莽虏聶茅聨赂氓潞垄忙聥卤莽录聛莽聜戮聵莽禄聣莽忙拢垄氓漏碌莽聤虏茂鹿陇莽聙職忙聧垄茅聧隆忙篓赂盲潞聹茅聳鹿忙聣鲁忙聶聸茅聬聫忙聼楼忙聼聸氓篓聠氓驴聯茅聳潞氓卤戮莽聸炉茅聧漏莽聜潞氓鹿聠猫忙戮?
    """
    if not isinstance(scene_id, str) or not scene_id.startswith("S(") or not scene_id.endswith(")"):
        return {}
    axes: dict[str, list[str]] = {}
    for token in scene_id[2:-1].split(","):
        normalized_token = str(token).strip()
        if len(normalized_token) >= 2 and normalized_token[0].isalpha():
            axes.setdefault(normalized_token[0], []).append(normalized_token)
    return axes


def _collect_search_metrics(
    *,
    experiment_output_root: Path,
    method_name: str,
    baseline_method_name: str = "ekf",
    baseline_metrics: dict[str, float] | None = None,
) -> dict[str, float]:
    """氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄忙驴聻忙聢聶氓聼聞茅聫聛氓茫聛聞茅聨戮猫路潞忙聻聭茅聳赂忙卢聫莽鹿聵茅聨庐忙楼聟忙聜陆茅聬垄莽聜鹿芒聜漏聡忙聼聸莽聙拢氓掳颅茅聳鲁猫聴聣茅陋聻忙聨聴氓鹿聵莽聙碌莽莽露聝茅聴聜氓聜聶忙碌聡茅隆聲忙聽颅氓麓鲁茅忙禄聞氓麓虏忙驴聽茅聺聸忙拢聸茅聫聠茂鹿聜忙聼聼茅隆聳莽聜虏盲潞掳忙驴隆猫聦聟聵氓搂聤忙麓陋氓麓聧聲茅聺聸氓聣掳茅聴聜盲陆赂氓潞聹莽录聧盲陆赂莽聮聡氓路聬聥氓漏垄猫路潞茂鹿娄茅聧聵氓聭聤莽鹿聸忙聺聢氓聣搂氓聢掳茅聳鹿忙碌聥芒聜卢忙聳楼忙聻聰茅聳潞氓卤禄氓聞虏茅聢路忙聢聽莽麓聮莽聙拢氓聞楼茅聳赂氓潞垄氓聤聴莽禄卤忙聨聰忙聥聽茅聨禄忙聨聺盲录聙氓漏碌芒聙鲁氓录露茅聧聰忙卢聯莽聧聨潞忙楼聙莽庐禄茅聳赂忙聮虏莽聞娄茅聧陇氓虏聞忙垄禄忙碌拢莽篓驴茅聳禄忙聮鲁莽聣聯氓庐聲忙聺驴茅隆聯莽聝聠氓職陇茅聴聛忙聬聻氓聞庐茅聫聟忙驴聢氓拢聲茅聮聵氓陇聢芒聜卢忙聨聭茅聰聺氓聭忙驴隆莽聜陋氓聙聭茅聧聤忙聸垄氓麓聬茅聧篓忙職娄茅聳驴忙驴聠莽禄聙氓聠聼聲忙驴聻脩聝氓茅隆聰忙聸聼莽鹿聺茅聬垄茅聺聸盲禄聹忙驴隆氓篓聠茫聢聽莽聲隆茅聨鲁氓漏聝忙陇陆茅隆垄忙聳驴茅聧隆忙篓禄莽聫聲茅聴聜氓聜聶莽陇聛茅聨虏芒聙鲁脙潞茅聰聲氓聜職莽茅聳禄忙聳驴氓篓录忙聽篓莽鹿聸氓莽掳禄茅聳鹿莽隆聟氓聼聳茅聤聡氓聥炉氓鹿聵莽聮潞莽聸聮茫聢垄莽麓聮氓娄陇氓卢炉聙忙驴漏聝忙陇陆氓庐聲茫聞娄莽鹿聺茅聧聬茂陆聠氓芦禄忙驴聽莽聰碌氓聛聸氓炉聣聛忙聧垄茅聧芦莽聠露莽鹿聠茅聳禄茫聢垄莽禄聽忙露垄忙聼隆忙戮露氓潞垄莽隆露茅聳赂氓漏聞氓聛陇氓忙麓氓篓麓茫聜聠莽聸聧漏芒聜陇陇忙聨聭氓茅聧聭莽聤禄忙聼隆氓氓潞隆氓鹿聴茅聳?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓麓茂陆聣忙陆禄茅聭掳氓聺聴氓录露茅聧芦忙露聵氓莽录聛莽聤禄莽庐聳茅聨庐氓陆聮氓麓露茅聧楼脩聞氓聶陆茅聳潞氓聴聬氓聙聺氓猫聶鹿忙聥聽茅聢楼猫聶芦莽聛聛茅聴聛莽篓驴忙碌路茅聫聛猫茅隆聬茂鹿聙莽庐聸忙陇陇忙聮露氓聛聼莽禄聬莽聜潞忙聼拢忙聬麓莽搂碌茅聳赂氓卢聡聵茅聨聰赂茅聰聰猫搂拢忙聥潞茅聳禄莽聤虏忙麓漏莽聛聫忙卢聭莽虏禄茅聨露猫聤楼莽聟聲茅聬聨茂陆聨氓聛聠氓篓虏忙聮聼聸茅聢鹿忙聝搂盲潞戮忙驴隆莽聜聙聳莽聰炉忙聨聴氓麓聬茅聨录盲陆赂氓聡聽茅聨录忙路聤聧茅聬聯莽聠赂莽鹿聺茅聧篓氓搂聡茅聳禄氓驴楼茅聶聞茅聨卤茫聞娄莽鹿聺茅聧聦茫聜聡氓聬聥茅聬聨猫忙麓聵莽聰炉忙聨聴茂陆聻氓漏碌氓忙聲聠氓猫路潞茅聧聥氓聭麓忙垄禄忙碌拢莽颅聥氓職聶茅聬聨忙露聮忙聶聺茅聧聽氓漏聜忙戮露氓漏聞茂录?baseline 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳禄莽聠赂氓茅聫聙盲禄聥忙聼拢氓娄聶鹿忙聻聝氓猫搂聞茂鹿垄茅聫聦茂陆聣茅聬聲盲陆鹿莽聳聛氓娄陇莽聤碌氓聛聻茅聬聴忙聸隆氓陆聛氓茫聜聠氓路禄茅聧聥忙聮鲁芒聰聭茅聢楼氓聠虏忙聜聭忙驴?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    experiment_output_root : Path
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜麓莽聣聽茅隆垄忙聸垄忙聲聝茅聢搂芒聜卢忙掳聤聡氓聥鹿聮茅聨麓忙驴聬盲禄录莽录聜盲陆虏氓聛聯?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聳莽禄聽莽聤禄忙聼聼茅聨鹿忙聞碌氓職聶茅聬聼忙卢聶莽庐聯茅聨忙聲聬忙聬麓芒聙鲁茅聴聛忙聬聻氓聤聧莽禄禄氓聽楼莽聧聫聧氓潞篓茅聢搂盲驴聤茅聭虏芒聲聝忙拢炉茅聳鹿盲陆赂盲禄聴聛氓聯聞忙聹卢氓篓虏忙忙聧垄氓篓聭茫聢隆氓聙路忙陇陇忙聨聧猫聝露猫陇掳氓聴聸忙垄禄忙碌拢盲戮聞戮氓娄陇莽聤虏忙聭聹莽禄庐忙聴聡茫聛聬茅聨潞忙聢拢忙聭聲氓漏碌莽聜麓氓聻炉茅聧篓氓陆聮氓麓隆茅聨露猫聤楼莽聟聫氓漏碌莽聜虏莽聛聺茅聧聰忙禄聢忙聼拢猫鹿聡忙聴聙氓掳碌莽录聛猫戮篓忙聧聡氓庐聲忙聨聭茅聧聴莽聝聠忙職驴隆莽聜聙陇茅聧聵茅鹿聝氓麓聴氓娄戮炉氓職聲茅聬聽氓聸篓莽楼麓氓娄虏氓聽楼忙聼聲猫鹿聡忙聸聻脨楼忙陇陇忙聮麓忙陇陆氓搂聤忙麓陋氓鹿聬忙聬麓茂陆聡莽聲碌氓漏碌芒聵聠氓聛聟茅聬聼脩聡氓鹿聭茅聤聫莽聤茅聳碌氓驴聲氓搂路茅聬聯忙聢聻莽鹿聺茅聤聫茂陆聠莽隆戮忙陇陇氓聣聼茅聧漏芒聜陇陇忙聨聧猫聝露茅聬聨聴忙聼隆莽聛聻莽聜莽聙鹿忙聸聻莽聢录忙驴隆忙聬聛氓娄路茅聤聣莽露聮茅聴聛猫氓颅漏茅忙聽麓掳氓娄陇芒聜卢茅隆聸搂芒聳聨茅聨潞忙聳戮猫聢陋茅聴聛猫忙碌路茅聨鲁忙聮露芒聜卢忙掳聧鲁茅聤聠氓聜聻忙聲聜茅聬拢氓虏聙氓陆?
    method_name : str
        茅聴聜氓聜職氓聙路氓篓麓氓聸搂莽聲路茅聬垄莽莽陋聴茅聳鹿茅聜娄氓聳聳茅聧篓忙驴聥氓鹿聳氓篓拢莽聛驴聻氓聭搂碌忙路聡氓漏聡氓娄露茅聧聸忙芦拢莽录聜盲陆潞氓搂鲁茅聧聴忙聤陆氓录禄氓篓聭忙篓路忙忙驴莽聣聤茅聫聛忙聝搂芒聰聭茅聬聬氓聫聣氓聶陆氓漏碌莽聜漏氓聛聯茅聴聛氓聯聞莽聼聣莽禄虏氓聙聼莽聛聮茅聴聛忙聝搂莽鹿聵茅聢搂莽篓聯忙陇鹿忙聤陆忙垄禄忙碌拢氓聭聤忙聠赂茅聳赂莽聤虏茅聟拢茅陋聻忙聽颅茅聰聰莽聲聦莽聲鲁茅聴聜氓聜聶莽聞娄莽聙碌脩聟忙聧垄茅聧聦莽聜麓氓聛聥氓漏碌莽聤虏忙碌聜茅聫聛氓聜聻忙聼拢氓娄炉猫聜漏氓赂聸茅聳禄忙聮麓忙麓聼茅聫聦氓露聣氓麓芦茅聧聲忙茅聴聛氓庐聽茅聬聴忙聮禄氓录禄茅聬聰莽垄卤忙聲聤忙驴聻氓聴聶氓聛聯莽录聧氓聽垄莽麓聯忙碌拢猫忙庐聵茅聳赂莽聤鲁氓聤聴茅聨鹿茫聢聽芒聰聭莽聙拢氓茫聜聠忙聦露茅聧聰氓潞篓氓麓楼茅聬聼忙聣鲁忙聰陆茅聳禄忙篓潞莽聛職茅聫聠芒聲聟忙聼聸莽聙拢忙麓麓茅聳鹿氓聥颅莽炉聙茅聳赂茫聝聭氓聠聜卢忙垄潞茅聧聯猫聦聨聸茅聳赂茂陆聨氓聶拢氓庐聲莽聰赂氓录麓茅聤聫茂录聞氓聨碌茅聴聛氓聭聤莽聭楼茅隆娄忙聣庐盲潞赂茅聰聰茅聰聛聯莽聭聶氓聥卢莽陇聢茅聳赂莽聤鲁莽聣聽氓庐聲忙麓陋氓聼聙茅隆聮盲陆碌盲潞聹茅聳鹿氓聯聞忙忙碌聹茅鹿聝忙垄潞茅聨赂氓潞垄氓氓篓麓忙禄聨盲潞聹茅茂陆聡忙聥聽氓篓麓氓氓漏碌氓娄芦忙卢聯氓麓聯茅聬垄氓聭聤氓聛聽忙驴庐忙漏聙忙拢驴茅聳颅忙聟?
    baseline_method_name : str, optional
        Baseline 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜麓莽聣聽茅隆垄忙聸垄忙聲聝茅聢搂氓茅聬聼忙聡聤氓陆聮忙聜漏氓庐赂氓聤聙莽录聜氓聜職莽搂碌茅聬聴忙聸垄芒聙娄莽聮潞茫聞娄氓麓聣茅聳禄忙聢聻脡聭茅聨路茅聛聧忙垄潞莽禄聥猫路篓莽莽录聛忙聧聡氓炉芒聵聣茅聤聫芒聲聜忙聶聺茅聴聛忙聦聨莽鹿聜氓娄芦忙露垄脣聡茅聤聣芒聲聜莽聟聼茅聳禄忙聳驴忙聭聹茅聨鲁忙聸聻芒聰聭茅聢楼猫聶芦氓聳聵茅聳赂芒聰戮氓聛聯氓娄聻茫聜聠氓路禄茅聧聥忙聮露莽麓聮茅聬聵猫聦卢脨搂莽聙鹿忙聸聼忙麓聵氓氓驴聸氓聼隆茅聧聦忙掳卢莽掳楼茅聴聜盲陆赂忙鹿卤茅聧聨茫聜聟忙驴聽忙掳颅氓聞聴茅聳赂芒聞聝茅聬聰氓聴聸忙聜鹿茅聯聫莽聭掳莽職聭莽聙鹿忙聬聻氓聞碌茅聫聦氓露聣忙聦聤忙戮露忙篓禄脨娄茅聴聛氓庐聽忙楼聽忙颅聦莽聛聮茅聴聛莽禄聵忙聦赂忙楼聽氓聭聤莽聯? ``"ekf"``茅聴?
    baseline_metrics : dict[str, float] | None, optional
        茅聴?茅聴聜氓聜職氓聙赂茅聧聤茅拢聨忙氓搂聵猫路潞莽聞碌忙陇陇忙聨聭氓聙赂忙碌聽忙禄聢忙聼陇氓篓虏猫莽聛隆茅聳潞氓聭聢录茅隆垄忙卢聬忙聟篓茅聨掳莽聸聨茅聴聜盲陆鹿氓氓篓虏氓露聣氓麓聭茅聨戮氓聻聸氓聛聯莽聭聶氓聥卢氓陇聞茅聳赂忙聮露氓聳聳氓庐?baseline 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓麓茂陆聣忙陆禄茅聭掳氓聺聴氓录露茅聧芦忙露聵氓莽录聛莽聤禄莽庐聳茅聨庐氓陆聮氓麓露茅聧楼脩聞氓聶陆茅聳潞氓聴聬氓聙聺氓猫聶鹿忙聥聽茅聢楼猫聶芦莽聛聛茅聴聛莽篓驴忙碌路茅聫聛猫茅隆聬茂鹿聙莽庐聸茅聳潞氓陇聨忙鲁聲茅聧聤莽聜潞忙垄潞茅聧聬氓聫聣忙聝聢茅聳赂莽聤鲁莽篓聯氓娄虏忙聞掳氓鹿聨茅聭潞茫聝垄忙聥禄忙驴聻猫戮戮莽碌驴茅聬隆忙聳驴茅聨掳莽聣聢莽聟聲茅聨录猫戮戮莽麓聲茅聰聸氓露聟莽聭聶氓聥卢莽陇聥忙陇陇茫聢隆茂鹿垄忙驴聜卢茅聳禄忙篓禄茅聧聼氓聽楼忙垄禄忙碌拢猫聶鹿氓赂聸忙陇陇忙麓篓茅聧聮忙聨聴茅聬聬忙聺聭盲潞聴茅聴聛茅聺聸忙驴隆氓聸漏莽虏禄忙楼拢氓聠漏莽聟聶茅聬聨莽聰碌茅聧聯茅聛聧忙聼拢茅聨潞氓露聡莽聼聶茅聳潞氓虏聥莽露聛茅陋聻忙聨聴氓录聧篓莽庐拢茅聴聜盲陆陆茅聧聽忙楼聟茫聙聝茅聧聸猫戮戮莽露聳忙驴聽茅聺聸氓聰聴聛茅聺聸茂录聫茅聧聦忙禄聟忙驴聽氓戮路忙聺驴芒聜卢忙楼聛忙聟篓茅聬聭猫聴聣莽拢禄茅聳禄忙聞聬芦莽禄聽忙聼楼忙聼聸茅聤聣氓垄掳莽聙鹿忙聮虏莽路職茅聫聦忙露垄氓娄路茅隆聰莽聟聨芒聮聮茅聴聛猫陆掳莽茅聧聤氓聺聴氓录禄氓篓聭茫聢聽莽聞潞茅聳赂忙聞篓忙聶聺氓篓聢茫聜聡莽麓聯忙碌拢氓聯聞脨拢茅聳禄忙聮炉芒聜聞聛忙陆聳莽录聜盲陆鹿莽禄隆芒聜聴聛氓聭聤氓聤聫茅聫聟忙禄聢氓麓聬茅聬聭猫聤楼芒聮聮氓篓麓茂陆聞猫颅娄茅聫聙莽聣聢莽鹿聸茅聧聸莽茅聳潞盲陆赂氓聲麓忙驴庐芒聜聳鲁茫聝篓氓掳聶莽禄聽忙掳卤莽麓聮茅聬聬氓聤陇忙陇陇忙聢聺忙麓颅茅聨聺聴氓聼聙茅隆聮芒聜卢茅聢鹿忙聢娄忙聜聶茅聫聣忙聢聽盲禄赂氓娄聻茫聜聨氓聨录氓篓虏忙拢聼茅聴聜盲戮聡氓聣聶莽禄聣莽聰赂氓聼聸茅聨麓茂赂陆莽聟聲忙驴聻忙聢聻茂鹿忙楼聝莽麓聮茅聧聸職忙聠戮茅聳潞氓卤戮莽篓聭猫聻拢茅聳赂茂赂禄氓聨戮茅聬聯忙聮禄忙聜?``p95``茅聴?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    dict[str, float]
        忙驴聽莽聰碌氓搂路茅聫聛氓聭聤忙聟篓茅聨戮氓聙露忙驴庐忙篓驴忙鲁聸茅聧職氓陆聮忙聼聧猫茅聧聯忙卢聯氓聼聻茅聨麓茂赂陆氓聙路氓娄芦莽聣聢莽鹿聜氓篓聢忙驴聢莽麓聯忙碌拢氓聯聞脨拢茅聳鹿莽聭掳忙麓陇茅聬拢氓鲁掳脙潞茅隆聫氓聥颅氓聤娄氓茫聜聠氓赂聤茅聴聞氓露聠莽娄聧氓氓聽聼莽聟聶茅聳禄忙聢聻脙陋茅聬聫茫聝漏忙聼聧猫茅聧聯忙驴聠莽聲路茅聬垄莽篓驴忙聻聛忙陇陇忙聳驴莽職路氓娄虏氓聽楼忙聼聲猫鹿聡氓篓戮氓聛聯茅聫聜茫聝楼盲潞戮茅聳禄茫聢聽忙聠隆茅聬聯忙聸聼茫聛聞茅隆聯莽聠路莽聛楼氓篓麓忙禄聞莽虏聯茅聫聦忙露聵芒聰聭忙驴聽氓聠戮盲禄漏茅聴聛茅聙聻氓卤陆氓垄陇陇忙聴聙莽聣聽氓庐聲盲录麓氓录麓茅聬聵忙聫聮莽聛聤氓漏碌莽聜虏忙莽禄禄忙聨聴氓录鲁茅聰聲氓聜聸莽聟聶忙陇陇忙聽芦氓聛聸忙碌聹莽聣聢盲驴聤茅聫聦茫聝搂莽聲潞忙驴庐氓漏聝忙陇陆氓庐聲莽聝陆茅聬聰氓聯楼氓陆搂茅聴聜盲陆赂忙聭聹忙驴聰赂忙聜搂茅聬聵氓路職聲茅聬聽氓聸篓莽楼麓氓娄虏氓聽聼莽鹿聺忙驴聽氓聜聸氓聶陆茅聳潞氓陇聦茅聢鹿忙聢娄忙聜聶茅聫聣忙聢聽盲禄赂茅聴聛猫聧陇氓聳聠茅聧聤忙聸垄茂陆聻氓漏碌氓卢聶聹茅聳赂忙聮聴氓聮聝莽麓鲁氓漏碌莽聜露忙聫禄虏猫聜漏氓聢聮氓篓聠忙聮鲁盲录聽茅聳戮氓驴聯忙鹿卤莽潞戮氓楼赂忙聜鹿茅聧楼莽職潞氓脩聞忙麓聼茅聫聦茂陆聣忙聲聬茅聧楼脩聝氓鹿聥茅聴聛猫氓聳聮茅聫聟忙驴聥氓鹿聫茅聬聵猫莽庐碌茅聳鹿莽聝聵忙垄禄氓庐聲氓聯聞芒聰聭茅聰聸氓聥潞录茅聳赂氓漏聝莽聣聽茅陋聻忙聞聜拢茅聧聬茫聞搂莽聧楼茅聳鹿氓聟录莽聲陋茅聧聰氓虏聙莽虏?``rmse``茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮莽聲聦忙聠赂氓庐楼氓聽垄盲潞卤茅聴聜盲陆赂忙聬芦茅聧聤氓陆聮氓篓聠氓驴拢莽聞陆茅聳潞氓露聝氓聤聧茅聧聶氓驴聯芒聰聭茅聬聵忙聫聮芒聜卢氓漏職忙聜掳?5``茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮莽聲聦忙聠赂氓庐楼氓聽垄盲潞卤茅聴聜盲陆赂忙聬芦茅聧聤氓陆聮氓篓聠氓驴拢莽聞陆茅聳潞氓露聝氓聤聧茅聧聶氓驴聯芒聰聭茅聬聵忙聫聮芒聜卢氓漏職氓聼聻氓炉陇莽聶路lure_rate``茅聴?
        ``mae``茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮莽聲聦忙聠赂氓庐楼氓聽垄盲潞卤茅聴聜盲陆赂忙聬芦茅聧聤氓陆聮氓篓聠氓驴拢莽聞陆茅聳潞氓露聝氓聤聧茅聧聶氓驴聯芒聰聭茅聬聵忙聫聮芒聜卢氓漏職氓聼聻氓庐赂氓禄赂_p95``茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮莽聲聦忙聠赂氓庐楼氓聽垄盲潞卤茅聴聜盲陆赂忙聬芦茅聧聤氓陆聮氓篓聠氓驴拢莽聞陆茅聳潞氓露聝氓聤聧茅聧聶氓驴聯芒聰聭茅聬聵忙聫聮芒聜卢氓漏職氓聼聻氓搂聤忙聛聼d_p95``茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮莽聲聦忙聠赂氓庐楼氓聽垄盲潞卤茅聴聜盲陆赂忙聬芦茅聧聤氓陆聮氓篓聠氓驴拢莽聞陆茅聳潞氓露聝氓聤聧茅聧聶氓驴聯芒聰聭茅聬聵忙聫聮芒聜卢氓漏職氓聼聻氓搂聤忙聛聼d_failure_rate``茅聴?

    茅聴聜氓聜職氓聙路氓篓麓氓聸茅聢搂莽篓聢茅聫聟莽聤禄茫聙聛氓篓聠忙聮鲁氓聠聙忙陇陇忙聮露氓聛聼茅聬聸茫聝漏忙垄潞茅聧聺氓聥颅芒聳聣茅聳赂忙篓驴氓聛聬忙戮鹿忙聸隆茫聛聬茅聨潞忙聮露氓聨碌茅聴聜氓聜職氓聙赂茅氓麓聺氓氓陇聥莽聟聲?
    ------
    ValueError
        茅聴?``baseline_metrics`` 氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄茅聳禄忙聞聬芦茅聧篓氓聜聻忙聼聸茅茂陆聠莽莽录聛莽聤卤氓聧鲁茅聫聦忙露聵芒聰聭茅聧聲氓搂垄茅聴聛忙麓禄氓聨陆茅聨鹿氓聸录禄茅聬聰猫聶鹿莽拢录茅聳碌氓驴聲氓搂碌茅聬聫氓聽垄莽麓?``p95`` 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜麓莽聣聽茅隆垄忙聸垄忙聲聝茅聢搂忙聜驴茅隆聲芒聜卢茅聫聦莽聠录氓鹿聠茅聬聴氓聽茅聴聛氓聯聞氓芦篓茅聧聨莽聰赂氓录禄茅聢楼忙聞聛氓聬聥茅聳赂忙聞漏氓聤聦茅隆忙垄潞茅聧聸氓漏聞忙聡聝莽录聛莽禄聵茂鹿垄茅聨庐茂驴聽茅聰聲芒聜聬聬氓漏聡忙聼聲忙驴聻脩聙氓聛聯忙驴隆忙聫聮莽聣聫莽禄卤忙聮麓忙聥聟茅聨禄忙聨聧氓茅聴聜盲陆赂忙聠隆氓篓虏茂鹿聜氓麓鹿茅聳露聠氓聛聜忙驴聻氓聴聵氓聻聼茅聧聥忙聮禄氓麓路茅脩聟氓聤赂氓茫聜聨氓聨录茅聧垄忙聼楼忙聜?
    """
    metric_views = _read_metric_table_views(experiment_output_root / "eval" / "metrics" / "metric_table.csv")  
    metric_rows = metric_views["scene_summary_rows"]  
    experiment_id = experiment_output_root.name  
    if experiment_id == "e9_dual_degradation":  
        hard_filters = ("A3", "N3")  
    elif experiment_id == "e2_async":  
        hard_filters = ("A3",)  
    elif experiment_id == "e3_nlos":  
        hard_filters = ("N3",)  
    else:  
        hard_filters = ()  
    if baseline_metrics is None:  
        baseline_p95 = _mean_metric_for_method(metric_rows, method_name=baseline_method_name, metric_name="p95")  
    else:  
        if "p95" not in baseline_metrics:  
            raise ValueError("baseline_metrics must expose p95 for search scoring")  
        baseline_p95 = float(baseline_metrics["p95"])  
    return {  
        "rmse": _mean_metric_for_method(metric_rows, method_name=method_name, metric_name="rmse"),  
        "p95": _mean_metric_for_method(metric_rows, method_name=method_name, metric_name="p95"),  
        "failure_rate": _mean_metric_for_method(metric_rows, method_name=method_name, metric_name="failure_rate"),  
        "mae": _mean_metric_for_method(metric_rows, method_name=method_name, metric_name="mae"),  
        "ekf_p95": baseline_p95,  
        "hard_p95": (  
            _mean_metric_for_method(  
                metric_rows,  
                method_name=method_name,  
                metric_name="p95",  
                scene_filters=hard_filters,  
            )  
            if hard_filters  
            else _mean_metric_for_method(metric_rows, method_name=method_name, metric_name="p95")  
        ),  
        "hard_failure_rate": (  
            _mean_metric_for_method(  
                metric_rows,  
                method_name=method_name,  
                metric_name="failure_rate",  
                scene_filters=hard_filters,  
            )  
            if hard_filters  
            else _mean_metric_for_method(metric_rows, method_name=method_name, metric_name="failure_rate")  
        ),  
    }  


def _search_candidate_signature(model_name: str, overrides: dict[str, Any], seed: int) -> str:
    """莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳鹿茅聴麓莽莽潞戮氓陆聮忙聼拢茅聨麓茂陆聟脨聯茅聳赂氓漏聜忙陆隆茅聫聦茫聞漏氓录麓茅聬聬忙碌聥氓聛聯猫陇掳忙聨聭莽拢禄茅聳驴莽聠潞氓搂碌茅聬聯氓驴聸忙聼聸茅茂陆聡莽庐聳莽聙鹿氓聭聤莽鹿聢茅聫聦忙露聵脦聰忙碌拢莽篓驴忙聭碌茅聴聛猫氓聳聴茅隆篓氓聭麓氓聼楼忙戮露氓篓戮忙聻聨忙驴隆氓聨搂忙驴庐忙麓垄禄?茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙碌拢茅聺漏猫聙驴茅聴聛莽搂碌茅聫聟芒聜鹿驴氓聞卤氓娄芦忙聴聜忙聧垄茅聧隆忙露聶莽聟聲忙驴聻氓聴聴忙碌職氓娄虏莽芒聰聭茅猫聜漏脨垄茅聳鲁猫搂聞氓聻驴茅隆垄忙卢聯忙聜路忙拢掳盲陆赂茅聴聜氓聜職氓聙路茅聭鲁氓聣聛氓聢聮茅隆聳忙禄聞莽庐聮氓漏碌莽聤聻录茅隆脙潞茅聫聧莽聟聵茅聳鹿猫戮戮茅聶聞茅聧聤莽聝陆忙聼聸氓篓聠氓驴拢忙篓聢忙驴隆莽鹿聺氓忙麓聜路茅聬聴氓聽聺忙職聢茅聴聛氓聹聧聼氓聺聴忙聜戮茅聬聭猫聴聣莽庐拢茅聳驴忙聸聴氓聛聯莽录聛莽聤虏茅聫聦芒聞聝氓戮聞莽聙拢盲禄录莽录聜盲陆鹿氓聤聴茅聧聥茫聜聣氓录禄茅聰聺氓聽垄莽聳聙茅聳潞氓聸漏氓聛聵茅聨麓莽聝陆忙垄潞莽禄聥忙氓聞聬茅聳鹿氓聭聤忙聜聜茅聧漏茫聢聽忙戮露氓忙陆聤茅聴聛茅聺聸莽聣聠氓娄虏聞氓鹿聫氓篓虏芒聮聭茅聳录氓搂聬氓聙聲茅聫聥忙聢聺茅聬聼莽聣聢忙莽聜麓茅聨庐芒聲聝莽麓聮氓娄聟聨氓鹿聢茅聴聜盲陆赂忙聬芦茅聧聼莽聤禄氓麓聭茅聨戮忙路潞聹茅聳碌氓氓聞潞氓娄虏氓楼赂忙聼聲茅聧楼茫聞楼氓搂聢莽录聛莽禄聵莽鹿聢氓庐聲氓聽聲氓娄赂猫茅陋聻忙忙垄潞茅聬聼忙聺驴忙聜聭茅聳禄茂陆聨盲潞庐聲忙聞掳氓录麓茅隆篓聫?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬氓聫聣莽聳聞茅聴聜盲陆赂忙聠隆茅聨赂氓聴聸氓麓篓茅隆聳忙禄聝氓鹿聠茅聴聜氓聜聶莽茅聨虏芒聙鲁脙潞茅聰聲氓聜職氓聜篓氓篓麓莽聤虏莽录聧忙聽颅忙聼隆茅聧楼芒聲聛忙聻聰莽录聛芒聶聙氓聛聯茅聴聜氓聜職氓聙赂茅聬聴氓漏聝莽卢聼氓娄陇氓聭颅莽拢驴茅聼芦氓聸篓忙聼搂莽禄禄氓聴聵茫聛聞茅聨掳氓聣聛莽搂露茅聳录莽聣聢氓炉搂茅聨卤茫聞娄莽鹿聺茅聧聸盲禄漏莽录聜盲戮聡氓聳聴茅聬聼芒聲聟莽聧庐氓娄聟录氓芦職茅聳录莽垄卤氓聣聶茅陋聻氓忙垄禄忙碌拢氓聭聤猫麓隆茅聳赂氓潞聸氓聙聲茅隆芦氓驴聸忙聡隆茅聤聢氓聮聛忙拢聹忙驴聻氓炉赂氓搂麓茅隆聭氓聴聸忙聜隆茅聫聡茫聢隆氓聙露茅聳禄忙聞陆聝茅聢路忙聴聢忙聼聧茅忙露聶莽茅聳鹿莽禄聽忙露垄忙聲聬茅聧聸莽聺聞茅聴聜盲陆赂忙聬聬颅氓陇聦氓戮聤忙楼拢氓聠陋氓碌聛氓漏碌莽聤虏忙麓娄茅聬聯忙聸垄忙聼隆茅聧聬茂陆聟盲潞戮茅聴聛忙聬聻氓聤聫茅聧漏忙聽虏職忙聺聢茫聞搂莽炉聙茅聳赂忙聸聵氓聤聦忙碌聹莽聜鹿聸茅聨麓莽聝聠盲录聠茅聳鹿氓炉赂氓聛聸茅聧篓忙聴聢忙聼聼莽录聛茫聢聽忙聻聼茅聳禄忙聮露氓聳聳茅聫聦茂陆聣氓录庐茅聧聦茫聜聡氓聤赂氓茫聜聝氓聞卤茅隆聭氓氓录禄茅聰聺氓陇聥氓聛聞茅聳赂忙卢聫茅聬聺忙掳露忙垄潞茅聧聺氓聥聛卤茅聳赂莽聤鲁莽聣聲茅聬拢氓鲁掳脦聰茅聧聸茅聴聛氓聟录莽楼麓茅聫聟忙驴聥氓聼垄猫鹿聡忙聸聽莽拢陆茅聳赂氓卤戮莽聭篓茅聧聫氓虏聙莽麓聮茅氓聭聛娄莽聙鹿忙聸聼莽鹿聜茅聢禄氓潞篓氓鹿聵莽聮潞猫聶鹿茫聜娄茅聴聜氓漏聨氓聛聞氓篓虏茂鹿聜氓录禄茅聤聤芒聲聤忙聲聥茅聴聛莽搂碌茅聬聯忙露聵莽聮潞盲戮炉氓聞聫茅聳禄氓驴聤忙聶聽茅聫聦茂陆聣茅聤聫莽聥聙盲录聝氓漏碌茂鹿楼氓猫聴聣脦聸茅聧聬茫聞垄忙茅聫聣氓聽聸氓聣卤茅聤聢氓陇聥芒聮聭莽录聛氓忙陆隆茅忙楼聝莽聴陋莽录聛茫聢聽氓鹿聵茅聳赂忙聨聯氓赂聴莽禄禄忙驴聠茅隆聮氓聥芦忙聲鹿茅聴聜盲陆赂忙聬芦氓篓虏茫聞漏氓麓掳忙聬麓茫聞楼莽聞碌忙陇陇忙聨聭氓聙鹿茅聫聠忙聟篓忙驴聽氓聜陇莽聟录莽聙鹿忙聸聻茫聜赂茅聨庐茫聢陇氓鹿聛茅聨潞氓聴聴盲潞戮茅聳鹿莽聝聵忙垄禄莽潞戮莽聟聨莽聮潞氓聯聞忙陇陇氓驴聲氓搂路莽禄卤忙聨聯忙陆聫茅聤聤茂鹿聙莽垄聧氓茫聜聠忙聦赂茅聤聢莽莽聲路茅聤聤芒聲聤忙聲聧忙驴漏聢莽碌陆氓氓露聠莽鹿聺?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    model_name : str
        忙驴聽莽聰碌氓搂路茅聫聛氓聭聤忙聟篓茅聬聭氓芒聜聜聻忙聦聤忙戮鹿芒聲聥莽露聛氓庐聲茫聞漏氓录露茅聨麓忙驴聢芒聜卢茅聤聢氓聭炉莽庐掳茅聳禄忙楼聙忙拢聺茅聨庐莽聜潞忙聜陆茅聬垄氓聻芦茂录聫莽录聜氓聜聶茅聶聧茅聧篓莽聭掳茅隆聲芒聜卢茅聫聦忙露垄氓聼聻茅聨炉脩聜氓拢聲茅聴聜氓聜聶猫聝露茅隆聽垄忙聧垄茅聨麓茂赂陆忙聶陋茅聴聜盲陆路莽麓禄虏氓聺聴氓麓聴氓娄莽鹿聠茅聳禄忙聢拢忙氓聰聴聛氓庐聽忙隆篓茅聭鲁氓聽聲氓麓聬茅聬聬莽聝聵芒聮聭莽录聛忙麓聳忙戮搂忙聼楼忙聼拢茅聬聰忙聺聭氓聤聹莽录聛氓聜聸氓赂隆氓庐聲茂驴聽忙聜聶茅聢潞氓聜聸忙聺?
    overrides : dict[str, Any]
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳禄忙聢拢氓搂陇茅聧聤氓聺聴忙聼篓茅聫聡忙楼聙盲潞戮氓茫聜聨氓聨录茅聬聫忙驴聤莽聛聮茅聴聛氓聟录莽楼麓茅聫聟忙驴聥忙聜隆莽聙拢芒聮聭茅聳赂忙聮麓氓陆聝忙碌聹忙驴聡莽鹿聸茅聧聶氓猫聙驴莽聙鹿忙聸聼氓聻驴茅隆垄忙聴聢氓麓录茅聬聰氓聯聞氓鹿聢茅聴聜盲陆鹿忙聻录茅聫聟忙露垄氓麓碌录忙聺聭莽庐聽茅聳赂忙聸篓氓聻卤茅聬聯忙聸垄忙聼拢茅聫聜忙聦聨氓聲聻茅聳碌氓聸篓莽鹿聢茅聫聦茂录聞盲禄娄茅聧聯脩聞氓聢聸茅聴聛莽鲁聲忙聶聸莽聙職忙聺驴莽聲路氓芒聜隆垄忙聴聢氓麓篓茅隆聯莽聠潞氓陆录聜氓聜職氓聙路莽禄聙盲陆鹿忙聶露忙碌聽氓莽拢禄茅聳鹿莽聜聣職茅聬聰氓聴聵忙聟篓氓猫炉虏盲潞戮茅聴聛氓聯聞忙聹篓虏莽聠路莽聲路茅聬聯氓潞聵忙聰陆茅聳赂茫聝聭氓卤戮脨聼茅聴聜氓聜聶莽聞娄茅聨庐氓聽聲氓麓聬茅聨掳忙聺驴莽拢禄茅聳鹿氓聣搂莽虏炉茅聢路忙聨聭脙鹿茅聰聺氓聭庐忙聝聢茅聳赂忙聴聯莽庐聭茅聢鹿忙聢娄茅聭虏芒聰卢脩聟芒聜忙麓聵茅聧聨氓楼赂脣聡莽聙聸忙篓赂盲潞聹茅聰聹氓漏職氓麓露茅聤聤茫聢隆氓芦录莽录聜盲陆潞猫聶聨茅聧聵氓楼赂氓鹿聤忙聬麓茫聢垄莽聭聹莽聣聢氓赂聴茅聬聯忙聸垄忙聜聴茅聰聺氓潞隆盲潞聹氓漏碌莽搂露茅聢?
    seed : int
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳鹿茅聴麓莽莽潞戮氓陆聮忙聼聼茅聴聜氓炉赂莽录聛莽聤碌氓篓聙茅陋聻忙聽篓忙陆聫茅聧聯脩聟脡聭茅聬聨猫忙麓聵茅聬聯茂驴聽氓录禄茅聬聰猫隆隆氓聛聯氓篓聭忙卢聵莽聞陆氓漏碌芒聙鲁忙麓颅茅聫聦茂驴聽氓聼聙茅隆聮氓聥卢忙聶聺茅聳赂氓卤戮莽篓聯茅聧聵忙聮禄忙聜路氓氓聠氓漏垄猫忙麓聼茅聨庐芒聞聝忙聝聰茅隆聰忙聝搂忙聲赂茅聴聛猫聧陇氓搂戮氓篓聟茂陆聡盲潞赂茅聤聤芒聲聛莽茅聳禄忙聴聡茫聙聙茅聧聥忙聮鲁芒聵聣氓篓聠忙聢聽莽聳聞氓漏碌茂鹿楼氓娄聻盲禄聥氓鹿聝氓氓聽聺莽聟聸氓篓麓茂陆聡莽隆露茅聨路茫聜聣忙垄禄忙碌拢猫聴聣氓聬鹿茅聳赂莽聠路忙陆隆氓炉聫聮茅聧聲忙聰鲁?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    str
        莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿氓氓卢碌路茅聬聴聴茫聙聮氓茂陆聟盲潞戮茅聴聛氓聯聞忙聹聬漏氓鹿聝茅聤聫茫聜聠莽聟楼茅聬聨茂陆聨盲潞聶氓篓拢忙垄禄忙碌拢莽颅聥氓職聝茅聳赂茂陆聫莽碌驴莽禄庐氓漏職氓录陆猫陇聫茅聫聠茂赂陆莽鹿聺忙驴聽氓聜聸茂录聧忙驴隆猫聦聛聯忙路聡氓漏聡忙聜聶忙拢掳忙驴聡莽露聢忙驴聽莽聰碌氓聛聸茅聰聲忙驴聽忙掳颅氓聛聞茅聳禄忙聮鲁忙碌路茅隆娄忙聺驴茅聤聢氓聴聴氓搂聣茅聳赂莽聤虏氓莽禄聜氓露聠茫聜拢茅聧聬芒聲聝氓聨碌茅聴聛猫氓聻卤氓忙篓驴茅聴聜盲陆潞茅聰聲茫聜聣忙聜聵氓搂聵猫戮篓氓麓虏忙驴聻忙聢聶氓聻鹿莽禄聽茂陆聣忙聼拢茅聨掳盲禄聸茅聳赂茂陆聢忙茅聢?氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄茅聳禄忙聞聬芦茅聧篓氓聜聻忙聼聸茅茂陆聠莽莽录聛莽聤鲁氓陆聝茅聤聠忙聨聭茅聰聺氓聭忙驴隆莽聜陋氓聙聭茅聧篓忙麓聡聺忙楼拢氓聠拢芒聜鹿聜氓娄赂茅聢潞盲陆赂莽陋聸氓娄聻茫聜聠忙麓聳茅聨鲁氓驴聲莽麓聻氓娄陇盲陆潞莽炉聙茅聳禄茫聢陇忙陆搂茅聢禄氓聽聼莽鹿聸忙碌拢氓聠虏忙碌聜茅聫聛氓聥聼聸氓篓聭氓聧聻莽聛拢茅聳禄忙拢聸茅聢搂氓聨聹茅聧聥忙聮禄忙聼聸茅聫聡茫聞楼氓垄掳茅聳赂忙篓潞氓麓卢茅聢鹿忙聢娄忙聜聶茅聫聣忙聢聽盲禄麓茅聬聨猫忙麓娄茅聧聯莽聠录忙聜掳茅隆聰氓聵聣忙聴聢氓麓篓茅隆聰忙聝搂氓鹿聢茅聴聛莽聭掳氓聬聥茅聬拢氓麓鹿茅聧聺氓聥颅忙職娄莽聙鹿芒聜卢茅聧聲氓聨赂??茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙聮虏莽茅聧篓忙驴聡茫聛聞茅聫聧茫聞搂茫聙聮莽录聛?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓聙驴茅聳驴忙聴聢忙聲庐茅聧聥忙聮露莽虏篓聭忙篓禄氓聶陆茅聳禄莽聭漏茅聫聦莽聠赂忙陆聫忙楼拢氓聠漏茅聴聛莽篓驴茅氓陇聬忙聜隆茅隆聬茂鹿聙莽聜聤茅聳碌氓篓搂芒聜卢忙驴聸卢忙垄潞莽禄聥忙卢戮氓聞聬茅聳驴忙聸聵氓聻陆氓氓漏職氓录麓茅聬聰茅莽陋聻茅聴聛莽鲁聲氓聣聺猫聯卤茅聳禄忙驴聡忙氓搂聤氓聯聞脙潞莽聮聡忙聫聮莽庐潞氓茫聜聝氓聤聦茅聰聲忙驴聽忙掳漏茅聳碌氓陇聢氓聻潞茅聫聜芒聜卢茅聴聜盲陆潞莽隆露茅聧聯忙驴聥茫聙聝茅聧隆忙陇禄忙聻聰忙驴聽氓漏聜莽聣聤茅聢路忙聢聽莽麓聮莽聙拢氓聞楼茅聳赂氓潞垄氓聤聶茅聫聦莽聠路莽聛聻猫聴聣忙聝聻莽录聜盲戮聡氓聶聧聶氓聠鹿聝茅聤聫茫聜聠氓聛聜茅聨录猫戮戮莽碌陆忙驴鹿聜忙垄禄忙碌拢氓聯楼茂录聧莽录聛氓聻露茅陋聻忙聢娄氓麓露茅芒聜聫聥盲戮聙忙聼聼茅聬聴氓聽聼莽路虏茅聳禄忙聞卢茂鹿聫聦茫聞漏氓录庐茅聧楼忙忙驴聽忙氓聻聧聴氓聠虏莽录聛忙篓禄氓氓篓聯忙掳录氓聻鲁茅聧聰忙聬聛忙垄潞茅聧聺氓聥颅氓垄卤茅聳赂忙聮麓莽鹿聠茅聤聡忙聞卢茅聧聠虏忙聥潞莽录聜盲陆赂茅忙卢聯氓麓聲茅聨掳莽聣聢莽聟聶莽录聛氓卢聬聯氓聟?氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄茅聳禄忙聞聬芦茅聧篓氓聜聻忙聜鹿忙聺聻忙聥聦忙驴聻氓聟录莽聣聹莽禄卤忙聮麓忙聥聟茅聭虏芒聞聝氓聻驴氓庐聲茂陆聢莽庐聸茅聫聡忙楼聙盲潞戮茅聳禄莽聠赂忙戮聵茅隆聫猫茅隆篓氓聽垄莽麓聯茅聨戮氓炉赂茅聬聨茂陆聨盲潞聹莽禄聬忙露聶莽鹿聺茅聬垄茅聺聸脨垄茅聳赂莽聡聛氓聻露茅聨?
        茅聴?``sha1`` 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聫茅聢搂莽聞娄莽禄禄忙驴聥氓麓聵茅聢潞盲陆赂盲录聨茅聴聜盲戮聙忙陆搂茅聬聴氓聴聸脣聡茅茂鹿聙莽聞?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬氓聫聣莽聳聞茅聴聜盲陆陆茅聧篓氓楼赂忙聜聵茅聨掳氓聳聠茅聳赂忙聸聻莽聛職忙聲聢茅聴聛莽聭掳氓聬聥茅聬拢氓麓潞茅聧聲莽聞陆茅聳禄忙聳驴氓聬聥茅聢路忙聢聽莽聴陋茅隆聯莽聜麓氓聶潞莽聭聹忙聤聤氓聭炉莽拢录茅聳禄忙篓潞氓聲驴茅聬聫猫戮漏莽麓?
    """
    ordered_items = sorted((str(key), value) for key, value in overrides.items())  
    flattened = "__".join(f"{key}={value}" for key, value in ordered_items)  
    digest = hashlib.sha1(f"{model_name}__{flattened}__seed={seed}".encode("utf-8")).hexdigest()[:16]  
    window_size = overrides.get("window.size", "na")  
    hidden_dim = overrides.get("network.hidden_dim", "na")  
    learning_rate = overrides.get("train.lr", "na")  
    return f"{model_name}__w{window_size}__h{hidden_dim}__lr{learning_rate}__s{seed}__{digest}"  


def _iter_neural_override_grid(
    model_name: str,
    *,
    neural_grid: dict[str, dict[str, Any]] | None = None,
    liquid_robustness_profiles: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聫茅聢搂莽聞娄莽禄禄忙驴聥氓麓聮氓娄陇盲陆潞盲潞聶忙驴聽莽聰碌氓聛聻茅聧篓氓麓鹿氓篓虏莽聳戮?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳鹿茅聴麓莽莽潞戮莽聭掳莽聙拢忙聧拢茅聳禄忙茅聹聣茅聳驴忙驴聠忙麓陇茅聧聰氓芒聜忙麓聵茅聬聯茂驴聽氓录禄茅聬聰氓聭聤莽露聡茅聳赂忙聮聴氓聠篓莽聟聜茅聴聜盲陆赂忙聬芦氓娄露垄氓麓聭茅聤聢氓聮聛忙陆聳莽录聜盲陆鹿脡聭忙驴聮庐忙聜路氓篓聠氓驴聯氓篓麓莽聤聶聽氓搂聤忙麓陋氓麓篓忙驴聽氓聠聫聠茅聴聛忙麓禄氓聨录茅聧聤氓聺聴氓娄麓忙赂職芒聜卢氓炉禄聥茅聬聬茅聺聸芒聜聜潞忙聜路氓氓聠戮忙篓聙莽聙碌忙聜聜氓炉麓录茅聬聰氓聯聞氓鹿聧茅聴聛氓聯聞茅聬聴氓聴聵忙聶露忙碌聽氓氓聛漏茅聫聣氓聽職忙聶聺氓娄聼聼莽聭聶氓聥芦氓搂娄忙驴聻氓聟录氓聤聴莽禄卤忙聨聰莽麓聮氓娄炉氓聟录氓聢聸茅聬聨猫忙麓聳氓庐聲莽聛聮茅聳禄莽聤虏忙麓漏莽聛聫忙卢聫芒聳聯茅聤聢氓陇聥芒聮聮氓篓麓茂陆聞猫颅娄茅聬聮莽聜聛聞茅聨麓忙驴聡忙楼聽莽聜麓氓聤聴茅聨路莽聤虏茅氓聟录茅聴聜盲陆赂忙聠隆氓篓虏茂鹿聜氓麓聭茅聧隆忙露聵忙卤虏茅聳赂芒聞聝莽篓聮茅聬聯氓聠聠赂氓漏聤氓聽聺莽陇聢忙驴聻忙聢聻猫聣戮茅聧篓忙驴聥忙聼隆茅聧聬茫聞娄氓聜篓氓篓麓忙禄聝莽露聤茅聫聦莽聠录忙聜聹氓娄忙職聴茅聴聛氓聭聤脡聭茅聨庐氓聜聻氓鹿聝氓娄陇氓聭炉氓碌聛茅聳赂氓聳聳忙驴庐氓潞隆芒聰聭茅聬聬猫聦卢忙鹿麓茅聳赂氓漏聝氓茅聢搂莽聲路忙驴聜莽氓聧鲁茅聳禄忙聳驴忙聬芦茅陋聻忙莽麓聯茅聧聦忙卢聭忙陇陇茂赂驴氓聧鲁茅陋聻忙聞掳氓鹿聨忙驴庐忙聞篓氓聛聯忙碌拢氓聣虏忙陆漏茅聳录莽篓驴莽聛職氓篓聟忙禄聞莽鹿聺茅聤聫茂陆聠莽隆戮茅聳驴忙聸陋莽庐聯氓庐聲忙驴聥忙聲聝茅聢搂芒聙聰茅聧聬脦搂茅聳赂芒聞聝茅聬聼茫聞漏忙聼拢忙聬麓茫聢聽氓職卢茅聳赂忙聮麓莽篓聯茅聧聮忙聨聯莽鹿聺氓搂聵忙芦聹忙驴聽茫聢拢忙鲁聸茅氓聸麓垄茅聬垄氓聭聤芒聮聮氓篓聯忙掳麓陇氓炉路篓莽麓聯茅聧聦忙露聶莽職聭茅聳录忙麓陋盲潞露茅聨鲁莽聜潞氓娄路茅聰聺氓聥颅莽聸聨茅聴聜氓漏聨氓聛聞氓篓虏茂赂戮莽虏聶茅聨掳忙聺驴莽庐聼氓娄陇茂陆聟氓聲聬聯忙卢聬茫聛聞茅聫聝芒聜卢莽聙碌脩聟忙露碌忙楼拢氓聠漏莽聟聝茅聳陆忙篓潞氓娄虏忙聬聛忙聭露茅聰聺氓莽聟聽忙驴聻忙聺聭氓篓聟氓聭聰聭?茅聴?忙驴聽莽聰碌氓搂路茅聫聛氓聭聤忙聟篓茅隆聯茫聢聽莽庐聳赂忙聞碌氓聳聳氓庐赂忙驴聡茫聛聞茅聫聝氓聜聸忙芦聲莽聭聹忙聤聤氓聸篓芒聮聮氓篓麓茂陆聡莽聯聰茅聧陇氓聠聼聸茅聬聵忙聞聛忙庐聰忙陇陇莽聛聺茅隆芦忙禄聢氓聼聙茅隆聮氓聥芦莽庐聳茅聳鹿芒聜聫聙盲禄聥忙聜聴茅聰聺氓潞聺莽掳陆茅聳赂忙聴聯莽庐聯氓搂聤氓聯聞莽聲路茅聧楼芒聲聛莽茅聴聛氓聹颅忙聡聯氓篓虏忙聜掳茅茅聰聛聯茅聰聺氓潞隆忙聻聼茅聳赂氓卢聤聶茅聫聦忙露垄氓鹿聡茅芒聜禄戮脩聞忙聟篓茅聰聺氓聭聳聵忙驴漏聞莽虏聨路氓聭麓忙聬鹿茅陋聻氓潞聯氓聛聯忙驴聽氓聸篓莽聟聲茅聳鹿忙聝搂莽禄聽莽聜潞忙聼聼茅忙聝搂氓聨麓茅聳潞盲陆赂氓聙驴茅陋聻氓聴職忙驴隆莽虏聯氓忙麓麓芦茅聧聲氓聻芦脨娄氓漏碌莽聜虏莽職路茅聧聯忙驴聡莽漏卤忙驴聽氓聸搂茅聠聡茅聳潞氓聸漏氓聛聸莽禄聬忙露聵茅聤聢氓聴聶氓垄录聛莽搂麓茅聢禄忙聮禄茅聬聬莽虏鹿忙聥禄?

    忙驴聽莽聰碌氓搂路茅聫聛氓聭聤忙聟篓茅聬聭猫聴聣忙聻聛茅聳鹿茅聴麓莽莽潞戮氓忙聼拢茅聨掳忙聝聢茅聳赂茫聞楼氓聙驴氓氓漏職氓麓录茅聬聰氓聣聛莽鹿聛茅聴聛忙聬聻氓聙聳氓篓虏莽聠录氓录禄氓篓聭茫聢漏忙聲聝茅聳驴忙驴聠忙茅隆娄茫聝漏忙垄潞莽录聛忙篓潞莽卢聮茅聳禄氓驴聰氓虏赂忙驴隆莽聰赂氓麓聼茅隆聳忙掳卤茅聴聛忙聦聨莽鹿聜氓娄芦忙露垄氓娄麓忙驴聽氓聸篓芒聮聭茅聳赂忙露聵茂鹿楼茅聬聯茫聝搂莽麓聮忙戮露氓漏聞莽搂潞忙楼聽莽聜虏茂鹿陇茅隆芦忙驴聢忙聧聞茅聯聫莽聭掳氓鹿聬茅聴聜盲陆赂忙聠隆忙赂職忙氓麓掳氓猫路潞氓聜篓茅聴聛莽搂碌茅聢路忙聢娄忙垄禄茅聧芦莽聠露莽路卤茅聳禄忙聨聴莽庐聳茅聫聦忙露聶忙聝聰茅聤聫脗掳盲禄垄茅聬聨猫忙麓聵氓篓虏莽聠录莽聧庐氓搂聴芒聜卢氓庐聲忙禄聞忙聥聟茅聬職氓炉赂氓潞聤莽录聜氓聜職氓聙赂茅聧聤莽聝陆忙聜聻茅聰聲盲陆鹿忙聡聦莽录聜氓聜聶猫茅聧聥忙聮禄忙聼聸茅聨掳茅聺聸忙聻聼茅聳赂氓漏聞忙聲禄茅聫聦茂陆聣氓茅聤聢氓聴聵茅聰聳禄茫聢漏氓聻碌氓篓聣忙露垄芒聙聰茅聧聬脦搂茅聳赂芒聞聝氓篓录忙聢娄忙垄潞莽禄聥忙卢戮氓聞聬茅聳鹿莽氓垄陆氓娄虏忙聞掳氓鹿聵茅聨掳盲陆赂忙聜聭茅聴聛氓聭聤盲戮聴茅聧篓忙聤陆氓麓聺茅莽聜麓芒聮聭茅聳赂忙驴聠氓芦漏聙盲录麓忙聼拢茅聢潞氓漏聜莽聛聽忙陇陇忙聸露聛茅隆垄忙掳露氓聼聙茅隆聮芒聜卢茅聬拢莽聝陆氓鹿聮茅聨麓氓聝碌茅聴聛忙聦聨莽鹿聜氓娄露垄氓聼聙茅隆聭氓聻庐氓碌聠忙驴庐氓漏聞莽虏炉茅聨路氓聭炉莽虏碌莽聙拢茅聴聜盲陆赂忙聠隆茅聫聦茂陆聣氓麓聬氓漏碌氓卢茅聳赂忙麓聵茅聧聤莽聝陆忙聼拢茅聨麓莽聝聠莽聞陆茅聳赂忙忙聲禄氓猫聶鹿莽麓聮氓娄聶鹿盲禄赂茅聳陆氓聠潞露茅聨莽庐聸莽聙拢茅聴聛氓聯聞忙聹卢莽禄聥忙聮禄氓聼聙茅隆聮氓漏聞莽搂碌茅聳赂氓卢聡聨茅聬拢莽聝陆氓麓聼茅忙露聮忙聥禄忙驴聻忙聮麓氓聼聝茅聧聥忙聮麓莽鹿聸忙碌拢氓聠虏氓聬聵氓篓聭忙篓聹忙聴聢氓麓篓茅莽聥聟莽麓庐茅聳禄莽聠赂莽虏聨忙戮搂忙楼聝忙聲庐茅聨潞忙聴聜盲赂聧忙陇陇忙聽聛聻茅聬聯忙聸聼莽鹿聸茅聨麓忙驴聠猫聢鹿莽聙碌氓聳職茅聢搂氓篓虏忙漏聙忙聲聝茅聤聫茫聜聟忙聻聛茅聳鹿莽聣聢忙戮聵氓忙驴聡茫聛聞茅聨潞氓聴聴盲潞戮氓?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    model_name : str
        忙驴聽莽聰碌氓搂路茅聫聛氓聭聤忙聟篓茅聬聭氓芒聜聜聻忙聦聤忙戮鹿芒聲聥莽露聛氓庐聲茫聞漏氓录露茅聨麓忙驴聢芒聜卢茅聤聢氓聭炉莽庐掳茅聳禄忙楼聙忙拢聺茅聨庐莽聜潞忙聜陆茅聬垄氓聻芦茂录聫莽录聜氓聜聶茅聶聧茅聧篓莽聭掳茅隆聲芒聜卢茅聫聦忙露垄氓聼聻茅聨炉脩聜氓拢聲茅聴聜氓聜聶猫聝露茅隆聽垄忙聧垄茅聨麓茂赂陆忙聶陋茅聴聜盲陆路莽麓禄虏氓聺聴氓麓聴氓娄莽鹿聠茅聳禄忙聢拢忙氓聰聴聛茅聺聸脛聸茅聧聸莽路聴聜氓聜聶莽茅聨录脙潞茅聤聤茂赂戮莽茅聳鲁脩聦氓聣職茅聨卤茫聞娄莽鹿聺茅聧聦忙露聶氓聞聯茅聳禄忙聮卤氓聙聴莽禄卤忙聮麓忙聥聟忙驴聢颅盲潞聧漏茫聢拢莽庐聸氓篓聭忙卢聦忙聥潞茅聴聛氓聭聤莽篓聭茅聰聲茫聝漏忙聜隆茅聤聣脩聟莽拢录氓猫路潞莽聛聫茂鹿聜忙聼隆莽聛聻氓聣搂忙麓麓氓漏碌芒聞聝忙聜聜茅聧漏芒聞聝忙聥聟忙赂職忙聶拢茅聴聜盲陆赂忙聬芦茅隆娄忙聜搂茅聧隆忙篓禄忙芦聽氓篓麓莽聤虏莽禄聽忙聽聼陇氓篓麓氓忙芦聡莽禄戮氓聣搂氓聨搂茅隆路篓忙聧聞茅聬職忙聺聭氓莽录聜盲陆赂氓娄聻盲禄聥氓录禄茅聰聸氓聥虏陋莽录聛忙麓聳忙碌聹茅鹿聝忙聼聼忙拢掳盲陆潞茅聳鹿氓聻驴氓忙麓麓聹茅聨禄忙聨聧莽掳录莽录聜盲陆陆茅聬聴忙聮鲁忙驴聽忙掳聧鲁茅聳碌氓驴聲氓搂麓茅聭掳氓聻聰聭忙聨鲁茅聧聤忙聞掳氓麓聭茅聨戮莽禄聵莽聟聸茅聳赂芒聵聠氓聨戮莽禄聣忙聼隆氓庐聙氓卢拢驴氓篓麓莽聥聟莽庐聯茅聨庐忙聝搂莽聲聺茅聢搂氓茅聧楼茫聞搂莽庐戮茅聬聨莽聰碌茅忙聮露莽麓聮茅聬聵猫聶芦氓麓聦忙莽聜麓氓聤聳莽禄禄忙驴聥氓麓聵茅聤聤脩聟莽聧聴聛猫炉虏莽鹿聮茅聧聥莽聜潞氓麓聲茅聯聫莽聭聹莽聣聢氓赂聴茅聢路忙聨聭脙鹿茅聰聺氓聸漏忙聭聜茅聳赂茫聜聠莽聭娄莽禄禄忙露職芒聜卢莽聰碌茅聧聵氓炉赂芒聜卢忙庐驴氓聶庐茅聧聥氓漏聡莽聧聫聧茫聞娄氓茅聳赂忙驴聞氓聞鲁茅聫聥氓聠垄禄忙碌拢氓聭聤猫麓隆茅聳赂氓卢聧聡氓庐聲忙聸垄氓麓聴茅聧聫莽聟聨氓录驴茅聴聛猫氓聞录忙驴庐茫聜聣忙聜隆氓篓聭茫聢隆氓聙露茅聳禄忙聞掳茅聧陇忙卢聯忙聼聸茅聫聝芒聜卢氓搂聵茫聞搂忙搂聬茅聨潞忙楼聙莽聞碌忙陇陇忙聨聭氓茅陋聻茫聢隆莽鹿聸茅聨麓莽聜碌忙聡聟茅聳赂忙漏聙盲潞露氓忙麓录卢茅聤聣茂赂陆莽聫聰茅聴聛氓聭聤莽聭娄茅聧聰忙卢聯莽聧聙拢忙聶聺茅聳鲁脩聦氓聣聶忙聲聢茅聳赂忙卢聫莽禄隆芒聜驴聽莽聰碌氓搂麓茅聧聤莽禄聵忙聶露氓篓聭茫聢隆莽聟聼?
    neural_grid : dict[str, dict[str, Any]] | None, optional
        茅聴?茅聴聜氓聜職氓聙赂茅聧聤茅拢聨忙氓搂聵猫路潞莽聞碌忙陇陇忙聨聭氓聙赂忙碌聽忙禄聢忙聼陇氓篓虏猫莽聛隆茅聳潞氓聭聢录茅隆垄忙卢聬忙聟篓茅聨掳莽聸聨茅聴聜盲陆鹿氓氓篓虏氓露聣氓麓聭茅聨戮氓聻聸氓聛聯莽聭聶氓聥卢氓陇聞茅聳赂忙聮露氓聳聳氓庐聲忙麓陋氓搂聙茅聤聢氓聭炉茅聳禄莽聤虏忙麓漏莽聛聫忙卢聯氓篓虏芒聮聭莽录聜盲陆鹿茂鹿漏茅聬聮盲禄聥忙聼隆忙碌拢猫搂聞氓聙聬茅聳鹿氓聛聻茅聤聢茂陆聣氓麓聵茅聢鹿莽聜卢氓芦录莽录聜氓聜職氓聙路茅聬聮茂鹿聜忙聲聥忙驴聽忙氓聳聳茅隆娄莽聰赂氓录禄茅聢漏氓聠漏氓陋麓茅聬聼忙卢聫氓茅聢搂氓聤聶茅聫聦莽聠录氓娄陇芒聜卢忙碌聹茅鹿聝忙垄禄忙碌拢氓聭聤氓聲聻莽录聛氓氓聻驴茅聨庐忙麓娄赂茅聢潞氓聜職盲潞聴氓漏碌莽聜虏忙茅聨赂氓忙聜隆茅聫聡茫聢垄氓聫聯茅聳赂茫聝拢氓聤聫忙戮鹿忙篓潞脙鹿氓氓聠搂潞茅聳潞氓虏聥莽露聛茅隆垄忙漏聙忙聜聶茅聬垄茅聺聸氓搂卤茅聴聜盲陆赂忙聬聫聢莽虏聶茅聨戮氓炉赂茅聳驴忙聴聜氓聛聯莽聙職忙掳露忙聼聼莽录聛忙篓潞盲驴炉茅聳禄氓潞篓莽禄卤忙聮禄氓麓聮氓篓聠忙聞聜掳忙驴隆莽聜虏茅隆聽拢莽聛聥茅聴聛氓聭聤氓聤娄茅聧聽忙掳卢莽聲碌忙赂職芒聜聨庐忙楼聟忙聲聬忙聬麓芒聞聝莽聛聧茅聴聛莽篓驴茂鹿娄茅聧聫忙篓潞忙驴聽忙掳卢茅聠聡茅聳禄忙聴聜氓聞陇茅聧聮忙露聵芒聰聭茅莽隆路忙聰禄忙驴隆莽聜掳莽虏聶茅聬聸氓鹿聡茅莽聠路氓聤聟茅聴聛忙聦聨莽鹿聜茅聧聤氓陆聮茂陆聻氓庐楼氓芒聮聮氓篓麓茂陆聢茅聤聡莽聠录忙聼聸氓娄聛聮氓聛垄莽聙鹿忙聸聼氓聤聳莽禄禄忙驴聠茅氓聜陇忙搂麓茅聴聜盲陆赂忙鹿卤茅聧聨茫聜聣茫聙聝猫鹿聡忙露聵忙聻聛莽聙鹿茫聝楼忙拢聸莽禄隆芒聜驴聽莽聰碌氓搂麓茅聧聤忙聬聛茅聢潞氓聜聵盲潞聹茅聨录忙路聻禄氓篓虏氓颅聵忙聟篓忙驴聽氓聠漏茫聛聺忙驴隆氓聲芦茅聢陆氓茅隆聮氓忙炉聡茅聴聜盲陆陆莽聙聸芒聲聟氓麓聶猫陇掳忙聨聭莽茅聳禄忙聴聜氓聨搂茅聫聞茫聝漏忙聼聲猫鹿聡茫聝搂忙聜录茅聳禄茫聜聜氓聛聯忙驴聽莽聰碌氓聛聻茅聧篓氓聽聲忙聜路猫陇聰氓庐聲?``_FAIR_NEURAL_GRID``茅聴?
    liquid_robustness_profiles : tuple[dict[str, Any], ...] | list[dict[str, Any]] | None, optional
        茅聴?茅聴聜氓聜職氓聙赂茅聧聤茅拢聨忙氓搂聵猫路潞莽聞碌忙陇陇忙聨聭氓聙赂忙碌聽忙禄聢忙聼陇氓篓虏猫莽聛隆茅聳潞氓聭聢录茅隆垄忙卢聬忙聟篓茅聨掳莽聸聨茅聴聜盲陆鹿氓氓篓虏氓露聣氓麓聭茅聨戮氓聻聸氓聛聯莽聭聶氓聥卢氓陇聞茅聳赂忙聮露氓聳聳氓庐聲忙麓陋氓搂聙茅聤聢氓聭炉茅聳禄莽聤虏忙麓漏莽聛聫忙卢聯茅聧聺氓聥篓忙聰陆茅聳禄忙莽虏篓聭脩聡忙聼聸茅聬聲盲陆赂莽聛娄茅聳驴忙驴聢氓聛聸茅氓聟录氓戮聞莽聙拢脩聞氓茅聴聜盲陆赂忙聠隆莽禄聥忙聢娄忙聲聝茅聰聲氓聜職莽庐聽茅聳赂忙聞碌氓聳聴茅聢路忙聢聽莽麓聮莽聙拢氓聞楼茅聳赂氓潞垄氓聤聶茅聫聦莽聠路莽聛聻猫聴聣忙聝聻莽录聜盲戮聡氓聶聧聶氓聠鹿聝茅聤聫茫聜聠氓聛聜茅聨录猫戮戮莽碌陆忙驴鹿聜忙垄禄忙碌拢莽颅聥氓職聝茅聳赂茫聞陇氓聧鲁猫聴聺茅聳禄茫聢聽忙聭聹氓庐聯盲戮聙忙聼聸茅聢漏氓聠娄盲禄聹忙驴隆忙聫聮莽聣聯茅聫聦忙聸隆莽庐聸茅聫聡莽聜聬茫聝娄茫聛聞茅聫聜氓聦隆猫潞驴庐氓漏聝忙陇陆茅聨聝陆氓录露莽聮潞茫聞搂芒聴陇氓漏碌莽聤庐氓聻录茅隆芦氓陇聤脙潞茅聬聰莽聟聨莽庐聳莽聙鹿氓聥炉氓聛聤茅聬聯茫聝漏忙聼聸茅聨掳茅聺聸忙聻聸莽聙碌氓莽禄禄忙驴聢氓聳聤氓娄炉氓聯聞芒聮聣茅聴聛氓聹颅茅聧聼氓聺聴忙聜戮茅聬聭猫聴聣莽茅聳鹿氓颅聵忙垄鹿茅隆芦氓露聟芒聰聭茅聬聬氓聫聣忙驴聻氓颅聵氓虏赂茅聧漏芒聜陇陇忙聨聧莽聞娄氓庐聲氓虏聞忙聼聼茅莽颅聥忙庐聹茅聳禄忙露卤氓聶拢氓庐聲氓陆聮茅聬聯氓潞陆聸茅聴聜氓聜聶猫聝露茅聨鲁忙聮禄氓麓炉茅聧搂忙莽聟聙茅聳驴忙驴聠莽禄聽忙聽篓盲驴聤茅忙驴聞芒聜卢芒聜聳潞氓卢聟拢茅聫聦莽聠录忙聜聶茅盲陆潞莽篓聡忙驴?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬氓庐聞忙聡聤氓陆聮氓麓露茅聤聤茂陆聞莽虏聯茅聴聛氓聭聤莽赂驴茅聧聨忙聫聮莽聲路茅聫聦茫聝娄莽聟聲茅聳鹿盲录麓忙陆搂茅聫聥忙聮禄忙聼聧猫陇聹茅聧聯忙掳聸茅聧聬忙聵楼茅聳鲁脩聦氓聣職茅聤聡氓聥鹿聮氓庐楼氓聴漏忙篓聜莽录聜?氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄茅聳禄忙篓驴莽禄聽氓聻聛聞猫鹿聡忙聮麓莽录聛芒聜卢忙驴聽氓聸篓莽聟聝莽聭聹忙禄聢氓麓聹氓搂聵猫戮篓氓麓虏忙驴聻忙聢聻莽聭楼莽禄露莽聜潞忙聜聴茅聰聺氓潞隆盲潞聻忙陇陇茂赂驴氓聧鲁茅聨庐?``_LIQUID_ROBUSTNESS_PROFILES``茅聴?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    list[dict[str, Any]]
        忙驴聽莽聰碌氓搂路茅聫聛氓聭聤忙聟篓茅聬聭猫聴聣忙聻聛茅聳鹿茅聴麓莽莽禄聙氓聴聸忙聼聼氓篓聢氓聻庐忙聻陇莽禄戮忙聝搂茫聜赂茅聨陆聮氓麓露猫茅聢禄忙聸垄忙聼聸忙聬麓茫聢隆茅聳潞氓虏聥莽麓聲忙碌聽茂鹿聜氓麓聬茅聬聰氓楼聣盲录聝忙驴隆莽聜陋氓聙陇茅聧聵氓陆聮氓麓聺茅聨麓忙驴聢忙職娄茅聳禄忙聮鲁莽掳露茅聫聙盲禄聥忙聼聸茅隆聬氓氓篓聭忙聮麓氓陆聝茅聢鹿忙聢娄忙聜漏茅聧篓忙炉聞茅聴聛莽莽禄聥忙聢拢莽聛聥氓漏碌莽聜虏忙茅聨赂氓聯楼莽虏禄忙碌录氓颅聵莽聟聶莽录聜盲陆碌氓聻鹿茅聫聥忙露職莽聴陋茅聨聦莽聟录茅聳潞氓虏聙氓聳聳茅聨庐忙卢聯忙碌拢猫聶鹿芒聳聫莽录?茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙碌拢茅聺漏猫聙驴茅聴聛莽搂麓茅聧聦茫聞搂忙聠赂茅聫聝氓聽聺莽庐聳忙驴聻氓路潞聹忙聝聼茅聴聛茅聺聸茅聧聽忙楼聝莽麓聻忙聬麓茫聞娄莽聟聼茅聳禄忙篓潞氓录露忙戮露氓聥颅莽麓聮忙碌拢猫搂聞氓掳娄莽聙碌忙聜聜氓炉麓录茅聬聰氓聯聞氓鹿聧茅聴聛猫炉虏忙碌路茅聫聛忙聬聛芒聙娄茅聧芦莽聠潞莽聭聹忙聴聢氓录禄氓庐楼氓聸漏莽拢录忙驴隆忙陇驴氓娄芦氓聠垄潞茅聧聺氓聥庐氓芦掳莽录聛氓陇聦氓聟聵莽炉聯氓篓聯忙掳拢氓聛聯茅聳潞氓卤戮莽聸陋聻忙篓录芒聜卢氓聻聜露茅聢搂氓赂聴忙路聡氓漏聡氓麓拢忙戮露氓漏聜氓娄陇莽聤碌氓聛聻莽聰聨聲莽聛聝茅聴聛茅聙聻氓卤陆氓垄掳莽录聛?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聳莽潞戮氓聺聴忙聼拢茅聤聫茫聢漏莽聞戮茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛莽篓驴茅氓陇聬氓录禄茅聰聼莽聤虏莽聜聤茅聳碌氓陇聢氓聞鲁忙碌聽茅鹿聝忙垄潞莽禄聥忙卢戮氓聞聧茅聳赂氓漏聝茅陋聻氓莽虏聶氓篓聠忙聝搂忙聜聭忙驴聽茫聢拢忙鲁聸茅聰聲茂鹿聜氓麓垄氓篓虏芒聮聭茅聳禄莽聠赂忙戮聵茅聢路忙聴聜莽麓聮茅隆聲氓聭颅莽聛娄茅聮聵莽聠录忙聼聸茅隆聬茫聜聟忙聻聰莽录聛莽聤禄莽庐聳茅聫聦莽聠潞芒聜聰碌茅聧聯茅聛聧忙聼拢茅聨潞忙聸聼茅聴聡氓陇聬忙聼拢氓娄聯聞忙職卤茅聳赂忙驴聡忙陇陆茅聫聦芒聙鲁莽聲聺莽聙拢脨聹茅聴聛猫炉隆氓聳聴茅聬聼脩聡氓聼聙茅隆聮氓聜聸忙職漏莽禄戮忙鲁聸猫聻聻茅聳碌氓篓戮忙聶聹茅聢路忙聢娄忙聼聸氓篓聭忙漏聢忙聰禄茅聳禄忙聮卤氓聞碌茅聫聦茫聞漏氓录麓茅聤聤茫聜聣氓聛聴茅聬聨猫忙麓聵氓盲禄聥氓麓潞茅聢?

    茅聴聜氓聜職氓聙路氓篓麓氓聸茅聢搂莽篓聢茅聫聟莽聤禄茫聙聛氓篓聠忙聮鲁氓聠聙忙陇陇忙聮露氓聛聼茅聬聸茫聝漏忙垄潞茅聧聺氓聥颅芒聳聣茅聳赂忙篓驴氓聛聬忙戮鹿忙聸隆茫聛聬茅聨潞忙聮露氓聨碌茅聴聜氓聜職氓聙赂茅氓麓聺氓氓陇聥莽聟聲?
    ------
    ValueError
        茅聴?``model_name`` 氓漏?``"liquid_ekf"`` 氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄茅聳禄忙聞聬芦茅聧篓氓聜聻忙聼聸茅茂陆聠莽莽录聛莽聤卤氓聧鲁茅聨庐氓陆聮氓麓露猫陇聧氓娄芦忙篓禄莽鹿聸茅聨麓忙卢聫莽聛漏莽录聛芒聜聧聬芒聰驴盲潞聹茅聳潞氓聠篓忙麓娄茅隆楼氓陇聬忙聼聸茅聨戮氓麓聡氓聼拢忙驴庐氓漏聟忙茅聨路莽聤禄氓聻戮氓庐聲氓聸娄莽聰禄茅聴聜盲陆潞氓聞碌茅聫聟芒聲聟氓麓鹿茅聧聸聼氓聛聵忙陇陇忙聸聴氓聛聯忙楼聽莽聜麓氓赂聮猫聻聳茅聳鲁脩聦氓聣聸莽禄聧陇忙聥聽莽聭聛盲陆潞忙聡聯茅隆聞漏氓麓聵茅聤聤茫聢隆氓炉聳茅聴聜盲陆陆猫陇掳氓聠庐氓鹿聣茅聰聼莽聤虏莽庐職茅聳潞氓聠篓莽聣聠忙聝聼茅聴聛茅聺聸莽禄篓忙麓陋氓麓聲茅聬垄莽篓驴猫聯聼忙驴聻忙聢聻莽聼篓聦忙聼聸茅聨戮忙陇禄氓聛聯氓漏碌忙麓娄莽禄禄忙驴聠芒聳聯茅聧篓盲禄漏氓茫聜聠忙麓娄茅聧聶氓聠庐氓麓潞茅聢搂茫聛聞茅聨潞氓聴聴盲潞戮莽录聜盲陆潞氓搂麓莽禄聣莽聭掳莽聲路茅聫聡莽聠赂莽露聡茅聳鲁猫搂聞氓聞鲁忙碌聹莽聜卢忙聟篓氓娄炉莽聟聨氓赂露茅聳禄氓驴楼莽碌驴莽禄卤忙聨聴茅聭虏芒聲聛芒聟卤茅聴聛茅聙聻氓卤陆氓垄戮莽录聜氓露聟忙氓庐聲忙禄聝芒聵聣氓忙拢聹茅聴聛氓聟录莽楼麓茅聫聜盲戮聙氓麓聭茅聨戮猫炉虏莽氓漏垄猫路潞茂鹿拢忙戮鹿忙聸垄忙垄禄忙赂職芒聜卢茅聢搂氓聛聸茅聭禄忙聶露莽聙碌莽鲁聲氓聛聯莽聭聶氓聥忙驴聻猫聦卢氓聳聬忙路聡氓漏聡氓麓录茅聫聡莽聜虏莽陋聸茅聴聛氓聯聞茅聧篓氓陆聮忙聲聧氓漏聤氓聽聼芒聮聭莽录聜盲陆鹿脙驴庐氓虏聞忙聜聵猫鹿聡忙聴聜忙聡聡忙陇陇茫聢隆莽聭漏氓庐聲茫聞漏氓篓聭忙卢聬忙聺赂茅聴聜盲陆潞莽虏炉茅聧聰忙聽搂氓篓聠忙聮麓氓聙露茅聳驴忙聸聻氓聙鹿茅聬聯茫聝娄忙聟篓氓搂聴氓聴聴氓鹿聴茅聳赂忙卢聫莽鹿聵茅聫聦茫聞楼芒聰聭茅聧隆忙篓潞氓漏碌芒聵聠氓聻陋茅聧聥忙聮禄忙垄?
    KeyError
        茅聴?``model_name`` 氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄茅聳禄忙聞聬芦茅聧篓氓聜聻忙聼聸茅茂陆聠莽莽录聛莽聤卤氓聧鲁茅聫聦忙露聵芒聰聭茅聧聲氓搂垄茅聴聛忙麓禄氓聨陆茅聨鹿氓聸录禄茅聢漏氓聠篓莽路聻茅聬聨茂陆聨盲潞聻茅聧聵忙聞掳忙垄潞茅聭虏芒聲聛茅聬聨莽卢聸氓聸搂脦娄茅聳赂忙聸篓氓聳職茅聬陇芒聜聴聛氓聹聛聯茅聨卤陆脦聸茅聤聢氓聸漏莽拢录氓娄陇茂陆聟氓聲掳茅聬拢猫聶鹿莽麓聮茅聬聵氓麓聡氓垄碌莽聙碌茅聨庐茫聢垄忙聜掳茅聢楼氓聟聟氓拢聲茅聴聛忙卤聡氓聻潞茅隆聰忙聽颅忙聜聻茅聨炉氓陆聮忙聜漏氓篓聠氓驴聯忙聻陇茅聳禄忙聳驴忙茅聨鲁掳芒聳聨猫鹿聡忙聯聝盲禄录忙驴聽忙氓聳聳氓篓虏氓录?氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆茫聙聜茅莽聜聳潞氓聸楼氓聻鹿莽潞戮莽禄聵莽鹿聸茅聨麓忙卢聫莽聛漏莽禄戮忙聝搂莽露聤茅聫聦忙露聹忙陇陇茫聞楼莽聣聠茅聬拢氓聻芦莽卢聼茅聢搂莽聧聫聧氓潞篓茅聢搂盲驴聤忙驴聜莽篓驴猫聢鹿忙陇陇忙聽芦氓聛聻茅聢路忙聢娄忙聼聸茅聨掳莽潞搂茅聳鹿莽聣聯茅聫聦忙露聶莽鹿聺茅聧聲忙庐聦茅聴聛氓庐聽莽禄聣氓陆聮莽聧庐氓娄聜漏莽拢录忙驴隆忙聰卤莽聙職氓楼赂忙垄禄忙碌拢氓聭聤猫麓隆忙陇陇莽聣聠茅聹聣茅聳禄忙聢拢忙茅聬聯芒聙鲁茅聭赂茅聺聸氓茅聳禄忙聮露氓聳聬忙路聡氓漏聡氓娄露茅聧聦忙掳芦氓拢聲莽录聜氓聜聶猫聝露莽禄聲聥氓娄?``grid[model_name]`` 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳鹿茅聴麓莽莽潞戮氓陆聮忙聼聼茅聴聜氓炉赂莽录聛莽聤碌氓篓聙茅聬聯氓麓露茅聤聤脩聙忙搂聢莽录聜盲陆虏氓聛聯氓漏垄猫聢碌氓聤聳茅聬聯忙露垄忙聼聸茅聤聣茫聢隆忙芦聟茅聫聧芒聙鲁茅聤聢氓潞隆盲潞聺忙驴聻猫聦卢氓聳聳氓氓漏職氓娄赂茅聢潞氓聜職盲潞聹茅聴聛氓聭聤莽鹿聜莽聙職氓聭麓氓搂聙茅聤聫茫聢垄茂录聹茅聴聛莽禄聵莽聭垄茅聧聥忙聮禄忙聼聸茅聬聵忙聞聛忙庐聵茅聳鹿氓鹿驴氓聻鹿茅聢陆氓陇聬氓茅聬聵忙氓聬聟茅聴聜盲陆陆莽禄禄忙聢聺茅聨赂氓潞拢忙驴聽氓漏聜忙拢颅茅聫聠忙聺驴芒聰聭茅聬聵莽聛卤茅聳赂芒聲聜莽聣聽氓庐聲忙驴聥氓录陆茅莽聠路盲潞聴茅聴聛茅聺聸忙驴隆氓聸漏莽虏禄忙楼拢氓聠漏莽聟聶茅聬聨莽聰碌茅聧聯氓聠虏茫聙聮氓芒聤聲氓篓拢氓聸搂茂鹿陋茅陋聻忙聮禄氓鹿聮茅聫聜猫路篓氓聨陆茅聴?
    """
    resolved_neural_grid = neural_grid or _FAIR_NEURAL_GRID  
    grid = resolved_neural_grid[model_name]  
    override_rows: list[dict[str, Any]] = []  
    if model_name == "liquid_ekf":  
        resolved_liquid_profiles = list(liquid_robustness_profiles or _LIQUID_ROBUSTNESS_PROFILES)  
        if not resolved_liquid_profiles:  
            raise ValueError("liquid_ekf search requires at least one robustness profile")  
        for window_size, hidden_dim, lr, robustness_profile in itertools.product(  
            grid["window"],  
            grid["hidden_dim"],  
            grid["lr"],  
            resolved_liquid_profiles,  
        ):  
            override_row = {  
                "window.size": int(window_size),  
                "network.hidden_dim": int(hidden_dim),  
                "train.lr": float(lr),  
            }  
            for key, value in robustness_profile.items():  
                if key == "name":  
                    continue  
                override_row[str(key)] = value  
            override_rows.append(override_row)  
        return override_rows  

    override_keys = list(grid)  
    override_values = [list(grid[key]) for key in override_keys]  
    for combination in itertools.product(*override_values):  
        raw_overrides = dict(zip(override_keys, combination, strict=True))  
        override_rows.append(  
            {  
                "window.size": int(raw_overrides["window"]),  
                "network.hidden_dim": int(raw_overrides["hidden_dim"]),  
                "train.lr": float(raw_overrides["lr"]),  
                "train.weight_decay": float(raw_overrides["weight_decay"]),  
            }  
        )  
    return override_rows  


def _safe_load_checkpoint_best_epoch(checkpoint_path: str) -> int:
    """Safely read best_epoch from a checkpoint payload."""
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        raise RuntimeError(f"unsafe checkpoint: {checkpoint_path}") from exc
    return payload.get("best_epoch") or 0


def _score_candidate_lexicographically(
    *,
    metrics_by_experiment: dict[str, dict[str, float]],
) -> tuple[float, float, float, float, float, float, float, float, float, float, float]:
    """茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聫茅聢搂莽聞娄莽禄禄忙驴聥氓麓聮氓娄陇盲陆潞盲潞聶忙驴聽莽聰碌氓聛聻茅聧篓氓麓鹿氓篓虏莽聳戮忙驴聽茅聺聸茅聬聯茫聝娄莽鹿聺茅聴聢忙露聶莽聙鹿氓聭聤莽鹿聢茅聫聦忙露聵脦聰忙碌拢莽篓驴忙聭碌茅聴聛猫氓聳聴茅隆篓氓聭麓氓聼楼忙戮露氓篓戮忙聻聨忙驴隆氓聨搂忙驴庐忙麓垄禄?茅聴聜氓聜職氓聙赂茅聧聤茅拢聨忙氓搂聵猫路潞莽聞碌忙陇陇忙聨聭氓聙赂忙碌聽忙禄聢忙聼陇氓篓虏猫莽聛隆茅聳潞氓聭聢录茅隆垄忙露垄忙聜聶莽聙碌莽氓鹿聴茅聴聜盲陆潞莽虏炉茅聫聦茫聞楼莽聮潞猫隆隆忙聻聧氓漏碌莽聤虏忙麓娄茅聬聯忙聸垄忙聼聧莽聭聶氓聥聤陇氓篓麓忙禄聟氓聛聯忙路聡氓漏聡忙聜聶茅隆聫氓聥篓氓聛聯茅聫聧脩聦氓聲聟氓漏碌莽聤鲁猫聣戮莽潞戮氓忙聼聼茅聬聬忙職聵氓篓麓忙禄聞莽虏聯茅聫聦茫聞漏氓录聧楼脩聦忙拢聢茅聴聛茅聙聻氓卤陆氓垄聬聨莽卢聸氓聸搂脦娄茅聳赂忙聸篓氓聳職茅聬陇芒聜聴聛氓聹聛聯氓篓录忙卢聫芒聳聧茅聰聺氓聸漏莽拢陆氓篓麓茂陆聡茅隆拢忙聤陆忙聼聸莽聙拢盲禄卤忙楼聽莽聜虏莽聣聯忙驴隆忙聬聛氓娄路茅隆聰猫聴聣莽录聯茅聴聜盲陆潞莽隆露茅聧聯忙驴聥氓娄碌茅聬聬氓聯聞莽陋聴氓漏碌莽聤虏忙麓娄茅聢路忙聢娄忙聼聼茅聭虏芒聲聛盲禄聹茅聳鲁脩聦氓聣職氓篓虏莽聠录茫聙聛茅聫聧茫聞楼莽路聻茅聳鹿茅聜娄氓聣聸茅聬拢氓聯聞芒聰聭忙聨鲁茅聧聤忙聸垄氓鹿聤忙聬麓茫聞楼莽庐虏茅聬聽氓聯聞茫聜陆猫陇掳忙聨聯氓聛聬茅聬聽盲陆陆忙芦聲茅聧聽忙掳露忙垄潞茅聧聺氓聥聞驴莽录聛猫鹿颅莽虏聶茅聧漏茫聝娄氓戮聞忙驴聻莽聜卢莽聙禄茅聴聛莽聭掳茅聧聥忙露聵芒聳聧氓娄陇氓聠陋莽拢陆茅聳赂氓卤戮莽聭搂茅隆娄茂赂陆茫聛聞茅聨麓忙驴聡氓職聡茅聮聵忙驴聥忙聼聼氓篓聢氓聻聻聼莽聙鹿忙聸聻氓录露莽禄禄忙驴聥忙氓篓聭氓聧聻氓聣掳莽录聜盲陆虏氓聛聯茅聬聨茂陆聨氓聳聴茅聬聯忙露垄忙聼聸茅聢漏氓聠陋忙聡聝茅聳鲁猫炉虏莽聣聯茅聫聦芒聙鲁莽聲聺莽聙拢脨聹茅聴聛猫炉隆氓聳聴茅聬聼脩聡氓聼聙茅隆聮氓聜聸忙職漏莽禄戮忙鲁聸猫聻聻茅聳碌氓篓戮忙聶聹茅聢路忙聢娄忙聼聸氓篓聭忙漏聢忙聰禄茅聳禄忙聮卤氓聞碌氓搂聤猫聶鹿忙聲氓聣聹茅聴聛猫炉隆氓聳聲茅聧聴忙聤陆茫聙聛氓猫路潞莽聞碌忙陇陇忙聨聠氓聛聯茅聳禄茂陆聡氓聟聵氓庐聲氓楼赂氓录麓茅聬聬莽垄聣莽麓虏茅聴聜盲戮聙忙陆搂猫聣聦茅聳潞氓聭庐莽虏聯氓庐聲忙聞掳忙聜陆茅聧聸氓漏聞氓聫聠氓漏碌莽聤禄忙陆搂氓娄驴聥氓娄陇芒聜卢茅聹聣忙驴聽氓漏聜氓茅聬聨聴忙聼隆氓庐聙氓卢搂卢莽录聛莽聤鲁莽聸陋聻忙篓录芒聜卢忙露聶脡聭茅聧聤莽聝陆忙垄禄忙碌拢氓聯楼忙聻聝忙驴隆氓芦掳猫聴聺茅聳禄茫聢聽忙聭聹氓庐聯盲陆鹿盲驴聤茅隆聳忙驴聠氓聬鹿忙陇陇芒聲聜莽聟隆茅聫聦茂陆聣氓鹿聡茅聴聢忙露聵莽录聜?茅聴聜氓聜聶忙碌聡氓庐聲莽聰赂氓麓掳茅聨掳氓聻聺茅聨录忙路卢氓聻潞氓篓聟莽聤鲁盲驴聤茅聤聢氓聭庐氓聶潞茅聳赂氓卢聙驴茅聫聦茫聞漏忙聜垄茅聧聺氓聥聭茫聝搂芒聜卢猫搂聞忙聦路莽禄露忙掳露氓录禄茅聬聰氓聟录氓聙禄忙驴庐忙楼聙忙拢聶茅聬拢氓聣聛莽麓聯忙碌拢氓聯聞脨拢氓漏碌莽聜虏茂鹿氓漏職忙聜垄莽聬聸芒聙鲁盲潞戮茅聳禄茫聢隆茅聬聮茅鹿聝氓麓聨录盲陆鹿芒聮聮茅聳赂氓卤戮猫聣戮茅聢搂氓聟聵茅聨庐莽聜潞忙聲聝茅聧聦忙露聵氓聣鲁茅聬聼忙禄聟氓莽聰聨聰莽虏聢忙戮露氓卤赂氓聛聯茅聧聫氓盲潞戮茅聴聛茅聙聻氓卤陆氓垄麓茅聳赂芒聰戮氓聛聯氓娄聻茫聜聠氓赂聤氓篓麓氓聸篓莽芦聼氓搂聴芒聜卢茅聫聦忙露職盲禄娄莽聮聡忙聫聮茅聴聛氓聯聞莽聻录茅聧聽忙聽颅莽聧聧隆忙掳录忙搂禄氓娄聻茫聜聫莽碌陆莽潞戮忙聝聙茅隆聫氓聽聺莽庐職莽聭聹氓卢麓聭茅聤聫茫聜聠莽聟聸茅聬聫莽聜虏氓垄陆氓篓虏氓芒聜氓聳聴氓忙卢聮莽聛聝忙驴聻猫戮戮莽露聙茅隆忙卢垄茅氓聥聮聮?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聧茅聢搂莽庐聯氓庐聲氓楼赂氓茅聢楼氓聠虏莽掳楼茅聴聜盲陆赂忙鹿卤忙戮搂忙氓搂聴芒聜卢茅聨聺聴茅聢搂氓录禄茅聬聰氓掳聛忙聻聰茅聳赂氓聳聴茅聬聫氓聽楼忙垄潞茅聨录莽聜虏氓娄陆莽录聛猫炉虏氓茅聧漏芒聜卢忙陇陇忙聨聠氓聛聯莽录聛莽聤虏莽搂鹿氓庐聲忙聸垄忙聼聠氓庐聯茫聝楼忙麓颅茅隆垄忙露聵茅聧聸莽聭聺茅聴聜盲陆赂忙聬隆娄盲录麓氓篓聠忙聮麓氓聛聞茅聳赂芒聞聝莽篓聮茅聧聥茂陆聣忙聼搂猫鹿聡忙聸聼氓職聙茅聳禄茫聞娄氓聻驴茅聫聦忙露垄氓录庐茅聧聦忙掳颅莽聛聢茅聴聛氓聯聞莽聻录茅聧聽忙聽颅氓鹿聝茅聧聯脩聜忙虏聳录忙聛聮氓聞鲁忙碌聹忙垄禄忙碌拢氓聭聤忙聝聢茅聳潞氓聽聣聸莽禄卢聯氓录陆茅茫聜聞莽卢聣氓漏碌莽聜麓氓聻聧篓氓陆聮氓麓隆茅聨露猫聤楼莽聟聫氓漏碌莽聜聭氓聠篓忙職聶氓氓聸漏莽麓聯茅聧聦忙掳卢芒聜聬聛氓麓聬忙氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯茅聳禄忙露卤氓聶拢氓炉禄聥茅聰聺氓聭卢莽虏隆忙驴聽莽聰碌氓聛聸氓娄卢聯氓鹿聬忙驴聽忙掳聛聜忙驴聻忙聢聶氓聻潞茅聬聯忙聸垄忙聼聼茅聨碌猫聶芦忙芦聟氓漏碌茂录聞忙聶芦茅聢搂忙聡聯茅聨虏茂录聞忙聧垄茅聧聲茅聳鹿莽聝聵莽禄聽忙露聶茫聛聞茅聨戮猫路潞脨拢茅聳赂忙聨聯莽聸炉氓搂聤忙麓陋氓麓聧聲忙陆露茅聴聛氓聭聤忙聼篓茅聬聜聻氓麓潞茅聤聣茂鹿聙莽路聻氓漏碌莽聜聻禄茅聧聽忙聽颅氓鹿聤茅聫聧莽聟聸氓篓麓茂陆聠氓聤聟茅聬聬莽聰碌莽拢陆茅聳赂氓卤戮猫聣戮茅聢搂莽禄庐氓聽聼莽卢聼茅聢搂忙茅聴聛氓聯聞茅隆聭忙卢聫茫聙聮忙驴聽忙掳聟垄忙驴隆氓掳篓莽禄卤忙掳颅莽鹿聸茅聨麓忙卢聫莽聛漏莽录聛芒聜卢茅聧聬芒聰驴盲潞聹茅聳潞氓聠篓忙麓娄茅隆楼氓脙鹿氓漏聤氓聠陋莽搂潞忙驴庐氓漏聝忙陇陆氓庐聲莽聝陆茅聬聵猫聦卢茅聴聛忙卤聡氓聼聞茅聧篓氓录虏茅聬聵氓路職聲茅聳鹿茅聴麓莽陇聛茅聬聮氓聻聛聞茅聨潞忙聢聺茅聳鲁茅聰聥氓赂聮茅聹聣茅聳驴忙驴聠忙聡聫莽聮聬芒聞聝莽鹿聺茅聢搂莽莽聙拢氓聨卤茅聴聛氓聯聞氓聙陆氓篓聣忙聸垄忙聜聵茅聳露聠莽聟聝莽聭聹忙禄聢氓麓聹氓搂聵茅忙聲聥莽聭聹氓潞篓氓鹿聠茅聫聝氓聽職芒聜鹿聜氓鹿聥茅聬聬氓颅聵莽聫娄茅聴聜氓聜聶莽陇聛茅聧職氓卢忙碌聽氓莽拢驴茅聴聛莽搂麓莽禄聙氓陇聬忙聼聸氓篓聭忙忙聭聽茅聳鲁茅聰聥氓聻驴茅聫聦莽聠潞莽虏聶茅聧篓氓聤聣氓茫聜聝氓聞卤氓娄卢聭忙聧垄氓篓聭忙卢聫莽路聻茅聬聨茂陆聨氓聛聠茅隆娄盲录麓忙聜聴莽聭聶氓聥卢莽陇聝忙驴聽茫聢隆茂鹿垄茅聰聺莽聜虏茅聧驴聢氓聤娄氓茫聜聠氓赂聮莽聙職忙聥聭茅聬聰氓聯楼莽庐戮茅聳鹿氓炉赂氓聛聼茅聬虏聞忙聼拢茅聤聢氓聜職盲潞戮茅聴聜氓聜聶忙碌聡茅隆芦氓陇聤莽聲路氓娄隆隆忙聻聳忙驴聻忙聢聶氓聻鹿茅聬聮氓氓录露茅聧芦忙露聵氓莽录聛猫忙拢聺茅聫聦忙聸垄氓麓录氓漏碌氓聸篓忙聜聶茅聬聨忙露聶莽鹿聜茅聢鹿忙聢娄忙聜聶茅聭录氓聥戮茅聴聛氓聭聤忙垄鹿茅聧篓氓聺聴忙聜掳茅芒聜盲禄聥茅聬聰氓掳聛氓聛聯茅聳驴氓盲潞聹茅聳鹿莽聝聵氓聻碌茅聢搂茅聨炉盲戮聙氓麓录茅聤聣茂赂鹿忙聥潞茅聴聛莽氓聛聬茅聧聥忙露垄氓麓聺氓娄陇氓聭庐莽聟聸茅聳赂忙露卤氓聳職氓篓虏忙聮庐忙聼聼氓庐聞忙聣茅聳鲁脩聦氓聣篓莽录聧氓聴聸氓麓掳氓娄陇氓聭聟聲?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    metrics_by_experiment : dict[str, dict[str, float]]
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬氓聫聣莽聳聞茅聴聜盲陆赂忙聠隆茅聨赂氓聴聸氓聼聙茅隆聮氓聥芦莽拢禄茅聳鹿莽聜聣職氓篓录芒聲聜茫聛聞茅聨戮猫路潞盲禄聮茅聳禄茫聝庐盲潞莽禄聵忙聥聟茅聧聸氓漏聜忙職聢氓漏碌莽聜虏氓录露茅聰聲茫聢陇茫聙聛氓篓聠忙聮鲁莽卤聧茅聳赂氓聳聯茅聧?茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙碌拢茅聺漏猫聙驴茅聴聛莽搂麓茅聧聦茫聞搂忙聠赂茅聫聝氓聽聺莽庐聳忙驴聻氓路潞聹忙聝聼茅聴聛茅聺聸茅聧聽忙楼聝莽麓聻忙聬麓茫聞娄莽聟聼茅聳禄忙篓潞氓录露忙戮露氓聥颅莽麓聮忙碌拢猫搂聞氓掳娄莽聙碌忙聜聜氓炉麓录茅聬聰氓聯聞氓鹿聧茅聴聛猫炉虏氓莽禄聥忙聨聴芒聙娄茅聧楼脩聛芒聜聨聴氓娄露茅聧楼茫聞楼忙聻聭茅聴聛氓聯聞氓聙陆氓篓聣忙聸垄氓录鲁茅聰聲氓聜聸莽聟聲忙陇陇忙聞露莽碌戮莽禄聙芒聜聳鹿氓聳聳氓忙麓氓篓聭忙掳卤忙驴聽忙氓聳聸氓篓聟忙聸聽莽虏漏茅聬聰莽聟聨氓聧鲁茅聳禄忙篓录忙聲赂茅聰聸忙驴聥忙垄潞莽禄聥忙聦聨忙鹿掳茅聳禄莽聠聺氓聸搂莽莽聙鹿茫聝楼茅聬聯茫聝娄忙聟篓氓娄炉氓聻聟聬茅聳禄忙聮露氓聳聳茅聫聦茫聞漏氓录麓氓娄陇芒聜卢忙碌聹氓聣搂莽麓聯茅聧聦忙掳卤脨垄茅聳驴忙聸篓盲潞聹茅隆聲茂陆聠莽鹿聺氓盲录聥茅聴聛氓陆聮莽禄聙盲戮聙氓鹿聝茅聨麓茂赂陆芒聮聭茅聳赂忙露聵茂鹿拢莽禄露茅聛聧忙聼聸茅聬聵氓聠虏氓聯潞氓漏碌芒聙鲁莽聢录茅隆垄忙聸垄氓娄露茅聧楼芒聲聥莽麓鲁氓漏碌莽聜露忙聫禄虏猫聜漏氓聢聮氓篓聠忙聮鲁盲录聽茅聳戮氓驴聯忙鹿卤莽潞戮氓楼赂忙聜鹿茅聧楼脩聝莽潞搂忙陇陇茫聝搂氓搂路莽禄卤忙聨聴氓篓聠忙聝搂忙庐聴聛氓庐聽忙茅隆娄氓聻芦忙聟篓茅聢搂忙聼拢氓娄炉氓聻職潞茅聳禄忙露聮忙聶聹莽禄禄忙驴聥忙聜陆茅聴聢忙露聵氓拢聥莽录聜盲陆陆氓掳陋忙驴聜聡莽虏聥氓氓聴聬氓聛聟茅聳赂忙聞碌盲潞聹猫陇掳氓聥垄潞茅聨录莽聜虏氓聤聵茅聳赂忙聳驴茅聟拢茅聧漏茫聝娄氓戮聞氓庐聙芒聜潞戮氓楼赂忙聼聧猫陇聹茅聧聯忙卢聯忙聜戮氓漏碌氓莽陇聥忙陇陇忙聨聭氓聙赂茅陋聻氓聽聼莽鹿聺茅聬垄茅聺聸盲禄娄茅聳赂茫聞漏忙陆搂茅聬拢莽聝陆茅聧聲莽楼娄氓漏碌脗掳氓聙聲茅聨鲁氓驴聸忙聜隆?        ``e9_dual_degradation``茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮莽聲聦忙聠赂氓庐楼氓聽垄盲潞卤茅聴聜盲陆赂忙聬芦茅聧聤氓陆聮氓篓聠氓驴拢莽聞陆茅聳潞氓露聝氓聤聧茅聧聶氓驴聯芒聰聭茅聬聵忙聫聮芒聜卢氓漏職氓聼聻?_async``茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮莽聲聦忙聠赂氓庐楼氓聽垄盲潞卤茅聴聜盲陆赂忙聬芦茅聧聤氓陆聮氓篓聠氓驴拢莽聞陆茅聳潞氓露聝氓聤聧茅聧聶氓驴聯芒聰聭茅聬聵忙聫聮芒聜卢氓漏職氓聼聻?_nlos``茅聴?
        ``e0_safe_mode`` 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓聙驴茅聳驴忙聸聻氓聙赂莽禄聽忙聽聼聸茅隆聬茂陆聠莽陇聙莽禄戮忙聝搂忙陆隆茅聫聦茫聜聡氓聳聤茅聧聴莽聜虏茅聴聛忙麓禄氓聨陆茅聬聼芒聲聟氓录禄茅聬聰莽聟聨莽庐楼忙陇陇忙聴聜芒聮聢茅聫聠忙垄潞莽录聛忙篓禄氓掳聳赂氓漏聜忙鹿卤茅聨鹿茫聢聽芒聰聭茅聧楼芒聲聝莽聵聢茅聴聛莽忙聹禄庐氓虏聞脗路茅聢搂忙垄?茅聴?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    tuple[float, ...]
        11 莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳鹿茅聴麓莽莽潞戮氓陆聮忙聼拢茅聨麓茂陆聟脨聯茅聳赂茫聜聟氓茅聫聦忙露垄茅聰聺氓聴聶茫聙聫忙驴聽茫聢拢氓聼聳茅聧聰忙聸聽莽虏禄氓篓聭茫聢隆莽聟聼忙驴隆茫聜聜氓聛聯茅聳禄忙楼聙忙拢聺氓庐聲忙禄聢氓娄赂茅聢潞氓聜聹氓聨碌茅聴聛莽篓驴莽鹿聴茅聧聥忙聞掳氓录鲁氓芒聜聫聦忙露垄氓录卢莽聮潞茫聞搂盲录聡莽录聜盲戮聡氓聳聳茅聬聴忙聮鲁莽聲路茅聧聴莽聜漏忙忙驴聻氓聴聵氓聻碌茅陋聻氓聽楼忙垄禄忙碌拢猫莽聛卤茅聳潞氓聭聸炉氓庐聲氓茅聫聣忙聢聻莽聙茫聜聠忙聼篓氓娄虏氓聽楼氓录聧芦忙驴聢氓茫聜聟莽鹿聬氓娄忙搂聬茅聧聬忙碌聥忙聰陆茅聳禄忙聞炉氓聼聳猫陇掳脩聟莽麓聮茅聨禄忙聨聯氓聛楼茅聳鲁茫聝炉莽庐聯氓庐聲氓楼赂氓茅聤聫莽麓鲁氓漏碌莽聜露忙聫禄虏猫聜漏氓聢聮氓篓聠忙聮鲁盲录聽茅聳戮氓驴聯忙鹿卤莽潞戮氓楼赂忙聜鹿茅聧楼茫聞楼莽碌禄茅聳潞氓聴聶氓聳聯莽禄卤忙聨聴茅聧聸莽卤聧茅聴聛猫炉隆氓聳聯茅聧篓莽禄聵莽娄聧茅聧聸氓漏聝氓茅聬聼忙卢聫氓芦?氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄茅聳禄忙聞聬芦茅聧篓氓聜聻忙聼聸茅茂陆聠莽莽录聛莽聤卤氓聧鲁茅聫聦忙露垄氓鹿聵氓娄陇芒聜卢茅聧聤氓氓麓聹茅聤聤芒聲聜芒聮聭茅聴聜氓聽聼盲赂鹿氓篓聭茫聢聽莽茅聳赂氓聛聠莽禄?
        茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼忙聲聥莽聭聹氓露聣忙鹿聛茅聴聛莽禄聵氓聻录氓娄芦氓聥虏禄茅聬聵莽禄聵莽聟聶茅聳鹿氓聠漏茅聴聛忙聬聻氓聞聹茅聧聸茅聴聜盲陆鹿氓莽禄禄氓聜聻忙聜搂猫鹿聡忙露聶氓聛陋茅聳鲁脩聦氓聣聼氓搂聤忙麓陋氓麓聧聲忙庐聹茅聴聛莽篓驴茅聨赂忙聬聛忙鹿聛氓漏碌莽聤聶氓聠禄垄茅聳鲁脩聦氓聣職茅聬聴忙禄聢氓鹿?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜麓莽聣聽茅隆垄忙聸垄氓娄露茅聧楼芒聲聝氓聨聽茅聴聜盲陆赂忙聬芦茅隆娄盲录麓氓麓碌氓忙麓聧聲氓鹿聸篓莽聟聲茅聳碌氓陇聸莽碌陆忙驴隆氓楼赂氓鹿聳茅聫聣忙聢潞卢忙驴漏聝忙陇陆氓庐聲氓麓聼茅茂陆聠茅聴聜盲陆潞莽虏炉茅隆篓氓聭颅脦聸氓漏聤氓聭庐氓麓虏忙驴庐忙漏聠氓聟聵茅聫聙盲禄聥忙聼拢茅聨掳莽聟聬莽聭聹忙聤聤氓聭拢录茅聴聤氓聨戮茅聬卢聬茅聬聬忙聬颅忙庐聰忙陇陇氓聤聭氓庐聲莽聟聨芒聰聭茅聧楼氓聫聭茅聴聜氓聜聶莽陇聛茅聨录氓麓茂赂陆氓聛聥氓漏碌莽聤虏忙麓娄茅聧聥氓聜聻忙聼隆茅聧楼茫聝楼莽聛聳赂忙卢聬忙聧聡茅聫聦忙露垄氓娄路茅聨麓忙驴聤氓茅聳赂忙聮露氓垄聴莽禄禄忙驴聠芒聳聯茅聧篓莽聛聙茅聴聛莽茅聨赂氓聣搂忙搂聬茅聨戮猫炉虏莽拢录忙驴聻氓聴聵氓赂聮茅聧聵茂录聞莽麓聯忙赂職氓聸搂忙芦聵茅聳赂茫聞陇氓聧鲁茅聰聺莽聜虏芒聰聭莽聙拢芒聲聟忙聼聧忙聺聻忙聥聦氓聟聦茅聳禄茫聜聟忙聡聵氓忙麓麓路茅隆聯芒聞聝茅聴聛氓聯楼氓搂碌茅聬聴忙禄聟氓聢聮茅聧聭漏氓聛聯茅聰聺氓潞隆盲潞聳氓篓麓忙禄聞莽虏聯茅聫聦芒聙鲁茅聧聬茫聞楼氓茫聜聫莽碌戮茅聬聯芒聙鲁忙驴聽忙掳拢录忙驴聻氓聴聵氓聻鹿茅聬聸茫聢聽芒聰聭茅聬聬忙聺驴氓掳聝茅聳赂氓驴聲氓聛聯茅聬漏茅芒聜庐聲忙聨聭茅聧聸氓搂赂忙驴聽莽聰碌氓聛聽茅聨禄氓戮聤忙碌聽氓莽庐聼茅聳驴忙露聵氓氓庐聲氓聫聣莽鹿聺茅聴聢忙露聶茂录聧忙驴聻氓掳聳忙聶聹茅聤聡氓聥鹿聵莽聮潞莽聸聮茫聢陇忙聜陆茅隆聳忙露聶氓麓聦茅聳潞氓虏聥莽碌戮茅聨聴聢忙聲聧茅聧聲忙庐聴聜盲陆潞茅聳掳茂陆聠莽茅聧聸聻猫聯聼忙驴聻忙聢聻氓聼聝茅聧聥忙聮禄忙聲聬忙聬麓芒聙虏莽掳录茅聬聨猫忙麓聳茅聬颅氓聜聻氓录禄氓篓聭茫聢聽莽聟聵茅聳鹿氓聜職忙驴聽莽垄聣莽麓聲茅聧聥忙聢娄氓麓聬茅聫聧忙職聹茅聳鹿莽聝聵忙聼隆茅聧聥忙聮鲁茅聬聯氓潞隆莽录?hard_p95茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮莽聲聦忙聠赂氓庐楼氓聽垄盲潞卤茅聴聜盲陆赂忙聬芦茅聧聼氓麓聬忙驴聽忙掳颅氓聞聴茅聳赂芒聞聝猫陇掳忙聨聯忙聶虏茅聳赂忙露卤氓聳聳茅聫聠氓聽垄莽麓聯忙碌拢猫聶鹿氓赂聸莽录聛猫炉虏氓聙驴茅聰聺莽聜虏芒聰聭莽聙拢氓聻芦忙聶拢茅聴聛莽禄聵脡聭猫陇掳氓聠虏颅茅聨戮莽禄聵芒聮聭茅聬聽氓聸篓莽聟隆茅隆聨忙莽麓聮茅聬聵猫聦搂莽聙鹿忙聸聼忙麓聼茅聫聦氓聴聴茅聧隆忙卢聫氓鹿聢茅聴聛猫氓聡陆莽录聧氓聴聸氓麓聭茅聧聸忙職娄莽聙鹿芒聜卢茅聧聲氓聨卤茅聴聛氓庐聽茅聧聨猫聶芦莽娄聧茅聬聬莽鹿聠茅聳禄忙聞碌盲潞聹茅聢搂莽聣聹茅聫聛氓鹿聮茅聫聜氓聯聞盲潞戮忙驴庐忙篓录氓聨搂忙戮搂忙聣庐莽麓?hard_failure_rate茅聴?
        NLOS hard_p95茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮莽聲聦忙聠赂氓庐楼氓聽垄盲潞卤茅聴聜盲陆赂忙聬芦茅聧聼氓麓聬忙驴聽忙掳颅氓聞聴茅聳赂芒聞聝猫陇掳忙聨聯忙聶虏茅聳赂茫聝聭氓聠篓忙驴隆莽聜聙陇茅聧聥氓聭潞茅聬聰莽聟聨猫聯聼茅聳禄忙聳驴氓聬聥茅聧聤茅聰聥茫聛聞茅聨潞氓聴聴盲潞戮忙驴聽芒聮聙氓卤陆莽聛隆莽录?hard_p95茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮莽聲聦忙聠赂氓庐楼氓聽垄盲潞卤茅聴聜盲陆赂忙聬芦茅聧聼氓麓聬忙驴聽忙掳颅氓聞聴茅聳赂芒聞聝猫陇掳忙聨聯忙聶虏茅聳赂忙露卤氓聳聳茅聫聠氓聽垄莽麓聯忙碌拢猫聶鹿氓赂聸莽录聛猫炉虏氓聙驴茅聰聺莽聜虏芒聰聭莽聙拢氓聻芦忙聶拢茅聴聛莽禄聵脡聭猫陇掳氓聠虏颅茅聨戮莽禄聵芒聮聭茅聬聽氓聸篓莽聟隆茅隆聨忙莽麓聮茅聬聵猫聦搂莽聙鹿忙聸聼忙麓聼茅聫聦氓聴聴茅聧隆忙卢聫氓鹿聢茅聴聛猫氓聡陆莽录聧氓聴聸氓麓聭茅聧聸忙職娄莽聙鹿芒聜卢茅聧聲氓聨卤茅聴聛氓庐聽茅聧聨猫聶芦莽娄聧茅聬聬莽鹿聠茅聳禄忙聞碌盲潞聹茅聢搂莽聣聹茅聫聛氓鹿聮茅聫聜氓聯聞盲潞戮忙驴庐忙篓录氓聨搂忙戮搂忙聣庐莽麓?rmse茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮莽聲聦忙聠赂氓庐楼氓聽垄盲潞卤茅聴聜盲陆赂忙聬芦茅聧聼氓麓聬忙驴聽忙掳颅氓聞聴茅聳赂芒聞聝猫陇掳忙聨聯忙聶虏茅聳赂茫聝聭氓聠篓忙驴隆莽聜聙陇茅聧聥氓聭潞茅聬聰莽聟聨猫聯聼茅聳禄忙聳驴氓聬聥茅聧聤茅聰聥茫聛聞茅聨潞氓聴聴盲潞戮忙驴聽芒聮聙氓卤陆莽聛隆莽录?p95茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮莽聲聦忙聠赂氓庐楼氓聽垄盲潞卤茅聴聜盲陆赂忙聬芦茅聧聼忙聜搂忙驴聽氓聸搂氓聬鹿茅聳赂氓聞虏茅聬聯忙卢聬茫聛聞茅聨潞忙聢颅忙搂鹿茅聳潞氓聴聬氓陆聝 p95茅聴?
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓氓聽聺莽拢驴忙驴隆茫聜聟忙聡聫茅聬聯忙聸聽芒聜卢氓聟聣氓聛聯茅聳鲁脩聦氓聣聼氓庐聲忙聢聺芒聵聣茅聤聫芒聶聙忙聥禄氓茫聜聠莽聣聹茅聧聥莽聜潞忙聜隆氓篓聠忙聮麓芒聮聮茅聳赂氓卤戮氓聡鹿茅聧陇莽聠录忙聼聸茅聫聜猫路篓脨搂茅聳潞氓虏聥莽露聛茅陋聻忙聨聴氓鹿聥茅隆聯莽聤虏忙聜聭茅聴聜盲陆赂忙聬芦茅聭禄莽虏聰茅聬聭茅聯掳茫聞娄莽卢聼茅聢搂氓录禄氓篓聭茫聢聽莽庐禄茅聬聽猫聶鹿氓聞陇茅聬聨氓忙垄?mae茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮莽聲聦忙聠赂氓庐楼氓聽垄盲潞卤茅聴聜盲陆赂忙聬芦茅聧聼氓麓聬忙驴聽忙掳颅氓聞聴茅聳赂芒聞聝茫聛聹茅聢搂氓赂聮茅隆驴聥忙聲聬茅聧聸茅聴聜氓聜聶忙碌聡茅忙聽麓鲁茅聰聺氓聸漏盲赂聧茅聳潞氓露聝氓聤聥莽禄禄氓聴聸忙聼聼茅聴聜氓炉赂茅聳禄忙聮麓莽篓聭茅聹聣茅聳驴忙驴聠氓茅隆聮莽聠赂盲驴聤氓莽聛娄忙驴庐氓漏聞莽虏炉莽禄聴莽聠录氓聼聙茅隆聮芒聜卢茅隆聸陋茅聳赂茂陆聟莽露聤氓忙麓麓篓忙驴聽氓聠篓茅聴聛氓聭聤脙陋莽录聧氓漏聜莽聲路茅聨麓忙驴聝忙茅聳碌氓驴聤忙陋搂茅聨路忙麓聼拢茅聬聵氓聫聣忙聬聬莽聙碌莽聢录茅陋聻忙聢娄忙聲聬忙戮露氓卢聨赂忙驴聻猫戮戮莽露聙茅隆氓麓聬茅聨掳忙庐陆茅聳禄忙聞聼卢茅聬聬莽聳碌芒聙娄茅聧搂忙楼聙莽陇聥忙陇陇忙聮露氓聛聻茅聬聨氓聣搂莽麓聯忙碌拢猫聶鹿氓赂聸莽录聛猫莽聣聠茅聬拢莽聝陆忙聜垄莽潞掳猫戮篓忙聶垄茅聴聛茅聙聻氓卤陆氓垄掳氓猫搂聞忙麓聼茅隆垄忙露聵茅聰聺氓聴聵忙聺赂茅聴聜盲陆潞莽虏炉茅隆聸漏氓聢聮茅忙掳拢忙莽聭聹忙聴聢氓鹿聝氓娄陇芒聜隆芦忙驴聥氓聼聙茅隆聮氓聥芦莽聜聤莽聭聹氓驴聸忙聲聧?
    """
    dual_metrics = metrics_by_experiment["e9_dual_degradation"]  
    if _resolve_paper_scope(_ACTIVE_PAPER_SCOPE) == "sim_e9_only":
        return (
            float(dual_metrics["hard_p95"]),
            float(dual_metrics["hard_failure_rate"]),
            float(dual_metrics["rmse"]),
            float(dual_metrics["p95"]),
            float(dual_metrics["mae"]),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    async_metrics = metrics_by_experiment["e2_async"]  
    nlos_metrics = metrics_by_experiment["e3_nlos"]  
    safe_metrics = metrics_by_experiment["e0_safe_mode"]  
    safe_reference = max(float(safe_metrics["ekf_p95"]), 1e-9)  
    safe_relative_margin = (float(safe_metrics["p95"]) - float(safe_metrics["ekf_p95"])) / safe_reference  
    safe_violation = max(0.0, safe_relative_margin - _SAFE_P95_RELATIVE_TOLERANCE)  
    return (  
        1.0 if safe_violation > 0.0 else 0.0,  
        safe_violation,  
        float(dual_metrics["hard_p95"]),  
        float(dual_metrics["hard_failure_rate"]),  
        float(nlos_metrics["hard_p95"]),  
        float(async_metrics["hard_p95"]),  
        float(dual_metrics["rmse"]),  
        float(async_metrics["p95"]),  
        float(nlos_metrics["p95"]),  
        float(dual_metrics["mae"]),  
        safe_relative_margin,  
    )  


def _resolve_conclusion_priority_metrics() -> list[str]:
    """Resolve the protocol-defined conclusion priority metric order."""
    protocol_cfg = load_experiment_protocol()
    raw_priority = list(protocol_cfg.get("conclusion_priority") or [])
    priority_metrics = [str(metric_name).strip() for metric_name in raw_priority if str(metric_name).strip()]
    if not priority_metrics:
        raise ValueError("experiment protocol must declare a non-empty conclusion_priority")
    return priority_metrics


def _summarize_held_out_method_metrics(
    metrics_by_experiment: dict[str, dict[str, float]],
) -> dict[str, float | int | None]:
    """Summarize held-out metrics for the final report."""
    summary: dict[str, float | int | None] = {
        "num_experiments": len(metrics_by_experiment),
    }
    for metric_name in ("p95", "failure_rate", "rmse", "mae"):
        values: list[float] = []
        for metrics in metrics_by_experiment.values():
            value = metrics.get(metric_name)
            if isinstance(value, (int, float)):
                numeric_value = float(value)
                if math.isfinite(numeric_value):
                    values.append(numeric_value)
        summary[f"mean_{metric_name}"] = (
            float(math.fsum(values) / len(values))  
            if values
            else None
        )
    return summary


def _build_held_out_protocol_rank_vector(
    method_summary: dict[str, float | int | None],
    *,
    priority_metrics: list[str],
) -> tuple[float, ...]:
    """Build a sortable rank key for held-out comparison rows."""
    rank_values: list[float] = []
    for metric_name in priority_metrics:
        value = method_summary.get(f"mean_{metric_name}")
        rank_values.append(
            float(value)
            if isinstance(value, (int, float)) and math.isfinite(float(value))
            else math.inf
        )
    rmse_value = method_summary.get("mean_rmse")
    rank_values.append(
        float(rmse_value)
        if isinstance(rmse_value, (int, float)) and math.isfinite(float(rmse_value))
        else math.inf
    )
    return tuple(rank_values)


def _search_scoring_config_names(method_name: str, *, include_safe: bool = True) -> tuple[str, ...]:
    """茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿氓聥聰聭茅聬聬氓聫聣芒聳聲氓篓麓忙禄聞莽虏聯氓庐聲忙漏聙氓聼聙茅隆聮氓聜聹莽拢陆氓篓麓茂陆聟忙聬芦茅聻聥氓茫聜聨氓聛聞茅隆娄茅聛聧氓鹿聭茅聤聫莽聤聰陆茅聳赂茫聝垄忙聶聺茅聬娄氓聽楼忙垄潞莽禄聥忙聦聨忙鹿掳莽录聛氓卢聻碌茅聢禄氓露聣忙聲聬忙戮露氓卢搂忙聥潞莽录聜盲陆赂茅隆聭忙卢聯氓麓聲猫鹿聡忙聸聽莽拢录氓漏垄猫路潞茂鹿娄莽禄聣莽聝聵茅聬聬忙聫聮忙職垄茅聳碌氓聸篓氓聤聶茅陋聻忙聨聴氓鹿聥茅聬聵忙碌聥氓聛聯茅聬聬莽聝聵氓聛隆忙驴聽氓漏聜氓聲掳忙聲聢氓漏碌茂鹿楼氓娄聻盲禄聥莽聧聫聧氓聙路茅聳鹿莽禄聵氓赂漏茅聬聨忙聤陆忙垄禄茅聧聦忙卢聭茅聨录氓颅聵莽虏聨氓炉掳氓漏聝莽娄聮莽聙拢氓聻芦忙聶聴聛忙聦聨莽鹿聜茅隆娄氓聧聻莽聲?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓聙驴茅聳驴忙聸聴氓聛聯忙陇陇氓聲麓茅陋聻忙聞颅忙聝聰茅聰聺氓潞垄茅聰陇忙驴聽莽聰碌氓搂麓茅聰聲茫聜聣忙聜搂氓漏聤氓聽聺忙聻聛茅聳赂忙聸篓氓聻卤茅聬聯忙聸垄忙聼聲忙戮露氓卢聛聧茅聳赂氓忙聧聡茅聫聦茂录聞芒聜卢茂陆聨氓聛聟茅隆楼氓聽楼忙聼隆氓庐聙芒聜聧聽忙聮禄氓聼聙茅隆聮氓聜聸忙職漏忙陇陇莽聣聬茅聢陋忙聮垄禄忙碌拢盲戮炉莽聞戮忙陇陇忙聢聺忙茅陋聻忙聢娄氓麓露猫茅聫聠茂鹿聜忙垄潞茅聧篓氓聞聭忙陇陇芒聲聝氓聟聵氓搂聤忙麓陋茅聢搂莽虏聰茅聬垄氓聭聤氓聛聜茅聳禄忙聳驴忙聭聹莽禄聽茅鹿聝忙聼聼莽聙碌莽盲禄搂茅聳鹿氓聠虏忙聡聵茅聫聦莽聠录氓麓聵茅聧聶氓陇聤忙聻聽茅聴聛氓聯聞莽聼聣莽禄卤忙聸聼莽茅聰聲氓聜職忙拢聵茅聳碌氓聽聺忙聡聧莽禄卤茂陆聣忙垄禄茅聧聦忙卢聭氓聤聧茅聳鹿莽聢录氓庐聲忙聸垄茅聢搂茅聨禄忙聨聺莽聟楼茅聳赂氓聳聨茅聢搂莽聛職茅聤聡氓聥炉氓鹿聵茅聧聴莽聜碌盲禄戮茅聴聛忙聤陆忙聰卤茅聧篓氓聺聴氓录禄氓篓聭忙篓路忙搂聢忙驴庐忙楼聙莽聣聯莽禄露氓颅聵莽鹿聺茅聤聫茂陆聨忙陆聬忙陇陇忙麓篓氓娄虏忙聞掳氓鹿聮茅聨戮氓麓聡莽陋聻茅聳禄氓潞露聠茅聧聥氓聥炉氓麓戮露忙聞聙碌茅聬聼忙卢聫氓莽禄聙芒聜录聜氓聜職莽搂麓茅聰聲氓娄麓忙赂職芒聜卢氓炉麓录氓漏碌氓聴職芒聜聻庐芒聰聭茅聢陆氓聴聴莽聛楼忙驴隆忙陇录氓聤聨茅聧聮忙聨聲忙驴庐氓漏聞莽虏炉莽禄聴莽聠录氓聼聙茅隆聮芒聜卢茅隆聸陋茅聳鲁茫聝炉莽庐聯茅隆垄忙漏聙忙聜漏茅聭录莽聯聵忙驴聽莽聰碌氓聛聸氓娄漏聝莽娄聧氓氓聭?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    method_name : str
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜麓莽聣聽茅隆垄忙聸垄忙聲聝茅聢搂氓茅聬聼忙聡聤氓陆聮忙聜漏氓庐赂氓聤聙莽录聜氓聜職莽搂碌茅聬聴忙聸垄芒聙娄莽聮潞茫聞娄氓麓聣茅聳禄忙聢聻脡聭茅聨路茅聛聧忙垄潞莽禄聥猫路篓莽莽录聛忙聧聡氓炉芒聵聣茅聤聫芒聲聜忙聶聺茅聴聛忙聦聨莽鹿聜氓娄芦忙露垄脣聡茅聤聣芒聲聜莽聟聼茅聳禄忙聳驴忙聭聹茅聨鲁忙聸聻芒聰聭茅聢楼猫聶芦氓聳聵茅聳赂芒聰戮氓聛聯氓娄聻茫聜聠氓路禄茅聧聥忙聮露莽麓聮茅聬聵猫聦卢脨搂莽聙鹿忙聸聼忙麓聵氓氓驴聸氓聼隆茅聧聦忙掳卢莽掳楼茅聴聜盲陆赂忙鹿卤茅聧聨茫聜聟忙驴聽忙掳颅氓聞聴茅聳赂芒聞聝莽篓聮茅聬聯忙聸垄忙聼隆茅聧楼茫聝楼氓忙楼聽莽聜麓茅聫聦忙露垄忙聜漏茅聧聬忙聫聮茅聴聛氓聯聞莽聼聣莽禄虏氓聙聼莽聛聝茅聴聛茅聙聻氓卤陆氓垄麓茅聳鹿氓聥炉聙茅聳赂忙聸篓盲潞聻茅聧聯茫聞楼茅聤聢氓聴聵氓掳碌茅聳赂氓潞垄氓聤聶茅聨庐茫聢陇氓麓卤氓篓聠忙聞卢忙聻聭茅聴聤忙麓娄氓猫聧陇莽虏禄忙楼拢氓聠漏莽聟聝莽聭聹忙禄聢氓麓聹氓篓聭茫聢聽莽聞?茅聴聜氓聜聶忙碌聡茅脩聡忙聼聤茅聰聺氓聮聛莽聞陆莽聭聹氓潞篓氓录虏茅聧芦忙聸篓忙聲聧忙驴聽氓漏聜忙聡聬茅聰聸忙禄聢忙聼拢忙聬麓莽搂碌氓篓麓忙禄聠忙鲁聸氓聧卤莽聭聹莽聣聢氓赂聴茅聬聯忙卢聬茫聛聞茅聫聧茫聞搂芒聤聲忙陇陇茫聝楼莽聣聠茅聢鹿忙聢娄氓聼聞茅聧聬芒聲聛芒聜卢忙聞聰聭茅隆聰莽聭掳盲潞戮茅聴聜盲陆潞莽虏炉茅聰職莽聙鹿忙聸篓氓聣聼猫聴聼忙驴氓聼聳茅聢路忙聨聭脙鹿茅聰聺氓聭聲聻茅聳鹿莽聣聯茅聨茅聬聯氓潞隆茅聴聛莽聭掳莽庐聧茅聧篓盲禄聥莽聧庐茅聨潞忙莽卤聧茅聳鲁脩聦氓聣聼氓氓聠虏茅聧芦莽聠潞氓聨卤茅聴聛忙聳楼忙聶聸茅聧聼盲录麓氓聼聤茅聫聡茫聢隆莽聟聼茅聳鹿忙聝搂氓麓卢茅聧聰忙禄聟莽麓聮莽录聛忙篓录忙麓麓忙楼聽莽聜虏茅聨篓潞脙潞莽聮聛氓麓聡忙聲聳莽录聜盲陆陆茅虏赂茅聨赂氓氓鹿聫茅聧聸氓颅聵忙聝聻茅聳禄莽聠赂氓拢聠茅氓楼赂忙聼拢忙聬麓芒聙鲁氓赂篓茅聳赂氓忙聧聡茅聫聦莽聠录忙垄禄莽聙碌氓聣虏忙搂录聜盲陆潞氓搂碌茅聬聴莽聤禄氓录禄茅聬聰莽垄卤忙聲聧茅聳赂芒聙鲁莽鹿聬忙碌聹茅鹿聝忙垄潞茅聧聺氓聥碌氓聞聫茅聴聛氓赂庐莽露聛氓氓漏職忙聜垄莽禄聥忙氓職陇茅聳禄氓潞垄莽篓聮猫聫陇茅聳鹿茅聰聥忙陇陆氓搂聤忙麓陋氓麓篓忙驴聽氓聥篓忙茅聴聛忙聦聨忙麓漏莽禄聽忙聮禄莽聧庐忙驴聽氓聸搂氓聧鲁茅聳禄忙篓录忙聲赂茅聰聸忙禄聢忙垄潞茅聭鹿忙聺驴忙驴聻氓陇聥莽篓聮莽禄聜氓露聣氓麓路茅莽聤碌茂录聹氓娄聻茫聜聠忙拢聛茅隆芦氓陇聤芒聳聧忙驴聽氓聠戮忙庐陆茅聳禄忙聞聫陋茅聭掳猫路篓芒聜忙麓聳茅聬聫忙掳卤忙聧垄氓漏碌氓莽聜聤茅聳碌氓篓驴氓聞庐茅聧聥忙聮禄氓麓鹿茅茫聜聡茅聴聛莽聭掳茅聧聨忙聞茅聧聲氓聫聫忙驴隆氓莽潞聣搂莽聴聨炉脩聦莽聟录茅聳潞氓虏聙氓聳聳茅陋聻氓聴職茅聬聽忙聞露莽麓聲莽禄卤忙聨聯忙陆聫茅聤聤茫聝搂莽聭虏氓漏碌莽聜碌氓聨搂莽禄禄忙篓禄盲驴聤茅聨录盲陆赂脦篓茅聳碌氓茅聶聞茅聧陇氓聽聼莽鹿聺?
    include_safe : bool, optional
        茅聴?茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙碌拢茅聺漏猫聙驴茅聫聣氓聽職忙聶聺茅聬聹氓忙聼隆茅聧聬茫聞楼芒聜卢莽聰赂氓麓聭茅聨戮猫聢碌氓聳聠茅聳赂忙聸篓氓聣聸茅隆娄莽鹿聺茅聬垄茅聺聸盲潞鹿茅聳赂氓卢聧聡茅聨忙聠麓茅聧聲茂录聵茅聴聛氓聭聤忙垄鹿茅聧篓氓聺聴氓娄麓忙碌拢氓聣虏忙陆篓茅聳鲁脩聦氓聣聼茅陋聻氓聠庐氓搂聙茅聬聵氓聻芦忙庐聺茅聴聛氓聯聞茅忙卢聫脦聸茅聨掳莽聣聢芒聮聮茅聳赂氓卤戮莽聭娄莽禄聛莽聣聢茫聛聞茅聫聡氓聴聴氓聨戮猫陇掳忙聨聯氓聙禄茅聳陆忙篓潞茅聬陇氓聸垄潞莽聭聶氓聥芦氓聤陇茅聳赂茫聜聝氓聻陆氓庐聲忙聝掳茅聳潞氓聴颅莽庐聯茅聫聦茫聞楼芒聰聭茅聧隆忙卢聫芒聟卤茅聴聛忙卤聡氓聙聬茅聧聥忙聮禄忙垄禄忙碌拢氓聭聤猫麓隆茅聳赂氓卢聧聡氓庐聲忙驴聠脦聰茅聧聸莽聧楼茅聴聛氓聯聞莽篓聛茅聧聵氓楼赂氓麓鹿茅聧聦莽聜麓莽聟聶忙陇陇忙聽搂忙茅聨鲁氓驴聯芒聳聯忙楼拢氓聠漏芒聮聭茅聳鹿猫聜漏氓聛聸茅聧聰忙楼聟忙聼隆茅聧聸芒聳聢莽聙碌茅聫聛忙聞掳氓麓卤氓娄聯聞氓娄鲁茅聴聜盲陆鹿忙聻录茅聫聟忙露垄氓麓庐聲氓聸篓氓聟聹茅聳鲁脩聦氓聣聼氓搂聤莽禄聵忙聥聟忙赂職氓聤赂氓茫聜聠氓聻碌茅聧聼忙聺聭莽聲路茅聨掳忙聰陆茅聬聨茂陆聨盲潞聻茅聬拢忙麓陋忙垄潞茅聧聺氓聥碌忙搂赂莽录聛茫聞漏盲潞露氓氓聭聤忙聝聰茅聤聤茂赂戮氓聨碌茅聴聛猫炉隆氓聻聨茅聧聸氓聳聳忙驴聽莽聰碌氓聛聸茅聨鲁氓驴聸忙聜搂茅聬聵猫莽庐聳忙驴隆茫聜聟忙聡聨莽禄聙氓聜職莽聮潞莽聦麓忙氓漏垄猫忙麓聼氓莽禄聵忙聥聟茅聧聸氓漏聜忙職聢氓漏碌莽聜虏氓录露莽禄聮禄茫聙聛茅聫聡莽聠潞茅聶聞茅聳赂忙露聵脜聢茅聨?``True``茅聴?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    tuple[str, ...]
        茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼忙聲聥莽聭聹氓潞篓茫聙聝茅聤聣脩聟莽聼聶茅聳赂忙聼聜聜氓聛聯茅聧搂忙楼聛芒聮聭忙陇陇忙聨聠氓聛聯莽录聛氓陇聤忙戮鹿忙聸聼莽鹿聺氓搂聵氓聨陋茅聴聛氓聣虏氓聛聟莽禄禄氓露聟脦拢猫陇掳忙聨聯莽聟聲茅聬聨茂陆聨氓聙聳茅聨麓莽聤鲁氓麓虏忙驴聻忙聢聶氓聻鹿茅陋聻茫聢隆盲驴聤茅隆聳忙驴聬氓職聶茅聴聜氓陇聤莽搂鹿氓搂聤猫聶鹿莽麓聮氓娄炉猫炉虏莽職聭茅聴聛莽茅聨鹿氓聸卢氓录禄茅聰聺氓聴聵氓聞聼茅聳驴忙露聵氓芦聫聠忙露垄忙垄潞莽禄聥氓聺聴莽聙鹿忙聸篓氓聣聸茅聧聶氓聭麓忙垄潞茅聴聢忙碌楼氓聽楼氓录虏氓篓聭茫聢聽莽陇聞茅聳禄忙篓录莽虏炉茅聬聯忙聸聽芒聜聟聣氓聛聯茅聳鲁脩聦氓聣聼氓庐聲忙聢娄忙聜聶茅聬聭忙篓潞盲潞聴茅聴聛氓聯聞忙麓篓茅聧聽忙聞掳氓麓拢猫鹿聡忙聳戮莽鹿聠忙陇陇忙聽篓莽虏聦忙碌聹氓楼赂忙聼聸茅聤聤茫聜聡莽聼聶莽聙碌茅聢陆氓陇聤脦聼氓氓聭聻聰莽录聛猫戮篓氓赂聮猫聻拢茅聳禄忙聢聻脡聭茅隆聰氓聜聹莽麓聯茅聧聦忙掳聜卢氓鲁掳脙鹿茅聧楼茫聞拢盲禄聢茅聳赂忙驴聞氓聞虏茅聫聠忙禄聢忙聼聧茅聤聣茫聝搂茂录聧茅聳潞氓聴庐聯茅聫聦茂陆聣氓录聧聦忙掳禄聵茅聴聛莽禄聵忙聦禄茅聰聲茫聢陇氓录禄茅聬聰莽垄卤忙聲聧茅聳赂芒聞聝茅聢搂忙聜聜猫聴聼?
    """
    if _resolve_paper_scope(_ACTIVE_PAPER_SCOPE) == "sim_e9_only":
        return _SIM_E9_ONLY_CORE_EXPERIMENT_CONFIGS
    if include_safe:  
        return _SEARCH_SCORING_EXPERIMENTS + (_SAFE_SCORING_EXPERIMENT,)  
    return _SEARCH_SCORING_EXPERIMENTS  


def _inject_self_baseline_safe_metrics(metrics_by_experiment: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮盲禄聥忙聼陇忙驴聞卢芒聜聽芦芒聜掳卢茅聫聦茂陆聣氓录聧聦忙掳禄聵莽录聜盲陆虏氓聛聯氓猫路潞氓茅聧聥忙聮禄莽聧聧篓氓搂聨氓茫聜聫莽录職茅聧聴氓聠虏莽聲路茅聫聡莽聠路莽路聻茅聳鹿茅聜娄氓聨戮茅聧聵氓聧聻芒聰聭茅聬聬忙聺聭莽聛娄茅聳禄莽聡聜氓莽禄卤莽聜潞氓鹿聥茅聰聰猫聴聣芒聜卢氓聻聟篓氓搂聴氓聴聴氓垄禄忙驴隆忙聫聮忙聠隆茅聤聡氓聥炉茅聢漏氓聠陋茫聙聤茅聴聛氓聹聡聯莽聙職盲录麓氓鹿聝茅芒聜氓聽聲氓鹿聥氓漏聤氓聭炉莽聟聭莽录聜氓聜職氓聙赂茅聧聤茅拢聨忙聥聥茅聫聦茫聝楼莽拢禄茅聳鹿氓聣搂莽虏炉茅聧聤莽聰赂忙聼篓氓漏碌氓聴聸茅聳潞氓卢聳聳茅聫?茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳茅聨潞忙聴聙忙陇陇忙聴聙氓聞鲁莽禄露茫聜聟芒聰聭茅聬聵氓庐聽氓拢聵莽录聛芒聜戮露忙聞聟聶茅聬聨忙露聶莽禄聽忙漏聙忙聼隆茅聧隆忙陋篓拢氓聸搂茂鹿陋忙驴隆氓聽聲氓麓篓茅氓聟录茅聴聛猫炉隆氓聻鲁茅聧聽忙聽颅氓录禄茅聰聼莽聤虏茅聠聡忙陇陇忙聞露莽麓聲茅聧聭氓陆聮忙垄潞茅聬聼忙聺驴忙搂禄茅聳鹿猫聶聬拢氓鲁掳茅聧芦忙驴聢茅聴聛氓聹聞陇茅聨录芒聲聟氓麓楼氓篓聠忙聮麓芒聮聮茅聳赂氓卤戮氓聡鹿茅聬聯忙聼聸莽聙拢茅聬聯茫聜聣忙聼篓茅聫聡莽聜虏芒聜聬聛忙聥聭茅聬聰氓聟录莽聟聫氓漏碌莽聜碌氓聛聯氓篓聟氓聴聸忙聼聸忙聬麓芒聵聟氓聤聼茅聳鹿氓聫聣忙聜聜茅聨聡氓麓录氓漏碌氓聽垄忙聡聳茅聴聜盲陆潞莽虏炉茅聨赂猫聧陇莽虏聶茅聨戮猫炉虏莽聞碌忙陇陇忙聨聠氓聛聯莽录聛莽聤虏莽搂鹿氓庐聲忙聸垄忙聼聠氓氓聭聳聡茅聴聤忙麓娄茅聫聦茫聞搂忙卢垄茅聬聬莽聝聵莽聟聲茅聳潞氓聸楼氓聤聦茅聧聵氓聯楼莽麓聮莽聮聡氓聫聣茅聳潞氓虏聥莽露聛茅陋聻氓聸聬聬莽聰碌茫聜路茅聴聜盲戮聙芒聜聜虏猫聥戮搂莽鹿聸莽聛聫忙聽陇氓聛聯茅聨掳盲陆赂忙庐篓氓茫聜聠氓聤聭茅聧漏氓聥炉氓麓楼莽聙拢莽聟聲忙驴聻忙聢聺氓麓聬聫芒聞聝莽鹿聸茅聧聠赂莽陇聥忙驴庐氓漏聝忙陇陆茅聫聦氓聭麓忙聜聶茅聨禄忙聨聯忙茅聴聛猫氓炉聮茅聧聯莽聠路氓篓虏忙驴聻忙聢聻猫聣戮茅隆拢氓聯聞茅聤聢氓驴聯莽聵聹茅聳赂茫聞楼莽拢聧聮忙聨聴氓麓录茅聤聣茫聝搂茂录聹茅聴聛莽禄聵氓聤聳氓篓录忙卢聬莽娄聮茅忙露聴莽鹿聠茅聳碌氓聽聺莽鹿聵氓忙聜聴莽禄聴忙露聵忙碌聜茅聫聛氓聜聻忙聼拢氓娄炉猫聜漏氓赂聸茅聳禄忙聮麓莽聸聫聦忙露垄氓聼聞茅聧聬莽聜卢茅聴聜氓氓聣娄茅聧聯莽聠录氓录禄茅聬聰莽垄卤忙聲聤莽录聛忙露聵茅聬陇盲戮聙忙聜聴莽聭聶氓聥茅聳潞芒聜卢忙聺聢芒聲聝莽聼聣茅聳鹿莽聝聵茅聬聮猫聜漏芒聜卢氓鹿驴氓聞卤氓娄露垄氓麓垄茅聮聵氓陇聠忙聰陆茅聳禄忙聞颅忙陆聶茅聬聫茂鹿聜忙聼拢茅聬聰忙驴聠忙麓陇莽禄聽莽聤禄忙聼聸茅聤聣氓茅聳禄忙聮麓莽聸聨庐忙漏聙忙聜聶氓篓麓氓卢潞戮茅聳禄忙聳驴忙碌录氓聠庐忙垄禄忙碌拢茅拢聨盲赂虏莽录聛忙聫聮氓垄陆茅聨鹿茫聢陇氓麓录茅聤聣茫聝娄忙搂聴聛茅聙聻氓卤陆氓垄麓茅聳潞?p95 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿氓氓卢茅隆聬盲禄聥忙聼篓茅聬聰氓聯聞脨垄莽禄戮忙聝搂茅聫聦忙露垄氓聼聞茅聧聫芒聲聜氓聻驴氓庐聲茂陆聢莽庐聸茅聫聡忙盲潞戮茅聬聼忙卢聫氓莽禄聙芒聜漏碌莽聜虏莽聹掳茅聧聰忙聢聺莽聲路茅聢楼忙聞聛忙陆漏茅聳录忙聬聛忙陆搂茅聢搂莽聛職莽禄禄忙露垄氓鹿聥茅聬聬氓聻芦氓聶陆茅聴聛莽禄聵氓赂聤莽禄卢聯氓录?茅聴聜氓聜職氓聙赂茅聧聤莽聝陆忙聡聴茅聧聯盲陆赂氓聻聺忙陇陇忙聽聛聸茅聮聝莽聠录忙聼篓茅聬聰氓聯聞脨垄莽禄戮忙聝搂茅聫聦莽聠录氓鹿聠猫陇聧茅隆拢氓麓聡莽聴陋茅聨鹿忙聞颅茅聴聡氓忙聼篓氓漏碌氓聴聵氓茅聳赂茫聜聜氓聛聯莽录聜氓聜聶猫茅聧聥忙聮禄忙聼聸茅聨掳茅聺聸忙聻聼茅聳禄忙聮卤氓聞碌茅聫聦茂陆聣氓录庐茅聢搂氓篓聠忙聮鲁莽茅聳驴忙聸聻氓聙鹿茅聬聯忙聸垄忙聼聧莽聭聶氓聥芦氓聤陇氓篓麓?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿氓氓卢茅隆聬盲禄聥忙聼篓茅聬聰氓聯聞脨垄莽录聛芒聜聧芦莽聠潞莽茅聳赂芒聞聝脡聭莽聛聫盲录麓忙聼聸忙驴聽氓聭聟聬莽录聛莽禄聵莽鹿聢氓娄芦氓聠篓芒聵聣茅聧聴莽聜虏氓拢聢茅聴聜盲陆潞莽聬職氓麓聺茅聨麓茂鹿聙猫聯聼茅聳潞氓聸楼氓聻鹿茅聳卤氓聸篓莽鹿聺茅聧篓氓搂聢莽禄聴忙聢聽莽麓聯茅聧聦忙卢聮氓聟聦氓漏碌莽庐聳茅聰聺莽聜麓氓戮聞莽聛聻忙聝搂氓潞聤氓漏碌莽聤禄忙陆搂氓篓虏茫聞搂氓录搂茅聢搂忙垄潞莽禄聥忙聦聨忙鹿掳莽录聛氓氓聻碌茅聢禄氓露聣忙聲聬忙戮露氓忙聥潞莽录聜盲陆赂茅忙卢聯氓麓聲猫鹿聡忙聸聽莽拢录氓猫路潞茂鹿娄莽禄聣莽聝聵茅聬聬忙聫聮忙職垄茅聳碌氓聸篓氓聤聶茅陋聻忙聨聴氓鹿聥茅聬聵忙碌聥氓聛聯茅聬聬莽聝聵氓聛隆忙驴聽氓漏聜氓聲掳忙聲聢氓漏碌茂鹿楼氓盲禄聥莽聧庐茅聫聧氓聙路茅聳鹿莽禄聵氓赂漏茅聬聨忙聤陆忙垄禄茅聧聦忙卢聭茅聨录氓颅聵莽虏聨氓炉掳氓漏聝莽娄聮莽聙拢盲禄录茅聴聛莽禄聵氓聻录氓娄芦氓聥聼聥茅聧楼莽聟聸茅聳赂茫聝垄忙聶聸茅聫聥氓潞隆脙鹿氓氓聠搂潞忙驴漏聜氓庐聲忙聨聭茅聧聴莽聝聠忙聺赂莽录聜氓聜聶莽茅隆聭氓聴聸氓鹿聬茅聧聯脩聝氓聻聺氓漏碌莽聤鲁猫聣戮莽禄聽猫聶芦盲驴聤茅聤聢氓聥聟聦茅聳?``e0_safe_mode`` 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯莽聙鹿忙聸聵氓聞鲁茅聢搂莽露聠忙碌聹氓聽聲忙聜垄茅聧隆忙篓禄氓聫聫忙驴隆莽聜露忙碌聹茅鹿聝忙垄潞莽聮聡芒聙鲁莽聛聽忙陇陇茂赂戮忙聲禄茅陋聻氓聠聼聞茅聧聬芒聲聛氓聤聟茅聴聛茅聺聸忙驴隆氓聸麓聺茅聧聸聼氓聙碌忙莽聜虏莽聛聺茅聧聰忙掳露忙聼聼茅聯聫氓搂聧茅聳鹿莽鹿聺茅陋聻氓卢聲聜莽聭聶茂陆聠氓芦录莽录聜氓聜聶莽茅隆聭氓聴聶芒聜卢聸氓聠驴猫聙驴茅聳禄忙楼聙莽聣聤茅聧聶氓驴聰盲驴聤茅隆聳忙禄聨莽聞戮氓漏碌氓聙驴茅聫聦忙露垄氓聼隆茅聧聬茫聞陇莽虏露茅聴聛氓庐聽忙茅隆娄氓聼垄忙聬麓茫聞楼莽庐拢茅聳禄忙篓潞莽聛職茅隆虏聞忙垄潞茅聭陆茫聜聟忙聻聭莽录聛莽聻录莽禄聴聢忙聜路茅聨碌忙庐戮氓茫聜聠莽聣聹茅聧聥忙露聶氓聞聴聜盲陆赂莽聳聡氓娄聴茫聞漏忙聡聴茅聧芦忙聸聻莽?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    metrics_by_experiment : dict[str, dict[str, float]]
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬氓聫聣莽聳聞茅聴聜盲陆赂忙聠隆茅聨赂氓聴聸氓聼聙茅隆聮氓聥芦莽拢禄茅聳鹿莽聜聣職氓篓录芒聲聜茫聛聞茅聨戮猫路潞盲禄聮茅聳禄茫聝庐盲潞莽禄聵忙聥聟茅聧聸氓漏聜忙職聢氓漏碌莽聜虏氓录露茅聰聲茫聢陇茫聙聛氓篓聠忙聮鲁莽卤聧茅聳赂氓聳聯茅聧?茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙碌拢茅聺漏猫聙驴茅聴聛莽搂麓茅聧聦茫聞搂忙聠赂茅聫聝氓聽聺莽庐聳忙驴聻氓路潞聹忙聝聼茅聴聛茅聺聸茅聧聽忙楼聝莽麓聻忙聬麓茫聞娄莽聟聼茅聳禄忙篓潞氓录露忙戮露氓聥颅莽麓聮忙碌拢猫搂聞氓掳娄莽聙碌忙聜聜氓炉麓录茅聬聰氓聯聞氓鹿聧茅聴聛猫炉虏氓莽禄聥忙聨聴芒聙娄茅聧楼脩聛芒聜聨聴氓娄露茅聧楼茫聞楼忙聻聭茅聴聛氓聯聞氓聙陆氓篓聣忙聸垄氓录鲁茅聰聲氓聜聸莽聟聲忙陇陇忙聞露莽碌戮莽禄聙芒聜聳鹿氓聳聳氓忙麓氓篓聭忙掳卤忙驴聽忙氓聳聸氓篓聟忙聸聽莽虏漏茅聬聰莽聟聨氓聧鲁茅聳禄忙篓录忙聲赂茅聰聸忙驴聥忙垄潞莽禄聥忙聦聨忙鹿掳茅聳禄莽聠聺氓聸搂莽莽聙鹿茫聝楼茅聬聯茫聝娄忙聟篓氓娄炉氓聻聟聬茅聳禄忙聮露氓聳聳茅聫聦茫聞漏氓录麓氓娄陇芒聜卢忙碌聹氓聣搂莽麓聯茅聧聦忙掳卤脨垄茅聳驴忙聸篓盲潞聹茅隆聲茂陆聠莽鹿聺氓盲录聥茅聴聛氓陆聮莽禄聙盲戮聙氓鹿聝茅聨麓茂赂陆芒聮聭茅聳赂忙露聵茂鹿拢莽禄露茅聛聧忙聼聸茅聬聵忙聞聛忙庐聹忙莽聜虏氓聲芦茅聢禄氓潞篓氓鹿聵莽禄庐猫搂聞氓茅聴聜盲陆潞莽虏聧篓莽聠录氓录虏氓篓聭莽露聳茅聳鹿莽聝聵茅聬聯忙聸垄忙聼聲忙戮露氓卢聛楼茅聳鹿氓鲁掳氓聧卤忙陇陇忙聽聛聻茅聢路忙聨聭脙鹿茅聰聺氓聥庐茅聳潞芒聜卢忙碌录氓颅聵莽聟聲茅聬聨茂陆聨氓聛聞忙驴露聡芒聜忙麓聵茅聧篓氓聺聴莽聧庐茅聧楼莽潞搂茅聬聽盲戮炉莽聟聣莽禄卤?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    dict[str, dict[str, float]]
        茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聟忙職娄茅聳禄茫聢垄莽潞戮氓漏職忙聼拢茅聨掳忙職漏茅聳禄莽聭漏茅聬聯氓麓露茅聤聤脩聙忙搂聢莽录聛莽聜聞鲁氓篓录茂驴聽氓录禄茅聰聺氓陇聤茅聳碌氓驴聤氓聣聺氓搂拢氓聥颅莽麓聯忙碌拢忙聫聮盲潞戮?``e0_safe_mode`` 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬氓聫聣莽聳聞茅聴聜盲陆陆茅聧篓氓楼赂忙聜聵茅聨掳莽聣聢氓聛隆茅聳鹿忙氓聳聴氓篓聢忙驴聥忙垄潞茅聳露職莽楼路忙驴聻氓聭麓聼茅聨篓禄莽卢聼茅聢搂茅聯聫猫氓聟聹茅聳赂忙露卤氓聳聳氓篓聭脩聟莽麓聯忙碌拢氓聯聞脨拢茅聳赂茫聞陇莽碌禄茅聬聫氓卢聰聭茅隆聰氓盲禄聹茅聳赂氓卢聧聡茅聫聦茂录聞盲禄娄莽禄聥氓聻聮聣茅聬聨氓聻芦忙戮聵莽聙職氓鹿聠茅聫聝氓聽聺忙陇陇氓聟聵茅聧聥忙聮麓莽鹿聺茅聧楼茫聞垄忙聥潞茅聴聛茅聺漏氓炉聦茅聧聵猫聴聣芒聳聧茅聫聡茫聢隆莽聟聲茅聳禄忙篓潞莽拢颅忙戮搂莽聰碌芒聜氓聳聸茅脩聡氓聼楼忙戮露氓漏聨莽聟楼茅聳赂忙露卤氓聻卤氓忙聢聻莽鹿聺茅聬垄茅聺聸盲禄娄茅聳赂茫聞楼莽聢录茅聨聝陆氓娄露茅聧聹盲陆鹿氓聝碌茅聴聛忙聦聨忙聼篓忙戮搂莽聲聦氓茅聮聵氓氓陆聮忙聲聬茅聧聲莽聟聯茅聴聛氓聭聤莽鹿聠氓篓聟茂陆聡忙茅聨潞忙聴聙氓芦職茅聳录莽垄卤氓聣聶茅隆拢忙麓陋忙垄潞忙碌录忙卢聭忙聜录茅聳赂氓驴聰脡聭茅聨卤茫聞漏氓聼聙茅隆聮氓聥卢莽聟聫茅聳赂莽鹿聝茅隆楼忙禄聝脙鹿氓氓聥聼聥氓篓拢氓聸搂茂鹿聨庐忙卢聫茅莽聟聨氓茅聴聛猫氓颅漏莽禄聥氓聽楼氓录虏茅聬聵猫聧陇氓碌聛氓猫路潞莽禄隆芒聜漏碌茂鹿漏茅聧聵忙聬聛氓鹿聫茅聯聫氓聹颅莽拢陆氓篓麓茂陆聟氓茅聧聨忙聮麓莽鹿聸忙戮露氓卢莽录聛莽虏炉莽禄禄忙驴聠茅隆聯莽聤虏氓鹿聬茅聴聜盲陆赂忙聠隆莽禄聥忙聨聭氓搂聴芒聜卢茅聨聠庐忙聬鹿茅聧娄茂录聹氓茫聜聠忙麓聳茅聨鲁氓潞篓莽聧娄陇氓聠拢氓聛聯茅聧篓莽路虏茅聬聨忙掳录氓聶拢茅聧漏芒聜陇陇忙聨聭茂鹿娄莽禄聣茅聺聸脙鹿氓漏聤氓聥颅莽庐聻忙?氓漏?茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聦忙碌聳茅聳碌氓篓搂氓聭炉莽聞录忙驴聻忙聮麓氓聼聝茅聧聥忙聮鲁芒聰聭茅聢楼氓麓聡氓聼拢莽聙鹿忙聸聵氓聤聧莽禄禄忙驴聡忙聥聟茅聯聫氓聹颅忙職?
    """
    if "e0_safe_mode" in metrics_by_experiment:  
        return metrics_by_experiment  
    reference_metrics = metrics_by_experiment.get("e9_dual_degradation") or metrics_by_experiment.get("e2_async") or {}  
    reference_p95 = max(float(reference_metrics.get("p95", 1.0)), 1e-9)  
    normalized_metrics = {  
        experiment_id: dict(metrics)  
        for experiment_id, metrics in metrics_by_experiment.items()  
    }  
    normalized_metrics["e0_safe_mode"] = {  
        "p95": reference_p95,  
        "ekf_p95": reference_p95,  
    }  
    return normalized_metrics  


def _build_base_estimator_cfg(estimator_name: str) -> dict[str, Any]:
    """茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聟忙職娄茅聳赂氓聛聟茅聧聶氓驴聸忙聼隆忙戮露氓忙庐驴聻氓路潞聹茅聬聯忙露垄忙聼聸氓芒聜聬聮茂鹿聜氓录虏茅隆聫氓聽聼氓聛隆忙驴聽氓漏聜氓茅聬聫忙聺聭芒聰聭茅聰聸氓聥職卢忙楼聽莽聜虏忙麓氓聣聛忙聧聞茅聤聤茫聝垄氓聛聯茅聬聬莽聰碌莽拢陆氓篓麓茂陆聟忙聬掳聫茅聴聛氓聭聤忙驴聻氓漏聜莽聮聡忙碌聥忙搂聢茅聳碌氓驴聲氓茅聤聤芒聲聜莽聟聶茅聬聨忙露聶莽禄聽忙聽聼聸茅聰聺氓聸搂氓聤聥茅聳鹿盲戮驴氓聤聭茅聫聛忙聞掳氓麓篓茅脩聝氓娄路莽录聜氓聜聶猫聝露莽禄聝聞茅隆聳忙掳录氓職聲茅聳赂忙麓聳莽禄聽茂陆聠莽鹿聺茅聴聢忙露聶忙聬聬茅聳鲁脩聦氓聣娄茅聧篓猫戮漏忙聧垄?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮盲禄聥忙聼陇忙驴聞卢芒聜聸聽忙卢垄茅聤聢氓聭聰陆茅聳禄忙篓潞莽聳聡忙戮鹿忙篓录莽麓聮茅聢搂氓戮聞茅聨掳莽录聜盲戮聡氓聳聸茅隆茅聧隆忙露聶莽聟聸茅聬聨茂陆聨氓聛聟茅隆楼氓聽楼忙聼隆氓庐聙芒聜聧聽忙聮禄氓聼聙茅隆聮盲陆潞芒聤聲茅聮聝茅聛聧忙聼聸忙聬麓茫聢隆莽陇聥茅聳潞氓虏聥莽露聛氓炉麓聮氓搂聵莽虏炉茅聴聜盲陆陆莽禄禄忙聞庐氓拢聲茅茫聢聽莽聞碌忙陇陇忙聨聭氓聳職氓忙聞鹿聸茅聧聶氓氓垄聬聯茫聜聣忙聼拢氓娄氓職聶莽禄戮忙聝搂茅聫聦莽聠录氓麓聹猫陇聫莽聰炉忙露垄忙聼聸忙驴聽氓聜陇莽聟录茅聳潞氓卤禄氓聙聺茅陋聻盲戮聙氓鹿聮茅聨麓忙驴聠脨聭茅聴聜盲陆赂忙聠隆猫聯卤茅聳赂忙聴聙莽聭漏茅陋聻氓聠篓脦聰茅聧聸莽楼娄茅聴聛氓聣虏莽聟聢茅聧聽忙聽篓忙聟篓忙聬麓芒聶聙忙聰陆茅聳禄忙聞掳聶氓漏聙忙聮聼聸茅聫聝芒聜卢茅聧篓莽聰赂茅聰聺氓莽庐鹿氓篓麓茂陆聟忙聭聹茅聰聸忙禄聝芒聰聭茅聬聬忙聺聭莽聛聼茅聳赂茫聞娄忙陇陆氓庐聲忙聞掳忙聜聹茅聭潞茫聝搂氓聨戮茅聴聛氓聭聤莽赂驴茅聧聨忙聼楼氓录鲁茅聢潞氓聭聟聫?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛氓露聣氓麓聼茅盲陆鹿忙驴聽氓戮路忙聺驴芒聜聞掳氓麓聭茅聨戮莽禄聵莽聟楼茅聳潞氓聸篓脙陋茅聬聫猫聦卢芒聜忙麓聹茅聧聫忙漏聙茫聙聛氓芒聜聨庐芒聲聟氓鹿聯茅聨潞忙聸聟忙聸垄忙垄禄忙碌拢莽颅聥氓職聶茅聬聨忙露聮忙聶聺莽禄卢聯氓鹿聮氓莽聝漏氓聛聯茅聰聲氓聜職莽聜聤忙陇陇忙聨聧猫茅聨路忙麓陋忙垄潞茅聧聸氓漏聞莽庐聯茅聬聨忙掳录氓聣聼茅聧漏芒聜陇陇忙聨聠氓聛聯茅聳禄忙聽芦氓聨搂茅聬拢莽聝陆氓鹿聡茅隆聭芒聲聜忙拢聝氓庐聲忙漏聙茅聧隆忙卢聫氓搂赂忙驴聽莽聰碌氓搂掳茅聧篓氓楼赂氓麓潞茅聫聧莽陇聣茅聳潞氓露聨氓聨录茅聬聯忙聸垄忙聼聼莽聙碌莽盲潞录氓篓麓忙禄聞莽虏聯茅聫聦莽聠录氓麓聧聲忙拢聻忙驴聻氓颅聵莽虏聯莽禄聽忙聽搂茅聯聫猫氓芦職茅聳录莽垄卤氓聣聸茅隆聰氓莽麓聯忙碌拢莽颅聥氓職聶茅聳禄忙忙拢聺茅聨庐茅鹿聝忙聜聹茅聮聵氓忙陆聤茅聴聛茅聺聸莽聣聠茅聨虏忙忙陇陇忙鲁聲茅聨鹿氓聥炉忙聜聨戮氓聫聧茅聳禄莽聜麓莽莽禄禄氓聽聺茅聯聫猫氓聟聹茅聳赂忙露卤氓聫聨氓娄聽楼忙垄潞莽聭聶氓聥聛聼氓篓麓忙禄聟氓聛聸茅茫聝娄莽鹿聺茅聧楼脩聡忙聲聹氓漏碌脗掳氓聙聬茅聧聥忙聮禄忙聼隆莽聙拢芒聲聝茅聯禄忙聽颅忙聼篓氓漏碌氓聴聵氓聶鹿茅聳潞氓聴聶氓聛陇茅聫聦忙露職芒聜卢茂陆聟氓莽禄隆芒聜卢茅聴聛氓聯聞莽聼聣莽禄禄忙驴聠莽聲路茅聧聸聻莽聟聸氓篓麓茂陆聣忙聰卤茅聧聲氓聠漏莽鹿聺茅聬垄茅聺聸脨楼茅聳潞盲陆鹿忙聜聜氓庐聲忙聢娄忙聜聶茅聼卢猫鹿卤聴茅聴聛氓聭聤氓聤娄茅聧聽忙楼聟氓录虏氓漏碌氓莽聟聫茅聳赂莽鹿聝茅隆楼莽聜麓莽鹿聸茅聧聸莽忙驴漏聞莽虏聨路氓聭聠麓茅聧聲莽麓聵茅聴聜盲陆赂忙聬聨鲁氓驴聸忙聝聞茅隆聳忙掳卤忙職娄忙陇陇忙聽篓莽篓聭茅隆聲茅聛聧忙聼隆忙戮露氓卢聛楼茅聳赂氓莽搂鹿氓猫聶鹿莽麓聮氓搂聴氓聴聶氓聤路茅聴聛猫陆掳莽茅聨陆忙禄聞莽芦聼茅聫聡莽聠潞茅聬聨茂陆聨氓聛聠茅聧聵茅聛聧忙聼拢猫鹿聡忙聸聻盲禄聹氓漏垄脩聜莽虏聶茅聧漏氓漏聡氓录麓茅聤聫芒聙鲁芒聜卢氓聽聲莽聟芦茅聧楼茅氓聢掳莽聙碌氓聶拢茅聫聦芒聙鲁莽聲聺莽聙拢脡聭氓娄陇莽聤碌氓聛聸茅聰聲氓录禄氓篓聭茫聢聽莽卤聧茅聳鲁脩聦氓聣聼氓庐聲氓陆聮忙聡聹茅聬垄莽忙聰?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    estimator_name : str
        氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄茅聳禄忙聞篓聤茅聬聮猫聤楼莽鹿聸茅聧隆忙篓禄氓掳颅茅聬聼忙卢聶莽庐聯茅聨庐忙楼聟忙聲聬忙聬麓芒聙虏莽掳禄茅聴?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮盲禄聥忙聼陇忙驴聞卢芒聜聸聽忙卢垄茅聤聢氓聭聰陆茅聳禄忙篓潞莽聳聡忙戮鹿忙篓录莽麓聮茅聢搂氓戮聞莽聛聻莽盲潞戮茅聳赂氓驴聯忙碌聹茅鹿聝忙垄潞茅聧聸氓漏聝氓陇聞茅聳赂氓卢茅聨聝搂脙潞茅隆聫氓聭陋忙聥潞茅聴聛氓聭聤莽鹿聜莽聙職氓漏碌氓聴聫忙聰陆忙陇陇忙聴聙氓聞鲁茅聧聵忙聮聼聼茅隆聳忙掳卤莽聞赂忙莽聜麓莽禄卤忙聨聭脦聼茅聬聯氓潞隆莽茅聴聜氓聜聶莽陇聛茅聨虏茂驴聽氓鹿聬茅聫聞氓驴聥盲潞鹿茅聳赂忙聞碌氓聳聳茅聧聦茫聞漏忙聼聼莽录聛茫聢聽忙聻聼茅聳?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    dict[str, Any]
        氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄茅聳禄忙聞篓聤茅聬聮猫聤楼莽鹿聸茅聧隆忙篓禄氓掳颅茅聬聼忙卢聶莽庐聯茅聨庐忙楼聟忙聲聬忙聬麓芒聙虏莽掳禄茅聴?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮盲禄聥忙聼陇忙驴聞卢芒聜聸聽忙卢垄茅聤聢氓聭聰陆茅聳禄忙篓潞莽聳聡忙戮鹿忙篓录莽麓聮茅聢搂氓戮聞茅聨掳氓聨碌茅聳潞氓莽聣聠忙戮搂盲禄聥忙聜戮茅聳露聠莽聟聲茅聬聫莽聜露忙驴庐氓聜聻忙聼隆莽聛聻莽聜卢莽陇聥莽聙鹿忙聸聽氓聨搂茅聢鹿忙聢娄氓麓露猫陇聫茅聬聸芒聲聝莽麓聯?氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄忙驴聻忙聢聶氓聻潞茅聧聥氓芒聰聭茅聬聵氓聫聣忙聬聬茅聳赂茫聞楼氓聙赂茅聹聣茅聳驴忙驴聠忙聡聨茅聫聠忙卢聭莽麓聮茅聨禄忙聨聰莽搂潞忙驴庐氓漏聝忙陇陆氓庐聲氓麓聼茅聧鹿聜忙垄潞莽录聛忙漏聠忙庐聰茅聬聨忙掳颅忙戮聵茅聬拢氓鲁掳芒聰聭茅聧隆忙卢聬忙庐聲茅聴聛氓聭聤忙麓聳茅聬聫忙掳露氓录虏氓氓聽聼芒聮聭茅聳赂忙聮麓氓陆聝忙碌聹忙驴聡莽鹿聸茅聧聶氓猫聙驴莽聙鹿忙聸聼氓聻驴茅聫聛忙聞戮聞莽聙拢氓鹿聧茅聴聜盲陆陆茅隆聰忙聽颅氓麓掳氓猫戮漏莽聨禄忙驴隆茫聜聟忙聡聫茅聢路?
    """
    return copy.deepcopy(  
        _load_yaml_config_cached(str((ROOT / "configs" / "models" / f"{estimator_name}.yaml").resolve()))  
    )  


def _probe_model_param_count(model_name: str, model_cfg: dict[str, Any]) -> int:
    """茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜麓莽聣聽茅隆垄忙聸垄忙聲聜茅聮聵氓氓聛聯茅聧篓氓聫聫忙驴庐忙楼聙忙拢聴忙碌聹忙禄聟芒聜卢猫搂聞忙聦赂氓娄驴聥氓录禄氓篓聭茫聢聽莽聟聵茅聳鹿氓聜職忙驴聽莽莽麓聲茅聧聥忙聢娄氓麓聬茅聫聧忙職聹茅聳鹿莽聝聵莽潞戮氓陆聮忙聼聸氓篓聭忙漏聢氓聤聼氓漏垄脩聧莽鹿聺氓搂聤莽禄聵忙聥聟茅聧聺氓聥禄聵茅聴聛猫莽鹿聬茅聬聴忙聮鲁茅聫聦茫聝娄氓聛聬茅聬聽氓聸陋氓掳聨陆忙聸聻芒聰聭茅聬聬忙聺聭莽聛聼茅聳赂茫聞篓茅聧聨忙垄禄忙赂職芒聜脩聡脣聡忙碌聽氓莽陋聴茅聳潞氓露聨氓聨录莽禄聙氓聽聺脙鹿茅聬聯氓潞拢忙聭聽茅聳鲁茅聰聥氓聻露茅聨卤茫聞漏忙聜路茅聬聴氓聽聼忙聜聶茅聴聛茅聙聻氓卤陆氓垄驴聻氓聫聣莽虏聯茅聧漏芒聜卢忙陇陇忙聨聧莽陇聛茅聧陇忙聼聸氓娄炉氓聯聞芒聮聰茅聳赂忙聨聯氓赂隆茅隆垄忙漏聙氓搂聙茅聢楼氓聟聟莽掳禄茅聴聛猫聧陇氓聛聻莽禄聥氓聽楼氓聼聙茅隆聮芒聜卢茅聧聵忙聽篓莽芦聼茅聫聡茫聢隆芒聮聭茅聳鹿莽忙碌路莽禄聽忙聮露莽麓聮茅聤聤脩聝氓赂露茅聴聡氓忙聼拢茅聨麓茂陆聟脨聯茅聳赂氓漏聜莽聛職茅聨卤茫聞楼茅聧隆忙莽庐禄茅聴聛莽聭掳氓聲驴茅聨鲁忙聞庐忙搂聬茅聨潞忙聮麓莽露聡茅聳碌氓莽庐聭茅聴聣氓露聣忙垄潞茅聬聼忙聺驴忙搂禄茅聳鹿猫聶聬拢莽聝陆茅聰聲芒聜禄聙氓卢聛聞茅聨潞忙聢聺芒聜猫聤?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫聫茅聬聼忙卢聫氓茅聧聶芒聜卢茅聴聛氓聯聞莽聼聣莽录聧盲陆赂茅氓聥聻聜忙陇陇忙聴聡忙拢聞茅聢搂氓聻露氓猫聶鹿莽麓聮氓娄莽垄聧茅聴聛莽篓驴茂鹿陇氓篓录茂驴聽氓鹿聝茅聰聼莽聥聙忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄忙聞卢茂鹿聫聦忙聸聼氓戮聞茅聧驴聠氓搂聨忙驴聽忙庐驴氓聳聯茅聧篓猫聧陇忙搂聬茅聨戮氓颅聵氓茅聳鹿莽禄聵氓赂聤忙戮鹿忙聸聻芒聰聭茅聬聵莽聛卤茅聳赂芒聲聜莽聣聽氓庐聲忙驴聥氓录陆茅隆聯莽聠路盲潞聴茅聴聛莽禄聵忙垄鹿茅聨聢碌莽娄聧氓氓聥漏盲潞聹茅聳鹿忙聣鲁忙聶聸茅聬聮忙麓聼拢忙聬麓茂录聞忙聭聽莽录聛莽禄聵莽鹿聢茅聨禄聥忙拢聝氓篓聭忙楼聙忙聭聥茅聴聜盲陆潞茅氓聴聸氓麓聺氓篓聠氓驴拢氓職聲忙陇陇忙聞漏氓聙聳莽聙職忙掳露忙聼陇莽潞掳氓聧聻氓忙驴聻氓聽楼莽庐聳氓?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿茅聳鹿茅聴麓莽莽潞戮莽聭掳莽聙拢忙聧拢茅聳禄忙茅聹聣茅聳驴忙驴聠忙麓陇茅聧聰氓芒聜忙麓聵茅聬聯茂驴聽氓录禄茅聬聰氓聟录氓聛聥茅聳赂氓聳聯茅聧聭茂驴聽忙垄潞莽录聛忙篓潞莽卢聮茅聳禄氓驴聰氓虏赂忙驴隆莽聰赂氓麓聼茅忙掳卤茅聬聨氓鹿驴氓聞卤茅聬聴氓聴聸氓麓垄茅聰聼莽聤鲁芒聮聭茅聳禄莽聠赂忙戮聵茅聫聠猫戮篓莽鹿聸氓庐赂氓录庐莽聙碌茅氓楼赂氓麓录茅隆聬茂陆聨芒聜聸卢忙垄禄忙碌拢氓聭聤氓聲聻茅聳鹿莽茅聨聝陆氓聼隆茅聧聲忙拢篓茅聴聜氓聜聶莽茅聨虏茂驴聽茫聙聝茅聧芦茫聝楼莽拢禄茅聳鹿莽聣聢忙戮聵茅聧聭莽聤禄忙聼隆茅聧聬茫聞搂氓垄聸茅聳禄忙聮麓莽聭漏茅聨忙聜陆氓篓聢猫路篓氓聤赂茅聬聻忙掳楼氓聞碌氓猫聶鹿莽虏聶氓篓聠忙聝搂氓聣卤茅聴聛氓聹聡聯氓篓虏莽聧庐茅聧聬氓聙禄茅聳陆忙篓潞茅莽聟聨茅聤聢氓聴聵芒聮聬茅聬聨忙掳卢茅聟拢忙驴隆忙聬聛氓聼隆茅聧聦忙氓芦录茅聴聜氓聜職氓聙赂茅聬聴氓聠录虏氓漏碌氓聽垄莽聼聯忙陇陇忙聨聧猫聝露莽禄聽茅鹿聝忙聼聸氓氓聠篓忙職聼莽录聛氓氓聶拢茅聫聦忙露垄氓聼聻茅聨聜氓氓漏碌莽聤碌忙聲赂茅聧聥忙漏聙氓麓聽茅聬聵猫炉虏氓路聺忙陇陇忙聽篓莽虏聽氓娄虏莽篓驴芒聰聭茅聬聵氓聻芦忙職漏氓漏碌茅聢搂氓聡鹿茅聧聽忙掳卤氓聢聮茅聧芦莽聠潞莽炉聙茅聳赂茫聝拢氓聤聨茅聧聵氓聧聻茅聤聢氓聴聴氓搂搂莽录聛忙聼楼莽庐聳氓庐聲忙聸垄氓录麓茅聬聰猫聶鹿茅聴聛莽聭掳莽聯篓茅聬聼茫聝漏忙聜聵茅聢潞氓聭庐莽聟聼茅聳鹿莽聝聵氓聻鹿忙碌聽忙露垄忙聼聲茅聧楼脩聝忙楼聽莽聜麓氓赂隆茅陋聻氓卢聰聭茅聨掳氓聛聟茅聴聜氓聜聶忙碌聡氓娄芦氓聥麓炉忙碌录忙聺聭莽陋聴茅聳潞氓露聨氓聨录茅聮聝莽聠录忙聼隆茅聧楼芒聲聛忙聻聨莽录聛氓聣聛氓聛聻莽禄禄忙露垄氓鹿聥茅聬聬猫路潞茅聴聛忙麓禄氓聨陆莽禄禄氓聽?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    model_name : str
        忙驴聽莽聰碌氓搂路茅聫聛氓聭聤忙聟篓茅聬聭氓芒聜聜聻忙聦聤忙戮鹿芒聲聥莽露聛氓庐聲茫聞漏氓录露茅聨麓忙驴聢芒聜卢茅聤聢氓聭炉莽庐掳茅聳禄忙楼聙忙拢聺茅聨庐莽聜潞忙聜陆茅聬垄氓聻芦茂录聫莽录聜氓聜聶茅聶聧茅聧篓莽聭掳茅隆聲芒聜卢茅聫聦忙露垄氓聼聻茅聨炉脩聜氓拢聲茅聴聜氓聜聶猫聝露茅隆聽垄忙聧垄茅聨麓茂赂陆忙聶陋茅聴聜盲陆路莽麓禄虏氓聺聴氓麓聴氓娄莽鹿聠茅聳禄忙聢拢忙氓聰聴聛氓庐聽忙隆篓茅聭鲁氓聽聲氓麓聬茅聬聬莽聝聵芒聮聭莽录聛忙麓聳忙戮搂忙聼楼忙聼拢茅聬聰忙聺聭氓聤聹莽录聛氓聜聸氓赂隆氓庐聲茂驴聽忙聜聶茅聢潞氓聜聸忙聺?
    model_cfg : dict[str, Any]
        忙驴聽莽聰碌氓搂路茅聫聛氓聭聤忙聟篓茅聬聭氓芒聜聜聻忙聦聤忙戮鹿芒聲聥莽露聛氓庐聲茫聞漏氓录露茅聨麓忙驴聢芒聜卢茅聤聢氓聭炉莽庐掳茅聳禄忙楼聙忙拢聺茅聨庐莽聜潞忙聜陆茅聬垄氓聻芦茂录聫莽录聜氓聜聶茅聶聧茅聧篓莽聭掳茅隆聲芒聜卢茅聫聦忙露垄氓聼聻茅聨炉脩聜氓拢聲茅聴聜氓聜聶猫聝露茅隆聽垄忙聧垄茅聨麓茂赂陆忙聶聺茅聳碌氓驴聲氓陋聧陇忙聸聽莽麓聯忙碌拢猫莽聞陆莽录聛芒聶聙氓聛聯茅聴聜盲陆潞茅聬卢氓楼赂氓麓聭氓篓聭茫聢隆氓聙路氓漏碌莽聤虏忙麓娄茅聬聯氓聠庐忙聼聸氓漏碌氓聴聴茅聳鲁脩聦氓聣聶茅隆颅莽聝陆氓麓潞茅聢搂茫聛聞茅聨潞忙聢颅忙搂赂莽聙碌氓聶拢茅聫聦芒聙鲁莽聲聺莽聙拢脡聭茅聴聛猫炉隆氓聻卤氓娄聠漏忙聟篓茅聢搂忙聼聧茅聧篓忙露聶忙芦聟忙陇陇忙聹卢忙路聡氓漏聡忙聜聶茅氓聥篓氓聛聯茅聨麓茂鹿聙莽陇聣莽聙鹿芒聜卢茅聧聲莽聳聡氓漏碌芒聵聠氓聻碌氓篓聟茂陆聣氓录鲁茅聰聲氓聜聸莽聟聫氓猫路潞忙氓篓聟氓聭麓忙聼拢茅聨潞忙聴聙忙職聳潞?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    int
        忙驴聽莽聰碌氓搂路茅聫聛氓聭聤忙聟篓茅聬聭氓芒聜聜聻忙聦聤忙戮鹿芒聲聥莽露聛氓庐聲茫聞漏氓录露茅聨麓忙驴聢芒聜卢茅聤聢氓聭炉莽庐掳茅聳禄忙楼聙忙拢聺茅聨庐莽聜潞忙聜陆茅聬垄氓聻芦茂录聫莽录聜氓聜聶茅聶聧茅聧篓莽聭掳茅隆聲芒聜卢茅聫聦忙露垄氓聼聻茅聨炉脩聜氓拢聲茅聴聜氓聜聶猫聝露茅隆聽垄忙聧垄茅聨麓茂赂陆忙聶聺茅聳碌氓驴聥氓聙聳茅聧聥忙篓禄茫聛聞茅聫聡忙盲潞戮茅聴聛莽鲁聲忙聳聛茅聧聥忙聮鲁茅聤聢氓聴聴莽卢聮茅聬聨忙掳录氓聫聝氓庐聲忙聢娄忙聲聬忙戮露氓卢聨卤茅聴聛茅聺聸莽禄虏猫聤楼盲驴聤茅聭虏芒聲聝莽拢录茅聳录莽垄卤氓聣聸莽聰炉忙露聵脟聨茅聧楼脩聧猫潞陇陇茫聢隆忙麓聼茅聫聛忙聞炉莽庐聸氓茂陆聟盲潞戮茅聳赂脩聞茅聬聯忙露垄忙聜聵茅聬聬忙莽聞戮氓脩聞莽聟隆茅聫聦莽聠路莽聲隆茅聬聴氓聽聺忙芦拢茅聴聛盲录聶莽露聡莽禄禄忙驴聠茅聧聯盲陆鹿氓麓聣茅聳碌氓漏聫莽碌陆忙碌聽氓驴聸忙垄禄茅聧聦忙掳聜卢茅拢聨忙氓搂聵莽聞陆茅聳录氓氓聙聲莽禄露茫聜聣忙聼聼茅隆聳氓聴聴忙聹聙碌氓虏聞忙垄潞茅聧聫茅聴麓猫聢陋茅聳赂氓潞垄氓拢聠茅聨鹿茫聢陇氓麓卤氓篓聭忙卢聭氓聨碌莽录聜氓聜聶茅聶聧茅聧篓氓陆聮氓录赂氓篓聭忙掳卤忙拢炉茅聳鹿氓聠漏氓聙录聜盲陆潞莽虏炉茅聬漏莽聲路茅聧聴莽聜聧莽聠录忙聬鹿茅聳露聶忙篓聤茅聴聜氓聜聶莽茅聨录茅聧楼茫聞搂莽茅聳鲁脩聦氓聣職茅聨卤茫聞娄莽鹿聺茅聧聸盲禄漏莽录聜盲戮聡氓聳聴茅聬聼脩聡氓鹿聭茅聧聲脦漏莽聭聹忙禄聢氓麓卢茅聧聸聼芒聮?茅聴?``model.params``茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓聙驴茅聳驴忙聴聢忙聲庐茅聧聥忙聮露莽虏篓聭忙篓禄氓聶陆茅聳禄莽聭漏茅聫聦莽聠潞芒聜聰碌忙碌聽脩聡忙垄禄茅聧聲茅聳潞氓虏聥莽鹿聺氓庐聲忙漏聙氓娄赂茅聤聣茫聢聽盲潞戮茅聳录氓氓聙聴忙露聯氓露聣忙聼隆茅聧聦忙露卤氓聞楼茅聳禄忙聳驴忙茅聨陆聮氓麓聧聲氓聞聯氓茫聜聫氓炉聦茅聧聯莽聠录氓录禄氓篓聭忙掳拢氓聛聯茅聰聺氓潞隆盲潞聺茅聬聫氓驴拢氓聫聝茅聫聦氓露聣忙聦聤忙戮露忙篓禄脨娄茅聴聛氓庐聽莽禄聣氓陆聮茫聙聛氓芒聜驴隆忙聬聛氓娄赂茅聤聫莽聤卤氓拢聲茅聴聛氓聹禄篓莽聝聵莽氓漏聤氓聽垄氓聫聯茅聳赂茫聝拢氓聤聧莽聛聫茅聺聸猫陇聨氓篓虏忙禄聟忙茅聨潞忙忙聲聝茅聳碌氓驴聤忙聡拢猫陇聰茅聫聦茂录聞盲禄娄茅聧聯脩聙忙搂聢茅聳禄忙聮卤氓聙聺茅聫聦莽聠赂忙陆聫茅聧聯脩聟忙職聽莽录聜盲陆潞氓搂路茅聧聽氓潞陇莽聛聝茅聴聛忙聦聨莽鹿聜茅聨鲁氓潞篓氓录鲁茅聬聬莽聝聵莽聟?
        ``model.parameters()`` 忙驴聽莽聰碌氓搂路茅聫聛氓聭聤忙聟篓茅聨戮氓聙露忙驴庐忙篓驴忙鲁聸茅聧職氓陆聮忙聼聧猫茅聧聯莽聠路茅聯聫氓聹颅莽聼聶茅聳赂忙聽陇氓聣聺茅聫聛茅鹿聝忙垄潞莽聮聡芒聙鲁忙聻聸茅聳赂氓漏聜忙陆隆茅聨庐盲录麓莽聮潞茫聞搂忙聺赂茅聴聛氓聯聞氓聙鹿茅忙卢聯氓麓碌茅聤聢氓芒聮聭茅聳赂忙驴聠氓氓娄虏氓陆聮忙聼?
    """
    model = create_model(model_name, model_cfg)  
    explicit_params = getattr(model, "params", None)  
    if explicit_params is not None:  
        return int(round(float(explicit_params)))  
    return int(sum(int(param.numel()) for param in model.parameters()))  


def _build_neural_capacity_alignment_report(  
    *,  
    neural_grid: dict[str, dict[str, Any]] | None = None,  
) -> dict[str, Any]:  
    resolved_neural_grid = neural_grid or _FAIR_NEURAL_GRID  
    lstm_hidden_dims = list(resolved_neural_grid["lstm_ekf"]["hidden_dim"])  
    liquid_hidden_dims = list(resolved_neural_grid["liquid_ekf"]["hidden_dim"])  
    if len(lstm_hidden_dims) != len(liquid_hidden_dims):  
        return {  
            "status": "blocked",  
            "reason": "mismatched_capacity_band_count",  
            "comparison_basis": "parameter_count",  
            "lstm_hidden_dim_candidates": lstm_hidden_dims,  
            "liquid_hidden_dim_candidates": liquid_hidden_dims,  
        }  

    try:  
        pairs = []  
        for lstm_hidden_dim, liquid_hidden_dim in zip(lstm_hidden_dims, liquid_hidden_dims, strict=True):  
            lstm_cfg = _build_base_estimator_cfg("lstm_ekf")  
            liquid_cfg = _build_base_estimator_cfg("liquid_ekf")  
            lstm_cfg.setdefault("network", {})["hidden_dim"] = int(lstm_hidden_dim)  
            liquid_cfg.setdefault("network", {})["hidden_dim"] = int(liquid_hidden_dim)  
            lstm_params = _probe_model_param_count("lstm_ekf", lstm_cfg)  
            liquid_params = _probe_model_param_count("liquid_ekf", liquid_cfg)  
            param_gap = abs(int(liquid_params) - int(lstm_params))  
            ratio_to_lstm = float(liquid_params) / max(float(lstm_params), 1.0)  
            pairs.append(  
                {  
                    "lstm_hidden_dim": int(lstm_hidden_dim),  
                    "liquid_hidden_dim": int(liquid_hidden_dim),  
                    "lstm_params": int(lstm_params),  
                    "liquid_params": int(liquid_params),  
                    "absolute_param_gap": int(param_gap),  
                    "ratio_to_lstm": float(ratio_to_lstm),  
                }  
            )  
    except Exception as exc:  
        return {  
            "status": "unavailable",  
            "reason": "param_probe_failed",  
            "comparison_basis": "parameter_count",  
            "lstm_hidden_dim_candidates": lstm_hidden_dims,  
            "liquid_hidden_dim_candidates": liquid_hidden_dims,  
            "error": f"{type(exc).__name__}: {exc}",  
        }  

    return {  
        "status": "ok",  
        "comparison_basis": "parameter_count",  
        "lstm_hidden_dim_candidates": lstm_hidden_dims,  
        "liquid_hidden_dim_candidates": liquid_hidden_dims,  
        "pairs": pairs,  
        "max_absolute_param_gap": max((pair["absolute_param_gap"] for pair in pairs), default=0),  
        "max_ratio_deviation": max((abs(pair["ratio_to_lstm"] - 1.0) for pair in pairs), default=0.0),  
    }  


def _build_search_budget_summary(  
    classical_search_result: dict[str, Any],  
    neural_search_results: dict[str, Any],  
) -> dict[str, Any]:  
    classical_audit = classical_search_result["search_audit"]  
    lstm_audit = neural_search_results["lstm_ekf"]["search_audit"]  
    liquid_audit = neural_search_results["liquid_ekf"]["search_audit"]  

    def _checkpoint_candidate_count(report: dict[str, Any]) -> int:
        checkpoint_selection = dict(report.get("checkpoint_selection") or {})
        candidate_count = checkpoint_selection.get("candidate_count")
        if candidate_count is None:
            return 0
        return int(candidate_count)

    def _collect_unique_neural_job_reports(audit: dict[str, Any]) -> list[dict[str, Any]]:
        unique_reports: list[dict[str, Any]] = []
        seen_pairs: set[tuple[str, int]] = set()

        def _append_report(report: dict[str, Any]) -> None:
            signature = str(report.get("signature") or "").strip()
            if not signature:
                raise ValueError("neural search report is missing signature")
            seed_value = int(report.get("seed", dict(report.get("overrides") or {}).get("train.seed", 0)))
            report_key = (signature, seed_value)
            if report_key in seen_pairs:
                return
            seen_pairs.add(report_key)
            unique_reports.append(report)

        for report in list(audit.get("grid_reports") or []):
            _append_report(report)
        for aggregated_report in list(audit.get("multiseed_reports") or []):
            for seed_report in list(aggregated_report.get("seed_reports") or []):
                _append_report(seed_report)
        return unique_reports

    def _classical_summary(method_name: str) -> dict[str, Any]:  
        reports = list(classical_audit[method_name]["grid_reports"])  
        per_candidate_scoring_runs = len(_search_scoring_config_names(method_name, include_safe=(method_name != "ekf")))  
        extra_selected_safe_runs = 1 if method_name == "ekf" else 0  
        scoring_run_count = len(reports) * per_candidate_scoring_runs  
        return {  
            "candidate_count": len(reports),  
            "per_candidate_train_runs": 0,  
            "search_train_run_count": 0,  
            "per_candidate_scoring_runs": per_candidate_scoring_runs,  
            "search_scoring_run_count": int(scoring_run_count + extra_selected_safe_runs),  
            "final_selected_retrain_count": 0,  
            "final_selected_safe_rescore_count": extra_selected_safe_runs,  
            "total_search_runs": int(scoring_run_count + extra_selected_safe_runs),  
        }  

    def _neural_summary(method_name: str, audit: dict[str, Any]) -> dict[str, Any]:  
        grid_reports = list(audit["grid_reports"])  
        unique_job_reports = _collect_unique_neural_job_reports(audit)
        top_k_multiseed = int(audit["top_k_multiseed"])  
        multiseed_values = list(audit["multiseed_values"])  
        top_candidate_count = min(top_k_multiseed, len(grid_reports))  
        multiseed_seed_count = len(multiseed_values)  
        scoring_runs_per_candidate = len(_search_scoring_config_names(method_name))  
        grid_signatures = {str(report.get("signature") or "") for report in grid_reports}  
        selected_top_reports = list(audit.get("top_reports") or [])[:top_candidate_count]  
        unique_multiseed_train_jobs = 0  
        for top_report in selected_top_reports:  
            signature = str(top_report.get("signature") or "")  
            if not signature:  
                raise ValueError(f"{method_name} top report is missing signature")  
            for seed in multiseed_values:  
                if int(seed) == 0 and signature in grid_signatures:  
                    continue  
                unique_multiseed_train_jobs += 1  
        search_train_run_count = len(grid_reports) + unique_multiseed_train_jobs  
        candidate_scoring_run_count = search_train_run_count * scoring_runs_per_candidate  
        checkpoint_selection_candidate_count = sum(
            _checkpoint_candidate_count(report)
            for report in unique_job_reports
        )
        checkpoint_selection_scoring_run_count = checkpoint_selection_candidate_count * scoring_runs_per_candidate
        final_checkpoint_selection = dict(audit.get("final_checkpoint_selection") or {})
        final_selected_checkpoint_candidate_count = int(final_checkpoint_selection.get("candidate_count") or 0)
        final_selected_checkpoint_selection_scoring_run_count = (
            final_selected_checkpoint_candidate_count * scoring_runs_per_candidate
        )
        search_scoring_run_count = candidate_scoring_run_count + checkpoint_selection_scoring_run_count
        checkpoint_rescoring_run_count = (
            checkpoint_selection_scoring_run_count + final_selected_checkpoint_selection_scoring_run_count
        )
        total_neural_budget_run_count = (
            search_train_run_count
            + search_scoring_run_count
            + 1
            + final_selected_checkpoint_selection_scoring_run_count
        )
        return {  
            "candidate_count": len(grid_reports),  
            "top_k_multiseed": top_k_multiseed,  
            "top_candidate_count_for_multiseed": top_candidate_count,  
            "multiseed_seed_values": multiseed_values,  
            "multiseed_candidate_count": top_candidate_count,  
            "unique_multiseed_train_job_count": int(unique_multiseed_train_jobs),  
            "search_train_run_count": int(search_train_run_count),  
            "candidate_scoring_run_count": int(candidate_scoring_run_count),  
            "checkpoint_selection_candidate_count": int(checkpoint_selection_candidate_count),  
            "checkpoint_selection_scoring_run_count": int(checkpoint_selection_scoring_run_count),  
            "search_scoring_run_count": int(search_scoring_run_count),  
            "per_candidate_scoring_runs": scoring_runs_per_candidate,  
            "final_selected_retrain_count": 1,  
            "final_selected_checkpoint_candidate_count": final_selected_checkpoint_candidate_count,  
            "final_selected_checkpoint_selection_scoring_run_count": int(final_selected_checkpoint_selection_scoring_run_count),  
            "checkpoint_rescoring_run_count": int(checkpoint_rescoring_run_count),  
            "checkpoint_rescoring_share_of_total_budget_runs": float(  
                checkpoint_rescoring_run_count / total_neural_budget_run_count
            ) if total_neural_budget_run_count > 0 else 0.0,
            "dominant_budget_driver": (  
                "checkpoint_rescoring"
                if checkpoint_rescoring_run_count > candidate_scoring_run_count
                else "candidate_scoring"
            ),
            "total_budgeted_run_count": int(total_neural_budget_run_count),  
            "selected_seed_policy": audit["selected_seed_policy"],  
            "epoch_candidate_stride": audit.get("epoch_candidate_stride"),  
            "resolved_training_budget": dict(audit.get("resolved_training_budget") or {}),
            "total_search_runs": int(search_train_run_count + search_scoring_run_count),  
        }  

    return {  
        "budget_counting_notes": [  
            "candidate_count counts checkpoint candidates produced per search report before any later top-k filtering.",
            "search_train_run_count counts search-time training runs across all candidate configurations and expanded seeds.",
            "candidate_scoring_run_count counts downstream candidate scoring runs, including checkpoint rescoring for e2/e3/e9/e0_safe style flows.",
            "checkpoint_selection_scoring_run_count counts extra epoch-checkpoint rescoring runs used to pick the final checkpoint within a selected candidate.",
            "search_scoring_run_count equals candidate_scoring_run_count plus checkpoint_selection_scoring_run_count.",
            "final_selected_retrain_count counts retraining runs for the finally selected candidate after search-time selection is complete.",
            "final_selected_checkpoint_selection_scoring_run_count counts epoch-checkpoint rescoring runs for the final selected retrain job.",
        ],  
        "ekf": _classical_summary("ekf"),  
        "robust_ekf": _classical_summary("robust_ekf"),  
        "fgo": _classical_summary("fgo"),  
        "lstm_ekf": _neural_summary("lstm_ekf", lstm_audit),  
        "liquid_ekf": _neural_summary("liquid_ekf", liquid_audit),  
    }  


def _measure_runtime_reference_span_minutes(reference_root: Path) -> float | None:
    """Measure the runtime span of reference artifacts in minutes."""
    if not reference_root.is_dir():
        return None
    file_paths = [path for path in reference_root.rglob("*") if path.is_file()]
    if not file_paths:
        return None
    ordered_files = sorted(file_paths, key=lambda path: path.stat().st_mtime)
    first_mtime = ordered_files[0].stat().st_mtime
    last_mtime = ordered_files[-1].stat().st_mtime
    total_minutes = (last_mtime - first_mtime) / 60.0
    if total_minutes <= 0.0:
        return None
    return float(total_minutes)


def _safe_float(value: Any) -> float | None:
    """Convert a value to float when possible, else return None."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_neural_runtime_reference_catalog() -> dict[str, Any]:
    """Build the neural runtime reference catalog from existing artifacts."""
    candidate_search_root = (
        ROOT
        / "outputs"
        / "paper_sim_only_fixcheck_20260530"
        / "train"
        / "lstm_ekf"
        / "search"
        / "lstm_ekf"
        / "lstm_ekf__w20__h64__lr0.001__s0__203f0380102dfe35"
    )
    final_selected_root = ROOT / "outputs" / "paper_sim_only_fixcheck_20260530" / "train" / "lstm_ekf" / "final_selected"
    candidate_report_path = candidate_search_root / "candidate_report.json"
    final_train_report_path = final_selected_root / "reports" / "lstm_ekf_train_report.json"
    references: list[dict[str, Any]] = []
    candidate_report: dict[str, Any] = {}

    if candidate_report_path.is_file():
        candidate_report = dict(_read_json(candidate_report_path) or {})
        train_report = dict(candidate_report.get("train_report") or {})
        checkpoint_selection = dict(candidate_report.get("checkpoint_selection") or {})
        span_minutes = _measure_runtime_reference_span_minutes(candidate_search_root)
        train_epochs = int(train_report.get("epochs") or 0)
        checkpoint_candidate_count = int(checkpoint_selection.get("candidate_count") or 0)
        scoring_runs_per_candidate = len(_search_scoring_config_names("lstm_ekf"))
        total_run_units = 1 + scoring_runs_per_candidate + checkpoint_candidate_count * scoring_runs_per_candidate
        references.append(
            {
                "reference_name": "lstm_historical_search_candidate_32epoch_with_checkpoint_selection",
                "model_name": "lstm_ekf",
                "reference_kind": "search_candidate",
                "path": str(candidate_search_root.resolve()),
                "observed_total_minutes": span_minutes,
                "train_epochs": train_epochs,
                "train_window_count": int(train_report.get("train_window_count") or 0),
                "val_window_count": int(train_report.get("val_window_count") or 0),
                "checkpoint_candidate_count": checkpoint_candidate_count,
                "scoring_runs_per_candidate": scoring_runs_per_candidate,
                "total_runtime_run_units": total_run_units,
                "minutes_per_total_unit": (
                    span_minutes / float(total_run_units)
                    if span_minutes is not None and total_run_units > 0
                    else None
                ),
                "minutes_per_epoch": (
                    span_minutes / float(train_epochs)
                    if span_minutes is not None and train_epochs > 0
                    else None
                ),
                "source_report": str(candidate_report_path.resolve()),
            }
        )

    if final_train_report_path.is_file():
        final_train_report = dict(_read_json(final_train_report_path) or {})
        final_checkpoint_candidate_count = len(list(final_train_report.get("epoch_candidate_paths") or []))
        if not final_checkpoint_candidate_count:
            final_checkpoint_candidate_count = int(
                dict(candidate_report.get("checkpoint_selection") or {}).get("candidate_count") or 0
            )
        span_minutes = _measure_runtime_reference_span_minutes(final_selected_root)
        train_epochs = int(final_train_report.get("epochs") or 0)
        scoring_runs_per_candidate = len(_search_scoring_config_names("lstm_ekf"))
        total_run_units = 1 + final_checkpoint_candidate_count * scoring_runs_per_candidate
        references.append(
            {
                "reference_name": "lstm_historical_final_selected_32epoch_with_checkpoint_selection",
                "model_name": "lstm_ekf",
                "reference_kind": "final_selected",
                "path": str(final_selected_root.resolve()),
                "observed_total_minutes": span_minutes,
                "train_epochs": train_epochs,
                "train_window_count": int(final_train_report.get("train_window_count") or 0),
                "val_window_count": int(final_train_report.get("val_window_count") or 0),
                "checkpoint_candidate_count": int(final_checkpoint_candidate_count),
                "scoring_runs_per_candidate": scoring_runs_per_candidate,
                "total_runtime_run_units": total_run_units,
                "minutes_per_total_unit": (
                    span_minutes / float(total_run_units)
                    if span_minutes is not None and total_run_units > 0
                    else None
                ),
                "minutes_per_epoch": (
                    span_minutes / float(train_epochs)
                    if span_minutes is not None and train_epochs > 0
                    else None
                ),
                "source_report": str(final_train_report_path.resolve()),
            }
        )

    return {
        "status": "ok" if references else "missing_reference",
        "reference_count": len(references),
        "references": references,
    }


def _project_neural_runtime_budget(
    model_name: str,
    budget_summary: dict[str, Any],
    *,
    reference_catalog: dict[str, Any],
) -> dict[str, Any]:
    """Project runtime for a single model from budget and reference artifacts."""
    model_budget = dict(budget_summary.get(model_name) or {})
    if not model_budget:
        return {"status": "missing_budget", "model_name": model_name}

    scoring_runs_per_candidate = int(model_budget.get("per_candidate_scoring_runs") or 0)
    total_budgeted_run_count = int(model_budget.get("total_budgeted_run_count") or 0)
    search_candidate_runs = (
        int(model_budget.get("search_train_run_count") or 0)
        + int(model_budget.get("candidate_scoring_run_count") or 0)
        + int(model_budget.get("checkpoint_selection_scoring_run_count") or 0)
    )
    final_selected_runs = (
        int(model_budget.get("final_selected_retrain_count") or 0)
        + int(model_budget.get("final_selected_checkpoint_selection_scoring_run_count") or 0)
    )
    references = [
        dict(reference_row)
        for reference_row in list(reference_catalog.get("references") or [])
        if str(reference_row.get("model_name") or "") in {model_name, "lstm_ekf"}
    ]
    search_reference = next(
        (
            reference
            for reference in references
            if str(reference.get("reference_kind") or "") == "search_candidate"
            and _safe_float(reference.get("minutes_per_total_unit")) is not None
        ),
        None,
    )
    final_reference = next(
        (
            reference
            for reference in references
            if str(reference.get("reference_kind") or "") == "final_selected"
            and _safe_float(reference.get("minutes_per_total_unit")) is not None
        ),
        None,
    )
    training_reference = next(
        (
            reference
            for reference in references
            if str(reference.get("reference_kind") or "") == "full_training"
            and _safe_float(reference.get("minutes_per_epoch")) is not None
        ),
        None,
    )
    estimated_search_minutes = None
    estimated_final_selected_minutes = None
    estimated_total_minutes = None
    total_hours_lower = None
    total_hours_upper = None
    notes: list[str] = []
    estimation_status = "budget_only"

    if search_reference is not None:
        search_unit_minutes = float(search_reference["minutes_per_total_unit"])
        estimated_search_minutes = search_unit_minutes * float(search_candidate_runs)
        notes.append(
            "search projection uses observed per-unit minutes from a real 32-epoch LSTM search candidate with checkpoint rescoring"
        )
    if final_reference is not None:
        final_unit_minutes = float(final_reference["minutes_per_total_unit"])
        estimated_final_selected_minutes = final_unit_minutes * float(final_selected_runs)
        notes.append(
            "final-selected projection uses observed per-unit minutes from a real 32-epoch final_selected retrain with checkpoint rescoring"
        )
    elif training_reference is not None and scoring_runs_per_candidate > 0:
        train_epochs = max(int(training_reference.get("train_epochs") or 0), 1)
        final_checkpoint_candidates = int(model_budget.get("final_selected_checkpoint_candidate_count") or 0)
        final_unit_count = 1 + final_checkpoint_candidates * scoring_runs_per_candidate
        training_minutes = float(training_reference["minutes_per_epoch"]) * float(train_epochs)
        estimated_final_selected_minutes = training_minutes * float(final_unit_count)
        notes.append(
            "final-selected projection falls back to extrapolating a historical training-only 256-epoch reference, so this branch is conservative and weakly grounded"
        )
    if estimated_search_minutes is not None and estimated_final_selected_minutes is not None:
        estimated_total_minutes = estimated_search_minutes + estimated_final_selected_minutes
        total_hours_lower = estimated_total_minutes / 60.0
        total_hours_upper = total_hours_lower * 1.25
        estimation_status = "estimated_from_observed_history"
        notes.append("upper bound adds a 25% buffer for model-specific and IO variance")
    elif estimated_search_minutes is not None:
        estimated_total_minutes = estimated_search_minutes
        total_hours_lower = estimated_total_minutes / 60.0
        total_hours_upper = total_hours_lower * 1.5
        estimation_status = "partial_estimate"
        notes.append("only the search branch is grounded by observed history; final_selected time is missing")
    if model_name != "lstm_ekf":
        notes.append("liquid_ekf currently reuses LSTM historical runtime references because no comparable liquid runtime artifact was found")

    return {
        "status": "ok",
        "model_name": model_name,
        "estimation_status": estimation_status,
        "total_budgeted_run_count": total_budgeted_run_count,
        "search_candidate_runs": search_candidate_runs,
        "final_selected_runs": final_selected_runs,
        "estimated_search_minutes": estimated_search_minutes,
        "estimated_final_selected_minutes": estimated_final_selected_minutes,
        "estimated_total_minutes": estimated_total_minutes,
        "estimated_total_hours_lower": total_hours_lower,
        "estimated_total_hours_upper": total_hours_upper,
        "observed_reference_runs": references,
        "notes": notes,
    }


def _build_neural_runtime_projection(budget_summary: dict[str, Any]) -> dict[str, Any]:
    """Build the aggregate neural runtime projection audit."""
    reference_catalog = _build_neural_runtime_reference_catalog()
    return {
        "status": "ok" if str(reference_catalog.get("status") or "") == "ok" else "budget_only",
        "reference_catalog_status": reference_catalog.get("status"),
        "reference_count": int(reference_catalog.get("reference_count") or 0),
        "references": list(reference_catalog.get("references") or []),
        "models": {
            "lstm_ekf": _project_neural_runtime_budget(
                "lstm_ekf",
                budget_summary,
                reference_catalog=reference_catalog,
            ),
            "liquid_ekf": _project_neural_runtime_budget(
                "liquid_ekf",
                budget_summary,
                reference_catalog=reference_catalog,
            ),
        },
    }


def _build_epoch_candidate_policy_audit(neural_search_surface: dict[str, Any]) -> dict[str, Any]:
    """Audit epoch-candidate policy semantics for the chosen search surface."""
    profile_name = str(neural_search_surface.get("profile_name") or "")
    checkpoint_selection_mode = str(neural_search_surface.get("checkpoint_selection_mode") or "downstream_val_rescore")
    requested_stride = neural_search_surface.get("epoch_candidate_stride")
    lstm_model_cfg = _load_yaml_config_cached(str((ROOT / "configs" / "models" / "lstm_ekf.yaml").resolve()))
    liquid_model_cfg = _load_yaml_config_cached(str((ROOT / "configs" / "models" / "liquid_ekf.yaml").resolve()))
    lstm_model_cfg_stride = dict(lstm_model_cfg.get("train") or {}).get("epoch_candidate_stride")
    liquid_model_cfg_stride = dict(liquid_model_cfg.get("train") or {}).get("epoch_candidate_stride")
    if checkpoint_selection_mode == "trainer_validation_best":
        resolved_stride = None
        resolution_rule = "paper profile reuses the trainer-selected validation-best checkpoint and does not add a second orchestration-level checkpoint rescoring layer"
    elif requested_stride is None:
        resolved_stride = 1
        resolution_rule = "profile omits explicit override and both trainers default train.epoch_candidate_stride to 1"
    else:
        resolved_stride = int(requested_stride)
        resolution_rule = "search surface explicitly overrides train.epoch_candidate_stride"
    return {
        "status": "ok",
        "profile_name": profile_name,
        "trainers": {
            "lstm_ekf": {
                "trainer_default_stride": 1,
                "trainer_default_stride_source": str((ROOT / "src" / "liquidloc" / "models" / "lstm" / "trainer.py").resolve()),
                "save_epoch_candidates_default": False,
            },
            "liquid_ekf": {
                "trainer_default_stride": 1,
                "trainer_default_stride_source": str((ROOT / "src" / "liquidloc" / "models" / "liquid" / "trainer.py").resolve()),
                "save_epoch_candidates_default": False,
            },
        },
        "orchestration_policy": {
            "search_forces_save_epoch_candidates": checkpoint_selection_mode == "downstream_val_rescore",
            "final_selected_retrain_forces_save_epoch_candidates": checkpoint_selection_mode == "downstream_val_rescore",
            "checkpoint_selection_mode": checkpoint_selection_mode,
            "requested_epoch_candidate_stride": requested_stride,
            "resolved_epoch_candidate_stride": resolved_stride,
            "resolution_rule": resolution_rule,
        },
        "config_surface": {
            "lstm_model_cfg_declares_epoch_candidate_stride": lstm_model_cfg_stride is not None,
            "liquid_model_cfg_declares_epoch_candidate_stride": liquid_model_cfg_stride is not None,
            "lstm_model_cfg_epoch_candidate_stride": int(lstm_model_cfg_stride) if lstm_model_cfg_stride is not None else None,
            "liquid_model_cfg_epoch_candidate_stride": int(liquid_model_cfg_stride) if liquid_model_cfg_stride is not None else None,
            "lstm_model_cfg_source": str((ROOT / "configs" / "models" / "lstm_ekf.yaml").resolve()),
            "liquid_model_cfg_source": str((ROOT / "configs" / "models" / "liquid_ekf.yaml").resolve()),
        },
        "fairness_implication": {
            "paper_profile_rescores_every_exported_epoch_candidate": checkpoint_selection_mode == "downstream_val_rescore" and requested_stride is None,
            "budget_growth_driver": (
                "checkpoint selection cost is protocol-defined by orchestration plus trainer default stride, not by hidden model-config drift"
                if checkpoint_selection_mode == "downstream_val_rescore"
                else "paper profile uses the trainer-selected validation-best checkpoint directly and avoids the second heavy checkpoint rescoring layer"
            ),
        },
    }


def _build_neural_checkpoint_selection_audit(neural_search_surface: dict[str, Any]) -> dict[str, Any]:
    """Audit checkpoint-selection semantics for the chosen search surface."""
    checkpoint_selection_mode = str(neural_search_surface.get("checkpoint_selection_mode") or "downstream_val_rescore")
    return {
        "status": "ok",
        "trainer_level_selection": {
            "enabled": True,
            "selection_surface": "val windows only",
            "selection_signal": "weighted validation selection_score with val_loss fallback",
            "best_checkpoint_written_during_training": True,
            "lstm_trainer_source": str((ROOT / "src" / "liquidloc" / "models" / "lstm" / "trainer.py").resolve()),
            "liquid_trainer_source": str((ROOT / "src" / "liquidloc" / "models" / "liquid" / "trainer.py").resolve()),
        },
        "paper_level_selection": {
            "enabled": checkpoint_selection_mode == "downstream_val_rescore",
            "selection_mode": checkpoint_selection_mode,
            "selection_surface": (
                ["e2_async", "e3_nlos", "e9_dual_degradation", "e0_safe_mode"]
                if checkpoint_selection_mode == "downstream_val_rescore"
                else ["trainer_validation_best"]
            ),
            "selection_signal": (
                "downstream localization score vector on val_ids across async/nlos/dual_degradation/safe"
                if checkpoint_selection_mode == "downstream_val_rescore"
                else "reuse trainer-selected validation-best checkpoint without a second downstream rescoring layer"
            ),
            "selection_scope": (
                "all exported epoch candidates plus best checkpoint if missing"
                if checkpoint_selection_mode == "downstream_val_rescore"
                else "trainer-selected best checkpoint only"
            ),
            "selection_code_source": "scripts/20_run_paper_experiments.py::_select_final_neural_checkpoint",
        },
        "implication": {
            "selection_layers": 2 if checkpoint_selection_mode == "downstream_val_rescore" else 1,
            "paper_level_is_additional_to_trainer_level": checkpoint_selection_mode == "downstream_val_rescore",
            "notes": [
                "trainer already chooses a best checkpoint during training",
                (
                    "paper script then re-ranks exported checkpoints with downstream localization metrics on the same val_ids"
                    if checkpoint_selection_mode == "downstream_val_rescore"
                    else "paper profile reuses the trainer-selected validation-best checkpoint directly"
                ),
                (
                    "the second layer expands validation-time compute substantially even though it does not touch held-out test_ids"
                    if checkpoint_selection_mode == "downstream_val_rescore"
                    else "this removes the second heavy validation-time rescoring layer from the paper profile"
                ),
            ],
        },
    }


def _iter_ekf_override_grid() -> list[dict[str, Any]]:  
    override_rows: list[dict[str, Any]] = []  
    for uwb_noise, vio_pos, vio_yaw in itertools.product(  
        _FAIR_EKF_GRID["uwb_noise"],  
        _FAIR_EKF_GRID["vio_pos"],  
        _FAIR_EKF_GRID["vio_yaw"],  
    ):  
        override_rows.append(  
            {  
                "measurement_noise.uwb": float(uwb_noise),  
                "measurement_noise.vio.pos": float(vio_pos),  
                "measurement_noise.vio.yaw": float(vio_yaw),  
            }  
        )  
    return override_rows  


def _iter_robust_override_grid(top_ekf_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:  
    override_rows: list[dict[str, Any]] = []  
    for ekf_report in top_ekf_reports:  
        base_noise_overrides = {  
            key: value  
            for key, value in dict(ekf_report["overrides"]).items()  
            if str(key).startswith("measurement_noise.")  
        }  
        for delta, quality_floor in itertools.product(  
            _FAIR_ROBUST_GRID["delta"],  
            _FAIR_ROBUST_GRID["quality_floor"],  
        ):  
            override_rows.append(  
                {  
                    **base_noise_overrides,  
                    "robust_weight.delta": float(delta),  
                    "gate.quality_floor": float(quality_floor),  
                }  
            )  
    return override_rows  


def _iter_fgo_override_grid(top_ekf_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:  
    override_rows: list[dict[str, Any]] = []  
    for ekf_report in top_ekf_reports:  
        base_noise_overrides = {  
            key: value  
            for key, value in dict(ekf_report["overrides"]).items()  
            if str(key).startswith("measurement_noise.")  
        }  
        for window_size, max_iters in itertools.product(  
            _FAIR_FGO_GRID["window_size"],  
            _FAIR_FGO_GRID["max_iters"],  
        ):  
            override_rows.append(  
                {  
                    **base_noise_overrides,  
                    "window_size": int(window_size),  
                    "optimizer.max_iters": int(max_iters),  
                }  
            )  
    return override_rows  


def _classical_candidate_signature(method_name: str, overrides: dict[str, Any]) -> str:  
    ordered_items = sorted((str(key), value) for key, value in overrides.items())  
    flattened = "__".join(f"{key}={value}" for key, value in ordered_items)  
    digest = hashlib.sha1(f"{method_name}__{flattened}".encode("utf-8")).hexdigest()[:16]  
    if method_name == "ekf":  
        uwb_noise = overrides.get("measurement_noise.uwb", "na")  
        vio_pos = overrides.get("measurement_noise.vio.pos", "na")  
        vio_yaw = overrides.get("measurement_noise.vio.yaw", "na")  
        return f"{method_name}__uwb{uwb_noise}__vp{vio_pos}__vy{vio_yaw}__{digest}"  
    if method_name == "robust_ekf":  
        delta = overrides.get("robust_weight.delta", "na")  
        quality_floor = overrides.get("gate.quality_floor", "na")  
        return f"{method_name}__d{delta}__q{quality_floor}__{digest}"  
    if method_name == "fgo":  
        window_size = overrides.get("window_size", "na")  
        max_iters = overrides.get("optimizer.max_iters", "na")  
        return f"{method_name}__w{window_size}__it{max_iters}__{digest}"  
    return f"{method_name}__{digest}"  


def _run_neural_search_candidate(  
    *,  
    model_name: str,  
    overrides: dict[str, Any],  
    seed: int,  
    prepare_root: Path,  
    raw_root: Path,  
    split_manifest: dict[str, Any],  
    search_root: Path,  
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
    scoring_config_names = _search_scoring_config_names(model_name)  
    method_name = model_name
    include_safe = _SAFE_SCORING_EXPERIMENT in scoring_config_names
    baseline_method_name = "ekf"
    candidate_signature = _search_candidate_signature(model_name, overrides, seed)  
    candidate_root = search_root / model_name / candidate_signature  
    report_path = candidate_root / "candidate_report.json"  
    cache_context = {  
        "model_name": model_name,  
        "seed": int(seed),  
        "overrides": dict(overrides),  
        "prepare_root": str(prepare_root.resolve()),  
        "raw_root": str(raw_root.resolve()),  
        "train_ids": list(split_manifest["train_ids"]),  
        "val_ids": list(split_manifest["val_ids"]),  
        "estimator_cfgs": estimator_cfgs,  
        "epochs": int(epochs),  
        "batch_size": int(batch_size),  
        "eval_batch_size": int(eval_batch_size),  
        "search_experiments": list(scoring_config_names),  
        "baseline_scoring_metrics_by_experiment": _normalize_baseline_scoring_metrics_for_cache(
            baseline_scoring_metrics_by_experiment
        ),  
        "checkpoint_selection_mode": str(checkpoint_selection_mode),  
        "epoch_candidate_stride": int(epoch_candidate_stride) if epoch_candidate_stride is not None else None,  
        "implementation_fingerprint": _resolve_search_implementation_fingerprint(  
            search_kind="neural",  
            implementation_name=model_name,  
            experiment_config_names=scoring_config_names,  
        ),  
    }  
    cached_report = _load_cached_search_report(report_path, cache_context=cache_context) if allow_cache else None  
    if cached_report is not None:  
        return cached_report  

    train_overrides = dict(overrides)  
    train_overrides["train.seed"] = int(seed)  
    if checkpoint_selection_mode == "downstream_val_rescore":  
        train_overrides["train.save_epoch_candidates"] = True  
    if checkpoint_selection_mode == "downstream_val_rescore" and epoch_candidate_stride is not None:  
        train_overrides["train.epoch_candidate_stride"] = int(epoch_candidate_stride)  
    train_result = _run_frontend_training(  
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
    checkpoint_selection = _select_final_neural_checkpoint(  
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
    train_report = dict(train_result.metadata["train_report"])  
    sample_report = dict(train_result.metadata.get("sample_report") or {})  
    training_flow_contract = dict(train_result.metadata.get("training_flow_contract") or {})  
    model_cfg = dict(checkpoint_selection["selected_model_cfg"])  
    scoring_seq_ids = list(split_manifest["val_ids"])  
    if not scoring_seq_ids:  
        raise RuntimeError(f"{model_name} search requires non-empty val_ids for fair scoring")  
    scoring_runs = [  
        _run_scoring_experiment(  
            config_name=config_name,  
            prepare_root=prepare_root,  
            raw_root=raw_root,  
            method_name=model_name,  
            estimator_cfgs=estimator_cfgs,  
            model_cfgs={model_name: model_cfg},  
            output_root=candidate_root / "scoring_runs",  
            seq_ids=scoring_seq_ids,  
            baseline_metrics=(  
                dict(baseline_scoring_metrics_by_experiment[Path(config_name).stem])  
                if baseline_scoring_metrics_by_experiment is not None  
                and Path(config_name).stem in baseline_scoring_metrics_by_experiment  
                else None  
            ),  
        )  
        for config_name in scoring_config_names  
    ]  
    metrics_by_experiment = {
        run["experiment_id"]: dict(run["metrics"])
        for run in scoring_runs
    }
    safe_metrics = metrics_by_experiment.get("e0_safe_mode")
    if safe_metrics is None and baseline_scoring_metrics_by_experiment is not None:
        safe_metrics = baseline_scoring_metrics_by_experiment.get("e0_safe_mode")
    if safe_metrics is None:
        metrics_by_experiment = _inject_self_baseline_safe_metrics(metrics_by_experiment)
        safe_metrics = metrics_by_experiment["e0_safe_mode"]
    elif "e0_safe_mode" not in metrics_by_experiment:
        metrics_by_experiment = dict(metrics_by_experiment)
        metrics_by_experiment["e0_safe_mode"] = dict(safe_metrics)
    if method_name == baseline_method_name and not include_safe:
        metrics_by_experiment = _inject_self_baseline_safe_metrics(metrics_by_experiment)
        safe_metrics = metrics_by_experiment["e0_safe_mode"]
    safe_p95 = float(safe_metrics.get("p95", 0.0))
    safe_ekf_p95 = float(safe_metrics.get("ekf_p95", safe_p95))
    candidate_report = {
        "method_name": method_name,
        "signature": candidate_signature,
        "overrides": dict(overrides),
        "cache_context": cache_context,
        "estimator_cfgs": estimator_cfgs,
        "baseline_method_name": baseline_method_name,
        "safe_scoring_included": include_safe,
        "scoring_seq_ids": scoring_seq_ids,
        "scoring_runs": scoring_runs,
        "score_vector": _score_candidate_lexicographically(metrics_by_experiment=metrics_by_experiment),
        "safe_margin_vs_ekf": safe_p95 - safe_ekf_p95,
        "safe_relative_margin": (safe_p95 - safe_ekf_p95) / max(safe_ekf_p95, 1e-9),
    }
    _write_json(report_path, candidate_report)
    return candidate_report  


def _select_final_neural_checkpoint(  
    *,  
    model_name: str,  
    train_result: Any,  
    prepare_root: Path,  
    raw_root: Path,  
    split_manifest: dict[str, Any],  
    estimator_cfgs: dict[str, dict[str, Any]],  
    selection_root: Path,  
    checkpoint_selection_mode: str = "downstream_val_rescore",  
    baseline_scoring_metrics_by_experiment: dict[str, dict[str, float]] | None = None,  
) -> dict[str, Any]:  
    train_report = dict(train_result.metadata["train_report"])  
    inference_device = _resolve_cuda_inference_device_from_train_report(train_report)  
    best_checkpoint_path = str(train_report.get("checkpoint_path") or "").strip()  
    if not best_checkpoint_path:  
        raise RuntimeError(f"{model_name} final checkpoint selection requires a non-empty checkpoint_path")  

    if checkpoint_selection_mode == "trainer_validation_best":  
        try:
            selected_payload = torch.load(best_checkpoint_path, map_location="cpu", weights_only=True)  
        except pickle.UnpicklingError as exc:
            raise RuntimeError(f"unsafe checkpoint: {best_checkpoint_path}") from exc
        selected_model_cfg = copy.deepcopy(  
            selected_payload.get("model_cfg")  
            or _load_yaml_config_cached(str((ROOT / "configs" / "models" / f"{model_name}.yaml").resolve()))  
        )  
        selected_model_cfg["checkpoint_path"] = best_checkpoint_path  
        selected_model_cfg["project_root"] = str(ROOT)  
        if inference_device is not None:  
            selected_model_cfg["inference_device"] = inference_device  
        selected_best_epoch = int(train_report.get("best_epoch") or selected_payload.get("best_epoch") or 0)  
        selection_report_path = str(  
            _write_json(  
                selection_root / "final_checkpoint_selection.json",  
                {  
                    "model_name": model_name,  
                    "selection_mode": "trainer_validation_best",  
                    "selected_checkpoint_path": best_checkpoint_path,  
                    "selected_best_epoch": selected_best_epoch,  
                    "candidate_count": 0,  
                    "candidate_reports": [],  
                    "trainer_best_selection_score": train_report.get("best_selection_score"),  
                    "notes": [  
                        "paper profile reuses the checkpoint already selected by the trainer on validation windows",  
                        "no second downstream rescoring layer is executed for epoch candidates",  
                    ],  
                },  
            )  
        )  
        train_report["paper_selected_checkpoint_path"] = best_checkpoint_path  
        train_report["paper_selected_best_epoch"] = selected_best_epoch  
        train_report["paper_selected_score_vector"] = []  
        train_report["paper_selected_metrics"] = {}  
        train_report["paper_checkpoint_selection_report"] = selection_report_path  
        train_report["checkpoint_path"] = best_checkpoint_path  
        train_report["best_ckpt"] = best_checkpoint_path  
        report_path = str(train_report.get("report_path") or "").strip()  
        if report_path:  
            from liquidloc.common.io_utils import dumps_json_text
            _write_json(Path(report_path), train_report)  
        train_result.metadata["train_report"] = train_report  
        return {  
            "train_result": train_result,  
            "selected_model_cfg": selected_model_cfg,  
            "selection_report": {  
                "selection_mode": "trainer_validation_best",  
                "selected_checkpoint_path": best_checkpoint_path,  
                "selected_best_epoch": selected_best_epoch,  
                "selected_score_vector": [],  
                "candidate_count": 0,  
                "selection_report_path": selection_report_path,  
            },  
        }  

    candidate_paths = [str(path) for path in list(train_report.get("epoch_candidate_paths") or []) if str(path).strip()]  
    if best_checkpoint_path and best_checkpoint_path not in candidate_paths:  
        candidate_paths.append(best_checkpoint_path)  
    if not candidate_paths:  
        raise RuntimeError(f"{model_name} final checkpoint selection requires at least one candidate checkpoint")  

    scoring_seq_ids = list(split_manifest["val_ids"])  
    if not scoring_seq_ids:  
        raise RuntimeError(f"{model_name} final checkpoint selection requires non-empty val_ids")  

    scoring_reports = []  
    for checkpoint_path in candidate_paths:  
        candidate_signature = Path(checkpoint_path).stem  
        candidate_root = selection_root / candidate_signature  
        model_cfg = _candidate_model_cfg_from_checkpoint(  
            model_name,  
            checkpoint_path,  
            inference_device=inference_device,  
        )  
        scoring_runs = [  
            _run_scoring_experiment(  
                config_name=config_name,  
                prepare_root=prepare_root,  
                raw_root=raw_root,  
                method_name=model_name,  
                estimator_cfgs=estimator_cfgs,  
                model_cfgs={model_name: model_cfg},  
                output_root=candidate_root,  
                seq_ids=scoring_seq_ids,  
                baseline_metrics=(  
                    dict(baseline_scoring_metrics_by_experiment[Path(config_name).stem])  
                    if baseline_scoring_metrics_by_experiment is not None  
                    and Path(config_name).stem in baseline_scoring_metrics_by_experiment  
                    else None  
                ),  
            )  
            for config_name in _search_scoring_config_names(model_name)  
        ]  
        metrics_by_experiment = {  
            run["experiment_id"]: dict(run["metrics"])  
            for run in scoring_runs  
        }  
        scoring_reports.append(  
            {  
                "checkpoint_path": checkpoint_path,  
                "best_epoch": int(  
                        _safe_load_checkpoint_best_epoch(checkpoint_path)  
                    ),  
                "score_vector": _score_candidate_lexicographically(metrics_by_experiment=metrics_by_experiment),  
                "selected_metrics": metrics_by_experiment,  
                "scoring_runs": scoring_runs,  
            }  
        )  

    scoring_reports.sort(key=lambda report: tuple(report["score_vector"]))  
    selected_checkpoint = scoring_reports[0]  
    selected_checkpoint_path = str(selected_checkpoint["checkpoint_path"])  
    try:
        selected_payload = torch.load(selected_checkpoint_path, map_location="cpu", weights_only=True)  
    except pickle.UnpicklingError as exc:
        raise RuntimeError(f"unsafe checkpoint: {selected_checkpoint_path}") from exc
    selected_model_cfg = copy.deepcopy(  
        selected_payload.get("model_cfg")  
        or _load_yaml_config_cached(str((ROOT / "configs" / "models" / f"{model_name}.yaml").resolve()))  
    )  
    selected_model_cfg["checkpoint_path"] = selected_checkpoint_path  
    selected_model_cfg["project_root"] = str(ROOT)  
    if inference_device is not None:  
        selected_model_cfg["inference_device"] = inference_device  

    train_report["paper_selected_checkpoint_path"] = selected_checkpoint_path  
    train_report["paper_selected_best_epoch"] = int(selected_checkpoint.get("best_epoch") or 0)  
    train_report["paper_selected_score_vector"] = list(selected_checkpoint["score_vector"])  
    train_report["paper_selected_metrics"] = selected_checkpoint["selected_metrics"]  
    train_report["paper_checkpoint_selection_report"] = str(  
        _write_json(  
            selection_root / "final_checkpoint_selection.json",  
            {  
                "model_name": model_name,  
                "scoring_seq_ids": scoring_seq_ids,  
                "candidate_reports": scoring_reports,  
                "selected_checkpoint_path": selected_checkpoint_path,  
                "selected_best_epoch": int(selected_checkpoint.get("best_epoch") or 0),  
                "selected_score_vector": list(selected_checkpoint["score_vector"]),  
            },  
        )  
    )  
    train_report["checkpoint_path"] = selected_checkpoint_path  
    train_report["best_ckpt"] = selected_checkpoint_path  
    report_path = str(train_report.get("report_path") or "").strip()  
    if report_path:  
        from liquidloc.common.io_utils import dumps_json_text
        _write_json(Path(report_path), train_report)  
    train_result.metadata["train_report"] = train_report  
    return {  
        "train_result": train_result,  
        "selected_model_cfg": selected_model_cfg,  
        "selection_report": {  
            "selected_checkpoint_path": selected_checkpoint_path,  
            "selected_best_epoch": int(selected_checkpoint.get("best_epoch") or 0),  
            "selected_score_vector": list(selected_checkpoint["score_vector"]),  
            "candidate_count": len(scoring_reports),  
            "selection_report_path": train_report["paper_checkpoint_selection_report"],  
        },  
    }  


def _aggregate_seed_reports(
    model_name: str,
    seed_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聫茅聢搂莽聞娄莽禄禄忙驴聥氓麓聮氓娄陇盲陆潞盲潞聶忙驴聽莽聰碌氓聛聻茅聧篓氓麓鹿氓篓虏莽聳戮?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿氓氓卢碌路茅聬聴聴脙鹿茅聰聺氓聽聼莽禄戮氓聣搂氓聨搂茅隆颅猫路篓忙聧聞忙赂職氓聣卤茅聳禄氓驴聯茅陋聫莽禄聽忙聮禄氓录禄茅聬聰莽垄卤忙聲聤茅聳赂氓驴聯氓聬聥茅聧聨忙聮禄忙垄潞莽聙鹿莽聥聙氓職聶茅聴聛氓赂露聛茅聬聸脦聼茅聯聫猫搂聞忙茅聴聛忙聬聻氓聞碌茅隆楼忙聮麓莽篓聣茅聨禄忙聨聠忙聰陆茅聳禄忙篓潞莽聛職茅聫聠芒聲聟忙聼聸莽聙拢芒聵聣茅聯禄莽聜麓莽鹿聸茅聧隆忙篓禄氓掳颅莽录聛莽聤卤氓聧鲁茅聫聦莽聠潞莽麓聮茅聤聫莽聤鲁莽聛聢莽录聛莽聜聞鲁茅隆聝陆氓录禄茅聬聰氓聟录氓聙路忙陇陇忙聨聧猫聝露莽禄聥氓聸漏莽麓聯忙碌拢莽篓驴茅聳赂氓卢篓氓聙聺茅陋聻氓聠篓茅聢搂氓鹿聝氓篓聠忙聮鲁莽茅聳鹿氓聯聞忙拢聶莽聙碌忙聽颅忙垄禄忙碌拢氓聭聤忙聝聢忙驴隆莽篓聮莽禄禄忙露垄氓聼聙茅隆聮氓聥卢莽聟聸莽聙鹿芒聜卢莽聙拢脨聹茅聴聛猫炉隆氓聳聮茅聫聟莽聤虏莽聲路茅聰聺氓聴聴莽路聻茅聬聨茂陆聨氓聛聞茅聢搂氓聟聵氓猫聶鹿忙聥聽茅聨禄忙聨聺氓聤聣茅聴聛莽聟聨莽露聤莽禄聽忙聽搂莽聲路猫陇掳忙聨聭莽拢录茅聳禄忙聞炉氓掳聶茅忙聞掳忙聼隆忙戮露氓漏聞氓垄聲氓脩聜莽虏聶茅陋聻氓聠聡聳茅聢潞氓聠漏茂录聫茅聴聛猫炉隆氓聻聨忙碌拢茅聲聬氓聢聮茅聴聜盲陆赂忙聬芦茅聬楼赂氓麓掳茅聫聧氓碌聛茅聳赂芒聞聝氓聡鹿氓娄虏猫聜漏莽麓聯忙碌拢莽篓驴芒聜卢氓聯楼莽虏聰茅聬聼茅聜娄氓職聲茅聬聽氓聸篓莽楼麓氓娄虏氓聽聼忙聟篓氓氓聽垄氓聢掳氓篓麓忙禄聢莽庐聳茅聫聦茫聞楼芒聰聭茅聧隆忙盲禄录氓茫聜聠氓赂聡茅聧篓氓陆聮氓聼聻茅聨麓茂赂陆氓聙路莽聙鹿氓聣虏氓聳聳氓篓聢猫聢碌莽鹿聺氓篓聢氓聻庐忙聻陇茅聳潞盲陆赂茅陋聻氓聴聸氓麓聼茅隆聳茂陆聠忙聶漏碌忙陇麓莽芒聳聯忙楼拢氓聠漏芒聮聭茅聴聜氓聽聼盲戮聴氓娄虏忙聮聼隆茅聧聸忙麓麓茅聳潞氓卤陆氓庐聲氓聽聲忙碌拢氓聯聞氓鹿聴茅聴聜盲陆赂忙聬芦茅聧聼氓麓聬茅聨录盲陆赂莽陇聣茅聬聨茂陆聨氓聳聴茅聬聯莽聤禄忙聼聸茅隆颅忙聢聺忙聻聼茅聳赂莽聤鲁脡聭茅聨卤茫聞娄莽鹿聺茅聧聸盲禄漏茅聴聛氓陆聮忙驴聻芒聜聳赂茫聞漏莽庐聳氓庐聲忙篓潞茅隆聰猫聤楼莽楼掳茅聴聜氓聜職氓聙路茅聬聮茂鹿聙茅聨赂氓潞拢忙驴聽氓漏聜忙聡聬莽禄聙氓漏聜芒聰聭茅聬聵氓聫聣忙聬聬茅聳潞氓聥漏氓聞碌茅聫?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    model_name : str
        茅聴聜氓聜職氓聙路氓篓麓氓聸搂莽聲路茅聬垄莽莽陋聴茅聳鹿茅聜娄氓聳聳茅聧篓忙驴聥氓鹿聳氓篓拢莽聛驴聻氓聭搂碌忙路聡氓漏聡氓娄露茅聧聸忙芦拢莽录聜盲陆潞氓搂鲁茅聧聴忙聤陆氓录禄氓篓聭忙篓路忙忙驴莽聣聤茅聫聛忙聝搂芒聰聭茅聬聬氓聫聣氓聶陆氓漏碌莽聜漏氓聛聯茅聴聛氓聯聞莽聼聣莽禄虏氓聙聼莽聛聮茅聴聛忙聝搂莽鹿聵茅聢搂莽篓聯忙陇鹿忙聤陆忙垄禄忙碌拢氓聭聤忙聠赂茅聳赂莽聤虏茅聟拢茅陋聻忙聽颅茅聰聰莽聲聦莽聲鲁茅聴聜氓聜聶莽聞娄莽聙碌脩聟忙聧垄茅聧聦忙忙露聵脦聰茅聧聸芒聜路潞芒聰聭茅聬聵忙聫聮莽禄戮忙聝搂氓聬聥茅聤聡氓聥炉氓录聧聦忙掳茅聴聛氓聯聞忙拢聧聯莽聠录氓录禄茅聰聸氓聥陋忙虏聳赂忙聮聴芒聜卢氓娄虏氓聽聺茅聤聢氓潞隆盲潞聺莽录聛猫炉虏氓聲麓氓炉聼楼氓麓录茅聫聡莽聝聠忙陆聤茅聴聛忙聦聨莽篓聭莽聙職茅聫聡茫聢隆芒聮聭茅聳赂忙聮麓氓陆聝忙碌聹莽聜潞忙聼聸茅聬聵忙聼楼氓聫聞茅聳鹿茂鹿垄忙驴隆莽聝陆氓聼隆茅聧聦忙露聵忙庐聶茅聴聜盲陆鹿氓莽禄禄氓聜聻脣聡忙碌录氓颅聵氓聛聜茅聳赂忙聞碌氓聳聴茅聬聯氓驴聯莽聮潞猫聶鹿氓垄聲茅聳赂忙聳禄氓聙聺茅聫聦茫聞楼芒聵聣氓娄陇鹿脩聡忙聼隆莽聛聻莽聲聦脨楼忙陇陇茂赂陆莽聟隆茅聫聦忙露聵脦聼忙赂職忙茅聴聛茅聙聻氓卤陆氓垄虏忙陇陇莽聟陇茅聼芦氓聸篓莽篓聭莽潞戮氓聺聴忙垄潞茅芒聜禄聣莽聰赂忙聜隆茅聫聡茫聢隆莽聟聶茅聳潞氓陇聤莽聛聺茅隆拢忙聤陆忙聼聸茅聨潞忙聮麓莽路虏茅聴?
    seed_reports : list[dict[str, Any]]
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬氓聫聣莽聳聞茅聴聜盲陆潞莽虏聧聰茂鹿聜氓麓聹氓篓聭茫聢隆芒聮聰茅聳赂忙聸篓氓聻卤茅聬聯忙露垄忙聼聸茅聨掳芒聲聛氓茅聳鲁脩聦氓聣聶茅隆鲁掳莽聲路茅聢楼忙聞聛忙陆漏茅聳录氓聯楼忙聲赂茅聧聵莽聰赂忙垄潞茅聧聸氓漏聝氓炉聶莽聙鹿茫聝楼忙麓拢莽聭聲氓聴聵莽鹿聺茅聬垄茅聺聸脨楼茅聳潞盲陆鹿忙聜聜氓庐聲忙聢聺芒聵聣氓娄禄聞莽篓聭茅聢禄氓潞篓氓鹿聵氓庐聲氓聸露氓聰露茅聴聜盲陆赂忙鹿卤茅聯聫氓潞篓氓麓掳茅聫聧氓漏碌莽聤鲁莽垄聧茅聬聯忙卢聯忙聼拢茅聨掳茅聺聸氓垄录聜氓露聠氓聻鹿莽禄卤忙聨聴氓麓聹忙碌拢茅聲聬莽虏露茅聴聛氓庐聽茅聧篓氓聺聴氓鹿聝茅聢潞氓聭茅聳鹿莽聝聵忙聦禄茅忙聢娄忙垄禄忙碌拢氓聭聤氓聲聻茅聳赂忙聢聺莽露聤氓庐聲忙聢娄氓麓篓茅忙露聵氓潞聤氓漏碌莽聤禄忙陆搂茅氓聭麓茂陆聻茅聧聸氓颅聵莽聟楼忙驴聽茅聺聸忙茅聧拢盲禄聥忙聼聸氓氓聴聴盲潞掳忙驴漏聜忙聬聟陇茅聬聽氓聸篓氓陆聝莽禄聽忙麓垄潞莽录聛忙篓禄氓掳聳赂氓漏聜忙鹿卤茅聨鹿茫聢聽芒聰聭茅聧楼芒聲聝莽聵聢茅聴聛莽忙聹禄聥忙聢聺芒聳聧猫陇聰氓氓聯聞脙潞莽聮聡忙聫聮莽茅聴聛猫忙聜聜莽禄聽忙聽麓潞茅聢搂茫聛聞茅聨潞氓聴聴盲潞戮莽录聜盲陆潞氓搂麓莽禄聣莽聭掳莽聲路茅聫聡茫聢隆莽聞录莽聙鹿茫聝楼忙聡聬茅聬陇氓聸垄潞茅聧娄忙驴聽茫聢隆莽禄庐氓漏職忙聬鹿茅聧聫氓盲潞戮忙莽聜虏莽聛聺茅聧聰忙掳露忙聼聼茅聧聬氓路搂赂茅聧聴茅聛聧忙聼聸茅莽聤虏氓聤聹茅聳禄忙聮麓莽聭漏茅聫聦茂陆聣氓鹿聡茅聴聢忙露聵忙驴聽芒聮聙氓卤陆莽聛娄茅聳潞氓卤戮莽篓聮茅聨炉忙聴聢氓茅聬聵氓聟赂忙隆聯莽录聜氓聜聶猫聝露莽禄聝聞茅隆聳忙掳卤忙職娄氓漏碌氓聜職忙聠隆茅聧聥氓聥芦莽聙拢氓聡鹿莽录聛猫戮篓莽鹿聝忙路聡氓漏聡氓娄露茅聧楼茫聝漏忙聼聸莽聙拢忙麓麓茅聳潞氓卤陆茂鹿聫聛忙聞掳氓麓卤茅聫聡莽聤碌莽聲戮茅聴聜盲陆潞莽虏炉茅聧篓莽聟聨脦聸茅聧聲忙職聹茅聴聜氓聜聶莽聞娄莽聙碌脩聝脙潞氓氓陇聤莽拢禄茅聳鹿氓聣搂莽虏聬聯忙露垄忙聼聸茅聢漏氓聠戮忙庐聵氓脩聧茅聶聞茅聨卤茫聞娄莽鹿聺茅聧聬芒聰驴盲禄垄茅聬聨猫搂聞忙麓聫茅聧聰氓露聡忙聧垄氓漏碌氓卢拢录忙驴聽氓漏聜忙聡聫茅聧陇氓聠篓芒聰聭茅聰聸氓聥芦盲潞录茅聳赂氓漏聝氓聻驴氓庐聲忙驴聠莽聲聺茅聧聲氓漏聜忙麓颅氓庐聲氓陆聮茅聰聸氓聥颅莽聲戮茅聴聜盲陆赂忙鹿卤茅聯聫氓潞篓氓麓掳茅聫聧莽聼聠茅聳赂忙聞碌氓聳聴茅聬聯忙露垄忙聼聸氓芒聜禄聙盲陆鹿忙聶露茅聨戮莽聟聶忙陇陇忙聴聙忙聻禄茅聭掳氓聺聴忙聼聼茅隆聰莽聲聦忙聡聡忙莽聜麓忙聧聡茅陋聻忙聨聴氓麓卤氓娄炉猫聶鹿忙搂潞忙驴聽莽聰碌氓搂路茅隆拢盲禄聥氓聼聙茅隆聮芒聜卢莽潞戮茂鹿垄茅聨戮莽聟聲茅聳潞氓聜聺忙鲁聲茅聬聮氓聭聤莽鹿聺茅聢搂莽卢聼茅聢搂茅聯聫氓聹拢录忙驴隆忙碌職忙碌聹莽聧庐茅聫聧忙聲聝茅聳碌氓氓聳聴氓搂拢忙聞聰聭茅聬聵氓聻芦忙職漏氓漏碌莽聜漏氓聛聯氓漏碌莽聜掳氓聦聽茅聧楼脩聟氓聤娄氓茫聜聠氓赂聤茅聭鲁氓聣聛莽虏禄茅聨戮莽禄聵莽聟聼茅聳驴忙驴聠忙麓陇茅聧聵忙聺聭芒聰聭茅聢漏氓聠漏氓聙聬茅聳潞盲陆赂氓聙驴氓庐聲忙禄聠忙驴隆忙聫聮忙麓颅氓搂聤莽禄聵忙聥聟忙赂職氓聤赂茅聴聛氓聯聞莽聣聹茅聧聯忙卢聯茂陆聻氓漏碌氓卢漏茅聳碌氓陇聤忙碌路茅聧聽忙聞庐莽虏聬聰莽聟聨莽聞碌忙陇陇忙聨聠氓聛聯忙陇陇?``scoring_runs``
        茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼忙聲聥莽聭聹氓驴聸氓聼聙茅隆聮盲陆潞芒聳聯茅聳潞氓聭鹿聞茅聧聮忙聨聭芒聳聨茅聨戮氓麓聡氓漏碌茂录聞氓聛聸茅聬篓莽聝陆氓麓?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓聙驴茅聳驴忙聴聢忙聲庐茅聧聥忙聮露莽虏篓聭忙篓禄氓聶陆茅聳禄莽聭漏茅聫聦莽聠赂忙陆聫忙楼拢氓聠漏茅聴聛忙聬聻氓聙聳茅聧聰忙楼聟氓聼聙茅隆聮氓聠戮忙陆聬忙驴聻忙碌聥忙聶聺莽禄卤莽聜麓莽卢聼茅聢搂莽聧聧聬茫聞垄忙聻聨茅聳鹿忙聝搂莽拢聧聵?茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳茅聨潞忙聴聙忙陇陇忙聸聴氓聛聯茅聳鹿猫聶聭篓茅聤聡忙聞掳氓鹿聮茅聨麓莽聥聙忙搂露忙驴聽莽聰碌氓篓聙氓篓录脩聡茅聧聸忙聰掳茅聴聜氓聜聶莽茅聨虏芒聙鲁脙潞茅聰聲氓聜職氓聜篓氓篓麓莽聤虏莽禄聬忙忙聼聸茅聢漏氓聠聤聲茅聳禄忙聮麓忙麓聼茅聫聦茂驴聽氓麓露茅聢潞盲陆聡茂陆聠莽鹿聸?``experiment_id`` 茅聴?``metrics``茅聴?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    dict[str, Any]
        茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬氓庐聞忙聡聤氓陆聮氓麓露茅聤聤脩聟氓陆戮茅聴聛氓聣虏氓聛聠茅聧聽氓聴聵莽茅聧聝娄莽聟聶茅聴聜氓聜職茅聧聰氓露聣忙聼聸茅聤聢氓聭掳氓芦掳茅聯禄忙聽聼篓氓漏碌氓聴聵氓聶鹿茅聳潞氓聴聵莽聭搂莽禄卤忙聨聴忙聜漏氓庐聲氓聸搂芒聮聦氓娄陇莽聤碌氓聛聻茅聬聴忙聸隆氓陆聛氓娄聻茫聜聠氓路禄茅聧聥忙聮鲁芒聰聭茅聢楼莽聜陆氓聛垄茅聳鹿氓聵虏茅聢禄氓潞隆芒聳聨茅聨麓莽聤鲁忙庐录忙驴隆茫聜聡氓搂路茅聧聥莽聜潞忙聜搂氓娄陇芒聜隆颅氓聸庐聸氓篓聭忙茅聬聼忙禄聝莽鹿聴茅聨陆忙垄禄茅聧聦忙卢聬莽录聛茅鲁聳氓聳職莽禄卤莽聜麓忙聥聟茅聧娄忙麓赂茅聴聛氓聣虏氓聛聟氓篓虏忙聽颅忙聥聭茅聬聰氓聟录莽聟聼茅聳潞氓聠篓忙麓陇忙碌聹氓陆聮忙聼聸氓篓聠忙聞颅莽陇聥茅聳潞氓虏聙氓聳聯莽禄卤忙聨聭脦聼茅聯聫氓聹聡聠氓漏碌莽聤鲁茅聧聽忙聽颅忙聼聤茅聰聺氓聮聛忙陆聳茅聳戮氓驴職茅聧聫忙禄聢忙聼聸氓芒聜禄聙盲陆赂芒聳聯氓漏碌氓聽垄莽拢陆氓篓麓茂陆聢忙莽聰露職莽麓聮莽聮聡猫聦忙陇陇忙聸陋莽露聛茅隆垄忙聸垄氓茅聢潞盲陆鹿氓聙聴聜盲陆鹿忙聜露茅聧聨氓忙垄陆茅聧聲莽陇聥茅聳赂忙聞碌氓聳聴茅聢路忙聢娄忙聼陇茅聨颅忙聨聭氓聣颅忙陇陇氓驴聯莽聟禄茅聧楼莽隆聙茅聳禄忙聞卢莽搂露茅聧聽忙聞虏颅茅聬聰莽聟聨莽聞碌忙陇陇忙聨聠氓聛聯忙陇?``model_name``茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞垄忙驴漏聠氓聣娄茅聬聮莽聲聦忙聠赂氓庐楼氓聽垄盲潞卤茅聴聜盲陆赂忙聬芦茅聧聼氓麓聬忙驴聽忙掳颅氓聞聴茅聳赂芒聞聝猫陇掳忙聨聯忙聶虏茅聳赂茫聝聭氓聠篓茅聴聜盲陆赂忙聬芦氓娄卢聭氓聢聮氓氓陇聤脦娄茅聳赂忙聸篓氓聻掳莽禄聽忙露垄忙聼聸茅聨戮猫聦露氓聟聦茅聮聝茅聛聧忙垄?``seed_reports``茅聴?
        ``aggregated_metrics``茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓聙驴茅聳驴忙聴聢忙聲庐茅聧聥忙聮露莽虏篓聭忙篓禄氓聶陆茅聳禄莽聭漏茅聫聦莽聠赂忙陆聫忙楼拢氓聠漏茅聴聛?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聨录盲陆赂莽拢鹿氓氓卢碌路茅聬聴聴脙鹿茅聰聺氓聽聼莽禄戮氓聣搂氓聨搂茅隆颅猫路篓忙聧聞忙赂職氓聣卤茅聳禄氓驴聯茅陋聫莽禄聽忙聮禄氓录禄茅聬聰莽垄卤忙聲聤茅聳赂氓驴聯氓聬聥茅聧聨忙聮禄忙垄潞莽聙鹿莽聥聙氓職聶茅聴聛氓赂露聛茅聬聸茅聢搂莽聧聬麓茫聞娄氓職聧茅聳碌氓茅聶聞茅聬拢氓聽聲忙垄禄茅聧聦忙掳卢芒聜聝陆忙聜聻茅聰聲氓聜職莽庐聳茅聳赂忙麓聳莽潞戮忙聦聨莽鹿聺忙驴聽氓聜聹氓茅聳禄忙聨聭莽聛職茅聤聡氓聥鹿聮茅聨麓忙驴聬盲禄录莽录聛莽聜聞鲁茅隆聝陆氓录禄茅聬聰莽聟聨莽茅聳鹿茅聜娄氓聣聸茅聧聵忙垄潞莽禄聥猫路篓莽莽录聛莽聼聜茅聢楼忙聴聢氓麓聵茅盲陆赂盲潞戮茅聳驴忙驴聠莽掳录莽禄篓氓楼赂忙聼聼茅聬搂氓聯楼莽搂驴庐?``score_vector``茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓聙驴茅聳驴忙聴聢忙聲庐茅聧聥忙聮露莽虏篓聭忙篓禄氓聶陆茅聳禄莽聭漏茅聫聦莽聠赂忙陆聫忙楼拢氓聠漏茅聴聛莽篓驴茅氓聭麓氓聼聻茅聨麓茂赂陆氓聛聬茅聳鹿莽禄聵氓赂漏忙碌聽茅鹿聝忙垄潞茅聧聺氓聥录聛忙聧聡氓氓漏職忙聜垄茅聬聯氓潞拢莽聲戮茅聬聼忙禄聝莽鹿聵茅陋聻忙忙聜漏莽录聛忙篓录氓聨卤茅聳鹿猫聣掳忙聫禄虏盲禄聥氓录赂氓篓聭茫聢隆莽聟聼茅聳鹿氓聻聲芦忙碌聹忙聣庐芒聜忙麓聳茅聬聳氓聟录盲驴聤茅聨录盲陆鹿忙聶聺茅聳鲁脩聦氓聣聼氓娄芦氓聥戮露氓卢搂忙聥潞莽录聜盲陆赂莽聛聫氓聭颅茅聤聣茂赂潞盲潞聹忙陇陇忙聮露氓聛聸氓娄虏莽禄聵茫聛聞茅茫聢垄莽茅聬聯茫聜聠茫聛聞猫鹿聡忙聮鲁忙陇陇忙聳驴莽聼聜氓搂聤忙麓陋氓麓路茅莽聜虏氓氓娄聻茫聜聝氓聤聦茅聬聴氓聴聸忙聜戮茅聬聭莽颅聥忙聲聜茅聳赂芒聞聝莽聭搂茅聰聸忙驴聡莽鹿聸忙聺聢氓聣搂氓莽录聛莽聭漏氓庐聲忙聬鹿茅聧聫氓盲潞戮茅聳赂氓驴聯忙碌聹茅鹿聝忙垄潞茅聧聸氓漏聞芒聵聣茅聳驴忙聸碌驴茅聨?

    茅聴聜氓聜職氓聙路氓篓麓氓聸茅聢搂莽篓聢茅聫聟莽聤禄茫聙聛氓篓聠忙聮鲁氓聠聙忙陇陇忙聮露氓聛聼茅聬聸茫聝漏忙垄潞茅聧聺氓聥颅芒聳聣茅聳赂忙篓驴氓聛聬忙戮鹿忙聸隆茫聛聬茅聨潞忙聮露氓聨碌茅聴聜氓聜職氓聙赂茅氓麓聺氓氓陇聥莽聟聲?
    ------
    ValueError
        茅聴?``seed_reports`` 氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄茅聳禄忙聞聬芦茅聧篓氓聜聻忙聼聸茅茂陆聠莽莽录聛莽聤鲁氓陆聝茅聹聣茅聳驴忙驴聠忙麓篓茅聤聠氓漏聡忙聼隆莽聙拢脨搂茅聳潞氓卤禄氓聤聦茅聢鹿忙聢娄氓麓卤茅聢潞氓聜聜茂鹿聜忙垄潞莽录聛忙篓禄氓掳掳茅聳驴忙聸聵氓聻陆氓氓漏職氓聻戮茅聨掳盲陆赂忙聜聭茅聴聛忙聬聻氓聞庐茅聫聟忙禄聟莽麓聻忙赂職芒聜氓漏職氓麓聮氓篓聠忙聢颅忙搂聴聛氓聭聤猫聣戮茅隆聭氓莽聧聧聬莽拢录忙驴庐忙篓驴氓聛聬茅聧聰莽聝陆忙聜路氓氓聥聳潞盲陆鹿氓聤聶茅聨卢聯氓麓聧聲忙陆聣茅聴聜盲陆赂氓茅聧聥氓聴聸氓麓掳茅聨戮猫炉虏氓聜篓茅聴聛莽搂碌茅聢路忙聢娄忙聼拢茅聬聰莽聟聨盲潞掳茅聳赂茫聜聡氓聳聳茅聫聦忙露職芒聜卢茂陆聨氓聛聯茅聭掳氓聺聴忙聼聼茅聯聫莽聼聮忙陇陇茫聢隆忙氓庐聲氓楼赂忙聜垄茅聧聺氓聥卢職?

    忙驴聽莽聰碌氓搂路茅聫聛氓聭聤忙聟篓茅聬聭猫聴聣忙聻聛茅聳鹿茅聴麓莽莽潞戮莽禄聵莽鹿聸茅聨麓忙卢聫莽聛陋茅聳禄忙聮麓忙麓聼茅聫聦茂驴聽氓麓聮氓漏碌氓聸漏茫聙聤茅聳潞盲陆赂莽聣聯茅聨?
    -----
    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜麓莽聣聽茅隆垄忙聸垄忙聲聝茅聢搂忙聡聹猫陇掳忙聨聯莽聟聸茅聬聫莽聜露茅聧聰忙掳卤莽麓聮茅聢搂氓麓聵茅聢潞氓聠陋莽聵聢茅聴聛茅聺聸茅陋聫莽禄卤忙聸聼忙聶露茅聧篓氓聫聫氓漏碌莽聤虏忙聡聫茅隆聫莽聤碌莽麓聮忙聺聢茫聞楼莽卢聮茅聯禄茂陆聣忙聼陇氓篓麓氓卢芦聡茅聳赂忙篓禄氓赂聮茅聢鹿忙聢娄忙聜漏茅隆聬茂陆聟茅聴聛氓聭聤忙麓聳茅聬聫忙掳露忙聜戮忙驴聜氓聭聤芒聮?``scoring_runs`` 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓聙驴茅聳驴忙聸聻氓聙鹿茅聧聨忙聢聽忙聠赂茅聬聴氓聽聺莽卢聮莽录聛芒聜戮露氓卤赂氓聛聯茅聧聫氓盲潞戮茅聴聛茅聙聻氓卤陆氓垄麓茅聳赂芒聰戮氓聛聯氓娄聻茫聜聠氓赂聤茅聬聮茂鹿聜忙聝聝茅聨麓莽聤碌莽拢录茅聳鲁脩聦氓聣職莽禄禄忙驴聥氓麓露茅聤聤脩聝氓鹿聢忙驴聽莽聰碌氓聛聻茅聧篓氓聽聲忙聜垄茅隆聮氓聜聸?茅聴聜氓聜職氓聙赂茅聧聤莽聝陆忙聡聴茅聧聸聴芒聜鹿聜氓麓录茅聤聫芒聙鲁盲禄聫氓娄聻茫聜聠氓聤搂莽禄聽忙聢聽莽虏聢忙戮露氓卤赂氓聛聯茅聧聫氓陇聤盲潞戮茅聴聛氓聭聤忙麓聳茅聬聫忙掳露氓录虏茅隆聫氓聽聼芒聮聭茅聳鹿莽篓驴忙碌路莽禄聽忙聮麓茫聛聞茅聰聰猫炉虏莽聛娄莽聙鹿忙聸陋莽鹿聺茅陋聻氓潞篓忙聡聻茅聤聣脩聝氓鹿聴茅聴聜盲陆赂忙聬聧聤氓聹聜卢莽卢聸氓聸搂莽庐職茅聳赂忙聡聡茅聧聥忙聮鲁芒聳聯茅聧篓莽聛聛茅聴聛氓聭聤忙垄鹿茅聧篓莽聰赂茂陆聻猫鹿聡忙聮聛忙麓聧聸莽聰禄茅聴聜盲陆陆茅忙聽麓掳茅聫聡莽聤虏氓聻聺茅聳禄茫聢聽忙聠隆茅聢路忙聢聽莽麓聮莽聙拢盲禄垄忙陇陇忙聨聥氓聻鹿茅聢鹿忙聢娄忙聜聶莽聮聡脩聠氓聬芦氓娄陇莽聤碌氓聛聸茅聧聼忙聺聭莽聲路莽禄聥猫忙聴聢氓聼聙茅隆聮氓聜聹莽聼聠茅聬聨茂陆聨氓聛聠莽禄聽茅鹿聝忙聼聸茅茂陆聡莽聼聦茅聳禄莽聭漏茅聫聦莽聠录茅聫聜茫聜聢芒聜聺聴忙聼聼茅莽聲聦忙聡聡莽聙碌莽聢录茅陋聻氓芒聰聭茅聧聽莽聣聢氓聻驴氓搂聤莽禄聵忙聥聟忙氓潞聺猫莽录聜盲陆潞氓搂碌茅聧篓氓聹茅聬聬氓莽炉聙茅聳赂茫聝聭氓聴聵忙聠掳茅聴聜盲陆潞氓聛篓茅聧聨忙聼楼氓麓聹氓搂聵猫路潞猫搂娄茅聬聨茂陆聨氓聳聴茅聬聯忙聸聼莽鹿聸茅聨麓忙驴聠猫聢鹿茅聳潞氓氓聙聺茅聫聦忙聸垄氓麓录茅隆聬莽掳陆莽录聜盲陆陆茅虏赂茅聨赂氓漏職氓鹿聫茅聧聸茅聬聨茂陆聨盲潞聺茅聨鲁忙卢聭莽麓聯茅聧聦忙卢聬莽楼掳忙陇陇忙聸聠氓聸篓氓聛聥茅聳鹿忙聧聬氓麓碌氓庐聯盲陆赂芒聰聭茅聬聵氓聫聣氓陇聞茅聳赂氓卢茅聫聦茂陆聣氓鹿聵茅聧聬氓路聰芦茅聴聛茅聺聸脜聢茅聫聟忙禄聢忙聜隆猫鹿聡忙露聶莽聟聲忙陇陇忙聞露莽碌驴莽禄聽忙聽聜卢莽聭掳忙聠隆莽禄禄氓聽楼氓录禄茅聬聰莽聟聨莽茅聳鹿莽聝聵忙垄禄忙忙驴?seed 茅聴?
    氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆茫聙聜茅莽聜聳潞氓聸楼氓聻鹿莽潞戮氓忙聼聼忙聺聢氓聣搂莽聲卤莽禄戮忙聝搂莽露聤茅聫聦茂驴聽氓麓露茅聤聣茫聝搂盲禄录莽录聜盲陆潞氓搂麓莽录聧氓漏聡氓录禄茅聤聢氓聬陇忙麓陋茅聧聬芒聲聛忙聻陇忙驴隆莽聜聙聭茅聧篓忙麓聲聝茅聤聫茫聜聟莽庐聳茅聳碌氓驴聥氓聻潞茅聧聥忙漏聙忙聼聧茅聢潞盲陆赂忙職聼氓篓麓忙禄聞莽聭漏氓猫聶鹿忙聥聽茅聫聦茫聝庐芒聜莽麓聮茅聬聵猫聦搂莽聙鹿忙聸聼忙麓聼茅聫聦氓聴聴茅聧聸忙聻聝茅聴聜氓漏聨氓聛聞氓篓虏茂赂戮莽虏聶茅聨麓莽聤虏莽聭聹莽聣聢氓赂聴茅聬聯忙聸聼盲驴聤茅聤聢氓聭職聲莽录聛氓聧聻莽拢聨虏忙聬聛忙聜搂茅聧驴聢莽聭篓莽录聜盲陆虏氓聛聯茅聳赂氓虏聙氓聛聻茅聬聯莽聤禄忙聜路氓篓聠氓驴聯莽录聛猫路潞氓录露茅聤聡氓聥娄路茅聰聰氓聣搂莽聳聞茅聴聛猫炉隆氓聳聳茅聧垄忙聼楼芒聙娄茅聤聤芒聲聛莽忙陇陇忙聨聭氓聙赂茅聧陇忙聨聯莽鹿聺茅聬垄茅聺聸盲禄聹忙驴隆茂鹿庐聲茫聢隆莽赂陆茅聧聸氓漏聜盲禄录茅聬聨莽聭掳氓茅聧聥氓漏聜茅聢潞氓聜聵盲潞聹茅聳鹿忙聧聬忙鲁聸茅聧聯茅聛聧忙聼聸茅聤聢氓聠篓芒聜卢忙驴聥氓录禄茅聰聺氓聴聵氓聞聼茅聳碌氓驴聤忙聡聡忙驴聸卢忙聼娄茅聧聬氓聯潞莽录聛莽禄聵莽聸炉氓庐聲茫聞漏茅聮聵氓茂录聯茅聴聜?茅聴聜氓聜職氓聙赂茅聧聤茅拢聨忙氓搂聵猫路潞莽聞碌忙陇陇忙聨聭氓聙赂忙碌聽忙禄聢忙聼陇氓篓虏猫莽聛隆茅聳潞氓聭聢录茅聨聡氓麓录氓篓聭忙莽聸聨茅聴聜盲陆潞忙聡聯茅隆聲氓麓聭茅聬聬莽聰碌莽聼聠茅聬聨茂陆聨氓聳聴茅聬聯忙聸垄忙聼聧莽聭聶氓聥聤陇氓篓麓忙禄聟氓聛聯忙路聡氓漏聡忙聜聶茅隆聫氓聥篓氓聛聯茅聫聧脩聦氓聲聟氓漏碌莽聤鲁猫聣戮莽潞戮氓忙聼聼茅聧聯脩聟氓赂聸茅聳禄忙聮麓忙麓聳茅聢鹿忙聢娄忙聜漏茅聨禄忙聨聧莽掳陆茅聴聛莽禄聵忙聦赂茅聤聢忙聬聛氓录禄茅聰聺氓聭陋忙搂聢茅聳赂忙楼聝氓聛聻茅聬聺忙驴聥忙聜聴莽聭聶氓聥茅聳赂氓潞隆茫聜赂莽炉聯氓篓聯忙掳拢氓聛聯茅聳潞氓卤戮莽聸炉茅聨聢陇忙聼篓莽聙拢莽聲禄茅聴聜盲陆赂忙聬聫聢莽虏聶茅聨麓茂鹿聙莽聟聺茅聨录忙路聙聴茅聬陇芒聜漏碌莽聜麓氓聻露猫聯卤茅聫聣氓聽聺莽聣聯氓搂聤猫聶鹿芒聜聹颅氓搂碌茅隆楼氓陇聥茫聛聞茅聰聰莽聜卢忙職聳禄忙露卤氓聳聳猫聻拢茅聳赂氓驴聲氓聨搂莽潞戮忙垄潞茅聧聸氓漏聞莽庐聯茅聬聨忙掳录氓聣聼茅聢楼忙聽颅氓麓卤氓篓聭忙卢聦忙聥潞?``experiment_id`` 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜麓莽聣聽茅隆垄忙聸垄氓娄露茅聧隆忙陇戮莽虏隆忙驴隆莽聜聙聳茅聧聰脩聟莽虏禄茅聨麓莽聤虏茅聳赂脩聞茅聬聯忙聸聼莽鹿聸茅聧隆忙楼聛莽忙驴隆氓聫聣忙陇陆茅聫聦茂陆聟芒聰聭忙驴聽氓聜聹氓聨聴聛茅聙聸莽漏露茅聧聰氓虏聞芒聙聰茅聧聬氓芦職茅聴聤氓聨录茅隆楼忙掳鹿聺?
    """
    if not seed_reports:
        raise ValueError(f"seed_reports must be non-empty for {model_name}")
    experiment_ids = [run["experiment_id"] for run in seed_reports[0]["scoring_runs"]]
    aggregated_metrics: dict[str, dict[str, float]] = {}
    for experiment_id in experiment_ids:
        metric_keys = list(seed_reports[0]["scoring_runs"][0]["metrics"])
        metrics_for_experiment: dict[str, float] = {}
        for metric_key in metric_keys:
            values = []
            for report in seed_reports:
                matching_run = next(
                    run for run in report["scoring_runs"]
                    if run["experiment_id"] == experiment_id
                )
                values.append(float(matching_run["metrics"][metric_key]))
            metrics_for_experiment[metric_key] = float(sum(values) / len(values))
        aggregated_metrics[experiment_id] = metrics_for_experiment

    return {
        "model_name": model_name,
        "seed_reports": seed_reports,
        "aggregated_metrics": aggregated_metrics,
        "score_vector": _score_candidate_lexicographically(metrics_by_experiment=aggregated_metrics),
    }


def _select_representative_seed_report(  
    seed_reports: list[dict[str, Any]],  
    aggregated_score_vector: tuple[float, ...],  
) -> dict[str, Any]:  
    if not seed_reports:  
        raise ValueError("seed_reports must be non-empty")  

    def _distance_to_aggregate(report: dict[str, Any]) -> tuple[float, int]:  
        distance = sum(abs(float(actual) - float(target)) for actual, target in zip(report["score_vector"], aggregated_score_vector))  
        return float(distance), int(report["seed"])  

    return min(seed_reports, key=_distance_to_aggregate)  


def _index_seed_reports_by_signature(  
    grid_reports: list[dict[str, Any]],  
) -> dict[str, dict[int, dict[str, Any]]]:  
    indexed_reports: dict[str, dict[int, dict[str, Any]]] = {}  
    for report in grid_reports:  
        signature = str(report.get("signature") or "").strip()  
        if not signature:  
            raise ValueError("grid report is missing signature")  
        overrides = dict(report.get("overrides") or {})  
        seed_value = int(report.get("seed", overrides.get("train.seed", 0)))  
        report["seed"] = seed_value
        indexed_reports.setdefault(signature, {})[seed_value] = report  
    return indexed_reports  


def _select_top_neural_grid_reports(  
    grid_reports: list[dict[str, Any]],  
    *,  
    top_k: int,  
) -> list[dict[str, Any]]:  
    if top_k < 1:  
        raise ValueError("top_k must be >= 1")  
    return sorted(grid_reports, key=lambda report: tuple(report["score_vector"]))[:top_k]  


def _probe_model_runtime_resource_meta(model_name: str, train_result: Any) -> dict[str, Any]:  
    train_report = dict(train_result.metadata["train_report"])  
    checkpoint_path = str(train_report.get("checkpoint_path") or "").strip()  
    base_meta = {  
        "status": "blocked" if not checkpoint_path else "unavailable",  
        "checkpoint_path": checkpoint_path or None,  
        "params": None,  
        "ram_peak": None,  
        "ram_peak_mb": None,  
    }  
    if not checkpoint_path:  
        base_meta["reason"] = "missing_checkpoint_path"  
        return base_meta  

    try:  
        model = create_model(model_name, {"checkpoint_path": checkpoint_path})  
    except Exception as exc:  
        base_meta["error"] = f"{type(exc).__name__}: {exc}"  
        return base_meta  

    base_meta.update(  
        {  
            "status": "ok",  
            "params": int(getattr(model, "params", 0)),  
            "ram_peak": float(getattr(model, "ram_peak", 0.0)),  
            "ram_peak_mb": float(getattr(model, "ram_peak_mb", 0.0)),  
        }  
    )  
    return base_meta  


def _search_best_neural_train_result(  
    *,  
    model_name: str,  
    prepare_root: Path,  
    raw_root: Path,  
    split_manifest: dict[str, Any],  
    output_root: Path,  
    estimator_cfgs: dict[str, dict[str, Any]],  
    device: str,  
    epochs: int,  
    batch_size: int,  
    eval_batch_size: int,  
    baseline_scoring_metrics_by_experiment: dict[str, dict[str, float]] | None = None,  
    neural_grid: dict[str, dict[str, Any]] | None = None,  
    liquid_robustness_profiles: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None,  
    top_k_multiseed: int | None = None,  
    neural_multiseeds: tuple[int, ...] | list[int] | None = None,  
    epoch_candidate_stride: int | None = None,  
    checkpoint_selection_mode: str = "downstream_val_rescore",  
    resolved_training_budget: dict[str, Any] | None = None,  
    allow_cache: bool = False,  
) -> dict[str, Any]:  
    resolved_neural_grid = neural_grid or _FAIR_NEURAL_GRID  
    resolved_liquid_profiles = list(liquid_robustness_profiles or _LIQUID_ROBUSTNESS_PROFILES)  
    resolved_top_k_multiseed = int(top_k_multiseed if top_k_multiseed is not None else _NEURAL_TOP_K_MULTI_SEED)  
    resolved_neural_multiseeds = tuple(  
        int(seed) for seed in (neural_multiseeds if neural_multiseeds is not None else _NEURAL_MULTI_SEEDS)  
    )  
    budget_metadata = dict(resolved_training_budget or {})  
    if resolved_top_k_multiseed < 1:  
        raise ValueError("top_k_multiseed must be >= 1")  
    if not resolved_neural_multiseeds:  
        raise ValueError("neural_multiseeds must be non-empty")  
    if epoch_candidate_stride is not None and int(epoch_candidate_stride) < 1:  
        raise ValueError("epoch_candidate_stride must be >= 1 when provided")  
    search_root = output_root / "search"  
    grid_reports = [  
        _run_neural_search_candidate(  
            model_name=model_name,  
            overrides=overrides,  
            seed=0,  
            prepare_root=prepare_root,  
            raw_root=raw_root,  
            split_manifest=split_manifest,  
            search_root=search_root,  
            estimator_cfgs=estimator_cfgs,  
            device=device,  
            epochs=epochs,  
            batch_size=batch_size,  
            eval_batch_size=eval_batch_size,  
            baseline_scoring_metrics_by_experiment=baseline_scoring_metrics_by_experiment,  
            epoch_candidate_stride=epoch_candidate_stride,  
            checkpoint_selection_mode=checkpoint_selection_mode,  
            allow_cache=allow_cache,  
        )  
        for overrides in _iter_neural_override_grid(  
            model_name,  
            neural_grid=resolved_neural_grid,  
            liquid_robustness_profiles=resolved_liquid_profiles,  
        )  
    ]  
    top_reports = _select_top_neural_grid_reports(  
        grid_reports,  
        top_k=min(resolved_top_k_multiseed, len(grid_reports)),  
    )  
    cached_seed_reports_by_signature = _index_seed_reports_by_signature(grid_reports)  

    multiseed_reports: list[dict[str, Any]] = []  
    for top_report in top_reports:  
        overrides = dict(top_report["overrides"])  
        top_signature = str(top_report["signature"])  
        seed_reports = []  
        for seed in resolved_neural_multiseeds:  
            cached_seed_report = cached_seed_reports_by_signature.get(top_signature, {}).get(int(seed))  
            if cached_seed_report is not None:  
                seed_reports.append(cached_seed_report)  
                continue  
            seed_reports.append(  
                _run_neural_search_candidate(  
                    model_name=model_name,  
                    overrides=overrides,  
                    seed=seed,  
                    prepare_root=prepare_root,  
                    raw_root=raw_root,  
                    split_manifest=split_manifest,  
                    search_root=search_root,  
                    estimator_cfgs=estimator_cfgs,  
                    device=device,  
                    epochs=epochs,  
                    batch_size=batch_size,  
                    eval_batch_size=eval_batch_size,  
                    baseline_scoring_metrics_by_experiment=baseline_scoring_metrics_by_experiment,  
                    epoch_candidate_stride=epoch_candidate_stride,  
                    checkpoint_selection_mode=checkpoint_selection_mode,  
                    allow_cache=allow_cache,  
                )  
            )  
        multiseed_reports.append(_aggregate_seed_reports(model_name, seed_reports))  

    multiseed_reports.sort(key=lambda report: tuple(report["score_vector"]))  
    best_multiseed = multiseed_reports[0]  
    best_seed_reports = list(best_multiseed["seed_reports"])  
    selected_report = _select_representative_seed_report(  
        best_seed_reports,  
        tuple(best_multiseed["score_vector"]),  
    )  

    selected_report_snapshot = copy.deepcopy(selected_report)

    final_output_root = output_root / "final_selected"  
    train_result = _run_frontend_training(  
        model_name=model_name,  
        prepare_root=prepare_root,  
        raw_root=raw_root,  
        split_manifest=split_manifest,  
        output_root=final_output_root,  
        estimator_cfg=copy.deepcopy(estimator_cfgs["ekf"]),  
        device=device,  
        epochs=epochs,  
        batch_size=batch_size,  
        eval_batch_size=eval_batch_size,  
        model_overrides={  
            **dict(selected_report_snapshot["overrides"]),  
            "train.seed": int(selected_report_snapshot["seed"]),  
            **(  
                {"train.save_epoch_candidates": True}  
                if checkpoint_selection_mode == "downstream_val_rescore"  
                else {}  
            ),  
            **(  
                {"train.epoch_candidate_stride": int(epoch_candidate_stride)}  
                if checkpoint_selection_mode == "downstream_val_rescore" and epoch_candidate_stride is not None  
                else {}  
            ),  
        },  
    )  
    final_checkpoint_selection = _select_final_neural_checkpoint(  
        model_name=model_name,  
        train_result=train_result,  
        prepare_root=prepare_root,  
        raw_root=raw_root,  
        split_manifest=split_manifest,  
        estimator_cfgs=estimator_cfgs,  
        selection_root=final_output_root / "paper_checkpoint_selection",  
        checkpoint_selection_mode=checkpoint_selection_mode,  
        baseline_scoring_metrics_by_experiment=baseline_scoring_metrics_by_experiment,  
    )  
    train_result = final_checkpoint_selection["train_result"]  
    search_audit = {  
        "model_name": model_name,  
        "grid_reports": grid_reports,  
        "top_reports": top_reports,  
        "multiseed_reports": multiseed_reports,  
        "selected_signature": selected_report_snapshot["signature"],  
        "selected_overrides": dict(selected_report_snapshot["overrides"]),  
        "selected_seed": int(selected_report_snapshot["seed"]),  
        "selected_seed_policy": "closest_to_multiseed_average",  
        "selected_score_vector": list(selected_report_snapshot["score_vector"]),  
        "selected_aggregated_score_vector": list(best_multiseed["score_vector"]),  
        "selected_seed_risk": {  
            "seed_values": [int(report["seed"]) for report in best_multiseed["seed_reports"]],  
            "seed_score_vectors": [list(report["score_vector"]) for report in best_multiseed["seed_reports"]],  
            "distance_to_multiseed_average": [  
                float(  
                    sum(  
                        abs(float(actual) - float(target))  
                        for actual, target in zip(report["score_vector"], best_multiseed["score_vector"])  
                    )  
                )  
                for report in best_multiseed["seed_reports"]  
            ],  
        },  
        "selected_metrics": {  
            run["experiment_id"]: run["metrics"]  
            for run in selected_report_snapshot["scoring_runs"]  
        },  
        "final_checkpoint_selection": final_checkpoint_selection["selection_report"],  
        "top_k_multiseed": int(resolved_top_k_multiseed),  
        "multiseed_values": [int(seed) for seed in resolved_neural_multiseeds],  
        "epoch_candidate_stride": int(epoch_candidate_stride) if epoch_candidate_stride is not None else None,  
        "resolved_training_budget": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "eval_batch_size": int(eval_batch_size),
            "budget_source": str(budget_metadata.get("budget_source") or "unknown"),
            "paper_grade": bool(budget_metadata.get("paper_grade")) if "paper_grade" in budget_metadata else None,
            "diagnostic_only": bool(budget_metadata.get("diagnostic_only")) if "diagnostic_only" in budget_metadata else None,
        },
        "grid_definition": copy.deepcopy(resolved_neural_grid[model_name]),  
        "liquid_robustness_profile_names": (  
            [str(profile.get("name") or "") for profile in resolved_liquid_profiles]  
            if model_name == "liquid_ekf"  
            else None  
        ),  
    }  
    _write_json(output_root / "search_audit.json", search_audit)  
    return {  
        "train_result": train_result,  
        "search_audit": search_audit,  
    }  


def _run_classical_search_candidate(
    *,
    method_name: str,
    overrides: dict[str, Any],
    prepare_root: Path,
    raw_root: Path,
    search_root: Path,
    split_manifest: dict[str, Any],
    baseline_ekf_cfg: dict[str, Any] | None = None,
    baseline_scoring_metrics_by_experiment: dict[str, dict[str, float]] | None = None,
    allow_cache: bool = False,
) -> dict[str, Any]:
    include_safe = method_name != "ekf"
    scoring_config_names = _search_scoring_config_names(method_name, include_safe=include_safe)
    candidate_signature = _classical_candidate_signature(method_name, overrides)
    candidate_root = search_root / method_name / candidate_signature
    report_path = candidate_root / "candidate_report.json"
    cache_context = {
        "method_name": method_name,
        "overrides": dict(overrides),
        "prepare_root": str(prepare_root.resolve()),
        "raw_root": str(raw_root.resolve()),
        "val_ids": list(split_manifest["val_ids"]),
        "baseline_ekf_cfg": baseline_ekf_cfg,
        "baseline_scoring_metrics_by_experiment": _normalize_baseline_scoring_metrics_for_cache(
            baseline_scoring_metrics_by_experiment
        ),
        "search_experiments": list(scoring_config_names),
        "implementation_fingerprint": _resolve_search_implementation_fingerprint(
            search_kind="classical",
            implementation_name=method_name,
            experiment_config_names=scoring_config_names,
        ),
    }
    cached_report = _load_cached_search_report(report_path, cache_context=cache_context) if allow_cache else None
    if cached_report is not None:
        return cached_report

    estimator_cfg = _apply_estimator_overrides(_build_base_estimator_cfg(method_name), overrides)
    estimator_cfgs = {method_name: estimator_cfg}
    baseline_method_name = "ekf"
    if method_name != "ekf":
        estimator_cfgs["ekf"] = copy.deepcopy(
            baseline_ekf_cfg if baseline_ekf_cfg is not None else _build_base_estimator_cfg("ekf")
        )
    else:
        baseline_method_name = method_name

    scoring_seq_ids = list(split_manifest["val_ids"])
    if not scoring_seq_ids:
        raise RuntimeError(f"{method_name} search requires non-empty val_ids for fair scoring")

    scoring_runs = [
        _run_scoring_experiment(
            config_name=config_name,
            prepare_root=prepare_root,
            raw_root=raw_root,
            method_name=method_name,
            estimator_cfgs=estimator_cfgs,
            model_cfgs={},
            output_root=candidate_root / "scoring_runs",
            baseline_method_name=baseline_method_name,
            seq_ids=scoring_seq_ids,
            baseline_metrics=(
                dict(baseline_scoring_metrics_by_experiment[Path(config_name).stem])
                if baseline_scoring_metrics_by_experiment is not None
                and method_name != baseline_method_name
                and Path(config_name).stem in baseline_scoring_metrics_by_experiment
                else None
            ),
        )
        for config_name in scoring_config_names
    ]

    metrics_by_experiment = {
        run["experiment_id"]: dict(run["metrics"])
        for run in scoring_runs
    }
    safe_metrics = metrics_by_experiment.get("e0_safe_mode")
    if safe_metrics is None and baseline_scoring_metrics_by_experiment is not None:
        safe_metrics = baseline_scoring_metrics_by_experiment.get("e0_safe_mode")
    if safe_metrics is None:
        metrics_by_experiment = _inject_self_baseline_safe_metrics(metrics_by_experiment)
        safe_metrics = metrics_by_experiment["e0_safe_mode"]
    elif "e0_safe_mode" not in metrics_by_experiment:
        metrics_by_experiment = dict(metrics_by_experiment)
        metrics_by_experiment["e0_safe_mode"] = dict(safe_metrics)
    if method_name == baseline_method_name and not include_safe:
        metrics_by_experiment = _inject_self_baseline_safe_metrics(metrics_by_experiment)
        safe_metrics = metrics_by_experiment["e0_safe_mode"]

    safe_p95 = float(safe_metrics.get("p95", 0.0))
    safe_ekf_p95 = float(safe_metrics.get("ekf_p95", safe_p95))

    candidate_report = {
        "method_name": method_name,
        "signature": candidate_signature,
        "overrides": dict(overrides),
        "cache_context": cache_context,
        "estimator_cfg": estimator_cfg,
        "baseline_method_name": baseline_method_name,
        "safe_scoring_included": include_safe,
        "scoring_seq_ids": scoring_seq_ids,
        "scoring_runs": scoring_runs,
        "score_vector": _score_candidate_lexicographically(metrics_by_experiment=metrics_by_experiment),
        "safe_margin_vs_ekf": safe_p95 - safe_ekf_p95,
        "safe_relative_margin": (safe_p95 - safe_ekf_p95) / max(safe_ekf_p95, 1e-9),
    }
    _write_json(report_path, candidate_report)
    return candidate_report


def _search_best_classical_estimator_cfgs(
  
    *,  
    prepare_root: Path,  
    raw_root: Path,  
    output_root: Path,  
    split_manifest: dict[str, Any],  
    allow_cache: bool = False,  
) -> dict[str, Any]:  
    search_root = output_root / "search"  
    ekf_reports = [  
        _run_classical_search_candidate(  
            method_name="ekf",  
            overrides=overrides,  
            prepare_root=prepare_root,  
            raw_root=raw_root,  
            search_root=search_root,  
            split_manifest=split_manifest,  
            allow_cache=allow_cache,  
        )  
        for overrides in _iter_ekf_override_grid()  
    ]  
    ekf_reports.sort(key=lambda report: tuple(report["score_vector"]))  
    top_ekf_reports = ekf_reports[:_EKF_TOP_K_FOR_CLASSICAL]  
    selected_ekf_report = ekf_reports[0]  
    selected_ekf_cfg = copy.deepcopy(selected_ekf_report["estimator_cfg"])  
    selected_ekf_safe_run = _run_scoring_experiment(  
        config_name=_SAFE_SCORING_EXPERIMENT,  
        prepare_root=prepare_root,  
        raw_root=raw_root,  
        method_name="ekf",  
        estimator_cfgs={"ekf": selected_ekf_cfg},  
        model_cfgs={},  
        output_root=search_root / "ekf" / selected_ekf_report["signature"] / "scoring_runs",  
        seq_ids=list(split_manifest["val_ids"]),  
    )  
    selected_ekf_metrics_by_experiment = {  
        run["experiment_id"]: dict(run["metrics"])  
        for run in selected_ekf_report["scoring_runs"]  
    }  
    selected_ekf_metrics_by_experiment[selected_ekf_safe_run["experiment_id"]] = dict(selected_ekf_safe_run["metrics"])  

    robust_reports = [  
        _run_classical_search_candidate(  
            method_name="robust_ekf",  
            overrides=overrides,  
            prepare_root=prepare_root,  
            raw_root=raw_root,  
            search_root=search_root,  
            split_manifest=split_manifest,  
            baseline_ekf_cfg=selected_ekf_cfg,  
            baseline_scoring_metrics_by_experiment=selected_ekf_metrics_by_experiment,  
            allow_cache=allow_cache,  
        )  
        for overrides in _iter_robust_override_grid(top_ekf_reports)  
    ]  
    robust_reports.sort(key=lambda report: tuple(report["score_vector"]))  
    selected_robust_report = robust_reports[0]  

    fgo_reports = [  
        _run_classical_search_candidate(  
            method_name="fgo",  
            overrides=overrides,  
            prepare_root=prepare_root,  
            raw_root=raw_root,  
            search_root=search_root,  
            split_manifest=split_manifest,  
            baseline_ekf_cfg=selected_ekf_cfg,  
            baseline_scoring_metrics_by_experiment=selected_ekf_metrics_by_experiment,  
            allow_cache=allow_cache,  
        )  
        for overrides in _iter_fgo_override_grid(top_ekf_reports)  
    ]  
    fgo_reports.sort(key=lambda report: tuple(report["score_vector"]))  
    selected_fgo_report = fgo_reports[0]  

    selected_estimator_cfgs = {  
        "ekf": selected_ekf_cfg,  
        "robust_ekf": copy.deepcopy(selected_robust_report["estimator_cfg"]),  
        "fgo": copy.deepcopy(selected_fgo_report["estimator_cfg"]),  
    }  
    search_audit = {  
        "ekf": {  
            "grid_reports": ekf_reports,  
            "selected_signature": selected_ekf_report["signature"],  
            "selected_overrides": dict(selected_ekf_report["overrides"]),  
            "selected_metrics": selected_ekf_metrics_by_experiment,
            "selected_score_vector": list(selected_ekf_report["score_vector"]),  
            "selected_safe_run": {  
                "experiment_id": selected_ekf_safe_run["experiment_id"],  
                "output_root": selected_ekf_safe_run["output_root"],  
                "eval_stage_name": selected_ekf_safe_run["eval_stage_name"],  
                "scene_sampling_seq_ids": list(selected_ekf_safe_run["scene_sampling_seq_ids"]),  
            },  
        },  
        "robust_ekf": {  
            "grid_reports": robust_reports,  
            "selected_signature": selected_robust_report["signature"],  
            "selected_overrides": dict(selected_robust_report["overrides"]),  
            "selected_metrics": selected_ekf_metrics_by_experiment,
            "selected_score_vector": list(selected_robust_report["score_vector"]),  
        },  
        "fgo": {  
            "grid_reports": fgo_reports,  
            "selected_signature": selected_fgo_report["signature"],  
            "selected_overrides": dict(selected_fgo_report["overrides"]),  
            "selected_metrics": {
                run["experiment_id"]: dict(run["metrics"])
                for run in selected_fgo_report["scoring_runs"]
            },
            "selected_score_vector": list(selected_fgo_report["score_vector"]),  
        },  
        "selected_estimator_cfgs": selected_estimator_cfgs,  
        "ekf_top_seed_signatures": [report["signature"] for report in top_ekf_reports],  
    }  
    _write_json(output_root / "search_audit.json", search_audit)  
    return {  
        "selected_estimator_cfgs": selected_estimator_cfgs,  
        "selected_ekf_metrics_by_experiment": selected_ekf_metrics_by_experiment,  
        "search_audit": search_audit,  
    }  


def _build_fairness_audit(  
    classical_search_result: dict[str, Any],  
    neural_search_results: dict[str, Any],  
    train_results: dict[str, Any],  
    *,  
    device: str,  
    neural_search_surface: dict[str, Any],  
) -> dict[str, Any]:  
    resolved_neural_grid = dict(neural_search_surface["neural_grid"])  
    resolved_liquid_profiles = list(neural_search_surface["liquid_robustness_profiles"])  
    search_budget_summary = _build_search_budget_summary(classical_search_result, neural_search_results)  
    liquid_profile_names = [str(profile.get("name") or "") for profile in resolved_liquid_profiles]  
    shared_budget_scope = "shared_data_shared_budget_surface_except_model_specific_profile_axis"  
    liquid_has_model_specific_profile_axis = bool(liquid_profile_names)  
    expected_methods = ["ekf", "robust_ekf", "fgo", "lstm_ekf", "liquid_ekf"]  
    actual_methods = [method_name for method_name in expected_methods if method_name in (set(classical_search_result.get("selected_estimator_cfgs") or {}) | set(train_results))]
    return {  
        "status": "ok",  
        "neural_search_profile": str(neural_search_surface["profile_name"]),  
        "neural_search_surface": copy.deepcopy(neural_search_surface),  
        "comparison_surface": {
            "expected_methods": expected_methods,
            "actual_methods": actual_methods,
            "all_expected_methods_present": actual_methods == expected_methods,
            "missing_methods": [method_name for method_name in expected_methods if method_name not in set(actual_methods)],
            "notes": [
                "expected_methods is the fixed five-method paper comparison surface",
                "actual_methods lists the methods that really entered the current fairness audit payload",
            ],
        },
        "budget_asymmetry_audit": {  
            "status": "ok",
            "fairness_scope": shared_budget_scope,
            "budget_asymmetry_disclosed": liquid_has_model_specific_profile_axis,
            "liquid_extra_profile_axis": {
                "enabled": liquid_has_model_specific_profile_axis,
                "axis_name": "robustness_profile",
                "cardinality": len(liquid_profile_names),
                "profile_names": liquid_profile_names,
                "budget_multiplier_relative_to_single_profile": len(liquid_profile_names) if liquid_profile_names else 0,
                "disclosed_in_neural_budget": liquid_has_model_specific_profile_axis,
            },
            "lstm_model_specific_budget_axes": [],
            "liquid_model_specific_budget_axes": ["robustness_profile"] if liquid_has_model_specific_profile_axis else [],
            "notes": [
                "Liquid keeps a model-specific robustness_profile axis that is disclosed as a structural search capability difference rather than hidden target shaping",
                "LSTM shares the same data chain, training budget regime, and multiseed policy, but does not expose the Liquid-only robustness_profile axis",
            ],
        },
        "neural_capacity_alignment": _build_neural_capacity_alignment_report(neural_grid=resolved_neural_grid),  
        "epoch_candidate_policy": _build_epoch_candidate_policy_audit(neural_search_surface),  
        "neural_checkpoint_selection": _build_neural_checkpoint_selection_audit(neural_search_surface),  
        "search_budget_summary": search_budget_summary,  
        "neural_runtime_projection": _build_neural_runtime_projection(search_budget_summary),  
        "selection_logic": {  
            "split_rule_source": str((ROOT / "scripts" / "default_split_rules.yaml").resolve()),  
            "experiment_protocol_source": str((ROOT / "configs" / "base" / "experiment_protocol.yaml").resolve()),  
            "search_scorer_source": "scripts/20_run_paper_experiments.py::_score_candidate_lexicographically",  
            "search_scorer_externalized": False,  
            "notes": [  
                "classical and neural search both select on sim train/val only",  
                "held-out test_ids are reserved for final main conclusion only",  
                "search scorer is currently frozen in orchestration code rather than external rules",  
                "neural search uses a multiseed winner and a representative final seed rather than best single-seed cherry-pick",  
            ],  
        },  
        "device_policy": {  
            "requested": device,  
            "gpu_training_only": ["lstm_ekf", "liquid_ekf"],  
            "cpu_default_stages": [  
                "prepare",  
                "split",  
                "core_eval",  
                "metrics",  
                "statistics",  
                "plotting",  
                "audit",  
                "consumer_verify",  
                "public_benchmark",  
            ],  
        },  
        "classical_budget": {  
            "ekf": {  
                "count": len(classical_search_result["search_audit"]["ekf"]["grid_reports"]),  
                "selected_signature": classical_search_result["search_audit"]["ekf"]["selected_signature"],  
                "selected_overrides": classical_search_result["search_audit"]["ekf"]["selected_overrides"],  
                **_FAIR_EKF_GRID,  
            },  
            "robust_ekf": {  
                "count": len(classical_search_result["search_audit"]["robust_ekf"]["grid_reports"]),  
                "selected_signature": classical_search_result["search_audit"]["robust_ekf"]["selected_signature"],  
                "selected_overrides": classical_search_result["search_audit"]["robust_ekf"]["selected_overrides"],  
                **_FAIR_ROBUST_GRID,  
            },  
            "fgo": {  
                "count": len(classical_search_result["search_audit"]["fgo"]["grid_reports"]),  
                "selected_signature": classical_search_result["search_audit"]["fgo"]["selected_signature"],  
                "selected_overrides": classical_search_result["search_audit"]["fgo"]["selected_overrides"],  
                **_FAIR_FGO_GRID,  
            },  
        },  
        "neural_budget": {  
            "lstm_ekf": {  
                "grid_size": len(neural_search_results["lstm_ekf"]["search_audit"]["grid_reports"]),  
                "top_k_multiseed": neural_search_results["lstm_ekf"]["search_audit"]["top_k_multiseed"],  
                "multiseed_values": list(neural_search_results["lstm_ekf"]["search_audit"]["multiseed_values"]),  
                "epoch_candidate_stride": neural_search_results["lstm_ekf"]["search_audit"].get("epoch_candidate_stride"),  
                "resolved_training_budget": dict(neural_search_results["lstm_ekf"]["search_audit"].get("resolved_training_budget") or {}),
                "budget_source": str((neural_search_results["lstm_ekf"]["search_audit"].get("resolved_training_budget") or {}).get("budget_source") or "profile_default"),
                "fairness_scope": shared_budget_scope,
                "budget_asymmetry_disclosed": liquid_has_model_specific_profile_axis,
                "model_specific_budget_axes": [],
                "selected_seed_policy": neural_search_results["lstm_ekf"]["search_audit"]["selected_seed_policy"],  
                "selected_seed_risk": neural_search_results["lstm_ekf"]["search_audit"]["selected_seed_risk"],  
                "selected_model_runtime_resource_meta": _probe_model_runtime_resource_meta(  
                    "lstm_ekf",  
                    train_results["lstm_ekf"],  
                ),  
                **resolved_neural_grid["lstm_ekf"],  
            },  
            "liquid_ekf": {  
                "grid_size": len(neural_search_results["liquid_ekf"]["search_audit"]["grid_reports"]),  
                "top_k_multiseed": neural_search_results["liquid_ekf"]["search_audit"]["top_k_multiseed"],  
                "multiseed_values": list(neural_search_results["liquid_ekf"]["search_audit"]["multiseed_values"]),  
                "epoch_candidate_stride": neural_search_results["liquid_ekf"]["search_audit"].get("epoch_candidate_stride"),  
                "resolved_training_budget": dict(neural_search_results["liquid_ekf"]["search_audit"].get("resolved_training_budget") or {}),
                "budget_source": str((neural_search_results["liquid_ekf"]["search_audit"].get("resolved_training_budget") or {}).get("budget_source") or "profile_default"),
                "fairness_scope": shared_budget_scope,
                "budget_asymmetry_disclosed": liquid_has_model_specific_profile_axis,
                "model_specific_budget_axes": ["robustness_profile"] if liquid_has_model_specific_profile_axis else [],
                "robustness_profile_count": len(liquid_profile_names),
                "selected_seed_policy": neural_search_results["liquid_ekf"]["search_audit"]["selected_seed_policy"],  
                "selected_seed_risk": neural_search_results["liquid_ekf"]["search_audit"]["selected_seed_risk"],  
                "selected_model_runtime_resource_meta": _probe_model_runtime_resource_meta(  
                    "liquid_ekf",  
                    train_results["liquid_ekf"],  
                ),  
                "window": list(resolved_neural_grid["liquid_ekf"]["window"]),  
                "hidden_dim": list(resolved_neural_grid["liquid_ekf"]["hidden_dim"]),  
                "lr": list(resolved_neural_grid["liquid_ekf"]["lr"]),  
                "independent_weight_decay_grid": False,  
                "coupled_weight_decay_values": sorted(  
                    {  
                        float(profile["train.weight_decay"])  
                        for profile in resolved_liquid_profiles  
                        if "train.weight_decay" in profile  
                    }  
                ),  
                "robustness_profiles": [  
                    {  
                        key: value  
                        for key, value in profile.items()  
                    }  
                    for profile in resolved_liquid_profiles  
                ],  
                "profile_coupling_note": (  
                    "Liquid couples train.weight_decay inside each robustness profile so bridge/cell knobs stay tied "  
                    "while the selected execution profile controls the search budget."  
                ),  
            },  
        },  
        "train_reports": {  
            model_name: {  
                "device": train_result.metadata["train_report"]["device"],  
                "requested_device": train_result.metadata["train_report"]["requested_device"],  
                "epochs_run": len(train_result.metadata["train_report"].get("train_epoch_losses") or []),  
                "early_stop_patience": train_result.metadata["train_report"].get("early_stop_patience"),  
            }  
            for model_name, train_result in train_results.items()  
        },  
        "selected_estimator_cfgs": classical_search_result["selected_estimator_cfgs"],  
    }  


def _build_baseline_tuning_audit(classical_search_result: dict[str, Any]) -> dict[str, Any]:
    """Build the baseline tuning audit block."""

    search_audit = dict(classical_search_result.get("search_audit") or {})
    selected_estimator_cfgs = dict(classical_search_result.get("selected_estimator_cfgs") or {})
    expected_methods = ["ekf", "robust_ekf", "fgo"]
    method_entries: dict[str, Any] = {}
    methods_with_nonempty_overrides: list[str] = []
    selected_override_count_total = 0
    for method_name in expected_methods:
        method_audit = dict(search_audit.get(method_name) or {})
        selected_overrides = dict(method_audit.get("selected_overrides") or {})
        selected_override_count = len(selected_overrides)
        selected_override_count_total += selected_override_count
        if selected_override_count > 0:
            methods_with_nonempty_overrides.append(method_name)
        method_entries[method_name] = {
            "selected_signature": str(method_audit.get("selected_signature") or ""),
            "selected_overrides": selected_overrides,
            "selected_override_count": selected_override_count,
            "grid_report_count": len(method_audit.get("grid_reports") or []),
            "selected_cfg_present": method_name in selected_estimator_cfgs,
            "selected_cfg_name": str((selected_estimator_cfgs.get(method_name) or {}).get("name") or method_name),
        }
    actual_methods = [method_name for method_name in expected_methods if method_name in selected_estimator_cfgs]
    return {
        "status": "ok" if actual_methods == expected_methods else "partial",
        "comparison_surface": {
            "expected_methods": expected_methods,
            "actual_methods": actual_methods,
            "all_expected_methods_present": actual_methods == expected_methods,
            "missing_methods": [method_name for method_name in expected_methods if method_name not in actual_methods],
        },
        "methods": method_entries,
        "shared_findings": {
            "baseline_reference_method": "ekf",
            "methods_with_nonempty_overrides": methods_with_nonempty_overrides,
            "selected_override_count_total": selected_override_count_total,
            "selected_cfg_methods": actual_methods,
            "all_methods_have_selected_cfgs": actual_methods == expected_methods,
        },
        "notes": [
            "baseline_tuning_audit exposes the selected classical search outputs that were actually chosen for downstream evaluation",
            "selected_overrides are the direct evidence of what changed from the base estimator defaults",
            "empty selected_overrides means the selected classical baseline stayed at the search-default configuration for that method",
        ],
    }


def _audit_payload_has_positive_signal(payload: Mapping[str, Any]) -> bool:
    """Return whether an audit payload exposes positive evidence."""

    for value in payload.values():
        if isinstance(value, bool):
            if value:
                return True
            continue
        if isinstance(value, (int, float)):
            if float(value) > 0.0:
                return True
            continue
        if isinstance(value, str):
            if value.strip():
                return True
            continue
        if isinstance(value, Mapping):
            if _audit_payload_has_positive_signal(value):
                return True
            continue
        if isinstance(value, (list, tuple, set)):
            if any(bool(item) for item in value):
                return True
            continue
    return False


def _teacher_quality_audit_is_effective(teacher_quality_audit: Mapping[str, Any]) -> bool:
    """Return whether teacher-quality evidence is present and effective."""

    if not teacher_quality_audit:
        return False
    status = str(teacher_quality_audit.get("status") or "").strip().lower()
    if status:
        return status == "ok"
    return _audit_payload_has_positive_signal(teacher_quality_audit)


def _training_stability_audit_is_effective(training_stability_audit: Mapping[str, Any]) -> bool:
    """Return whether training-stability evidence is present and effective."""

    if not training_stability_audit:
        return False
    status = str(training_stability_audit.get("status") or "").strip().lower()
    if status != "ok":
        return False
    finite_train_losses = training_stability_audit.get("finite_train_losses")
    if finite_train_losses is not None and not bool(finite_train_losses):
        return False
    return _audit_payload_has_positive_signal(training_stability_audit)


def _split_audit_is_effective(split_audit: Mapping[str, Any]) -> bool:
    """Return whether split-audit evidence is present and effective."""

    if not split_audit:
        return False
    split_strategy = str(split_audit.get("split_strategy") or "").strip()
    return bool(split_strategy)


def _build_data_chain_evidence_audit(
    *,
    split_manifest: dict[str, Any],
    leak_report: dict[str, Any],
    train_results: dict[str, Any],
) -> dict[str, Any]:
    """Build the teacher-quality evidence audit block."""

    split_manifest = dict(split_manifest or {})
    leak_report = dict(leak_report or {})
    models: dict[str, Any] = {}
    models_with_sample_report: list[str] = []
    models_with_split_audit: list[str] = []
    models_with_teacher_quality_audit: list[str] = []
    observed_split_strategies: list[str] = []
    for model_name, train_result in train_results.items():
        metadata = dict(getattr(train_result, "metadata", {}) or {})
        sample_report = dict(metadata.get("sample_report") or {})
        split_audit = dict(sample_report.get("split_audit") or {})
        teacher_quality_audit = dict(sample_report.get("teacher_quality_audit") or {})
        sample_report_effective = bool(sample_report)
        split_audit_effective = _split_audit_is_effective(split_audit)
        teacher_quality_effective = _teacher_quality_audit_is_effective(teacher_quality_audit)
        if sample_report_effective:
            models_with_sample_report.append(model_name)
        if split_audit_effective:
            models_with_split_audit.append(model_name)
            split_strategy = str(split_audit.get("split_strategy") or "")
            if split_strategy:
                observed_split_strategies.append(split_strategy)
        if teacher_quality_effective:
            models_with_teacher_quality_audit.append(model_name)
        missing_evidence = []
        if not sample_report_effective:
            missing_evidence.append("sample_report")
        if not split_audit_effective:
            missing_evidence.append("split_audit")
        if not teacher_quality_effective:
            missing_evidence.append("teacher_quality_audit")
        models[model_name] = {
            "evidence_strength": "direct" if not missing_evidence else "partial",
            "missing_evidence": missing_evidence,
            "usable_sample_count": sample_report.get("usable_sample_count"),
            "usable_sample_count_by_modality": dict(sample_report.get("usable_sample_count_by_modality") or {}),
            "train_window_count": sample_report.get("train_window_count"),
            "val_window_count": sample_report.get("val_window_count"),
            "train_split_ids": list(sample_report.get("train_split_ids") or []),
            "val_split_ids": list(sample_report.get("val_split_ids") or []),
            "split_audit": split_audit,
            "teacher_quality_audit": teacher_quality_audit,
            "geometry_bias_teacher": dict(sample_report.get("geometry_bias_teacher") or {}),
        }
    expected_models = sorted(train_results)
    unique_split_strategies = sorted(set(item for item in observed_split_strategies if item))
    return {
        "status": (
            "ok"
            if bool(leak_report.get("is_clean")) and all(models[name]["evidence_strength"] == "direct" for name in expected_models)
            else "partial"
        ),
        "global_data_chain": {
            "split_manifest_present": bool(split_manifest),
            "train_ids": list(split_manifest.get("train_ids") or []),
            "val_ids": list(split_manifest.get("val_ids") or []),
            "test_ids": list(split_manifest.get("test_ids") or []),
            "leak_report_is_clean": bool(leak_report.get("is_clean")),
            "leak_report": leak_report,
        },
        "expected_models": expected_models,
        "models": models,
        "shared_findings": {
            "models_with_sample_report": models_with_sample_report,
            "models_with_split_audit": models_with_split_audit,
            "models_with_teacher_quality_audit": models_with_teacher_quality_audit,
            "observed_split_strategies": unique_split_strategies,
            "shared_split_strategy_consistent": len(unique_split_strategies) <= 1,
            "all_models_have_sample_report": len(models_with_sample_report) == len(expected_models),
            "all_models_have_split_audit": len(models_with_split_audit) == len(expected_models),
            "all_models_have_teacher_quality_audit": len(models_with_teacher_quality_audit) == len(expected_models),
        },
        "notes": [
            "data_chain_evidence_audit combines the top-level split/leak contract with per-model sample_report evidence",
            "teacher_quality_audit and split_audit come only from the train pipeline sample_report chain",
            "partial means the paper run metadata cannot yet prove the full data chain for every trained model",
        ],
    }


def _build_training_stability_evidence_audit(train_results: dict[str, Any]) -> dict[str, Any]:
    """Build the training-stability evidence audit block."""

    def _artifact_exists(value: Any) -> bool:
        path_text = str(value or "").strip()
        return bool(path_text) and Path(path_text).is_file()

    models: dict[str, Any] = {}
    models_with_stability_audit: list[str] = []
    models_with_loss_diagnostics: list[str] = []
    models_with_epoch_predictions: list[str] = []
    all_epoch_counts: dict[str, int] = {}
    for model_name, train_result in train_results.items():
        metadata = dict(getattr(train_result, "metadata", {}) or {})
        train_report = dict(metadata.get("train_report") or {})
        training_stability_audit = dict(train_report.get("training_stability_audit") or {})
        loss_diagnostics_path = str(train_report.get("loss_diagnostics_path") or "")
        epoch_predictions_path = str(train_report.get("epoch_predictions_vs_targets_path") or "")
        training_stability_effective = _training_stability_audit_is_effective(training_stability_audit)
        if training_stability_effective:
            models_with_stability_audit.append(model_name)
        if _artifact_exists(loss_diagnostics_path):
            models_with_loss_diagnostics.append(model_name)
        if _artifact_exists(epoch_predictions_path):
            models_with_epoch_predictions.append(model_name)
        epoch_count = len(train_report.get("train_epoch_losses") or [])
        all_epoch_counts[model_name] = epoch_count
        missing_evidence = []
        if not training_stability_effective:
            missing_evidence.append("training_stability_audit")
        if not _artifact_exists(loss_diagnostics_path):
            missing_evidence.append("loss_diagnostics_path")
        if not _artifact_exists(epoch_predictions_path):
            missing_evidence.append("epoch_predictions_vs_targets_path")
        if epoch_count <= 0:
            missing_evidence.append("train_epoch_losses")
        models[model_name] = {
            "trainer_mode": str(train_report.get("trainer_mode") or ""),
            "evidence_strength": "direct" if not missing_evidence else "partial",
            "missing_evidence": missing_evidence,
            "epochs_configured": train_report.get("epochs"),
            "epochs_run": epoch_count,
            "best_epoch": train_report.get("best_epoch"),
            "best_loss": train_report.get("best_loss"),
            "best_selection_score": train_report.get("best_selection_score"),
            "early_stop_patience": train_report.get("early_stop_patience"),
            "early_stopped": train_report.get("early_stopped"),
            "loss_diagnostics_path": loss_diagnostics_path,
            "epoch_predictions_vs_targets_path": epoch_predictions_path,
            "training_stability_audit": training_stability_audit,
        }
    expected_models = sorted(train_results)
    return {
        "status": "ok" if all(models[name]["evidence_strength"] == "direct" for name in expected_models) else "partial",
        "expected_models": expected_models,
        "models": models,
        "shared_findings": {
            "models_with_training_stability_audit": models_with_stability_audit,
            "models_with_loss_diagnostics": models_with_loss_diagnostics,
            "models_with_epoch_predictions": models_with_epoch_predictions,
            "all_models_have_training_stability_audit": len(models_with_stability_audit) == len(expected_models),
            "all_models_have_loss_diagnostics": len(models_with_loss_diagnostics) == len(expected_models),
            "all_models_have_epoch_predictions": len(models_with_epoch_predictions) == len(expected_models),
            "epoch_counts": all_epoch_counts,
        },
        "notes": [
            "training_stability_evidence_audit surfaces whether each trained model exported explicit convergence and stability artifacts",
            "direct evidence requires a training_stability_audit payload plus both loss_diagnostics and epoch_predictions_vs_targets artifacts",
            "partial means the paper-level report cannot yet prove stable convergence for every trained model",
        ],
    }


def _build_comparison_readiness_audit(
    *,
    fairness_audit: dict[str, Any],
    baseline_tuning_audit: dict[str, Any],
    data_chain_evidence_audit: dict[str, Any],
    training_stability_evidence_audit: dict[str, Any],
    training_evidence_audit: dict[str, Any],
    measurement_noise_linkage_audit: dict[str, Any],
    evaluation_coverage_audit: dict[str, Any],
    held_out_core_evaluation: dict[str, Any],
    main_conclusion: dict[str, Any],
) -> dict[str, Any]:
    """Build the measurement-noise linkage audit block."""

    fairness_surface = dict(fairness_audit.get("comparison_surface") or {})
    coverage_surface = dict(evaluation_coverage_audit.get("comparison_surface") or {})
    main_surface = dict(main_conclusion.get("comparison_surface") or {})
    expected_methods = list(
        fairness_surface.get("expected_methods")
        or coverage_surface.get("expected_methods")
        or main_surface.get("expected_methods")
        or []
    )
    fairness_complete = bool(fairness_surface.get("all_expected_methods_present"))
    coverage_complete = bool(coverage_surface.get("all_expected_methods_present"))
    main_complete = bool(main_surface.get("all_expected_methods_present"))
    baseline_tuning_status = str(baseline_tuning_audit.get("status") or "")
    data_chain_status = str(data_chain_evidence_audit.get("status") or "")
    training_stability_status = str(training_stability_evidence_audit.get("status") or "")
    training_evidence_status = str(training_evidence_audit.get("status") or "")
    measurement_noise_linkage_status = str(measurement_noise_linkage_audit.get("status") or "")
    held_out_status = str(held_out_core_evaluation.get("status") or "")
    main_status = str(main_conclusion.get("status") or "")
    public_surface_status = str(main_conclusion.get("public_surface_status") or "")
    cross_factorial_surface_status = str(main_conclusion.get("cross_factorial_surface_status") or "")
    conclusion_ready = (
        fairness_complete
        and baseline_tuning_status == "ok"
        and data_chain_status == "ok"
        and training_stability_status == "ok"
        and training_evidence_status == "ok"
        and measurement_noise_linkage_status == "ok"
        and coverage_complete
        and main_complete
        and held_out_status == "ok"
        and main_status == "ok"
        and public_surface_status == "ok"
        and cross_factorial_surface_status == "ok"
    )
    return {
        "status": "ok" if conclusion_ready else "partial",
        "comparison_surface": {
            "expected_methods": expected_methods,
            "fairness_actual_methods": list(fairness_surface.get("actual_methods") or []),
            "coverage_leaderboard_methods": list(coverage_surface.get("leaderboard_methods") or []),
            "main_leaderboard_methods": list(main_surface.get("leaderboard_methods") or []),
            "fairness_all_expected_methods_present": fairness_complete,
            "coverage_all_expected_methods_present": coverage_complete,
            "main_all_expected_methods_present": main_complete,
        },
        "conclusion_gate": {
            "held_out_status": held_out_status,
            "main_conclusion_status": main_status,
            "public_surface_status": public_surface_status,
            "cross_factorial_surface_status": cross_factorial_surface_status,
            "baseline_tuning_status": baseline_tuning_status,
            "data_chain_status": data_chain_status,
            "training_stability_status": training_stability_status,
            "training_evidence_status": training_evidence_status,
            "measurement_noise_linkage_status": measurement_noise_linkage_status,
            "claim_recommendation": str(main_conclusion.get("claim_recommendation") or ""),
            "liquid_rank": held_out_core_evaluation.get("liquid_rank"),
            "leader_method": held_out_core_evaluation.get("leader_method"),
            "ready_for_full_fair_comparison_claim": conclusion_ready,
        },
        "missing_requirements": [
            requirement
            for requirement, satisfied in (
                ("fairness_expected_methods_complete", fairness_complete),
                ("baseline_tuning_status_ok", baseline_tuning_status == "ok"),
                ("data_chain_status_ok", data_chain_status == "ok"),
                ("training_stability_status_ok", training_stability_status == "ok"),
                ("training_evidence_status_ok", training_evidence_status == "ok"),
                ("measurement_noise_linkage_status_ok", measurement_noise_linkage_status == "ok"),
                ("coverage_expected_methods_complete", coverage_complete),
                ("main_conclusion_expected_methods_complete", main_complete),
                ("held_out_status_ok", held_out_status == "ok"),
                ("main_conclusion_status_ok", main_status == "ok"),
                ("public_surface_status_ok", public_surface_status == "ok"),
                ("cross_factorial_surface_status_ok", cross_factorial_surface_status == "ok"),
            )
            if not satisfied
        ],
        "notes": [
            "comparison_readiness_audit is the top-level gate for whether the current run supports a full five-method paper claim",
            "it stays partial when any of fairness surface, baseline-tuning evidence, data-chain evidence, training-stability evidence, training-flow evidence, measurement-noise linkage evidence, held-out ranking, public surface, or cross-factorial surface is incomplete",
            "this audit summarizes whether lower-level audits jointly support a publication-grade comparison claim and therefore requires comparison-surface completeness, baseline-tuning completeness, measurement-noise linkage completeness, and training-evidence completeness",
        ],
    }


def _build_baseline_tuning_readiness_summary(baseline_tuning_audit: dict[str, Any]) -> dict[str, Any]:
    """Build the baseline-tuning readiness summary block."""

    comparison_surface = dict(baseline_tuning_audit.get("comparison_surface") or {})
    shared_findings = dict(baseline_tuning_audit.get("shared_findings") or {})
    return {
        "status": str(baseline_tuning_audit.get("status") or ""),
        "expected_methods": list(comparison_surface.get("expected_methods") or []),
        "actual_methods": list(comparison_surface.get("actual_methods") or []),
        "all_expected_methods_present": bool(comparison_surface.get("all_expected_methods_present")),
        "all_methods_have_selected_cfgs": bool(shared_findings.get("all_methods_have_selected_cfgs")),
        "methods_with_nonempty_overrides": list(shared_findings.get("methods_with_nonempty_overrides") or []),
        "selected_override_count_total": int(shared_findings.get("selected_override_count_total") or 0),
    }


def _build_data_chain_readiness_summary(data_chain_evidence_audit: dict[str, Any]) -> dict[str, Any]:
    """Build the data-chain readiness summary block."""

    global_data_chain = dict(data_chain_evidence_audit.get("global_data_chain") or {})
    shared_findings = dict(data_chain_evidence_audit.get("shared_findings") or {})
    return {
        "status": str(data_chain_evidence_audit.get("status") or ""),
        "expected_models": list(data_chain_evidence_audit.get("expected_models") or []),
        "leak_report_is_clean": bool(global_data_chain.get("leak_report_is_clean")),
        "all_models_have_sample_report": bool(shared_findings.get("all_models_have_sample_report")),
        "all_models_have_split_audit": bool(shared_findings.get("all_models_have_split_audit")),
        "all_models_have_teacher_quality_audit": bool(shared_findings.get("all_models_have_teacher_quality_audit")),
        "shared_split_strategy_consistent": bool(shared_findings.get("shared_split_strategy_consistent")),
        "observed_split_strategies": list(shared_findings.get("observed_split_strategies") or []),
    }


def _build_training_stability_readiness_summary(
    training_stability_evidence_audit: dict[str, Any],
) -> dict[str, Any]:
    """Build the training-stability readiness summary block."""

    shared_findings = dict(training_stability_evidence_audit.get("shared_findings") or {})
    return {
        "status": str(training_stability_evidence_audit.get("status") or ""),
        "expected_models": list(training_stability_evidence_audit.get("expected_models") or []),
        "all_models_have_training_stability_audit": bool(
            shared_findings.get("all_models_have_training_stability_audit")
        ),
        "all_models_have_loss_diagnostics": bool(shared_findings.get("all_models_have_loss_diagnostics")),
        "all_models_have_epoch_predictions": bool(shared_findings.get("all_models_have_epoch_predictions")),
        "epoch_counts": dict(shared_findings.get("epoch_counts") or {}),
    }


def _build_training_evidence_readiness_summary(training_evidence_audit: dict[str, Any]) -> dict[str, Any]:
    """Build the training-evidence readiness summary block."""

    shared_findings = dict(training_evidence_audit.get("shared_findings") or {})
    return {
        "status": str(training_evidence_audit.get("status") or ""),
        "expected_models": list(training_evidence_audit.get("expected_models") or []),
        "models_with_split_audit": list(shared_findings.get("models_with_split_audit") or []),
        "models_with_teacher_quality_audit": list(shared_findings.get("models_with_teacher_quality_audit") or []),
        "models_with_training_stability_audit": list(
            shared_findings.get("models_with_training_stability_audit") or []
        ),
        "models_with_training_flow_contract": list(shared_findings.get("models_with_training_flow_contract") or []),
        "shared_split_strategy_consistent": bool(shared_findings.get("shared_split_strategy_consistent")),
    }


def _build_measurement_noise_linkage_audit(
    train_results: dict[str, Any],
    evaluation_coverage_audit: dict[str, Any],
) -> dict[str, Any]:
    """Build the measurement-noise linkage audit from training and coverage evidence."""

    expected_models = ["liquid_ekf", "lstm_ekf"]
    models: dict[str, Any] = {}
    models_with_target_contract: list[str] = []
    models_with_sample_report: list[str] = []
    models_with_axis_linkage_contract: list[str] = []
    for model_name in expected_models:
        train_result = train_results.get(model_name)
        metadata = dict(getattr(train_result, "metadata", {}) or {}) if train_result is not None else {}
        target_contract = dict(metadata.get("target_contract") or {})
        sample_report = dict(metadata.get("sample_report") or {})
        head_sources = dict(target_contract.get("head_sources") or {})
        confidence_materialization = str(target_contract.get("confidence_materialization") or "")
        has_target_contract = bool(target_contract)
        has_sample_report = bool(sample_report)
        if has_target_contract:
            models_with_target_contract.append(model_name)
        if has_sample_report:
            models_with_sample_report.append(model_name)
        axis_linkage_requirements = {
            "risk_is_pre_bridge_alignment_only": str(target_contract.get("risk_semantics") or "") == "alignment_risk_only_pre_bridge_base_risk",
            "observation_risk_stays_in_bridge_and_scaling_consumers": "observation_risk remains an audit trace" in str(head_sources.get("risk") or ""),
            "scaling_is_noise_inflation_only": (
                str(target_contract.get("scaling_semantics") or "")
                == "teacher_free_modality_specific_noise_inflation_only"
            ),
            "uwb_scaling_mentions_async": "modality_gap_dt staleness" in str(head_sources.get("uwb_scaling") or ""),
            "uwb_scaling_mentions_geometry": "geometric bias severity" in str(head_sources.get("uwb_scaling") or ""),
            "vio_scaling_mentions_async": "modality_gap_dt staleness" in str(head_sources.get("vio_scaling") or ""),
            "vio_scaling_mentions_visual": (
                "tracked_features" in str(head_sources.get("vio_scaling") or "")
                and "reproj_err" in str(head_sources.get("vio_scaling") or "")
            ),
            "risk_head_documents_bridge_consumption": "downstream bridge noise logic" in str(head_sources.get("risk") or ""),
            "confidence_materialization_is_scaling_heads": "modality-specific scaling heads" in confidence_materialization,
        }
        if has_target_contract and all(axis_linkage_requirements.values()):
            models_with_axis_linkage_contract.append(model_name)
        models[model_name] = {
            "evidence_strength": (
                "direct"
                if has_target_contract and has_sample_report and all(axis_linkage_requirements.values())
                else ("partial" if has_target_contract or has_sample_report else "missing")
            ),
            "target_contract_present": has_target_contract,
            "sample_report_present": has_sample_report,
            "risk_semantics": str(target_contract.get("risk_semantics") or ""),
            "scaling_semantics": str(target_contract.get("scaling_semantics") or ""),
            "confidence_materialization": confidence_materialization,
            "head_sources": {
                "risk": str(head_sources.get("risk") or ""),
                "uwb_scaling": str(head_sources.get("uwb_scaling") or ""),
                "vio_scaling": str(head_sources.get("vio_scaling") or ""),
            },
            "axis_linkage_requirements": axis_linkage_requirements,
            "bridge_thresholds_present": bool(target_contract.get("bridge_thresholds")),
            "geometry_bias_teacher": dict(sample_report.get("geometry_bias_teacher") or {}),
            "teacher_quality_audit": dict(sample_report.get("teacher_quality_audit") or {}),
            "missing_evidence": [
                item
                for item, ok in (
                    ("target_contract", has_target_contract),
                    ("sample_report", has_sample_report),
                    ("axis_linkage_contract", all(axis_linkage_requirements.values())),
                )
                if not ok
            ],
        }

    core_axis_coverage = dict(evaluation_coverage_audit.get("core_axis_coverage") or {})
    held_out_axis_coverage = dict(evaluation_coverage_audit.get("held_out_axis_coverage") or {})
    status = (
        "ok"
        if len(models_with_axis_linkage_contract) == len(expected_models)
        and bool(core_axis_coverage.get("async"))
        and bool(core_axis_coverage.get("nlos"))
        and bool(core_axis_coverage.get("visual"))
        else "partial"
    )
    return {
        "status": status,
        "expected_models": expected_models,
        "models": models,
        "coverage": {
            "core_axis_coverage": core_axis_coverage,
            "held_out_axis_coverage": held_out_axis_coverage,
        },
        "shared_findings": {
            "models_with_target_contract": models_with_target_contract,
            "models_with_sample_report": models_with_sample_report,
            "models_with_axis_linkage_contract": models_with_axis_linkage_contract,
            "all_models_have_target_contract": len(models_with_target_contract) == len(expected_models),
            "all_models_have_sample_report": len(models_with_sample_report) == len(expected_models),
            "all_models_have_axis_linkage_contract": len(models_with_axis_linkage_contract) == len(expected_models),
            "core_axis_noise_surface_complete": bool(core_axis_coverage.get("async"))
            and bool(core_axis_coverage.get("nlos"))
            and bool(core_axis_coverage.get("visual")),
        },
        "notes": [
            "measurement_noise_linkage_audit proves that async, NLOS, and visual degradation are represented through the training-side target contract and not only through scenario naming",
            "risk remains alignment-only, while modality-specific scaling heads carry async staleness, geometric degradation, and visual degradation into the bridge noise-consumption path",
            "core axis coverage is included so the linkage contract can be interpreted together with the executed paper scenario surface",
        ],
    }


def _build_comparison_readiness_summary(comparison_readiness_audit: dict[str, Any]) -> dict[str, Any]:
    """Build the comparison-readiness summary block."""

    conclusion_gate = dict(comparison_readiness_audit.get("conclusion_gate") or {})
    return {
        "status": str(comparison_readiness_audit.get("status") or ""),
        "ready_for_full_fair_comparison_claim": bool(conclusion_gate.get("ready_for_full_fair_comparison_claim")),
        "claim_recommendation": str(conclusion_gate.get("claim_recommendation") or ""),
        "leader_method": str(conclusion_gate.get("leader_method") or ""),
        "liquid_rank": conclusion_gate.get("liquid_rank"),
        "held_out_status": str(conclusion_gate.get("held_out_status") or ""),
        "main_conclusion_status": str(conclusion_gate.get("main_conclusion_status") or ""),
        "public_surface_status": str(conclusion_gate.get("public_surface_status") or ""),
        "cross_factorial_surface_status": str(conclusion_gate.get("cross_factorial_surface_status") or ""),
        "baseline_tuning_status": str(conclusion_gate.get("baseline_tuning_status") or ""),
        "data_chain_status": str(conclusion_gate.get("data_chain_status") or ""),
        "training_stability_status": str(conclusion_gate.get("training_stability_status") or ""),
        "training_evidence_status": str(conclusion_gate.get("training_evidence_status") or ""),
        "measurement_noise_linkage_status": str(conclusion_gate.get("measurement_noise_linkage_status") or ""),
        "missing_requirements": list(comparison_readiness_audit.get("missing_requirements") or []),
    }


def _build_run_grade_contract(
    *,
    requested_public_datasets: list[str],
    allow_incomplete_paper_run: bool,
    neural_search_surface: Mapping[str, Any],
    comparison_readiness: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the run-grade contract for the current paper experiment run."""

    training_budget = dict(neural_search_surface.get("training_budget") or {})
    paper_grade_budget = bool(training_budget.get("paper_grade", False))
    diagnostic_only_profile = bool(training_budget.get("diagnostic_only", False))
    required_public_datasets = list(_SUPPORTED_PUBLIC_DATASETS)
    complete_public_surface_requested = set(requested_public_datasets) == set(required_public_datasets)
    comparison_ready = bool(comparison_readiness.get("ready_for_full_fair_comparison_claim", False))

    downgrade_drivers: list[str] = []
    if diagnostic_only_profile:
        downgrade_drivers.append("diagnostic_only_profile")
    if allow_incomplete_paper_run:
        downgrade_drivers.append("allow_incomplete_paper_run")
    if not complete_public_surface_requested:
        downgrade_drivers.append("public_surface_not_requested_in_full")

    if downgrade_drivers:
        requested_execution_grade = "diagnostic_only"
    elif paper_grade_budget:
        requested_execution_grade = "paper_grade"
    else:
        requested_execution_grade = "non_paper_budget"

    return {
        "requested_execution_grade": requested_execution_grade,
        "paper_grade_budget": paper_grade_budget,
        "diagnostic_only_profile": diagnostic_only_profile,
        "allow_incomplete_paper_run": bool(allow_incomplete_paper_run),
        "required_public_datasets": required_public_datasets,
        "requested_public_datasets": list(requested_public_datasets),
        "complete_public_surface_requested": complete_public_surface_requested,
        "ready_for_full_fair_comparison_claim": comparison_ready,
        "downgrade_drivers": downgrade_drivers,
        "claim_gate_blockers": list(comparison_readiness.get("missing_requirements") or []),
        "notes": [
            "paper_grade_budget describes the training budget regime only and does not by itself authorize publication-grade claims",
            "requested_execution_grade describes the actual execution resource tier and is independent from whether downstream evidence supports a publication-grade claim",
            "ready_for_full_fair_comparison_claim remains the top-level publication gate and must stay true before any full five-method paper claim is made",
        ],
    }


def _build_training_evidence_audit(
    train_results: dict[str, Any],
) -> dict[str, Any]:
    """Build the aggregate training-evidence audit block."""

    models: dict[str, Any] = {}
    split_strategies: dict[str, str] = {}
    models_with_split_audit: list[str] = []
    models_with_teacher_quality_audit: list[str] = []
    models_with_training_stability_audit: list[str] = []
    models_with_training_flow_contract: list[str] = []
    for model_name, train_result in train_results.items():
        metadata = dict(getattr(train_result, "metadata", {}) or {})
        train_report = dict(metadata.get("train_report") or {})
        sample_report = dict(metadata.get("sample_report") or {})
        training_flow_contract = dict(metadata.get("training_flow_contract") or {})
        split_audit = dict(sample_report.get("split_audit") or train_report.get("split_audit") or {})
        teacher_quality_audit = dict(sample_report.get("teacher_quality_audit") or {})
        training_stability_audit = dict(train_report.get("training_stability_audit") or {})
        split_audit_effective = _split_audit_is_effective(split_audit)
        teacher_quality_effective = _teacher_quality_audit_is_effective(teacher_quality_audit)
        training_stability_effective = _training_stability_audit_is_effective(training_stability_audit)
        if split_audit_effective:
            models_with_split_audit.append(model_name)
            split_strategy = str(split_audit.get("split_strategy") or "")
            if split_strategy:
                split_strategies[model_name] = split_strategy
        if teacher_quality_effective:
            models_with_teacher_quality_audit.append(model_name)
        if training_stability_effective:
            models_with_training_stability_audit.append(model_name)
        if training_flow_contract:
            models_with_training_flow_contract.append(model_name)
        missing_evidence = []
        if not train_report:
            missing_evidence.append("train_report")
        if not split_audit_effective:
            missing_evidence.append("split_audit")
        if not teacher_quality_effective:
            missing_evidence.append("teacher_quality_audit")
        if not training_stability_effective:
            missing_evidence.append("training_stability_audit")
        if not training_flow_contract:
            missing_evidence.append("training_flow_contract")
        models[model_name] = {
            "trainer_mode": str(train_report.get("trainer_mode") or training_flow_contract.get("trainer_mode") or ""),
            "evidence_strength": "direct" if not missing_evidence else "partial",
            "missing_evidence": missing_evidence,
            "train_report_path": str(train_report.get("report_path") or ""),
            "training_flow_contract_path": str(train_report.get("training_flow_contract_path") or ""),
            "split_audit": split_audit,
            "teacher_quality_audit": teacher_quality_audit,
            "training_stability_audit": training_stability_audit,
            "usable_sample_count": sample_report.get("usable_sample_count"),
            "usable_sample_count_by_modality": dict(sample_report.get("usable_sample_count_by_modality") or {}),
        }
    expected_models = sorted(train_results)
    observed_split_strategies = sorted(set(value for value in split_strategies.values() if value))
    return {
        "status": "ok" if all(models[name]["evidence_strength"] == "direct" for name in expected_models) else "partial",
        "expected_models": expected_models,
        "models": models,
        "shared_findings": {
            "models_with_split_audit": models_with_split_audit,
            "models_with_teacher_quality_audit": models_with_teacher_quality_audit,
            "models_with_training_stability_audit": models_with_training_stability_audit,
            "models_with_training_flow_contract": models_with_training_flow_contract,
            "observed_split_strategies": observed_split_strategies,
            "shared_split_strategy_consistent": len(observed_split_strategies) <= 1,
        },
        "notes": [
            "training_evidence_audit surfaces whether split, teacher-quality, training-stability, and training-flow evidence are directly available from train pipeline artifacts",
            "partial means the current train_result payload is missing one or more direct evidence objects even if training completed",
        ],
    }


def _resolve_main_conclusion_methods(  
    estimator_cfgs: dict[str, dict[str, Any]],  
    model_cfgs: dict[str, dict[str, Any]],  
) -> list[str]:  
    ordered_methods = ("ekf", "robust_ekf", "fgo", "lstm_ekf", "liquid_ekf")  
    available_methods = set(estimator_cfgs) | set(model_cfgs)  
    return [method_name for method_name in ordered_methods if method_name in available_methods]  


def _normalize_leaderboard_method_names(leaderboard: list[dict[str, Any]]) -> list[str]:
    """Normalize leaderboard method names to the comparison-surface contract."""
    normalized_methods: list[str] = []
    for row in leaderboard:
        normalized_method = str(row.get("method_name") or "").strip()
        if normalized_method:
            normalized_methods.append(normalized_method)
    return normalized_methods


def _iter_ekf_init_sensitivity_profiles() -> list[dict[str, Any]]:  
    return [  
        {"name": "nominal", "overrides": {}},  
        {"name": "pos_small", "overrides": {"init_state.px": 0.5, "init_state.py": -0.5}},  
        {"name": "pos_large", "overrides": {"init_state.px": 1.5, "init_state.py": -1.5}},  
        {"name": "yaw_small_pos", "overrides": {"init_state.yaw": math.radians(10.0)}},
        {"name": "yaw_small_neg", "overrides": {"init_state.yaw": math.radians(-10.0)}},
        {"name": "yaw_large_pos", "overrides": {"init_state.yaw": math.radians(25.0)}},
        {"name": "yaw_large_neg", "overrides": {"init_state.yaw": math.radians(-25.0)}},
        {"name": "vel_bias", "overrides": {"init_state.vx": 0.5, "init_state.vy": -0.5}},  
    ]  


def _run_ekf_init_sensitivity_audit(  
    *,  
    split_manifest: dict[str, Any],  
    prepare_root: Path,  
    raw_root: Path,  
    estimator_cfgs: dict[str, dict[str, Any]],  
    output_root: Path,  
) -> dict[str, Any]:  
    test_ids = list(split_manifest.get("test_ids") or [])  
    if not test_ids:  
        return {  
            "status": "blocked",  
            "reason": "missing_test_ids",  
            "message": "EKF init sensitivity audit requires held-out test_ids",  
            "held_out_test_ids": [],  
        }  

    base_ekf_cfg = estimator_cfgs.get("ekf")  
    if not isinstance(base_ekf_cfg, dict):  
        return {  
            "status": "blocked",  
            "reason": "missing_ekf_cfg",  
            "message": "EKF init sensitivity audit requires selected ekf estimator cfg",  
            "held_out_test_ids": test_ids,  
        }  

    profile_reports: list[dict[str, Any]] = []  
    nominal_metrics: dict[str, dict[str, float]] | None = None  
    for profile in _iter_ekf_init_sensitivity_profiles():  
        profile_estimator_cfg = _apply_estimator_overrides(base_ekf_cfg, dict(profile["overrides"]))  
        profile_runs = [  
            _run_scoring_experiment(  
                config_name=config_name,  
                prepare_root=prepare_root,  
                raw_root=raw_root,  
                method_name="ekf",  
                estimator_cfgs={"ekf": profile_estimator_cfg},  
                model_cfgs={},  
                output_root=output_root / profile["name"],  
                seq_ids=test_ids,  
            )  
            for config_name in _EKF_INIT_SENSITIVITY_CONFIGS  
        ]  
        metrics_by_experiment = {  
            run["experiment_id"]: dict(run["metrics"])  
            for run in profile_runs  
        }  
        if profile["name"] == "nominal":  
            nominal_metrics = metrics_by_experiment  
        profile_reports.append(  
            {  
                "profile_name": profile["name"],  
                "overrides": dict(profile["overrides"]),  
                "score_vector": list(_score_candidate_lexicographically(metrics_by_experiment=metrics_by_experiment)),  
                "metrics_by_experiment": metrics_by_experiment,  
                "runs": [  
                    {  
                        "experiment_id": run["experiment_id"],  
                        "output_root": run["output_root"],  
                        "eval_stage_name": run["eval_stage_name"],  
                        "scene_sampling_seq_ids": list(run["scene_sampling_seq_ids"]),  
                    }  
                    for run in profile_runs  
                ],  
            }  
        )  

    if nominal_metrics is None:  
        raise RuntimeError("EKF init sensitivity audit failed to collect nominal profile metrics")  

    sensitivity_summary = []  
    for config_name in _EKF_INIT_SENSITIVITY_CONFIGS:  
        experiment_id = Path(config_name).stem  
        experiment_p95_values = {  
            report["profile_name"]: float(report["metrics_by_experiment"][experiment_id]["p95"])  
            for report in profile_reports  
        }  
        experiment_failure_values = {  
            report["profile_name"]: float(report["metrics_by_experiment"][experiment_id]["failure_rate"])  
            for report in profile_reports  
        }  
        nominal_p95 = max(float(nominal_metrics[experiment_id]["p95"]), 1e-9)  
        worst_profile = max(experiment_p95_values.items(), key=lambda item: item[1])[0]  
        best_profile = min(experiment_p95_values.items(), key=lambda item: item[1])[0]  
        sensitivity_summary.append(  
            {  
                "experiment_id": experiment_id,  
                "nominal_p95": float(nominal_metrics[experiment_id]["p95"]),  
                "nominal_failure_rate": float(nominal_metrics[experiment_id]["failure_rate"]),  
                "max_p95": max(experiment_p95_values.values()),  
                "min_p95": min(experiment_p95_values.values()),  
                "max_failure_rate": max(experiment_failure_values.values()),  
                "min_failure_rate": min(experiment_failure_values.values()),  
                "max_relative_p95_degradation": (  
                    max(experiment_p95_values.values()) - float(nominal_metrics[experiment_id]["p95"])  
                )  
                / nominal_p95,  
                "worst_profile": worst_profile,  
                "best_profile": best_profile,  
            }  
        )  

    return {  
        "status": "ok",  
        "held_out_test_ids": test_ids,  
        "profiles": profile_reports,  
        "summary": sensitivity_summary,  
    }  


def _build_embedded_runtime_proxy_audit(eval_root: Path, output_root: Path) -> dict[str, Any]:  
    runtime_table_path = eval_root / "plotting_inputs" / "runtime_table.json"  
    if not runtime_table_path.is_file():  
        return {  
            "status": "blocked",  
            "reason": "missing_runtime_table",  
            "message": "embedded runtime proxy audit requires plotting_inputs/runtime_table.json",  
            "runtime_table_path": str(runtime_table_path),  
        }  

    from liquidloc.common.io_utils import read_json
    runtime_rows = read_json(runtime_table_path)  
    if not isinstance(runtime_rows, list) or not runtime_rows:  
        return {  
            "status": "blocked",  
            "reason": "empty_runtime_table",  
            "message": "embedded runtime proxy audit requires non-empty runtime rows",  
            "runtime_table_path": str(runtime_table_path),  
        }  

    aggregated_by_method: dict[str, dict[str, float]] = {}  
    counts_by_method: dict[str, int] = {}  
    for row in runtime_rows:  
        method_name = str(row["method_name"])  
        summary = aggregated_by_method.setdefault(  
            method_name,  
            {  
                "latency_mean": 0.0,  
                "latency_p95": 0.0,  
                "params": 0.0,  
                "ram_peak": 0.0,  
            },  
        )  
        counts_by_method[method_name] = counts_by_method.get(method_name, 0) + 1  
        summary["latency_mean"] += float(row["latency_mean"])  
        summary["latency_p95"] += float(row["latency_p95"])  
        summary["params"] = float(row["params"])  
        summary["ram_peak"] = max(summary["ram_peak"], float(row["ram_peak"]))  

    aggregated_rows = []  
    for method_name, summary in aggregated_by_method.items():  
        case_count = counts_by_method[method_name]  
        latency_mean = summary["latency_mean"] / case_count  
        latency_p95 = summary["latency_p95"] / case_count  
        proxy_rows = []  
        for proxy_profile in _EMBEDDED_RUNTIME_PROXY_PROFILES:  
            fits_memory = summary["ram_peak"] <= float(proxy_profile["ram_limit_mb"])  
            fits_params = summary["params"] <= float(proxy_profile["params_limit"])  
            proxy_rows.append(  
                {  
                    "profile_name": proxy_profile["name"],  
                    "estimate_kind": "heuristic_proxy_not_hardware_measurement",  
                    "latency_scale": float(proxy_profile["latency_scale"]),  
                    "estimated_latency_mean_ms": latency_mean * float(proxy_profile["latency_scale"]),  
                    "estimated_latency_p95_ms": latency_p95 * float(proxy_profile["latency_scale"]),  
                    "ram_limit_mb": float(proxy_profile["ram_limit_mb"]),  
                    "params_limit": float(proxy_profile["params_limit"]),  
                    "fits_memory": fits_memory,  
                    "fits_params": fits_params,  
                    "fit_status": "fit" if (fits_memory and fits_params) else "blocked",  
                }  
            )  
        aggregated_rows.append(  
            {  
                "method_name": method_name,  
                "case_count": case_count,  
                "desktop_latency_mean_ms": latency_mean,  
                "desktop_latency_p95_ms": latency_p95,  
                "params": summary["params"],  
                "ram_peak_mb": summary["ram_peak"],  
                "proxy_estimates": proxy_rows,  
            }  
        )  

    return {  
        "status": "ok",  
        "runtime_table_path": str(runtime_table_path),  
        "notes": [  
            "proxy estimates are explicit heuristics, not hardware measurements",  
            "memory and parameter fit gates are reported separately from latency scaling",  
        ],  
        "methods": sorted(aggregated_rows, key=lambda row: row["method_name"]),  
    }  


def _run_cross_factorial_interaction_audit(  
    *,  
    split_manifest: dict[str, Any],  
    prepare_root: Path,  
    raw_root: Path,  
    estimator_cfgs: dict[str, dict[str, Any]],  
    model_cfgs: dict[str, dict[str, Any]],  
    output_root: Path,  
) -> dict[str, Any]:  
    test_ids = list(split_manifest.get("test_ids") or [])  
    if not test_ids:  
        return {  
            "status": "blocked",  
            "reason": "missing_test_ids",  
            "message": "cross-factorial interaction audit requires held-out test_ids",  
            "held_out_test_ids": [],  
        }  

    run = _run_core_experiment(  
        config_name=_CROSS_FACTORIAL_CONFIG,  
        prepare_root=prepare_root,  
        raw_root=raw_root,  
        estimator_cfgs=estimator_cfgs,  
        model_cfgs=model_cfgs,  
        output_root=output_root,  
        seq_ids=test_ids,  
    )  
    metric_table_path = Path(run["output_root"]) / "eval" / "metrics" / "metric_table.csv"  
    metric_views = _read_metric_table_views(metric_table_path)  
    metric_rows = metric_views["scene_summary_rows"]  
    cell_metrics: dict[str, dict[str, dict[str, float]]] = {}  
    for row in metric_rows:  
        scene_id = str(row.get("scene_id") or "")  
        method_name = str(row.get("method_name") or "")  
        metric_name = str(row.get("metric") or "")  
        if not scene_id or not method_name or not metric_name:  
            continue  
        cell_metrics.setdefault(scene_id, {}).setdefault(method_name, {})[metric_name] = float(row["value"])  

    required_metrics = ("p95", "failure_rate", "rmse", "mae")  
    methods = _resolve_main_conclusion_methods(estimator_cfgs, model_cfgs)  
    scene_rows: list[dict[str, Any]] = []  
    coverage = {"A": set(), "N": set(), "V": set()}  
    collection_errors: list[dict[str, Any]] = []  
    leader_counts = {  
        metric_name: {method_name: 0 for method_name in methods}  
        for metric_name in ("p95", "failure_rate")  
    }  
    method_aggregate_values = {  
        method_name: {metric_name: [] for metric_name in required_metrics}  
        for method_name in methods  
    }  

    for scene_id, scene_method_metrics in sorted(cell_metrics.items()):  
        axes = _parse_scene_axes(scene_id)  
        for axis_name in ("A", "N", "V"):  
            axis_levels = axes.get(axis_name)  
            if axis_levels:  
                coverage[axis_name].update(axis_levels)  
        leader_by_metric: dict[str, str] = {}  
        for metric_name in required_metrics:  
            candidates: list[tuple[str, float]] = []  
            for method_name in methods:  
                metric_value = scene_method_metrics.get(method_name, {}).get(metric_name)  
                if metric_value is None:  
                    collection_errors.append(  
                        {  
                            "scene_id": scene_id,  
                            "method_name": method_name,  
                            "metric_name": metric_name,  
                        }  
                    )  
                    continue  
                method_aggregate_values[method_name][metric_name].append(float(metric_value))  
                candidates.append((method_name, float(metric_value)))  
            if len(candidates) == len(methods):  
                leader_method = min(candidates, key=lambda item: item[1])[0]  
                leader_by_metric[metric_name] = leader_method  
                if metric_name in leader_counts:  
                    leader_counts[metric_name][leader_method] += 1  

        scene_rows.append(  
            {  
                "scene_id": scene_id,  
                "axes": axes,  
                "leader_by_metric": leader_by_metric,  
                "method_metrics": {  
                    method_name: {  
                        metric_name: scene_method_metrics.get(method_name, {}).get(metric_name)  
                        for metric_name in required_metrics  
                    }  
                    for method_name in methods  
                },  
            }  
        )  

    expected_scene_count = 4 * 4 * 4  
    unique_scene_count = len(scene_rows)  
    coverage_ok = unique_scene_count == expected_scene_count and all(len(coverage[axis_name]) == 4 for axis_name in coverage)  
    if collection_errors:  
        coverage_ok = False  

    method_summary = {  
        method_name: {  
            f"mean_{metric_name}": (  
                float(sum(values) / len(values))  
                if values  
                else None  
            )  
            for metric_name, values in metric_map.items()  
        }  
        for method_name, metric_map in method_aggregate_values.items()  
    }  
    liquid_p95_wins = int(leader_counts["p95"].get("liquid_ekf", 0))  
    liquid_failure_wins = int(leader_counts["failure_rate"].get("liquid_ekf", 0))  
    scene_count_denominator = max(unique_scene_count, 1)  
    return {  
        "status": "ok" if coverage_ok else "blocked",  
        "reason": None if coverage_ok else "incomplete_factorial_coverage",  
        "message": (  
            "cross-factorial A/N/V interaction audit completed on held-out test_ids"  
            if coverage_ok  
            else "cross-factorial interaction audit did not cover the full 4x4x4 A/N/V matrix cleanly"  
        ),  
        "held_out_test_ids": test_ids,  
        "experiment_id": Path(_CROSS_FACTORIAL_CONFIG).stem,  
        "output_root": run["output_root"],  
        "eval_stage_name": run["eval_stage_name"],  
        "metric_table_path": str(metric_table_path),  
        "scene_count": unique_scene_count,  
        "expected_scene_count": expected_scene_count,  
        "axes_coverage": {axis_name: sorted(levels) for axis_name, levels in coverage.items()},  
        "leader_counts": leader_counts,  
        "liquid_scene_win_rates": {  
            "p95": liquid_p95_wins / scene_count_denominator,  
            "failure_rate": liquid_failure_wins / scene_count_denominator,  
        },  
        "method_summary": method_summary,  
        "scene_rows": scene_rows,  
        "collection_errors": collection_errors,  
    }  


def _run_held_out_core_evaluation(  
    *,  
    split_manifest: dict[str, Any],  
    prepare_root: Path,  
    raw_root: Path,  
    estimator_cfgs: dict[str, dict[str, Any]],  
    model_cfgs: dict[str, dict[str, Any]],  
    output_root: Path,  
) -> dict[str, Any]:  
    test_ids = list(split_manifest.get("test_ids") or [])  
    split_provenance = {  
        "split_kind": "held_out_test_ids",
        "family_isolation_required": True,
        "test_ids": list(test_ids),
        "train_ids": list(split_manifest.get("train_ids") or []),
        "val_ids": list(split_manifest.get("val_ids") or []),
    }
    if not test_ids:  
        return {  
            "status": "blocked",  
            "reason": "missing_test_ids",  
            "message": "final main conclusion must use held-out test_ids; val-only conclusions are not allowed",  
            "held_out_test_ids": [],  
            "split_manifest": split_manifest,  
            "split_provenance": split_provenance,
        }  

    methods = _resolve_main_conclusion_methods(estimator_cfgs, model_cfgs)  
    if "ekf" not in methods:  
        raise RuntimeError("held-out main conclusion requires ekf as baseline reference")  
    priority_metrics = _resolve_conclusion_priority_metrics()  

    held_out_runs = [  
        _run_core_experiment(  
            config_name=config_name,  
            prepare_root=prepare_root,  
            raw_root=raw_root,  
            estimator_cfgs=estimator_cfgs,  
            model_cfgs=model_cfgs,  
            output_root=output_root,  
            seq_ids=test_ids,  
        )  
        for config_name in _HELD_OUT_CONCLUSION_CONFIGS  
    ]  

    metrics_by_method: dict[str, dict[str, dict[str, float]]] = {method_name: {} for method_name in methods}  
    collection_errors: list[dict[str, str]] = []  
    for run in held_out_runs:  
        experiment_output_root = Path(run["output_root"])  
        for method_name in methods:  
            try:  
                metrics_by_method[method_name][run["experiment_id"]] = _collect_search_metrics(  
                    experiment_output_root=experiment_output_root,  
                    method_name=method_name,  
                )  
            except RuntimeError as exc:  
                collection_errors.append(  
                    {  
                        "experiment_id": run["experiment_id"],  
                        "method_name": method_name,  
                        "error": str(exc),  
                    }  
                )  

    incomplete_methods = [  
        method_name  
        for method_name, metrics_by_experiment in metrics_by_method.items()  
        if any(Path(config_name).stem not in metrics_by_experiment for config_name in _HELD_OUT_CONCLUSION_CONFIGS)  
    ]  
    if collection_errors or incomplete_methods:  
        return {  
            "status": "blocked",  
            "reason": "held_out_metrics_incomplete",  
            "message": "held-out main conclusion could not collect a complete metric set for every method",  
            "held_out_test_ids": test_ids,  
            "split_manifest": split_manifest,  
            "split_provenance": split_provenance,
            "held_out_runs": [  
                {  
                    "experiment_id": run["experiment_id"],  
                    "output_root": run["output_root"],  
                    "eval_stage_name": run["eval_stage_name"],  
                    "scene_sampling_seq_ids": list(run["scene_sampling_seq_ids"]),  
                }  
                for run in held_out_runs  
            ],  
            "collection_errors": collection_errors,  
            "incomplete_methods": incomplete_methods,  
        }  

    leaderboard = sorted(  
        (  
            {  
                "method_name": method_name,  
                "score_vector": list(  
                    _build_held_out_protocol_rank_vector(
                        _summarize_held_out_method_metrics(metrics_by_method[method_name]),
                        priority_metrics=priority_metrics,
                    )
                ),
                "protocol_summary": _summarize_held_out_method_metrics(metrics_by_method[method_name]),  
                "search_score_vector": list(  
                    _score_candidate_lexicographically(metrics_by_experiment=metrics_by_method[method_name])
                ),
                "metrics_by_experiment": metrics_by_method[method_name],  
            }  
            for method_name in methods  
        ),  
        key=lambda row: tuple(row["score_vector"]) + (str(row["method_name"]),),  
    )  
    leader = leaderboard[0]["method_name"]
    liquid_rank: int | None = None
    for index, row in enumerate(leaderboard):
        if row["method_name"] == "liquid_ekf":
            liquid_rank = index + 1
            break
    return {  
        "status": "ok" if leader == "liquid_ekf" else "degraded",  
        "reason": None if leader == "liquid_ekf" else "liquid_not_global_first_on_held_out",  
        "message": (  
            "liquid_ekf is held-out leader on the paper conclusion matrix"  
            if leader == "liquid_ekf"  
            else "held-out ranking does not support a global-first Liquid claim; shrink the claim to truthful slices or mechanism evidence"  
        ),  
        "held_out_test_ids": test_ids,  
        "split_manifest": split_manifest,  
        "split_provenance": split_provenance,  
        "conclusion_priority": list(priority_metrics),  
        "leader_method": leader,  
        "liquid_rank": liquid_rank,  
        "claim_recommendation": "global_first" if leader == "liquid_ekf" else "shrink_to_truthful_scope",  
        "leaderboard": leaderboard,  
        "held_out_runs": [  
            {  
                "experiment_id": run["experiment_id"],  
                "output_root": run["output_root"],  
                "eval_stage_name": run["eval_stage_name"],  
                "scene_sampling_seq_ids": list(run["scene_sampling_seq_ids"]),  
            }  
            for run in held_out_runs  
        ],  
    }  


def _build_main_conclusion_report(  
    *,  
    split_manifest: dict[str, Any],  
    held_out_core_evaluation: dict[str, Any],  
    cross_factorial_interactions: dict[str, Any],  
    public_runs: list[dict[str, Any]],  
    requested_public_datasets: list[str],  
) -> dict[str, Any]:  
    completed_public_datasets = [  
        str(run["dataset_name"])  
        for run in public_runs  
    ]  
    missing_required_public_datasets = [  
        dataset_name  
        for dataset_name in _SUPPORTED_PUBLIC_DATASETS  
        if dataset_name not in set(completed_public_datasets)  
    ]  
    held_out_status = str(held_out_core_evaluation["status"])  
    held_out_reason = held_out_core_evaluation.get("reason")  
    held_out_message = held_out_core_evaluation.get("message")  
    ordered_methods = ("ekf", "robust_ekf", "fgo", "lstm_ekf", "liquid_ekf")  
    leaderboard = list(held_out_core_evaluation.get("leaderboard") or [])  
    leaderboard_methods = _normalize_leaderboard_method_names(leaderboard)  
    comparison_methods_present = [method_name for method_name in ordered_methods if method_name in set(leaderboard_methods)]  
    cross_factorial_status = str(cross_factorial_interactions.get("status") or "missing")  
    cross_factorial_reason = cross_factorial_interactions.get("reason")  
    cross_factorial_message = cross_factorial_interactions.get("message")  
    if requested_public_datasets:  
        public_conclusion_source = (  
            "public_dataset_eval"  
            if not missing_required_public_datasets  
            else "incomplete_public_surface"  
        )  
        public_surface_status = "ok" if not missing_required_public_datasets else "incomplete"  
    else:  
        public_conclusion_source = "public_benchmarks_skipped"  
        public_surface_status = "skipped"  
    cross_factorial_conclusion_source = (  
        "held_out_cross_factorial_eval"  
        if cross_factorial_status == "ok"  
        else "cross_factorial_surface_incomplete"  
    )  
    if held_out_status == "ok" and cross_factorial_status == "ok" and not missing_required_public_datasets:  
        status = "ok"  
        reason = held_out_reason  
        message = held_out_message  
    elif held_out_status == "ok" and cross_factorial_status != "ok":  
        status = "degraded"  
        reason = str(cross_factorial_reason or "cross_factorial_surface_incomplete")  
        message = (  
            str(cross_factorial_message)  
            if str(cross_factorial_message or "").strip()  
            else "held-out core ranking is available, but cross-factorial A/N/V evidence is incomplete so the claim must stay below a full cross-scene statement"  
        )  
    elif held_out_status == "ok":  
        status = "degraded"  
        if requested_public_datasets:  
            reason = "public_surface_incomplete"  
            message = (  
                "public benchmark surface incomplete: "  
                f"requested={list(requested_public_datasets)}, "  
                f"completed={completed_public_datasets}, "  
                f"missing_required={missing_required_public_datasets}"  
            )  
        else:  
            reason = "public_benchmarks_skipped"  
            message = (  
                "public benchmark surface skipped: "  
                f"required={list(_SUPPORTED_PUBLIC_DATASETS)}"  
            )  
    else:  
        status = held_out_status  
        reason = held_out_reason  
        message = held_out_message  
    claim_recommendation = "global_first" if status == "ok" else "shrink_to_truthful_scope"  

    return {  
        "status": status,  
        "reason": reason,  
        "message": message,  
        "split_manifest": split_manifest,  
        "split_provenance": held_out_core_evaluation.get("split_provenance"),  
        "held_out_test_ids": list(held_out_core_evaluation.get("held_out_test_ids") or []),  
        "core_conclusion_source": "held_out_test_ids",  
        "cross_factorial_conclusion_source": cross_factorial_conclusion_source,  
        "cross_factorial_surface_status": cross_factorial_status,  
        "public_conclusion_source": public_conclusion_source,  
        "public_surface_status": public_surface_status,  
        "required_public_datasets": list(_SUPPORTED_PUBLIC_DATASETS),  
        "requested_public_datasets": list(requested_public_datasets),  
        "completed_public_datasets": completed_public_datasets,  
        "missing_required_public_datasets": missing_required_public_datasets,  
        "cross_factorial_reason": cross_factorial_reason,  
        "cross_factorial_message": cross_factorial_message,  
        "selection_logic": {  
            "split_rule_source": str((ROOT / "scripts" / "default_split_rules.yaml").resolve()),  
            "experiment_protocol_source": str((ROOT / "configs" / "base" / "experiment_protocol.yaml").resolve()),  
            "conclusion_priority": list(held_out_core_evaluation.get("conclusion_priority") or []),  
            "held_out_ranker_source": "scripts/20_run_paper_experiments.py::_build_held_out_protocol_rank_vector",  
            "search_scorer_source": "scripts/20_run_paper_experiments.py::_score_candidate_lexicographically",  
        },  
        "leader_method": held_out_core_evaluation.get("leader_method"),  
        "liquid_rank": held_out_core_evaluation.get("liquid_rank"),  
        "claim_recommendation": claim_recommendation,  
        "leaderboard": leaderboard,  
        "comparison_surface": {
            "expected_methods": list(ordered_methods),
            "leaderboard_methods": leaderboard_methods,
            "comparison_methods_present": comparison_methods_present,
            "all_expected_methods_present": comparison_methods_present == list(ordered_methods),
            "missing_methods": [method_name for method_name in ordered_methods if method_name not in set(comparison_methods_present)],
        },
        "collection_errors": held_out_core_evaluation.get("collection_errors"),  
        "incomplete_methods": held_out_core_evaluation.get("incomplete_methods"),  
        "core_runs": list(held_out_core_evaluation.get("held_out_runs") or []),  
        "public_runs": [  
            {  
                "dataset_name": run["dataset_name"],  
                "output_root": run["output_root"],  
                "eval_stage_name": run["eval_stage_name"],  
            }  
            for run in public_runs  
        ],  
    }  


def _build_evaluation_coverage_audit(
    *,
    core_runs: list[dict[str, Any]],
    held_out_core_evaluation: dict[str, Any],
    cross_factorial_interactions: dict[str, Any],
    requested_public_datasets: list[str],
    public_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the comparison-readiness summary block."""

    core_experiment_ids = [str(run.get("experiment_id")) for run in core_runs]
    held_out_experiment_ids = [
        str(run.get("experiment_id"))
        for run in list(held_out_core_evaluation.get("held_out_runs") or [])
    ]
    ordered_methods = ("ekf", "robust_ekf", "fgo", "lstm_ekf", "liquid_ekf")
    leaderboard = list(held_out_core_evaluation.get("leaderboard") or [])
    leaderboard_methods = _normalize_leaderboard_method_names(leaderboard)
    comparison_methods_present = [method_name for method_name in ordered_methods if method_name in set(leaderboard_methods)]
    completed_public_datasets = [str(run.get("dataset_name")) for run in public_runs]
    covered_axes = {
        "async": "e2_async" in core_experiment_ids,
        "nlos": "e3_nlos" in core_experiment_ids,
        "visual": "e4_visual" in core_experiment_ids,
        "dual_degradation": "e9_dual_degradation" in core_experiment_ids,
        "cross_factorial": str(cross_factorial_interactions.get("status") or "missing") == "ok",
    }
    held_out_coverage = {
        "main_table": "e1_main_table" in held_out_experiment_ids,
        "async": "e2_async" in held_out_experiment_ids,
        "nlos": "e3_nlos" in held_out_experiment_ids,
        "dual_degradation": "e9_dual_degradation" in held_out_experiment_ids,
    }
    missing_requested_public = [
        dataset_name
        for dataset_name in requested_public_datasets
        if dataset_name not in set(completed_public_datasets)
    ]
    core_surface_complete = all(bool(covered_axes.get(axis_name)) for axis_name in covered_axes)
    held_out_surface_complete = all(bool(held_out_coverage.get(axis_name)) for axis_name in held_out_coverage)
    comparison_surface_complete = comparison_methods_present == list(ordered_methods)
    public_surface_complete = not missing_requested_public
    missing_requirements = [
        requirement
        for requirement, satisfied in (
            ("core_async_covered", bool(covered_axes.get("async"))),
            ("core_nlos_covered", bool(covered_axes.get("nlos"))),
            ("core_visual_covered", bool(covered_axes.get("visual"))),
            ("core_dual_degradation_covered", bool(covered_axes.get("dual_degradation"))),
            ("cross_factorial_covered", bool(covered_axes.get("cross_factorial"))),
            ("held_out_main_table_covered", bool(held_out_coverage.get("main_table"))),
            ("held_out_async_covered", bool(held_out_coverage.get("async"))),
            ("held_out_nlos_covered", bool(held_out_coverage.get("nlos"))),
            ("held_out_dual_degradation_covered", bool(held_out_coverage.get("dual_degradation"))),
            ("comparison_surface_complete", comparison_surface_complete),
            ("requested_public_surface_complete", public_surface_complete),
        )
        if not satisfied
    ]
    return {
        "status": (
            "ok"
            if core_surface_complete
            and held_out_surface_complete
            and comparison_surface_complete
            and public_surface_complete
            else "partial"
        ),
        "core_experiment_ids": core_experiment_ids,
        "held_out_experiment_ids": held_out_experiment_ids,
        "core_axis_coverage": covered_axes,
        "held_out_axis_coverage": held_out_coverage,
        "coverage_summary": {
            "core_surface_complete": core_surface_complete,
            "held_out_surface_complete": held_out_surface_complete,
            "comparison_surface_complete": comparison_surface_complete,
            "requested_public_surface_complete": public_surface_complete,
        },
        "comparison_surface": {
            "expected_methods": list(ordered_methods),
            "leaderboard_methods": leaderboard_methods,
            "comparison_methods_present": comparison_methods_present,
            "all_expected_methods_present": comparison_surface_complete,
            "missing_methods": [method_name for method_name in ordered_methods if method_name not in set(comparison_methods_present)],
        },
        "held_out_test_ids": list(held_out_core_evaluation.get("held_out_test_ids") or []),
        "cross_factorial_status": str(cross_factorial_interactions.get("status") or "missing"),
        "cross_factorial_reason": cross_factorial_interactions.get("reason"),
        "requested_public_datasets": list(requested_public_datasets),
        "completed_public_datasets": completed_public_datasets,
        "missing_requested_public_datasets": missing_requested_public,
        "missing_requirements": missing_requirements,
        "coverage_notes": [
            "core axis coverage summarizes whether async, NLOS, visual degradation, and dual degradation were executed in the main paper matrix",
            "held-out axis coverage summarizes whether the held-out conclusion suite includes async, NLOS, and dual-degradation checkpoints in addition to e1_main_table",
            "cross-factorial coverage is sourced from the dedicated held-out A/N/V interaction audit rather than inferred from the core matrix",
            "public coverage tracks only datasets explicitly requested for the current run",
        ],
    }


def _resolve_primary_paper_artifact_roots(  
    *,  
    core_runs: list[dict[str, Any]],  
    held_out_core_evaluation: dict[str, Any],  
) -> dict[str, Any]:  
    held_out_runs = list(held_out_core_evaluation.get("held_out_runs") or [])  
    held_out_main_run = next(  
        (run for run in held_out_runs if str(run.get("experiment_id")) == "e1_main_table"),  
        None,  
    )  
    if held_out_main_run is not None:  
        experiment_output_root = Path(held_out_main_run["output_root"]).resolve()  
        return {  
            "source": "held_out_e1_main_table",  
            "experiment_output_root": str(experiment_output_root),  
            "core_root": str((experiment_output_root / "core").resolve()),  
            "eval_root": str((experiment_output_root / "eval").resolve()),  
            "scene_sampling_seq_ids": list(held_out_main_run.get("scene_sampling_seq_ids") or []),  
        }  
    raise RuntimeError("paper run could not locate held-out e1_main_table outputs for artifact consumers")  


def _summarize_seq_generalization_metric_table(  
    metric_table_path: Path,  
    *,  
    methods: list[str],  
    allowed_seq_ids: list[str] | None = None,  
) -> dict[str, Any]:  
    if not metric_table_path.is_file():  
        return {  
            "status": "blocked",  
            "reason": "missing_metric_table",  
            "message": "seq generalization audit requires eval/metrics/metric_table.csv",  
            "metric_table_path": str(metric_table_path),  
            "allowed_seq_ids": list(allowed_seq_ids or []),  
        }  

    metric_views = _read_metric_table_views(metric_table_path)  
    metric_rows = metric_views["repeat_summary_rows"]  
    required_metrics = ("rmse", "p95", "p99", "failure_rate", "mae")  
    metric_values_by_variant: dict[str, dict[str, dict[str, list[float]]]] = {}  
    support_by_variant: dict[str, dict[str, dict[str, Any]]] = {}  
    collection_errors: list[dict[str, Any]] = []  
    allowed_seq_id_set = {str(seq_id) for seq_id in list(allowed_seq_ids or [])} if allowed_seq_ids is not None else None  

    for row in metric_rows:  
        seq_id = str(row.get("seq_id") or "").strip()  
        method_name = str(row.get("method_name") or "").strip()  
        metric_name = str(row.get("metric") or "").strip()  
        if not seq_id or not method_name:  
            continue  
        if allowed_seq_id_set is not None and seq_id not in allowed_seq_id_set:  
            continue  
        
        raw_scene_id = str(row.get("scene_id") or "").strip()  
        variant_key = f"{raw_scene_id}::{seq_id}" if raw_scene_id else seq_id  
        support_row = support_by_variant.setdefault(variant_key, {}).setdefault(  
            method_name,  
            {  
                "case_count": 0,  
                "overlap_ratio_sum": 0.0,  
                "reliability_statuses": set(),  
                "seen_case_refs": set(),  
            },  
        )  
        case_ref = str(row.get("case_ref") or "").strip()  
        if case_ref and case_ref not in support_row["seen_case_refs"]:  
            support_row["seen_case_refs"].add(case_ref)  
            support_row["case_count"] += 1  
            try:  
                overlap_ratio = float(row.get("overlap_ratio") or 0.0)  
                if not math.isfinite(overlap_ratio):  
                    raise ValueError("overlap_ratio must be finite")
                support_row["overlap_ratio_sum"] += overlap_ratio  
            except (TypeError, ValueError):  
                collection_errors.append(  
                    {  
                        "seq_id": seq_id,  
                        "method_name": method_name,  
                        "case_ref": case_ref,  
                        "field": "overlap_ratio",  
                    }  
                )  
            reliability_status = str(row.get("reliability_status") or "").strip()  
            if reliability_status:  
                support_row["reliability_statuses"].add(reliability_status)  

        if metric_name not in required_metrics:  
            continue  
        metric_values_by_variant.setdefault(variant_key, {}).setdefault(method_name, {}).setdefault(metric_name, []).append(  
            float(row["value"])  
        )  

    variant_rows: list[dict[str, Any]] = []  
    leader_counts = {  
        metric_name: {method_name: 0 for method_name in methods}  
        for metric_name in ("p95", "failure_rate")  
    }  
    for variant_key in sorted(metric_values_by_variant):  
        method_rows: dict[str, dict[str, Any]] = {}  
        leader_by_metric: dict[str, str] = {}  
        for method_name in methods:  
            metric_lists = metric_values_by_variant[variant_key].get(method_name)  
            if metric_lists is None:  
                collection_errors.append(  
                    {  
                        "variant_key": variant_key,  
                        "method_name": method_name,  
                        "field": "missing_method_metrics",  
                    }  
                )  
                continue  
            missing_metrics = [metric_name for metric_name in required_metrics if metric_name not in metric_lists]  
            if missing_metrics:  
                collection_errors.append(  
                    {  
                        "variant_key": variant_key,  
                        "method_name": method_name,  
                        "field": "missing_metrics",  
                        "metrics": missing_metrics,  
                    }  
                )  
                continue  
            support_row = support_by_variant.get(variant_key, {}).get(method_name, {})  
            case_count = int(support_row.get("case_count") or 0)  
            overlap_ratio_sum = float(support_row.get("overlap_ratio_sum") or 0.0)  
            method_rows[method_name] = {  
                metric_name: float(sum(metric_lists[metric_name]) / len(metric_lists[metric_name]))  
                for metric_name in required_metrics  
            }  
            method_rows[method_name]["case_count"] = case_count  
            method_rows[method_name]["mean_overlap_ratio"] = (  
                overlap_ratio_sum / case_count if case_count > 0 else None  
            )  
            method_rows[method_name]["reliability_statuses"] = sorted(  
                support_row.get("reliability_statuses") or []  
            )  

        for metric_name in ("p95", "failure_rate"):  
            if all(method_name in method_rows for method_name in methods):  
                leader_method = min(  
                    ((method_name, float(method_rows[method_name][metric_name])) for method_name in methods),  
                    key=lambda item: item[1],  
                )[0]  
                leader_by_metric[metric_name] = leader_method  
                leader_counts[metric_name][leader_method] += 1  

        variant_rows.append(  
            {  
                "variant_key": variant_key,  
                "leader_by_metric": leader_by_metric,  
                "method_metrics": method_rows,  
            }  
        )  

    variant_count = len(variant_rows)  
    if variant_count == 0:  
        return {  
            "status": "blocked",  
            "reason": "empty_metric_table",  
            "message": "seq generalization audit found no per-seq metrics",  
            "metric_table_path": str(metric_table_path),  
            "allowed_seq_ids": list(allowed_seq_ids or []),  
            "collection_errors": collection_errors,  
        }  
    return {  
        "status": "ok",  
        "reason": None,  
        "message": "scene variant generalization summary collected per-variant metrics",  
        "metric_table_path": str(metric_table_path),  
        "allowed_seq_ids": list(allowed_seq_ids or []),  
        "variant_count": variant_count,  
        "leader_counts": leader_counts,  
        "variant_rows": variant_rows,  
        "collection_errors": collection_errors,  
    }  


def _build_held_out_seq_generalization_audit(  
    *,  
    held_out_core_evaluation: dict[str, Any],  
    methods: list[str],  
) -> dict[str, Any]:  
    held_out_runs = list(held_out_core_evaluation.get("held_out_runs") or [])  
    held_out_main_run = next(  
        (run for run in held_out_runs if str(run.get("experiment_id")) == "e1_main_table"),  
        None,  
    )  
    if held_out_main_run is None:  
        return {  
            "status": "blocked",  
            "reason": "missing_held_out_main_table",  
            "message": "held-out seq generalization audit requires a held-out e1_main_table run",  
            "held_out_test_ids": list(held_out_core_evaluation.get("held_out_test_ids") or []),  
        }  

    held_out_summary = _summarize_seq_generalization_metric_table(  
        Path(held_out_main_run["output_root"]) / "eval" / "metrics" / "metric_table.csv",  
        methods=methods,  
        allowed_seq_ids=list(held_out_core_evaluation.get("held_out_test_ids") or []),  
    )  
    if held_out_summary["status"] != "ok":  
        held_out_summary["held_out_test_ids"] = list(held_out_core_evaluation.get("held_out_test_ids") or [])  
        return held_out_summary  

    variant_count = int(held_out_summary["variant_count"])  
    status = "ok" if variant_count >= 2 else "limited"  
    message = (  
        "held-out scene variant generalization audit covers at least two test variants"  
        if status == "ok"  
        else "held-out scene variant generalization audit is limited because only one test variant is available"  
    )  
    return {  
        "status": status,  
        "reason": None if status == "ok" else "single_test_variant",  
        "message": message,  
        "metric_table_path": held_out_summary["metric_table_path"],  
        "held_out_test_ids": list(held_out_core_evaluation.get("held_out_test_ids") or []),  
        "variant_count": variant_count,  
        "leader_counts": held_out_summary["leader_counts"],  
        "variant_rows": held_out_summary["variant_rows"],  
        "collection_errors": held_out_summary["collection_errors"],  
    }  


def _build_unseen_seq_generalization_audit(  
    *,  
    core_runs: list[dict[str, Any]],  
    held_out_core_evaluation: dict[str, Any],  
    split_manifest: dict[str, Any],  
    methods: list[str],  
) -> dict[str, Any]:  
    validation_seq_ids = list(split_manifest.get("val_ids") or [])  
    held_out_test_ids = list(held_out_core_evaluation.get("held_out_test_ids") or split_manifest.get("test_ids") or [])  
    full_main_run = next((run for run in core_runs if str(run.get("experiment_id")) == "e1_main_table"), None)  
    if full_main_run is None:  
        return {  
            "status": "blocked",  
            "reason": "missing_full_main_table",  
            "message": "unseen seq generalization audit requires the full-matrix e1_main_table run",  
            "validation_unseen_seq_ids": validation_seq_ids,  
            "held_out_test_ids": held_out_test_ids,  
        }  

    held_out_runs = list(held_out_core_evaluation.get("held_out_runs") or [])  
    held_out_main_run = next(  
        (run for run in held_out_runs if str(run.get("experiment_id")) == "e1_main_table"),  
        None,  
    )  
    if held_out_main_run is None:  
        return {  
            "status": "blocked",  
            "reason": "missing_held_out_main_table",  
            "message": "unseen seq generalization audit requires the held-out e1_main_table run",  
            "validation_unseen_seq_ids": validation_seq_ids,  
            "held_out_test_ids": held_out_test_ids,  
        }  

    validation_summary = _summarize_seq_generalization_metric_table(  
        Path(full_main_run["output_root"]) / "eval" / "metrics" / "metric_table.csv",  
        methods=methods,  
        allowed_seq_ids=validation_seq_ids,  
    )  
    held_out_summary = _summarize_seq_generalization_metric_table(  
        Path(held_out_main_run["output_root"]) / "eval" / "metrics" / "metric_table.csv",  
        methods=methods,  
        allowed_seq_ids=held_out_test_ids,  
    )  
    if validation_summary["status"] != "ok":  
        return {  
            **validation_summary,  
            "validation_unseen_seq_ids": validation_seq_ids,  
            "held_out_test_ids": held_out_test_ids,  
        }  
    if held_out_summary["status"] != "ok":  
        return {  
            **held_out_summary,  
            "validation_unseen_seq_ids": validation_seq_ids,  
            "held_out_test_ids": held_out_test_ids,  
        }  

    combined_variant_rows = [  
        {**row, "seq_bucket": "validation_unseen"} for row in validation_summary["variant_rows"]  
    ] + [{**row, "seq_bucket": "held_out_unseen"} for row in held_out_summary["variant_rows"]]  
    combined_leader_counts = {  
        metric_name: {method_name: 0 for method_name in methods}  
        for metric_name in ("p95", "failure_rate")  
    }  
    for row in combined_variant_rows:  
        for metric_name, leader_method in dict(row.get("leader_by_metric") or {}).items():  
            if metric_name in combined_leader_counts and leader_method in combined_leader_counts[metric_name]:  
                combined_leader_counts[metric_name][leader_method] += 1  

    combined_variant_count = len(combined_variant_rows)  
    status = "ok" if combined_variant_count >= 2 else "limited"  
    return {  
        "status": status,  
        "reason": None if status == "ok" else "insufficient_unseen_variant_count",  
        "message": (  
            "unseen variant audit combines validation-unseen and held-out-unseen variants for broader visibility"  
            if status == "ok"  
            else "unseen variant audit found fewer than two unseen variants"  
        ),  
        "validation_unseen_seq_ids": validation_seq_ids,  
        "held_out_test_ids": held_out_test_ids,  
        "combined_unseen_seq_ids": validation_seq_ids + held_out_test_ids,  
        "combined_unseen_variant_count": combined_variant_count,  
        "leader_counts": combined_leader_counts,  
        "variant_rows": combined_variant_rows,  
        "bucket_summaries": {  
            "validation_unseen": validation_summary,  
            "held_out_unseen": held_out_summary,  
        },  
        "honesty_notes": [  
            "validation-unseen trajectories were used during model selection and are reported for cross-trajectory visibility only",  
            "held-out-unseen trajectories remain the only unseen bucket that supports the final main conclusion",  
        ],  
        "collection_errors": validation_summary["collection_errors"] + held_out_summary["collection_errors"],  
    }  


def _run_core_experiment(  
    *,  
    config_name: str,  
    prepare_root: Path,  
    raw_root: Path,  
    estimator_cfgs: dict[str, dict[str, Any]],  
    model_cfgs: dict[str, dict[str, Any]],  
    output_root: Path,  
    seq_ids: list[str] | None = None,  
) -> dict[str, Any]:  
    experiment_cfg = copy.deepcopy(_load_experiment_cfg_cached(config_name))  
    scene_tasks = [copy.deepcopy(task) for task in _sample_scenes_cached(config_name)]  
    prepare_manifest = _load_prepare_manifest_cached(str(prepare_root.resolve()))  
    resolved_seq_ids = resolve_experiment_seq_ids_from_prepare_manifest(  
        prepare_manifest,
        experiment_cfg=experiment_cfg,
        seq_ids=seq_ids,
    )
    if not resolved_seq_ids:  
        raise RuntimeError(f"{Path(config_name).stem} requires non-empty seq_ids")  
    _, events_by_seq_id, ground_truth_by_seq_id, source_report_by_seq_id = _load_scoring_inputs_cached(  
        str(prepare_root.resolve()),  
        str(raw_root.resolve()),  
        tuple(resolved_seq_ids),  
    )  
    expanded_scene_tasks = expand_scene_tasks_across_seq_ids(scene_tasks, resolved_seq_ids)  
    events_by_seq_id = copy.deepcopy(events_by_seq_id)  
    ground_truth_by_seq_id = copy.deepcopy(ground_truth_by_seq_id)  
    source_report_by_seq_id = copy.deepcopy(source_report_by_seq_id)  

    experiment_output_root = output_root / Path(config_name).stem  
    core_result = CorePipeline().run(  
        {  
            "project_root": ROOT,  
            "experiment_cfg": experiment_cfg,  
            "scene_tasks": expanded_scene_tasks,  
            "events_by_seq_id": events_by_seq_id,  
            "ground_truth_by_seq_id": ground_truth_by_seq_id,  
            "source_report_by_seq_id": source_report_by_seq_id,  
            "methods": list(experiment_cfg["methods"]),  
            "estimator_cfgs": estimator_cfgs,  
            "model_cfgs": model_cfgs,  
            "output_root": str(experiment_output_root / "core"),
            "retain_prediction_bundles": False,  
        }  
    )  
    eval_result = EvalPipeline().run(  
        {  
            "project_root": ROOT,  
            "prediction_bundles": _resolve_prediction_bundles(core_result.metadata),  
            "ground_truth_root": str(raw_root),  
            "ground_truth_by_seq_id": ground_truth_by_seq_id,  
            "output_root": str(experiment_output_root / "eval"),  
        }  
    )  
    return {  
        "experiment_id": Path(config_name).stem,  
        "core_result": core_result,  
        "eval_result": eval_result,  
        "output_root": str(experiment_output_root),  
        "scene_sampling_seq_ids": resolved_seq_ids,  
        "eval_stage_name": eval_result.stage_name,  
    }  


def _run_scoring_experiment(  
    *,  
    config_name: str,  
    prepare_root: Path,  
    raw_root: Path,  
    method_name: str,  
    estimator_cfgs: dict[str, dict[str, Any]],  
    model_cfgs: dict[str, dict[str, Any]],  
    output_root: Path,  
    baseline_method_name: str = "ekf",  
    seq_ids: list[str] | None = None,  
    baseline_metrics: dict[str, float] | None = None,  
) -> dict[str, Any]:  
    experiment_cfg = copy.deepcopy(_load_experiment_cfg_cached(config_name))  
    scene_tasks = [copy.deepcopy(task) for task in _sample_scenes_cached(config_name)]  
    prepare_manifest = _load_prepare_manifest_cached(str(prepare_root.resolve()))  
    resolved_seq_ids = resolve_experiment_seq_ids_from_prepare_manifest(  
        prepare_manifest,
        experiment_cfg=experiment_cfg,
        seq_ids=seq_ids,
    )
    if not resolved_seq_ids:  
        raise RuntimeError(f"{Path(config_name).stem} scoring requires non-empty seq_ids")  
    _, events_by_seq_id, ground_truth_by_seq_id, source_report_by_seq_id = _load_scoring_inputs_cached(  
        str(prepare_root.resolve()),  
        str(raw_root.resolve()),  
        tuple(resolved_seq_ids),  
    )  
    expanded_scene_tasks = expand_scene_tasks_across_seq_ids(scene_tasks, resolved_seq_ids)  
    events_by_seq_id = copy.deepcopy(events_by_seq_id)  
    ground_truth_by_seq_id = copy.deepcopy(ground_truth_by_seq_id)  
    source_report_by_seq_id = copy.deepcopy(source_report_by_seq_id)  
    experiment_output_root = output_root / Path(config_name).stem  
    methods = (  
        [method_name]  
        if method_name == baseline_method_name or baseline_metrics is not None  
        else [baseline_method_name, method_name]  
    )  
    core_result = CorePipeline().run(  
        {  
            "project_root": ROOT,  
            "experiment_cfg": experiment_cfg,  
            "scene_tasks": expanded_scene_tasks,  
            "events_by_seq_id": events_by_seq_id,  
            "ground_truth_by_seq_id": ground_truth_by_seq_id,  
            "source_report_by_seq_id": source_report_by_seq_id,  
            "methods": methods,  
            "estimator_cfgs": estimator_cfgs,  
            "model_cfgs": model_cfgs,  
            "output_root": str(experiment_output_root / "core"),
            "retain_prediction_bundles": False,  
        }  
    )  
    eval_result = EvalPipeline().run(  
        {  
            "project_root": ROOT,  
            "prediction_bundles": _resolve_prediction_bundles(core_result.metadata),  
            "ground_truth_root": str(raw_root),  
            "ground_truth_by_seq_id": ground_truth_by_seq_id,  
            "output_root": str(experiment_output_root / "eval"),  
        }  
    )  
    metrics = _collect_search_metrics(  
        experiment_output_root=experiment_output_root,  
        method_name=method_name,  
        baseline_method_name=baseline_method_name,  
        baseline_metrics=baseline_metrics,  
    )  
    _trim_search_scoring_artifacts(experiment_output_root)  
    return {  
        "experiment_id": Path(config_name).stem,  
        "metrics": metrics,  
        "output_root": str(experiment_output_root),  
        "seq_ids": resolved_seq_ids,  
        "baseline_reused": bool(baseline_metrics is not None and method_name != baseline_method_name),  
        "core_stage_name": core_result.stage_name,  
        "eval_stage_name": eval_result.stage_name,  
        "scene_sampling_seq_ids": resolved_seq_ids,  
    }  


def _trim_search_scoring_artifacts(experiment_output_root: Path) -> None:
    """茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聟忙職娄茅聳禄茫聢垄莽禄聬忙楼聝芒聜卢氓鹿驴氓聞卤茅聨路氓茫聙聮忙驴聽忙掳聟聶茅聳禄忙聢聻茂鹿漏氓篓聢忙聸垄忙聼聸忙驴聽氓聜聸氓聛楼茅聳潞氓卤禄氓聤聭氓炉麓聮氓搂聵茅聴聜盲陆赂忙聬芦氓娄卢聭氓聢聮茅聨戮猫炉虏猫聯聼莽聙鹿茫聝楼莽聣聹氓娄虏氓楼赂忙垄潞莽录聛忙篓禄氓掳掳茅聳禄莽聠虏忙聶聽氓炉芒聵聣氓氓聭聺赂氓漏碌茂鹿漏茅聧聫忙露聵脨搂氓娄陇氓聭庐芒聮聭忙驴聻猫聦露茅陋聻忙聽聼拢茅隆聯莽聜虏芒聜聻庐莽聮聡忙碌聥忙搂聢忙驴庐忙漏聠氓聛聟茅聧聲氓聠庐忙聼拢茅聬聵氓聫聣氓莽聭聹忙聺驴氓聲麓茅聰聺莽聜潞氓麓露猫陇聣茅聫聙猫聤楼茫聛聞茅聫聝氓聜聵芒聜茅聤聣芒聲聜莽聟聫茅聳赂氓聳聬茅聧聤忙聞?茅聴聜氓聜職氓聙路氓篓麓氓聸茅隆驴聥忙聲聝茅聧聦茂赂戮莽篓聫忙驴聽茫聢拢氓聼聳茅聧聰忙聽麓聭茅聤聢氓莽聟聸茅聳赂茫聝垄忙聶聸氓掳聫茅聴聛莽禄聵氓赂聮茅聰聲茫聝搂忙聧垄氓篓聭茫聢聽氓鹿垄忙驴隆忙忙碌聽忙聮庐忙垄潞茅聨录莽聜虏氓莽录聛猫莽聣聯氓猫炉虏忙戮露氓漏聞氓聨赂忙驴聻猫戮戮莽碌陆茅聨虏芒聙鲁芒聳聯茅聧聸聼芒聮聭莽聙鹿忙聸聻忙麓篓莽聰炉忙露聶忙聟篓忙驴聽氓聜陇莽聟录茅聳赂芒聰戮氓聛聯氓娄聻茫聜聠氓路禄茅聧聥忙聮露莽麓聮茅聬聵猫聦卢脨搂莽聙鹿忙聸聼忙麓聼茅聫聦氓聴聴茅聧聸忙聻聝茅聴聛莽聭掳氓聬聥茅聬拢氓麓聺氓庐聙氓聥聛陋茅聳禄忙聞碌氓聣聸莽禄聽茅鹿聝忙聼聸茅聢漏氓聟漏忙聢聽盲潞赂忙碌录氓颅聵莽聟聝茅聴聛氓聻篓麓忙聼隆莽聛聻氓聣搂忙麓麓氓漏碌茂录聞氓聟聵茅隆垄忙露聵茅聧聬茫聞搂氓聨聮忙驴聽莽聰碌氓聛聸茅聬隆茫聞搂氓聢聮茅聨戮氓聣聛盲赂聧茅聳潞氓露聨氓聨录茅聮聝莽聠录忙聼篓氓漏碌氓聴漏忙搂鹿茅聳赂氓氓芦掳茅聫聦忙露聵芒聵聣氓搂聴氓聽聺氓搂聸茅聴聛忙聦聨莽篓聮茅聬漏氓篓虏氓聛聜茅聨录猫戮戮莽碌戮茅聬聨芒聰聭茅聬聬氓路庐忙搂鹿茅聳禄芒聲聤氓聻露氓庐聲忙麓陋氓搂聙茅聤聢氓聭炉茅聳禄莽聤虏忙麓漏莽聛聫忙卢聯氓篓虏芒聮聭莽录聜盲陆鹿脙驴庐氓虏聞忙聜聵猫鹿聡忙聴聜忙聡聟莽录聛莽露聛茅聨庐忙卢聯忙聜聹氓娄陆掳莽聰禄忙驴聽莽聰碌氓搂麓茅聰聲茫聜聣氓麓碌录氓颅聵氓聙露茅聫聣氓聽職忙聶聺莽潞戮氓楼赂氓录露茅聧芦忙掳芦聟氓篓麓莽聤聶氓聸篓莽聟聝莽聭聹忙禄聢氓麓聹茅聤聤脩聝莽路陇茅聳赂脩聞茅聧?茅聴聜氓聜職氓聙赂茅聧聤莽聝陆忙聡聴茅聧聸聻氓聜篓莽聭聹忙聴聢氓鹿聝氓娄戮炉忙陆漏茅聬聽氓聯聞忙聬芦茅聬聨忙垄潞茅聧聯猫聦卢氓聨聸茅聳赂茂陆聨氓聳聳茅隆聸卢氓录陆茅莽聠潞氓聨戮忙驴聽忙氓聳聴茅聧聰忙聸垄氓聼聙茅隆聮盲戮聡忙庐聵莽录聛莽聣聯氓庐聲莽聠录氓篓聭忙掳鹿聴忙驴聽莽忙搂聟茅聧篓盲录麓忙聜聵氓漏碌氓猫搂娄茅聳赂茂赂禄氓聨赂茅聧聥忙聮露忙聠麓茅聧聲莽录聜盲戮聡氓聳聴茅聨赂忙聬聛茂陆聻猫鹿聡忙露聶忙聝聻茅聳赂茂赂禄氓聨戮茅聰聸忙禄聝芒聰聭茅聢陆氓聴聴莽聛娄茅聳潞氓聭掳氓掳聶茅聧聮忙聨聴氓麓录茅聫聡莽聠虏忙聥潞茅聴聛氓聭聤莽鹿聜莽聙職氓聼聙茅隆聮盲陆鹿氓聙聬莽聙鹿忙聸聼氓聤聶氓庐聲莽篓驴脦聰茅聢搂氓录掳茅聤聣芒聲聜莽聟聝?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬茅聬聭猫聤楼氓職聞茅聳赂忙麓聳莽禄聽莽聤禄忙聼聼茅聨鹿忙聞碌氓職聶茅聬聼忙卢聶莽庐聯茅聨忙聲聬忙聬麓芒聙鲁茅聴聛忙聬聻氓聤聧莽禄禄氓聽楼莽聧聫聧氓潞篓茅聢搂盲驴聤茅聭虏芒聲聝忙拢炉茅聳鹿盲陆赂盲禄聴聛氓聯聞忙聹卢氓篓虏忙忙聧垄氓篓聭茫聢隆氓聙路忙陇陇忙聨聧猫聝露猫陇掳氓聭炉莽麓聯茅聧聦忙卢聮氓聟聦氓漏碌莽聛職莽禄禄忙露聶莽鹿聺茅聧楼脩聠忙聭聲茅聴聛莽禄聵忙聼篓茅聧職氓氓麓聬莽录聛忙篓鹿莽鹿聠忙陇陇忙聽篓莽鹿聝茅莽聤虏脙鹿茅聬聵氓聯聞盲潞鹿莽录聛猫戮篓氓赂隆茅聨庐忙卢聯忙碌拢氓聯聞氓職聮莽录聜氓聜聶莽陇聛茅隆娄莽麓聻忙驴聽氓聜陇氓職聲茅聬聽氓聸篓莽楼麓氓娄虏氓聽聼忙聟篓氓娄陇芒聜卢氓娄卢聯氓麓聯茅聬垄氓聭聤芒聮聭茅聳赂茂赂禄氓聨陆茅莽聠潞莽麓聯氓庐楼氓聥颅莽陋聰莽聙碌茅聫聛忙聞戮聞忙驴聽氓潞垄莽露聜茅聴聜盲陆潞莽虏炉茅隆聽垄盲潞赂氓篓聠忙聮鲁忙卤虏??茅聴?``eval/metrics/metric_table.csv`` 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛氓露聣氓麓聼茅茫聢潞掳茅聳禄氓潞隆氓聨聹茅聧聥忙聮禄忙聼聸茅聫聡茫聞楼氓垄驴聻氓聽露聛氓搂聤猫聶鹿氓聦聳茅聫聡莽聠赂忙戮聮茅聴聛莽茅聨赂忙聬聛芒聙聰茅聧聬脦搂茅聳赂芒聞聝茅聬聼忙聞掳忙垄潞茅聬聯氓潞隆忙職卤茅聳禄忙聽搂茫聜赂茅聬聸莽鹿聺茅聧楼脩聟氓聙聻氓茫聜聠氓赂聤茅聭鲁氓聽聲氓麓垄茅聨戮莽禄聵氓聛隆忙驴聽氓漏聜氓芦聬颅忙聨聰芒聜卢猫搂聞忙麓聵莽禄虏聞氓聼垄忙聬麓茫聞楼莽莽聙鹿茫聝楼忙聡聫茅聧聽忙漏聙忙聼聼茅茫聞楼忙麓陇茅聳卤氓聸卢忙聼拢茅聫聜氓聻搂禄氓篓麓莽聟聨茅聟拢氓莽禄聵莽卢聼茅聢搂猫茅聨聡氓麓卤氓篓麓忙聺驴忙麓隆垄忙露聵茅聬聯氓潞垄忙芦聤忙驴聽莽聰碌氓篓聙氓篓录脩聝脦聸氓庐聙氓聥禄漏氓篓麓莽聤虏茅聬聯莽聠赂盲驴聤茅忙掳聛楼茅聳赂忙驴聤忙芦拢莽禄卤忙聨聯忙聥聟氓庐聞忙聺驴莽楼聴聛氓聯聞忙聹卢莽禄聥忙聢拢莽聛聮茅聴聛莽聟聨茅聧聤忙聸垄莽聧庐氓娄陇氓聭庐芒聮聮氓篓聯忙掳拢氓聛聯忙驴聻盲陆鲁氓聸拢芒聜卢茂鹿聜氓麓录茅聤聣茫聝搂茂录聼茅聳潞氓陇聤莽聣聞茅聧聰氓潞拢莽虏禄忙楼拢氓聠漏氓聙碌忙驴聻忙聢聻茅氓聥篓忙聻聰茅聳禄忙莽虏聬聯忙聸聻芒聰聭茅聬聵忙聫聮忙職聼莽录聛莽聤碌氓聛聼茅聢搂茅聧聽忙露垄猫茅聨聝搂芒聰聭莽聙拢氓茫聜聟莽鹿聬氓娄卢聫茅聨麓茂赂陆芒聮聮氓篓麓茂陆聡莽聯聰氓篓录忙聞掳忙聼聸忙聬麓茫聢聽芒聳聲莽聙鹿忙聸篓氓聻鲁茅聨鹿氓聥娄赂猫陇聫茅聬拢猫路潞芒聰聭茅隆聰莽颅聥莽聞戮茅聳赂芒聲聜莽聣聽茅聧漏忙露垄氓鹿聥茅聰聲芒聜禄戮脩聡忙聼拢茅聨掳莽露聭忙陇陇氓潞垄莽虏聯茅聫聦忙露聹莽庐聸茅聨戮莽聭掳氓茫聜聨氓聫聙茅聧聨氓脣聡茅聯聫盲潞聹茅聳碌氓氓聞虏茅聧陇氓聸聼聼茅隆聲芒聜卢莽禄聽忙聽搂莽聲路猫陇掳忙聨聺忙聲聝茅聳碌氓聽聺忙麓娄茅聧陇氓虏聞忙垄禄忙赂職芒聜卢茅聢搂氓聛聸茅聭禄忙聶露莽聙碌莽鲁聲氓聛?
    氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄茅聳禄忙聞聬芦茅聧篓氓聜聻忙聼聸茅茂陆聠莽莽录聛莽聤卤氓聧鲁茅聫聦忙露聵芒聰聭茅聧聲氓搂垄茅聴聛忙麓禄氓聨陆茅聨鹿氓聸录禄茅聬聰猫聶鹿莽拢录茅聳碌氓驴聲氓搂碌茅聬聫氓聽楼忙垄潞氓搂鹿氓聸搂芒聜卢茫聜聣氓麓鲁茅聰聺氓猫聯聼莽聙鹿茫聝楼莽聣聹氓娄虏氓聣搂莽麓聯忙碌拢氓聯聞芒聳聢莽录聛莽聤鲁莽聣聽氓炉忙聜垄莽聬聸芒聙鲁盲潞戮忙驴聻忙聢聻莽聭聬聮莽聲聦莽麓聮茅聬聵猫露聛芒聙娄茅聭鲁忙陋录聜氓聜職莽搂鲁莽禄露忙掳卢?``core/predictions`` 茅聴?``eval/plotting_inputs`` 茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛莽莽禄卤忙聸垄氓鹿聣氓氓虏聙莽聳聙忙驴聻忙聢聻茅聬拢氓聯聞茅聤聢氓聴聵莽拢碌茅聳赂氓卢聦禄茅聤聡氓聥炉氓搂聙茅聢漏氓聠陋茅聴聛猫陆掳莽聞娄茅聧聰忙卢聫莽聲路茅聤聤芒聲聤忙聲聧忙驴聽氓漏聜忙聡聨莽禄聛茂鹿聜忙垄禄忙碌拢猫聴聣氓聬鹿茅聳赂莽聤鲁氓聤聴茅聧聮忙聨聭莽聲聺茅聢搂氓鹿聭茅聤聫茫聜聟莽庐聸忙陇陇忙聴聜氓聨搂茅聬陇茅鹿聝忙垄禄茅聧聦忙卢聮氓聟聦茅聳鹿猫聶芦忙聧聡氓庐聲忙聼楼忙聡聽茅隆聮盲陆赂盲禄聫氓茫聜聠氓聤搂莽禄虏氓聺聴茫聙聟茅聭录氓聳聬茅聳禄忙楼聙莽聣聠莽禄聴忙掳露忙聼拢茅聨戮氓忙麓麓茅聳潞氓卤戮莽聸炉忙驴隆莽聝陆忙聲聬茅聧聸忙聥赂忙驴聽忙庐驴氓聳聴茅隆聽聺芒聳聰氓篓聭忙掳?
    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛莽莽禄禄忙驴聥氓录禄氓氓陇聤盲录聽茅聳赂忙聸篓氓聣聶氓篓聟茂陆聣忙垄禄茅聧聦忙掳卢盲潞聻茅聳赂茫聞楼莽聢录氓炉忙聲聬忙戮露氓漏聞茅聴聛忙聦聨莽鹿聜氓娄聸茅聧聲芒聮聭茅聳赂忙驴聠氓茅聧聨氓潞拢莽麓聮茅聭赂茅聺聸氓聯潞莽聙碌茅聢陆氓陇聤脦聼茅聳驴氓卢碌忙陆聰茅聴聜盲陆赂忙聠隆茅隆篓氓聽楼氓麓聭茅聬聬莽聝聵氓聙露茅聳赂氓聳聯莽禄隆芒聜聴聛茅聺聸莽鹿聢茅聧篓忙麓陋氓麓碌茅聢搂忙垄潞莽录聛忙漏聠忙庐聲茅聳鹿莽聭掳忙麓陇茅?茅聴聜氓聜職氓聙赂茅聧聤莽聝陆忙聜聻茅聰聲氓聜職莽庐聳茅聳赂忙麓聳莽禄聙氓陇聦芒聜卢氓聟聣氓聛聯茅聳鲁脩聦氓聣聸氓娄虏忙聞掳忙聜聶莽聙碌氓聯楼莽聵聢茅聴聛莽篓驴忙聹卢莽禄露聟芒聳聯茅聨炉脩聙忙聰陆茅聳鲁猫聴聣忙茅聬聫芒聞聝盲驴聤茅聬聬猫聢碌茅聳禄茂陆聟氓聵虏茅隆聞漏氓麓卤茅聢潞氓聜聜忙垄禄忙碌拢氓聯楼忙聻聝忙陇陇忙聸聠氓聭庐氓麓虏茅聳鹿氓炉赂氓搂碌氓庐聲氓聫聣莽鹿聺茅聴聢忙露卤氓聞聫忙陇陇忙聡聵茅聫聦茂陆聟茅聢搂猫茅聨漏聙茅聢搂氓篓虏氓路聺氓漏碌莽聤虏氓聙赂茅隆芦氓聭麓忙垄潞茅聧聺氓聥卢氓聶潞莽录聛氓卢赂聻莽禄氓鹿聮氓娄陇芒聜禄聽莽禄聵茫聛聞茅聫聝氓聸聠忙聨聴氓鹿聫氓篓虏芒聮聭茅聳赂茂赂禄氓聫聨茅聬聮莽聜鹿聸莽聙碌莽氓聨麓茅聳赂芒聰驴氓聞虏氓炉掳氓聥炉氓鹿聡茅莽聤虏氓鹿聴忙驴隆莽聜陋氓聙聳茅聨赂茅鹿聝氓麓聭氓篓聭茫聢聽莽聞碌忙陇陇忙聨聠氓聛聯忙驴聽芒聜卢茅聳露聟莽聞碌忙陇陇忙聨聭氓氓篓聭脩聡忙聼聲茅聧聠赂氓聙聬莽聙鹿忙聸聼忙陇陆茅聫聛忙聮禄忙聜漏茅聨禄忙聨聰莽聧漏忙驴隆莽聜聙聳茅聨赂莽聜聢聤茅隆聯莽聜聧茂陆聣茅聬聬氓炉赂氓聨陆茅聳鹿猫聣掳忙聫禄卤忙聸垄忙聜戮茅隆聯茫聢隆莽聟聲茅聨录莽虏鹿芒聜卢氓庐聽氓聼聻茅聳戮猫聧陇氓聛聻莽禄禄忙露垄氓麓卤氓娄莽垄聧氓漏碌茅聬聯茂驴聽氓录禄茅聰聸氓聥虏陋茅聳赂忙聮聴氓聥聻卤茅聳禄氓潞垄茅聧拢氓麓聹氓篓聭茫聢聽莽庐聳禄忙篓驴莽禄聽莽聰赂忙聼聼茅聬聭忙莽庐聳莽录聜氓露聟氓聙禄莽禄卤忙聮禄氓麓聮氓搂聵氓聛聯忙陇聥氓潞垄忙驴篓潞莽聣聬聭猫聴聣莽拢录茅聳禄忙聞炉氓掳聶茅聬聯忙聢娄忙垄潞茅聧聸氓漏聜氓搂聜茅聳赂忙聳驴茅聟拢氓炉聤陆忙聲聜茅聬拢氓虏聙茅聴聛莽鲁聲氓聣聺猫聯卤茅聬聫氓驴拢氓聨录茅聹聣茅聳录忙碌聥忙聶聸茅聢禄氓聽楼忙聼隆莽聛聻氓聣搂忙麓麓莽聙碌忙聦鲁茅聨聢陇忙聼篓莽聙拢莽掳陆忙驴聽莽聰碌氓搂路茅聫聛氓聭聤忙聟篓茅聬聭猫聴聣忙聻聛茅聳鹿茅聴麓莽莽潞戮莽禄聵莽鹿聸茅聨麓忙卢聫莽聛陋茅聳赂氓茅聫聦莽聠录忙聜聶茅芒聜戮搂忙聼拢茅聨潞忙聴聙忙芦聲茅聳鹿氓聫聣忙聜聜氓炉麓录茅聤聫莽聤碌茫聜娄茅聴聜盲陆赂忙聠隆茅聧聰氓驴聸氓录虏氓氓聽聺莽拢驴茅聳禄忙聳驴氓聬聥茅聬聯氓驴聯芒聰聭茅聬聬猫聦卢盲禄垄茅聳赂忙聳禄氓聞鲁茅聨聰聭莽聙拢忙聥潞茅聳禄莽聤虏忙麓漏莽聛聫忙卢聭莽虏禄茅聨露猫聤楼莽聟聲茅聬聨茂陆聨氓聣聶忙碌聽猫戮漏芒聜忙麓聵氓篓虏莽聠录莽聧庐忙聬麓茫聞娄氓職聧茅聳碌氓陇聢氓聞庐茅聧聥忙聮禄忙聡聹茅聧聫氓聯楼莽庐職茅聴聛茅聺聸莽聣聠茅聨鲁氓驴聸氓麓鲁氓篓虏莽聟聲茅聬聨茂陆聟氓垄聨莽禄聣忙聼隆莽聛聻氓聣搂忙麓麓氓漏碌茂录聞氓聟聵茅陋聻氓卢聰聭茅聧隆忙篓禄茅聴聜盲陆赂忙聬聫聢莽虏聶茅聨麓茂鹿聙莽聟聺茅聨录忙路聙聴茅聬陇芒聜聴聛茅聺聸茅聧聰氓虏聞氓鹿聠茅聧聝娄莽聟聼茅聨录忙路聙陇氓忙聞鹿聸氓篓麓氓忙芦聡莽录聛猫戮漏氓聛聻莽禄聴莽聠录氓聼聙茅隆聮芒聜卢茅隆聲茂陆聣茅聰聲芒聜卢莽禄聽忙露垄忙聼隆忙戮露氓漏聞盲禄垄莽录聛忙聞颅莽聸猫聶鹿莽麓聮氓娄聹颅莽鹿聛茅聴聛茅聺漏莽露聡莽禄虏盲禄聥茂陆?
    """
    removable_roots = (
        experiment_output_root / "core" / "predictions",
        experiment_output_root / "eval" / "plotting_inputs",
    )
    for removable_root in removable_roots:
        if removable_root.exists():
            shutil.rmtree(removable_root, ignore_errors=False)


def _run_public_benchmark(  
    *,  
    dataset_name: str,  
    raw_root: Path,  
    estimator_cfgs: dict[str, dict[str, Any]],  
    model_cfgs: dict[str, dict[str, Any]],  
    output_root: Path,  
) -> dict[str, Any]:  
    dataset_cfg = copy.deepcopy(  
        _load_yaml_config_cached(str((ROOT / "configs" / "datasets" / f"{dataset_name}.yaml").resolve()))  
    )  
    experiment_cfg = copy.deepcopy(  
        _load_yaml_config_cached(str((ROOT / "configs" / "experiments" / f"e7_{dataset_name}.yaml").resolve()))  
    )  
    readiness_fn = {
        "miluv": inspect_miluv_raw_readiness,
        "ntu_viral": inspect_ntu_viral_raw_readiness,
    }.get(dataset_name)
    if readiness_fn is None:
        raise ValueError(f"unsupported public dataset '{dataset_name}' for readiness inspection")
    readiness_report = readiness_fn(raw_root)
    seq_ids = resolve_public_eval_seq_ids(dataset_name, experiment_cfg, "full")
    ready_seq_ids = list(readiness_report.get("sequence_ids") or [])
    if not ready_seq_ids:  
        raise RuntimeError(f"{dataset_name} readiness report exposed no seq_ids")  
    missing_ready_seq_ids = [seq_id for seq_id in seq_ids if seq_id not in ready_seq_ids]
    if missing_ready_seq_ids:
        raise RuntimeError(
            f"{dataset_name} readiness report is missing frozen public benchmark seq_ids: {missing_ready_seq_ids}"
        )

    dataset_output_root = output_root / dataset_name  
    public_result = PublicBenchmarkPipeline().run(  
        {  
            "project_root": ROOT,  
            "dataset_name": dataset_name,  
            "raw_root": str(raw_root),  
            "field_mapping": dataset_cfg["field_mapping"],  
            "seq_ids": seq_ids,  
            "methods": list(experiment_cfg["methods"]),  
            "estimator_cfgs": estimator_cfgs,  
            "model_cfgs": model_cfgs,  
            "mode": "full",  
            "output_root": str(dataset_output_root / "public"),  
        }  
    )  
    eval_result = EvalPipeline().run(  
        {  
            "project_root": ROOT,  
            "prediction_bundles": _resolve_prediction_bundles(public_result.metadata),  
            "ground_truth_root": str(raw_root),  
            "output_root": str(dataset_output_root / "eval"),  
        }  
    )  
    return {  
        "dataset_name": dataset_name,  
        "public_result": public_result,  
        "eval_result": eval_result,  
        "output_root": str(dataset_output_root),  
        "eval_stage_name": eval_result.stage_name,  
    }  


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    """茅聴聜氓聜職氓聙赂茅聧聤氓鲁掳脙鹿茅聧楼脩聟忙職娄茅聳赂氓聛聟茅聧聶氓驴聸忙聼隆忙戮露氓忙庐驴聻氓路潞聹茅聬聯忙露垄忙聼聸氓芒聜聬聮茂鹿聜氓录虏茅隆聫氓聽聼氓聛隆忙驴聽氓漏聜氓茅聬聫忙聺聭芒聰?CSV 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜麓莽聣聽茅隆垄忙聸垄忙聲聝茅聢搂氓茅聧娄莽拢录茅聬聨茂陆聨氓聛聯莽禄卤芒聲聜莽鹿聸氓庐赂莽掳录茅聳潞氓聭庐莽聟隆茅聫聦忙露聵芒聵聣茅聧聶忙氓聡鹿茅聴聛猫炉虏茅茅隆鲁掳氓篓虏忙驴聻忙聢聻忙掳篓茅聬陇氓聥炉忙垄潞莽禄聥氓聦隆忙聰禄忙陇陇茫聞楼忙聡聵茅聨庐茅鹿聝氓麓鹿茅隆聬茫聜聡忙聡聺茅聴聛茅聙聻氓卤陆氓垄麓莽聙碌茅聨庐茫聢陇氓麓芦茅聤聣脩聝茅陋聞茅聴聜盲陆潞莽隆露茅聧聯忙驴聥忙聜路猫陇聨莽禄聥氓陇聤忙戮露氓卢聨赂茅聳禄氓驴聲氓聛聽茅脩聡氓麓聺氓氓聽聼莽聟聼茅聳鹿忙聝搂莽聯篓莽禄聙氓卢聼隆莽聛聻莽聜卢莽陇聝莽录聛忙聴聜氓聳聳茅隆垄忙露聵茅隆聮盲陆鹿莽录聜氓聜聶莽路隆茅聧聽忙聽聜楼茅聬聯氓潞庐忙陆聳忙驴聻氓路潞聺茅聧聤氓忙聼聧忙聺聻忙聣麓掳忙驴聻氓聟录氓虏赂氓忙麓鹿聳茅聬聬忙聫聮氓娄搂茅聴聛氓聭聤莽虏聢茅聬聮茂鹿聛芒聟露氓漏碌?茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛氓露聣氓麓聼茅盲陆鹿忙驴聽氓戮路忙聺驴芒聜聸垄氓鹿聤氓庐聙氓聥聞碌忙陇陇忙聨聠氓聛聯茅聳赂莽聡聛氓聻掳茅隆聲茫聝漏氓麓聼茅忙掳卤莽聳聙氓娄聻茫聜聠氓聻聼茅聫聜氓聜聻氓麓聲茅聬垄莽猫聯聼忙驴聻忙聢聶氓聻鹿莽禄?

    茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聫聛忙聮禄忙聜漏茅聧聬猫鹿聰禄茅聴聜盲戮聙忙陆搂茅隆娄氓录虏氓篓聭忙掳聳赂茂赂禄氓聨陆茅聧聽忙聞掳忙聼拢氓娄陇芒聜聬聴氓聴聶莽虏颅茅聨潞忙聮麓氓聫?
    ----------
    path : Path
        CSV 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜麓莽聣聽茅隆垄忙聸垄忙聲聝茅聢搂氓茅聧娄莽拢录茅聬聨茂陆聨氓聛聯莽禄卤芒聲聜莽鹿聸氓庐赂莽掳录茅聳潞氓聭庐莽聟隆茅聫聦忙露聵芒聵聣茅聧聶忙氓聡鹿茅聴聛猫炉虏茅茅隆鲁掳氓篓虏忙驴聻忙聢聻忙掳篓茅聬陇氓聥炉忙垄潞茅聨录莽聜虏氓茅聳鹿莽垄卤氓聛聤茅聨庐?莽录聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓麓茂陆聢氓篓聟莽聤禄氓鹿聳氓篓拢氓莽禄戮忙聝搂茅聫聦忙聸垄氓麓录氓漏碌氓聸露忙莽录?

    茅聴聜氓聜職氓聙赂茅聧聤茅拢聨莽虏聢忙赂職芒聜卢茅陋聻忙聽颅盲陆聧茅聧楼忙聲聝茅聳驴忙聸聴氓聛聯茅聳禄茫聜聡氓聛聯莽禄禄忙驴聡莽鹿聺茅聧聦忙禄聝茅聴聛猫聴聣氓聲掳茅聧聽忙聽录禄茅聫聡莽聠潞莽庐戮茅聳赂氓聳聳忙戮鹿?
    -------
    list[dict[str, Any]]
        忙驴聽莽聰碌氓搂路茅聫聛氓聭聤忙聟篓茅聬聭猫聴聣忙聻聛茅聳鹿茅聴麓莽莽禄聙氓聴聸忙聼聼氓篓聢氓聻庐忙聻陇莽禄戮?氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢禄茅聳禄忙篓驴莽禄聽氓聻聛聞猫鹿聡忙聮麓莽录聛猫潞虏氓聙聺茅聫聦忙露垄氓鹿聵氓娄陇芒聜卢茅聧聼忙聜聵忙驴聽氓聸篓芒聮聭茅聳赂忙聮麓氓陆聝忙碌聹忙聽颅忙聼聸茅聤聤茫聞娄茅聳鲁脩聦氓聣職茅聭鹿茅鹿聝芒聙娄茅聬聭猫聴聣猫聯聼茅聳驴忙驴聠氓娄漏聙忙聼聼莽禄聥氓聻聳赂莽聤鲁氓聤聶氓搂聤氓聯聞莽聲路茅聧楼芒聲聛莽氓漏碌芒聵聠氓聛聞茅聧聼忙聺聭莽聮聡氓路聬聥茅聳赂氓聛聟茅隆聨氓聸卢忙垄禄忙碌拢氓聭聤氓聲聻茅聳鹿忙颅聦茅隆驴聥氓娄赂猫茅聧?茅聴聜氓聜職氓聙赂茅聧聤莽聝陆忙聡聴茅聧聸聻莽拢驴茅聳赂氓聛聟茅聧聶氓驴聲芒聜卢氓鹿驴氓聞卤茅隆娄忙聜聶忙驴聽氓聥卢氓聳聬茅聼芦氓聸篓猫聸聥茅聧楼忙篓聞氓漏碌茂鹿陇茅隆鲁掳莽聲路茅聨潞忙聢陆漏茅聳赂忙氓聞聬茅聫聠茫聜聣忙垄禄忙碌拢氓聭聤忙聝聢茅聳鹿氓聠虏莽鹿聮茅聨鹿茫聢陇氓麓聼茅莽聜虏莽聛聤茅聴聛氓聣虏氓聛聠茅聧聽氓潞拢忙茅聬聬莽聝聵莽聟聶茅聴聛莽庐聭茅陋聻忙聞掳忙聼聼?茅聴聜氓聜職氓聙路氓篓麓氓聸搂莽聲路茅聧篓氓聫聫茅聳禄茫聢垄莽禄聙氓陇聥盲驴聤茅聤聢氓聭庐氓聶鹿莽录聛忙聞聫聦芒聙鲁莽聲碌茅隆聰氓聟录氓聶潞忙驴聻氓聽聼氓聞鲁茅聢鹿忙聢娄茅聫聜茫聜聞莽路碌茅聴聛莽篓驴忙碌路茅聫聛猫茅隆聲芒聜卢氓庐聲莽聟聨芒聰聭茅聧芦忙驴聠脨聲氓漏碌脗掳氓聙聴忙驴庐莽聝陆氓麓聭氓篓聭茫聢聽莽聳聳潞氓陇聥氓聼聢茅聧陇忙聸垄忙聼聼莽录聛茫聢聽忙聻聸忙陇陇忙聡聵茅聫?

    忙驴聽莽聰碌氓搂路茅聫聛氓聭聤忙聟篓茅聬聭猫聴聣忙聻聛茅聳鹿茅聴麓莽莽潞戮莽禄聵莽鹿聸茅聨麓忙卢聫莽聛陋茅聳禄忙聮麓忙麓聼茅聫聦茂驴聽氓麓聮氓漏碌氓聸漏茫聙聤茅聳潞盲陆赂莽聣聯茅聨?
    -----
    氓漏?``_read_metric_rows`` 茅聴聜氓聜職氓聙赂茅聧聤忙聬聛氓麓聬忙陇聥氓潞拢莽聼聠氓篓聯忙掳拢氓聛聯忙楼聽莽聜虏茅聧篓茅鹿聝氓鹿聡忙碌拢氓聹篓聛氓漏碌莽聤碌忙聲赂忙驴庐莽聰赂忙聡聺茅聧聯脩聟莽聨禄忙驴隆茫聜聟忙聡聫茅聬聯忙露垄忙聼聸茅聤聣茂陆聟氓聤職茅聳禄氓驴聤忙聶聺莽禄卤忙聨聴氓聼聙茅隆聮氓聥芦莽拢录忙驴聻忙聢聻莽禄聽忙掳聤聢氓聴聶氓莽录聛猫氓聙驴氓庐聲茫聝篓茫聛聬茅聨潞忙聮露氓聨聯莽录聜氓聜聶莽聞娄猫聯卤忙陇陇茫聝楼莽聣聯茅聫聦茂录聞盲禄娄茅聬拢氓卤聺茫聝娄茫聛聞猫莽禄露忙掳露氓录禄氓篓聭茫聢聽莽茅聬聽猫聶鹿氓聞陇茅聬聫氓聽聺脦聰茅聬聵茅聺聸盲禄聹茅聳驴忙聸聵氓聙聺茅聧漏茫聝娄氓戮聞忙驴聻莽聜卢莽聙禄茅聴聛氓陆聮氓聛聤茅聧聽忙掳鲁氓聙麓茅聴聜氓聜職氓聙路莽禄聙盲戮聙氓鹿聣茅聰聼莽聤虏莽茅聳赂芒聞聝莽篓聭氓忙驴聡茫聛聞茅聨戮氓芒聤聲茅聳潞氓聜聴氓聛聯莽录聜氓聜職氓聙路莽禄聙盲戮聙氓鹿聳茅茫聢漏莽聞陆茅聳鲁茫聝楼莽拢聫聠?氓漏碌莽聤碌忙聲赂忙驴庐莽聝陆氓录芦茅聨录盲陆赂莽拢驴茅聳鹿氓炉赂氓搂麓莽禄露茫聜聣忙聼娄氓娄炉盲戮拢娄忙驴聻氓路庐盲潞聺茅聫聟忙禄聢忙聼拢茅聨掳茅聺聸氓垄聨录忙聺聭茂鹿陋氓搂聤忙麓陋氓麓篓忙驴聽氓聠篓茅聴聛忙聬聻氓聤聧氓漏垄忙篓禄忙聶楼茅聴聛氓聯聞猫茅聧聨忙聼楼忙聜隆茅聧聬芒聰驴盲潞聹忙陇陇忙聢拢脨聮茅聧聥忙聮鲁莽聲路茅聧楼芒聰聞漏忙垄潞莽聮聡芒聙鲁忙聻聯茅聳潞氓聭炉氓搂麓茅聬拢氓鲁掳芒聰聭茅聧隆芒聲聜氓聝碌茅聴聛莽聟聨忙聭聹茅聫聛忙聬聛氓麓垄茅聧聴莽聜漏忙聰陆忙陇陇忙聴聙忙聻禄忙赂職忙露垄忙聼聸茅聨戮猫路潞忙聻聸莽聙鹿忙聸聼忙陇陆茅聧漏芒聜陇陇忙聨聭氓聙禄莽潞戮莽聲聦芒聜卢氓鹿驴氓聞卤茅聨路忙聢娄莽聟氓聥聟聲茅聳鲁猫氓聞鲁忙碌聹氓聣搂莽麓聯茅聧聦忙卢聭忙陇陇忙聢聺忙拢聺茅陋聻忙聢娄氓麓露茅隆聮芒聜卢莽禄聽忙聽颅忙聼拢茅聨麓茂陆聠莽录聛莽聤虏茅聨庐忙楼聟忙氓篓聭忙卢聬氓聳聬氓茫聜聠忙垄鹿氓篓虏莽聠路茅聯聫猫氓聳聠茅聳赂忙聸篓氓聛聞忙驴聢聽茅聤聢氓聴聵氓陇聞茅聳赂忙卢聫氓聲麓茅聬聸脦聰茅聧聸氓娄聻茫聜聠忙垄禄茅聫聟氓麓垄忙碌聽氓卢聮聭茅聴聜氓聽聼盲赂鹿氓篓聭茫聢聽莽茅聳鲁氓聯聞氓聲漏茅聳颅忙垄禄茅聧聦忙卢聬莽楼掳忙陇陇忙聸聼莽聣聽氓庐聲盲录麓氓录麓茅聬聵忙聫聮莽聛聤氓漏碌莽聜虏忙茅聧聰忙聨聰忙聧垄茅聧隆忙露聶莽聟聶茅聴聜氓聜職茅聧聰氓露聣忙聼隆茅聧聸氓聫聙茅聳潞氓卤戮莽篓聯忙碌聽茂鹿聜氓麓聹猫陇聫茅聧聶忙驴聡莽鹿聺?
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _run_summary_and_figures(eval_root: Path, output_root: Path) -> dict[str, Any]:
    """Run summary and figure publication after fail-closed verification."""

    def _normalize_plot_manifest(payload: Any, *, consumer_name: str) -> dict[str, Any]:
        if isinstance(payload, (str, Path)):
            return {"figure_path": str(Path(payload)), "metrics": []}
        if not isinstance(payload, dict):
            raise TypeError(f"{consumer_name} manifest must be a mapping or path-like")
        if payload.get("skipped"):
            return dict(payload)
        figure_path = payload.get("figure_path")
        if not isinstance(figure_path, (str, Path)):
            raise KeyError(f"{consumer_name} manifest missing figure_path")
        normalized_payload = dict(payload)
        normalized_payload["figure_path"] = str(Path(figure_path))
        metrics = normalized_payload.get("metrics")
        if metrics is None:
            normalized_payload["metrics"] = []
        elif isinstance(metrics, str):
            normalized_payload["metrics"] = [metrics]
        elif isinstance(metrics, dict):
            raise TypeError(f"{consumer_name} manifest metrics must be a sequence, not a mapping")
        elif not isinstance(metrics, list):
            normalized_payload["metrics"] = list(metrics)
        return normalized_payload

    def _normalize_case_group_payload(payload: Any, *, group_name: str) -> dict[str, str]:
        if isinstance(payload, (str, Path)):
            return {"figure_path": str(Path(payload))}
        if not isinstance(payload, dict):
            raise TypeError(f"{group_name} case figure entry must be a mapping or path-like")
        figure_path = payload.get("figure_path")
        if not isinstance(figure_path, (str, Path)):
            raise KeyError(f"{group_name} case figure entry missing figure_path")
        return {"figure_path": str(Path(figure_path))}

    def _build_canonical_case_manifest(payload: Any, *, group_name: str) -> dict[str, Any]:
        normalized_payload = _normalize_case_group_payload(payload, group_name=group_name)
        if isinstance(payload, dict) and "case_refs" in payload:
            case_refs = _coerce_case_refs(payload.get("case_refs"))
            return {
                "figure_path": normalized_payload["figure_path"],
                "case_refs": case_refs,
            }
        return normalized_payload

    def _coerce_case_refs(payload: Any) -> list[Any]:
        if payload is None:
            return []
        if isinstance(payload, str):
            return [payload]
        if isinstance(payload, dict):
            raise TypeError("summary case_refs must be a sequence, not a mapping")
        try:
            return list(payload)
        except TypeError as exc:
            raise TypeError("summary case_refs must be a sequence, string, or null") from exc

    def _rebase_path_string(path_text: str, *, old_root: Path, new_root: Path) -> str:
        resolved_path = Path(path_text).resolve()
        try:
            relative_path = resolved_path.relative_to(old_root.resolve())
        except ValueError:
            return str(resolved_path)
        return str((new_root / relative_path).resolve())

    def _rebase_plot_manifest_paths(
        payload: dict[str, Any],
        *,
        old_root: Path,
        new_root: Path,
    ) -> dict[str, Any]:
        rebased_payload = dict(payload)
        figure_path = rebased_payload.get("figure_path")
        if isinstance(figure_path, (str, Path)):
            rebased_payload["figure_path"] = _rebase_path_string(
                str(figure_path),
                old_root=old_root,
                new_root=new_root,
            )
        return rebased_payload

    def _rebase_case_manifest_paths(
        payload: dict[str, dict[str, Any]],
        *,
        old_root: Path,
        new_root: Path,
    ) -> dict[str, dict[str, Any]]:
        rebased_payload: dict[str, dict[str, Any]] = {}
        for group_name, group_payload in payload.items():
            rebased_payload[group_name] = _rebase_plot_manifest_paths(
                group_payload,
                old_root=old_root,
                new_root=new_root,
            )
        return rebased_payload

    def _coerce_manifest_metric_list(payload: Any, *, field_name: str) -> list[Any]:
        if payload is None:
            return []
        if isinstance(payload, str):
            return [payload]
        if isinstance(payload, dict):
            raise TypeError(f"{field_name} must be a sequence, string, or null")
        try:
            return list(payload)
        except TypeError as exc:
            raise TypeError(f"{field_name} must be a sequence, string, or null") from exc

    staging_root = output_root / "_summary_and_figures_staging"
    if staging_root.exists():
        shutil.rmtree(staging_root, ignore_errors=False)
    staging_summary_dir = staging_root / "summary"
    staging_figures_dir = staging_root / "figures"
    commit_root = output_root / "_summary_and_figures_commit"
    commit_summary_dir = commit_root / "summary"
    commit_figures_dir = commit_root / "figures"
    final_summary_dir = output_root / "summary"
    final_figures_dir = output_root / "figures"
    summary_backup_dir = output_root / "_summary_backup"
    figures_backup_dir = output_root / "_figures_backup"

    try:
        metric_rows = _read_csv_rows(eval_root / "metrics" / "metric_table.csv")
        if not metric_rows:
            raise ValueError("metric_table.csv must be non-empty")
        required_metric_keys = {"case_ref", "seq_id", "scene_id", "method_name", "metric", "value"}
        missing_metric_keys = sorted(required_metric_keys - set(metric_rows[0].keys()))
        if missing_metric_keys:
            raise KeyError(f"metric_table.csv missing required columns: {missing_metric_keys}")

        statistics_payload = _read_json(eval_root / "statistics" / "statistics_table.json")
        if not isinstance(statistics_payload, dict):
            raise TypeError("statistics_table.json must contain a mapping")
        required_statistics_keys = ("method_summary", "pairwise_tests")
        missing_statistics_keys = [key for key in required_statistics_keys if key not in statistics_payload]
        if missing_statistics_keys:
            raise KeyError(f"statistics_table.json missing required keys: {missing_statistics_keys}")

        selected_cases = _read_json(eval_root / "cases" / "selected_cases.json")
        if not isinstance(selected_cases, dict):
            raise TypeError("selected_cases.json must contain a mapping")
        required_case_groups = ("main_cases", "failure_cases", "boundary_cases")
        missing_case_groups = [name for name in required_case_groups if name not in selected_cases]
        if missing_case_groups:
            raise KeyError(f"selected_cases.json missing required groups: {missing_case_groups}")

        main_table_input_path = eval_root / "plotting_inputs" / "main_table.json"
        if not main_table_input_path.is_file():
            raise RuntimeError("summary and figure consumers require eval/plotting_inputs/main_table.json")
        main_table_rows = _read_json(main_table_input_path)
        if not isinstance(main_table_rows, list) or not main_table_rows:
            raise ValueError("main_table.json must contain a non-empty list")
        if not isinstance(main_table_rows[0], dict):
            raise TypeError("main_table.json rows must be mappings")

        runtime_rows = _read_json(eval_root / "plotting_inputs" / "runtime_table.json")
        if not isinstance(runtime_rows, list) or not runtime_rows:
            raise ValueError("runtime_table.json must contain a non-empty list")
        if not isinstance(runtime_rows[0], dict):
            raise TypeError("runtime_table.json rows must be mappings")
        required_runtime_keys = {"case_ref", "latency_mean", "latency_p50", "latency_p95", "params", "ram_peak"}
        missing_runtime_keys = sorted(required_runtime_keys - set(runtime_rows[0].keys()))
        if missing_runtime_keys:
            raise KeyError(f"runtime_table.json missing required columns: {missing_runtime_keys}")

        sweep_rows = None
        sweep_input_path = eval_root / "plotting_inputs" / "sweep_table.json"
        if sweep_input_path.is_file():
            sweep_rows = _read_json(sweep_input_path)
            if not isinstance(sweep_rows, list) or not sweep_rows:
                raise ValueError("sweep_table.json must contain a non-empty list")
            if not isinstance(sweep_rows[0], dict):
                raise TypeError("sweep_table.json rows must be mappings")

        trajectory_bundle = None
        trajectory_input_path = eval_root / "plotting_inputs" / "trajectory_bundle.json"
        if trajectory_input_path.is_file():
            trajectory_bundle = _read_json(trajectory_input_path)
            if not isinstance(trajectory_bundle, dict):
                raise TypeError("trajectory_bundle.json must contain a mapping")

        summary_payload = build_summary(main_table_rows, statistics_payload, selected_cases)
        if not isinstance(summary_payload, dict):
            raise TypeError("summary payload must be a mapping")
        summary_stats = summary_payload.get("summary_stats", {})
        if not isinstance(summary_stats, dict):
            raise TypeError("summary_stats must be a mapping")
        case_refs = _coerce_case_refs(summary_payload.get("case_refs"))
        if not isinstance(case_refs, list):
            raise TypeError("summary case_refs must resolve to a list")

        summary_path = Path(_write_json(staging_summary_dir / "summary.json", {"summary": summary_payload}))
        figures_root = staging_figures_dir
        figures_root.mkdir(parents=True, exist_ok=True)

        main_table_manifest = _normalize_plot_manifest(
            render_main_table_figure(main_table_rows, {"figure_path": figures_root / "main_table.png"}),
            consumer_name="main_table",
        )
        runtime_manifest = _normalize_plot_manifest(
            render_runtime_figure(runtime_rows, {"figure_path": figures_root / "runtime.svg", "return_manifest": True}),
            consumer_name="runtime",
        )
        calibration_manifest = _normalize_plot_manifest(
            render_calibration_figure(metric_rows, {"figure_path": figures_root / "calibration.svg", "return_manifest": True}),
            consumer_name="calibration",
        )
        sweep_manifest = None
        if sweep_rows is not None:
            sweep_manifest = _normalize_plot_manifest(
                render_sweep_figure(sweep_rows, {"figure_path": figures_root / "sweep.svg", "return_manifest": True}),
                consumer_name="sweep",
            )
        trajectory_manifest = None
        if trajectory_bundle is not None:
            trajectory_manifest = _normalize_plot_manifest(
                render_trajectory_figure(trajectory_bundle, {"figure_path": figures_root / "trajectory.png"}),
                consumer_name="trajectory",
            )

        case_manifest = render_case_figures(selected_cases, {"figure_path": figures_root / "cases.png"})
        if not isinstance(case_manifest, dict):
            raise TypeError("case figure manifest must be a mapping")
        normalized_case_manifest: dict[str, dict[str, Any]] = {}
        for group_name in required_case_groups:
            if group_name not in case_manifest:
                raise KeyError(f"case figure manifest missing expected group: {group_name}")
            normalized_case_manifest[group_name] = _build_canonical_case_manifest(
                case_manifest[group_name],
                group_name=group_name,
            )

        main_table_figure_path = Path(main_table_manifest["figure_path"])
        runtime_figure_path = Path(runtime_manifest["figure_path"])
        calibration_figure_path = Path(calibration_manifest["figure_path"])
        if not main_table_figure_path.is_file():
            raise RuntimeError(f"main table figure missing after render: {main_table_figure_path}")
        if not runtime_figure_path.is_file():
            raise RuntimeError(f"runtime figure missing after render: {runtime_figure_path}")
        if not calibration_figure_path.is_file():
            raise RuntimeError(f"calibration figure missing after render: {calibration_figure_path}")
        if sweep_manifest is not None:
            sweep_figure_path = Path(sweep_manifest["figure_path"])
            if not sweep_figure_path.is_file():
                raise RuntimeError(f"sweep figure missing after render: {sweep_figure_path}")
        if trajectory_manifest is not None:
            trajectory_figure_path = Path(trajectory_manifest["figure_path"])
            if not trajectory_figure_path.is_file():
                raise RuntimeError(f"trajectory figure missing after render: {trajectory_figure_path}")
        for group_name, group_payload in normalized_case_manifest.items():
            figure_path = Path(group_payload["figure_path"])
            if not figure_path.is_file():
                raise RuntimeError(f"{group_name} case figure missing after render: {figure_path}")

        figure_manifest_payload: dict[str, Any] = {
            "main_table": main_table_manifest,
            "runtime": runtime_manifest,
            "calibration": calibration_manifest,
            "cases": normalized_case_manifest,
        }
        if sweep_manifest is not None:
            figure_manifest_payload["sweep"] = sweep_manifest
        if trajectory_manifest is not None:
            figure_manifest_payload["trajectory"] = trajectory_manifest

        final_figure_manifest_payload: dict[str, Any] = {
            "main_table": _rebase_plot_manifest_paths(
                main_table_manifest,
                old_root=staging_figures_dir,
                new_root=final_figures_dir,
            ),
            "runtime": _rebase_plot_manifest_paths(
                runtime_manifest,
                old_root=staging_figures_dir,
                new_root=final_figures_dir,
            ),
            "calibration": _rebase_plot_manifest_paths(
                calibration_manifest,
                old_root=staging_figures_dir,
                new_root=final_figures_dir,
            ),
            "cases": _rebase_case_manifest_paths(
                normalized_case_manifest,
                old_root=staging_figures_dir,
                new_root=final_figures_dir,
            ),
        }
        if sweep_manifest is not None:
            final_figure_manifest_payload["sweep"] = _rebase_plot_manifest_paths(
                sweep_manifest,
                old_root=staging_figures_dir,
                new_root=final_figures_dir,
            )
        if trajectory_manifest is not None:
            final_figure_manifest_payload["trajectory"] = _rebase_plot_manifest_paths(
                trajectory_manifest,
                old_root=staging_figures_dir,
                new_root=final_figures_dir,
            )
        manifest_path = Path(
            _write_json(
                staging_figures_dir / "figure_manifest.json",
                {"figure_manifest": final_figure_manifest_payload},
            )
        )
        if commit_root.exists():
            shutil.rmtree(commit_root, ignore_errors=False)
        if summary_backup_dir.exists():
            shutil.rmtree(summary_backup_dir, ignore_errors=False)
        if figures_backup_dir.exists():
            shutil.rmtree(figures_backup_dir, ignore_errors=False)
        commit_root.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging_summary_dir), str(commit_summary_dir))
        shutil.move(str(staging_figures_dir), str(commit_figures_dir))

        had_summary_dir = final_summary_dir.exists()
        had_figures_dir = final_figures_dir.exists()
        published_summary_dir = False
        published_figures_dir = False
        try:
            if had_summary_dir:
                final_summary_dir.replace(summary_backup_dir)
            if had_figures_dir:
                final_figures_dir.replace(figures_backup_dir)
            commit_summary_dir.replace(final_summary_dir)
            published_summary_dir = True
            commit_figures_dir.replace(final_figures_dir)
            published_figures_dir = True
        except Exception:
            if published_summary_dir and final_summary_dir.exists():
                final_summary_dir.replace(commit_summary_dir)
            if published_figures_dir and final_figures_dir.exists():
                final_figures_dir.replace(commit_figures_dir)
            if had_summary_dir and summary_backup_dir.exists() and not final_summary_dir.exists():
                summary_backup_dir.replace(final_summary_dir)
            if had_figures_dir and figures_backup_dir.exists() and not final_figures_dir.exists():
                figures_backup_dir.replace(final_figures_dir)
            raise

        if summary_backup_dir.exists():
            shutil.rmtree(summary_backup_dir, ignore_errors=False)
        if figures_backup_dir.exists():
            shutil.rmtree(figures_backup_dir, ignore_errors=False)
        if commit_root.exists():
            shutil.rmtree(commit_root, ignore_errors=False)
        return {
            "summary_path": str((final_summary_dir / summary_path.name).resolve()),
            "figure_manifest_path": str((final_figures_dir / manifest_path.name).resolve()),
            "summary": {
                "case_refs": list(case_refs),
                "summary_stats_keys": sorted(summary_stats.keys()),
            },
            "plot_manifests": {
                "runtime_metrics": _coerce_manifest_metric_list(
                    final_figure_manifest_payload["runtime"].get("metrics"),
                    field_name="runtime manifest metrics",
                ),
                "calibration_metrics": _coerce_manifest_metric_list(
                    final_figure_manifest_payload["calibration"].get("metrics"),
                    field_name="calibration manifest metrics",
                ),
                "sweep": final_figure_manifest_payload.get("sweep"),
                "trajectory": final_figure_manifest_payload.get("trajectory"),
            },
        }
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
        if commit_root.exists():
            shutil.rmtree(commit_root, ignore_errors=True)
        if summary_backup_dir.exists():
            shutil.rmtree(summary_backup_dir, ignore_errors=True)
        if figures_backup_dir.exists():
            shutil.rmtree(figures_backup_dir, ignore_errors=True)


def _run_high_level_consumer_verification(*, input_root: Path, ground_truth_root: Path, output_root: Path) -> dict[str, Any]:
    """Run the high-level consumer verification script and fail closed on contract violations."""

    class _HighLevelConsumerVerificationError(RuntimeError):
        """Mirror the contract enforced by 19_verify_high_level_consumers.py."""

        def __init__(self, payload: dict[str, Any]) -> None:
            self.payload = payload
            super().__init__(str(payload))

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "19_verify_high_level_consumers.py"),
            "--input-root",
            str(input_root),
            "--ground-truth-root",
            str(ground_truth_root),
            "--output-root",
            str(output_root),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        stdout_payload = _loads_json_from_stdout(
            completed.stdout,
            source_name="19_verify_high_level_consumers.py",
        )
    except RuntimeError:
        raise

    if completed.returncode != 0:
        if isinstance(stdout_payload, dict):
            raise _HighLevelConsumerVerificationError(dict(stdout_payload))
        raise RuntimeError(f"19_verify_high_level_consumers.py failed: {stdout_payload}")
    if not isinstance(stdout_payload, dict):
        raise RuntimeError(
            f"19_verify_high_level_consumers.py returned invalid payload type: {type(stdout_payload).__name__}"
        )
    if str(stdout_payload.get("status") or "") != "ok":
        raise RuntimeError(f"19_verify_high_level_consumers.py returned non-ok status payload: {stdout_payload}")
    if str(stdout_payload.get("stage") or "") != "high_level_consumers":
        raise RuntimeError(
            f"19_verify_high_level_consumers.py returned unexpected stage payload: {stdout_payload}"
        )
    if int(stdout_payload.get("exit_code", -1)) != 0:
        raise RuntimeError(
            f"19_verify_high_level_consumers.py returned non-zero exit_code payload: {stdout_payload}"
        )
    required_keys = ("artifacts", "route_bridge", "summary", "plot_manifests", "eval_stage_name")
    missing_keys = [key for key in required_keys if key not in stdout_payload]
    if missing_keys:
        raise RuntimeError(
            f"19_verify_high_level_consumers.py payload missing required keys: {missing_keys}; payload={stdout_payload}"
        )
    return stdout_payload


def main(argv: list[str] | None = None) -> int:  
    parser = argparse.ArgumentParser(description="Run the paper-grade experiment chain with real-data gates")  
    parser.add_argument("--output-root", default=None)  
    parser.add_argument("--sim-raw-root", default=None)  
    parser.add_argument("--miluv-raw-root", default=None)  
    parser.add_argument("--ntu-viral-raw-root", default=None)  
    parser.add_argument("--miluv-official-root", default=None)  
    parser.add_argument("--ntu-viral-official-root", default=None)  
    parser.add_argument("--miluv-runtime-root", default=None)  
    parser.add_argument("--ntu-viral-runtime-root", default=None)  
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")  
    parser.add_argument("--public-datasets", nargs="*", choices=_SUPPORTED_PUBLIC_DATASETS, default=None)  
    parser.add_argument("--skip-public-benchmarks", action="store_true")  
    parser.add_argument("--reuse-search-cache", action="store_true")  
    parser.add_argument("--allow-incomplete-paper-run", action="store_true")  
    parser.add_argument("--paper-scope", choices=("full", "sim_e9_only"), default="full")  
    parser.add_argument("--neural-search-profile", choices=("paper", "small_eval", "medium_eval"), default="paper")  
    parser.add_argument("--auto-materialize-sim", dest="auto_materialize_sim", action="store_true")  
    parser.add_argument("--no-auto-materialize-sim", dest="auto_materialize_sim", action="store_false")  
    parser.add_argument("--lstm-epochs", type=int, default=None)  
    parser.add_argument("--liquid-epochs", type=int, default=None)  
    parser.add_argument("--batch-size", type=int, default=None)  
    parser.add_argument("--eval-batch-size", type=int, default=None)  
    parser.add_argument("--seed", type=int, default=0)  
    parser.set_defaults(auto_materialize_sim=True)  
    args = parser.parse_args(argv)  

    _wrapper_scope = globals().get("_ACTIVE_PAPER_SCOPE")
    paper_scope = _set_active_paper_scope(_wrapper_scope if _wrapper_scope is not None else args.paper_scope)

    output_root = _resolve_output_root(args.output_root)  
    output_root.mkdir(parents=True, exist_ok=True)  
    seed_manifest = _build_run_seed_manifest(args.seed)  
    _write_json(output_root / "audits" / "seed_manifest.json", seed_manifest)  
    requested_public_datasets = _resolve_requested_public_datasets(  
        args.public_datasets,  
        skip_public_benchmarks=bool(args.skip_public_benchmarks),  
    )  
    _wrapper_requested_public_datasets = globals().get("_WRAPPER_REQUESTED_PUBLIC_DATASETS")
    if _wrapper_requested_public_datasets is not None:
        requested_public_datasets = list(_wrapper_requested_public_datasets)
    neural_search_surface = _resolve_neural_search_surface(args.neural_search_profile)  
    resolved_training_budget = _resolve_training_budget_overrides(
        neural_search_surface,
        lstm_epochs=args.lstm_epochs,
        liquid_epochs=args.liquid_epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
    )
    neural_search_surface = copy.deepcopy(neural_search_surface)
    neural_search_surface["training_budget"] = dict(resolved_training_budget)
    sim_raw_root = _resolve_dataset_raw_root("sim", args.sim_raw_root)  
    default_sim_raw_root = _resolve_dataset_raw_root("sim", None)  
    miluv_raw_root = _resolve_dataset_raw_root("miluv", args.miluv_raw_root)  
    ntu_viral_raw_root = _resolve_dataset_raw_root("ntu_viral", args.ntu_viral_raw_root)  
    public_roots = {  
        "miluv": {
            "official_root": _resolve_optional_path(args.miluv_official_root, flag_name="--miluv-official-root"),
            "landed_raw_root": str(miluv_raw_root),
            "runtime_root": _resolve_optional_path(args.miluv_runtime_root, flag_name="--miluv-runtime-root"),
        },
        "ntu_viral": {
            "official_root": _resolve_optional_path(args.ntu_viral_official_root, flag_name="--ntu-viral-official-root"),
            "landed_raw_root": str(ntu_viral_raw_root),
            "runtime_root": _resolve_optional_path(args.ntu_viral_runtime_root, flag_name="--ntu-viral-runtime-root"),
        },
    }
    preflight_paper_run_runtime_context = _build_paper_run_runtime_context(
        seed_manifest_path=str((output_root / "audits" / "seed_manifest.json").resolve()),
        env_report_path=None,
        readiness_report_path=None,
        sim_materialization_report_path=None,
        paper_scope=paper_scope,
        requested_public_datasets=requested_public_datasets,
        reuse_search_cache=bool(args.reuse_search_cache),
        allow_incomplete_paper_run=bool(args.allow_incomplete_paper_run),
        neural_search_profile=str(args.neural_search_profile),
        neural_search_surface=neural_search_surface,
        public_roots=public_roots,
        resolved_training_budget=resolved_training_budget,
    )
    contract_violations = _collect_full_paper_run_contract_violations(  
        requested_public_datasets=requested_public_datasets,  
        skip_public_benchmarks=bool(args.skip_public_benchmarks),  
        reuse_search_cache=bool(args.reuse_search_cache),  
        allow_incomplete_paper_run=bool(args.allow_incomplete_paper_run),  
        requested_device=str(args.device),  
        requested_neural_search_profile=str(args.neural_search_profile),  
        lstm_epochs=args.lstm_epochs,  
        liquid_epochs=args.liquid_epochs,  
        batch_size=args.batch_size,  
        eval_batch_size=args.eval_batch_size,  
    )  
    if contract_violations:  
        contract_block_reason = _build_paper_run_contract_block_reason(contract_violations)
        return _write_blocked_paper_run_report(  
            output_root,  
            {  
                "status": "blocked",  
                "stage": "paper_run_contract",  
                "reason": contract_block_reason["reason"],  
                "message": contract_block_reason["message"],  
                "contract_violations": contract_violations,  
                "required_public_datasets": list(_SUPPORTED_PUBLIC_DATASETS),  
                "skip_public_benchmarks": bool(args.skip_public_benchmarks),  
                **preflight_paper_run_runtime_context,
            },  
        )  

    current_stage = "env_check"
    completed_stages: list[str] = []
    partial_stage_artifacts: dict[str, Any] = dict(preflight_paper_run_runtime_context)
    neural_search_results: dict[str, Any] = {}
    try:
        env_report = _run_env_check(output_root)  
        env_report_path = str(Path(env_report["report_path"]).resolve())
        partial_stage_artifacts["env_report"] = env_report_path
        completed_stages.append(current_stage)

        current_stage = "sim_materialization"
        sim_materialization_report = None  
        if args.auto_materialize_sim and can_materialize_sim_raw(sim_raw_root) and _sim_root_has_only_placeholder(sim_raw_root):  
            sim_materialization_report = _materialize_sim_raw(sim_raw_root, output_root)  
        if sim_materialization_report is not None:
            partial_stage_artifacts["sim_materialization_report"] = str(
                (output_root / "audits" / "sim_materialization.json").resolve()
            )
            completed_stages.append(current_stage)

        current_stage = "data_readiness"
        readiness_reports = {  
            "sim": _inspect_sim_raw_readiness(sim_raw_root),  
            "miluv": inspect_miluv_raw_readiness(miluv_raw_root),  
            "ntu_viral": inspect_ntu_viral_raw_readiness(ntu_viral_raw_root),  
        }  
        sim_contract_report = dict(readiness_reports["sim"].get("sim_materialized_contract_report") or {})
        repo_default_sim_raw_root = (ROOT / "data" / "raw" / "sim").resolve()
        should_refresh_default_sim_raw = (
            bool(args.auto_materialize_sim)
            and args.sim_raw_root is None
            and sim_raw_root == default_sim_raw_root
            and sim_raw_root == repo_default_sim_raw_root
            and bool(sim_contract_report)
            and not bool(sim_contract_report.get("is_valid"))
            and sim_raw_root.is_dir()
        )
        if should_refresh_default_sim_raw:  
            if sim_raw_root.exists():
                shutil.rmtree(sim_raw_root)
            current_stage = "sim_materialization"
            sim_materialization_report = _materialize_sim_raw(sim_raw_root, output_root)
            partial_stage_artifacts["sim_materialization_report"] = str(
                (output_root / "audits" / "sim_materialization.json").resolve()
            )
            if "sim_materialization" not in completed_stages:
                completed_stages.append("sim_materialization")
            readiness_reports["sim"] = _inspect_sim_raw_readiness(sim_raw_root)
            current_stage = "data_readiness"
        for dataset_name, report in readiness_reports.items():  
            if dataset_name == "sim":  
                report["required_for_run"] = True  
            else:  
                report["required_for_run"] = dataset_name in requested_public_datasets  
        readiness_report_path = _write_json(output_root / "audits" / "data_readiness.json", readiness_reports)  
        readiness_report_path = str(Path(readiness_report_path).resolve())
        partial_stage_artifacts["readiness_report"] = readiness_report_path
        paper_run_runtime_context = _build_paper_run_runtime_context(
            seed_manifest_path=str((output_root / "audits" / "seed_manifest.json").resolve()),
            env_report_path=env_report_path,
            readiness_report_path=readiness_report_path,
            sim_materialization_report_path=(
                str((output_root / "audits" / "sim_materialization.json").resolve())
                if sim_materialization_report is not None
                else None
            ),
            paper_scope=paper_scope,
            requested_public_datasets=requested_public_datasets,
            reuse_search_cache=bool(args.reuse_search_cache),
            allow_incomplete_paper_run=bool(args.allow_incomplete_paper_run),
            neural_search_profile=str(args.neural_search_profile),
            neural_search_surface=neural_search_surface,
            public_roots=public_roots,
            resolved_training_budget=resolved_training_budget,
        )
        partial_stage_artifacts.update(paper_run_runtime_context)
        required_readiness_reports = {  
            dataset_name: report  
            for dataset_name, report in readiness_reports.items()  
            if bool(report.get("required_for_run"))  
        }  
        if any(report["status"] != "ready" for report in required_readiness_reports.values()):  
            blocked_report = {  
                "status": "blocked",  
                "stage": "data_readiness",  
                "completed_stages": list(completed_stages),  
                "datasets": {name: report["status"] for name, report in readiness_reports.items()},  
                "dataset_reasons": {  
                    name: list(report.get("reasons") or [])  
                    for name, report in readiness_reports.items()  
                },  
                **partial_stage_artifacts,
            }  
            return _write_blocked_paper_run_report(
                output_root,
                blocked_report,
            )  
        completed_stages.append(current_stage)
    except Exception as exc:
        failure_report = {
            "status": "failed",
            "stage": current_stage,
            "reason": "unhandled_exception",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "completed_stages": list(completed_stages),
            **partial_stage_artifacts,
        }
        return _write_failed_paper_run_report(output_root, failure_report)

    current_stage = "sim_prepare"
    partial_stage_artifacts = dict(paper_run_runtime_context)

    try:
        sim_prepare = _run_prepare("sim", sim_raw_root, output_root / "prepare" / "sim")  
        partial_stage_artifacts["sim_prepare_artifacts"] = sim_prepare.artifacts
        completed_stages.append(current_stage)

        current_stage = "public_prepare"
        public_prepare_results = {}  
        if "miluv" in requested_public_datasets:  
            public_prepare_results["miluv"] = _run_prepare("miluv", miluv_raw_root, output_root / "prepare" / "miluv")  
        if "ntu_viral" in requested_public_datasets:  
            public_prepare_results["ntu_viral"] = _run_prepare(  
                "ntu_viral",  
                ntu_viral_raw_root,  
                output_root / "prepare" / "ntu_viral",  
            )  
        partial_stage_artifacts["miluv_prepare_artifacts"] = (
            public_prepare_results["miluv"].artifacts if "miluv" in public_prepare_results else None
        )
        partial_stage_artifacts["ntu_viral_prepare_artifacts"] = (
            public_prepare_results["ntu_viral"].artifacts if "ntu_viral" in public_prepare_results else None
        )
        completed_stages.append(current_stage)

        current_stage = "split_manifest"
        split_manifest, leak_report = _run_split(sim_prepare, output_root / "splits")  
        partial_stage_artifacts["split_manifest"] = split_manifest
        partial_stage_artifacts["leak_report"] = leak_report
        if not list(split_manifest.get("test_ids") or []):  
            return _write_blocked_paper_run_report(  
                output_root,  
                {  
                    "status": "blocked",  
                    "stage": "split_manifest",  
                    "completed_stages": list(completed_stages),  
                    "reason": "missing_test_ids",  
                    "message": "paper-grade full run requires non-empty held-out test_ids",  
                    **partial_stage_artifacts,
                    "split_manifest": split_manifest,  
                    "leak_report": leak_report,  
                },  
            )  
        completed_stages.append(current_stage)

        current_stage = "classical_search"
        classical_search_result = _search_best_classical_estimator_cfgs(  
            prepare_root=output_root / "prepare" / "sim",  
            raw_root=sim_raw_root,  
            output_root=output_root / "classical_search",  
            split_manifest=split_manifest,  
            allow_cache=bool(args.reuse_search_cache),  
        )  
        partial_stage_artifacts["classical_search_audit"] = str((output_root / "classical_search" / "search_audit.json").resolve())
        estimator_cfgs = classical_search_result["selected_estimator_cfgs"]  
        baseline_scoring_metrics_by_experiment = classical_search_result.get("selected_ekf_metrics_by_experiment")  
        partial_stage_artifacts["selected_estimator_cfgs"] = estimator_cfgs
        partial_stage_artifacts["classical_search"] = {
            "search_audit_path": str((output_root / "classical_search" / "search_audit.json").resolve()),
            "selected_estimator_cfgs": estimator_cfgs,
        }
        completed_stages.append(current_stage)

        current_stage = "neural_search_liquid"
        neural_search_results["liquid_ekf"] = _search_best_neural_train_result(
            model_name="liquid_ekf",
            prepare_root=output_root / "prepare" / "sim",
            raw_root=sim_raw_root,
            split_manifest=split_manifest,
            output_root=output_root / "train" / "liquid_ekf",
            estimator_cfgs=estimator_cfgs,
            device=args.device,
            epochs=int(resolved_training_budget["liquid_epochs"]),
            batch_size=int(resolved_training_budget["batch_size"]),
            eval_batch_size=int(resolved_training_budget["eval_batch_size"]),
            baseline_scoring_metrics_by_experiment=baseline_scoring_metrics_by_experiment,
            neural_grid=neural_search_surface["neural_grid"],
            liquid_robustness_profiles=neural_search_surface["liquid_robustness_profiles"],
            top_k_multiseed=int(neural_search_surface["top_k_multiseed"]),
            neural_multiseeds=tuple(int(seed) for seed in neural_search_surface["multiseed_values"]),
            epoch_candidate_stride=neural_search_surface["epoch_candidate_stride"],
            checkpoint_selection_mode=neural_search_surface["checkpoint_selection_mode"],
            resolved_training_budget=resolved_training_budget,
            allow_cache=bool(args.reuse_search_cache),
        )
        partial_stage_artifacts["train_reports"] = {
            **dict(partial_stage_artifacts.get("train_reports") or {}),
            "liquid_ekf": neural_search_results["liquid_ekf"]["train_result"].metadata["train_report"],
        }
        partial_stage_artifacts["neural_search"] = {
            "liquid_ekf": {
                "search_audit_path": str((output_root / "train" / "liquid_ekf" / "search_audit.json").resolve()),
                "selected_signature": neural_search_results["liquid_ekf"]["search_audit"]["selected_signature"],
                "selected_overrides": neural_search_results["liquid_ekf"]["search_audit"]["selected_overrides"],
                "selected_seed": neural_search_results["liquid_ekf"]["search_audit"]["selected_seed"],
                "selected_score_vector": neural_search_results["liquid_ekf"]["search_audit"]["selected_score_vector"],
            },
        }
        neural_search_results["liquid_ekf"]["search_audit"]["resolved_training_budget"] = {
            "epochs": int(resolved_training_budget["liquid_epochs"]),
            "batch_size": int(resolved_training_budget["batch_size"]),
            "eval_batch_size": int(resolved_training_budget["eval_batch_size"]),
            "budget_source": str(resolved_training_budget["budget_source"]),
            "paper_grade": bool(resolved_training_budget["paper_grade"]),
            "diagnostic_only": bool(resolved_training_budget["diagnostic_only"]),
        }
        current_stage = "neural_search_lstm"
        neural_search_results["lstm_ekf"] = _search_best_neural_train_result(
            model_name="lstm_ekf",
            prepare_root=output_root / "prepare" / "sim",
            raw_root=sim_raw_root,
            split_manifest=split_manifest,
            output_root=output_root / "train" / "lstm_ekf",
            estimator_cfgs=estimator_cfgs,
            device=args.device,
            epochs=int(resolved_training_budget["lstm_epochs"]),
            batch_size=int(resolved_training_budget["batch_size"]),
            eval_batch_size=int(resolved_training_budget["eval_batch_size"]),
            baseline_scoring_metrics_by_experiment=baseline_scoring_metrics_by_experiment,
            neural_grid=neural_search_surface["neural_grid"],
            liquid_robustness_profiles=neural_search_surface["liquid_robustness_profiles"],
            top_k_multiseed=int(neural_search_surface["top_k_multiseed"]),
            neural_multiseeds=tuple(int(seed) for seed in neural_search_surface["multiseed_values"]),
            epoch_candidate_stride=neural_search_surface["epoch_candidate_stride"],
            checkpoint_selection_mode=neural_search_surface["checkpoint_selection_mode"],
            resolved_training_budget=resolved_training_budget,
            allow_cache=bool(args.reuse_search_cache),
        )
        neural_search_results["lstm_ekf"]["search_audit"]["resolved_training_budget"] = {
            "epochs": int(resolved_training_budget["lstm_epochs"]),
            "batch_size": int(resolved_training_budget["batch_size"]),
            "eval_batch_size": int(resolved_training_budget["eval_batch_size"]),
            "budget_source": str(resolved_training_budget["budget_source"]),
            "paper_grade": bool(resolved_training_budget["paper_grade"]),
            "diagnostic_only": bool(resolved_training_budget["diagnostic_only"]),
        }
        train_results = {
            model_name: result["train_result"]
            for model_name, result in neural_search_results.items()
        }
        partial_stage_artifacts["train_reports"] = {
            model_name: {
                "train_report_path": str((output_root / "train" / model_name / "reports" / f"{model_name}_train_report.json").resolve()),
                "search_audit_path": str((output_root / "train" / model_name / "search_audit.json").resolve()),
            }
            for model_name in train_results
        }
        partial_stage_artifacts["neural_search"] = {
            model_name: {
                "search_audit_path": str((output_root / "train" / model_name / "search_audit.json").resolve()),
                "selected_signature": result["search_audit"]["selected_signature"],
                "selected_overrides": result["search_audit"]["selected_overrides"],
                "selected_seed": result["search_audit"]["selected_seed"],
                "selected_score_vector": result["search_audit"]["selected_score_vector"],
            }
            for model_name, result in neural_search_results.items()
        }
        completed_stages.append(current_stage)
        model_cfgs = _build_model_cfgs_from_train_reports(train_results)
        partial_stage_artifacts["model_cfgs"] = model_cfgs

        current_stage = "core_experiments"
        core_runs = [
            _run_core_experiment(
                config_name=config_name,
                prepare_root=output_root / "prepare" / "sim",
                raw_root=sim_raw_root,
                estimator_cfgs=estimator_cfgs,
                model_cfgs=model_cfgs,
                output_root=output_root / "core_experiments",
            )
            for config_name in _core_experiment_config_names_for_scope(paper_scope)
        ]
        partial_stage_artifacts["core_runs"] = [
            {
                "experiment_id": run["experiment_id"],
                "core_stage_name": run["core_result"].stage_name,
                "eval_stage_name": run["eval_result"].stage_name,
                "output_root": run["output_root"],
            }
            for run in core_runs
        ]
        completed_stages.append(current_stage)

        current_stage = "public_benchmarks"
        public_runs = [
            _run_public_benchmark(
                dataset_name=dataset_name,
                raw_root=miluv_raw_root if dataset_name == "miluv" else ntu_viral_raw_root,
                estimator_cfgs=estimator_cfgs,
                model_cfgs=model_cfgs,
                output_root=output_root / "public_benchmarks",
            )
            for dataset_name in requested_public_datasets
        ]
        partial_stage_artifacts["public_runs"] = [
            {
                "dataset_name": run["dataset_name"],
                "public_stage_name": run["public_result"].stage_name,
                "eval_stage_name": run["eval_result"].stage_name,
                "output_root": run["output_root"],
            }
            for run in public_runs
        ]
        completed_stages.append(current_stage)

        current_stage = "held_out_core_evaluation"
        held_out_core_evaluation = _run_held_out_core_evaluation(
            split_manifest=split_manifest,
            prepare_root=output_root / "prepare" / "sim",
            raw_root=sim_raw_root,
            estimator_cfgs=estimator_cfgs,
            model_cfgs=model_cfgs,
            output_root=output_root / "held_out_core_experiments",
        )
        partial_stage_artifacts["held_out_core_evaluation"] = held_out_core_evaluation
        partial_stage_artifacts["held_out_core_runs"] = list(held_out_core_evaluation.get("held_out_runs") or [])
        if str(held_out_core_evaluation.get("status") or "") == "blocked":
            return _write_blocked_paper_run_report(
                output_root,
                {
                    "status": "blocked",
                    "stage": current_stage,
                    "completed_stages": list(completed_stages),
                    **partial_stage_artifacts,
                    "reason": held_out_core_evaluation.get("reason") or "held_out_core_evaluation_blocked",
                    "message": held_out_core_evaluation.get("message")
                    or "held-out core evaluation blocked before paper artifact consumers could run",
                    "held_out_test_ids": list(held_out_core_evaluation.get("held_out_test_ids") or []),
                    "split_manifest": held_out_core_evaluation.get("split_manifest", split_manifest),
                    "split_provenance": held_out_core_evaluation.get("split_provenance"),
                    "collection_errors": held_out_core_evaluation.get("collection_errors"),
                    "incomplete_methods": held_out_core_evaluation.get("incomplete_methods"),
                },
            )
        completed_stages.append(current_stage)

        current_stage = "summary_and_figures"
        primary_paper_artifacts = _resolve_primary_paper_artifact_roots(
            core_runs=core_runs,
            held_out_core_evaluation=held_out_core_evaluation,
        )
        partial_stage_artifacts["paper_artifact_source"] = primary_paper_artifacts
        summary_and_figures = _run_summary_and_figures(
            Path(primary_paper_artifacts["eval_root"]),
            output_root,
        )
        partial_stage_artifacts["summary_and_figures"] = summary_and_figures
        completed_stages.append(current_stage)

        current_stage = "audit_and_reporting"
        current_stage = "ekf_init_sensitivity_audit"
        ekf_init_sensitivity = _run_ekf_init_sensitivity_audit(
            split_manifest=split_manifest,
            prepare_root=output_root / "prepare" / "sim",
            raw_root=sim_raw_root,
            estimator_cfgs=estimator_cfgs,
            output_root=output_root / "ekf_init_sensitivity",
        )
        ekf_init_sensitivity_path = _write_json(output_root / "audits" / "ekf_init_sensitivity.json", ekf_init_sensitivity)
        partial_stage_artifacts["ekf_init_sensitivity_audit"] = ekf_init_sensitivity_path
        completed_stages.append(current_stage)
        current_stage = "embedded_runtime_proxy_audit"
        embedded_runtime_proxy = _build_embedded_runtime_proxy_audit(
            Path(primary_paper_artifacts["eval_root"]),
            output_root,
        )
        embedded_runtime_proxy_path = _write_json(
            output_root / "audits" / "embedded_runtime_proxy.json",
            embedded_runtime_proxy,
        )
        partial_stage_artifacts["embedded_runtime_proxy_audit"] = embedded_runtime_proxy_path
        completed_stages.append(current_stage)
        current_stage = "cross_factorial_interactions_audit"
        cross_factorial_interactions = _run_cross_factorial_interaction_audit(
            split_manifest=split_manifest,
            prepare_root=output_root / "prepare" / "sim",
            raw_root=sim_raw_root,
            estimator_cfgs=estimator_cfgs,
            model_cfgs=model_cfgs,
            output_root=output_root / "cross_factorial_interactions",
        )
        cross_factorial_interactions_path = _write_json(
            output_root / "audits" / "cross_factorial_interactions.json",
            cross_factorial_interactions,
        )
        partial_stage_artifacts["cross_factorial_interactions_audit"] = cross_factorial_interactions_path
        completed_stages.append(current_stage)
        current_stage = "consumer_verify"
        consumer_report = _run_high_level_consumer_verification(
            input_root=Path(primary_paper_artifacts["core_root"]),
            ground_truth_root=sim_raw_root,
            output_root=output_root / "consumer_verify",
        )
        partial_stage_artifacts["consumer_verify"] = consumer_report
        partial_stage_artifacts["high_level_consumers"] = consumer_report
        completed_stages.append(current_stage)
        current_stage = "fairness_audit"
        fairness_audit = _build_fairness_audit(
            classical_search_result,
            neural_search_results,
            train_results,
            device=args.device,
            neural_search_surface=neural_search_surface,
        )
        fairness_audit_path = _write_json(output_root / "audits" / "fairness_audit.json", fairness_audit)
        partial_stage_artifacts["fairness_audit"] = fairness_audit_path
        completed_stages.append(current_stage)
        current_stage = "baseline_tuning_audit"
        baseline_tuning_audit = _build_baseline_tuning_audit(classical_search_result)
        baseline_tuning_audit_path = _write_json(output_root / "audits" / "baseline_tuning_audit.json", baseline_tuning_audit)
        partial_stage_artifacts["baseline_tuning_audit"] = baseline_tuning_audit_path
        completed_stages.append(current_stage)
        current_stage = "data_chain_evidence_audit"
        data_chain_evidence_audit = _build_data_chain_evidence_audit(
            split_manifest=split_manifest,
            leak_report=leak_report,
            train_results=train_results,
        )
        data_chain_evidence_audit_path = _write_json(
            output_root / "audits" / "data_chain_evidence_audit.json",
            data_chain_evidence_audit,
        )
        partial_stage_artifacts["data_chain_evidence_audit"] = data_chain_evidence_audit_path
        completed_stages.append(current_stage)
        current_stage = "training_stability_evidence_audit"
        training_stability_evidence_audit = _build_training_stability_evidence_audit(train_results)
        training_stability_evidence_audit_path = _write_json(
            output_root / "audits" / "training_stability_evidence_audit.json",
            training_stability_evidence_audit,
        )
        partial_stage_artifacts["training_stability_evidence_audit"] = training_stability_evidence_audit_path
        completed_stages.append(current_stage)
        current_stage = "training_evidence_audit"
        training_evidence_audit = _build_training_evidence_audit(train_results)
        training_evidence_audit_path = _write_json(
            output_root / "audits" / "training_evidence_audit.json",
            training_evidence_audit,
        )
        partial_stage_artifacts["training_evidence_audit"] = training_evidence_audit_path
        completed_stages.append(current_stage)
        current_stage = "held_out_seq_generalization_audit"
        held_out_seq_generalization = _build_held_out_seq_generalization_audit(
            held_out_core_evaluation=held_out_core_evaluation,
            methods=_resolve_main_conclusion_methods(estimator_cfgs, model_cfgs),
        )
        held_out_seq_generalization_path = _write_json(
            output_root / "audits" / "held_out_seq_generalization.json",
            held_out_seq_generalization,
        )
        partial_stage_artifacts["held_out_seq_generalization_audit"] = held_out_seq_generalization_path
        completed_stages.append(current_stage)
        current_stage = "unseen_seq_generalization_audit"
        unseen_seq_generalization = _build_unseen_seq_generalization_audit(
            core_runs=core_runs,
            held_out_core_evaluation=held_out_core_evaluation,
            split_manifest=split_manifest,
            methods=_resolve_main_conclusion_methods(estimator_cfgs, model_cfgs),
        )
        unseen_seq_generalization_path = _write_json(
            output_root / "audits" / "unseen_seq_generalization.json",
            unseen_seq_generalization,
        )
        partial_stage_artifacts["unseen_seq_generalization_audit"] = unseen_seq_generalization_path
        completed_stages.append(current_stage)
        current_stage = "evaluation_coverage_audit"
        evaluation_coverage_audit = _build_evaluation_coverage_audit(
            core_runs=core_runs,
            held_out_core_evaluation=held_out_core_evaluation,
            cross_factorial_interactions=cross_factorial_interactions,
            requested_public_datasets=requested_public_datasets,
            public_runs=public_runs,
        )
        evaluation_coverage_audit_path = _write_json(
            output_root / "audits" / "evaluation_coverage_audit.json",
            evaluation_coverage_audit,
        )
        partial_stage_artifacts["evaluation_coverage_audit"] = evaluation_coverage_audit_path
        completed_stages.append(current_stage)
        current_stage = "measurement_noise_linkage_audit"
        measurement_noise_linkage_audit = _build_measurement_noise_linkage_audit(
            train_results,
            evaluation_coverage_audit,
        )
        measurement_noise_linkage_audit_path = _write_json(
            output_root / "audits" / "measurement_noise_linkage_audit.json",
            measurement_noise_linkage_audit,
        )
        partial_stage_artifacts["measurement_noise_linkage_audit"] = measurement_noise_linkage_audit_path
        completed_stages.append(current_stage)
        current_stage = "main_conclusion"
        main_conclusion = _build_main_conclusion_report(
            split_manifest=split_manifest,
            held_out_core_evaluation=held_out_core_evaluation,
            cross_factorial_interactions=cross_factorial_interactions,
            public_runs=public_runs,
            requested_public_datasets=requested_public_datasets,
        )
        main_conclusion_path = _write_json(output_root / "audits" / "main_conclusion.json", main_conclusion)
        partial_stage_artifacts["main_conclusion"] = main_conclusion_path
        completed_stages.append(current_stage)
        current_stage = "comparison_readiness_audit"
        comparison_readiness_audit = _build_comparison_readiness_audit(
            fairness_audit=fairness_audit,
            baseline_tuning_audit=baseline_tuning_audit,
            data_chain_evidence_audit=data_chain_evidence_audit,
            training_stability_evidence_audit=training_stability_evidence_audit,
            training_evidence_audit=training_evidence_audit,
            measurement_noise_linkage_audit=measurement_noise_linkage_audit,
            evaluation_coverage_audit=evaluation_coverage_audit,
            held_out_core_evaluation=held_out_core_evaluation,
            main_conclusion=main_conclusion,
        )
        comparison_readiness_audit_path = _write_json(
            output_root / "audits" / "comparison_readiness_audit.json",
            comparison_readiness_audit,
        )
        partial_stage_artifacts["comparison_readiness_audit"] = comparison_readiness_audit_path
        completed_stages.append(current_stage)
        current_stage = "comparison_readiness_summary"
        baseline_tuning_readiness = _build_baseline_tuning_readiness_summary(baseline_tuning_audit)
        partial_stage_artifacts["baseline_tuning_readiness"] = baseline_tuning_readiness
        data_chain_readiness = _build_data_chain_readiness_summary(data_chain_evidence_audit)
        partial_stage_artifacts["data_chain_readiness"] = data_chain_readiness
        training_stability_readiness = _build_training_stability_readiness_summary(training_stability_evidence_audit)
        partial_stage_artifacts["training_stability_readiness"] = training_stability_readiness
        training_evidence_readiness = _build_training_evidence_readiness_summary(training_evidence_audit)
        partial_stage_artifacts["training_evidence_readiness"] = training_evidence_readiness
        comparison_readiness = _build_comparison_readiness_summary(comparison_readiness_audit)
        partial_stage_artifacts["comparison_readiness"] = comparison_readiness
        completed_stages.append(current_stage)
        current_stage = "run_grade_contract"
        run_grade_contract = _build_run_grade_contract(
            requested_public_datasets=requested_public_datasets,
            allow_incomplete_paper_run=bool(args.allow_incomplete_paper_run),
            neural_search_surface=neural_search_surface,
            comparison_readiness=comparison_readiness,
        )
        partial_stage_artifacts["run_grade_contract"] = run_grade_contract
        completed_stages.append(current_stage)
        current_stage = "audit_and_reporting"
        final_status = (
            "ok"
            if (
                main_conclusion["status"] == "ok"
                and str(run_grade_contract.get("requested_execution_grade") or "") == "paper_grade"
                and bool(run_grade_contract.get("ready_for_full_fair_comparison_claim"))
            )
            else "degraded"
        )
        partial_stage_artifacts["final_status"] = final_status
        final_completed_stages = [*completed_stages, current_stage]

        final_report = {
            "status": final_status,
            "completed_stages": final_completed_stages,
            **paper_run_runtime_context,
            "sim_prepare_artifacts": sim_prepare.artifacts,
            "miluv_prepare_artifacts": (
                public_prepare_results["miluv"].artifacts if "miluv" in public_prepare_results else None
            ),
            "ntu_viral_prepare_artifacts": (
                public_prepare_results["ntu_viral"].artifacts if "ntu_viral" in public_prepare_results else None
            ),
            "split_manifest": split_manifest,
            "leak_report": leak_report,
            "classical_search": {
                "search_audit_path": str((output_root / "classical_search" / "search_audit.json").resolve()),
                "selected_estimator_cfgs": estimator_cfgs,
            },
            "train_reports": {
                model_name: {
                    "train_report_path": str((output_root / "train" / model_name / "reports" / f"{model_name}_train_report.json").resolve()),
                    "search_audit_path": str((output_root / "train" / model_name / "search_audit.json").resolve()),
                }
                for model_name in train_results
            },
            "neural_search": {
                model_name: {
                    "search_audit_path": str((output_root / "train" / model_name / "search_audit.json").resolve()),
                    "selected_signature": result["search_audit"]["selected_signature"],
                    "selected_overrides": result["search_audit"]["selected_overrides"],
                    "selected_seed": result["search_audit"]["selected_seed"],
                    "selected_score_vector": result["search_audit"]["selected_score_vector"],
                }
                for model_name, result in neural_search_results.items()
            },
            "paper_artifact_source": primary_paper_artifacts,
            "fairness_audit": fairness_audit_path,
            "baseline_tuning_audit": baseline_tuning_audit_path,
            "baseline_tuning_readiness": baseline_tuning_readiness,
            "data_chain_evidence_audit": data_chain_evidence_audit_path,
            "data_chain_readiness": data_chain_readiness,
            "training_stability_evidence_audit": training_stability_evidence_audit_path,
            "training_stability_readiness": training_stability_readiness,
            "training_evidence_audit": training_evidence_audit_path,
            "training_evidence_readiness": training_evidence_readiness,
            "measurement_noise_linkage_audit": measurement_noise_linkage_audit_path,
            "held_out_core_evaluation": held_out_core_evaluation,
            "core_runs": [
                {
                    "experiment_id": run["experiment_id"],
                    "core_stage_name": run["core_result"].stage_name,
                    "eval_stage_name": run["eval_result"].stage_name,
                    "output_root": run["output_root"],
                }
                for run in core_runs
            ],
            "held_out_core_runs": list(held_out_core_evaluation.get("held_out_runs") or []),
            "public_runs": [
                {
                    "dataset_name": run["dataset_name"],
                    "public_stage_name": run["public_result"].stage_name,
                    "eval_stage_name": run["eval_result"].stage_name,
                    "output_root": run["output_root"],
                }
                for run in public_runs
            ],
            "summary_and_figures": summary_and_figures,
            "ekf_init_sensitivity_audit": ekf_init_sensitivity_path,
            "embedded_runtime_proxy_audit": embedded_runtime_proxy_path,
            "cross_factorial_interactions_audit": cross_factorial_interactions_path,
            "held_out_seq_generalization_audit": held_out_seq_generalization_path,
            "unseen_seq_generalization_audit": unseen_seq_generalization_path,
            "evaluation_coverage_audit": evaluation_coverage_audit_path,
            "main_conclusion": main_conclusion_path,
            "comparison_readiness_audit": comparison_readiness_audit_path,
            "comparison_readiness": comparison_readiness,
            "run_grade_contract": run_grade_contract,
            "consumer_verify": consumer_report,
            "high_level_consumers": consumer_report,
        }
        _write_final_report(output_root, final_report)
        completed_stages = final_completed_stages
        try:
            from liquidloc.common.io_utils import dumps_json_text
            print(dumps_json_text(final_report))
        except Exception:
            
            pass
        return 0
    except Exception as exc:
        if type(exc).__name__ == "_HighLevelConsumerVerificationError" and hasattr(exc, "payload"):
            structured_payload = dict(getattr(exc, "payload"))
            child_stage = str(structured_payload.get("stage") or "unknown")
            child_blocker = str(structured_payload.get("blocker") or "consumer verification failed")
            child_detail = str(structured_payload.get("detail") or "")
            failure_report = {
                "status": "failed",
                "stage": current_stage,
                "reason": "consumer_verify_failed",
                "message": f"{child_stage}: {child_blocker}" if child_stage else child_blocker,
                "completed_stages": list(completed_stages),
                **partial_stage_artifacts,
                "consumer_verify_stage": child_stage,
                "consumer_verify_blocker": child_blocker,
                "consumer_verify_detail": child_detail,
                "consumer_verify_exit_code": int(structured_payload.get("exit_code", 1)),
                "consumer_verify": structured_payload,
                "high_level_consumers": structured_payload,
            }
            return _write_failed_paper_run_report(output_root, failure_report)
        if "comparison_readiness_audit_path" in locals():
            partial_stage_artifacts["comparison_readiness_audit"] = comparison_readiness_audit_path
        if "comparison_readiness" in locals():
            partial_stage_artifacts["comparison_readiness"] = comparison_readiness
        if "run_grade_contract" in locals():
            partial_stage_artifacts["run_grade_contract"] = run_grade_contract
        failure_report = {
            "status": "failed",
            "stage": current_stage,
            "reason": "unhandled_exception",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "completed_stages": list(completed_stages),
            **partial_stage_artifacts,
        }
        return _write_failed_paper_run_report(output_root, failure_report)




_CORE_EXPERIMENT_CONFIGS = tuple(
    config_name
    for config_name in _CORE_EXPERIMENT_CONFIGS
    if config_name not in {"e10_modality_recovery.yaml", "e11_phase_switch.yaml"}
)


if __name__ == "__main__":  
    raise SystemExit(main())  

