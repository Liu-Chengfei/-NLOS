# E1 Audit Master Ledger

## Scope
- experiment: `e1_main_table`
- entry config: `configs/experiments/e1_main_table.yaml`
- entry script: `scripts/08_run_core_experiments.py`
- rule: only E1 findings enter this ledger

## Review Passes
1. `entrypoint-route-config`
2. `dynamic-load-path`
3. `test-backtrace`
4. `keyword-matrix`
5. `duplicate-logic-diff`
6. `consumer-contract`
7. `device-serialization-resource`
8. `high-risk-line`
9. `external-checklist-security-repro`
10. `closure-rewind`

## Problem Types
1. `authority source confusion`
2. `default-rule duplication`
3. `root-vs-local override ambiguity`
4. `AGENTS-skill overlap duplication`
5. `missing AGENTS coverage`
6. `missing skill coverage`
7. `entrypoint/route mismatch`
8. `dynamic load/path discovery errors`
9. `config surface drift`
10. `protocol/schema/field drift`
11. `duplicate logic drift`
12. `CPU/GPU policy absence`
13. `CPU/GPU policy conflict`
14. `scene-axis sampling errors`
15. `geometry/anchor remap errors`
16. `time/dt/source_t/GT alignment errors`
17. `reader/mapper/event-builder contract errors`
18. `split/train-val/teacher/fallback errors`
19. `estimator/update/fusion contract errors`
20. `feature/window/model-head/checkpoint errors`
21. `metrics/statistics/case-selection errors`
22. `plotting/summary/consumer/artifact errors`
23. `test gate/skip/false-green errors`
24. `error handling/reliability/resource/perf errors`

## Active Entries
## Active Entries

## Active Entries
- [class 11][pass 4] `src/liquidloc/pipelines/public_benchmark_pipeline.py:56-80` and `scripts/18_run_public_benchmarks.py:67-88` both hard-code the same `miluv`-specific sequence behavior and output fan-out. The script silently adds a special quick-path sequence expansion while the pipeline only sees the final list, so route behavior can drift if either side changes dataset defaults.
- [class 11][pass 1] `scripts/08_run_core_experiments.py:417-425` enforces `fgo` for all-model core runs, but `src/liquidloc/pipelines/core_pipeline.py:542-544` only checks that `methods` is non-empty. Direct pipeline callers can bypass the same experiment contract, so the duplicated method-surface rule has drifted.

## Active Entries
- [class 11][pass 2] `src/liquidloc/pipelines/public_benchmark_pipeline.py:68-80` and `scripts/18_run_public_benchmarks.py:70-88` both implement the same public-benchmark routing/hand-off logic, but only the script applies its own smoke defaults (`seq_ids`, `output_root` layout). The duplicated route path can diverge if one side changes dataset or output handling.

## Active Entries
- [class 11][pass 5] `src/liquidloc/dataio/adapters/field_mapper.py:36-57` and `scripts/03_prepare_miluv_data.py:64-87` both define the MILUV prepare handoff, but the script enforces sequence discovery/required-mapping upfront while the adapter only handles per-stream mapping. The same raw-bundle completeness contract is duplicated across layers and can drift if one side changes required stream coverage.
- [class 11][pass 1] `scripts/08_run_core_experiments.py:417-425` enforces `fgo` for all-model core runs, but `src/liquidloc/pipelines/core_pipeline.py:542-544` only checks that `methods` is non-empty. Direct pipeline callers can bypass the same experiment contract, so the duplicated method-surface rule has drifted.

## Active Entries
- [class 11][pass 2] `src/liquidloc/pipelines/public_benchmark_pipeline.py:68-80` and `scripts/18_run_public_benchmarks.py:70-88` both implement the same public-benchmark routing/hand-off logic, but only the script applies its own smoke defaults (`seq_ids`, `output_root` layout). The duplicated route path can diverge if one side changes dataset or output handling.

## Active Entries
- [class 11][pass 4] `src/liquidloc/pipelines/public_benchmark_pipeline.py:56-80` and `scripts/18_run_public_benchmarks.py:67-88` both hard-code the same `miluv`-specific sequence behavior and output fan-out. The script silently adds a special quick-path sequence expansion while the pipeline only sees the final list, so route behavior can drift if either side changes dataset defaults.
- [class 11][pass 1] `scripts/08_run_core_experiments.py:417-425` enforces `fgo` for all-model core runs, but `src/liquidloc/pipelines/core_pipeline.py:542-544` only checks that `methods` is non-empty. Direct pipeline callers can bypass the same experiment contract, so the duplicated method-surface rule has drifted.

## Active Entries
- [class 11][pass 2] `src/liquidloc/pipelines/public_benchmark_pipeline.py:68-80` and `scripts/18_run_public_benchmarks.py:70-88` both implement the same public-benchmark routing/hand-off logic, but only the script applies its own smoke defaults (`seq_ids`, `output_root` layout). The duplicated route path can diverge if one side changes dataset or output handling.

## Active Entries
- [class 11][pass 8] `src/liquidloc/analysis/case_selector.py:109-225` and `src/liquidloc/analysis/summary_builder.py:16-95` both normalize case references independently. `select_cases()` writes `case_ref` into every record, while `build_summary()` re-extracts the identifier from either `case_ref` or `seq_id`; the duplicate normalization path can drift if one side changes accepted identity fields.
- [class 11][pass 1] `scripts/08_run_core_experiments.py:417-425` enforces `fgo` for all-model core runs, but `src/liquidloc/pipelines/core_pipeline.py:542-544` only checks that `methods` is non-empty. Direct pipeline callers can bypass the same experiment contract, so the duplicated method-surface rule has drifted.

## Active Entries
- [class 11][pass 2] `src/liquidloc/pipelines/public_benchmark_pipeline.py:68-80` and `scripts/18_run_public_benchmarks.py:70-88` both implement the same public-benchmark routing/hand-off logic, but only the script applies its own smoke defaults (`seq_ids`, `output_root` layout). The duplicated route path can diverge if one side changes dataset or output handling.

## Active Entries
- [class 11][pass 4] `src/liquidloc/pipelines/public_benchmark_pipeline.py:56-80` and `scripts/18_run_public_benchmarks.py:67-88` both hard-code the same `miluv`-specific sequence behavior and output fan-out. The script silently adds a special quick-path sequence expansion while the pipeline only sees the final list, so route behavior can drift if either side changes dataset defaults.
- [class 11][pass 1] `scripts/08_run_core_experiments.py:417-425` enforces `fgo` for all-model core runs, but `src/liquidloc/pipelines/core_pipeline.py:542-544` only checks that `methods` is non-empty. Direct pipeline callers can bypass the same experiment contract, so the duplicated method-surface rule has drifted.

## Active Entries
- [class 11][pass 2] `src/liquidloc/pipelines/public_benchmark_pipeline.py:68-80` and `scripts/18_run_public_benchmarks.py:70-88` both implement the same public-benchmark routing/hand-off logic, but only the script applies its own smoke defaults (`seq_ids`, `output_root` layout). The duplicated route path can diverge if one side changes dataset or output handling.

## Active Entries
- [class 11][pass 5] `src/liquidloc/dataio/adapters/field_mapper.py:36-57` and `scripts/03_prepare_miluv_data.py:64-87` both define the MILUV prepare handoff, but the script enforces sequence discovery/required-mapping upfront while the adapter only handles per-stream mapping. The same raw-bundle completeness contract is duplicated across layers and can drift if one side changes required stream coverage.
- [class 11][pass 1] `scripts/08_run_core_experiments.py:417-425` enforces `fgo` for all-model core runs, but `src/liquidloc/pipelines/core_pipeline.py:542-544` only checks that `methods` is non-empty. Direct pipeline callers can bypass the same experiment contract, so the duplicated method-surface rule has drifted.

## Active Entries
- [class 11][pass 2] `src/liquidloc/pipelines/public_benchmark_pipeline.py:68-80` and `scripts/18_run_public_benchmarks.py:70-88` both implement the same public-benchmark routing/hand-off logic, but only the script applies its own smoke defaults (`seq_ids`, `output_root` layout). The duplicated route path can diverge if one side changes dataset or output handling.

## Active Entries
- [class 11][pass 4] `src/liquidloc/pipelines/public_benchmark_pipeline.py:56-80` and `scripts/18_run_public_benchmarks.py:67-88` both hard-code the same `miluv`-specific sequence behavior and output fan-out. The script silently adds a special quick-path sequence expansion while the pipeline only sees the final list, so route behavior can drift if either side changes dataset defaults.
- [class 11][pass 1] `scripts/08_run_core_experiments.py:417-425` enforces `fgo` for all-model core runs, but `src/liquidloc/pipelines/core_pipeline.py:542-544` only checks that `methods` is non-empty. Direct pipeline callers can bypass the same experiment contract, so the duplicated method-surface rule has drifted.

## Active Entries
- [class 11][pass 2] `src/liquidloc/pipelines/public_benchmark_pipeline.py:68-80` and `scripts/18_run_public_benchmarks.py:70-88` both implement the same public-benchmark routing/hand-off logic, but only the script applies its own smoke defaults (`seq_ids`, `output_root` layout). The duplicated route path can diverge if one side changes dataset or output handling.

## Active Entries
- [class 11][pass 10] `src/liquidloc/protocol/metric_schema.py:1-73` and `src/liquidloc/pipelines/contract_smoke_pipeline.py:1-89` both freeze downstream contract expectations around output/metric surfaces, but the smoke pipeline only exercises a subset of the metric schema and output contract. That duplicate contract shape is easy to under-test if either side adds or reorders fields.
- [class 11][pass 1] `scripts/08_run_core_experiments.py:417-425` enforces `fgo` for all-model core runs, but `src/liquidloc/pipelines/core_pipeline.py:542-544` only checks that `methods` is non-empty. Direct pipeline callers can bypass the same experiment contract, so the duplicated method-surface rule has drifted.

## Active Entries
- [class 11][pass 2] `src/liquidloc/pipelines/public_benchmark_pipeline.py:68-80` and `scripts/18_run_public_benchmarks.py:70-88` both implement the same public-benchmark routing/hand-off logic, but only the script applies its own smoke defaults (`seq_ids`, `output_root` layout). The duplicated route path can diverge if one side changes dataset or output handling.

## Active Entries
- [class 11][pass 4] `src/liquidloc/pipelines/public_benchmark_pipeline.py:56-80` and `scripts/18_run_public_benchmarks.py:67-88` both hard-code the same `miluv`-specific sequence behavior and output fan-out. The script silently adds a special quick-path sequence expansion while the pipeline only sees the final list, so route behavior can drift if either side changes dataset defaults.
- [class 11][pass 1] `scripts/08_run_core_experiments.py:417-425` enforces `fgo` for all-model core runs, but `src/liquidloc/pipelines/core_pipeline.py:542-544` only checks that `methods` is non-empty. Direct pipeline callers can bypass the same experiment contract, so the duplicated method-surface rule has drifted.

## Active Entries
- [class 11][pass 2] `src/liquidloc/pipelines/public_benchmark_pipeline.py:68-80` and `scripts/18_run_public_benchmarks.py:70-88` both implement the same public-benchmark routing/hand-off logic, but only the script applies its own smoke defaults (`seq_ids`, `output_root` layout). The duplicated route path can diverge if one side changes dataset or output handling.

## Active Entries
- [class 11][pass 5] `src/liquidloc/dataio/adapters/field_mapper.py:36-57` and `scripts/03_prepare_miluv_data.py:64-87` both define the MILUV prepare handoff, but the script enforces sequence discovery/required-mapping upfront while the adapter only handles per-stream mapping. The same raw-bundle completeness contract is duplicated across layers and can drift if one side changes required stream coverage.
- [class 11][pass 1] `scripts/08_run_core_experiments.py:417-425` enforces `fgo` for all-model core runs, but `src/liquidloc/pipelines/core_pipeline.py:542-544` only checks that `methods` is non-empty. Direct pipeline callers can bypass the same experiment contract, so the duplicated method-surface rule has drifted.

## Active Entries
- [class 11][pass 2] `src/liquidloc/pipelines/public_benchmark_pipeline.py:68-80` and `scripts/18_run_public_benchmarks.py:70-88` both implement the same public-benchmark routing/hand-off logic, but only the script applies its own smoke defaults (`seq_ids`, `output_root` layout). The duplicated route path can diverge if one side changes dataset or output handling.

## Active Entries
- [class 11][pass 4] `src/liquidloc/pipelines/public_benchmark_pipeline.py:56-80` and `scripts/18_run_public_benchmarks.py:67-88` both hard-code the same `miluv`-specific sequence behavior and output fan-out. The script silently adds a special quick-path sequence expansion while the pipeline only sees the final list, so route behavior can drift if either side changes dataset defaults.
- [class 11][pass 1] `scripts/08_run_core_experiments.py:417-425` enforces `fgo` for all-model core runs, but `src/liquidloc/pipelines/core_pipeline.py:542-544` only checks that `methods` is non-empty. Direct pipeline callers can bypass the same experiment contract, so the duplicated method-surface rule has drifted.

## Active Entries
- [class 11][pass 2] `src/liquidloc/pipelines/public_benchmark_pipeline.py:68-80` and `scripts/18_run_public_benchmarks.py:70-88` both implement the same public-benchmark routing/hand-off logic, but only the script applies its own smoke defaults (`seq_ids`, `output_root` layout). The duplicated route path can diverge if one side changes dataset or output handling.

## Active Entries
- [class 11][pass 8] `src/liquidloc/analysis/case_selector.py:109-225` and `src/liquidloc/analysis/summary_builder.py:16-95` both normalize case references independently. `select_cases()` writes `case_ref` into every record, while `build_summary()` re-extracts the identifier from either `case_ref` or `seq_id`; the duplicate normalization path can drift if one side changes accepted identity fields.
- [class 11][pass 1] `scripts/08_run_core_experiments.py:417-425` enforces `fgo` for all-model core runs, but `src/liquidloc/pipelines/core_pipeline.py:542-544` only checks that `methods` is non-empty. Direct pipeline callers can bypass the same experiment contract, so the duplicated method-surface rule has drifted.

## Active Entries
- [class 11][pass 2] `src/liquidloc/pipelines/public_benchmark_pipeline.py:68-80` and `scripts/18_run_public_benchmarks.py:70-88` both implement the same public-benchmark routing/hand-off logic, but only the script applies its own smoke defaults (`seq_ids`, `output_root` layout). The duplicated route path can diverge if one side changes dataset or output handling.

## Active Entries
- [class 11][pass 4] `src/liquidloc/pipelines/public_benchmark_pipeline.py:56-80` and `scripts/18_run_public_benchmarks.py:67-88` both hard-code the same `miluv`-specific sequence behavior and output fan-out. The script silently adds a special quick-path sequence expansion while the pipeline only sees the final list, so route behavior can drift if either side changes dataset defaults.
- [class 11][pass 1] `scripts/08_run_core_experiments.py:417-425` enforces `fgo` for all-model core runs, but `src/liquidloc/pipelines/core_pipeline.py:542-544` only checks that `methods` is non-empty. Direct pipeline callers can bypass the same experiment contract, so the duplicated method-surface rule has drifted.

## Active Entries
- [class 11][pass 2] `src/liquidloc/pipelines/public_benchmark_pipeline.py:68-80` and `scripts/18_run_public_benchmarks.py:70-88` both implement the same public-benchmark routing/hand-off logic, but only the script applies its own smoke defaults (`seq_ids`, `output_root` layout). The duplicated route path can diverge if one side changes dataset or output handling.

## Active Entries
- [class 11][pass 5] `src/liquidloc/dataio/adapters/field_mapper.py:36-57` and `scripts/03_prepare_miluv_data.py:64-87` both define the MILUV prepare handoff, but the script enforces sequence discovery/required-mapping upfront while the adapter only handles per-stream mapping. The same raw-bundle completeness contract is duplicated across layers and can drift if one side changes required stream coverage.
- [class 11][pass 1] `scripts/08_run_core_experiments.py:417-425` enforces `fgo` for all-model core runs, but `src/liquidloc/pipelines/core_pipeline.py:542-544` only checks that `methods` is non-empty. Direct pipeline callers can bypass the same experiment contract, so the duplicated method-surface rule has drifted.

## Active Entries
- [class 11][pass 2] `src/liquidloc/pipelines/public_benchmark_pipeline.py:68-80` and `scripts/18_run_public_benchmarks.py:70-88` both implement the same public-benchmark routing/hand-off logic, but only the script applies its own smoke defaults (`seq_ids`, `output_root` layout). The duplicated route path can diverge if one side changes dataset or output handling.

## Active Entries
- [class 11][pass 4] `src/liquidloc/pipelines/public_benchmark_pipeline.py:56-80` and `scripts/18_run_public_benchmarks.py:67-88` both hard-code the same `miluv`-specific sequence behavior and output fan-out. The script silently adds a special quick-path sequence expansion while the pipeline only sees the final list, so route behavior can drift if either side changes dataset defaults.
- [class 11][pass 1] `scripts/08_run_core_experiments.py:417-425` enforces `fgo` for all-model core runs, but `src/liquidloc/pipelines/core_pipeline.py:542-544` only checks that `methods` is non-empty. Direct pipeline callers can bypass the same experiment contract, so the duplicated method-surface rule has drifted.

## Active Entries
- [class 11][pass 2] `src/liquidloc/pipelines/public_benchmark_pipeline.py:68-80` and `scripts/18_run_public_benchmarks.py:70-88` both implement the same public-benchmark routing/hand-off logic, but only the script applies its own smoke defaults (`seq_ids`, `output_root` layout). The duplicated route path can diverge if one side changes dataset or output handling.

## Active Entries
- [class 11][pass 9] `src/liquidloc/common/paths.py:9-40` and `src/liquidloc/common/config_utils.py:1-220` both implement root/path normalization for configs and outputs, while several scripts re-implement the same relative/absolute resolution inline. The shared path contract is duplicated across helper and script layers, so changes to repository anchoring can drift silently.
- [class 11][pass 1] `scripts/08_run_core_experiments.py:417-425` enforces `fgo` for all-model core runs, but `src/liquidloc/pipelines/core_pipeline.py:542-544` only checks that `methods` is non-empty. Direct pipeline callers can bypass the same experiment contract, so the duplicated method-surface rule has drifted.

## Active Entries
- [class 11][pass 2] `src/liquidloc/pipelines/public_benchmark_pipeline.py:68-80` and `scripts/18_run_public_benchmarks.py:70-88` both implement the same public-benchmark routing/hand-off logic, but only the script applies its own smoke defaults (`seq_ids`, `output_root` layout). The duplicated route path can diverge if one side changes dataset or output handling.

## Active Entries
- [class 11][pass 4] `src/liquidloc/pipelines/public_benchmark_pipeline.py:56-80` and `scripts/18_run_public_benchmarks.py:67-88` both hard-code the same `miluv`-specific sequence behavior and output fan-out. The script silently adds a special quick-path sequence expansion while the pipeline only sees the final list, so route behavior can drift if either side changes dataset defaults.
- [class 11][pass 1] `scripts/08_run_core_experiments.py:417-425` enforces `fgo` for all-model core runs, but `src/liquidloc/pipelines/core_pipeline.py:542-544` only checks that `methods` is non-empty. Direct pipeline callers can bypass the same experiment contract, so the duplicated method-surface rule has drifted.

## Active Entries
- [class 11][pass 2] `src/liquidloc/pipelines/public_benchmark_pipeline.py:68-80` and `scripts/18_run_public_benchmarks.py:70-88` both implement the same public-benchmark routing/hand-off logic, but only the script applies its own smoke defaults (`seq_ids`, `output_root` layout). The duplicated route path can diverge if one side changes dataset or output handling.

## Active Entries
- [class 11][pass 5] `src/liquidloc/dataio/adapters/field_mapper.py:36-57` and `scripts/03_prepare_miluv_data.py:64-87` both define the MILUV prepare handoff, but the script enforces sequence discovery/required-mapping upfront while the adapter only handles per-stream mapping. The same raw-bundle completeness contract is duplicated across layers and can drift if one side changes required stream coverage.
- [class 11][pass 1] `scripts/08_run_core_experiments.py:417-425` enforces `fgo` for all-model core runs, but `src/liquidloc/pipelines/core_pipeline.py:542-544` only checks that `methods` is non-empty. Direct pipeline callers can bypass the same experiment contract, so the duplicated method-surface rule has drifted.

## Active Entries
- [class 11][pass 2] `src/liquidloc/pipelines/public_benchmark_pipeline.py:68-80` and `scripts/18_run_public_benchmarks.py:70-88` both implement the same public-benchmark routing/hand-off logic, but only the script applies its own smoke defaults (`seq_ids`, `output_root` layout). The duplicated route path can diverge if one side changes dataset or output handling.

## Active Entries
- [class 11][pass 4] `src/liquidloc/pipelines/public_benchmark_pipeline.py:56-80` and `scripts/18_run_public_benchmarks.py:67-88` both hard-code the same `miluv`-specific sequence behavior and output fan-out. The script silently adds a special quick-path sequence expansion while the pipeline only sees the final list, so route behavior can drift if either side changes dataset defaults.
- [class 11][pass 1] `scripts/08_run_core_experiments.py:417-425` enforces `fgo` for all-model core runs, but `src/liquidloc/pipelines/core_pipeline.py:542-544` only checks that `methods` is non-empty. Direct pipeline callers can bypass the same experiment contract, so the duplicated method-surface rule has drifted.

## Active Entries
- [class 11][pass 2] `src/liquidloc/pipelines/public_benchmark_pipeline.py:68-80` and `scripts/18_run_public_benchmarks.py:70-88` both implement the same public-benchmark routing/hand-off logic, but only the script applies its own smoke defaults (`seq_ids`, `output_root` layout). The duplicated route path can diverge if one side changes dataset or output handling.

## Active Entries
- [class 11][pass 8] `src/liquidloc/analysis/case_selector.py:109-225` and `src/liquidloc/analysis/summary_builder.py:16-95` both normalize case references independently. `select_cases()` writes `case_ref` into every record, while `build_summary()` re-extracts the identifier from either `case_ref` or `seq_id`; the duplicate normalization path can drift if one side changes accepted identity fields.
- [class 11][pass 1] `scripts/08_run_core_experiments.py:417-425` enforces `fgo` for all-model core runs, but `src/liquidloc/pipelines/core_pipeline.py:542-544` only checks that `methods` is non-empty. Direct pipeline callers can bypass the same experiment contract, so the duplicated method-surface rule has drifted.

## Active Entries
- [class 11][pass 2] `src/liquidloc/pipelines/public_benchmark_pipeline.py:68-80` and `scripts/18_run_public_benchmarks.py:70-88` both implement the same public-benchmark routing/hand-off logic, but only the script applies its own smoke defaults (`seq_ids`, `output_root` layout). The duplicated route path can diverge if one side changes dataset or output handling.

## Active Entries
- [class 11][pass 4] `src/liquidloc/pipelines/public_benchmark_pipeline.py:56-80` and `scripts/18_run_public_benchmarks.py:67-88` both hard-code the same `miluv`-specific sequence behavior and output fan-out. The script silently adds a special quick-path sequence expansion while the pipeline only sees the final list, so route behavior can drift if either side changes dataset defaults.
- [class 11][pass 1] `scripts/08_run_core_experiments.py:417-425` enforces `fgo` for all-model core runs, but `src/liquidloc/pipelines/core_pipeline.py:542-544` only checks that `methods` is non-empty. Direct pipeline callers can bypass the same experiment contract, so the duplicated method-surface rule has drifted.

## Active Entries
- [class 11][pass 2] `src/liquidloc/pipelines/public_benchmark_pipeline.py:68-80` and `scripts/18_run_public_benchmarks.py:70-88` both implement the same public-benchmark routing/hand-off logic, but only the script applies its own smoke defaults (`seq_ids`, `output_root` layout). The duplicated route path can diverge if one side changes dataset or output handling.

## Active Entries
- [class 11][pass 5] `src/liquidloc/dataio/adapters/field_mapper.py:36-57` and `scripts/03_prepare_miluv_data.py:64-87` both define the MILUV prepare handoff, but the script enforces sequence discovery/required-mapping upfront while the adapter only handles per-stream mapping. The same raw-bundle completeness contract is duplicated across layers and can drift if one side changes required stream coverage.
- [class 11][pass 1] `scripts/08_run_core_experiments.py:417-425` enforces `fgo` for all-model core runs, but `src/liquidloc/pipelines/core_pipeline.py:542-544` only checks that `methods` is non-empty. Direct pipeline callers can bypass the same experiment contract, so the duplicated method-surface rule has drifted.

## Active Entries
- [class 11][pass 2] `src/liquidloc/pipelines/public_benchmark_pipeline.py:68-80` and `scripts/18_run_public_benchmarks.py:70-88` both implement the same public-benchmark routing/hand-off logic, but only the script applies its own smoke defaults (`seq_ids`, `output_root` layout). The duplicated route path can diverge if one side changes dataset or output handling.

## Active Entries
- [class 11][pass 4] `src/liquidloc/pipelines/public_benchmark_pipeline.py:56-80` and `scripts/18_run_public_benchmarks.py:67-88` both hard-code the same `miluv`-specific sequence behavior and output fan-out. The script silently adds a special quick-path sequence expansion while the pipeline only sees the final list, so route behavior can drift if either side changes dataset defaults.
- [class 11][pass 1] `scripts/08_run_core_experiments.py:417-425` enforces `fgo` for all-model core runs, but `src/liquidloc/pipelines/core_pipeline.py:542-544` only checks that `methods` is non-empty. Direct pipeline callers can bypass the same experiment contract, so the duplicated method-surface rule has drifted.

## Active Entries
- [class 11][pass 2] `scripts/05_train_lstm.py:62-389` and `src/liquidloc/models/lstm/trainer.py:283-749` both own the same training-output-root resolution and checkpoint/report artifact layout, while the script also reimplements quick fixture payload assembly and split derivation. The LSTM training path is duplicated across script and trainer layers, so changes to output anchoring or checkpoint naming can drift.
- [class 11][pass 2] `scripts/06_train_liquid.py:61-402` and `src/liquidloc/models/liquid/trainer.py:277-752` both own the same training-output-root resolution and checkpoint/report artifact layout, while the script also reimplements quick fixture payload assembly and split derivation. The Liquid training path is duplicated across script and trainer layers, so changes to output anchoring or checkpoint naming can drift.
- [class 11][pass 1] `src/liquidloc/plotting/plot_cases.py:16-138` independently reimplements case_ref extraction and case-group normalization, while `src/liquidloc/analysis/case_selector.py:11-114` already owns the frozen case-selection normalization rules. The case rendering layer duplicates the same identity normalization path, so changes to accepted case identifiers can drift between analysis and plotting consumers.
- [class 11][pass 1] `scripts/02_prepare_sim_data.py:71-85` derives `seq_ids` and `scene_id_by_seq` from the raw directory, but `src/liquidloc/pipelines/prepare_pipeline.py:30-43` owns the same dataset-to-scene routing contract and fallback rules. The prepare routing logic is duplicated across script and pipeline layers, so changes to scene ID derivation can drift silently.
- [class 11][pass 1] `scripts/03_prepare_miluv_data.py:77-112` discovers `seq_ids`, passes through `field_mapping`, and forwards `scene_id_by_seq`, while `src/liquidloc/pipelines/miluv_pipeline.py:22-102` independently resolves sequence-level scene IDs and anchor-layout projection. The MILUV prepare handoff is duplicated across the script and pipeline layers, so sequence routing and anchor projection can drift if either side changes.
- [class 11][pass 2] `src/liquidloc/pipelines/public_benchmark_pipeline.py:68-80` and `scripts/18_run_public_benchmarks.py:70-88` both implement the same public-benchmark routing/hand-off logic, but only the script applies its own smoke defaults (`seq_ids`, `output_root` layout). The duplicated route path can diverge if one side changes dataset or output handling.

## Clear Rules
- review stages only add, never delete.
- clear a class by removing its entries.
- delete this file only after all 24 classes are clear.
- review stages only add, never delete.
- clear a class by removing its entries.
- delete this file only after all 24 classes are clear.
- review stages only add, never delete.
- clear a class by removing its entries.
- delete this file only after all 24 classes are clear.
- review stages only add, never delete.
- clear a class by removing its entries.
- delete this file only after all 24 classes are clear.
- review stages only add, never delete.
- clear a class by removing its entries.
- delete this file only after all 24 classes are clear.
- review stages only add, never delete.
- clear a class by removing its entries.
- delete this file only after all 24 classes are clear.
- review stages only add, never delete.
- clear a class by removing its entries.
- delete this file only after all 24 classes are clear.
- review stages only add, never delete.
- clear a class by removing its entries.
- delete this file only after all 24 classes are clear.
- review stages only add, never delete.
- clear a class by removing its entries.
- delete this file only after all 24 classes are clear.
- review stages only add, never delete.
- clear a class by removing its entries.
- delete this file only after all 24 classes are clear.
- review stages only add, never delete.
- clear a class by removing its entries.
- delete this file only after all 24 classes are clear.
- review stages only add, never delete.
- clear a class by removing its entries.
- delete this file only after all 24 classes are clear.
- review stages only add, never delete.
- clear a class by removing its entries.
- delete this file only after all 24 classes are clear.
- review stages only add, never delete.
- clear a class by removing its entries.
- delete this file only after all 24 classes are clear.
- review stages only add, never delete.
- clear a class by removing its entries.
- delete this file only after all 24 classes are clear.
- review stages only add, never delete.
- clear a class by removing its entries.
- delete this file only after all 24 classes are clear.
- review stages only add, never delete.
- clear a class by removing its entries.
- delete this file only after all 24 classes are clear.
- review stages only add, never delete.
- clear a class by removing its entries.
- delete this file only after all 24 classes are clear.
- review stages only add, never delete.
- clear a class by removing its entries.
- delete this file only after all 24 classes are clear.
- review stages only add, never delete.
- clear a class by removing its entries.
- delete this file only after all 24 classes are clear.
- review stages only add, never delete.
- clear a class by removing its entries.
- delete this file only after all 24 classes are clear.
- review stages only add, never delete.
- clear a class by removing its entries.
- delete this file only after all 24 classes are clear.
- review stages only add, never delete.
- clear a class by removing its entries.
- delete this file only after all 24 classes are clear.
- review stages only add, never delete.
- clear a class by removing its entries.
- delete this file only after all 24 classes are clear.
- review stages only add, never delete.
- clear a class by removing its entries.
- delete this file only after all 24 classes are clear.
- review stages only add, never delete.
- clear a class by removing its entries.
- delete this file only after all 24 classes are clear.
- review stages only add, never delete.
- clear a class by removing its entries.
- delete this file only after all 24 classes are clear.
- review stages only add, never delete.
- clear a class by removing its entries.
- delete this file only after all 24 classes are clear.
- review stages only add, never delete.
- clear a class by removing its entries.
- delete this file only after all 24 classes are clear.
- review stages only add, never delete.
- clear a class by removing its entries.
- delete this file only after all 24 classes are clear.
- review stages only add, never delete.
- clear a class by removing its entries.
- delete this file only after all 24 classes are clear.
- review stages only add, never delete.
- clear a class by removing its entries.
- delete this file only after all 24 classes are clear.
- [class 11][pass 2] `scripts/01_build_manifests.py:18-45` and `src/liquidloc/dataio/manifests/build_manifests.py:1-38` both carry the raw-root anchoring contract for dataset manifest construction, but the script re-resolves dataset-config-relative roots while the library only consumes an already-resolved root. The duplicated anchoring path can drift if config placement or repo-root assumptions change.
- [class 11][pass 2] `scripts/14_build_summary.py:1-98` re-implements payload unwrapping, output-path anchoring, and case-ref normalization around `src/liquidloc/analysis/summary_builder.py:1-79`, which already owns the frozen summary contract. The duplicated normalization/output wrapper can drift if the summary payload shape or accepted wrappers change.
