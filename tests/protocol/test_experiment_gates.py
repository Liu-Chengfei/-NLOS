from __future__ import annotations

"""实验门控（experiment_gates）测试模块。

测试覆盖范围：
- quick/full 模式的门控规则
- 训练/评估的分裂角色验证
- 公开基准测试的数据集白名单
- 调参禁止与共享分裂要求

被测模块：liquidloc.protocol.experiment_gates"""

from pathlib import Path

import pytest

from liquidloc.common.config_utils import find_project_root
from liquidloc.protocol.experiment_gates import (
    _FAILURE_THRESHOLD_MAX_M,
    _PUBLIC_BENCHMARK_ALLOWED_DATASETS,
    assert_anchor_uniform_source,
    get_default_failure_threshold_m,
    load_experiment_protocol,
    normalize_run_mode,
    normalize_eval_request,
    normalize_public_benchmark_request,
    normalize_train_request,
)

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / 'fixtures' / 'datasets' / 'miluv'
_PROJECT_ROOT = find_project_root()


def _make_valid_protocol_cfg():
    """Build a complete valid experiment protocol config for testing."""
    return {
        'protocol_version': 2,
        'quick_full_rule': 'quick_smoke_scale__full_real_execution_required',
        'failure_sample_policy': 'retain_and_audit',
        'aggregation_order': ['single_run', 'repeat_summary', 'scene_summary', 'experiment_conclusion'],
        'conclusion_priority': ['p95', 'failure_rate', 'rmse', 'mae'],
        'public_benchmark': {
            'allowed_datasets': ['miluv', 'ntu_viral'],
            'frozen_eval_split': 'frozen_public_eval',
            'tuning_forbidden': True,
            'shared_split_required': True,
        },
        'training': {
            'allowed_split_roles': ['train', 'val'],
            'forbidden_split_roles': ['test', 'external', 'frozen_public_eval'],
        },
        'evaluation': {
            'require_prediction_bundles': True,
            'default_failure_threshold_m': 1.0,
            'failure_threshold_max_m': 10.0,
            'require_ground_truth_unless_smoke': True,
        },
    }


def test_load_experiment_protocol_rejects_mismatched_protocol_version(tmp_path):
    """冻结实验协议版本不匹配时应在 loader 入口拒绝。"""
    tmp_dir = _PROJECT_ROOT / "outputs" / "_test_tmp" / "experiment_gates"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = tmp_dir / 'bad_experiment_protocol.yaml'
    protocol_path.write_text(
        '\n'.join([
            'protocol_version: 999',
            'quick_full_rule: quick_smoke_scale__full_real_execution_required',
            'failure_sample_policy: retain_and_audit',
            'aggregation_order: [single_run, repeat_summary, scene_summary, experiment_conclusion]',
            'conclusion_priority: [p95, failure_rate, rmse, mae]',
            'public_benchmark:',
            '  frozen_eval_split: frozen_public_eval',
            '  tuning_forbidden: true',
            '  shared_split_required: true',
            'training:',
            '  allowed_split_roles: [train, val]',
            '  forbidden_split_roles: [test, external, frozen_public_eval]',
            'evaluation:',
            '  require_prediction_bundles: true',
            '  default_failure_threshold_m: 1.0',
            '  require_ground_truth_unless_smoke: true',
        ]),
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='experiment protocol version must be 2'):
        load_experiment_protocol(protocol_path)


def test_load_experiment_protocol_rejects_blank_string_path():
    """拒绝测试：load experiment protocol。\n\n验证被测功能对 load experiment protocol 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    with pytest.raises(ValueError, match='experiment protocol path must not be blank'):
        load_experiment_protocol('   ')


# ---------------------------------------------------------------------------
# normalize_train_request 测试
# ---------------------------------------------------------------------------

def test_train_gate_normal_case():
    """正常训练请求应通过校验。"""
    report = normalize_train_request({'split_ids': ['mini_seq'], 'split_role': 'train'})
    assert report['split_role'] == 'train'
    assert report['num_split_ids'] == 1


def test_train_gate_empty_split_ids_rejected():
    """空 split_ids 应被拒绝。"""
    with pytest.raises(ValueError, match='split_ids must be non-empty'):
        normalize_train_request({'split_ids': [], 'split_role': 'train'})


def test_train_gate_string_split_ids_rejected():
    """拒绝测试：train gate string split ids。\n\n验证被测功能对不合法的 train gate string split ids 输入正确抛出异常，\n防止无效参数通过验证。
    """
    with pytest.raises(TypeError, match='split_ids must be a list or tuple'):
        normalize_train_request({'split_ids': 'mini_seq', 'split_role': 'train'})


def test_train_gate_mapping_split_ids_rejected():
    """拒绝测试：train gate mapping split ids。\n\n验证被测功能对不合法的 train gate mapping split ids 输入正确抛出异常，\n防止无效参数通过验证。
    """
    with pytest.raises(TypeError, match='split_ids must be a list or tuple'):
        normalize_train_request({'split_ids': {'mini_seq': 1}, 'split_role': 'train'})


def test_train_gate_split_ids_are_normalized_and_deduplicated():
    """分裂测试：train gate。\n\n验证 train gate 的训练/验证分裂逻辑，\n确保分裂策略和审计正确。
    """
    report = normalize_train_request({
        'split_ids': [' mini_seq ', 'mini_seq', 'mini_seq_02'],
        'train_split_ids': [' mini_seq '],
        'val_split_ids': [' mini_seq_02 '],
        'split_role': 'train',
    })
    assert report['split_ids'] == ['mini_seq', 'mini_seq_02']
    assert report['train_split_ids'] == ['mini_seq']
    assert report['val_split_ids'] == ['mini_seq_02']


def test_train_gate_split_role_is_normalized():
    """分裂测试：train gate。\n\n验证 train gate 的训练/验证分裂逻辑，\n确保分裂策略和审计正确。
    """
    report = normalize_train_request({'split_ids': ['mini_seq'], 'split_role': ' train '})
    assert report['split_role'] == 'train'

    val_report = normalize_train_request({'split_ids': ['mini_seq'], 'split_role': 'val '})
    assert val_report['split_role'] == 'val'


def test_train_gate_blank_split_role_rejected():
    """拒绝测试：train gate blank split role。\n\n验证被测功能对不合法的 train gate blank split role 输入正确抛出异常，\n防止无效参数通过验证。
    """
    with pytest.raises(ValueError, match='training split_role must not be blank'):
        normalize_train_request({'split_ids': ['mini_seq'], 'split_role': '   '})


def test_train_gate_blank_split_id_item_rejected():
    """拒绝测试：train gate blank split id item。\n\n验证被测功能对不合法的 train gate blank split id item 输入正确抛出异常，\n防止无效参数通过验证。
    """
    with pytest.raises(ValueError, match='split_ids items must not be blank'):
        normalize_train_request({'split_ids': ['   '], 'split_role': 'train'})


def test_train_gate_disallowed_split_role_rejected():
    """不在 allowed_split_roles 中的角色应被拒绝。"""
    with pytest.raises(ValueError, match='split_role not allowed'):
        normalize_train_request({'split_ids': ['mini_seq'], 'split_role': 'hack'})


def test_train_gate_forbidden_split_role_rejected():
    """在 forbidden_split_roles 中的角色（如 frozen_public_eval）应被拒绝。

    冻结协议中 allowed_split_roles 与 forbidden_split_roles 无交集，
    因此 forbidden 角色会先被 allowed 白名单拦截。
    """
    with pytest.raises(ValueError, match='split_role not allowed'):
        normalize_train_request(
            {'split_ids': ['mini_seq'], 'split_role': 'frozen_public_eval'},
        )


def test_train_gate_val_role_allowed():
    """val 角色应被允许（协议 allowed_split_roles: [train, val]）。"""
    report = normalize_train_request({'split_ids': ['seq_v1'], 'split_role': 'val'})
    assert report['split_role'] == 'val'


# ---------------------------------------------------------------------------
# normalize_eval_request 测试
# ---------------------------------------------------------------------------

def test_eval_gate_invalid_case():
    """缺少 ground_truth_root 的非 smoke 评测请求应被拒绝。"""
    with pytest.raises(ValueError):
        normalize_eval_request({'prediction_bundles': [{'seq_id': 'x'}]})


def test_eval_gate_smoke_mode_without_ground_truth():
    """smoke 模式下缺少 ground_truth_root 应被允许。"""
    report = normalize_eval_request({
        'prediction_bundles': [{'seq_id': 'x'}],
        'smoke_mode': True,
    })
    assert report['num_prediction_bundles'] == 1


def test_eval_gate_zero_failure_threshold_rejected():
    """failure_threshold=0 应被拒绝。"""
    with pytest.raises(ValueError, match='failure_threshold must be positive'):
        normalize_eval_request({
            'prediction_bundles': [{'seq_id': 'x'}],
            'ground_truth_root': FIXTURE_ROOT,
            'failure_threshold': 0,
        })


def test_eval_gate_negative_failure_threshold_rejected():
    """负数 failure_threshold 应被拒绝。"""
    with pytest.raises(ValueError, match='failure_threshold must be positive'):
        normalize_eval_request({
            'prediction_bundles': [{'seq_id': 'x'}],
            'ground_truth_root': FIXTURE_ROOT,
            'failure_threshold': -1.0,
        })


def test_eval_gate_excessive_failure_threshold_rejected():
    """超过上限的 failure_threshold 应被拒绝。"""
    with pytest.raises(ValueError, match='failure_threshold must not exceed'):
        normalize_eval_request({
            'prediction_bundles': [{'seq_id': 'x'}],
            'ground_truth_root': FIXTURE_ROOT,
            'failure_threshold': _FAILURE_THRESHOLD_MAX_M + 1.0,
        })


def test_eval_gate_non_finite_failure_threshold_rejected():
    """非有限的 failure_threshold 应被拒绝。"""
    with pytest.raises(ValueError, match='failure_threshold must be finite'):
        normalize_eval_request({
            'prediction_bundles': [{'seq_id': 'x'}],
            'ground_truth_root': FIXTURE_ROOT,
            'failure_threshold': float("nan"),
        })


def test_eval_gate_bool_failure_threshold_rejected():
    """bool 伪装的 failure_threshold 应被拒绝。"""
    with pytest.raises(TypeError, match='failure_threshold must be a real number'):
        normalize_eval_request({
            'prediction_bundles': [{'seq_id': 'x'}],
            'ground_truth_root': FIXTURE_ROOT,
            'failure_threshold': True,
        })


def test_eval_gate_non_numeric_string_failure_threshold_rejected():
    """拒绝测试：eval gate non numeric string failure threshold。\n\n验证被测功能对不合法的 eval gate non numeric string failure threshold 输入正确抛出异常，\n防止无效参数通过验证。
    """
    with pytest.raises(TypeError, match='failure_threshold must be a real number'):
        normalize_eval_request({
            'prediction_bundles': [{'seq_id': 'x'}],
            'ground_truth_root': FIXTURE_ROOT,
            'failure_threshold': 'abc',
        })


def test_eval_gate_failure_threshold_at_max_allowed():
    """恰好等于上限的 failure_threshold 应被允许。"""
    report = normalize_eval_request({
        'prediction_bundles': [{'seq_id': 'x'}],
        'ground_truth_root': FIXTURE_ROOT,
        'failure_threshold': _FAILURE_THRESHOLD_MAX_M,
    })
    assert report['failure_threshold'] == _FAILURE_THRESHOLD_MAX_M


def test_eval_gate_custom_failure_threshold_within_range():
    """合理范围内的自定义 failure_threshold 应被允许。"""
    report = normalize_eval_request({
        'prediction_bundles': [{'seq_id': 'x'}],
        'ground_truth_root': FIXTURE_ROOT,
        'failure_threshold': 0.5,
    })
    assert report['failure_threshold'] == 0.5


def test_get_default_failure_threshold_reads_protocol_cfg():
    """协议冻结 default_failure_threshold_m 为 1.0，函数应从默认协议中读取该值。"""
    report = get_default_failure_threshold_m()
    assert report == pytest.approx(1.0)


def test_get_default_failure_threshold_rejects_non_positive_protocol_value():
    """拒绝测试：get default failure threshold。\n\n验证被测功能对 get default failure threshold 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    cfg = _make_valid_protocol_cfg()
    cfg['evaluation']['default_failure_threshold_m'] = 0.0
    with pytest.raises(ValueError, match='default_failure_threshold_m must be positive'):
        get_default_failure_threshold_m(cfg)


def test_get_default_failure_threshold_rejects_non_finite_protocol_value():
    """拒绝测试：get default failure threshold。\n\n验证被测功能对 get default failure threshold 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    cfg = _make_valid_protocol_cfg()
    cfg['evaluation']['default_failure_threshold_m'] = float("inf")
    with pytest.raises(ValueError, match='default_failure_threshold_m must be finite'):
        get_default_failure_threshold_m(cfg)


def test_get_default_failure_threshold_rejects_bool_protocol_value():
    """拒绝测试：get default failure threshold。\n\n验证被测功能对 get default failure threshold 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    cfg = _make_valid_protocol_cfg()
    cfg['evaluation']['default_failure_threshold_m'] = True
    with pytest.raises(TypeError, match='default_failure_threshold_m must be a real number'):
        get_default_failure_threshold_m(cfg)


def test_get_default_failure_threshold_rejects_non_numeric_string_protocol_value():
    """拒绝测试：get default failure threshold。\n\n验证被测功能对 get default failure threshold 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    cfg = _make_valid_protocol_cfg()
    cfg['evaluation']['default_failure_threshold_m'] = 'abc'
    with pytest.raises(TypeError, match='default_failure_threshold_m must be a real number'):
        get_default_failure_threshold_m(cfg)


def test_eval_gate_nonexistent_ground_truth_root_rejected():
    """不存在的 ground_truth_root 应在入口 gate 被拒绝。"""
    with pytest.raises(ValueError, match='ground_truth_root must point to an existing directory'):
        normalize_eval_request({
            'prediction_bundles': [{'seq_id': 'x'}],
            'ground_truth_root': 'C:/definitely/not/exist',
        })


def test_eval_gate_ground_truth_root_is_normalized():
    report = normalize_eval_request({
        'prediction_bundles': [{'seq_id': 'x'}],
        'ground_truth_root': ' tests/fixtures/datasets/miluv ',
    })
    assert report['ground_truth_root'] == 'tests/fixtures/datasets/miluv'


def test_eval_gate_blank_ground_truth_root_rejected():
    """拒绝测试：eval gate blank ground truth root。\n\n验证被测功能对不合法的 eval gate blank ground truth root 输入正确抛出异常，\n防止无效参数通过验证。
    """
    with pytest.raises(ValueError, match='ground_truth_root must not be blank'):
        normalize_eval_request({
            'prediction_bundles': [{'seq_id': 'x'}],
            'ground_truth_root': '   ',
        })


def test_eval_gate_empty_prediction_bundles_rejected():
    """空 prediction_bundles 在协议要求时应被拒绝。"""
    with pytest.raises(ValueError, match='prediction_bundles must be non-empty'):
        normalize_eval_request({
            'prediction_bundles': [],
            'ground_truth_root': FIXTURE_ROOT,
        })


def test_eval_gate_string_prediction_bundles_rejected():
    """拒绝测试：eval gate string prediction bundles。\n\n验证被测功能对不合法的 eval gate string prediction bundles 输入正确抛出异常，\n防止无效参数通过验证。
    """
    with pytest.raises(TypeError, match='prediction_bundles must be a list or tuple'):
        normalize_eval_request({
            'prediction_bundles': 'bundle.json',
            'ground_truth_root': FIXTURE_ROOT,
        })


def test_eval_gate_mapping_prediction_bundles_rejected():
    """拒绝测试：eval gate mapping prediction bundles。\n\n验证被测功能对不合法的 eval gate mapping prediction bundles 输入正确抛出异常，\n防止无效参数通过验证。
    """
    with pytest.raises(TypeError, match='prediction_bundles must be a list or tuple'):
        normalize_eval_request({
            'prediction_bundles': {'bundle': 1},
            'ground_truth_root': FIXTURE_ROOT,
        })


# ---------------------------------------------------------------------------
# normalize_public_benchmark_request 测试
# ---------------------------------------------------------------------------

def test_public_gate_invalid_split_case():
    """非冻结切分应被拒绝。"""
    with pytest.raises(ValueError, match='split must be'):
        normalize_public_benchmark_request({'dataset_name': 'miluv', 'seq_ids': ['mini_seq'], 'split': 'test'})


def test_public_gate_split_is_normalized():
    """分裂测试：public gate。\n\n验证 public gate 的训练/验证分裂逻辑，\n确保分裂策略和审计正确。
    """
    report = normalize_public_benchmark_request({
        'dataset_name': 'miluv',
        'seq_ids': ['mini_seq'],
        'split': ' frozen_public_eval ',
    }, dataset_entry={'frozen_public_eval_seq_ids': ['mini_seq']})
    assert report['split'] == 'frozen_public_eval'


def test_public_gate_blank_split_rejected():
    """拒绝测试：public gate blank split。\n\n验证被测功能对不合法的 public gate blank split 输入正确抛出异常，\n防止无效参数通过验证。
    """
    with pytest.raises(ValueError, match='public benchmark split must not be blank'):
        normalize_public_benchmark_request({
            'dataset_name': 'miluv',
            'seq_ids': ['mini_seq'],
            'split': '   ',
        })


def test_public_gate_empty_dataset_name_rejected():
    """空 dataset_name 应被拒绝。"""
    with pytest.raises(ValueError, match="non-empty 'dataset_name'"):
        normalize_public_benchmark_request({'dataset_name': '', 'seq_ids': ['mini_seq']})


def test_public_gate_disallowed_dataset_name_rejected():
    """不在允许列表中的 dataset_name 应被拒绝。"""
    with pytest.raises(ValueError, match='dataset_name not allowed'):
        normalize_public_benchmark_request({'dataset_name': 'unknown_dataset', 'seq_ids': ['seq1']})


def test_public_gate_allowed_dataset_miluv():
    """miluv 数据集应在允许列表中，校验应通过。"""
    report = normalize_public_benchmark_request({
        'dataset_name': 'miluv',
        'seq_ids': ['seq1'],
    }, dataset_entry={'frozen_public_eval_seq_ids': ['seq1']})
    assert report['dataset_name'] == 'miluv'


def test_public_gate_allowed_dataset_ntu_viral():
    """ntu_viral 数据集应在允许列表中，校验应通过。"""
    report = normalize_public_benchmark_request({
        'dataset_name': 'ntu_viral',
        'seq_ids': ['seq1'],
    }, dataset_entry={'frozen_public_eval_seq_ids': ['seq1']})
    assert report['dataset_name'] == 'ntu_viral'


def test_public_gate_dataset_name_case_insensitive():
    """数据集名称应大小写不敏感匹配（注册表使用小写，但上游可能传大写）。"""
    # 大写 MILUV 应该能通过校验
    report = normalize_public_benchmark_request({
        'dataset_name': 'MILUV',
        'seq_ids': ['seq1'],
    }, dataset_entry={'frozen_public_eval_seq_ids': ['seq1']})
    assert report['dataset_name'] == 'miluv'


def test_public_gate_empty_seq_ids_rejected():
    """空 seq_ids 应被拒绝。"""
    with pytest.raises(ValueError, match='seq_ids must be non-empty'):
        normalize_public_benchmark_request({'dataset_name': 'miluv', 'seq_ids': []})


def test_public_gate_string_seq_ids_rejected():
    """拒绝测试：public gate string seq ids。\n\n验证被测功能对不合法的 public gate string seq ids 输入正确抛出异常，\n防止无效参数通过验证。
    """
    with pytest.raises(TypeError, match='seq_ids must be a list or tuple'):
        normalize_public_benchmark_request({'dataset_name': 'miluv', 'seq_ids': 'seq1'})


def test_public_gate_mapping_seq_ids_rejected():
    """拒绝测试：public gate mapping seq ids。\n\n验证被测功能对不合法的 public gate mapping seq ids 输入正确抛出异常，\n防止无效参数通过验证。
    """
    with pytest.raises(TypeError, match='seq_ids must be a list or tuple'):
        normalize_public_benchmark_request({'dataset_name': 'miluv', 'seq_ids': {'seq1': 1}})


def test_public_gate_blank_seq_id_item_rejected():
    """拒绝测试：public gate blank seq id item。\n\n验证被测功能对不合法的 public gate blank seq id item 输入正确抛出异常，\n防止无效参数通过验证。
    """
    with pytest.raises(ValueError, match='seq_ids items must not be blank'):
        normalize_public_benchmark_request({'dataset_name': 'miluv', 'seq_ids': ['   ']})


def test_public_gate_seq_ids_are_normalized_and_deduplicated():
    """序列 ID 测试：public gate。\n\n验证 public gate 的序列 ID 处理，\n确保归一化和去重正确。
    """
    report = normalize_public_benchmark_request({
        'dataset_name': 'miluv',
        'seq_ids': [' mini_seq ', 'mini_seq', 'mini_seq_02 '],
    }, dataset_entry={'frozen_public_eval_seq_ids': ['mini_seq', 'mini_seq_02']})
    assert report['seq_ids'] == ['mini_seq', 'mini_seq_02']


def test_public_gate_tuning_mode_true_rejected():
    """tuning_mode=True 在协议禁止时应被拒绝。"""
    with pytest.raises(ValueError, match='tuning_mode is forbidden'):
        normalize_public_benchmark_request({
            'dataset_name': 'miluv',
            'seq_ids': ['seq1'],
            'tuning_mode': True,
        })


def test_public_gate_tuning_mode_string_true_rejected():
    """tuning_mode='true'（字符串）在协议禁止时应被拒绝。"""
    with pytest.raises(ValueError, match='tuning_mode is forbidden'):
        normalize_public_benchmark_request({
            'dataset_name': 'miluv',
            'seq_ids': ['seq1'],
            'tuning_mode': 'true',
        })


def test_public_gate_tuning_mode_string_true_with_spaces_rejected():
    """tuning_mode=' true ' 在协议禁止时也应被拒绝。"""
    with pytest.raises(ValueError, match='tuning_mode is forbidden'):
        normalize_public_benchmark_request({
            'dataset_name': 'miluv',
            'seq_ids': ['seq1'],
            'tuning_mode': ' true ',
        })


def test_public_gate_tuning_mode_numeric_one_rejected():
    """tuning_mode=1 应被视为开启 tuning 并拒绝。"""
    with pytest.raises(ValueError, match='tuning_mode is forbidden'):
        normalize_public_benchmark_request({
            'dataset_name': 'miluv',
            'seq_ids': ['seq1'],
            'tuning_mode': 1,
        })


def test_public_gate_tuning_mode_string_one_rejected():
    """tuning_mode='1' 应被视为开启 tuning 并拒绝。"""
    with pytest.raises(ValueError, match='tuning_mode is forbidden'):
        normalize_public_benchmark_request({
            'dataset_name': 'miluv',
            'seq_ids': ['seq1'],
            'tuning_mode': '1',
        })


def test_public_gate_tuning_mode_zero_allowed():
    """tuning_mode=0 不应被视为开启 tuning，应通过校验。"""
    report = normalize_public_benchmark_request({
        'dataset_name': 'miluv',
        'seq_ids': ['seq1'],
        'tuning_mode': 0,
    }, dataset_entry={'frozen_public_eval_seq_ids': ['seq1']})
    assert report['dataset_name'] == 'miluv'


def test_public_gate_tuning_mode_off_allowed():
    """tuning_mode='off' 不应被视为开启 tuning，应通过校验。"""
    report = normalize_public_benchmark_request({
        'dataset_name': 'miluv',
        'seq_ids': ['seq1'],
        'tuning_mode': 'off',
    }, dataset_entry={'frozen_public_eval_seq_ids': ['seq1']})
    assert report['dataset_name'] == 'miluv'


def test_public_gate_tuning_mode_false_allowed():
    """tuning_mode=False 不应被视为开启 tuning，应通过校验。"""
    report = normalize_public_benchmark_request({
        'dataset_name': 'miluv',
        'seq_ids': ['seq1'],
        'tuning_mode': False,
    }, dataset_entry={'frozen_public_eval_seq_ids': ['seq1']})
    assert report['dataset_name'] == 'miluv'


def test_public_gate_dataset_entry_none_vs_empty_dict():
    """dataset_entry=None 和 dataset_entry={} 都缺少 frozen_public_eval_seq_ids，应被拒绝。"""
    with pytest.raises(ValueError, match='frozen_public_eval_seq_ids'):
        normalize_public_benchmark_request(
            {'dataset_name': 'miluv', 'seq_ids': ['seq1']},
            dataset_entry=None,
        )
    with pytest.raises(ValueError, match='frozen_public_eval_seq_ids'):
        normalize_public_benchmark_request(
            {'dataset_name': 'miluv', 'seq_ids': ['seq1']},
            dataset_entry={},
        )


def test_public_gate_dataset_entry_with_keys():
    """传入非空 dataset_entry 时应正确记录其字段列表。"""
    report = normalize_public_benchmark_request(
        {'dataset_name': 'miluv', 'seq_ids': ['seq1']},
        dataset_entry={'reader': 'x', 'prepare_script': 'y', 'frozen_public_eval_seq_ids': ['seq1']},
    )
    assert report['dataset_entry_keys'] == ['frozen_public_eval_seq_ids', 'prepare_script', 'reader']  # 排序后的字段列表


def test_public_gate_non_mapping_dataset_entry_rejected():
    """拒绝测试：public gate non mapping dataset entry。\n\n验证被测功能对不合法的 public gate non mapping dataset entry 输入正确抛出异常，\n防止无效参数通过验证。
    """
    with pytest.raises(TypeError, match='dataset_entry must be a mapping when provided'):
        normalize_public_benchmark_request(
            {'dataset_name': 'miluv', 'seq_ids': ['seq1']},
            dataset_entry=['reader'],
        )


def test_public_benchmark_allowed_datasets_contains_expected():
    """允许列表应至少包含 miluv 和 ntu_viral。"""
    assert 'miluv' in _PUBLIC_BENCHMARK_ALLOWED_DATASETS
    assert 'ntu_viral' in _PUBLIC_BENCHMARK_ALLOWED_DATASETS


def test_public_benchmark_allowed_datasets_excludes_util():
    """util 已注册为补充数据集，但不属于当前官方公开 benchmark 面。"""
    assert 'util' not in _PUBLIC_BENCHMARK_ALLOWED_DATASETS


def test_train_gate_invalid_mode_rejected():
    """拒绝测试：train gate invalid mode。\n\n验证被测功能对不合法的 train gate invalid mode 输入正确抛出异常，\n防止无效参数通过验证。
    """
    with pytest.raises(ValueError, match='mode must be one of'):
        normalize_train_request({
            'split_ids': ['mini_seq'],
            'split_role': 'train',
            'mode': 'paper',
        })


def test_eval_gate_invalid_mode_rejected():
    """拒绝测试：eval gate invalid mode。\n\n验证被测功能对不合法的 eval gate invalid mode 输入正确抛出异常，\n防止无效参数通过验证。
    """
    with pytest.raises(ValueError, match='mode must be one of'):
        normalize_eval_request({
            'prediction_bundles': [{'seq_id': 'x'}],
            'ground_truth_root': FIXTURE_ROOT,
            'mode': 'paper',
        })


def test_eval_gate_bool_mode_rejected():
    """拒绝测试：eval gate bool mode。\n\n验证被测功能对不合法的 eval gate bool mode 输入正确抛出异常，\n防止无效参数通过验证。
    """
    with pytest.raises(ValueError, match='mode must be one of'):
        normalize_eval_request({
            'prediction_bundles': [{'seq_id': 'x'}],
            'ground_truth_root': FIXTURE_ROOT,
            'mode': False,
        })


def test_normalize_run_mode_blank_string_rejected():
    """拒绝测试：normalize run mode blank string。\n\n验证被测功能对不合法的 normalize run mode blank string 输入正确抛出异常，\n防止无效参数通过验证。
    """
    with pytest.raises(ValueError, match='mode must be one of'):
        normalize_run_mode('   ', default_mode='full')


def test_eval_gate_default_mode_is_full():
    """模式测试：eval gate default。\n\n验证 eval gate default 的模式验证，\n确保仅允许 quick/full 模式。
    """
    report = normalize_eval_request({
        'prediction_bundles': [{'seq_id': 'x'}],
        'ground_truth_root': FIXTURE_ROOT,
    })
    assert report['mode'] == 'full'


def test_eval_gate_explicit_quick_mode_preserved():
    """显式测试：eval gate。\n\n验证 eval gate 的显式参数优先级，\n确保显式指定覆盖隐式推断。
    """
    report = normalize_eval_request({
        'prediction_bundles': [{'seq_id': 'x'}],
        'ground_truth_root': FIXTURE_ROOT,
        'mode': 'quick',
    })
    assert report['mode'] == 'quick'


def test_public_gate_invalid_mode_rejected():
    """拒绝测试：public gate invalid mode。\n\n验证被测功能对不合法的 public gate invalid mode 输入正确抛出异常，\n防止无效参数通过验证。
    """
    with pytest.raises(ValueError, match='mode must be one of'):
        normalize_public_benchmark_request({
            'dataset_name': 'miluv',
            'seq_ids': ['seq1'],
            'mode': 'paper',
        })


def test_public_gate_default_mode_is_full():
    """模式测试：public gate default。\n\n验证 public gate default 的模式验证，\n确保仅允许 quick/full 模式。
    """
    report = normalize_public_benchmark_request({
        'dataset_name': 'miluv',
        'seq_ids': ['seq1'],
    }, dataset_entry={'frozen_public_eval_seq_ids': ['seq1']})
    assert report['mode'] == 'full'


def test_public_gate_explicit_quick_mode_preserved():
    """显式测试：public gate。\n\n验证 public gate 的显式参数优先级，\n确保显式指定覆盖隐式推断。
    """
    report = normalize_public_benchmark_request({
        'dataset_name': 'miluv',
        'seq_ids': ['seq1'],
        'mode': 'quick',
    }, dataset_entry={'frozen_public_eval_seq_ids': ['seq1']})
    assert report['mode'] == 'quick'


def test_eval_gate_string_smoke_mode_rejected():
    """拒绝测试：eval gate string smoke mode。\n\n验证被测功能对不合法的 eval gate string smoke mode 输入正确抛出异常，\n防止无效参数通过验证。
    """
    with pytest.raises(TypeError, match='smoke_mode must be a boolean'):
        normalize_eval_request({
            'prediction_bundles': [{'seq_id': 'x'}],
            'smoke_mode': 'true',
        })


def test_assert_anchor_uniform_source_na_in_3_4_5_passes():
    """§8.1 L1360 Na∈{3,4,5} 落在推荐范围时门禁通过。"""
    layout = {'anchor_positions': [[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]]}
    report = assert_anchor_uniform_source({'ekf': layout, 'fgo': layout})
    assert report['uniform'] is True
    assert report['na_in_range'] is True
    assert report['na_count'] == 3


def test_assert_anchor_uniform_source_na_ge_8_raises():
    """§8.1 L1360 Na≥8 高冗余优 GDOP 时门禁 raise。"""
    layout = {'anchor_positions': [[float(i), 0.0] for i in range(8)]}
    with pytest.raises(ValueError, match='anchor_count violation: Na=8 not in'):
        assert_anchor_uniform_source({'ekf': layout, 'fgo': layout})


def test_assert_anchor_uniform_source_na_lt_3_raises():
    """§8.1 L1360 Na<3 欠定压力不足时门禁 raise。"""
    layout = {'anchor_positions': [[0.0, 0.0], [1.0, 0.0]]}
    with pytest.raises(ValueError, match='anchor_count violation: Na=2 not in'):
        assert_anchor_uniform_source({'ekf': layout, 'fgo': layout})


def test_assert_anchor_uniform_source_different_layouts_raises():
    """Layout mismatch 时门禁 raise（与 Na 校验独立）。"""
    layout_a = {'anchor_positions': [[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]]}
    layout_b = {'anchor_positions': [[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]]}
    with pytest.raises(ValueError, match='methods using different anchor layouts'):
        assert_anchor_uniform_source({'ekf': layout_a, 'fgo': layout_b})


def test_assert_anchor_uniform_source_no_raise_when_disabled():
    """raise_on_violation=False 时只返回 report 不 raise。"""
    layout = {'anchor_positions': [[float(i), 0.0] for i in range(8)]}
    report = assert_anchor_uniform_source(
        {'ekf': layout, 'fgo': layout}, raise_on_violation=False
    )
    assert report['uniform'] is True
    assert report['na_in_range'] is False
    assert report['na_count'] == 8


# ========== §8.2.0 政策5/6 assert_seed_required / assert_seed_decoupling ==========

def test_assert_seed_required_all_seeded_passes():
    """§8.2.0 政策5: 全部序列含 finite-numeric seed → pass."""
    from liquidloc.protocol.experiment_gates import assert_seed_required
    report = assert_seed_required({'seq1': 42, 'seq2': 7, 'seq3': '13'})
    assert report['all_seeded'] is True
    assert report['n_sequences'] == 3
    assert report['n_unseeded'] == 0


def test_assert_seed_required_none_seed_raises():
    """§8.2.0 政策5: 含 None seed → raise."""
    from liquidloc.protocol.experiment_gates import assert_seed_required
    import pytest
    with pytest.raises(ValueError, match='seed_required violation'):
        assert_seed_required({'seq1': 42, 'seq2': None})


def test_assert_seed_required_empty_raises():
    """§8.2.0 政策5: 空字典 → raise."""
    from liquidloc.protocol.experiment_gates import assert_seed_required
    import pytest
    with pytest.raises(ValueError, match='seed_required violation'):
        assert_seed_required({})


def test_assert_seed_required_root_none_raises():
    """§8.2.0 政策5: None root → raise."""
    from liquidloc.protocol.experiment_gates import assert_seed_required
    import pytest
    with pytest.raises(ValueError, match='seed_required violation'):
        assert_seed_required(None)


def test_assert_seed_required_no_raise_when_disabled():
    """§8.2.0 政策5: raise_on_violation=False 时只返回 report 不抛."""
    from liquidloc.protocol.experiment_gates import assert_seed_required
    report = assert_seed_required({'seq1': None}, raise_on_violation=False)
    assert report['all_seeded'] is False
    assert report['n_unseeded'] == 1
    assert 'seq1' in report['unseeded_seq_ids']


def test_assert_seed_decoupling_distinct_seeds_passes():
    """§8.2.0 政策6: 三个不同种子 → pass."""
    from liquidloc.protocol.experiment_gates import assert_seed_decoupling
    report = assert_seed_decoupling(42, 7, 13)
    assert report['decoupled'] is True
    assert report['n_unique_seeds'] == 3
    assert report['collisions'] == []


def test_assert_seed_decoupling_fully_coupled_raises():
    """§8.2.0 政策6: 三种子相同（完全耦合）→ raise."""
    from liquidloc.protocol.experiment_gates import assert_seed_decoupling
    import pytest
    with pytest.raises(ValueError, match='seed_decoupling violation'):
        assert_seed_decoupling(42, 42, 42)


def test_assert_seed_decoupling_partial_collision_passes():
    """§8.2.0 政策6: 部分碰撞（两同）但非完全耦合 → pass + collisions 标记."""
    from liquidloc.protocol.experiment_gates import assert_seed_decoupling
    report = assert_seed_decoupling(42, 42, 13)
    assert report['decoupled'] is True
    assert report['fully_coupled'] is False
    assert len(report['collisions']) >= 1
    assert 'trajectory==nlos(42.0)' in report['collisions']


def test_assert_seed_decoupling_no_raise_when_disabled():
    """§8.2.0 政策6: raise_on_violation=False 时只返回 report 不抛."""
    from liquidloc.protocol.experiment_gates import assert_seed_decoupling
    report = assert_seed_decoupling(42, 42, 42, raise_on_violation=False)
    assert report['decoupled'] is False
    assert report['fully_coupled'] is True


def test_assert_seed_decoupling_string_seeds_passes():
    """§8.2.0 政策6: 字符串种子可数值化 → 比较生效."""
    from liquidloc.protocol.experiment_gates import assert_seed_decoupling
    report = assert_seed_decoupling('42', '7', '13')
    assert report['decoupled'] is True
    assert report['seeds']['trajectory'] == 42.0
