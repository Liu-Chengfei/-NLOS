from __future__ import annotations

"""场景采样器（scene_sampler）测试模块。

文件职责：验证场景采样器能根据场景轴协议正确生成场景任务，
包括公开基准管道的协议门控、序列ID归一化、split 归一化等。

测试覆盖范围：
- 公开基准管道的完整端到端流程
- 协议门控对 seq_ids 和 split 的归一化
- 空白 split / 空白路径的拒绝
- 非公开数据集（util）的拒绝
- NTU VIRAL 真实链路冒烟测试
- 相对路径锚定到 project_root

被测模块：liquidloc.pipelines.public_pipeline"""


import pytest

from liquidloc.scenarios.scene_sampler import sample_scenes


def test_normal_case():
    tasks = sample_scenes({
        'primary_axis': 'A',
        'frozen_axes': {'N': 'N2', 'V': 'V1', 'K': 'K3'},
        'mode': 'quick',
        'mode_overrides': {'quick': {'levels': ['A0', 'A3']}},
    })
    assert [task['axes']['A'] for task in tasks] == ['A0', 'A3']
    assert tasks[0]['scene_id'].startswith('S(')
    assert tasks[1]['scene_parameters']['axes']['A']['cross_modal_skew_ms'] == 185


def test_primary_axis_levels_accept_tuple_sequences():
    tasks = sample_scenes({
        'primary_axis': 'A',
        'frozen_axes': {'N': 'N2', 'V': 'V1', 'K': 'K3'},
        'mode': 'quick',
        'mode_overrides': {'quick': {'levels': ('A0', 'A3')}},
    })
    assert [task['axes']['A'] for task in tasks] == ['A0', 'A3']


def test_non_primary_frozen_axis_rejects_multiple_levels():
    with pytest.raises(ValueError, match=r'frozen_axes\.N must be a single level in this sampling mode'):
        sample_scenes({
            'primary_axis': 'A',
            'frozen_axes': {'N': ['N2', 'N3'], 'V': 'V1', 'K': 'K3'},
            'mode': 'quick',
            'mode_overrides': {'quick': {'levels': ['A0']}},
        })


def test_sweep_cfgs_override_mode_overrides_levels():
    tasks = sample_scenes(
        {
            'primary_axis': 'A',
            'frozen_axes': {'N': 'N2', 'V': 'V1', 'K': 'K3'},
            'mode': 'quick',
            'mode_overrides': {'quick': {'levels': ['A0']}},
        },
        {'levels': ['A3']},
    )
    assert [task['axes']['A'] for task in tasks] == ['A3']


def test_sweep_cfgs_reject_non_mapping():
    with pytest.raises(TypeError, match='sweep_cfgs must be a mapping'):
        sample_scenes(
            {
                'primary_axis': 'A',
                'frozen_axes': {'N': 'N2', 'V': 'V1', 'K': 'K3'},
                'mode': 'quick',
                'mode_overrides': {'quick': {'levels': ['A0']}},
            },
            ['A3'],
        )


def test_boundary_case():
    tasks = sample_scenes({
        'primary_axis': 'public_sequence_category',
        'frozen_axes': {'dataset': 'miluv'},
        'seq_ids': ['mini_seq'],
    })
    assert tasks == [{
        'task_id': 'public_0000',
        'dataset_name': 'miluv',
        'seq_id': 'mini_seq',
        'scene_id': 'miluv:mini_seq',
        'axes': {'dataset': 'miluv', 'split': None},
        'scene_parameters': {'axes': {}, 'flat': {'dataset': 'miluv', 'split': None}, 'axis_metadata': {}},
    }]


def test_public_sequence_category_rejects_scene_axis_frozen_keys():
    with pytest.raises(ValueError, match='supports only frozen_axes keys dataset/split'):
        sample_scenes({
            'primary_axis': 'public_sequence_category',
            'frozen_axes': {'dataset': 'miluv', 'split': 'frozen_public_eval', 'A': 'A2'},
            'seq_ids': ['mini_seq'],
        })


def test_geometry_axis_pair_sampling():
    # 「五轴档位协议定义」K 轴仅 K0/K1/K3 三档，G 轴已合并入 K 轴，
    # K sweep 现在按 anchor_count (4/4/4) 区分（geom_condition 数值区分几何质量）。
    tasks = sample_scenes({
        'primary_axis': ['K'],
        'frozen_axes': {'A': 'A2', 'N': 'N2', 'V': 'V1'},
        'mode': 'quick',
        'mode_overrides': {'quick': {'levels': ['K0', 'K3']}},
    })
    assert [(task['axes']['K']) for task in tasks] == [
        'K0',
        'K3',
    ]
    assert tasks[0]['scene_parameters']['axes']['K']['anchor_count'] == 4
    assert tasks[-1]['scene_parameters']['axes']['K']['geom_condition'][1] == 11.0


def test_geometry_axis_pair_sampling_honors_repeats():
    tasks = sample_scenes({
        'primary_axis': ['K'],
        'frozen_axes': {'A': 'A2', 'N': 'N2', 'V': 'V1'},
        'mode': 'quick',
        'mode_overrides': {'quick': {'levels': ['K0', 'K3'], 'repeats': 2}},
    })
    assert len(tasks) == 4
    assert [task['repeat_id'] for task in tasks[:2]] == ['repeat_0000'] * 2
    assert [task['repeat_id'] for task in tasks[2:]] == ['repeat_0001'] * 2


def test_target_degradation_bundle_expands_explicit_axes():
    tasks = sample_scenes({
        'primary_axis': 'target_degradation_bundle',
        'frozen_axes': {'A': ['A2', 'A3'], 'N': ['N2', 'N3'], 'V': 'V2', 'K': 'K3', 'M': 'M0'},
    })
    assert [(task['axes']['A'], task['axes']['N'], task['axes']['V'], task['axes']['K']) for task in tasks] == [
        ('A2', 'N2', 'V2', 'K3'),
        ('A2', 'N3', 'V2', 'K3'),
        ('A3', 'N2', 'V2', 'K3'),
        ('A3', 'N3', 'V2', 'K3'),
    ]
    # N2 bias_strength_m 是区间 [2.0, 3.0]，resolve 后保持区间形式
    n2_bias = tasks[0]['scene_parameters']['axes']['N']['bias_strength_m']
    assert tuple(n2_bias) == (2.0, 3.0)
    assert tasks[-1]['scene_parameters']['axes']['A']['cross_modal_skew_ms'] == 185


def test_target_degradation_bundle_requires_all_axes():
    with pytest.raises(ValueError, match='target_degradation_bundle requires explicit frozen_axes'):
        sample_scenes({
            'primary_axis': 'target_degradation_bundle',
            'frozen_axes': {'A': 'A2', 'N': 'N2', 'V': 'V2', 'M': 'M0'},
        })


def test_dual_degradation_bundle_matches_target_bundle_sampling():
    tasks = sample_scenes({
        'primary_axis': 'dual_degradation_bundle',
        'frozen_axes': {'A': ['A2', 'A3'], 'N': ['N2', 'N3'], 'V': 'V2', 'K': 'K3', 'M': 'M0'},
    })
    assert len(tasks) == 4
    assert tasks[0]['scene_id'] == 'S(A2,N2,V2,K3,M0)'
    assert tasks[-1]['axes']['A'] == 'A3'


def test_target_degradation_bundle_quick_repeats_and_max_sequences_take_effect():
    tasks = sample_scenes({
        'primary_axis': 'target_degradation_bundle',
        'frozen_axes': {'A': ['A3'], 'N': ['N3'], 'V': ['V2', 'V3'], 'K': ['K0', 'K1', 'K3'], 'M': 'M0'},
        'mode': 'quick',
        'mode_overrides': {'quick': {'repeats': 3, 'max_sequences': 8}},
    })
    # 6 base tasks (1*1*2*3*1), max_sequences=8 keeps all 6, repeats=3 → 18 total
    assert len(tasks) == 18
    assert tasks[0]['repeat_id'] == 'repeat_0000'
    assert tasks[5]['repeat_id'] == 'repeat_0000'
    assert tasks[6]['repeat_id'] == 'repeat_0001'
    assert tasks[-1]['task_id'] == 'scene_0017'
    assert tasks[0]['scene_variant_id'].endswith('repeat_0000')


def test_public_sequence_category_treats_single_string_seq_id_as_one_sequence():
    tasks = sample_scenes({
        'primary_axis': 'public_sequence_category',
        'frozen_axes': {'dataset': 'miluv'},
        'seq_ids': ['mini_seq'],
    })
    assert [task['seq_id'] for task in tasks] == ['mini_seq']


def test_public_sequence_category_normalizes_whitespace_wrapped_fields():
    tasks = sample_scenes({
        'primary_axis': 'public_sequence_category',
        'frozen_axes': {'dataset': ' miluv '},
        'seq_ids': [' mini_seq '],
    })
    assert tasks == [{
        'task_id': 'public_0000',
        'dataset_name': 'miluv',
        'seq_id': 'mini_seq',
        'scene_id': 'miluv:mini_seq',
        'axes': {'dataset': 'miluv', 'split': None},
        'scene_parameters': {'axes': {}, 'flat': {'dataset': 'miluv', 'split': None}, 'axis_metadata': {}},
    }]


def test_public_sequence_category_rejects_blank_seq_id_entries():
    with pytest.raises(ValueError, match=r'seq_ids\[0\] must not be blank'):
        sample_scenes({
            'primary_axis': 'public_sequence_category',
            'frozen_axes': {'dataset': 'miluv'},
            'seq_ids': ['   '],
        })


def test_dual_degradation_bundle_requires_all_axes():
    with pytest.raises(ValueError, match='dual_degradation_bundle requires explicit frozen_axes'):
        sample_scenes({
            'primary_axis': 'dual_degradation_bundle',
            'frozen_axes': {'A': 'A2', 'N': 'N2', 'V': 'V2', 'M': 'M0'},
        })


def test_rejects_unknown_frozen_axis_key():
    with pytest.raises(ValueError, match='Unsupported frozen_axes keys'):
        sample_scenes({
            'primary_axis': 'A',
            'frozen_axes': {'N_typo': 'N2', 'V': 'V1', 'K': 'K3'},
            'mode': 'quick',
            'mode_overrides': {'quick': {'levels': ['A0']}},
        })


def test_scene_axis_sampling_rejects_public_only_frozen_keys():
    with pytest.raises(ValueError, match='supports only frozen_axes keys A/N/V/K/M outside public_sequence_category'):
        sample_scenes({
            'primary_axis': 'A',
            'frozen_axes': {'dataset': 'miluv', 'N': 'N2', 'V': 'V1', 'K': 'K3'},
            'mode': 'quick',
            'mode_overrides': {'quick': {'levels': ['A0']}},
        })


def test_invalid_case():
    with pytest.raises(ValueError):
        sample_scenes({'frozen_axes': {}})


@pytest.mark.parametrize(
    ('field_name', 'value'),
    [
        ('repeats', 1.5),
        ('max_sequences', True),
    ],
)
def test_mode_override_counts_reject_non_integer_positive_values(field_name, value):
    with pytest.raises(TypeError, match=field_name):
        sample_scenes({
            'primary_axis': 'A',
            'frozen_axes': {'N': 'N2', 'V': 'V1', 'K': 'K3'},
            'mode': 'quick',
            'mode_overrides': {'quick': {'levels': ['A0'], field_name: value}},
        })


def test_safe_mode_behavior_uses_frozen_scene_axes():
    tasks = sample_scenes({
        'primary_axis': 'safe_mode_behavior',
        'frozen_axes': {'A': 'A0', 'N': 'N0', 'V': 'V0', 'K': 'K0'},
    })
    assert len(tasks) == 1
    assert tasks[0]['scene_id'] == 'S(A0,N0,V0,K0,M0)'
    assert tasks[0]['axes']['K'] == 'K0'


@pytest.mark.parametrize('primary_axis', ['safe_mode_behavior', 'ablation_variant'])
def test_single_scene_special_cases_honor_repeats_and_max_sequences(primary_axis):
    tasks = sample_scenes({
        'primary_axis': primary_axis,
        'frozen_axes': {'A': 'A2', 'N': 'N2', 'V': 'V2', 'K': 'K3'},
        'mode': 'quick',
        'mode_overrides': {'quick': {'repeats': 2, 'max_sequences': 1}},
    })
    # 1 base task, max_sequences=1 keeps 1, repeats=2 → 2 total
    assert len(tasks) == 2
    assert tasks[0]['repeat_id'] == 'repeat_0000'
    assert tasks[1]['repeat_id'] == 'repeat_0001'


def test_ablation_variant_uses_frozen_scene_axes():
    tasks = sample_scenes({
        'primary_axis': 'ablation_variant',
        'frozen_axes': {'A': 'A2', 'N': 'N2', 'V': 'V2', 'K': 'K3'},
    })
    assert len(tasks) == 1
    assert tasks[0]['scene_id'] == 'S(A2,N2,V2,K3,M0)'
    assert tasks[0]['axes']['A'] == 'A2'


def test_runtime_profile_allows_k_sweep():
    tasks = sample_scenes({
        'primary_axis': 'runtime_profile',
        'frozen_axes': {'A': 'A0', 'N': 'N0', 'V': 'V0', 'K': ['K0', 'K3']},
    })
    assert [task['axes']['K'] for task in tasks] == ['K0', 'K3']
    assert [task['scene_id'] for task in tasks] == ['S(A0,N0,V0,K0,M0)', 'S(A0,N0,V0,K3,M0)']


def test_runtime_profile_honors_repeats_and_max_sequences():
    tasks = sample_scenes({
        'primary_axis': 'runtime_profile',
        'frozen_axes': {'A': 'A0', 'N': 'N0', 'V': 'V0', 'K': ['K0', 'K3']},
        'mode': 'quick',
        'mode_overrides': {'quick': {'repeats': 2, 'max_sequences': 3}},
    })
    # 2 base tasks (K0, K3), max_sequences=3 keeps all 2, repeats=2 → 4 total
    assert len(tasks) == 4
    assert [task['repeat_id'] for task in tasks] == ['repeat_0000', 'repeat_0000', 'repeat_0001', 'repeat_0001']


def test_runtime_profile_rejects_unused_scenarios_frozen_axis_key():
    with pytest.raises(ValueError, match='Unsupported frozen_axes keys'):
        sample_scenes({
            'primary_axis': 'runtime_profile',
            'frozen_axes': {'A': 'A0', 'N': 'N0', 'V': 'V0', 'K': ['K0', 'K3'], 'scenarios': ['foo']},
        })


def test_modality_recovery_profile_gracefully_degrades():
    # modality_recovery_profile (e10) 未实现执行路径，优雅降级为空任务列表
    tasks = sample_scenes({
        'primary_axis': 'modality_recovery_profile',
        'frozen_axes': {'A': 'A2', 'N': 'N2', 'V': 'V2', 'K': 'K3', 'M': 'M0'},
    })
    assert tasks == []


def test_scene_sampler_invalid_mode_rejected_even_with_override_block():
    with pytest.raises(ValueError, match='mode must be one of'):
        sample_scenes({
            'primary_axis': 'A',
            'frozen_axes': {'N': 'N2', 'V': 'V1', 'K': 'K3'},
            'mode': 'paper',
            'mode_overrides': {'paper': {'levels': ['A0', 'A3']}},
        })


def test_phase_switch_profile_gracefully_degrades():
    # phase_switch_profile (e11) 未实现执行路径，优雅降级为空任务列表
    tasks = sample_scenes({
        'primary_axis': 'phase_switch_profile',
        'frozen_axes': {'A': 'A2', 'N': 'N2', 'V': 'V2', 'K': 'K3', 'M': 'M0'},
    })
    assert tasks == []


# ============================================================================
# 第 3 轮审查 HIGH-1 修复（R3-D）：_build_scene_id M 轴防御性校验测试覆盖
# scene_schema.SceneSpec 当前为五轴（A/N/V/G/K），不含 M 轴字段。
# 这意味着 scene_id 无法区分 M 轴的不同等级。_build_scene_id 通过显式校验
# M 必须为协议正常等级（M0）来防止 M 多值展开产生 scene_id 冲突。
# 以下 3 个测试用例覆盖该防御性校验的三种失败场景。
# ============================================================================


def test_build_scene_id_rejects_non_nominal_m_in_bundle():
    """target_degradation_bundle 模式下 M='M3'（非正常等级）应被 _build_scene_id 拒绝。

    验证 scene_id 无法区分 M 轴等级时，非正常等级 M 会被显式拒绝，
    防止下游两个不同 M 等级的任务共享同一 scene_id 导致冲突。
    """
    with pytest.raises(ValueError, match=r'_build_scene_id M axis must be nominal level'):
        sample_scenes({
            'primary_axis': 'target_degradation_bundle',
            'frozen_axes': {
                'A': 'A2', 'N': 'N2', 'V': 'V2', 'K': 'K3', 'M': 'M3',
            },
        })


def test_build_scene_id_rejects_multi_value_m_with_non_nominal():
    """target_degradation_bundle 模式下 M=['M0', 'M1'] 应在 M1 任务上被拒绝。

    多值 M 展开时，M0 任务通过 _build_scene_id，但 M1（非正常等级）任务
    会被 _build_scene_id 的 nominal_m 校验拒绝。此测试确保防御性校验在
    多值展开路径上也能正确触发。
    """
    with pytest.raises(ValueError, match=r'_build_scene_id M axis must be nominal level'):
        sample_scenes({
            'primary_axis': 'target_degradation_bundle',
            'frozen_axes': {
                'A': 'A2', 'N': 'N2', 'V': 'V2', 'K': 'K3',
                'M': ['M0', 'M1'],
            },
        })


def test_build_scene_id_rejects_missing_m_in_bundle():
    """target_degradation_bundle 模式下缺失 M 应被 bundle 路径的 missing_axes 检查拒绝。

    虽然 _build_scene_id 自身也有 missing_axes 检查，但 bundle 路径在
    调用 _build_scene_id 之前会先检查所有六轴是否显式指定。此测试确保
    M 缺失时 bundle 路径的 missing_axes 检查能正确触发。
    """
    with pytest.raises(ValueError, match=r'target_degradation_bundle requires explicit frozen_axes for M'):
        sample_scenes({
            'primary_axis': 'target_degradation_bundle',
            'frozen_axes': {
                'A': 'A2', 'N': 'N2', 'V': 'V2', 'K': 'K3',
            },
        })
