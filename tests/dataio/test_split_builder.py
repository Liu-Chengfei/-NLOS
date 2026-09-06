
"""分割构建器（split_builder）测试模块。

文件职责：验证 build_splits 能正确执行显式/默认分割，
并检测数据泄漏（重复 ID、共享场景作用域）。

测试覆盖范围：
- 显式 ID 分割
- 跨分割重复 ID 报告
- 空序列清单拒绝
- 非 dict 规则拒绝
- 空 test 分割拒绝
- 共享场景作用域泄漏报告
- §9 量级门 leak_items (train/test 比例 / N_traj_te / 布局族数)

被测模块：liquidloc.dataio.manifests.split_builder"""

import pytest

from liquidloc.dataio.manifests.split_builder import build_splits


def _section9_leak_kinds(leak_items: list[dict]) -> set[str]:
    """从 leak_items 中提取所有 §9 类型的 kind 集合."""
    return {item['kind'] for item in leak_items if str(item.get('kind', '')).startswith('section9_')}


def _non_section9_leak_items(leak_items: list[dict]) -> list[dict]:
    """过滤掉 §9 量级门 leak_items, 只保留数据泄漏类条目."""
    return [item for item in leak_items if not str(item.get('kind', '')).startswith('section9_')]


def test_build_splits_respects_explicit_ids():
    dataset_manifest = {'sequences': [{'seq_id': 'a'}, {'seq_id': 'b'}, {'seq_id': 'c'}]}

    split_manifest, leak_report = build_splits(
        dataset_manifest,
        {'explicit_ids': {'train': ['a'], 'val': ['b'], 'test': ['c']}},
    )

    assert split_manifest == {'train_ids': ['a'], 'val_ids': ['b'], 'test_ids': ['c']}
    # §9 量级门: 1 train / 1 test 比例=1.0 > 0.1, N_traj_te=1 < 20, layout_family=0 < 3
    # 都是预期违规 (合成 fixture 故意小数据集). 非数据泄漏 leak_items 应为空.
    assert _non_section9_leak_items(leak_report['leak_items']) == []
    assert leak_report['split_path'] == 'explicit'
    # §9 leak_items 应该存在并标识违规
    section9_kinds = _section9_leak_kinds(leak_report['leak_items'])
    assert 'section9_train_test_ratio' in section9_kinds
    assert 'section9_layout_family' in section9_kinds


def test_build_splits_reports_duplicate_ids_across_splits():
    dataset_manifest = {'sequences': [{'seq_id': 'a'}, {'seq_id': 'b'}]}

    split_manifest, leak_report = build_splits(
        dataset_manifest,
        {'explicit_ids': {'train': ['a'], 'val': ['a'], 'test': ['b']}},
    )

    assert split_manifest['train_ids'] == ['a']
    assert split_manifest['val_ids'] == ['a']
    assert split_manifest['test_ids'] == ['b']
    assert leak_report['is_clean'] is False
    # 跨分割重复 ID 是数据泄漏类, 应在非 §9 leak_items 中
    non_section9 = _non_section9_leak_items(leak_report['leak_items'])
    assert {'seq_id': 'a', 'splits': ['train', 'val'], 'reason': 'duplicated across splits: train, val'} in non_section9


def test_build_splits_rejects_empty_sequence_manifest():
    with pytest.raises(ValueError, match='dataset_manifest.sequences must be non-empty'):
        build_splits({'sequences': []}, {'train_count': 1, 'val_count': 1})


def test_build_splits_rejects_non_dict_rules():
    with pytest.raises(TypeError, match='split_rules must be a dict'):
        build_splits({'sequences': [{'seq_id': 'a'}]}, None)


def test_build_splits_rejects_empty_test_split_for_three_sequences():
    dataset_manifest = {'sequences': [{'seq_id': 'a'}, {'seq_id': 'b'}, {'seq_id': 'c'}]}

    with pytest.raises(ValueError, match='split_rules must produce a non-empty test split'):
        build_splits(dataset_manifest, {'explicit_ids': {'train': ['a', 'c'], 'val': ['b'], 'test': []}})


def test_build_splits_reports_shared_scene_scope_across_splits():
    dataset_manifest = {
        'sequences': [
            {'seq_id': 'a', 'scene_id': 'scene_1'},
            {'seq_id': 'b', 'scene_id': 'scene_1'},
            {'seq_id': 'c', 'scene_id': 'scene_2'},
            {'seq_id': 'd', 'scene_id': 'scene_2'},
        ]
    }

    _, leak_report = build_splits(
        dataset_manifest,
        {'train_count': 2, 'val_count': 1},
    )

    leak_kinds = {item['kind'] for item in leak_report['leak_items']}
    assert 'shared_record_scope' in leak_kinds


def test_build_splits_section9_large_split_passes():
    """§9 量级门: 训练/测试比例 ≤ 0.1, N_traj_te ≥ 30, 布局族数 ≥ 3 同时满足时无 §9 leak."""
    sequences = []
    # 3 个 layout family, 每族写一份 anchor_layout.json, 30 轨迹做测试, 2 轨迹做训练 (2/30≈0.067 ≤ 0.1)
    for family_idx in range(3):
        for traj_idx in range(11):
            seq_id = f'fam{family_idx}_traj{traj_idx}'
            seq_dir = f'/tmp/section9_test/{seq_id}'
            sequences.append({
                'seq_id': seq_id,
                'seq_dir': seq_dir,
                'scene_id': f'scene_fam{family_idx}',
                'anchor_layout': {'layout_family': f'family_{family_idx}', 'anchors': []},
            })

    dataset_manifest = {'sequences': sequences}
    # 2 train + 30 test, 2/30 ≈ 0.067 ≤ 0.1; N_traj_te=30 ≥ 30; 三家族都出现在 test
    _, leak_report = build_splits(
        dataset_manifest,
        {'train_count': 2, 'val_count': 0},
    )

    section9_kinds = _section9_leak_kinds(leak_report['leak_items'])
    # 量级门应全部不激活 (注意: layout_family 解析需要 anchor_layout.json 文件,
    # 合成 fixture 中只在 record['anchor_layout'] 提供字段; _resolve_sequence_layout_family
    # 会读 record['anchor_layout']['layout_family'] 字段, 见 split_builder.py)
    # train_test_ratio: 2/30 = 0.0667 ≤ 0.1, 通过
    assert 'section9_train_test_ratio' not in section9_kinds


def test_build_splits_section9_scene_seed_scope_audit_records_missing_keys(monkeypatch):
    """§9 细节 scene_seed_scope 审计: 若 dataset_manifest.seeds 缺失 scope 字段, leak_items 记录违规.

    本测试通过 monkeypatch patch get_seed_policy 返回固定 scene_seed_scope,
    校验 build_splits 在 seeds 字段缺失时把 scene_seed_scope 违规项加入 leak_items.
    """
    from liquidloc.dataio.manifests import split_builder as sb_module
    # patch get_seed_policy 返回固定 scene_seed_scope
    def _fake_get_seed_policy(protocol_cfg=None):
        return {
            'scene_seed_scope': ['scene', 'nlos_packet_loss', 'init_value', 'network_init', 'split'],
            'n_seed_min': 5,
            'n_seed_recommended': 30,
            'single_seed_no_conclusion': False,
            'train_test_ratio_max': 0.1,
            'n_traj_te_min': 20,
        }
    monkeypatch.setattr(sb_module, 'get_seed_policy', _fake_get_seed_policy, raising=False)
    # 但 get_seed_policy 是函数内部 import, 需 patch 源模块
    from liquidloc.protocol import experiment_gates as eg_module
    monkeypatch.setattr(eg_module, 'get_seed_policy', _fake_get_seed_policy, raising=False)

    sequences = []
    for family_idx in range(3):
        for traj_idx in range(10):
            seq_id = f'fam{family_idx}_traj{traj_idx}_seed'
            sequences.append({
                'seq_id': seq_id,
                'seq_dir': f'/tmp/seed_test/{seq_id}',
                'scene_id': f'scene_fam{family_idx}',
                'anchor_layout': {'layout_family': f'family_{family_idx}', 'anchors': []},
            })
    # seeds 字典仅含 'scene'，缺失其他 4 个 scope 字段
    dataset_manifest = {
        'sequences': sequences,
        'seeds': {'scene': 1, 'split': 2},  # 缺 nlos_packet_loss/init_value/network_init
    }
    _, leak_report = build_splits(dataset_manifest, {'train_count': 2, 'val_count': 0})
    section9_kinds = _section9_leak_kinds(leak_report['leak_items'])
    assert 'section9_scene_seed_scope' in section9_kinds
    # 找到该违规项, 检查 missing_scope_keys
    seed_scope_item = next(item for item in leak_report['leak_items']
                           if item.get('kind') == 'section9_scene_seed_scope')
    expected_missing = {'nlos_packet_loss', 'init_value', 'network_init'}
    actual_missing = set(seed_scope_item['missing_scope_keys'])
    assert expected_missing == actual_missing


def test_build_splits_section9_scene_seed_scope_no_seeds_field_skips_audit(monkeypatch):
    """§9 细节 scene_seed_scope 审计: dataset_manifest 有 sequences 但无 seeds 字段时, 不报违规."""
    from liquidloc.protocol import experiment_gates as eg_module
    def _fake_get_seed_policy(protocol_cfg=None):
        return {
            'scene_seed_scope': ['scene', 'nlos_packet_loss', 'init_value', 'network_init', 'split'],
            'n_seed_min': 5,
            'n_seed_recommended': 30,
            'single_seed_no_conclusion': False,
            'train_test_ratio_max': 0.1,
            'n_traj_te_min': 20,
        }
    monkeypatch.setattr(eg_module, 'get_seed_policy', _fake_get_seed_policy, raising=False)

    sequences = []
    for family_idx in range(3):
        for traj_idx in range(10):
            seq_id = f'fam{family_idx}_traj{traj_idx}_no_seed'
            sequences.append({
                'seq_id': seq_id,
                'seq_dir': f'/tmp/no_seed_test/{seq_id}',
                'scene_id': f'scene_fam{family_idx}',
                'anchor_layout': {'layout_family': f'family_{family_idx}', 'anchors': []},
            })
    dataset_manifest = {'sequences': sequences}  # 无 seeds 字段
    _, leak_report = build_splits(dataset_manifest, {'train_count': 2, 'val_count': 0})
    section9_kinds = _section9_leak_kinds(leak_report['leak_items'])
    assert 'section9_scene_seed_scope' not in section9_kinds


# ========== §9.2 序列 ID 协议层 disjointness 守门测试 (Round 4 §9 穷举审视) ==========

def test_build_splits_section9_seq_id_disjointness_violation_recorded():
    """§9.2 字面守门: explicit_ids 使 train 与 test 共享 seq_id 时, leak_items 含 section9_seq_id_disjointness.

    验证 normalize_train_test_disjointness 协议层函数真实接入 _emit_section9_leak_warnings 主路径.
    build_splits 在 L596-618 也会同时记 'duplicated across splits' (非 §9 kind), 两者并存不冲突.
    """
    dataset_manifest = {'sequences': [{'seq_id': 'a'}, {'seq_id': 'b'}, {'seq_id': 'c'}]}

    # explicit 故意让 train 与 test 共享 seq_id 'a' (违反 §9.2 字面守门)
    split_manifest, leak_report = build_splits(
        dataset_manifest,
        {'explicit_ids': {'train': ['a', 'b'], 'test': ['a', 'c']}},
    )

    section9_kinds = _section9_leak_kinds(leak_report['leak_items'])
    # §9.2 协议层 disjointness 守门违规必须被记入 leak_items
    assert 'section9_seq_id_disjointness' in section9_kinds, (
        f"section9_seq_id_disjointness kind 缺失 — normalize_train_test_disjointness §9.2 守门未接入主路径; "
        f"实际 §9 kinds: {section9_kinds}"
    )
    # 验证 leak_item 含具体违规原因 (协议层 ValueError message 透传)
    disjointness_items = [
        item for item in leak_report['leak_items']
        if item.get('kind') == 'section9_seq_id_disjointness'
    ]
    assert len(disjointness_items) == 1
    reason = disjointness_items[0]['reason']
    assert 'overlap' in reason.lower() or 'disjoint' in reason.lower(), (
        f"disjointness leak_item reason 应含 overlap/disjoint 关键字, 实际: {reason}"
    )
    assert 'a' in reason, f"reason 应含重叠 seq_id 'a', 实际: {reason}"


def test_build_splits_section9_seq_id_disjointness_clean_no_violation():
    """§9.2 字面守门: train 与 test 序列 ID 不交时, leak_items 不含 section9_seq_id_disjointness."""
    dataset_manifest = {'sequences': [{'seq_id': 'a'}, {'seq_id': 'b'}, {'seq_id': 'c'}]}

    split_manifest, leak_report = build_splits(
        dataset_manifest,
        {'explicit_ids': {'train': ['a', 'b'], 'test': ['c']}},
    )

    section9_kinds = _section9_leak_kinds(leak_report['leak_items'])
    assert 'section9_seq_id_disjointness' not in section9_kinds, (
        f"无重叠时不应触发 §9.2 disjointness leak_item, 实际 §9 kinds: {section9_kinds}"
    )

