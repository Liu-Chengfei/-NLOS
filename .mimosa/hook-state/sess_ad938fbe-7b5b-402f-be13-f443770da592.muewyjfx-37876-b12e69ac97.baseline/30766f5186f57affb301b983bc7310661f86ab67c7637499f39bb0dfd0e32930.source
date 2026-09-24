from __future__ import annotations

"""分割构建器语义重叠（split_builder semantic overlap）测试模块。

文件职责：验证 build_splits 能检测跨分割的布局族共享，
并按布局族分组进行默认分割。

测试覆盖范围：
- 跨分割的共享布局族报告
- 按布局族分组的默认分割
- §9 量级门 leak_items 在小型合成数据集上的预期行为

被测模块：liquidloc.dataio.manifests.split_builder"""


import json

from liquidloc.dataio.manifests.split_builder import build_splits


def _non_section9_leak_items(leak_items: list[dict]) -> list[dict]:
    """过滤掉 §9 量级门 leak_items, 只保留数据泄漏类条目."""
    return [item for item in leak_items if not str(item.get('kind', '')).startswith('section9_')]


def test_build_splits_reports_shared_layout_family_across_splits(tmp_path):
    seq_a_dir = tmp_path / 'seq_a'
    seq_b_dir = tmp_path / 'seq_b'
    seq_c_dir = tmp_path / 'seq_c'
    for seq_dir in (seq_a_dir, seq_b_dir, seq_c_dir):
        seq_dir.mkdir()

    (seq_a_dir / 'anchor_layout.json').write_text(json.dumps({'base_layout_id': 'family_alpha'}), encoding='utf-8')
    (seq_b_dir / 'anchor_layout.json').write_text(json.dumps({'base_layout_id': 'family_alpha'}), encoding='utf-8')
    (seq_c_dir / 'anchor_layout.json').write_text(json.dumps({'base_layout_id': 'family_beta'}), encoding='utf-8')

    dataset_manifest = {
        'sequences': [
            {'seq_id': 'a', 'seq_dir': str(seq_a_dir)},
            {'seq_id': 'b', 'seq_dir': str(seq_b_dir)},
            {'seq_id': 'c', 'seq_dir': str(seq_c_dir)},
        ]
    }

    split_manifest, leak_report = build_splits(
        dataset_manifest,
        {'explicit_ids': {'train': ['a'], 'val': ['b'], 'test': ['c']}},
    )

    assert split_manifest == {'train_ids': ['a'], 'val_ids': ['b'], 'test_ids': ['c']}
    assert leak_report['is_clean'] is False
    # §9 量级门: 1 train / 1 test = 1.0 > 0.1, N_traj_te=1 < 20, layout_family=1 < 3
    # 数据泄漏类条目应只含 shared_layout_family 一项 (跨 train/val 共享 family_alpha).
    assert _non_section9_leak_items(leak_report['leak_items']) == [
        {
            'kind': 'shared_layout_family',
            'layout_family': 'family_alpha',
            'seq_ids': ['a', 'b'],
            'splits': ['train', 'val'],
            'reason': 'layout family family_alpha spans multiple splits: train, val',
        }
    ]


def test_build_splits_groups_default_splits_by_layout_family_when_metadata_is_complete(tmp_path):
    family_specs = {
        'family_alpha': ['alpha_1', 'alpha_2'],
        'family_beta': ['beta_1', 'beta_2'],
        'family_gamma': ['gamma_1', 'gamma_2'],
    }
    sequences = []
    for family_id, seq_ids in family_specs.items():
        for seq_id in seq_ids:
            seq_dir = tmp_path / seq_id
            seq_dir.mkdir()
            (seq_dir / 'anchor_layout.json').write_text(json.dumps({'base_layout_id': family_id}), encoding='utf-8')
            sequences.append({'seq_id': seq_id, 'seq_dir': str(seq_dir)})

    split_manifest, leak_report = build_splits(
        {'sequences': sequences},
        {'train_count': 1, 'val_count': 1},
    )

    assert split_manifest == {
        'train_ids': ['alpha_1', 'alpha_2'],
        'val_ids': ['beta_1', 'beta_2'],
        'test_ids': ['gamma_1', 'gamma_2'],
    }
    # §9 量级门: 2 train / 2 test = 1.0 > 0.1, N_traj_te=2 < 20, layout_family=1 < 3 (test 只有 gamma)
    # 数据泄漏类条目应为空 (各分割互斥且无共享布局族).
    assert _non_section9_leak_items(leak_report['leak_items']) == []
    assert leak_report['split_path'] == 'family_grouped'
