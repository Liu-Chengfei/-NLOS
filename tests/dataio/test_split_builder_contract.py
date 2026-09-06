
"""分割构建器合同（split_builder contract）测试模块。

文件职责：验证 build_splits 生成的分割满足
互斥性和验证分割保留等合同要求。

测试覆盖范围：
- 3 序列生成互斥的 train/val/test 分割
- 2 序列保留验证分割
- §9 量级门泄漏项在小型合成数据集上的预期行为

被测模块：liquidloc.dataio.manifests.split_builder"""

from liquidloc.dataio.manifests.split_builder import build_splits


def _non_section9_leak_items(leak_items: list[dict]) -> list[dict]:
    """过滤掉 §9 量级门 leak_items, 只保留数据泄漏类条目."""
    return [item for item in leak_items if not str(item.get('kind', '')).startswith('section9_')]


def test_build_splits_generates_disjoint_sets():
    dataset_manifest = {
        'sequences': [
            {'seq_id': 'a'},
            {'seq_id': 'b'},
            {'seq_id': 'c'},
        ]
    }
    split_manifest, leak_report = build_splits(dataset_manifest, {'train_count': 1, 'val_count': 1})
    assert split_manifest['train_ids'] == ['a']
    assert split_manifest['val_ids'] == ['b']
    assert split_manifest['test_ids'] == ['c']
    # §9 量级门: 1 train / 1 test = 1.0 > 0.1, N_traj_te=1 < 20, layout_family=0 < 3
    # 合成 fixture 故意小数据集, §9 违规是预期行为.
    # 数据泄漏类条目应为空 (无跨分割重复或共享场景作用域).
    assert _non_section9_leak_items(leak_report['leak_items']) == []


def test_build_splits_preserves_validation_split_for_two_sequences():
    dataset_manifest = {
        'sequences': [
            {'seq_id': 'a'},
            {'seq_id': 'b'},
        ]
    }
    split_manifest, leak_report = build_splits(dataset_manifest, {'train_count': 3, 'val_count': 2})
    assert split_manifest['train_ids'] == ['a']
    assert split_manifest['val_ids'] == ['b']
    assert split_manifest['test_ids'] == []
    # §9 量级门: 1 train / 0 test → ratio=inf > 0.1, N_traj_te=0 < 20
    # 数据泄漏类条目应为空.
    assert _non_section9_leak_items(leak_report['leak_items']) == []
