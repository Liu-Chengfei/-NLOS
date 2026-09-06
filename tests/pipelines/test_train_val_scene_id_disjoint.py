from __future__ import annotations

"""H24c 真改 §0.2 / §27「scene_id 不可泄漏」cross-set 守门测试模块。

文件职责：验证 _split_train_and_val_samples + _build_split_audit 的 H24c 真改：
- 显式 split_ids 切分下 train/val 不应共享 scene_id（cross-set 泄漏守门，抛 ValueError）
- 单序列时间切分（fall-through）下不守门（同 seq_id 内时间切片允许共享 scene_id）
- _build_split_audit 输出 shared_scene_ids / scene_id_disjoint / train_scene_ids / val_scene_ids 字段
- 缺 scene_id 字段的旧 sample 向后兼容（视为 None 跳过守门）

被测模块：liquidloc.pipelines.train_pipeline._split_train_and_val_samples
        liquidloc.pipelines.train_pipeline._build_split_audit
"""

import pytest

from liquidloc.pipelines.train_pipeline import (
    _split_train_and_val_samples,
    _build_split_audit,
)


def _samples_with_scene_ids(seq_id: str, scene_ids: list[str]) -> list[dict]:
    """构造带 scene_id 字段的 sample 列表，event_time 按 index 递增。"""
    return [
        {
            'seq_id': seq_id,
            'scene_id': sid,
            'event_time': float(i) * 0.1,
            'modality': 'uwb',
            'window_tensor': {},
            'target_intermediate': {},
            'target_trace': {},
            'source_report': {},
        }
        for i, sid in enumerate(scene_ids)
    ]


def test_cross_set_scene_id_overlap_rejected_explicit_split():
    """显式 train_split_ids / val_split_ids 切分下 train/val 共享 scene_id 必须抛 ValueError。"""
    # seq_alpha 全部 scene_id=S(A0,N1,V0,K0,M0)
    # seq_beta  全部 scene_id=S(A0,N1,V0,K0,M0)  ← 与 seq_alpha 共享
    samples = _samples_with_scene_ids(
        'seq_alpha', ['S(A0,N1,V0,K0,M0)'] * 3
    ) + _samples_with_scene_ids(
        'seq_beta', ['S(A0,N1,V0,K0,M0)'] * 3
    )
    with pytest.raises(ValueError, match=r"cross-set leakage"):
        _split_train_and_val_samples(
            samples,
            split_ids=['seq_alpha', 'seq_beta'],
            train_split_ids=['seq_alpha'],
            val_split_ids=['seq_beta'],
        )


def test_cross_set_scene_id_disjoint_accepted_explicit_split():
    """train/val 各自独占 scene_id 必须通过守门，不抛错。"""
    samples = _samples_with_scene_ids(
        'seq_alpha', ['S(A0,N1,V0,K0,M0)'] * 3
    ) + _samples_with_scene_ids(
        'seq_beta', ['S(A0,N2,V0,K0,M0)'] * 3  # 不同 N_level
    )
    train_samples, val_samples, train_ids, val_ids = _split_train_and_val_samples(
        samples,
        split_ids=['seq_alpha', 'seq_beta'],
        train_split_ids=['seq_alpha'],
        val_split_ids=['seq_beta'],
    )
    assert train_ids == ['seq_alpha']
    assert val_ids == ['seq_beta']
    train_sids = {s['scene_id'] for s in train_samples}
    val_sids = {s['scene_id'] for s in val_samples}
    assert train_sids == {'S(A0,N1,V0,K0,M0)'}
    assert val_sids == {'S(A0,N2,V0,K0,M0)'}
    assert train_sids & val_sids == set()


def test_single_sequence_time_split_does_not_gate_scene_id():
    """单序列时间切分（fall-through）下不守门 scene_id（同 seq_id 内允许共享 scene_id）。"""
    samples = _samples_with_scene_ids('seq_only', ['S(A0,N1,V0,K0,M0)'] * 6)
    # 没传 train_split_ids / val_split_ids → 走 single_sequence_time_split fall-through
    train_samples, val_samples, train_ids, val_ids = _split_train_and_val_samples(
        samples,
        split_ids=['seq_only'],
    )
    # 单序列 fall-through 不应抛错，且 train/val 都来自同一 seq_only
    assert len(train_samples) + len(val_samples) == len(samples)
    assert train_ids == ['seq_only']
    assert val_ids == ['seq_only']


def test_build_split_audit_emits_scene_id_disjoint_fields():
    """_build_split_audit 必须输出 shared_scene_ids / scene_id_disjoint / train_scene_ids / val_scene_ids。"""
    train_samples = _samples_with_scene_ids('seq_a', ['S(A0,N1,V0,K0,M0)'] * 3)
    val_samples = _samples_with_scene_ids('seq_b', ['S(A0,N2,V0,K0,M0)'] * 3)
    audit = _build_split_audit(
        split_ids=['seq_a', 'seq_b'],
        requested_train_ids=['seq_a'],
        requested_val_ids=['seq_b'],
        resolved_train_ids=['seq_a'],
        resolved_val_ids=['seq_b'],
        train_samples=train_samples,
        val_samples=val_samples,
    )
    assert audit['split_strategy'] == 'explicit_sequence_split'
    assert audit['shared_scene_ids'] == []
    assert audit['scene_id_disjoint'] is True
    assert audit['train_scene_ids'] == ['S(A0,N1,V0,K0,M0)']
    assert audit['val_scene_ids'] == ['S(A0,N2,V0,K0,M0)']


def test_build_split_audit_flags_shared_scene_id():
    """train/val 共享 scene_id 时 audit.scene_id_disjoint 必须为 False。"""
    train_samples = _samples_with_scene_ids('seq_a', ['S(A0,N1,V0,K0,M0)'] * 3)
    val_samples = _samples_with_scene_ids('seq_b', ['S(A0,N1,V0,K0,M0)'] * 3)  # 共享 scene_id
    audit = _build_split_audit(
        split_ids=['seq_a', 'seq_b'],
        requested_train_ids=['seq_a'],
        requested_val_ids=['seq_b'],
        resolved_train_ids=['seq_a'],
        resolved_val_ids=['seq_b'],
        train_samples=train_samples,
        val_samples=val_samples,
    )
    assert audit['shared_scene_ids'] == ['S(A0,N1,V0,K0,M0)']
    assert audit['scene_id_disjoint'] is False


def test_build_split_audit_handles_missing_scene_id_field():
    """旧 sample 缺 scene_id 字段时 audit 视为 None 跳过守门（向后兼容）。"""
    train_samples = [{'seq_id': 'seq_a', 'event_time': 0.0}, {'seq_id': 'seq_a', 'event_time': 1.0}]
    val_samples = [{'seq_id': 'seq_b', 'event_time': 0.0}, {'seq_id': 'seq_b', 'event_time': 1.0}]
    audit = _build_split_audit(
        split_ids=['seq_a', 'seq_b'],
        requested_train_ids=['seq_a'],
        requested_val_ids=['seq_b'],
        resolved_train_ids=['seq_a'],
        resolved_val_ids=['seq_b'],
        train_samples=train_samples,
        val_samples=val_samples,
    )
    assert audit['shared_scene_ids'] == []
    assert audit['scene_id_disjoint'] is True
    assert audit['train_scene_ids'] == []
    assert audit['val_scene_ids'] == []


def test_cross_set_overlap_with_mixed_scene_ids_reports_correct_shared_set():
    """train/val 各含多 scene_id 时仅交集共享值触发守门 ValueError。

    构造：
    - train (seq_a): scene_id ∈ {S_A, S_B}
    - val   (seq_b): scene_id ∈ {S_B, S_C}
    shared = {S_B}，应抛 ValueError 含 S_B。
    """
    samples = _samples_with_scene_ids('seq_a', ['S(A0,N1,V0,K0,M0)', 'S(A0,N2,V0,K0,M0)']) + \
              _samples_with_scene_ids('seq_b', ['S(A0,N2,V0,K0,M0)', 'S(A0,N3,V0,K0,M0)'])
    with pytest.raises(ValueError) as exc_info:
        _split_train_and_val_samples(
            samples,
            split_ids=['seq_a', 'seq_b'],
            train_split_ids=['seq_a'],
            val_split_ids=['seq_b'],
        )
    msg = str(exc_info.value)
    assert 'S(A0,N2,V0,K0,M0)' in msg, f"ValueError 应含共享 scene_id S(A0,N2,V0,K0,M0)：{msg}"
