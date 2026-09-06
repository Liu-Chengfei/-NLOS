from __future__ import annotations

"""预处理输入（prepared_inputs）测试模块。

文件职责：验证 load_source_report_by_seq_id 能正确加载
源报告并携带锚点布局元数据。

测试覆盖范围：
- 锚点布局元数据正确携带
- 源报告字段完整性

被测模块：liquidloc.common.prepared_inputs"""


import json

from liquidloc.common.prepared_inputs import load_source_report_by_seq_id


def test_load_source_report_by_seq_id_carries_anchor_layout_metadata(tmp_path):
    raw_root = tmp_path / 'raw'
    seq_root = raw_root / 'seq_a'
    seq_root.mkdir(parents=True)
    (seq_root / 'anchor_layout.json').write_text(
        json.dumps(
            {
                'layout_id': 'seq_a_layout',
                'anchor_ids': [0, 1],
                'anchor_positions': [[0.0, 0.0], [1.0, 2.0]],
                'source': 'local_anchor_layout_json',
            }
        ),
        encoding='utf-8',
    )
    prepare_manifest = {
        'sequences': {
            'seq_a': {
                'seq_id': 'seq_a',
                'scene_id': 'scene_a',
            }
        }
    }

    report_by_seq_id = load_source_report_by_seq_id(
        raw_root,
        prepare_manifest,
        ['seq_a'],
        default_source='sim_prepare_bridge',
    )

    report = report_by_seq_id['seq_a']
    assert report['source'] == 'sim_prepare_bridge'
    assert report['prepare_sequence'] == {'seq_id': 'seq_a', 'scene_id': 'scene_a'}
    assert report['anchor_layout']['layout_id'] == 'seq_a_layout'
    assert report['anchor_layout_metadata']['layout_id'] == 'seq_a_layout'
    assert report['anchor_layout_source'] == 'raw_anchor_layout_json'
    assert report['anchor_layout_metadata_source'] == 'raw_anchor_layout_json'
    assert report['anchor_layout_position_dim'] == 2
