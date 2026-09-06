
"""公开数据集注册表（public_dataset_registry）测试模块。

文件职责：验证公开数据集注册表的加载、查询、归一化
和就绪状态检查功能。

测试覆盖范围：
- 加载注册表并获取条目
- 数据集名称大小写归一化
- 默认发现路径
- 缺失数据集抛出 KeyError
- util 数据集注册为公开数据集
- 列出所有公开数据集
- 就绪状态保持 smoke 范围
- 非官方基准面数据集标记为 not_ready

被测模块：liquidloc.dataio.registry.public_dataset_registry"""

from pathlib import Path

from liquidloc.dataio.registry import public_dataset_registry as registry_mod
from liquidloc.dataio.registry.public_dataset_registry import (
    get_dataset_entry,
    inspect_public_dataset_readiness,
    load_public_dataset_registry,
    normalize_public_dataset_name,
)


def test_load_registry_and_get_entry():
    registry = load_public_dataset_registry(Path(__file__).resolve().parents[2] / 'configs' / 'datasets' / 'public_dataset_registry.yaml')
    entry = get_dataset_entry('miluv', registry)
    assert entry['dataset_name'] == 'miluv'
    assert entry['reader'].endswith('miluv_reader.py')
    ntu_entry = get_dataset_entry('ntu_viral', registry)
    assert ntu_entry['dataset_name'] == 'ntu_viral'
    assert ntu_entry['reader'].endswith('ntu_viral_reader.py')
    assert ntu_entry['prepare_script'] == 'scripts/03_prepare_ntu_viral_data.py'


def test_get_dataset_entry_normalizes_case():
    registry = load_public_dataset_registry(Path(__file__).resolve().parents[2] / 'configs' / 'datasets' / 'public_dataset_registry.yaml')
    entry = get_dataset_entry('MILUV', registry)
    assert entry['dataset_name'] == 'miluv'
    assert entry['reader'].endswith('miluv_reader.py')


def test_normalize_public_dataset_name_lowercases_and_strips():
    assert normalize_public_dataset_name('  NTU_VIRAL  ') == 'ntu_viral'


def test_load_registry_uses_default_discovery_path(monkeypatch):
    project_root = Path('C:/fake/project-root')
    captured = {}

    def fake_get_project_root():
        return project_root

    def fake_load_yaml_config(registry_path):
        captured['registry_path'] = registry_path
        return {'datasets': {'miluv': {'reader': 'x', 'useful_assets': ['a.json'], 'release_tier': 'smoke_only', 'frozen_public_eval_seq_ids': ['seq1']}}}

    monkeypatch.setattr(registry_mod, 'get_project_root', fake_get_project_root)
    monkeypatch.setattr(registry_mod, 'load_yaml_config', fake_load_yaml_config)

    registry = load_public_dataset_registry(None)

    assert captured['registry_path'] == project_root / 'configs' / 'datasets' / 'public_dataset_registry.yaml'
    assert registry['datasets']['miluv']['reader'] == 'x'


def test_missing_dataset_raises():
    registry = {'datasets': {'miluv': {'reader': 'x'}}}
    try:
        get_dataset_entry('util', registry)
    except KeyError as exc:
        assert 'Unknown public dataset' in str(exc)
    else:
        raise AssertionError('Expected missing dataset to raise KeyError')


def test_util_dataset_is_registered_as_public_dataset():
    registry = load_public_dataset_registry(Path(__file__).resolve().parents[2] / 'configs' / 'datasets' / 'public_dataset_registry.yaml')
    assert 'util' in registry['datasets']


def test_registry_lists_public_datasets():
    registry = load_public_dataset_registry(Path(__file__).resolve().parents[2] / 'configs' / 'datasets' / 'public_dataset_registry.yaml')
    # 注册表 2026-09 扩展：除官方公开面（miluv/ntu_viral/util）外，还包含
    # 补充路由入口 sim 与 sim_e9_protocol_20260726（协议级仿真数据集，非官方公开
    # benchmark 面，仅用于 §6/§8 主表与 §12 消融的仿真路由）。
    assert registry_mod.list_public_dataset_names(registry) == [
        'miluv', 'ntu_viral', 'sim', 'sim_e9_protocol_20260726', 'util',
    ]


def test_public_dataset_readiness_stays_smoke_scoped():
    registry = {
        'datasets': {
            'miluv': {'reader': 'x', 'release_tier': 'smoke_only', 'paper_full_ready': False},
            'ntu_viral': {'reader': 'y', 'release_tier': 'full_ready', 'paper_full_ready': True, 'public_benchmark_smoke_ready': False},
        }
    }

    miluv_ready = inspect_public_dataset_readiness('miluv', registry)
    ntu_ready = inspect_public_dataset_readiness('ntu_viral', registry)

    assert miluv_ready['public_benchmark_smoke_ready'] is True
    assert miluv_ready['paper_full_ready'] is False
    assert ntu_ready['public_benchmark_smoke_ready'] is False
    assert ntu_ready['paper_full_ready'] is True


def test_registered_dataset_outside_official_public_benchmark_surface_stays_not_ready():
    registry = {
        'datasets': {
            'util': {
                'reader': 'x',
                'release_tier': 'smoke_only',
                'paper_full_ready': False,
                'public_benchmark_smoke_ready': True,
            },
        }
    }

    util_ready = inspect_public_dataset_readiness('util', registry)

    assert util_ready['public_benchmark_smoke_ready'] is False
    assert util_ready['status'] == 'not_ready'
    assert util_ready['reasons'] == ['dataset_outside_official_public_benchmark_surface']
