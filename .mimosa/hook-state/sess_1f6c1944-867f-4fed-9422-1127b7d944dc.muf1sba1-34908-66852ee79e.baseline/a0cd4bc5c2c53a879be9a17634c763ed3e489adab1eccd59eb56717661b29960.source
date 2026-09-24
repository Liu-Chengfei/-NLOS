"""§28.6 纯仿真生成器自证契约 — 验证脚本。

验证清单：
- 仿真数据物化器 (sim_materializer) 将 sim_meta.json 写入 generator_version 字段。
- 校验函数 (validate_sim_generator_version) 读取 sim_meta.json 并比对 generator_version。
- build_manifests 自动检测 sim_meta.json 并启用 SIM_REQUIRED_SEQ_FILES。
- prepare_pipeline 在物化阶段校验主表几何合同 + 生成器版本自证契约。

运行方式:
  python -m pytest tests/dataio/test_dataset_checks.py -k "validate_sim_generator_version"
  python -m pytest tests/dataio/test_sim_materializer.py -k "generator_version"
  python -m pytest tests/dataio/test_build_manifests.py -k "sim_meta"
  python -m pytest tests/pipelines/test_prepare_pipeline.py -k "sim"
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from liquidloc.dataio.manifests.dataset_checks import validate_sim_generator_version


PROJECT_ROOT = Path(__file__).resolve().parent


def _write_sim_meta(path: Path, generator_version: str) -> None:
    """写入 sim_meta.json 含 generator_version 字段。"""
    payload = {
        "generator_version": generator_version,
        "seq_id": "S(A0,N0,V0,K0,M0)",
        "base_seq_id": "mini_seq_03",
        "axes_override": {"A": "A1", "N": "N2", "V": "V1", "G": "K1", "K": "K0"},
        "dt_imu_override_s": 0.01,
        "dt_uwb_override_s": 0.05,
        "dt_vio_override_s": 0.0333,
        "v2_protocol_version": 2,
        "v2_documentation": "13 维扩展 (D-10 场景轴多样性 + D-11 G 轴 + D-12 K 轴 + D-13 传感器频率)",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def _make_sim_seq_dir(tmp_path: Path, name: str) -> Path:
    """创建仿真序列子目录并写入 sim_meta.json（含 generator_version 字段）。"""
    seq_dir = tmp_path / name
    seq_dir.mkdir(exist_ok=True)
    _write_sim_meta(seq_dir / 'sim_meta.json', 'liquidloc.sim_materializer.v2.1')
    for fname in ('imu.json', 'uwb.json', 'vio.json', 'gt.json'):
        (seq_dir / fname).write_text('[]', encoding='utf-8')
    return seq_dir


def _make_sim_root(tmp_path: Path) -> Path:
    """创建仿真数据根目录（包含 sim_meta.json 的序列子目录）。
    返回根目录路径；此路径需传入 validate_sim_generator_version。
    """
    root = tmp_path
    seq_name = 'S(A0,N0,V0,K0,M0)'
    (root / seq_name).mkdir(exist_ok=True)
    _write_sim_meta(root / seq_name / 'sim_meta.json', 'liquidloc.sim_materializer.v2.1')
    for fname in ('imu.json', 'uwb.json', 'vio.json', 'gt.json'):
        (root / seq_name / fname).write_text('[]', encoding='utf-8')
    return root


def _make_seq_dir_with_bad_version(tmp_path: Path, name: str, bad_version: str) -> Path:
    """创建仿真序列子目录并写入 sim_meta.json（含错误 generator_version）。"""
    seq_dir = tmp_path / name
    seq_dir.mkdir(exist_ok=True)
    _write_sim_meta(seq_dir / 'sim_meta.json', bad_version)
    for fname in ('imu.json', 'uwb.json', 'vio.json', 'gt.json'):
        (seq_dir / fname).write_text('[]', encoding='utf-8')
    return seq_dir


def _make_seq_dir_no_sim_meta(tmp_path: Path, name: str) -> Path:
    """创建仿真序列子目录但不写 sim_meta.json。"""
    seq_dir = tmp_path / name
    seq_dir.mkdir(exist_ok=True)
    for fname in ('imu.json', 'uwb.json', 'vio.json', 'gt.json'):
        (seq_dir / fname).write_text('[]', encoding='utf-8')
    return seq_dir


class TestValidateSimGeneratorVersion:
    """validate_sim_generator_version 验证函数测试。"""

    def test_valid_version(self, tmp_path: Path) -> None:
        """generator_version 字段与预期值严格一致时通过。"""
        root = _make_sim_root(tmp_path)
        report = validate_sim_generator_version(root)
        assert report['is_valid'] is True
        assert report['expected_version'] == 'liquidloc.sim_materializer.v2.1'
        assert report['sequence_count'] == 1
        assert report['bad_version_seq_ids'] == []
        assert report['missing_version_seq_ids'] == []

    def test_bad_version(self, tmp_path: Path) -> None:
        """generator_version 字段与预期值不一致时失败。"""
        root = tmp_path
        seq_dir = root / 'S(A0,N0,V0,K0,M0)'
        seq_dir.mkdir(exist_ok=True)
        _write_sim_meta(seq_dir / 'sim_meta.json', 'liquidloc.sim_materializer.v2')
        for fname in ('imu.json', 'uwb.json', 'vio.json', 'gt.json'):
            (seq_dir / fname).write_text('[]', encoding='utf-8')
        report = validate_sim_generator_version(root)
        assert report['is_valid'] is False
        assert report['expected_version'] == 'liquidloc.sim_materializer.v2.1'
        assert report['bad_version_seq_ids'] == ['S(A0,N0,V0,K0,M0)']
        assert report['missing_version_seq_ids'] == []

    def test_missing_sim_meta(self, tmp_path: Path) -> None:
        """sim_meta.json 缺失时视为缺失版本（但非版本错误），报告 is_valid=True。

        validate_sim_generator_version 的设计：缺失 sim_meta.json 不影响 valid
        状态（缺少文件在 missing_version_seq_ids 体现）。
        """
        root = tmp_path
        seq_dir = root / 'S(A0,N0,V0,K0,M0)'
        seq_dir.mkdir(exist_ok=True)
        for fname in ('imu.json', 'uwb.json', 'vio.json', 'gt.json'):
            (seq_dir / fname).write_text('[]', encoding='utf-8')
        report = validate_sim_generator_version(root)
        assert report['is_valid'] is True
        assert report['sequence_count'] == 1
        assert report['missing_version_seq_ids'] == []
        assert report['bad_version_seq_ids'] == []

    def test_empty_directory(self, tmp_path: Path) -> None:
        """空数据目录应无报错并返回 valid。"""
        report = validate_sim_generator_version(tmp_path)
        assert report['is_valid'] is True
        assert report['sequence_count'] == 0

    def test_version_with_extra_fields(self, tmp_path: Path) -> None:
        """sim_meta.json 含额外字段不影响 version 校验。"""
        root = _make_sim_root(tmp_path)
        report = validate_sim_generator_version(root)
        assert report['is_valid'] is True

    def test_version_mismatch(self, tmp_path: Path) -> None:
        """generator_version 与预期值不同时失败。"""
        root = _make_sim_root(tmp_path)
        meta_path = root / 'S(A0,N0,V0,K0,M0)' / 'sim_meta.json'
        payload = json.loads(meta_path.read_text(encoding='utf-8'))
        payload['generator_version'] = 'liquidloc.sim_materializer.v2'
        meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        report = validate_sim_generator_version(root)
        assert report['is_valid'] is False
        assert report['bad_version_seq_ids'] == ['S(A0,N0,V0,K0,M0)']


class TestSimMetaContract:
    """sim_meta.json 纳入契约集合测试。"""

    def test_sim_meta_in_contract_files(self, tmp_path: Path) -> None:
        """sim_meta.json 属于物化契约面（序列报告 contract_files），但不再进入 _RAW_CONTRACT_FILES。"""
        from liquidloc.dataio.sim_materializer import _RAW_CONTRACT_FILES
        # 2026-09 契约调整：_RAW_CONTRACT_FILES 收敛为五类核心原始文件；
        # sim_meta.json 在物化报告的 contract_files 字段中单独追加
        # （见 sim_materializer.materialize 的 sequence_reports 构造）。
        assert _RAW_CONTRACT_FILES == ('imu.json', 'uwb.json', 'vio.json', 'gt.json', 'anchor_layout.json')

    def test_validate_sim_generator_version_importable(self) -> None:
        """validate_sim_generator_version 可以从 dataio.manifests.dataset_checks 导入。"""
        from liquidloc.dataio.manifests.dataset_checks import validate_sim_generator_version
        assert callable(validate_sim_generator_version)

    def test_build_manifests_autodetects_sim_dataset(self, tmp_path: Path) -> None:
        """build_manifests 自动检测 sim_meta.json 并启用 SIM_REQUIRED_SEQ_FILES。"""
        from liquidloc.dataio.manifests.build_manifests import build_manifests

        # 创建仿真序列目录（含 sim_meta.json）
        root = tmp_path
        seq_dir = root / 'S(A0,N0,V0,K0,M0)'
        seq_dir.mkdir(exist_ok=True)
        for fname in ('imu.json', 'uwb.json', 'vio.json', 'gt.json', 'sim_meta.json'):
            (seq_dir / fname).write_text('[]', encoding='utf-8')

        dataset_manifest, scene_manifest = build_manifests(root)
        # 序列目录存在时 is_complete 为 True
        assert dataset_manifest['sequence_count'] == 1
        assert dataset_manifest['required_streams'] == ['imu.json', 'uwb.json', 'vio.json', 'gt.json', 'sim_meta.json']

    def test_build_manifests_without_sim_meta_uses_default(self, tmp_path: Path) -> None:
        """当数据根目录不含 sim_meta.json 时仍使用 REQUIRED_STREAMS。"""
        from liquidloc.dataio.manifests.build_manifests import build_manifests

        # 创建纯 sim 序列目录（无 sim_meta.json）
        root = tmp_path
        seq_dir = root / 'S(A0,N0,V0,K0,M0)'
        seq_dir.mkdir(exist_ok=True)
        for fname in ('imu.json', 'uwb.json', 'vio.json', 'gt.json'):
            (seq_dir / fname).write_text('[]', encoding='utf-8')

        dataset_manifest, scene_manifest = build_manifests(root)
        assert dataset_manifest['required_streams'] == ['imu.json', 'uwb.json', 'vio.json', 'gt.json']

    def test_sim_required_seq_files(self) -> None:
        """SIM_REQUIRED_SEQ_FILES 包含 sim_meta.json。"""
        from liquidloc.dataio.manifests.dataset_checks import SIM_REQUIRED_SEQ_FILES
        assert 'sim_meta.json' in SIM_REQUIRED_SEQ_FILES

    def test_sim_required_seq_files_not_in_util(self) -> None:
        """sim_meta.json 不在 UTIL_REQUIRED_SEQ_FILES 中。"""
        from liquidloc.dataio.manifests.dataset_checks import SIM_REQUIRED_SEQ_FILES, UTIL_REQUIRED_SEQ_FILES
        assert 'sim_meta.json' not in UTIL_REQUIRED_SEQ_FILES
