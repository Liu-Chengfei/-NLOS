from __future__ import annotations

"""数据根路径锚定（data_root_path_anchoring）测试模块。

测试覆盖范围：
- 数据根路径的解析与锚定
- 相对路径与绝对路径的处理
- 项目根目录的确定

被测模块：scripts.data_root_path_anchoring"""

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_module(script_name: str, alias: str):
    path = ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_manifests_anchors_repo_dataset_config_to_repo_root(tmp_path, monkeypatch):
    """清单测试：build。\n\n验证 build 的清单生成，\n确保场景参数和序列信息被正确持久化。
    """
    module = _load_module("01_build_manifests.py", "build_manifests_anchor_repo")
    repo_root = tmp_path / "repo"
    cfg_path = repo_root / "configs" / "datasets" / "miluv.yaml"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text("raw_root: data/raw/miluv\n", encoding="utf-8")
    data_root = repo_root / "data" / "raw" / "miluv"
    data_root.mkdir(parents=True)

    captured = {}

    def _fake_load_yaml_config(path):
        assert path == cfg_path
        return {"raw_root": "data/raw/miluv"}

    def _fake_build_manifests(resolved_root, required_streams=None):
        captured["data_root"] = resolved_root
        return {"data_root": str(resolved_root), "sequence_count": 0, "sequences": []}, {"scene_count": 0, "scenes": []}

    monkeypatch.setattr(module, "ROOT", repo_root)
    monkeypatch.setattr(module, "load_yaml_config", _fake_load_yaml_config)
    monkeypatch.setattr(module, "build_manifests", _fake_build_manifests)

    output_root = tmp_path / "out"
    assert module.main(["--dataset-config", str(cfg_path), "--output-root", str(output_root)]) == 0

    assert captured["data_root"] == data_root.resolve()


def test_build_manifests_keeps_external_dataset_config_relative_to_config_dir(tmp_path, monkeypatch):
    """保持测试：build manifests。\n\n验证 build manifests 的保持行为，\n确保特定属性在处理过程中不变。
    """
    module = _load_module("01_build_manifests.py", "build_manifests_anchor_external")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    cfg_dir = tmp_path / "external_cfg"
    cfg_dir.mkdir()
    cfg_path = cfg_dir / "miluv.yaml"
    cfg_path.write_text("raw_root: relative/raw\n", encoding="utf-8")
    expected_root = cfg_dir / "relative" / "raw"

    captured = {}

    def _fake_load_yaml_config(path):
        assert path == cfg_path
        return {"raw_root": "relative/raw"}

    def _fake_build_manifests(resolved_root, required_streams=None):
        captured["data_root"] = resolved_root
        return {"data_root": str(resolved_root), "sequence_count": 0, "sequences": []}, {"scene_count": 0, "scenes": []}

    monkeypatch.setattr(module, "ROOT", repo_root)
    monkeypatch.setattr(module, "load_yaml_config", _fake_load_yaml_config)
    monkeypatch.setattr(module, "build_manifests", _fake_build_manifests)

    output_root = tmp_path / "out_external"
    assert module.main(["--dataset-config", str(cfg_path), "--output-root", str(output_root)]) == 0

    assert captured["data_root"] == expected_root.resolve()


def test_prepare_util_anchors_repo_dataset_config_to_repo_root(tmp_path, monkeypatch):
    """UTIL 数据集测试：prepare。\n\n验证 prepare 的 UTIL 数据集准备，\n确保 flow 和 tof 数据被正确处理。
    """
    module = _load_module("17_prepare_util_data.py", "prepare_util_anchor_repo")
    repo_root = tmp_path / "repo"
    cfg_path = repo_root / "configs" / "datasets" / "util.yaml"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text("raw_root: data/raw/util\nfield_mapping:\n  imu: {}\n", encoding="utf-8")
    raw_root = repo_root / "data" / "raw" / "util"
    seq_dir = raw_root / "mini_seq"
    seq_dir.mkdir(parents=True)

    def _fake_load_yaml_config(path):
        assert path == cfg_path
        return {"raw_root": "data/raw/util", "field_mapping": {"imu": {}}}

    def _fake_read_util_sequence(seq_id, resolved_root):
        assert seq_id == "mini_seq"
        assert Path(str(resolved_root)).resolve() == raw_root.resolve()
        # 必须包含 UTIL_REQUIRED_RAW_KEYS 的全部键 (imu_raw/uwb_raw/flow_raw/gt_raw),
        # 且每条流都必须有非空行, 否则 run_dataset_checks 会以 empty_streams 拒绝.
        return ({
            'imu_raw': [{'timestamp': 0.0, 'ax': 0.0, 'ay': 0.0, 'gz': 0.0}],
            'uwb_raw': [{'timestamp': 0.1, 'anchor_id': 1, 'range': 2.0, 'valid': True}],
            'flow_raw': [{'timestamp': 0.2, 'vx': 0.0, 'vy': 0.0}],
            'gt_raw': [{'timestamp': 0.3, 'px': 0.0, 'py': 0.0, 'yaw': 0.0}],
        }, {'is_complete': True})

    def _fake_map_external_fields(raw_bundle, field_mapping):
        assert field_mapping == {"imu": {}}
        # 真实 map_external_fields 会做字段补全 (uwb quality 默认值等); fake 直接
        # 把每条 raw 行补成 internal_bundle 兼容的最小字段集, 不依赖 field_mapping 的细节.
        # build_flow_events 需要 dx/dy/quality; build_uwb_events 需要 quality.
        internal = {
            'imu_raw': [{'timestamp': r.get('timestamp', 0.0), 'ax': r.get('ax', 0.0), 'ay': r.get('ay', 0.0), 'gz': r.get('gz', 0.0)} for r in raw_bundle.get('imu_raw', [])],
            'uwb_raw': [{**r, 'quality': r.get('quality', 1.0)} for r in raw_bundle.get('uwb_raw', [])],
            'flow_raw': [{'timestamp': r.get('timestamp', 0.0), 'dx': r.get('dx', 0.0), 'dy': r.get('dy', 0.0), 'quality': r.get('quality', 1.0)} for r in raw_bundle.get('flow_raw', [])],
            'gt_raw': list(raw_bundle.get('gt_raw', [])),
        }
        return internal, {'is_complete': True}

    monkeypatch.setattr(module, "ROOT", repo_root)
    monkeypatch.setattr(module, "load_yaml_config", _fake_load_yaml_config)
    monkeypatch.setattr(module, "read_util_sequence", _fake_read_util_sequence)
    monkeypatch.setattr(module, "map_external_fields", _fake_map_external_fields)
    # 脚本最终走 prepare_pipeline.run(), 内部 import 自己的 read_util_sequence/map_external_fields,
    # 必须同步旁路管道层的这两个引用, 否则 fake 不会触发.
    monkeypatch.setattr("liquidloc.pipelines.prepare_pipeline.read_util_sequence", _fake_read_util_sequence)
    monkeypatch.setattr("liquidloc.pipelines.prepare_pipeline.map_external_fields", _fake_map_external_fields)
    # build_manifests 默认会扫盘读取序列目录里的 imu/uwb/flow/gt 文件; 测试只关心路径锚定,
    # 旁路 build_manifests, 直接给出 mini_seq 的占位记录.
    def _fake_build_manifests(data_root, required_streams=None):
        return (
            {'data_root': str(data_root), 'sequence_count': 1, 'required_streams': list(required_streams or ()), 'sequences': [{'seq_id': 'mini_seq', 'files': list(required_streams or ())}]},
            {'scene_count': 0, 'scenes': []},
        )
    monkeypatch.setattr("liquidloc.pipelines.prepare_pipeline.build_manifests", _fake_build_manifests)

    output_root = tmp_path / "util_out"
    assert module.main(["--config", str(cfg_path), "--seq-ids", "mini_seq", "--output-root", str(output_root)]) == 0

    summary = json.loads((output_root / "prepare_summary.json").read_text(encoding="utf-8"))
    assert summary["raw_root"] == str(raw_root.resolve())


def test_prepare_util_keeps_external_config_relative_to_config_dir(tmp_path, monkeypatch):
    """保持测试：prepare util。\n\n验证 prepare util 的保持行为，\n确保特定属性在处理过程中不变。
    """
    module = _load_module("17_prepare_util_data.py", "prepare_util_anchor_external")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    cfg_dir = tmp_path / "external_cfg"
    cfg_dir.mkdir()
    cfg_path = cfg_dir / "util.yaml"
    cfg_path.write_text("raw_root: relative/raw\nfield_mapping:\n  imu: {}\n", encoding="utf-8")
    raw_root = cfg_dir / "relative" / "raw"
    raw_root.mkdir(parents=True)

    def _fake_load_yaml_config(path):
        assert path == cfg_path
        return {"raw_root": "relative/raw", "field_mapping": {"imu": {}}}

    def _fake_read_util_sequence(seq_id, resolved_root):
        assert Path(str(resolved_root)).resolve() == raw_root.resolve()
        # 提供非空流以满足 run_dataset_checks, 与 anchors 测试保持同步.
        return ({
            'imu_raw': [{'timestamp': 0.0, 'ax': 0.0, 'ay': 0.0, 'gz': 0.0}],
            'uwb_raw': [{'timestamp': 0.1, 'anchor_id': 1, 'range': 2.0, 'valid': True}],
            'flow_raw': [{'timestamp': 0.2, 'vx': 0.0, 'vy': 0.0}],
            'gt_raw': [{'timestamp': 0.3, 'px': 0.0, 'py': 0.0, 'yaw': 0.0}],
        }, {'is_complete': True})

    def _fake_map_external_fields(raw_bundle, field_mapping):
        # 与 anchors 测试同步: 简单 fake map 直接补出 internal_bundle 兼容字段集.
        internal = {
            'imu_raw': [{'timestamp': r.get('timestamp', 0.0), 'ax': r.get('ax', 0.0), 'ay': r.get('ay', 0.0), 'gz': r.get('gz', 0.0)} for r in raw_bundle.get('imu_raw', [])],
            'uwb_raw': [{**r, 'quality': r.get('quality', 1.0)} for r in raw_bundle.get('uwb_raw', [])],
            'flow_raw': [{'timestamp': r.get('timestamp', 0.0), 'dx': r.get('dx', 0.0), 'dy': r.get('dy', 0.0), 'quality': r.get('quality', 1.0)} for r in raw_bundle.get('flow_raw', [])],
            'gt_raw': list(raw_bundle.get('gt_raw', [])),
        }
        return internal, {'is_complete': True}

    monkeypatch.setattr(module, "ROOT", repo_root)
    monkeypatch.setattr(module, "load_yaml_config", _fake_load_yaml_config)
    monkeypatch.setattr(module, "read_util_sequence", _fake_read_util_sequence)
    monkeypatch.setattr(module, "map_external_fields", _fake_map_external_fields)
    # 脚本最终走 prepare_pipeline.run(), 旁路管道层的这两个引用, fake 才会被触发.
    monkeypatch.setattr("liquidloc.pipelines.prepare_pipeline.read_util_sequence", _fake_read_util_sequence)
    monkeypatch.setattr("liquidloc.pipelines.prepare_pipeline.map_external_fields", _fake_map_external_fields)
    # 同步旁路 build_manifests, 避免扫盘读取磁盘上不存在的序列文件.
    def _fake_build_manifests(data_root, required_streams=None):
        return (
            {'data_root': str(data_root), 'sequence_count': 1, 'required_streams': list(required_streams or ()), 'sequences': [{'seq_id': 'mini_seq', 'files': list(required_streams or ())}]},
            {'scene_count': 0, 'scenes': []},
        )
    monkeypatch.setattr("liquidloc.pipelines.prepare_pipeline.build_manifests", _fake_build_manifests)

    output_root = tmp_path / "util_out_external"
    assert module.main(["--config", str(cfg_path), "--seq-ids", "mini_seq", "--output-root", str(output_root)]) == 0

    summary = json.loads((output_root / "prepare_summary.json").read_text(encoding="utf-8"))
    assert summary["raw_root"] == str(raw_root.resolve())


def test_public_benchmark_normalizes_relative_raw_root_to_repo_root(tmp_path, monkeypatch):
    """归一化测试：public benchmark。\n\n验证 public benchmark 的归一化处理，\n确保空白/重复等输入被正确清洗。
    """
    module = _load_module("18_run_public_benchmarks.py", "public_benchmark_anchor")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    raw_root = repo_root / "data" / "raw" / "util"
    raw_root.mkdir(parents=True)
    monkeypatch.setattr(module, "ROOT", repo_root)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    output_root = tmp_path / "public_out"
    assert module.main(["--dataset-name", "util", "--raw-root", "data/raw/util", "--output-root", str(output_root)]) == 2

    readiness_report = json.loads((output_root / "util_readiness.json").read_text(encoding="utf-8"))
    assert readiness_report["raw_root"] == str(raw_root.resolve())
    assert readiness_report["dataset_name"] == "util"
    assert readiness_report["reasons"] == ["unsupported_public_dataset"]
