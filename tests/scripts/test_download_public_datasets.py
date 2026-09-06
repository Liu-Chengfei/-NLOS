from __future__ import annotations

"""公开数据集下载脚本（download_public_datasets）测试模块。

测试覆盖范围：
- 数据集下载的参数验证
- 下载目标路径的处理

被测模块：scripts.download_public_datasets"""

import importlib.util
import json
from pathlib import Path
import urllib.request


ROOT = Path(__file__).resolve().parents[2]


def _load_script():
    script_path = ROOT / 'scripts' / '16_download_public_datasets.py'
    spec = importlib.util.spec_from_file_location('download_public_datasets_script', script_path)
    script = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(script)
    return script


def _write_stream_bundle(seq_dir: Path):
    seq_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        'imu.json': [{'timestamp': 0.0, 'ax': 0.1, 'ay': 0.0, 'gz': 0.01}],
        'uwb.json': [{'timestamp': 0.1, 'anchor_id': 1, 'range': 2.0, 'valid': True, 'quality': 0.9}],
        'vio.json': [{'timestamp': 0.2, 'dx': 0.05, 'dy': 0.0, 'dyaw': 0.0, 'quality': 0.8, 'tracked_features': 40, 'reproj_err': 0.5}],
        'gt.json': [{'timestamp': 0.3, 'px': 0.0, 'py': 0.0, 'yaw': 0.0}],
    }
    for name, payload in payloads.items():
        (seq_dir / name).write_text(json.dumps(payload), encoding='utf-8')


def _registry(*, official_root: Path, landed_raw_root: Path, adapter_script: Path | None = None):
    return {
        'datasets': {
            'miluv': {
                'reader': 'src/liquidloc/dataio/readers/miluv_reader.py',
                'prepare_script': 'scripts/03_prepare_miluv_data.py',
                'benchmark_script': 'scripts/18_run_public_benchmarks.py',
                'download_script': 'scripts/16_download_public_datasets.py',
                'official_root': str(official_root / 'miluv'),
                'landed_raw_root': str(landed_raw_root / 'miluv'),
                'useful_assets': ['imu.json', 'uwb.json', 'vio.json', 'gt.json', 'anchor_layout.json'],
                'vio_backend': 'json',
                'adapter_script': str(adapter_script) if adapter_script is not None else None,
            },
            'ntu_viral': {
                'reader': 'src/liquidloc/dataio/readers/ntu_viral_reader.py',
                'prepare_script': 'scripts/03_prepare_ntu_viral_data.py',
                'benchmark_script': 'scripts/18_run_public_benchmarks.py',
                'download_script': 'scripts/16_download_public_datasets.py',
                'official_root': str(official_root / 'ntu_viral'),
                'landed_raw_root': str(landed_raw_root / 'ntu_viral'),
                'useful_assets': ['imu.json', 'uwb.json', 'vio.json', 'gt.json'],
                'vio_backend': 'json',
                'adapter_script': str(adapter_script) if adapter_script is not None else None,
            },
        }
    }


def test_convert_and_full_can_land_existing_official_sequence_dirs_without_adapter(tmp_path, monkeypatch):
    """无依赖测试：convert and full can land existing official sequence dirs。\n\n验证 convert and full can land existing official sequence dirs 在缺少依赖时的降级行为，\n确保回退策略正确。
    """
    script = _load_script()
    official_root = tmp_path / 'official'
    landed_raw_root = tmp_path / 'landed'
    runtime_root = tmp_path / 'runtime'
    seq_dir = official_root / 'ntu_viral' / 'seq_001'
    _write_stream_bundle(seq_dir)
    registry = _registry(official_root=official_root, landed_raw_root=landed_raw_root)
    monkeypatch.setattr(script, 'load_public_dataset_registry', lambda: registry)

    convert_exit = script.main([
        '--dataset-name', 'ntu_viral',
        '--official-root', str(official_root / 'ntu_viral'),
        '--landed-raw-root', str(landed_raw_root / 'ntu_viral'),
        '--runtime-root', str(runtime_root / 'convert'),
        '--mode', 'convert',
    ])
    convert_report = json.loads((runtime_root / 'convert' / 'landing_report.json').read_text(encoding='utf-8'))

    full_exit = script.main([
        '--dataset-name', 'ntu_viral',
        '--official-root', str(official_root / 'ntu_viral'),
        '--landed-raw-root', str(landed_raw_root / 'ntu_viral'),
        '--runtime-root', str(runtime_root / 'full'),
        '--mode', 'full',
    ])
    full_report = json.loads((runtime_root / 'full' / 'landing_report.json').read_text(encoding='utf-8'))

    assert convert_exit == 0
    assert convert_report['status'] == 'ready'
    assert convert_report['convert_report']['mode'] == 'existing_official_root_copy'
    assert convert_report['convert_report']['reasons'] == []
    assert full_exit == 0
    assert full_report['status'] == 'ready'
    assert full_report['convert_report']['mode'] == 'existing_official_root_copy'
    assert full_report['convert_report']['reasons'] == []
    landed_seq_dir = landed_raw_root / 'ntu_viral' / 'seq_001'
    assert landed_seq_dir.is_dir()
    for filename in ('imu.json', 'uwb.json', 'vio.json', 'gt.json'):
        assert (landed_seq_dir / filename).is_file()


def test_convert_blocks_when_existing_official_root_has_no_sequence_dirs_and_adapter_is_missing(tmp_path, monkeypatch):
    """缺失测试：convert blocks when existing official root has no sequence dirs and adapter is。\n\n验证 convert blocks when existing official root has no sequence dirs and adapter is 在数据缺失时的处理，\n确保缺失数据不影响整体流程。
    """
    script = _load_script()
    official_root = tmp_path / 'official'
    landed_raw_root = tmp_path / 'landed'
    runtime_root = tmp_path / 'runtime'
    miluv_official_root = official_root / 'miluv'
    miluv_official_root.mkdir(parents=True)
    (miluv_official_root / 'bundle.zip').write_bytes(b'zip-placeholder')
    registry = _registry(official_root=official_root, landed_raw_root=landed_raw_root)
    monkeypatch.setattr(script, 'load_public_dataset_registry', lambda: registry)

    convert_exit = script.main([
        '--dataset-name', 'miluv',
        '--official-root', str(miluv_official_root),
        '--landed-raw-root', str(landed_raw_root / 'miluv'),
        '--runtime-root', str(runtime_root / 'convert'),
        '--mode', 'convert',
    ])
    convert_report = json.loads((runtime_root / 'convert' / 'landing_report.json').read_text(encoding='utf-8'))

    assert convert_exit == 2
    assert convert_report['status'] == 'blocked'
    assert convert_report['fetch_report']['source_kind'] == 'manual_import_required'
    assert convert_report['convert_report']['reasons'] == ['official_fetch_not_implemented']
    assert not (landed_raw_root / 'miluv').exists()


def test_all_runs_individually_and_aggregates_summary(tmp_path, monkeypatch):
    script = _load_script()
    official_root = tmp_path / 'official'
    landed_raw_root = tmp_path / 'landed'
    runtime_root = tmp_path / 'runtime'
    for dataset_name in ('miluv', 'ntu_viral'):
        _write_stream_bundle(official_root / dataset_name / 'seq_001')
    registry = _registry(official_root=official_root, landed_raw_root=landed_raw_root)
    monkeypatch.setattr(script, 'load_public_dataset_registry', lambda: registry)

    exit_code = script.main([
        '--dataset-name', 'all',
        '--official-root', str(official_root),
        '--landed-raw-root', str(landed_raw_root),
        '--runtime-root', str(runtime_root),
        '--mode', 'verify',
    ])
    summary = json.loads((runtime_root / 'public_landing_summary.json').read_text(encoding='utf-8'))

    assert exit_code == 2
    assert summary['dataset_name'] == 'all'
    assert summary['dataset_names'] == ['miluv', 'ntu_viral']
    assert summary['datasets']['miluv']['dataset_name'] == 'miluv'
    assert summary['datasets']['ntu_viral']['dataset_name'] == 'ntu_viral'
    assert summary['datasets']['miluv']['mode'] == 'verify'
    assert summary['datasets']['ntu_viral']['mode'] == 'verify'


def test_verify_reports_blocked_when_landed_raw_missing(tmp_path, monkeypatch):
    """缺失测试：verify reports blocked when landed raw。\n\n验证 verify reports blocked when landed raw 在数据缺失时的处理，\n确保缺失数据不影响整体流程。
    """
    script = _load_script()
    official_root = tmp_path / 'official'
    landed_raw_root = tmp_path / 'landed'
    runtime_root = tmp_path / 'runtime'
    _write_stream_bundle(official_root / 'ntu_viral' / 'seq_001')
    registry = _registry(official_root=official_root, landed_raw_root=landed_raw_root)
    monkeypatch.setattr(script, 'load_public_dataset_registry', lambda: registry)

    exit_code = script.main([
        '--dataset-name', 'ntu_viral',
        '--official-root', str(official_root / 'ntu_viral'),
        '--landed-raw-root', str(landed_raw_root / 'ntu_viral'),
        '--runtime-root', str(runtime_root),
        '--mode', 'verify',
    ])
    report = json.loads((runtime_root / 'landing_report.json').read_text(encoding='utf-8'))

    assert exit_code == 2
    assert report['status'] == 'not_ready'
    assert report['readiness_report']['status'] == 'not_ready'


def test_ntu_viral_fetch_manifest_keeps_only_flight_archives(monkeypatch):
    """保持测试：ntu viral fetch manifest。\n\n验证 ntu viral fetch manifest 的保持行为，\n确保特定属性在处理过程中不变。
    """
    script = _load_script()
    monkeypatch.setattr(
        script,
        '_load_json_url',
        lambda _url: {
            'data': {
                'latestVersion': {
                    'files': [
                        {'dataFile': {'id': 1, 'filename': 'eee_01.zip', 'filesize': 10, 'contentType': 'application/zip'}},
                        {'dataFile': {'id': 2, 'filename': 'calib_stereo.zip', 'filesize': 11, 'contentType': 'application/zip'}},
                        {'dataFile': {'id': 3, 'filename': 'notes.txt', 'filesize': 12, 'contentType': 'text/plain'}},
                        {'dataFile': {'id': 4, 'filename': 'tnp_03.zip', 'filesize': 13, 'contentType': 'application/zip'}},
                    ]
                }
            }
        },
    )

    report = script._build_ntu_viral_fetch_manifest()

    assert report['status'] == 'ready'
    assert [item['filename'] for item in report['selected_files']] == ['eee_01.zip', 'tnp_03.zip']
    assert [item['reason'] for item in report['skipped_files']] == ['skip_calibration_asset', 'skip_non_flight_asset']


def test_fetch_reports_existing_official_root_without_network(tmp_path):
    """无依赖测试：fetch reports existing official root。\n\n验证 fetch reports existing official root 在缺少依赖时的降级行为，\n确保回退策略正确。
    """
    script = _load_script()
    official_root = tmp_path / 'official'
    landed_raw_root = tmp_path / 'landed'
    _write_stream_bundle(official_root / 'seq_001')
    registry = _registry(official_root=official_root, landed_raw_root=landed_raw_root)

    report = script._fetch_official_assets('miluv', official_root, registry)

    assert report['status'] == 'ready'
    assert report['source_kind'] == 'existing_official_root'
    assert report['entry_count'] == 1


def test_stream_download_retries_after_curl_partial_failure(tmp_path, monkeypatch):
    script = _load_script()
    destination = tmp_path / 'eee_01.zip'
    payload = b'abcdef'

    monkeypatch.setattr(script.shutil, 'which', lambda _name: 'curl.exe')

    class _Completed:
        returncode = 56
        stderr = 'curl partial transfer'

    def _fake_run(*_args, **_kwargs):
        partial_path = destination.with_suffix(destination.suffix + '.part')
        partial_path.write_bytes(payload[:3])
        return _Completed()

    monkeypatch.setattr(script.subprocess, 'run', _fake_run)

    class _FakeResponse:
        status = 206

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, _size=-1):
            if getattr(self, '_done', False):
                return b''
            self._done = True
            return payload[3:]

    def _fake_urlopen(request, timeout=0):
        assert timeout == script._SOCKET_TIMEOUT_SECONDS
        assert isinstance(request, urllib.request.Request)
        assert request.headers.get('Range') == 'bytes=3-'
        return _FakeResponse()

    monkeypatch.setattr(script.urllib.request, 'urlopen', _fake_urlopen)

    report = script._stream_download_with_resume(
        'https://example.test/eee_01.zip',
        destination,
        expected_size=len(payload),
    )

    assert report['status'] == 'ready'
    assert report['path'] == str(destination)
    assert destination.read_bytes() == payload
    assert report['attempts'][0]['backend'] == 'curl'
    assert report['attempts'][0]['status'] == 'retry'
    assert report['attempts'][-1]['backend'] == 'urllib'


def test_fetch_reports_blocked_when_official_root_is_locked(tmp_path, monkeypatch):
    """报告测试：fetch。\n\n验证 fetch 的报告生成，\n确保审计信息被正确记录。
    """
    script = _load_script()
    official_root = tmp_path / 'official'
    landed_raw_root = tmp_path / 'landed'
    official_root.mkdir(parents=True, exist_ok=True)
    lock_path = official_root / script._OFFICIAL_FETCH_LOCK_NAME
    lock_path.write_text(json.dumps({'pid': 12345, 'dataset_name': 'ntu_viral'}), encoding='utf-8')
    registry = _registry(official_root=official_root, landed_raw_root=landed_raw_root)

    monkeypatch.setattr(script, '_pid_is_alive', lambda pid: pid == 12345)

    report = script._fetch_official_assets('ntu_viral', official_root, registry)

    assert report['status'] == 'blocked'
    assert report['source_kind'] == 'official_fetch_lock'
    assert report['reasons'] == ['official_root_busy']
    assert report['lock_owner']['pid'] == 12345


def test_fetch_ntu_viral_uses_configured_concurrency_and_sorts_reports(tmp_path, monkeypatch):
    """使用测试：fetch ntu viral。\n\n验证被测功能正确使用 fetch ntu viral，\n确保内部依赖被正确调用。
    """
    script = _load_script()
    official_root = tmp_path / 'official'
    official_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        script,
        '_build_ntu_viral_auto_fetch_manifest',
        lambda: {
            'dataset_name': 'ntu_viral',
            'status': 'ready',
            'gate_action': 'pass',
            'source_kind': 'official_metadata_api',
            'metadata_url': 'https://official.example.test',
            'selected_files': [
                {'filename': 'tnp_03.zip', 'filesize': 10, 'download_url': 'https://example.test/tnp_03.zip'},
                {'filename': 'eee_01.zip', 'filesize': 10, 'download_url': 'https://example.test/eee_01.zip'},
            ],
            'skipped_files': [],
            'reasons': [],
        },
    )

    observed_calls: list[str] = []

    def _fake_stream_download(url, destination, *, expected_size):
        observed_calls.append(destination.name)
        destination.write_bytes(b'x' * int(expected_size))
        return {
            'path': str(destination),
            'status': 'ready',
            'action': 'downloaded',
            'bytes_written': int(expected_size),
            'final_size': int(expected_size),
            'expected_size': int(expected_size),
            'attempts': [],
        }

    monkeypatch.setattr(script, '_stream_download_with_resume', _fake_stream_download)

    report = script._fetch_ntu_viral_official_assets(official_root, max_concurrent_downloads=2)

    assert report['status'] == 'ready'
    assert report['max_concurrent_downloads'] == 2
    assert sorted(observed_calls) == ['eee_01.zip', 'tnp_03.zip']
    assert [item['filename'] for item in report['downloaded_files']] == ['eee_01.zip', 'tnp_03.zip']


def test_build_ntu_viral_onedrive_manifest_keeps_only_flight_archives(monkeypatch):
    """保持测试：build ntu viral onedrive manifest。\n\n验证 build ntu viral onedrive manifest 的保持行为，\n确保特定属性在处理过程中不变。
    """
    script = _load_script()
    monkeypatch.setattr(
        script,
        '_load_text_url',
        lambda url: '{"driveInfo":{".driveUrl":"https://sharepoint.example.test/_api/v2.0/drives/demo",".driveAccessToken":"access_token=abc"}}',
    )
    # 实现内部走 urllib.request.urlopen 取 children, 因此需要旁路 urlopen 而非 _load_json_url.
    # 用一个 fake context manager 把假驱动器内容回灌给 with urlopen(...) as resp 调用.
    import contextlib

    fake_items_payload = {
        'value': [
            {'name': 'eee_01.zip', 'size': 123, '@content.downloadUrl': 'https://download/eee_01.zip', 'file': {'mimeType': 'application/zip'}, 'id': '1'},
            {'name': 'calib_stereo.zip', 'size': 456, '@content.downloadUrl': 'https://download/calib.zip', 'file': {'mimeType': 'application/zip'}, 'id': '2'},
            {'name': 'notes.txt', 'size': 12, '@content.downloadUrl': 'https://download/notes.txt', 'file': {'mimeType': 'text/plain'}, 'id': '3'},
        ]
    }
    import json as _json

    class _FakeResp:
        def __init__(self, payload):
            self._payload = _json.dumps(payload).encode("utf-8")

        def read(self):
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _fake_urlopen(req, timeout=None):
        return _FakeResp(fake_items_payload)

    monkeypatch.setattr(script.urllib.request, "urlopen", _fake_urlopen)

    report = script._build_ntu_viral_onedrive_fetch_manifest()

    assert report['source_kind'] == 'official_onedrive_share'
    assert [item['filename'] for item in report['selected_files']] == ['eee_01.zip']
    assert [item['reason'] for item in report['skipped_files']] == ['skip_calibration_asset', 'skip_non_flight_asset']


def test_build_ntu_viral_auto_fetch_manifest_merges_official_sources(monkeypatch):
    """官方测试：build ntu viral auto fetch manifest merges。\n\n验证 build ntu viral auto fetch manifest merges 使用官方数据的正确性，\n确保与官方格式兼容。
    """
    script = _load_script()
    monkeypatch.setattr(
        script,
        '_build_ntu_viral_onedrive_fetch_manifest',
        lambda: {
            'dataset_name': 'ntu_viral',
            'status': 'ready',
            'gate_action': 'pass',
            'source_kind': 'official_onedrive_share',
            'metadata_url': 'https://official.example.test/onedrive',
            'selected_files': [
                {'filename': 'eee_01.zip', 'filesize': 1, 'download_url': 'https://onedrive/eee_01.zip'},
            ],
            'skipped_files': [{'filename': 'calib_stereo.zip', 'reason': 'skip_calibration_asset'}],
            'reasons': [],
        },
    )
    monkeypatch.setattr(
        script,
        '_build_ntu_viral_fetch_manifest',
        lambda: {
            'dataset_name': 'ntu_viral',
            'status': 'ready',
            'gate_action': 'pass',
            'source_kind': 'official_metadata_api',
            'metadata_url': 'https://official.example.test/repository',
            'selected_files': [
                {'filename': 'eee_01.zip', 'filesize': 2, 'download_url': 'https://repo/eee_01.zip'},
                {'filename': 'tnp_03.zip', 'filesize': 3, 'download_url': 'https://repo/tnp_03.zip'},
            ],
            'skipped_files': [{'filename': 'notes.txt', 'reason': 'skip_non_flight_asset'}],
            'reasons': [],
        },
    )

    report = script._build_ntu_viral_auto_fetch_manifest()

    assert report['source_kind'] == 'official_onedrive_share+official_metadata_api'
    assert report['auto_selected_source'] == 'merged_official_sources'
    assert [item['filename'] for item in report['selected_files']] == ['eee_01.zip', 'tnp_03.zip']
    assert report['selected_files'][0]['download_url'] == 'https://onedrive/eee_01.zip'
