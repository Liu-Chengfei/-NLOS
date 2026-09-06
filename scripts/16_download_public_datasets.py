"""Download, convert, and verify public datasets.

网络说明 / Network Requirements:
- 自动下载依赖 Dataverse API（https://dataverse.ntu.edu.sg）和 OneDrive 共享链接。
- 两者在部分网络环境（如公司防火墙、某些地区）可能被阻止。
- 若无法访问，须手动放置数据：
  1. NTU VIRAL: `data/public/official/ntu_viral/{eee_aa_xx, ny02_aa_xx}/`（-flight archives as JSON）
  2. MILUV:    `data/public/official/miluv/{exp_01_*, ...}/`（序列目录）
  3. UTIL:     `data/public/official/util/{identification_xx, flight_xx}/`
  脚本会自动检测 `data/raw/<dataset>/` 目录是否存在，若存在则跳过下载。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from liquidloc.common.config_utils import load_yaml_config
from liquidloc.common.io_utils import dumps_json_text, loads_json_text, read_json, write_json
from liquidloc.common.paths import get_standard_dirs
from liquidloc.common.validation import validate_path_component
from liquidloc.dataio.readers.miluv_reader import inspect_miluv_raw_readiness
from liquidloc.dataio.readers.ntu_viral_reader import inspect_ntu_viral_raw_readiness
from liquidloc.dataio.registry.public_dataset_registry import (
    get_dataset_entry,
    load_public_dataset_registry,
)

_OFFICIAL_FETCH_LOCK_NAME = ".fetch_lock.json"
_SOCKET_TIMEOUT_SECONDS = 30
# 论文官方仓库实际托管在 researchdata.ntu.edu.sg（不是 dataverse.ntu.edu.sg 的 :persistentId API，
# 后者对自动化请求返回 HTML 而非 JSON）。每个 flight 序列是独立的 datafile 下载端点。
# 序列名 → datafile ID 映射来自 ntu-aris/ntu_viral_dataset README 表格。
_NTU_VIRAL_SEQUENCE_DATAFILE_IDS = {
    "eee_01": 68133,
    "eee_02": 68131,
    "eee_03": 68132,
    "nya_01": 68144,
    "nya_02": 68138,
    "nya_03": 68142,
    "sbs_01": 68139,
    "sbs_02": 68140,
    "sbs_03": 68143,
    "rtp_01": 98194,
    "rtp_02": 98191,
    "rtp_03": 98192,
    "tnp_01": 98188,
    "tnp_02": 98189,
    "tnp_03": 98190,
    "spms_01": 98195,
    "spms_02": 98196,
    "spms_03": 98193,
}
_NTU_VIRAL_DATAFILE_BASE_URL = "https://researchdata.ntu.edu.sg/api/access/datafile/"
_NTU_VIRAL_ONEDRIVE_SHARE_URL = (
    "https://ntuedusg-my.sharepoint.com/:f:/g/personal/"
    "cle003_e_ntu_edu_sg/EiJtY9L3g9JAtJHmF8X0jLABvZrSLYrFm0Y3JbPvGJjMxQ"
)
_FLIGHT_ARCHIVE_PREFIXES = ("eee_", "tnp_")
_CALIBRATION_PREFIXES = ("calib_",)


def _resolve_raw_root(dataset_name: str, raw_root_override: str | None) -> Path:
    """Resolve a dataset raw root from CLI override or dataset config."""
    if raw_root_override:
        return Path(raw_root_override).resolve()
    dataset_cfg = load_yaml_config(ROOT / "configs" / "datasets" / f"{dataset_name}.yaml")
    configured_root = Path(dataset_cfg["raw_root"])
    if not configured_root.is_absolute():
        configured_root = ROOT / configured_root
    return configured_root.resolve()


def _load_json_url(url: str) -> dict:
    """Load a JSON payload from a URL."""
    with urllib.request.urlopen(url, timeout=_SOCKET_TIMEOUT_SECONDS) as resp:
        return loads_json_text(resp.read().decode("utf-8"))


def _load_text_url(url: str) -> str:
    """Load a text payload from a URL."""
    with urllib.request.urlopen(url, timeout=_SOCKET_TIMEOUT_SECONDS) as resp:
        return resp.read().decode("utf-8")


def _write_sha256_sidecar(destination: Path) -> str:
    """Compute sha256 of a downloaded file and write a `.sha256` sidecar file.

    非阻塞：任何异常都返回空字符串，不影响下载主流程。
    用于在下载完整性环节新增 checksum 校验，远端未提供 sha256 时本地落盘。
    """
    try:
        sha = hashlib.sha256()
        with destination.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                sha.update(chunk)
        digest = sha.hexdigest()
        sidecar = destination.with_suffix(destination.suffix + ".sha256")
        sidecar.write_text(f"{digest}  {destination.name}\n", encoding="utf-8")
        return digest
    except Exception:
        return ""


def _pid_is_alive(pid: int) -> bool:
    """Check whether a process ID is still alive.

    Windows 上 os.kill(pid, 0) 会因 signal 0 不支持而抛 ValueError，
    改用 ctypes.OpenProcess 探测进程存活；POSIX 仍走 os.kill(pid, 0)。
    """
    if os.name == "nt":
        try:
            import ctypes
            # PROCESS_QUERY_LIMITED_INFORMATION，足够用于探测进程存活而不需要更高权限。
            access = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(access, False, pid)
            if not handle:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError, ValueError):
        return False
    return True


def _is_flight_archive(filename: str) -> bool:
    """Return True when a file is an NTU VIRAL flight archive."""
    return any(filename.startswith(prefix) for prefix in _FLIGHT_ARCHIVE_PREFIXES)


def _is_calibration_asset(filename: str) -> bool:
    """Return True when a file is an NTU VIRAL calibration asset."""
    return any(filename.startswith(prefix) for prefix in _CALIBRATION_PREFIXES)


def _classify_ntu_viral_file(filename: str) -> tuple[str, str]:
    """Classify NTU VIRAL assets for download selection."""
    if _is_flight_archive(filename):
        return "selected", ""
    if _is_calibration_asset(filename):
        return "skipped", "skip_calibration_asset"
    return "skipped", "skip_non_flight_asset"


def _build_ntu_viral_fetch_manifest() -> dict:
    """Build an NTU VIRAL fetch manifest from the Dataverse API."""
    try:
        data = _load_json_url(_NTU_VIRAL_DATAVERSE_URL)
    except Exception as exc:
        return {
            "dataset_name": "ntu_viral",
            "status": "unavailable",
            "gate_action": "skip",
            "source_kind": "official_metadata_api",
            "metadata_url": _NTU_VIRAL_DATAVERSE_URL,
            "selected_files": [],
            "skipped_files": [],
            "reasons": [f"api_error: {exc}"],
        }

    files_data = data.get("data", {}).get("latestVersion", {}).get("files", [])
    selected_files = []
    skipped_files = []
    for entry in files_data:
        data_file = entry.get("dataFile", {})
        filename = data_file.get("filename", "")
        category, reason = _classify_ntu_viral_file(filename)
        if category == "selected":
            file_id = data_file.get("id", "")
            selected_files.append(
                {
                    "filename": filename,
                    "filesize": int(data_file.get("filesize", 0)),
                    "content_type": data_file.get("contentType", ""),
                    "file_id": file_id,
                    "download_url": (
                        f"https://dataverse.ntu.edu.sg/api/access/datafile/{file_id}"
                    ),
                }
            )
        else:
            skipped_files.append({"filename": filename, "reason": reason})
    return {
        "dataset_name": "ntu_viral",
        "status": "ready",
        "gate_action": "pass",
        "source_kind": "official_metadata_api",
        "metadata_url": _NTU_VIRAL_DATAVERSE_URL,
        "selected_files": selected_files,
        "skipped_files": skipped_files,
        "reasons": [],
    }


def _build_ntu_viral_onedrive_fetch_manifest() -> dict:
    """Build an NTU VIRAL fetch manifest from the official OneDrive share."""
    try:
        sharing_info = loads_json_text(_load_text_url(_NTU_VIRAL_ONEDRIVE_SHARE_URL))
        drive_info = sharing_info.get("driveInfo", {})
        drive_url = drive_info.get(".driveUrl", "")
        access_token = drive_info.get(".driveAccessToken", "")
        if not drive_url:
            return {
                "dataset_name": "ntu_viral",
                "status": "unavailable",
                "gate_action": "skip",
                "source_kind": "official_onedrive_share",
                "selected_files": [],
                "skipped_files": [],
                "reasons": ["missing_drive_url"],
            }
        # 使用 Authorization: Bearer header 而非 URL query string，
        # 避免 access_token 在 HTTPError 异常消息中泄露到日志。
        # OneDrive sharing API 返回的 .driveAccessToken 形如 "access_token=eyJ...",
        # 需要去掉前缀只保留 token 值用于 Bearer header。
        token_value = access_token
        if isinstance(token_value, str) and token_value.startswith("access_token="):
            token_value = token_value[len("access_token="):]
        items_req = urllib.request.Request(
            f"{drive_url}/root/children",
            headers={"Authorization": f"Bearer {token_value}"},
        )
        with urllib.request.urlopen(items_req, timeout=_SOCKET_TIMEOUT_SECONDS) as resp:
            items_data = loads_json_text(resp.read().decode("utf-8"))
    except Exception as exc:
        return {
            "dataset_name": "ntu_viral",
            "status": "unavailable",
            "gate_action": "skip",
            "source_kind": "official_onedrive_share",
            "selected_files": [],
            "skipped_files": [],
            "reasons": [f"onedrive_error: {exc}"],
        }

    selected_files = []
    skipped_files = []
    for item in items_data.get("value", []):
        filename = item.get("name", "")
        category, reason = _classify_ntu_viral_file(filename)
        if category == "selected":
            file_info = item.get("file", {})
            selected_files.append(
                {
                    "filename": filename,
                    "filesize": int(item.get("size", 0)),
                    "download_url": item.get("@content.downloadUrl", ""),
                    "mime_type": file_info.get("mimeType", "")
                    if isinstance(file_info, dict)
                    else "",
                    "item_id": item.get("id", ""),
                }
            )
        else:
            skipped_files.append({"filename": filename, "reason": reason})
    return {
        "dataset_name": "ntu_viral",
        "status": "ready",
        "gate_action": "pass",
        "source_kind": "official_onedrive_share",
        "selected_files": selected_files,
        "skipped_files": skipped_files,
        "reasons": [],
    }


def _build_ntu_viral_auto_fetch_manifest() -> dict:
    """Merge OneDrive and Dataverse NTU VIRAL manifests."""
    onedrive_manifest = _build_ntu_viral_onedrive_fetch_manifest()
    dataverse_manifest = _build_ntu_viral_fetch_manifest()

    onedrive_by_name = {
        item["filename"]: item for item in onedrive_manifest.get("selected_files", [])
    }
    dataverse_by_name = {
        item["filename"]: item for item in dataverse_manifest.get("selected_files", [])
    }

    merged_files = []
    seen_names = set()
    for name, item in onedrive_by_name.items():
        merged_files.append(item)
        seen_names.add(name)
    for name, item in dataverse_by_name.items():
        if name not in seen_names:
            merged_files.append(item)
            seen_names.add(name)

    merged_skipped = list(onedrive_manifest.get("skipped_files", []))
    seen_skip_names = {item["filename"] for item in merged_skipped}
    for item in dataverse_manifest.get("skipped_files", []):
        if item["filename"] not in seen_skip_names:
            merged_skipped.append(item)
            seen_skip_names.add(item["filename"])

    merged_reasons = list(onedrive_manifest.get("reasons", []))
    merged_reasons.extend(dataverse_manifest.get("reasons", []))
    merged_status = (
        "ready"
        if onedrive_manifest.get("status") == "ready"
        or dataverse_manifest.get("status") == "ready"
        else "unavailable"
    )
    return {
        "dataset_name": "ntu_viral",
        "status": merged_status,
        "gate_action": "pass" if merged_status == "ready" else "skip",
        "source_kind": "official_onedrive_share+official_metadata_api",
        "auto_selected_source": "merged_official_sources",
        "selected_files": merged_files,
        "skipped_files": merged_skipped,
        "reasons": merged_reasons,
    }


def _get_sequence_required_assets(dataset_entry: dict) -> list[str]:
    """Return the required per-sequence raw assets for a dataset."""
    useful_assets = dataset_entry.get("useful_assets") or []
    required_assets = []
    for asset_name in useful_assets:
        if not isinstance(asset_name, str):
            continue
        asset_name = asset_name.strip()
        if not asset_name or asset_name == "anchor_layout.json":
            continue
        required_assets.append(asset_name)
    return required_assets


def _list_sequence_dirs(root: Path, required_assets: list[str]) -> list[Path]:
    """List child directories that satisfy the dataset raw contract."""
    if not root.is_dir():
        return []
    sequence_dirs = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and all((child / asset_name).is_file() for asset_name in required_assets):
            sequence_dirs.append(child)
    return sequence_dirs


def _list_ntu_viral_archives(root: Path) -> list[Path]:
    """List downloaded NTU VIRAL flight archives."""
    if not root.is_dir():
        return []
    return [
        child
        for child in sorted(root.iterdir())
        if child.is_file() and _is_flight_archive(child.name)
    ]


def _inspect_dataset_raw_readiness(dataset_name: str, raw_root: Path, dataset_entry: dict) -> dict:
    """Inspect raw readiness using the dataset-specific contract."""
    if dataset_name == "miluv":
        return inspect_miluv_raw_readiness(str(raw_root))
    if dataset_name == "ntu_viral":
        return inspect_ntu_viral_raw_readiness(str(raw_root))

    required_assets = _get_sequence_required_assets(dataset_entry)
    sequence_dirs = _list_sequence_dirs(raw_root, required_assets)
    return {
        "dataset_name": dataset_name,
        "raw_root": str(raw_root),
        "status": "ready" if sequence_dirs else "not_ready",
        "gate_action": "pass" if sequence_dirs else "blocked",
        "reasons": [] if sequence_dirs else ["missing_required_streams"],
        "sequence_count": len(sequence_dirs),
        "ready_sequence_count": len(sequence_dirs),
        "required_assets": required_assets,
    }


def _stream_download_with_resume(url: str, destination: Path, *, expected_size: int) -> dict:
    """Download a file with curl-first resume fallback."""
    attempts = []
    destination.parent.mkdir(parents=True, exist_ok=True)

    curl_path = shutil.which("curl")
    if curl_path:
        partial_path = destination.with_suffix(destination.suffix + ".part")
        try:
            result = subprocess.run(
                [curl_path, "-fSL", "-o", str(partial_path), url],
                capture_output=True,
                timeout=_SOCKET_TIMEOUT_SECONDS * 3,
            )
            if result.returncode == 0:
                if partial_path.is_file():
                    shutil.move(str(partial_path), str(destination))
                attempts.append({"backend": "curl", "status": "ok"})
                sha256_digest = _write_sha256_sidecar(destination)
                return {
                    "status": "ready",
                    "path": str(destination),
                    "attempts": attempts,
                    "sha256": sha256_digest,
                }

            existing_bytes = partial_path.stat().st_size if partial_path.is_file() else 0
            attempts.append(
                {"backend": "curl", "status": "retry", "partial_bytes": existing_bytes}
            )
            if existing_bytes > 0:
                req = urllib.request.Request(url, headers={"Range": f"bytes={existing_bytes}-"})
                with urllib.request.urlopen(req, timeout=_SOCKET_TIMEOUT_SECONDS) as resp:
                    with destination.open("wb") as handle:
                        handle.write(partial_path.read_bytes())
                        while True:
                            chunk = resp.read(8192)
                            if not chunk:
                                break
                            handle.write(chunk)
                if partial_path.is_file():
                    partial_path.unlink()
                attempts.append(
                    {
                        "backend": "urllib",
                        "status": "ok",
                        "resumed_from": existing_bytes,
                    }
                )
                sha256_digest = _write_sha256_sidecar(destination)
                return {
                    "status": "ready",
                    "path": str(destination),
                    "attempts": attempts,
                    "sha256": sha256_digest,
                }
        except Exception as exc:
            attempts.append({"backend": "curl", "status": "error", "error": str(exc)})

    try:
        with urllib.request.urlopen(url, timeout=_SOCKET_TIMEOUT_SECONDS) as resp:
            with destination.open("wb") as handle:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    handle.write(chunk)
        attempts.append({"backend": "urllib", "status": "ok"})
        sha256_digest = _write_sha256_sidecar(destination)
        return {
            "status": "ready",
            "path": str(destination),
            "attempts": attempts,
            "sha256": sha256_digest,
        }
    except Exception as exc:
        attempts.append({"backend": "urllib", "status": "error", "error": str(exc)})
        return {
            "status": "failed",
            "path": str(destination),
            "attempts": attempts,
            "expected_size": int(expected_size),
        }


def _fetch_ntu_viral_official_assets(
    official_root: Path, max_concurrent_downloads: int = 2
) -> dict:
    """Download NTU VIRAL official flight archives into official_root."""
    manifest = _build_ntu_viral_auto_fetch_manifest()
    if manifest.get("status") != "ready":
        return {
            "status": "unavailable",
            "max_concurrent_downloads": max_concurrent_downloads,
            "downloaded_files": [],
            "failed_files": [],
            "reasons": manifest.get("reasons", []),
        }

    official_root.mkdir(parents=True, exist_ok=True)
    downloaded_files = []
    failed_files = []
    for file_info in sorted(manifest.get("selected_files", []), key=lambda x: x.get("filename", "")):
        filename = file_info["filename"]
        download_url = file_info.get("download_url", "")
        expected_size = int(file_info.get("filesize", 0))
        # 路径穿越防护：filename 来自远端 API（OneDrive/Dataverse），
        # 必须校验不含 ../、/、\\ 等穿越字符，避免写出 official_root 之外。
        validate_path_component(filename, name="filename")
        destination = official_root / filename
        if not download_url:
            failed_files.append(
                {
                    "filename": filename,
                    "filesize": expected_size,
                    "reason": "missing_download_url",
                }
            )
            continue
        report = _stream_download_with_resume(
            download_url,
            destination,
            expected_size=expected_size,
        )
        if report.get("status") == "ready":
            downloaded_files.append(
                {
                    "filename": filename,
                    "filesize": expected_size,
                    "path": report["path"],
                }
            )
        else:
            failed_files.append(
                {
                    "filename": filename,
                    "filesize": expected_size,
                    "attempts": report.get("attempts", []),
                }
            )

    status = "ready" if downloaded_files and not failed_files else "failed"
    reasons = []
    if failed_files:
        reasons.append("official_download_failed")
    if not downloaded_files:
        reasons.append("no_downloadable_official_assets")
    return {
        "status": status,
        "max_concurrent_downloads": max_concurrent_downloads,
        "downloaded_files": downloaded_files,
        "failed_files": failed_files,
        "reasons": reasons,
    }


def _fetch_official_assets(dataset_name: str, official_root: Path, registry: dict) -> dict:
    """Fetch or inspect official assets for one dataset."""
    lock_path = official_root / _OFFICIAL_FETCH_LOCK_NAME
    if lock_path.is_file():
        try:
            lock_data = read_json(lock_path)
            lock_pid = lock_data.get("pid")
            if lock_pid is not None and _pid_is_alive(lock_pid):
                return {
                    "status": "blocked",
                    "source_kind": "official_fetch_lock",
                    "reasons": ["official_root_busy"],
                    "lock_owner": lock_data,
                }
        except ValueError:
            pass

    dataset_entry = registry.get("datasets", {}).get(dataset_name, {})
    required_assets = _get_sequence_required_assets(dataset_entry)
    sequence_dirs = _list_sequence_dirs(official_root, required_assets)
    if sequence_dirs:
        return {
            "status": "ready",
            "source_kind": "existing_official_root",
            "entry_count": len(sequence_dirs),
            "required_assets": required_assets,
        }

    if dataset_name == "ntu_viral":
        existing_archives = _list_ntu_viral_archives(official_root)
        if existing_archives:
            return {
                "status": "ready",
                "source_kind": "existing_official_archives",
                "archive_count": len(existing_archives),
                "landing_ready": False,
                "reasons": ["archive_payload_requires_unpack"],
            }
        fetch_report = _fetch_ntu_viral_official_assets(official_root)
        if fetch_report.get("status") == "ready":
            return {
                **fetch_report,
                "source_kind": "downloaded_official_archives",
                "archive_count": len(fetch_report.get("downloaded_files", [])),
                "landing_ready": False,
                "reasons": list(fetch_report.get("reasons", []))
                + ["archive_payload_requires_unpack"],
            }
        return fetch_report

    return {
        "status": "blocked",
        "source_kind": "manual_import_required",
        "reasons": ["official_fetch_not_implemented"],
    }


def _has_sequence_dirs(root: Path) -> bool:
    """Return True when a directory contains prepared sequence children."""
    if not root.is_dir():
        return False
    for child in root.iterdir():
        if child.is_dir() and (child / "imu.json").is_file():
            return True
    return False


def _run_adapter_script(
    adapter_script: str,
    official_root: Path,
    landed_raw_root: Path,
) -> dict:
    """Run a per-dataset adapter script as a subprocess.

    adapter_script must be a path to a Python file exposing a callable named
    ``unpack_<dataset>_archives(official_root, landed_raw_root, [seq_ids]) -> dict``.
    See scripts/19_unpack_ntu_viral_archives.py for the reference contract.

    Failures (non-zero exit or missing module) degrade gracefully into a
    ``blocked``/``adapter_failed`` report so the orchestration can continue
    with the next dataset rather than aborting the whole pipeline.
    """
    script_path = (ROOT / adapter_script).resolve() if not Path(adapter_script).is_absolute() else Path(adapter_script)
    if not script_path.is_file():
        return {
            "mode": "adapter_missing",
            "reasons": [f"adapter_script_not_found:{script_path}"],
        }
    cmd = [
        sys.executable,
        str(script_path),
        "--official-root",
        str(official_root),
        "--landed-raw-root",
        str(landed_raw_root),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 单数据集 1h 上限，足够 unzip 多 GB archive。
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "mode": "adapter_failed",
            "reasons": ["adapter_timeout"],
            "timeout_seconds": 3600,
            "error": str(exc),
        }
    if result.returncode != 0:
        return {
            "mode": "adapter_failed",
            "reasons": [f"adapter_exit_code_{result.returncode}"],
            "stderr": result.stderr[-2000:] if result.stderr else "",
        }
    try:
        adapter_report = loads_json_text(result.stdout)
    except Exception as exc:
        return {
            "mode": "adapter_failed",
            "reasons": ["adapter_stdout_unparseable"],
            "error": str(exc),
            "stdout_head": result.stdout[:1000],
            "stderr_tail": result.stderr[-1000:] if result.stderr else "",
        }
    adapter_status = str(adapter_report.get("status", "unknown"))
    if adapter_status in ("ready", "partial"):
        return {
            "mode": "adapter_invoked",
            "adapter_status": adapter_status,
            "adapter_report": adapter_report,
            "reasons": [] if adapter_status == "ready" else ["adapter_partial"],
        }
    return {
        "mode": "adapter_failed",
        "reasons": [f"adapter_status_{adapter_status}"],
        "adapter_report": adapter_report,
    }


def _run_single_dataset(
    dataset_name: str,
    official_root: Path,
    landed_raw_root: Path,
    runtime_root: Path,
    mode: str,
    registry: dict,
) -> dict:
    """Run fetch, convert, verify, or full mode for one dataset."""
    runtime_root.mkdir(parents=True, exist_ok=True)
    report: dict = {"dataset_name": dataset_name, "mode": mode}
    dataset_entry = registry.get("datasets", {}).get(dataset_name, {})

    if mode == "fetch":
        fetch_report = _fetch_official_assets(dataset_name, official_root, registry)
        report["fetch_report"] = fetch_report
        report["status"] = fetch_report.get("status", "unknown")
        report_path = runtime_root / "landing_report.json"
        write_json(report_path, report)
        return report

    if mode in ("convert", "full"):
        fetch_report = _fetch_official_assets(dataset_name, official_root, registry)
        report["fetch_report"] = fetch_report
        convert_report: dict = {"mode": "", "reasons": []}

        if _has_sequence_dirs(official_root):
            convert_report["mode"] = "existing_official_root_copy"
            landed_raw_root.mkdir(parents=True, exist_ok=True)
            for child in sorted(official_root.iterdir()):
                if child.is_dir() and (child / "imu.json").is_file():
                    dest = landed_raw_root / child.name
                    if not dest.exists():
                        shutil.copytree(str(child), str(dest))
        else:
            adapter_script = dataset_entry.get("adapter_script")
            fetch_status = fetch_report.get("status")
            fetch_reasons = list(fetch_report.get("reasons", []))
            if fetch_status != "ready":
                convert_report["mode"] = "blocked"
                convert_report["reasons"] = fetch_reasons or [f"fetch_status_{fetch_status}"]
            elif not adapter_script:
                convert_report["mode"] = "blocked"
                convert_report["reasons"] = ["adapter_missing"]
            else:
                # archive_payload_requires_unpack 是 NTU VIRAL 等数据集 archive 解包信号；
                # 此时调用 adapter 脚本完成 archive→sequence 目录转换。
                adapter_result = _run_adapter_script(
                    adapter_script, official_root, landed_raw_root,
                )
                convert_report["mode"] = adapter_result["mode"]
                convert_report["reasons"] = adapter_result.get("reasons", [])
                if "adapter_report" in adapter_result:
                    convert_report["adapter_report"] = adapter_result["adapter_report"]
                if "adapter_status" in adapter_result:
                    convert_report["adapter_status"] = adapter_result["adapter_status"]

        report["convert_report"] = convert_report
        if convert_report.get("mode") == "blocked":
            report["status"] = "blocked"
            report_path = runtime_root / "landing_report.json"
            write_json(report_path, report)
            return report

    if mode in ("verify", "full"):
        readiness_report = _inspect_dataset_raw_readiness(
            dataset_name,
            landed_raw_root,
            dataset_entry,
        )
        report["readiness_report"] = readiness_report

    if mode == "convert":
        convert_mode = report.get("convert_report", {}).get("mode")
        # 这些状态意味着 archive 没有真正解包成可被 reader 消费的序列目录，
        # 视为已阻塞，对外暴露 blocked。
        blocked_modes = {"blocked", "adapter_missing", "adapter_failed"}
        report["status"] = "blocked" if convert_mode in blocked_modes else "ready"
    elif mode in ("verify", "full"):
        report["status"] = report.get("readiness_report", {}).get("status", "not_ready")
    else:
        report["status"] = "ready"

    report_path = runtime_root / "landing_report.json"
    write_json(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Download, convert and verify public datasets")
    parser.add_argument("--dataset-name", default="miluv")
    parser.add_argument("--official-root", default=None)
    parser.add_argument("--landed-raw-root", default=None)
    parser.add_argument("--runtime-root", default=None)
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--mode", choices=("fetch", "convert", "verify", "full"), default=None)
    args = parser.parse_args(argv)

    from liquidloc.common.tee_logger import print_args, print_dict
    print_args(args, "16_download_public_datasets")

    print("[16_pub_download] 开始 | dataset_name=" + str(args.dataset_name) + " mode=" + str(args.mode or "verify"), flush=True)

    if args.mode is None and not args.official_root and not args.landed_raw_root:
        print("[16_pub_download] 完成 | 返回码=<legacy_readiness_check>", flush=True)
        return _legacy_readiness_check(args)

    mode = args.mode or "verify"
    registry_cfg = load_public_dataset_registry()
    print_dict(registry_cfg, "公开数据集注册表 (registry)")
    print("[16_pub_download] 注册表已加载 | mode=" + str(mode) + " dataset_name=" + str(args.dataset_name), flush=True)

    if args.dataset_name == "all":
        print("[16_pub_download] 正在处理所有数据集", flush=True)
        dataset_names = list(registry_cfg.get("datasets", {}).keys())
        datasets_reports = {}
        for ds_name in dataset_names:
            ds_entry = registry_cfg.get("datasets", {}).get(ds_name, {})
            official_root = (
                Path(args.official_root) / ds_name
                if args.official_root
                else Path(ds_entry.get("official_root", ROOT / "data" / "raw" / ds_name))
            )
            landed_raw_root = (
                Path(args.landed_raw_root) / ds_name
                if args.landed_raw_root
                else Path(ds_entry.get("landed_raw_root", ROOT / "data" / "raw" / ds_name))
            )
            runtime_root = (
                Path(args.runtime_root)
                if args.runtime_root
                else (ROOT / "outputs" / "public_dataset_readiness")
            )
            ds_report = _run_single_dataset(
                ds_name,
                official_root,
                landed_raw_root,
                runtime_root / ds_name,
                mode,
                registry_cfg,
            )
            datasets_reports[ds_name] = ds_report

        runtime_root = (
            Path(args.runtime_root)
            if args.runtime_root
            else (ROOT / "outputs" / "public_dataset_readiness")
        )
        runtime_root.mkdir(parents=True, exist_ok=True)
        summary = {
            "dataset_name": "all",
            "dataset_names": dataset_names,
            "datasets": datasets_reports,
        }
        summary_path = runtime_root / "public_landing_summary.json"
        write_json(summary_path, summary)
        print(dumps_json_text(summary))
        print("[16_pub_download] 完成 | 返回码=2", flush=True)
        return 2

    dataset_name = args.dataset_name
    dataset_entry = get_dataset_entry(dataset_name, registry_cfg)
    print_dict(dataset_entry, "数据集条目 (dataset_entry)")
    official_root = (
        Path(args.official_root)
        if args.official_root
        else Path(dataset_entry.get("official_root", ROOT / "data" / "raw" / dataset_name))
    )
    landed_raw_root = (
        Path(args.landed_raw_root)
        if args.landed_raw_root
        else Path(dataset_entry.get("landed_raw_root", ROOT / "data" / "raw" / dataset_name))
    )
    runtime_root = (
        Path(args.runtime_root)
        if args.runtime_root
        else (ROOT / "outputs" / "public_dataset_readiness")
    )
    print("[16_pub_download] 正在处理单个数据集: " + str(dataset_name), flush=True)
    report = _run_single_dataset(
        dataset_name,
        official_root,
        landed_raw_root,
        runtime_root,
        mode,
        registry_cfg,
    )
    print_dict({"dataset_name": args.dataset_name, "official_root": str(official_root) if official_root else None, "landed_raw_root": str(landed_raw_root) if landed_raw_root else None, "runtime_root": str(runtime_root) if runtime_root else None, "mode": mode}, "下载路径与模式")
    print(
        dumps_json_text(
            {
                "dataset_name": dataset_name,
                "status": report.get("status", "unknown"),
                "mode": mode,
            }
        )
    )
    print("[16_pub_download] 完成 | 返回码=" + str(0 if report.get("status") in ("ready",) else 2), flush=True)
    return 0 if report.get("status") in ("ready",) else 2


def _legacy_readiness_check(args) -> int:
    """Legacy raw-readiness-only path."""
    dataset_name = args.dataset_name
    registry_cfg = load_public_dataset_registry()
    dataset_entry = get_dataset_entry(dataset_name, registry_cfg)
    raw_root = _resolve_raw_root(dataset_name, args.raw_root)
    readiness_report = _inspect_dataset_raw_readiness(
        dataset_name,
        Path(raw_root),
        dataset_entry,
    )
    readiness_report["dataset_entry"] = dataset_entry
    readiness_report["check_scope"] = "raw_readiness_only"

    dirs = get_standard_dirs()
    output_root = Path(args.output_root or dirs["outputs"] / "public_dataset_readiness")
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / f"{dataset_name}_raw_readiness.json"
    write_json(report_path, readiness_report)
    print(
        dumps_json_text(
            {
                "dataset_name": dataset_name,
                "raw_root": str(raw_root),
                "status": readiness_report["status"],
                "gate_action": readiness_report["gate_action"],
                "reasons": readiness_report["reasons"],
                "sequence_count": readiness_report["sequence_count"],
                "ready_sequence_count": readiness_report["ready_sequence_count"],
                "report_path": str(report_path),
            }
        )
    )
    return 0 if readiness_report["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
