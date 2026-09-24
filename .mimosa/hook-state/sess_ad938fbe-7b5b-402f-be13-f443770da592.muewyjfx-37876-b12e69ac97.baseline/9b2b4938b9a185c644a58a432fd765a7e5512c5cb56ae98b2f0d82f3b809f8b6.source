"""断点恢复模块。

提供并发调度器的 checkpoint resume 能力。
调度器在训练中断后可从此模块恢复，无需全量重跑。

与 `src/liquidloc/models/liquid/_checkpoint_utils.py` 的对接方式：
- 该文件负责原子保存与校验
- 本文件提供面向调度器的高层 resume 入口
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from liquidloc.common.io_utils import loads_json_text
from liquidloc.common.types import ModelIntermediate


def load_checkpoint_state(
    checkpoint_dir: Path,
    unit_id: str,
) -> dict[str, Any] | None:
    """从 checkpoint 目录加载指定 unit 的断点状态。

    返回 None 表示无可用断点（从头开始），否则返回包含
    'progress'、'last_completed'、'pending_units' 等字段的字典。
    """
    state_file = checkpoint_dir / f"{unit_id}_checkpoint.json"
    if not state_file.is_file():
        return None
    try:
        payload: dict[str, Any] = json.loads(
            loads_json_text(state_file.read_text(encoding="utf-8"))
        )
    except Exception:
        return None
    # 基本结构校验
    required_keys = {"progress", "last_completed"}
    if not required_keys.issubset(payload.keys()):
        return None
    return payload


def save_checkpoint_state(
    checkpoint_dir: Path,
    unit_id: str,
    progress: dict[str, Any],
    last_completed: int,
    pending_units: list[int],
) -> Path:
    """保存当前 unit 的断点状态到 checkpoint 目录。

    写入原子操作（先写临时文件再 rename），防止半文件。
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_file = checkpoint_dir / f"{unit_id}_checkpoint.json"
    tmp_file = checkpoint_dir / f"{unit_id}_checkpoint.tmp"
    payload = {
        "progress": progress,
        "last_completed": last_completed,
        "pending_units": pending_units,
    }
    tmp_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tmp_file.replace(state_file)
    return state_file


def resume_from_checkpoint(
    checkpoint_dir: Path,
    total_units: int,
    start_unit: int,
) -> int:
    """并发调度器调用入口：决定从第几个 unit 继续运行。

    返回 0 表示全量重跑（无断点），否则返回下一个 unit 索引。
    """
    state = load_checkpoint_state(checkpoint_dir, "concurrent_scheduler")
    if state is None:
        return 0
    last_completed = int(state.get("last_completed", -1))
    # 如果 last_completed >= start_unit，从 last_completed+1 继续
    return max(start_unit, last_completed + 1)
