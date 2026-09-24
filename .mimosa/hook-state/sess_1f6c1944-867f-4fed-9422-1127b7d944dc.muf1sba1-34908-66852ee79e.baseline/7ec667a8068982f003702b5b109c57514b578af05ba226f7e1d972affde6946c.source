"""从准备好的数据根目录构建轻量级数据集清单和场景清单。

这个模块会扫描数据根目录，记录有哪些序列目录存在，检查每个序列是
否包含必需的流文件，并输出两份紧凑结果：一份按序列组织的清单，一
份按场景组织的清单。这里故意保持简单，因为它服务的是准备流程，需
要的是可预测、适合脚手架使用的摘要，而不是完整的元数据模型。

上游依赖:
- 文件系统上的数据根目录（包含序列子目录）
- 每个序列子目录中的流文件（imu.json, uwb.json, vio.json, gt.json 等）

下游调用者:
- manifests.split_builder（需要数据集清单中的序列列表来构建切分）
- manifests.__init__.py（重新导出 build_manifests 和 REQUIRED_STREAMS）
- 准备流程脚本（需要清单来确认数据完整性）

核心变量:
- REQUIRED_STREAMS: 标准数据集默认要求的流文件名元组
- UTIL_REQUIRED_STREAMS: 工具型数据集要求的流文件名元组
- dataset_manifest: 按序列组织的数据集清单
- scene_manifest: 按场景组织的场景清单
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from liquidloc.dataio.manifests.dataset_checks import (
    REQUIRED_SEQ_FILES as REQUIRED_STREAMS,
    SIM_REQUIRED_SEQ_FILES as SIM_REQUIRED_STREAMS,
    UTIL_REQUIRED_SEQ_FILES as UTIL_REQUIRED_STREAMS,
    resolve_layout_family_from_dir,
)
from liquidloc.dataio.readers import list_sequence_dirs


def _resolve_layout_family(seq_dir: Path) -> Optional[str]:
    """从序列目录下的 anchor_layout.json 解析布局族编号。

    参数：
        seq_dir: 序列目录路径。

    返回：
        布局族编号字符串；无法解析时返回 None。
    """
    return resolve_layout_family_from_dir(seq_dir)


def build_manifests(
    data_root: str | Path,
    required_streams: Optional[tuple[str, ...]] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """扫描数据根目录，返回数据集清单和场景清单。

    参数：
        data_root：包含序列文件夹的根目录，可以是字符串路径或 Path 对象。
        required_streams：可选覆盖项，用来替换每个序列必须包含的文件集合。
            默认使用 REQUIRED_STREAMS。不允许传入空元组（会放宽合同）。

    返回：
        一个二元组 (dataset_manifest, scene_manifest)：
        - dataset_manifest: 数据集清单字典，包含 data_root、sequence_count、
          required_streams 和 sequences 字段
        - scene_manifest: 场景清单字典，包含 scene_count 和 scenes 字段

    异常：
        FileNotFoundError：当指定的数据根目录不存在或不是目录时抛出。
        TypeError：当 data_root 既不是字符串也不是 Path 对象时抛出。
        ValueError：当 required_streams 为空元组时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "data_root": str(data_root),
        "required_streams": required_streams,
    }, "build_manifests")
    if not isinstance(data_root, (str, Path)):
        raise TypeError("data_root must be a string or Path object")

    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Data root is not a directory: {os.fspath(root)}')

    # BUG-008 根因 (2026-09-06 §10 阶段 12 审计): sim_e9_main 顶层是 seed0..seed9 目录, 真正的
    # sim_meta.json 在 seed/seq 嵌套深处. has_sim_meta 必须递归扫描 (rglob),
    # 不能只看顶层 seq_dir / sim_meta.json.
    _top_seq_dirs = list_sequence_dirs(root)
    has_sim_meta = any(
        (seq_dir / 'sim_meta.json').is_file() for seq_dir in _top_seq_dirs
    ) or any(
        _sub.is_file()
        for seed_dir in _top_seq_dirs
        for _sub in seed_dir.rglob('sim_meta.json')
    )
    # §28.6 自证契约：当未显式指定 required_streams 时，自动检测仿真数据集
    # 根目录（任一序列目录含 sim_meta.json 即视为仿真路径），仿真路径必须
    # 满足 SIM_REQUIRED_SEQ_FILES（含 sim_meta.json），公开/真实数据集仍
    # 使用 REQUIRED_STREAMS。显式传入 required_streams 时不覆盖调用方意图。
    #
    # 阶段 12 全面审计修复 (2026-09-06 §10): 嵌套 sim 序列扫描必须在 effective_streams
    # 计算前完成, 否则 required_streams 非 None 时 (如 prepare_pipeline L342 写死传
    # REQUIRED_STREAMS) 直接跳过嵌套扫描, 走 list_sequence_dirs(root) 只返 10 seed 顶层
    # 目录. 这里把嵌套扫描逻辑移到 if/else 之外, 总是先做嵌套/平面识别.
    if has_sim_meta:
        _seq_dirs: list[Path] = []
        for seed_dir in _top_seq_dirs:
            if (seed_dir / 'sim_meta.json').is_file():
                _seq_dirs.append(seed_dir)
            else:
                _seq_dirs.extend(
                    child
                    for child in seed_dir.iterdir()
                    if child.is_dir() and (child / 'sim_meta.json').is_file()
                )
        _seq_dirs = sorted(_seq_dirs)
    else:
        _seq_dirs = _top_seq_dirs
    # 决定 required_streams 走哪条 (sim → SIM_REQUIRED_SEQ_FILES, 其他 → REQUIRED_STREAMS)
    if required_streams is None:
        effective_streams = SIM_REQUIRED_STREAMS if has_sim_meta else REQUIRED_STREAMS
    else:
        effective_streams = required_streams
    if not effective_streams:
        raise ValueError('required_streams must not be empty (would silently widen the contract)')

    sequence_records: list[dict[str, Any]] = []
    scene_index: dict[str, list[str]] = {}

    for seq_dir in _seq_dirs:
        file_names = sorted(path.name for path in seq_dir.iterdir() if path.is_file())
        missing_files = [name for name in effective_streams if name not in file_names]
        layout_family = _resolve_layout_family(seq_dir)
        if layout_family is not None:
            scene_id = layout_family
            scene_id_source = 'layout_family'
        else:
            scene_id = seq_dir.name
            scene_id_source = 'seq_dir_name'

        # BUG-008: sim_e9_main 嵌套结构下 seq_id 用正斜杠 (跨平台) 而非纯目录名
        # (Windows 下 relative_to 返回反斜杠, 不与 read_sim_sequence 路径拼接兼容)
        _rel_path = str(seq_dir.relative_to(root)).replace(os.sep, '/')
        record: dict[str, Any] = {
            'seq_id': _rel_path,
            'scene_id': scene_id,
            'scene_id_source': scene_id_source,
            'seq_dir': os.fspath(seq_dir),
            'file_names': file_names,
            'missing_files': missing_files,
            'is_complete': not missing_files,
        }
        sequence_records.append(record)
        scene_index.setdefault(scene_id, []).append(_rel_path)

    dataset_manifest: dict[str, Any] = {
        'data_root': os.fspath(root),
        'sequence_count': len(sequence_records),
        'required_streams': list(effective_streams),
        'sequences': sequence_records,
    }

    scene_manifest: dict[str, Any] = {
        'scene_count': len(scene_index),
        'scenes': [],
    }

    for scene_id, seq_ids in sorted(scene_index.items()):
        scene_manifest['scenes'].append({
            'scene_id': scene_id,
            'seq_ids': seq_ids,
        })

    return dataset_manifest, scene_manifest