"""准备阶段产物回放工具。

职责：
    把 prepare pipeline 产出的 manifest、事件和原始真值重新拼回
    后续 pipeline 需要的输入结构，方便复现和检查。

上游依赖：
    - Python 标准库 json / copy / pathlib / typing — JSON 读写、深拷贝与路径处理
    - liquidloc.common.constants — normalize_run_mode 运行模式归一化

下游调用者：
    - liquidloc.pipelines.train      — 训练流水线用本模块加载准备好的事件与真值
    - liquidloc.pipelines.eval       — 评估流水线用本模块加载预测输入
    - scripts / smoke tests          — 脚本用本模块检查准备阶段产物完整性

核心变量：
    - 无模块级变量，全部通过函数返回
"""

from __future__ import annotations  # 允许类型注解在文件内后面再定义或引用。

import json  # 用标准库读取 JSON 工件。
from copy import deepcopy  # 用深拷贝避免修改原始元数据。
from pathlib import Path  # 用 Path 统一处理路径。
from typing import Any  # 用 Any 接住不确定结构。

from liquidloc.common.validation import is_integer as _is_integer, is_string_like  # numpy 2.0+ 兼容的整数和字符串类型检查。
from liquidloc.common.validation import validate_path_component  # 路径组件穿越校验。
from liquidloc.common.constants import normalize_run_mode  # 纯字符串级运行模式归一化（common 层单源真相）。

__all__ = (
    "load_prepare_manifest",
    "resolve_seq_ids_from_prepare_manifest",
    "resolve_experiment_seq_ids_from_prepare_manifest",
    "load_prepared_events_by_seq_id",
    "load_ground_truth_by_seq_id",
    "load_source_report_by_seq_id",
    "expand_scene_tasks_across_seq_ids",
)


def _load_json(path: Path) -> Any:  # 读取一个必须存在的 JSON 工件。
    """读取一个必须存在的 JSON 工件。"""
    if not path.is_file():  # 文件不存在或不是普通文件都不行。
        raise FileNotFoundError(f"required artifact not found: {path}")  # 找不到工件就报错。
    from liquidloc.common.io_utils import read_json
    return read_json(path)  # 直接按严格 JSON 语义读取工件。


def _load_pickle_gz(path: Path) -> Any:  # 读取一个必须存在的 Pickle+Gzip 工件。
    """读取 .pkl.gz 格式的序列化工件。"""
    if not path.is_file():
        raise FileNotFoundError(f"required artifact not found: {path}")
    import gzip
    import pickle
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


def _load_events_file(prepare_root: Path, seq_id: str) -> list[dict[str, Any]]:
    """加载单个序列的事件数据，优先使用 .pkl.gz，回退到 .json。

    返回值必须是 list[dict[str, Any]]，任何格式不匹配都会抛出 TypeError。
    """
    # BUG-015 修复 (2026-09-06 §10 阶段 12 审计): sim 嵌套 seq_id 'seed0/sim_curve_01'
    # 在 prepare 阶段被 BUG-009 修复替换 / 为 __ 作文件系统安全名 (seed0__sim_curve_01).
    # 这里读取时也做同样替换才能匹配磁盘文件. 不替换则 FileNotFoundError.
    safe_seq_id = seq_id.replace('/', '__') if '/' in seq_id else seq_id
    pkl_gz_path = prepare_root / f"{safe_seq_id}_events.pkl.gz"
    json_path = prepare_root / f"{safe_seq_id}_events.json"
    if pkl_gz_path.is_file():
        payload = _load_pickle_gz(pkl_gz_path)
        if not isinstance(payload, list):
            raise TypeError(f"{seq_id}_events.pkl.gz must contain a list payload, got {type(payload).__name__}")
        return payload
    if json_path.is_file():
        payload = _load_json(json_path)
        if not isinstance(payload, list):
            raise TypeError(f"{seq_id}_events.json must contain a list payload, got {type(payload).__name__}")
        return payload
    raise FileNotFoundError(
        f"Neither {safe_seq_id}_events.pkl.gz nor {safe_seq_id}_events.json found in {prepare_root}"
    )


def load_prepare_manifest(prepare_root: str | Path) -> dict[str, Any]:  # 读取准备阶段总清单。
    """读取准备阶段总清单，并保证返回值是映射。

    Args:
        prepare_root (str | Path): 准备阶段输出目录的路径，
            该目录下必须存在 prepare_manifest.json 文件。

    Returns:
        dict[str, Any]: 清单字典，包含 sequences 等顶层键。

    Raises:
        FileNotFoundError: 当 prepare_manifest.json 不存在时抛出。
        TypeError: 当清单顶层不是映射（dict）时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"prepare_root": str(prepare_root)}, "load_prepare_manifest 入口参数")
    prepare_root_path = Path(prepare_root).resolve()  # 先把输入路径统一成绝对路径。
    payload = _load_json(prepare_root_path / "prepare_manifest.json")  # 读取固定名的总清单。
    if not isinstance(payload, dict):  # 顶层必须是映射。
        raise TypeError(f"prepare_manifest.json must contain a mapping payload, got {type(payload).__name__}: {prepare_root_path}")  # 不是映射就报错。
    # 校验必需键，避免结构漂移时错误延迟到下游消费端才暴露。
    _REQUIRED_MANIFEST_KEYS = ("sequences",)
    missing_keys = [k for k in _REQUIRED_MANIFEST_KEYS if k not in payload]
    if missing_keys:
        raise ValueError(f"prepare_manifest.json missing required keys: {missing_keys}")
    # sequences 的值必须是映射，否则下游 .keys() 调用会抛不友好的 AttributeError。
    if not isinstance(payload["sequences"], dict):
        raise TypeError(f"prepare_manifest.sequences must be a mapping, got {type(payload['sequences']).__name__}")
    return payload  # 返回清单字典。


def resolve_seq_ids_from_prepare_manifest(
    prepare_manifest: dict[str, Any],
    seq_ids: list[str] | None = None,
) -> list[str]:  # 返回最终要用的序列 ID 列表。
    """从准备清单里解析可用序列 ID。

    如果调用方未指定 seq_ids，则返回清单中的全部序列；
    如果指定了子集，则校验子集中的每个 ID 都存在于清单中。

    Args:
        prepare_manifest (dict[str, Any]): 准备阶段总清单，必须包含
            非空的 sequences 键。
        seq_ids (list[str] | None): 调用方指定的序列子集，None 表示使用全部。

    Returns:
        list[str]: 最终要使用的序列 ID 列表，已去除空白、空项和重复项，
            并保持调用方给定顺序。

    Raises:
        ValueError: 当清单中序列为空，或请求的 seq_ids 有缺失时抛出。
    """
    available_sequences = list((prepare_manifest.get("sequences") or {}).keys())  # 先取出清单里的全部序列。
    if not available_sequences:  # 如果清单里没有序列，就说明准备阶段不完整。
        raise ValueError("prepare_manifest.sequences must be non-empty, but the sequences mapping is empty or missing")  # 空序列表不允许。
    if seq_ids is None:  # 调用方没指定子集时，就默认用全部可用序列。
        return available_sequences  # 直接返回全部序列。

    normalized_seq_ids: list[str] = []  # 收集清洗后且保持顺序唯一的序列 ID。
    seen_seq_ids: set[str] = set()  # 避免同一序列重复进入后续采样链。
    for seq_id in seq_ids:  # 逐个清洗调用方请求的序列。
        normalized_seq_id = str(seq_id).strip()  # 统一转成去空白字符串。
        if not normalized_seq_id or normalized_seq_id in seen_seq_ids:  # 空项或重复项都直接跳过。
            continue
        # BUG-014 修复 (2026-09-06 §10 阶段 12 审计): sim 嵌套 seq_id 含 '/', 跳过 path traversal 校验.
        if '/' not in normalized_seq_id:
            validate_path_component(normalized_seq_id, name="seq_id")
        normalized_seq_ids.append(normalized_seq_id)  # 保留首次出现的序列。
        seen_seq_ids.add(normalized_seq_id)  # 记录已见过的序列。

    missing_seq_ids = [seq_id for seq_id in normalized_seq_ids if seq_id not in available_sequences]  # 找出请求但不存在的序列。
    if missing_seq_ids:  # 只要有缺失就不能继续。
        raise ValueError(  # 这里明确告诉调用方缺了哪些序列。
            "requested seq_ids are missing from prepare_manifest: " + ", ".join(missing_seq_ids)  # 把缺失项拼出来。
        )  # raise 参数结束。
    return normalized_seq_ids  # 返回清洗后的合法序列列表。


def _coerce_positive_int(value: Any, *, name: str) -> int:  # 校验 mode_overrides 里的正整数配置。
    """把 mode_overrides 中的正整数配置校验并转成 int。"""
    if not _is_integer(value):  # bool 和非整数都拒绝，兼容 numpy 2.0+ 的 np.integer。
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:  # 上限必须是正整数。
        raise ValueError(f"{name} must be positive when provided")
    return int(value)  # 统一转成 Python int。


def resolve_experiment_seq_ids_from_prepare_manifest(  # 按实验 quick/full 合同解析要跑的序列列表。
    prepare_manifest: dict[str, Any],  # 准备阶段总清单。
    *,
    experiment_cfg: dict[str, Any] | None = None,  # 当前实验配置。
    seq_ids: list[str] | None = None,  # 调用方显式指定的序列子集。
    default_mode: str = "full",  # 缺省模式名。
) -> list[str]:
    """在 prepare manifest 的可用序列上叠加实验级 quick/full 序列预算合同。

    规则：
    1. 如果调用方显式传入 ``seq_ids``，直接按显式子集校验并返回，不再套用
       ``mode_overrides.max_sequences``，避免覆盖更高优先级的显式实验子集。
    2. 如果未显式传入 ``seq_ids``，则先取 prepare manifest 的全部可用序列，
       再读取当前实验 active mode 的 ``mode_overrides.max_sequences`` 做截断。

    注意：本函数承载了实验级 quick/full 截断逻辑，属于 common 层的边界情况。
    截断规则由 ``experiment_cfg`` 参数驱动，而非硬编码业务规则。截断仅裁
    序列维度，不改 scene_tasks 本体，不涉及指标口径或数据合同变更。
    """
    available_seq_ids = resolve_seq_ids_from_prepare_manifest(prepare_manifest, seq_ids)
    if seq_ids is not None or not experiment_cfg:  # 显式序列子集优先，或没有实验配置时直接返回。
        return available_seq_ids

    mode = normalize_run_mode(experiment_cfg.get("mode"), default_mode=default_mode)  # 取当前实验真正生效的模式。
    mode_cfg = dict((experiment_cfg.get("mode_overrides") or {}).get(mode) or {})  # 读取该模式的覆盖配置。
    max_sequences = mode_cfg.get("max_sequences")
    if max_sequences is None:  # 没有限制就保持全部可用序列。
        return available_seq_ids
    max_count = _coerce_positive_int(max_sequences, name=f"mode_overrides.{mode}.max_sequences")
    return available_seq_ids[:max_count]  # 只裁序列维度，不改 scene_tasks 本体。


class _LazyEventsDict(dict):
    """惰性加载事件字典：按需从磁盘读取事件，避免一次性加载全部数据。

    由于 1860 个序列的事件文件总大小约 47 GB，一次性加载会超出内存限制。
    本类继承 dict，只在访问某个 seq_id 时才加载对应的事件文件，
    并保持已加载的事件以字典缓存。超过 CAP 个序列后用 LRU 策略驱逐最旧的。
    """
    _CAP = 8  # 缓存上限：超过 8 个序列时驱逐最久未访问的（每个 ~25MB，8 个约 200MB 上界）

    def __init__(self, prepare_root: Path, seq_ids: list[str]):
        super().__init__()
        self._prepare_root = prepare_root
        self._seq_ids = list(seq_ids)
        self._loaded_keys: set[str] = set()
        self._lru_order: list[str] = []  # 最近访问顺序，最旧在前

    def __getitem__(self, seq_id: str) -> list[dict[str, Any]]:
        if seq_id not in self._loaded_keys:
            # 首次访问时从磁盘加载（优先 .pkl.gz，回退到 .json）
            payload = _load_events_file(self._prepare_root, seq_id)
            # LRU 驱逐：超过 CAP 时 evict 最旧序列
            if len(self._lru_order) >= self._CAP:
                evict_key = self._lru_order.pop(0)
                super().__delitem__(evict_key)
                self._loaded_keys.discard(evict_key)
            super().__setitem__(seq_id, payload)
            self._loaded_keys.add(seq_id)
            self._lru_order.append(seq_id)
        else:
            # 已加载：更新 LRU 顺序（移到末尾表示最近访问）
            self._lru_order.remove(seq_id)
            self._lru_order.append(seq_id)
        return super().__getitem__(seq_id)

    def get(self, seq_id: str, default=None):
        try:
            return self[seq_id]
        except (FileNotFoundError, TypeError):
            return default

    def __contains__(self, seq_id: str) -> bool:
        # BUG-015 修复: 嵌套 sim seq_id 文件名用 __ 替换 /, 这里同步处理.
        safe_seq_id = seq_id.replace('/', '__') if '/' in seq_id else seq_id
        return (
            seq_id in self._loaded_keys
            or (self._prepare_root / f"{safe_seq_id}_events.pkl.gz").is_file()
            or (self._prepare_root / f"{safe_seq_id}_events.json").is_file()
        )

    def __iter__(self):
        return iter(self._seq_ids)

    def __len__(self):
        return len(self._seq_ids)

    def __repr__(self):
        return f"_LazyEventsDict(len={len(self._seq_ids)}, loaded={len(self._loaded_keys)})"

    def keys(self):
        return iter(self._seq_ids)

    def values(self):
        for key in self._seq_ids:
            yield self[key]

    def items(self):
        for key in self._seq_ids:
            yield key, self[key]


def load_prepared_events_by_seq_id(
    prepare_root: str | Path,
    seq_ids: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """按序列 ID 读取准备阶段事件列表（惰性加载）。

    对每个序列 ID，从 prepare_root 下读取 {seq_id}_events.json 文件。
    采用惰性加载策略，只在访问某个 seq_id 时才加载对应的事件文件，
    避免一次性加载全部数据导致内存溢出。

    Args:
        prepare_root (str | Path): 准备阶段输出目录的路径。
        seq_ids (list[str]): 要读取的序列 ID 列表。

    Returns:
        _LazyEventsDict: seq_id 到事件列表的惰性映射，
            每个事件是一个字典。

    Raises:
        FileNotFoundError: 当某个序列的事件文件不存在时抛出。
        TypeError: 当事件文件顶层不是列表时抛出。
    """

    prepare_root_path = Path(prepare_root).resolve()  # 统一转成绝对路径。
    for seq_id in seq_ids:  # 校验所有 seq_id 有效。
        # BUG-014 修复 (2026-09-06 §10 阶段 12 审计): sim_e9_main 嵌套 seq_id 含 '/' (seed0/sim_curve_01),
        # build_manifests 已生成正确的嵌套 seq_id, 这里对嵌套路径跳过通用 path traversal 检查.
        if '/' not in seq_id:
            validate_path_component(seq_id, name="seq_id")
    return _LazyEventsDict(prepare_root_path, seq_ids)  # 返回惰性加载字典。


class _LazyGroundTruthDict(dict):
    """惰性加载真值字典：按需从磁盘读取真值，避免一次性加载全部数据。

    采用 LRU 缓存策略，超过 CAP 个序列时驱逐最久未访问的（每个 ~1MB，CAP=64 约 64MB 上界）。
    """
    _CAP = 64  # 真值文件较小（~1MB），缓存上限放宽到 64

    def __init__(self, raw_root: str | Path, seq_ids: list[str]):
        super().__init__()
        self._raw_root = Path(raw_root).resolve()
        self._seq_ids = list(seq_ids)
        self._loaded_keys: set[str] = set()
        self._lru_order: list[str] = []

    def __getitem__(self, seq_id: str) -> list[dict[str, float]]:
        if seq_id not in self._loaded_keys:
            payload = _load_json(self._raw_root / seq_id / "gt.json")
            if not isinstance(payload, list):
                raise TypeError(f"{seq_id}/gt.json must contain a list payload, got {type(payload).__name__}")
            if len(self._lru_order) >= self._CAP:
                evict_key = self._lru_order.pop(0)
                super().__delitem__(evict_key)
                self._loaded_keys.discard(evict_key)
            super().__setitem__(seq_id, payload)
            self._loaded_keys.add(seq_id)
            self._lru_order.append(seq_id)
        else:
            self._lru_order.remove(seq_id)
            self._lru_order.append(seq_id)
        return super().__getitem__(seq_id)

    def get(self, seq_id: str, default=None):
        try:
            return self[seq_id]
        except (FileNotFoundError, TypeError):
            return default

    def __contains__(self, seq_id: str) -> bool:
        return seq_id in self._loaded_keys or (self._raw_root / seq_id / "gt.json").is_file()

    def __iter__(self):
        return iter(self._seq_ids)

    def __len__(self):
        return len(self._seq_ids)

    def __repr__(self):
        return f"_LazyGroundTruthDict(len={len(self._seq_ids)}, loaded={len(self._loaded_keys)})"

    def keys(self):
        return iter(self._seq_ids)

    def values(self):
        for key in self._seq_ids:
            yield self[key]

    def items(self):
        for key in self._seq_ids:
            yield key, self[key]


def load_ground_truth_by_seq_id(  # 按序列 ID 读取原始真值。
    raw_root: str | Path,  # 原始数据根目录。
    seq_ids: list[str],  # 要读取的序列 ID 列表。
) -> dict[str, list[dict[str, Any]]]:  # 返回 seq_id -> ground truth 的映射。
    """按序列 ID 读取原始真值轨迹（惰性加载）。

    采用惰性加载策略，只在访问某个 seq_id 时才从磁盘加载真值文件，
    避免一次性加载全部数据导致内存溢出。

    Args:
        raw_root (str | Path): 原始数据根目录。
        seq_ids (list[str]): 要读取的序列 ID 列表。

    Returns:
        _LazyGroundTruthDict: seq_id -> 真值列表的惰性映射。
    """
    raw_root_path = Path(raw_root).resolve()  # 统一转成绝对路径。
    for seq_id in seq_ids:  # 校验所有 seq_id 有效。
        # BUG-014 修复 (2026-09-06 §10 阶段 12 审计): 嵌套 sim seq_id 含 '/', 跳过 path traversal 校验.
        if '/' not in seq_id:
            validate_path_component(seq_id, name="seq_id")
    return _LazyGroundTruthDict(raw_root_path, seq_ids)  # 返回惰性加载真值字典。


def load_source_report_by_seq_id(
    raw_root: str | Path,
    prepare_manifest: dict[str, Any],
    seq_ids: list[str],
    *,
    default_source: str,
) -> dict[str, dict[str, Any]]:
    """为每个序列构造来源报告。"""
    raw_root_path = Path(raw_root).resolve()  # 统一转成绝对路径。
    sequence_meta = dict(prepare_manifest.get("sequences") or {})  # 拿到准备清单里的序列元数据。
    source_report_by_seq_id: dict[str, dict[str, Any]] = {}  # 收集每个序列的来源报告。
    for seq_id in seq_ids:  # 逐个序列处理。
        report = {  # 先构造最基础的报告。
            "source": default_source,  # 默认来源标签。
            "prepare_sequence": deepcopy(dict(sequence_meta.get(seq_id) or {})),  # 复制准备阶段的序列元数据。
            "anchor_layout_metadata_available": False,  # 默认无锚点布局元数据，与 miluv_reader.py 的显式默认值模式一致。
        }  # 基础报告结束。
        # 从 prepare_manifest 的序列元数据中提取 scene_parameters，下沉到 source_report。
        # 直接从已 deepcopy 的 report["prepare_sequence"] 中读取，避免重复读取同一条目。
        scene_parameters = report["prepare_sequence"].get("scene_parameters")  # 取出协议展开后的场景参数。
        if scene_parameters is not None:  # 如果存在，写入 source_report。
            report["scene_parameters"] = deepcopy(scene_parameters)  # 深拷贝避免污染。
        anchor_layout_path = raw_root_path / seq_id / "anchor_layout.json"  # 看看这个序列有没有锚点布局文件。
        if anchor_layout_path.is_file():  # 如果存在，就补充锚点布局信息。
            anchor_layout = _load_json(anchor_layout_path)  # 读取锚点布局 JSON。
            if not isinstance(anchor_layout, dict):  # 布局文件顶层必须是映射。
                raise TypeError(f"anchor_layout.json for {seq_id} must contain a mapping payload, got {type(anchor_layout).__name__}")  # 类型不对就报错。
            report["anchor_layout"] = deepcopy(anchor_layout)  # 防御性拷贝，避免与 anchor_layout_metadata 共享引用。
            report["anchor_layout_source"] = "raw_anchor_layout_json"  # 标明来源是原始 JSON。
            report["anchor_layout_metadata"] = deepcopy(anchor_layout)  # 再放一份元数据副本。
            report["anchor_layout_metadata_source"] = "raw_anchor_layout_json"  # 元数据来源也写清楚。
            report["anchor_layout_metadata_available"] = True  # 标记锚点布局元数据可用，供下游条件判断。
            anchor_positions = anchor_layout.get("anchor_positions") or []  # 取出锚点坐标列表。
            if (  # 如果锚点坐标看起来像二维或更高维数组。
                isinstance(anchor_positions, list)  # 必须是列表。
                and anchor_positions  # 列表不能空。
                and isinstance(anchor_positions[0], list)  # 第一项也要是列表。
                and anchor_positions[0]  # 第一项也不能空。
            ):  # 结构检查结束。
                report["anchor_layout_position_dim"] = len(anchor_positions[0])  # 记录坐标维度，便于下游判断。
        source_report_by_seq_id[seq_id] = report  # 把这个序列的报告放进结果字典。
    return source_report_by_seq_id  # 返回来源报告。


def expand_scene_tasks_across_seq_ids(  # 把场景任务展开到多个序列上。
    scene_tasks: list[dict[str, Any]],  # 输入的场景任务列表。
    seq_ids: list[str],  # 目标序列列表。
) -> list[dict[str, Any]]:  # 返回展开后的任务列表。
    """把场景任务展开到多个序列 ID 上。

    对每个基础场景任务，如果任务自身指定了 seq_id 则只展开到那一个序列，
    否则复制到全部目标序列。每个展开后的任务会获得唯一的 task_id 和
    scene_variant_id。

    Args:
        scene_tasks (list[dict[str, Any]]): 基础场景任务列表，每个任务
            必须是字典，可包含可选的 seq_id 键。
        seq_ids (list[str]): 目标序列 ID 列表，当任务未指定 seq_id 时
            展开到这些序列上。

    Returns:
        list[dict[str, Any]]: 展开后的任务列表，每个任务包含：
            - task_id (str): 格式为 "scene_{编号:04d}" 的唯一标识。
            - seq_id (str): 分配到的序列 ID。
            - scene_variant_id (str): 格式为 "{scene_id}::{seq_id}" 的变体标识。

    Raises:
        TypeError: 当 scene_tasks 中的某项不是字典时抛出。
        ValueError: 当 scene_tasks 中的某项缺少 scene_id 键时抛出。

    注意：task_id 使用 4 位零填充格式（:04d），确保 10000 个以内任务编号
    不产生排序歧义。如需与其他模块的 2 位格式（:02d）对齐，需同步修改。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "scene_tasks_count": len(scene_tasks) if isinstance(scene_tasks, (list, tuple)) else None,
        "seq_ids": seq_ids,
    }, "expand_scene_tasks_across_seq_ids 入口参数")
    expanded_tasks: list[dict[str, Any]] = []  # 收集展开后的任务。
    task_index = 0  # 用连续编号生成 task_id。
    for base_task_index, base_task in enumerate(scene_tasks):  # 逐个基础场景任务展开。
        if not isinstance(base_task, dict):  # 每个任务必须是字典。
            raise TypeError(f"scene_tasks[{base_task_index}] must be a mapping, got {type(base_task).__name__}")  # 类型不对就报错。
        if "scene_id" not in base_task:  # scene_id 是场景任务的必需键。
            raise ValueError(f"scene_tasks[{base_task_index}] must contain a 'scene_id' key, got keys: {sorted(base_task.keys())}")  # 缺少 scene_id 就报错。
        base_task_seq_id = base_task.get("seq_id")  # 看这个任务自己有没有固定 seq_id。
        if is_string_like(base_task_seq_id) and str(base_task_seq_id).strip():  # 任务指定了有效 seq_id。
            normalized_task_seq_id = str(base_task_seq_id).strip()  # 去空白。
            if normalized_task_seq_id not in seq_ids:  # 校验指定的 seq_id 是否在可用序列中。
                raise ValueError(f"scene_tasks[{base_task_index}] specifies seq_id {normalized_task_seq_id!r} which is not in the available seq_ids: {seq_ids}")  # 不在可用序列中就报错。
            target_seq_ids = [normalized_task_seq_id]  # 只展开到那一个序列。
        else:  # 任务未指定 seq_id，展开到全部序列。
            target_seq_ids = list(seq_ids)
        for seq_id in target_seq_ids:  # 针对每个目标序列复制一个任务。
            task = deepcopy(base_task)  # 复制任务，避免污染原始基础任务。
            task["task_id"] = f"scene_{task_index:04d}"  # 给展开后的任务一个稳定编号。
            task["seq_id"] = seq_id  # 把当前目标序列写进去。
            if not task.get("scene_variant_id"):  # 如果没有现成的变体编号，就自己拼一个。
                task["scene_variant_id"] = f"{task['scene_id']}::{seq_id}"  # 用 scene_id 和 seq_id 组合成变体 id。
            else:  # 如果已有 scene_variant_id，就在后面追加 seq_id。
                task["scene_variant_id"] = f"{task['scene_variant_id']}::{seq_id}"  # 让变体编号更明确。
            expanded_tasks.append(task)  # 把这个展开任务加入结果列表。
            task_index += 1  # 下一个任务编号加一。
    return expanded_tasks  # 返回展开后的任务列表。
