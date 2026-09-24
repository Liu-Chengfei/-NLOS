"""公开数据集注册表的加载与查询工具。

本模块负责加载 configs/datasets/public_dataset_registry.yaml 中定义的
公开数据集注册表，并提供按名称查询、枚举和就绪状态检查等能力。注册表
是数据集发现和准入控制的中心：只有注册在案的数据集才能进入公开基准
烟雾测试和论文级全量验证。

核心概念：
- **注册表（registry）**：YAML 文件中定义的数据集条目集合，每个条目
  包含数据集名称、发布层级（release_tier）、就绪标记等元数据。
- **发布层级（release_tier）**：smoke_only / full_ready / paper_ready，
  控制数据集能参与哪些验证流程。
- **公开基准烟雾就绪（public_benchmark_smoke_ready）**：只有同时满足
  "在协议允许的公开基准数据集面上" 和 "注册表标记为烟雾就绪" 两个条件，
  才算真正可用于公开基准烟雾测试。

上游依赖：
- configs/datasets/public_dataset_registry.yaml（注册表定义）
- liquidloc.common.config_utils（YAML 配置加载）
- liquidloc.common.paths（项目根目录解析）
- liquidloc.protocol.experiment_gates（normalize_public_dataset_name 数据集名称规范化）

下游调用者：
- dataio.registry.__init__.py（重新导出查询和就绪检查函数）
- 准备流程脚本（查询数据集条目和就绪状态）
- 公开验证流程（确认数据集准入资格）
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from liquidloc.common.config_utils import load_yaml_config
from liquidloc.common.paths import get_project_root
from liquidloc.common.validation import is_bool_like, is_integer, is_string_like
from liquidloc.protocol.experiment_gates import (
    get_public_benchmark_allowed_datasets,
    normalize_public_dataset_name,  # 名称规范化已迁至协议层，本模块保留重导出。
)

_SMOKE_READY_TIER = "smoke_only"
_FULL_READY_TIERS = {"full_ready", "paper_ready"}
# 已知的发布层级取值集合，用于在加载时校验 release_tier 拼写正确性。
_KNOWN_RELEASE_TIERS = frozenset({_SMOKE_READY_TIER} | _FULL_READY_TIERS)
# 可选布尔字段：若存在则必须为布尔类型，拒绝 truthiness 强转。
_OPTIONAL_BOOL_FIELDS = ("paper_full_ready", "public_benchmark_smoke_ready")


def load_public_dataset_registry(
    registry_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """加载公开数据集注册表 YAML 并校验顶层结构。

    参数：
        registry_path: 注册表 YAML 文件路径；为 None 时使用项目默认路径
            configs/datasets/public_dataset_registry.yaml。

    返回：
        注册表配置字典，顶层应包含 datasets 键，其值为数据集条目字典。

    异常：
        ValueError: 注册表为空或顶层 datasets 键缺失/无效时抛出。
        FileNotFoundError: 注册表文件不存在时抛出。
        TypeError: registry_path 类型不支持时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"registry_path": str(registry_path) if registry_path is not None else None}, "load_public_dataset_registry")
    if registry_path is None:
        registry_path = get_project_root() / "configs" / "datasets" / "public_dataset_registry.yaml"
    elif is_string_like(registry_path):
        registry_path = Path(str(registry_path))

    if not isinstance(registry_path, Path):
        raise TypeError("registry_path must be a string, Path, or None")

    registry_cfg = load_yaml_config(registry_path)
    datasets = registry_cfg.get("datasets")

    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("public dataset registry must contain a non-empty datasets mapping")

    # 校验每个数据集条目包含必需字段。
    # 合同声明：必需字段集合在此代码中定义并强制执行，是注册表数据合同的
    # 唯一执行点。协议文档（experiment_gates）定义的是"哪些数据集允许进入
    # 公开基准面"，而此处定义的是"注册表条目必须具备哪些字段才能合法"。
    # 两者职责不同，不可互相替代。
    _REQUIRED_DATASET_KEYS = {"reader", "useful_assets", "release_tier"}
    # 必需字段的值类型约束：reader 为非空字符串，useful_assets 为列表，
    # release_tier 为非空字符串。
    _REQUIRED_KEY_TYPES = {
        "reader": (str, "a non-empty string"),
        "useful_assets": (list, "a list"),
        "release_tier": (str, "a non-empty string"),
    }
    # 从协议层获取官方公开基准数据集面，避免硬编码与协议漂移。
    official_public_benchmark_datasets = _get_official_public_benchmark_datasets()
    for ds_name, ds_entry in datasets.items():
        if not isinstance(ds_entry, dict):
            raise ValueError(f"registry entry '{ds_name}' must be a mapping")
        missing = _REQUIRED_DATASET_KEYS - set(ds_entry.keys())
        if missing:
            raise ValueError(f"registry entry '{ds_name}' missing required keys: {sorted(missing)}")
        # 校验必需字段的值类型和有效性。
        for key, (expected_type, type_desc) in _REQUIRED_KEY_TYPES.items():
            value = ds_entry[key]
            if not isinstance(value, expected_type):
                raise ValueError(
                    f"registry entry '{ds_name}'.{key} must be {type_desc}, got {type(value).__name__}"
                )
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"registry entry '{ds_name}'.{key} must not be blank")
            if isinstance(value, list) and len(value) == 0:
                raise ValueError(f"registry entry '{ds_name}'.{key} must not be empty")
        # 校验 useful_assets 元素类型：每个元素必须是非空字符串，防止
        # 非字符串元素（如数字、字典）混入资产列表导致下游路由异常。
        for asset_index, asset in enumerate(ds_entry["useful_assets"]):
            if not is_string_like(asset):
                raise ValueError(
                    f"registry entry '{ds_name}'.useful_assets[{asset_index}] must be a string, "
                    f"got {type(asset).__name__}"
                )
            if not str(asset).strip():
                raise ValueError(
                    f"registry entry '{ds_name}'.useful_assets[{asset_index}] must not be blank"
                )
        # 校验 release_tier 取值在已知集合内，避免拼写错误（如 smok_only）
        # 静默导致 registry_smoke_ready 默认为 False。
        release_tier_value = ds_entry["release_tier"]
        if release_tier_value not in _KNOWN_RELEASE_TIERS:
            raise ValueError(
                f"registry entry '{ds_name}'.release_tier must be one of "
                f"{sorted(_KNOWN_RELEASE_TIERS)}, got {release_tier_value!r}"
            )
        # 校验可选布尔字段：若存在则必须为布尔类型，拒绝 truthiness 强转
        # （避免 YAML 中误写 paper_full_ready: "false" 字符串被 bool() 判为 True）。
        for bool_field in _OPTIONAL_BOOL_FIELDS:
            if bool_field in ds_entry:
                raw_value = ds_entry[bool_field]
                if raw_value is not None and not is_bool_like(raw_value):
                    raise TypeError(
                        f"registry entry '{ds_name}'.{bool_field} must be a boolean, "
                        f"got {type(raw_value).__name__}"
                    )

        # 公开基准面数据集必须包含 frozen_public_eval_seq_ids。
        # 数据集面来源为协议层 _get_official_public_benchmark_datasets()，
        # 与 experiment_protocol.yaml 的 public_benchmark.allowed_datasets 对齐。
        if ds_name in official_public_benchmark_datasets and "frozen_public_eval_seq_ids" not in ds_entry:
            raise ValueError(f"public benchmark dataset '{ds_name}' must contain frozen_public_eval_seq_ids")

    return registry_cfg


def get_dataset_entry(dataset_name: str, registry_cfg: dict[str, Any]) -> dict[str, Any]:
    """按名称从注册表中取出一条数据集条目（深拷贝），并附加标准化名称。

    参数：
        dataset_name: 数据集名称，会被标准化后用于查找。
        registry_cfg: 已加载的注册表配置字典。

    返回：
        该数据集条目的副本，包含额外添加的 dataset_name 字段。

    异常：
        KeyError: 数据集名称不在注册表中时抛出，附带可用名称列表。
        ValueError: 注册表条目中已存在 dataset_name 字段且与标准化名称冲突时抛出。
        TypeError: 参数类型不正确时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "dataset_name": dataset_name,
        "registry_cfg_keys": list(registry_cfg.keys()) if isinstance(registry_cfg, dict) else None,
    }, "get_dataset_entry")
    if not isinstance(registry_cfg, dict):
        raise TypeError("registry_cfg must be a dict")

    normalized_name = normalize_public_dataset_name(dataset_name)
    datasets = registry_cfg.get("datasets", {})

    # 校验 datasets 字段类型：若为字符串，`in` 运算会做子串匹配导致误命中，
    # 必须显式拒绝非映射类型。
    if not isinstance(datasets, dict):
        raise TypeError("registry_cfg['datasets'] must be a mapping")

    if normalized_name not in datasets:
        available = ", ".join(sorted(datasets.keys()))
        raise KeyError(f"Unknown public dataset: {normalized_name}. Available: {available}")

    entry = deepcopy(datasets[normalized_name])

    # 校验条目类型：非字典条目会导致后续 .get 调用失败，需提前拒绝。
    if not isinstance(entry, dict):
        raise TypeError(
            f"registry entry '{normalized_name}' must be a mapping, got {type(entry).__name__}"
        )

    if "dataset_name" in entry and entry["dataset_name"] != normalized_name:
        raise ValueError(
            f"Registry entry already has dataset_name={entry['dataset_name']!r} "
            f"which conflicts with normalized name {normalized_name!r}"
        )

    entry["dataset_name"] = normalized_name
    return entry


def _normalize_public_seq_id_list(raw_seq_ids: object, *, field_name: str) -> list[str]:
    """校验并标准化公开序列合同中的 seq_ids 列表。

    参数：
        raw_seq_ids: 待校验的序列 ID 列表，可以是任何类型。
        field_name: 字段名，用于错误消息中标识来源。

    返回：
        标准化后的序列 ID 列表，所有 ID 均为非空字符串且无重复。

    异常：
        TypeError: raw_seq_ids 不是序列类型或包含非字符串元素时抛出。
        ValueError: raw_seq_ids 为空列表或包含空字符串、重复 ID、
            含空字节的 ID 时抛出。
    """
    if isinstance(raw_seq_ids, (str, bytes)) or not isinstance(raw_seq_ids, Sequence):
        raise TypeError(f"{field_name} must be a sequence of non-empty strings")

    normalized_seq_ids: list[str] = []
    seen_seq_ids: set[str] = set()

    for index, seq_id in enumerate(raw_seq_ids):
        if not is_string_like(seq_id):
            raise TypeError(f"{field_name}[{index}] must be a string")

        normalized_seq_id = str(seq_id).strip()
        if not normalized_seq_id:
            raise ValueError(f"{field_name}[{index}] must not be blank")
        # 拒绝含空字节的 ID，防止注入和路径穿越风险（空字节可截断字符串）。
        if "\x00" in normalized_seq_id:
            raise ValueError(f"{field_name}[{index}] must not contain null bytes")

        if normalized_seq_id in seen_seq_ids:
            raise ValueError(f"{field_name} must not contain duplicate seq_ids: {normalized_seq_id}")

        normalized_seq_ids.append(normalized_seq_id)
        seen_seq_ids.add(normalized_seq_id)

    if not normalized_seq_ids:
        raise ValueError(f"{field_name} must not be empty")

    return normalized_seq_ids


def resolve_public_eval_seq_ids(
    dataset_name: str,
    experiment_cfg: Optional[dict[str, Any]] = None,
    mode: str = "default",
) -> list[str]:
    """按统一合同解析公开或补充公开路线的评测序列。

    解析优先级（从高到低）：
    1. experiment_cfg.mode_overrides[mode].seq_ids
    2. experiment_cfg.seq_ids
    3. public_dataset_registry.datasets[dataset_name].frozen_public_eval_seq_ids

    参数：
        dataset_name: 数据集名称。
        experiment_cfg: 实验配置字典，可为 None。
        mode: 运行模式名称，默认为 "default"。模式名称会做大小写归一化
            （小写+去空白），以与 protocol 层 normalize_run_mode 的归一化
            哲学对齐，避免 "Default" 与 "default" 被视为不同模式。

    返回：
        标准化后的评测序列 ID 列表。

    异常：
        ValueError: 无法找到 seq_ids 配置或配置无效时抛出。
        TypeError: 参数类型不正确时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "dataset_name": dataset_name,
        "mode": mode,
        "experiment_cfg_keys": list(experiment_cfg.keys()) if isinstance(experiment_cfg, dict) else None,
    }, "resolve_public_eval_seq_ids")
    normalized_dataset_name = normalize_public_dataset_name(dataset_name)

    if experiment_cfg is not None and not isinstance(experiment_cfg, dict):
        raise TypeError("experiment_cfg must be a dict or None")

    cfg = dict(experiment_cfg or {})

    if mode is None:
        raise ValueError("mode must be a non-empty string")

    if not is_string_like(mode):
        raise TypeError("mode must be a string")

    # 模式名称大小写归一化，与 protocol 层 normalize_run_mode 的归一化哲学对齐。
    normalized_mode = str(mode).strip().lower()
    if not normalized_mode:
        raise ValueError("mode must be a non-empty string")

    # 类型守卫：mode_overrides 必须是映射，避免字符串被 dict() 误解释为
    # 字符序列（dict("abc") 会抛出令人困惑的 ValueError）。
    raw_mode_overrides = cfg.get("mode_overrides")
    if raw_mode_overrides is None:
        mode_overrides: dict[str, Any] = {}
    elif not isinstance(raw_mode_overrides, dict):
        raise TypeError("experiment_cfg.mode_overrides must be a mapping or null")
    else:
        mode_overrides = raw_mode_overrides

    raw_mode_cfg = mode_overrides.get(normalized_mode)
    if raw_mode_cfg is None:
        mode_cfg: dict[str, Any] = {}
    elif not isinstance(raw_mode_cfg, dict):
        raise TypeError(
            f"experiment_cfg.mode_overrides.{normalized_mode} must be a mapping or null"
        )
    else:
        mode_cfg = raw_mode_cfg

    raw_seq_ids: Any = None
    field_name: str = ""

    raw_seq_ids = mode_cfg.get("seq_ids")
    field_name = f"mode_overrides.{normalized_mode}.seq_ids"

    if raw_seq_ids is None:
        raw_seq_ids = cfg.get("seq_ids")
        field_name = "experiment_cfg.seq_ids"

    # §9 修复: 显式空列表 (experiment_cfg.seq_ids = []) 时回退到 registry frozen,
    # 而非直接 raise; 显式空列表的语义是"暂未指定"而非"显式拒绝".
    if isinstance(raw_seq_ids, list) and len(raw_seq_ids) == 0:
        raw_seq_ids = None

    if raw_seq_ids is None:
        registry_cfg = load_public_dataset_registry()
        dataset_entry = get_dataset_entry(normalized_dataset_name, registry_cfg)
        raw_seq_ids = dataset_entry.get("frozen_public_eval_seq_ids")
        field_name = f"public_dataset_registry.datasets.{normalized_dataset_name}.frozen_public_eval_seq_ids"

    if raw_seq_ids is None:
        raise ValueError(
            "public_sequence_category requires explicit seq_ids in experiment_cfg "
            "or frozen_public_eval_seq_ids in the public dataset registry"
        )

    seq_ids = _normalize_public_seq_id_list(raw_seq_ids, field_name=field_name)

    max_sequences = mode_cfg.get("max_sequences")
    if max_sequences is None:
        return seq_ids

    if not is_integer(max_sequences):
        raise TypeError(f"mode_overrides.{normalized_mode}.max_sequences must be an integer or null")

    if max_sequences <= 0:
        raise ValueError(f"mode_overrides.{normalized_mode}.max_sequences must be positive when provided")

    return seq_ids[:max_sequences]


def list_public_dataset_names(registry_cfg: dict[str, Any]) -> list[str]:
    """返回注册表中所有公开数据集名称的排序列表。

    参数：
        registry_cfg: 已加载的注册表配置字典。

    返回：
        按字母排序的数据集名称列表；注册表无效时返回空列表。

    异常：
        TypeError: registry_cfg 不是字典时抛出。
    """
    if not isinstance(registry_cfg, dict):
        raise TypeError("registry_cfg must be a dict")

    datasets = registry_cfg.get("datasets", {})
    if not isinstance(datasets, dict):
        return []

    # 对名称做标准化（小写+去空白），与 get_dataset_entry 的查找键保持一致，
    # 避免 "MILUV" 和 "miluv" 同时存在时返回不一致的名称。
    normalized_names: list[str] = []
    seen: set[str] = set()
    for raw_name in datasets.keys():
        if not is_string_like(raw_name):
            continue
        normalized = str(raw_name).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        normalized_names.append(normalized)
    return sorted(normalized_names)


@lru_cache(maxsize=1)
def _get_official_public_benchmark_datasets() -> frozenset[str]:
    """从实验协议中加载官方公开基准允许的数据集面（带缓存）。

    缓存说明：lru_cache 不感知文件系统变更，修改 experiment_protocol.yaml
    后需重启进程才能生效（与 scene_axis_protocol 的缓存策略一致）。

    返回：
        包含所有官方允许的公开基准数据集名称的不可变集合。
    """
    # 不传参调用，走 _resolve_protocol_cfg(None) → _load_default_experiment_protocol_cached
    # 缓存路径，避免显式 load_experiment_protocol() 触发的双重校验。
    return frozenset(get_public_benchmark_allowed_datasets())


def _read_registry_bool(
    entry: dict[str, Any], key: str, default: bool, dataset_name: str
) -> bool:
    """从注册表条目读取布尔字段，严格校验类型，拒绝 truthiness 强转。

    与 protocol 层 _read_protocol_boolean_flag 哲学一致：只接受显式布尔值，
    避免 YAML 中误写 "false" 字符串被 bool() 判为 True 的语义反转风险。

    参数：
        entry: 注册表条目字典。
        key: 要读取的布尔字段键名。
        default: 键不存在或为 None 时的默认值。
        dataset_name: 数据集名称，用于构造错误信息。

    返回：
        bool: 读取并校验后的布尔值。

    异常：
        TypeError: 字段值存在但不是布尔类型时抛出。
    """
    if key not in entry:
        return default
    raw_value = entry[key]
    if raw_value is None:
        return default
    if not is_bool_like(raw_value):
        raise TypeError(
            f"registry entry '{dataset_name}'.{key} must be a boolean, "
            f"got {type(raw_value).__name__}"
        )
    return bool(raw_value)


def inspect_public_dataset_readiness(
    dataset_name: str,
    registry_cfg: dict[str, Any],
) -> dict[str, Any]:
    """汇总单个公开数据集的烟雾/论文就绪状态。

    就绪判断逻辑：
    1. 从注册表条目读取 release_tier 和 paper_full_ready 标记。
    2. 判断注册表层面的烟雾就绪（release_tier 为 smoke_only 或属于
       _FULL_READY_TIERS，或显式标记 public_benchmark_smoke_ready）。
    3. 与协议允许的公开基准数据集面取交集，只有同时满足"在协议面上"
       和"注册表标记就绪"才算 public_benchmark_smoke_ready。

    守门范围说明：本函数只做"准入级"就绪检查（tier + 协议面交集），
    不检查 frozen_public_eval_seq_ids 评测序列合同。评测序列合同的
    守门由下游 protocol 层 normalize_public_benchmark_request 负责，
    两者职责不同，不可互相替代。

    参数：
        dataset_name: 数据集名称。
        registry_cfg: 已加载的注册表配置字典。

    返回：
        就绪状态报告字典，包含：
        - dataset_name: 标准化后的数据集名称
        - release_tier: 发布层级
        - public_benchmark_smoke_ready: 是否可用于公开基准烟雾测试
        - paper_full_ready: 是否标记为论文全量就绪
        - paper_full_ready_allowed: 是否允许参与论文级全量验证
        - status: 状态描述（ready_for_public_benchmark_smoke 或 not_ready）
        - reasons: 未就绪原因列表

    异常：
        KeyError: 数据集名称不在注册表中时抛出。
        TypeError: 参数类型不正确或布尔字段非布尔类型时抛出。
        ValueError: 注册表条目中已存在 dataset_name 字段且与标准化名称冲突时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "dataset_name": dataset_name,
        "registry_cfg_keys": list(registry_cfg.keys()) if isinstance(registry_cfg, dict) else None,
    }, "inspect_public_dataset_readiness")
    if not isinstance(registry_cfg, dict):
        raise TypeError("registry_cfg must be a dict")

    entry = get_dataset_entry(dataset_name, registry_cfg)

    canonical_name = str(entry.get("dataset_name") or "")
    release_tier = str(entry.get("release_tier") or _SMOKE_READY_TIER)

    # 严格校验布尔字段类型，拒绝 truthiness 强转。
    paper_full_ready = _read_registry_bool(
        entry, "paper_full_ready", default=False, dataset_name=canonical_name
    )

    # registry_smoke_ready 默认推导：所有已知 tier（smoke_only/full_ready/paper_ready）
    # 均默认为 True，只有显式 public_benchmark_smoke_ready: False 才能阻断。
    default_smoke_ready = release_tier in _FULL_READY_TIERS or release_tier == _SMOKE_READY_TIER
    registry_smoke_ready = _read_registry_bool(
        entry, "public_benchmark_smoke_ready", default=default_smoke_ready, dataset_name=canonical_name
    )

    official_public_benchmark_datasets = _get_official_public_benchmark_datasets()

    public_smoke_ready = (
        canonical_name in official_public_benchmark_datasets and registry_smoke_ready
    )

    reasons: list[str] = []
    if not public_smoke_ready:
        # 使用独立 if 而非 elif，确保同时存在多个未就绪原因时全部报告，
        # 提升审计可追溯性。
        if canonical_name not in official_public_benchmark_datasets:
            reasons.append("dataset_outside_official_public_benchmark_surface")
        if not registry_smoke_ready:
            reasons.append("registry_not_marked_for_public_benchmark_smoke")

    return {
        "dataset_name": canonical_name,
        "release_tier": release_tier,
        "public_benchmark_smoke_ready": public_smoke_ready,
        "paper_full_ready": paper_full_ready,
        "paper_full_ready_allowed": release_tier in _FULL_READY_TIERS and paper_full_ready,
        "status": "ready_for_public_benchmark_smoke" if public_smoke_ready else "not_ready",
        "reasons": reasons,
    }
