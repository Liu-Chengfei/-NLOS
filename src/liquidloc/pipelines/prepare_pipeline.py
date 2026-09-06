"""原始数据准备流水线。

这个流水线负责把不同数据集的原始输入读出来，做字段映射、数据检查、
事件构建和清单生成，最后把每条序列的事件和整体验证清单写到输出目录。
它是很多后续 pipeline 的前置步骤。

上游依赖:
- liquidloc.dataio.readers: 各数据集的原始数据读取器
- liquidloc.dataio.adapters.field_mapper: 外部字段到内部字段的映射
- liquidloc.dataio.adapters.event_builder: 事件构建与合并
- liquidloc.dataio.manifests: 数据集清单生成与检查

下游调用者:
- liquidloc.pipelines.public_benchmark_pipeline: 公开基准路由流水线
- 任何需要从原始数据生成事件序列的脚本

核心变量:
- dataset_name: 数据集名称，决定走哪条读取路径
- seq_ids: 需要准备的序列 ID 列表
- raw_root: 原始数据根目录
- field_mapping: 外部字段到内部字段的映射表
"""

from __future__ import annotations  # 允许在标注里直接引用当前模块类型。

import json  # 用于写事件和清单文件。
import gzip  # 用于压缩事件产物（G7.1：替代 JSON，空间降至 ~10-20%）。
import pickle  # 用于事件高效序列化。
from collections.abc import Mapping  # 用于校验字段映射合同，避免错误结构拖到更深链路才报错。
from copy import deepcopy  # 用于深拷贝 scene_parameters，避免所有事件共享同一可变字典引用。
from pathlib import Path  # 用于处理输出目录路径。
from typing import Any  # 用于给字典做宽松标注。

from liquidloc.common.constants import DATASET_NAME_MILUV, DATASET_NAME_NTU_VIRAL, DATASET_NAME_SIM, DATASET_NAME_UTIL  # 数据集名字常量（单源真相，D9 漂移根因修复）。
from liquidloc.common.io_utils import dumps_json_text  # 严格 JSON 序列化工具（清单文件仍用 JSON 写）。
from liquidloc.common.paths import get_standard_dirs, resolve_output_root  # 解析标准目录和统一输出根目录。
from liquidloc.common.types import StageResult  # 流水线统一返回结构。
from liquidloc.common.validation import is_string_like  # 统一判断字符串类型。
from liquidloc.common.validation import validate_path_component  # 校验路径组件不含穿越字符，防止 seq_id 拼接到文件路径时路径穿越。
from liquidloc.dataio.adapters.event_builder import build_imu_events  # 把 IMU 原始行转成事件。
from liquidloc.dataio.adapters.event_builder import build_uwb_events  # 把 UWB 原始行转成事件。
from liquidloc.dataio.adapters.event_builder import build_vio_events  # 把 VIO 原始行转成事件。
from liquidloc.dataio.adapters.event_builder import merge_and_finalize_events  # 合并并整理事件序列。
from liquidloc.dataio.adapters.field_mapper import map_external_fields  # 做外部字段到内部字段的映射。
from liquidloc.dataio.manifests.build_manifests import REQUIRED_STREAMS  # 标准数据集需要的流。
from liquidloc.dataio.manifests.build_manifests import UTIL_REQUIRED_STREAMS  # util 数据集需要的流。
from liquidloc.dataio.manifests.build_manifests import build_manifests  # 生成数据清单。
from liquidloc.dataio.manifests.dataset_checks import REQUIRED_RAW_KEYS  # 标准数据集必须有的原始键。
from liquidloc.dataio.manifests.dataset_checks import UTIL_REQUIRED_RAW_KEYS  # util 数据集必须有的原始键。
from liquidloc.dataio.manifests.dataset_checks import (
    inspect_sim_materialized_contract,  # SIM 物化合同校验（主表 K1/K3 几何档位，五轴档位协议）。
    validate_sim_generator_version,  # §28.6 仿真生成器版本自证契约校验。
)
from liquidloc.dataio.manifests.dataset_checks import run_dataset_checks  # 运行数据集合法性检查。
from liquidloc.dataio.readers.miluv_reader import read_miluv_sequence  # 读 MILUV 原始数据。
from liquidloc.dataio.readers.ntu_viral_reader import read_ntu_viral_sequence  # 读 NTU VIRAL 原始数据。
from liquidloc.dataio.readers.sim_reader import read_sim_sequence  # 读仿真数据。
from liquidloc.dataio.readers.util_reader import read_util_sequence  # 读 util 数据。
from liquidloc.interfaces.pipeline_api import PipelineAPI, normalize_pipeline_cfg  # 流水线接口基类与统一配置规整 helper。
from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS  # 桥接层业务阈值（flow_missing_dyaw_quality_penalty）。
from liquidloc.protocol.scene_schema import decode_scene  # 解析 scene_id，恢复 A/N/V/G/K 轴级上下文。
from liquidloc.protocol.scene_axis_protocol import attach_scene_parameters  # 展开简写轴值为标准场景参数对象。


def _resolve_scene_id(cfg: dict[str, Any], dataset_name: str, seq_id: str) -> str:
    """为某条序列解析场景 ID。

    Args:
        cfg: 当前流水线配置，可能包含 scene_id_by_seq 或 scene_id。
        dataset_name: 数据集名称，用于构造默认场景 ID。
        seq_id: 当前序列 ID。

    Returns:
        解析后的场景 ID 字符串。

    Raises:
        ValueError: sim 数据集没有显式场景 ID 时报错。
    """
    by_seq_id = cfg.get('scene_id_by_seq') or {}  # 先看是否有按序列定制的映射。
    if seq_id in by_seq_id:  # 显式映射优先。
        scene_id = by_seq_id[seq_id]  # 取出这一条序列对应的场景 ID。
        if not is_string_like(scene_id) or not str(scene_id).strip():  # 映射值必须是非空字符串。
            raise ValueError(f'scene_id_by_seq[{seq_id!r}] must be a non-empty string')
        return str(scene_id).strip()  # 返回规范化后的场景 ID，避免空白污染下游场景协议解析。

    scene_id = cfg.get('scene_id')  # 再看是否有统一的场景 ID。
    if is_string_like(scene_id) and str(scene_id).strip():  # 非空字符串才算合法。
        return str(scene_id).strip()  # 返回规范化后的统一场景 ID。

    if dataset_name == DATASET_NAME_SIM:  # sim 数据集要求必须显式指定场景 ID。
        raise ValueError('sim prepare_pipeline requires explicit scene_id or scene_id_by_seq')
    return f'{dataset_name}:{seq_id}'  # 其他数据集默认用 dataset:seq_id。


def _resolve_output_root(cfg: dict[str, Any], default_name: str) -> Path:
    """解析输出根目录，委托给 common.paths.resolve_output_root 统一处理。"""
    return resolve_output_root(cfg, default_name)


def _flow_rows_as_vio_rows(flow_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 flow 原始行改写成 VIO 行结构。

    某些数据集（如 UTIL）提供的是光流增量而非 VIO 增量，
    这个函数把 flow 行的字段映射到 VIO 行的同名字段，
    使下游 build_vio_events 可以统一处理。

    注意：flow 模态通常缺少 dyaw、tracked_features、reproj_err 字段。
    缺失字段按 feature_missing_policy.numeric_fill_value (0.0) 填充，
    同时在 missing_mask 中标记为 1，使下游能区分"真实值"和"补零"。
    这遵守 sensors.yaml 中 missing_semantics_carrier: missing_mask 的约定。
    missing_mask 采用与 sensors.yaml vio_fields [dx, dy, dyaw, quality,
    tracked_features, reproj_err] 一一对齐的列表形式，使用 1/0 整数
    （1=缺失补零，0=真实值），与 build_imu_events 的 missing_mask
    格式保持一致，便于下游统一消费。

    field_mapper 对缺失字段统一填充 None 占位，因此不能用
    row.get(key, default) 取值——当键存在但值为 None 时，dict.get
    返回 None 而非 default，会导致 float(None)/int(None) 抛 TypeError。
    这里通过 has_* 标志显式区分"键不存在/值为 None"与"真实值"。

    Args:
        flow_rows: 原始 flow 行列表，每行包含 timestamp/dx/dy/quality 等字段。

    Returns:
        转换后的 VIO 行列表，字段名与 VIO 行一致。
    """
    _FLOW_MISSING_DYAW_QUALITY_PENALTY = BRIDGE_THRESHOLDS["flow_missing_dyaw_quality_penalty"]  # flow 缺少 dyaw 时的质量惩罚值，从桥接层阈值表读取。
    vio_rows: list[dict[str, Any]] = []  # 收集转换后的 VIO 行。
    for row in flow_rows or []:  # 没有 flow 行时就返回空列表。
        timestamp = row['timestamp']  # 时间戳直接沿用。
        dx = row['dx']  # 平移增量沿用。
        dy = row['dy']  # 平移增量沿用。
        has_dyaw = 'dyaw' in row and row['dyaw'] is not None
        dyaw = float(row['dyaw']) if has_dyaw else 0.0  # 航向增量没有就给 0；has_dyaw 已排除 None，避免 float(None) TypeError。
        _quality_raw = row.get('quality')  # 先取原始值，None 时不能用 float(None)。
        quality = float(_quality_raw) if _quality_raw is not None else 1.0  # 质量分数沿用，缺失时默认 1.0。
        # flow 缺少 dyaw 时下调 quality，表示旋转信息不可靠。
        if not has_dyaw and quality > _FLOW_MISSING_DYAW_QUALITY_PENALTY:
            quality = _FLOW_MISSING_DYAW_QUALITY_PENALTY
        has_tracked_features = 'tracked_features' in row and row['tracked_features'] is not None
        # 缺失时填 0（整数占位），存在时保留原值交由 build_vio_events 的 _require_integer_field 校验整数语义，
        # 不在此处用 int() 强转，避免 int(3.7)=3 的静默截断掩盖数据质量问题。
        tracked_features = row['tracked_features'] if has_tracked_features else 0
        has_reproj_err = 'reproj_err' in row and row['reproj_err'] is not None
        reproj_err = float(row['reproj_err']) if has_reproj_err else 0.0  # 重投影误差统一转浮点数；has_reproj_err 已排除 None。
        vio_row = {
            'timestamp': timestamp,  # 时间戳。
            'dx': dx,  # x 方向位移。
            'dy': dy,  # y 方向位移。
            'dyaw': dyaw,  # 航向变化。
            'quality': quality,  # 质量评分。
            'tracked_features': tracked_features,  # 跟踪特征数。
            'reproj_err': reproj_err,  # 重投影误差。
            # missing_mask：与 sensors.yaml vio_fields [dx, dy, dyaw, quality,
            # tracked_features, reproj_err] 一一对应的列表，
            # 1 表示该位置为缺失补零，0 表示真实值。
            # 遵守 sensors.yaml 中 missing_semantics_carrier: missing_mask 的约定。
            # 列表格式与 build_imu_events 的 missing_mask 一致，便于下游统一消费。
            'missing_mask': [
                0,  # dx — flow_fields 始终提供 dx。
                0,  # dy — flow_fields 始终提供 dy。
                1 if not has_dyaw else 0,  # flow 缺少 dyaw 时标记缺失。
                0,  # quality — flow_fields 始终提供 quality。
                1 if not has_tracked_features else 0,  # flow 缺少 tracked_features 时标记缺失。
                1 if not has_reproj_err else 0,  # flow 缺少 reproj_err 时标记缺失。
            ],
        }
        if 'source_t' in row:  # flow 行显式带了原始来源时间时，必须继续传下去供下游对齐真值。
            vio_row['source_t'] = row['source_t']
        vio_rows.append(vio_row)  # 把转换结果加入输出。
    return vio_rows  # 返回转换后的列表。


def _reject_unsupported_util_tof_bridge(internal_bundle: dict[str, Any], seq_id: str) -> None:
    """拒绝静默丢弃 UTIL ToF 原始观测。"""
    tof_rows = internal_bundle.get('tof_raw', [])
    if tof_rows:
        raise ValueError(
            f'UTIL sequence {seq_id!r} contains tof_raw, but prepare_pipeline only supports '
            'imu/uwb/vio events. UTIL ToF cannot be bridged to uwb without anchor_id and has '
            'no dedicated event modality, so continuing would silently drop measurement data.'
        )


def _raise_if_dataset_check_failed(check_report: dict[str, Any], *, seq_id: str) -> None:
    """把 dataset_checks 的失败结果提升成准备阶段硬闸门。"""
    if check_report.get('is_valid'):
        return
    missing_streams = list(check_report.get('missing_streams') or [])
    empty_streams = list(check_report.get('empty_streams') or [])
    bad_streams = list(check_report.get('bad_streams') or [])
    detail_parts: list[str] = []
    if missing_streams:
        detail_parts.append(f'missing_streams={missing_streams}')
    if empty_streams:
        detail_parts.append(f'empty_streams={empty_streams}')
    if bad_streams:
        detail_parts.append(f'bad_streams={bad_streams}')
    detail = ', '.join(detail_parts) or repr(check_report)
    raise ValueError(f'sequence {seq_id!r} failed dataset checks: {detail}')


def _enforce_sim_materialized_contract(raw_root: Any) -> None:
    """对 SIM 数据集强校验主表几何合同（K∈{K1,K3}，五轴档位协议 G 已并入 K）。

    用户审查标准 #3 要求 inspect_sim_materialized_contract 在 prepare 阶段
    强校验。五轴档位协议下合同从单一 K0 改为欠定主表允许集（K1/K3），阻断旧 2 锚
    或优几何高冗余 raw。

    §8 / fail-loud 守卫：inspect_sim_materialized_contract 契约声明返回 dict[str, Any]，
    但若上游实现 bug 致返回 None / 非 Mapping（被 silent-skip swallow），原实现
    `sim_contract_report.get('is_valid')` 会抛 `AttributeError("'NoneType' object has
    no attribute 'get'")` ——虽意外 raise 但错误信号模糊，用户难以诊断"inspect 上游
    bug"vs"is_valid=False"。修复：显式 None/非 Mapping 守卫 + 上下文 RuntimeError，
    让数据完整性破坏显形而非借 AttributeError 透传。
    """
    sim_contract_report = inspect_sim_materialized_contract(raw_root)  # 阻断 stale/非法几何 raw。
    # BUG-004 修复 (2026-09-06 §10 审计): sim_e9_main 是 seed0..seed9 嵌套结构,
    # raw_root 顶层为多 seed 容器 (每个 seedN/ 内含 sim_curve_01/ 等 60 序列)。
    # 顶层 raw_root 本身无 anchor_layout.json, 直接 contract 检查把 10 个 seed
    # 当成 10 序列, 100% 报 missing_anchor_layout。当 raw_root 下含 seed*/ 子目录时,
    # 应遍历每个 seed 目录分别检查 (任一不过则 raise)。
    if isinstance(raw_root, (str, Path)):
        _raw_root_path = Path(raw_root)
        _seed_subdirs = sorted(
            d for d in _raw_root_path.iterdir()
            if d.is_dir() and d.name.startswith('seed')
        ) if _raw_root_path.is_dir() else []
        if _seed_subdirs and all(
            (sd / 'anchor_layout.json').is_file() == False
            for sd in _seed_subdirs
        ):
            # raw_root 是 seed 容器, 对每个 seed 分别检查
            sim_contract_report = {
                'is_valid': True,
                'sequence_count': 0,
                'seed_results': {},
            }
            all_valid = True
            for sd in _seed_subdirs:
                _seed_report = inspect_sim_materialized_contract(sd)
                sim_contract_report['seed_results'][sd.name] = _seed_report
                sim_contract_report['sequence_count'] += _seed_report.get('sequence_count', 0)
                if not _seed_report.get('is_valid', False):
                    all_valid = False
                    sim_contract_report['is_valid'] = False
                    break
            if not all_valid:
                sim_contract_report['is_valid'] = False
    if not isinstance(sim_contract_report, Mapping):
        # §8 fail-loud：inspect 非 Mapping 返回值不应被 .get() AttributeError 偶然 raise
        # ——显式 RuntimeError 含原 report 上下文，让上游 inspect 实现 bug 显形而非
        # 误读为下游 contract violation。
        raise RuntimeError(
            'sim_materialized_contract inspection integrity breach: '
            'inspect_sim_materialized_contract 返回非 Mapping（应为 dict[str, Any]）；'
            f'got type={type(sim_contract_report).__name__}, value={sim_contract_report!r}; '
            '上游 inspect 实现 bug，未遵循其声明返回类型——需修复 inspect 实现而非此守卫。'
        )
    if not sim_contract_report.get('is_valid'):
        raise RuntimeError(
            'sim raw_root does not satisfy the paper-grade materialized SIM contract '
            '(allowed K∈{K1,K3}); '
            f"report={sim_contract_report}. Re-run scripts/02_generate_sim_raw.py or point "
            '--raw-root to a protocol-trajectory undetermined-geometry sim raw directory.'
        )

    # §28.6 自证契约：仿真生成器版本必须与当前代码库严格一致。
    gen_report = validate_sim_generator_version(raw_root)
    if not isinstance(gen_report, Mapping):
        raise RuntimeError(
            'sim_generator_version inspection integrity breach: '
            'validate_sim_generator_version 返回非 Mapping（应为 dict[str, Any]）；'
            f'got type={type(gen_report).__name__}, value={gen_report!r}; '
            '上游 validate 实现 bug，未遵循其声明返回类型——需修复 validate 实现而非此守卫。'
        )
    if not gen_report.get('is_valid'):
        raise RuntimeError(
            'sim raw_root does not satisfy the §28.6 self-certifying generator contract: '
            f"expected version={gen_report.get('expected_version')!r}, "
            f"bad_version_seq_ids={gen_report.get('bad_version_seq_ids')}, "
            f"missing_version_seq_ids={gen_report.get('missing_version_seq_ids')}. "
            'Re-run scripts/02_generate_sim_raw.py or point --raw-root to a '
            'sim raw directory generated by the current sim_materializer version.'
        )


class PreparePipeline(PipelineAPI):
    """原始数据准备流水线。

    负责把不同数据集（sim/miluv/ntu_viral/util）的原始输入读出来，
    做字段映射、数据检查、事件构建和清单生成。
    """

    def run(self, pipeline_cfg: dict | None = None, runtime_context: dict | None = None) -> StageResult:
        """执行数据准备闭环。

        Args:
            pipeline_cfg: 运行配置，必须包含 dataset_name、seq_ids、raw_root。
            runtime_context: 运行时上下文，保留接口但不使用。

        Returns:
            StageResult，包含事件文件路径和清单元数据。

        Raises:
            ValueError: 数据集名称不合法、序列为空或重复、数据校验失败时。
            TypeError: 字段映射或事件载荷类型不匹配时。
            FileNotFoundError: 原始数据文件或目录不存在时。
            KeyError: 事件字段缺失时。
        """
        cfg = normalize_pipeline_cfg(pipeline_cfg)  # 统一成字典，并拒绝非映射配置。
        from liquidloc.common.tee_logger import print_dict
        print_dict({
            "dataset_name": cfg.get("dataset_name"),
            "seq_ids": cfg.get("seq_ids"),
            "raw_root": str(cfg.get("raw_root")) if cfg.get("raw_root") else None,
            "has_field_mapping": "field_mapping" in cfg,
            "output_root": str(cfg.get("output_root")) if cfg.get("output_root") else None,
            "scene_id": cfg.get("scene_id"),
        }, "PreparePipeline.run 入口参数")
        dataset_name = cfg.get('dataset_name')  # 读取数据集名称。
        if dataset_name not in {DATASET_NAME_SIM, DATASET_NAME_MILUV, DATASET_NAME_NTU_VIRAL, DATASET_NAME_UTIL}:  # 只支持这四种数据集。
            raise ValueError('dataset_name must be one of: sim, miluv, ntu_viral, util')

        seq_ids = list(cfg.get('seq_ids') or [])  # 需要准备的序列列表。
        if not seq_ids:  # 序列不能为空。
            raise ValueError('seq_ids must be a non-empty list')

        output_root = _resolve_output_root(cfg, 'prepare_smoke')  # 解析输出目录。
        output_root.mkdir(parents=True, exist_ok=True)  # 确保输出根目录存在。

        raw_root = cfg.get('raw_root')  # 原始数据根目录必须由上层传入。
        if raw_root is None:  # raw_root 是 value contract 必填项，缺失时应抛 ValueError 而非 KeyError。
            raise ValueError('raw_root is required')
        if dataset_name == DATASET_NAME_SIM:  # SIM 数据集在 prepare 阶段强校验主表欠定几何合同，字段漂移立即 abort（用户审查标准 #3）。
            _enforce_sim_materialized_contract(raw_root)
        field_mapping = cfg.get('field_mapping')  # 外部字段映射。
        if dataset_name == DATASET_NAME_UTIL:  # util 数据集有单独的必需流定义。
            required_streams = UTIL_REQUIRED_STREAMS  # util 的必需流。
            required_raw_keys = UTIL_REQUIRED_RAW_KEYS  # util 的必需原始键。
        else:
            required_streams = REQUIRED_STREAMS  # 标准数据集的必需流。
            required_raw_keys = REQUIRED_RAW_KEYS  # 标准数据集的必需原始键。
        if len(set(seq_ids)) != len(seq_ids):  # 序列 ID 不允许重复。
            raise ValueError('seq_ids must not contain duplicates')

        # B04: 测试轨迹数 ≥ 20 条有效计分轨（手册 §0.4 / §9 / §14）。
        # 豁免通道（2026-09-06 AUDIT_REPORT §4 修门位置错误）: 真实实验路径仍 ≥20 硬门；
        # smoke_mode / quick_full_rule='quick_smoke_scale__full_real_execution_required' 视为
        # 冒烟/单轨调通场景, 跳过 B04 计数检查, 与 eval_pipeline.py:615 口径对齐.
        cfg_quick_full_rule = cfg.get('quick_full_rule') if isinstance(cfg, Mapping) else None
        b04_exempt = bool(cfg.get('smoke_mode')) or cfg_quick_full_rule == 'quick_smoke_scale__full_real_execution_required'
        if not b04_exempt and len(seq_ids) < 20:
            raise ValueError(
                f'B04 violation: {len(seq_ids)} seq_ids provided, minimum 20 required per B04 '
                f'(test trajectory count ≥ 20 valid scorable trajectories per §9 / §14)'
            )

        artifacts: list[str] = []  # 收集输出产物路径。
        per_sequence: dict[str, Any] = {}  # 收集每条序列的处理结果。
        for seq_id in seq_ids:  # 逐条序列处理。
            # BUG-008 修复 (2026-09-06 §10 阶段 12 审计): sim_e9_main 是 seed0/sim_curve_01 嵌套结构,
            # seq_id 含路径分隔符. sim 路径下 read_sim_sequence 自己接受嵌套路径; 通用
            # validate_path_component 仅对 miluv/ntu_viral/util 等单层 seq_id 生效.
            if dataset_name != DATASET_NAME_SIM:  # sim 走 sim_reader 自己路径校验, 跳过通用 path traversal 检查.
                validate_path_component(seq_id, name='seq_id')  # 校验 seq_id 不含路径穿越字符，防止下游拼接 {seq_id}_events.json 时路径穿越，与 core_pipeline.py / eval_pipeline.py 模式一致。
            # BUG-009 修复 (2026-09-06 §10 阶段 12 审计): sim 嵌套 seq_id 'seed0/sim_curve_01'
            # 直接作 pickle 文件名不合法 (gzip FileNotFoundError). 这里把 / 替换为 __
            # 得到文件系统安全名, 但下游路径拼接 (read_sim_sequence, sim_meta 路径) 用原始 seq_id.
            safe_seq_id = seq_id.replace('/', '__') if '/' in seq_id else seq_id
            scene_id = _resolve_scene_id(cfg, dataset_name, seq_id)  # 解析当前序列对应的场景 ID。
            if dataset_name == DATASET_NAME_SIM:  # sim 走专门的读取器。
                # BUG-005 修复 (2026-09-06 §10 审计): sim_e9_main 是 seed0..seed9 嵌套结构,
                # seq_id='seed0__sim_curve_01' (09 脚本自动扫时用 __ 分隔避免 / 被 validate_path_component 拒).
                # read_sim_sequence 需还原为 'seed0/sim_curve_01' 路径拼接.
                _resolved_seq_id = seq_id.replace('__', '/') if '__' in seq_id else seq_id
                raw_bundle, read_report = read_sim_sequence(_resolved_seq_id, raw_root)  # 读取 sim 原始 bundle 和读取报告。
                internal_bundle = dict(raw_bundle)  # sim 不需要字段映射；浅拷贝避免与 raw_bundle 共享同一可变 dict。
            elif dataset_name == DATASET_NAME_MILUV:  # MILUV 需要字段映射。
                if not isinstance(field_mapping, Mapping) or not field_mapping:  # MILUV 必须提供非空字段映射。
                    raise ValueError("field_mapping must be a non-empty mapping for MILUV")
                normalized_field_mapping = dict(field_mapping)  # 复制一份映射表，避免后续链路改写调用方对象。
                raw_bundle, read_report = read_miluv_sequence(seq_id, raw_root)  # 读取 MILUV 原始 bundle。
                internal_bundle, mapping_report = map_external_fields(raw_bundle, normalized_field_mapping)  # 字段映射。
                read_report['mapping_report'] = mapping_report  # 把映射报告挂回读取报告。
            elif dataset_name == DATASET_NAME_NTU_VIRAL:  # NTU VIRAL 也需要字段映射。
                if not isinstance(field_mapping, Mapping) or not field_mapping:  # NTU VIRAL 必须提供非空字段映射。
                    raise ValueError("field_mapping must be a non-empty mapping for NTU VIRAL")
                normalized_field_mapping = dict(field_mapping)  # 复制一份映射表，避免后续链路改写调用方对象。
                raw_bundle, read_report = read_ntu_viral_sequence(seq_id, raw_root)  # 读取 NTU VIRAL 原始 bundle。
                internal_bundle, mapping_report = map_external_fields(raw_bundle, normalized_field_mapping)  # 字段映射。
                read_report['mapping_report'] = mapping_report  # 把映射报告挂回读取报告。
            else:  # util 也走映射路径。
                if not isinstance(field_mapping, Mapping) or not field_mapping:  # UTIL 也必须提供非空字段映射。
                    raise ValueError("field_mapping must be a non-empty mapping for UTIL")
                normalized_field_mapping = dict(field_mapping)  # 复制一份映射表，避免后续链路改写调用方对象。
                raw_bundle, read_report = read_util_sequence(seq_id, raw_root)  # 读取 util 原始 bundle。
                internal_bundle, mapping_report = map_external_fields(raw_bundle, normalized_field_mapping)  # 字段映射。
                read_report['mapping_report'] = mapping_report  # 把映射报告挂回读取报告。

            check_report = run_dataset_checks(internal_bundle, required_raw_keys=required_raw_keys)  # 先做数据合法性检查。
            _raise_if_dataset_check_failed(check_report, seq_id=seq_id)  # 检查失败时立刻阻断，不能继续产出失真事件。
            if dataset_name == DATASET_NAME_UTIL:
                _reject_unsupported_util_tof_bridge(internal_bundle, seq_id)
            imu_events = build_imu_events(internal_bundle.get('imu_raw', []), scene_id, seq_id)  # 构建 IMU 事件。
            uwb_events = build_uwb_events(internal_bundle.get('uwb_raw', []), scene_id, seq_id)  # 构建 UWB 事件。
            vio_source_rows = internal_bundle.get('vio_raw', []) or _flow_rows_as_vio_rows(internal_bundle.get('flow_raw', []))  # 没有 vio 就用 flow 转。
            vio_events = build_vio_events(vio_source_rows, scene_id, seq_id)  # 构建 VIO 事件。
            events = merge_and_finalize_events([imu_events, uwb_events, vio_events])  # 合并并整理最终事件序列。

            # 展开 scene_axis_protocol，将 scene_parameters 写入事件 meta 和序列产物。
            scene_parameters = None  # 默认无场景参数。
            try:
                scene_spec = decode_scene(scene_id)  # 尝试解码 scene_id（5 轴 S(A,N,V,G,K)）。
                # decode_scene 只解析 A/N/V/G/K 5 轴；M 轴不在 scene_code 内，但
                # attach_scene_parameters 要求 6 轴齐全。这里用 nominal level M0
                # 补全，与默认协议的"无缺失"baseline 一致（modality_drop_prob=0.0）。
                from liquidloc.protocol.scene_axis_protocol import get_nominal_levels  # 局部导入避免顶层循环依赖。
                _nominal_levels = get_nominal_levels()  # 读取每个轴的 nominal level 名。
                scene_parameters = attach_scene_parameters({  # 展开协议为标准场景参数对象。
                    'A': scene_spec.A_level,
                    'N': scene_spec.N_level,
                    'V': scene_spec.V_level,
                    'K': scene_spec.K_value,  # §2.1 G→K 合并：G 轴并入 K 轴。
                    'M': _nominal_levels.get('M', 'M0'),  # M 轴不在 scene_code 中，用 nominal 兜底。
                }).to_dict()  # SceneParameters → 纯 dict，确保 JSON 可序列化。
            except (ValueError, TypeError, KeyError):  # scene_id 不可解码时静默跳过。
                pass
            if scene_parameters is not None:  # 成功展开时，把 scene_parameters 写入每个事件的 meta。
                for event in events:  # 逐个事件写入场景参数。
                    meta = event.get('meta')  # 取出 meta。
                    if isinstance(meta, dict):  # 只有字典型 meta 才能写入。
                        meta['scene_parameters'] = deepcopy(scene_parameters)  # 深拷贝，避免所有事件共享同一可变字典引用。

            seq_payload = {  # 这一条序列的输出摘要。
                'seq_id': seq_id,  # 序列 ID。
                'scene_id': scene_id,  # 场景 ID。
                'read_report': read_report,  # 读取报告。
                'check_report': check_report,  # 检查报告。
                'event_count': len(events),  # 最终事件数量。
            }
            if scene_parameters is not None:  # 把展开后的场景参数写入序列产物。
                seq_payload['scene_parameters'] = deepcopy(scene_parameters)  # 深拷贝，避免与事件 meta 共享可变引用。
            if isinstance(raw_bundle.get('anchor_layout_raw'), dict):  # 任何数据集只要提供了锚点布局就保留。
                seq_payload['anchor_layout'] = dict(raw_bundle['anchor_layout_raw'])  # 浅拷贝避免与 raw_bundle 共享嵌套可变 dict。
            per_sequence[seq_id] = seq_payload  # 挂到总表里。

            seq_path = output_root / f'{safe_seq_id}_events.pkl.gz'  # BUG-009: 嵌套 sim seq_id 含 /, 文件名替换为 __. 每条序列单独写一份压缩 Pickle 文件（G7.1：替代 JSON，空间降至 ~10-20%）。
            with gzip.open(seq_path, 'wb', compresslevel=3) as f:
                pickle.dump(events, f, protocol=pickle.HIGHEST_PROTOCOL)  # 高效序列化后写入压缩文件。
            artifacts.append(str(seq_path))  # 记录产物路径。

        dataset_manifest, scene_manifest = build_manifests(raw_root, required_streams=required_streams)  # 生成清单。
        manifest_records = dataset_manifest.get('sequences', [])  # 取出清单里的序列记录。
        manifest_records_by_seq = {record['seq_id']: record for record in manifest_records}  # 按 seq_id 建索引。
        dataset_sequences: list[dict[str, Any]] = []  # 收集本次需要的序列记录。
        for seq_id in seq_ids:  # 只保留本次请求的序列。
            record = manifest_records_by_seq.get(seq_id)  # 从索引里找记录。
            if record is None:  # 如果清单里找不到就报错。
                raise ValueError(f'sequence {seq_id!r} missing from dataset manifest under raw_root')
            scene_id = per_sequence[seq_id]['scene_id']  # 取本次处理得到的场景 ID。
            dataset_sequence = {**record, 'scene_id': scene_id}  # 把 scene_id 补进去。
            seq_scene_parameters = per_sequence[seq_id].get('scene_parameters')  # 取协议展开后的场景参数。
            if seq_scene_parameters is not None:  # 如果存在，也补进 dataset_manifest 的序列记录。
                dataset_sequence['scene_parameters'] = seq_scene_parameters  # 协议展开结果持久化到清单。
            dataset_sequences.append(dataset_sequence)  # 放入本次数据集序列列表。
        dataset_manifest = {  # 重新包装数据集清单。
            **dataset_manifest,  # 保留原字段。
            'sequence_count': len(dataset_sequences),  # 序列数量改成本次选中的数量。
            'sequences': dataset_sequences,  # 只保留本次相关序列。
        }
        scene_to_seq_ids: dict[str, list[str]] = {}  # 收集每个 scene 对应哪些 seq_id。
        for seq_id, seq_payload in per_sequence.items():  # 逐条序列回填场景索引。
            scene_id = seq_payload['scene_id']  # 取当前序列的场景 ID。
            scene_to_seq_ids.setdefault(scene_id, []).append(seq_id)  # 把序列归到对应场景下。
        scene_items = sorted(scene_to_seq_ids.items())  # 按场景 ID 排序，保证输出稳定。
        scenes = [  # 重新整理场景清单。
            {
                'scene_id': scene_id,  # 场景 ID。
                'seq_ids': scene_seq_ids,  # 场景下的序列列表。
            }
            for scene_id, scene_seq_ids in scene_items
        ]
        scene_manifest = {  # 组装场景清单。
            'scene_count': len(scene_to_seq_ids),  # 场景数量。
            'scenes': scenes,  # 场景列表。
        }
        manifest_payload = {  # 汇总最终清单。
            'dataset_manifest': dataset_manifest,  # 数据集清单。
            'scene_manifest': scene_manifest,  # 场景清单。
            'sequences': per_sequence,  # 每条序列的处理结果。
        }
        manifest_path = output_root / 'prepare_manifest.json'  # 清单文件输出位置。
        manifest_json = dumps_json_text(manifest_payload)  # 先序列化成 JSON。
        manifest_path.write_text(manifest_json, encoding='utf-8')  # 写出清单。
        artifacts.append(str(manifest_path))  # 把清单也记为产物。

        return StageResult(  # 返回阶段结果。
            stage_name='prepare_pipeline',  # 阶段名。
            artifacts=artifacts,  # 产物路径。
            metadata=manifest_payload,  # 元数据。
        )


def run(pipeline_cfg):
    """兼容旧入口的直接运行函数。

    Args:
        pipeline_cfg: 传给 PreparePipeline.run 的配置字典。

    Returns:
        PreparePipeline().run(...) 的原始返回值。
    """
    return PreparePipeline().run(pipeline_cfg)  # 实例化后执行。
