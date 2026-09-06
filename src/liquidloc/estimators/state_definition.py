"""
文件：src/liquidloc/estimators/state_definition.py

【文件职责】
这个文件专门定义整条估计链共享的状态顺序、维度和索引映射。
它是预测、UWB 更新、VIO 更新、EKF 核心和 FGO 核心共同依赖的状态契约源头。

【本文件绝对不负责】
不做预测，不做更新，不做优化，只提供状态定义和契约校验。

【上游依赖】
configs/base/task.yaml、liquidloc.common.constants、liquidloc.protocol.task_contract。

【下游调用者】
predict_step.py、uwb_update_step.py、vision_update_step.py、ekf_core.py、fgo_core.py。

【输入对象定义】
- 无直接业务输入；它读取共享 task contract 并校验公共常量注册表。

【输出对象定义】
- state_items、state_index_map、state_dim、cov_dim、noise_dim

【核心变量定义】
- state_items
- state_index_map
- state_dim
- cov_dim
- noise_dim

【实现要求】
- state_items 的顺序必须和 configs/base/task.yaml 完全一致。
- 一旦发现不一致就直接报错，不能悄悄改口径。
"""

from __future__ import annotations  # 允许类型标注使用前向引用。

from types import MappingProxyType  # 用来创建只读映射，防止运行时被改写。

from liquidloc.common.config_utils import find_project_root  # 基于 marker 文件查找项目根目录。
from liquidloc.common.constants import validate_constant_registry  # 校验常量注册表是否完整。
from liquidloc.protocol.task_contract import get_state_items, load_task_contract  # 从共享任务合同读取状态定义。


_TASK_CONFIG_PATH = (find_project_root() / "configs" / "base" / "task.yaml").resolve()  # task.yaml 的绝对路径，resolve() 在模块加载时完成，与 protocol 层风格一致。
# 前提指导 §1.1+§2.3 全体同增 10 维硬合同：必需状态键不再只校验 5 维核心键，
# 必须把 §2.3 紧耦合扩维项 uwb_clock_bias / vio_scale 一并校验。
# 旧版 (_REQUIRED_STATE_KEYS = ("px", "py", "vx", "vy", "yaw")) 只校验 5 维核心
# 是 §1.1 防御性硬合同缺口——若 task.yaml 漏掉 uwb_clock_bias/vio_scale 但保留 5 维核心，
# 旧校验不会报错，而下游 predict_step/uwb_update_step/vision_update_step/estimator_factory
# 都依赖 10 维状态，会静默漂移成"5 维状态 + 8 维回退兜底"的破合同模式。
# 现扩为 10 维全体同增必需键，与 task.yaml state_items 当前 10 维硬合同对齐。
_REQUIRED_STATE_KEYS = (
    "px", "py", "vx", "vy", "yaw",  # §1.1 主表 5 维核心键
    "bax", "bay", "bg",  # §1.1 主表加计/陀螺偏置 3 项
    "uwb_clock_bias", "vio_scale",  # §2.3 紧耦合扩维 2 项
)  # 全体同增 10 维必需键，缺少任何一个会立即报错。

state_items = get_state_items()  # 从共享任务合同获取冻结的状态顺序元组，这是整条估计链的状态契约源头。


def _load_config_state_items() -> tuple[str, ...]:
    """从共享 task contract 读取 state_definition.state_items。

    返回
    -------
    tuple[str, ...]
        配置文件中定义的状态项顺序元组。
    """
    contract = load_task_contract(_TASK_CONFIG_PATH)  # 加载 task.yaml 合同。
    state_definition = contract["state_definition"]  # 取出状态定义子节。
    return tuple(state_definition["state_items"])  # 转成不可变元组，保证顺序不被改写。


def _validate_upstream_contracts() -> None:
    """校验常量注册表和共享任务合同是否与本地冻结定义一致。

    这个函数在构建 state_index_map 时被调用，确保以下一致性：
    1. common.constants 注册表完整（没有遗漏的常量名）
    2. task.yaml 的 state_items 与本地冻结定义完全一致
    3. 两边都没有重复项
    4. 必需的状态键（px, py, vx, vy, yaw）都存在

    注意：predict_step.py 等下游消费方通过 ``_IDX = {name: i for i, name in enumerate(state_items)}``
    动态派生索引，不存在硬编码数字索引，因此本函数不做硬编码索引校验。

    异常
    ------
    RuntimeError
        当常量注册表不完整时抛出。
    ValueError
        当状态项重复、冻结定义与配置不一致或缺少必需状态键时抛出。
    """
    registry_report = validate_constant_registry()  # 检查常量注册表是否完整。
    if not registry_report["is_complete"]:  # 注册表不完整说明有常量名遗漏或契约违反。
        missing_names = ", ".join(registry_report["missing_names"])  # 拼出缺失的常量名列表。
        contract_errors = "; ".join(registry_report["contract_errors"])  # 拼出契约错误列表。
        parts = []
        if missing_names:
            parts.append(f"missing_names=[{missing_names}]")
        if contract_errors:
            parts.append(f"contract_errors=[{contract_errors}]")
        raise RuntimeError(f"common.constants registry is incomplete: {'; '.join(parts)}")

    config_state_items = _load_config_state_items()  # 配置中的状态项顺序，复用 _load_config_state_items 保证单一真相。
    if len(config_state_items) != len(set(config_state_items)):  # 配置中不能有重复的状态项。
        raise ValueError("task.yaml state_items contains duplicate entries")  # 重复项会导致索引映射歧义。

    if len(state_items) != len(set(state_items)):  # 本地冻结定义也不能有重复项。
        raise ValueError("state_definition.py state_items contains duplicate entries")  # 重复项会导致索引映射歧义。

    if config_state_items != state_items:  # 配置和本地冻结定义必须完全一致，顺序也要相同。
        raise ValueError(
            "Frozen state_items in state_definition.py do not match configs/base/task.yaml"
        )  # 不一致说明有人改了一边但没同步另一边。

    missing_keys = [key for key in _REQUIRED_STATE_KEYS if key not in config_state_items]  # 检查必需键是否都在。
    if missing_keys:  # 缺少必需键会导致估计链无法运行。
        raise ValueError(f"task.yaml is missing required state keys: {missing_keys}")  # 必需键不能缺失。


def get_state_index_map() -> dict[str, int]:
    """返回冻结状态项到索引的映射。

    返回
    -------
    dict[str, int]
        状态名到其在状态向量中索引位置的映射字典。
    """
    _validate_upstream_contracts()  # 先校验一致性，确保映射基于正确的状态定义。
    return {state_item: index for index, state_item in enumerate(state_items)}  # 按冻结顺序生成索引映射。


state_index_map = MappingProxyType(get_state_index_map())  # 冻结为只读映射，防止运行时被意外改写，保证整条估计链使用一致的索引。
state_dim = len(state_items)  # 状态维度，供矩阵构造使用。
cov_dim = state_dim  # 协方差维度，与状态维度一致。
noise_dim = state_dim  # 噪声维度，供预测和更新步骤对齐。

from liquidloc.common.tee_logger import print_dict  # 局部导入参数打印工具。

print_dict(  # 打印状态定义全部冻结常量，供审计核对。
    {
        "state_items": list(state_items),
        "state_index_map": dict(state_index_map),
        "state_dim": state_dim,
        "cov_dim": cov_dim,
        "noise_dim": noise_dim,
    },
    "state_definition.py 常量",
    prefix="[配置]",
)
