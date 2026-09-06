"""流水线阶段抽象接口合同。

职责：
    定义所有可运行流水线阶段（pipeline stage）必须遵守的抽象合同。
    每个具体流水线（如 prepare、train、eval 等）都必须继承 PipelineAPI
    并实现 run 方法，以保证统一的调用面和返回类型。

上游依赖：
    - Python 标准库 abc              — 提供抽象基类和抽象方法装饰器
    - liquidloc.common.types         — 提供 StageResult 类型，作为 run 方法的返回类型

下游调用者：
    - liquidloc.pipelines.*          — 各具体流水线实现类继承 PipelineAPI
    - liquidloc.interfaces.__init__  — 统一导出 PipelineAPI 给外部使用
    - scripts / notebooks            — 通过 PipelineAPI 类型标注做依赖注入

核心变量：
    - PipelineAPI  — 流水线抽象基类，只定义一个 run 方法
"""

from __future__ import annotations  # 允许本文件内的类型注解使用前向引用，避免循环导入。

from abc import ABC, abstractmethod  # ABC 用于定义抽象基类，abstractmethod 用于标记必须由子类实现的方法。
from collections.abc import Mapping  # 统一校验流水线配置必须是映射，避免静默吞掉错误输入。
from typing import Any  # Any 用于 normalize_pipeline_cfg 的返回类型注解 dict[str, Any]，与 model_api.py 已修复模式对齐（D9 单源口径）。

from liquidloc.common.types import StageResult  # StageResult 是阶段性产物摘要类型，包含 stage_name、artifacts、metadata。

__all__ = ("PipelineAPI", "normalize_pipeline_cfg")  # 明确对外导出接口基类和统一配置规整 helper。


def normalize_pipeline_cfg(pipeline_cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    """把流水线配置规整为字典，并拒绝非映射输入。

    参数
    ----------
    pipeline_cfg : Mapping[str, Any] | None
        流水线配置，可以是映射或 ``None``。``None`` 统一视为空配置。

    返回
    -------
    dict[str, Any]
        从映射浅拷贝而来的字典副本；调用方修改顶层键不会污染原对象，
        但嵌套可变值仍与原对象共享引用（与 metric_factory 浅拷贝口径一致）。

    异常
    ------
    TypeError
        当 *pipeline_cfg* 既不是映射也不是 ``None`` 时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "pipeline_cfg_type": type(pipeline_cfg).__name__,
        "pipeline_cfg_keys": list(pipeline_cfg.keys()) if hasattr(pipeline_cfg, "keys") else None,
    }, "normalize_pipeline_cfg 入口参数")
    if pipeline_cfg is None:  # 允许不传配置，统一视为空配置。
        return {}
    if not isinstance(pipeline_cfg, Mapping):  # 非映射输入必须显式拒绝，不能静默洗成空字典。
        raise TypeError(
            f"pipeline_cfg must be a mapping or None, got {type(pipeline_cfg).__name__}"
        )  # 错误消息含实际类型，与 estimator_factory._require_mapping / metric_factory.create_metric_calculator 已修复模式对齐。
    return dict(pipeline_cfg)  # shallow copy：顶层键独立，嵌套值仍共享；与 metric_factory.create_metric_calculator 口径一致。


class PipelineAPI(ABC):  # 继承 ABC 使其成为抽象基类，不能直接实例化。
    """所有可运行流水线阶段的公共抽象合同。

    任何流水线阶段（如数据准备、模型训练、评估等）都必须继承此类
    并实现 run 方法。这样上层调度器可以用统一接口调用不同阶段，
    而不关心具体实现细节。

    上游：
        - 被具体流水线类继承（如 PreparePipeline、TrainPipeline、EvalPipeline）
    下游：
        - 调度器通过 run() 统一驱动各阶段
    """

    @abstractmethod  # 标记为抽象方法，子类必须实现，否则子类也无法实例化。
    def run(self, pipeline_cfg: dict | None = None, runtime_context: dict | None = None) -> StageResult:
        """运行一个流水线阶段并返回其阶段结果。

        这是流水线阶段的核心入口方法。子类必须实现此方法，
        在其中完成该阶段的全部工作（如数据准备、模型训练、指标评估等），
        并返回一个 StageResult 描述本阶段的产物和元数据。

        Args:
            pipeline_cfg (dict | None): 流水线配置字典，包含本阶段运行所需的
                超参数和路径等信息。默认为 None，表示使用子类内部默认配置。
            runtime_context (dict | None): 运行时上下文，预留给上层编排传入动态信息。
                当前各子类实现均不直接依赖此参数（设备选择由 pipeline_cfg 决定）。
                默认为 None，表示没有额外运行时信息。

        Returns:
            StageResult: 阶段性产物摘要，包含：
                - stage_name (str): 阶段名称，如 "prepare_pipeline"、"train_pipeline"、"eval_pipeline"、"core_pipeline"
                - artifacts (list[str]): 本阶段产出的文件路径列表
                - metadata (dict[str, Any]): 附加元数据字典

        Raises:
            TypeError: 子类未实现 run 时，由 ABC 机制在实例化该子类时抛出
                （本抽象方法体为占位，调用本身不抛异常）。
        """
        from liquidloc.common.tee_logger import print_dict
        print_dict({
            "has_pipeline_cfg": pipeline_cfg is not None,
            "has_runtime_context": runtime_context is not None,
        }, "PipelineAPI.run 入口参数")
        ...  # 省略号占位，表示此方法由子类实现，基类不提供默认行为。
