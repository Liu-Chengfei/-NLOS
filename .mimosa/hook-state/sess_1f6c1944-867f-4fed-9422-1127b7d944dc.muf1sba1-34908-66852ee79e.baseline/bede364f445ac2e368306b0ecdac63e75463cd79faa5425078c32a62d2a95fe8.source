"""interfaces 子包的统一导出入口。

职责：
    把三类抽象接口（EstimatorAPI、ModelAPI、PipelineAPI）统一导出，
    方便外部代码直接引用标准协议入口，而不需要知道每个接口的具体文件名。
    它存在的原因是把接口定义收束到一个稳定入口，避免调用方到处记内部文件名。

上游依赖：
    - liquidloc.interfaces.estimator_api  — 估计器抽象接口定义
    - liquidloc.interfaces.model_api      — 模型抽象接口定义
    - liquidloc.interfaces.pipeline_api   — 流水线抽象接口定义

下游调用者：
    - liquidloc.estimators.*   — 估计器实现类通过继承 EstimatorAPI 遵守合同
    - liquidloc.models.*       — 模型实现类通过继承 ModelAPI 遵守合同
    - liquidloc.pipelines.*    — 流水线实现类通过继承 PipelineAPI 遵守合同
    - scripts / notebooks      — 脚本和笔记本通过本入口获取接口类型做类型标注

核心变量：
    - EstimatorAPI          — 估计器抽象接口，定义 reset / step / get_state 三个方法
    - ModelAPI              — 模型抽象接口，定义 reset / infer_intermediate 两个方法
    - PipelineAPI           — 流水线抽象接口，定义 run 一个方法
    - normalize_pipeline_cfg — 流水线配置规整 helper，把 None / 映射统一转为 dict
"""

from liquidloc.interfaces.estimator_api import EstimatorAPI  # 估计器抽象接口，定义状态估计的标准合同。
from liquidloc.interfaces.model_api import ModelAPI  # 模型抽象接口，定义中间态推断的标准合同。
from liquidloc.interfaces.pipeline_api import PipelineAPI, normalize_pipeline_cfg  # 流水线抽象接口及配置规整 helper。

__all__ = ("EstimatorAPI", "ModelAPI", "PipelineAPI", "normalize_pipeline_cfg")  # 明确 interfaces 子包对外允许导出的接口名。
