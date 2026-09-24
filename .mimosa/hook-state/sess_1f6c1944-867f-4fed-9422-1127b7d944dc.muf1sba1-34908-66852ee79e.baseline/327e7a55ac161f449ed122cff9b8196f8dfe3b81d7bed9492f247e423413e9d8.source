"""pipelines 包的统一导出入口。

这个模块不实现具体流程，只负责把常用 pipeline 类集中导出，
让上层代码可以通过 `liquidloc.pipelines` 直接拿到标准入口。
其中 `EvalPipeline` 用惰性导入，是为了避免循环导入和不必要的启动开销。
"""

from liquidloc.pipelines.contract_smoke_pipeline import ContractSmokePipeline  # 导出最小合同烟雾测试流水线。
from liquidloc.pipelines.core_pipeline import CorePipeline  # 导出核心推理/评测调度流水线。
from liquidloc.pipelines.miluv_pipeline import MiluvPipeline  # 导出 MILUV 专用外部验证流水线。
from liquidloc.pipelines.prepare_pipeline import PreparePipeline  # 导出原始数据准备流水线。
from liquidloc.pipelines.public_benchmark_pipeline import PublicBenchmarkPipeline  # 导出公开基准路由流水线。
from liquidloc.pipelines.train_pipeline import TrainPipeline  # 导出训练流水线的统一入口。

__all__ = (  # 对外公开的符号列表，方便 `from ... import *` 和文档生成。
    "ContractSmokePipeline",  # 合同烟雾测试入口。
    "CorePipeline",  # 核心处理入口。
    "MiluvPipeline",  # MILUV 验证入口。
    "PreparePipeline",  # 数据准备入口。
    "PublicBenchmarkPipeline",  # 公开基准入口。
    "TrainPipeline",  # 训练入口。
)


def __getattr__(name: str):  # 按需解析未直接导入的属性。
    """对延迟导出的 pipeline 做懒加载。

    这里只对 `EvalPipeline` 做特殊处理，避免循环引用和不必要的导入开销。
    """
    if name == "EvalPipeline":  # 只有请求这个名字时才导入评测流水线。
        from liquidloc.pipelines.eval_pipeline import EvalPipeline  # 在真正需要时再导入。

        globals()[name] = EvalPipeline  # 缓存到模块全局，避免重复导入。
        return EvalPipeline  # 把类对象直接返回给调用方。
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")  # 其他名字按 Python 约定报属性错误。


__all__ = __all__ + ("EvalPipeline",)  # 把懒加载导出的名字也补进公开接口。
