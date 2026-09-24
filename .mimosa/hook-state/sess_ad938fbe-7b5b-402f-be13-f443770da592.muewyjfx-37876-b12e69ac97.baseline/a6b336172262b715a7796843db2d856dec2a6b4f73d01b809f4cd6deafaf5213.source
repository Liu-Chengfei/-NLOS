"""估计器抽象接口的对外导出入口。

职责：
    本模块仅作为 EstimatorAPI 的对外统一导出入口（re-export）。
    EstimatorAPI 的权威定义已迁移至 liquidloc.estimators.estimator_api，
    以消除 estimators → interfaces 的逆向依赖。

    保留此文件是为了向后兼容：所有从 liquidloc.interfaces.estimator_api
    导入 EstimatorAPI 的代码无需修改。

上游依赖：
    - liquidloc.estimators.estimator_api  — EstimatorAPI 的权威定义位置

下游调用者：
    - liquidloc.interfaces.__init__  — 统一导出 EstimatorAPI 给外部使用
    - 外部代码通过本模块导入 EstimatorAPI（向后兼容）
"""

from liquidloc.estimators.estimator_api import EstimatorAPI  # 从权威定义位置 re-export，保持向后兼容。

__all__ = ("EstimatorAPI",)  # 明确本模块对外只导出 EstimatorAPI 一个名字。


if __name__ == "__main__":  # 仅在直接运行时打印重导出信息，避免 import 侧效应。
    from liquidloc.common.tee_logger import print_dict
    print_dict({"re_export": "EstimatorAPI", "source": "liquidloc.estimators.estimator_api"}, "estimator_api 重导出")
