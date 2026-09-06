"""
analysis 子包的统一导出入口。

这个文件把 ablation、calibration、case 选择、显著性检验和 summary 构建函数
集中导出，方便上层脚本和测试直接按 `liquidloc.analysis` 取用常见分析 API。
它本身不做分析计算，只负责整理对外公开的函数名。
"""

from liquidloc.analysis.ablation_analysis import align_ablation_tables, build_ablation_table, compare_one_ablation  # 导出消融比较相关函数。
from liquidloc.analysis.calibration_analysis import align_traces, build_calibration_report, compute_trace_correlations  # 导出校准分析相关函数。
from liquidloc.analysis.case_selector import select_cases  # 导出案例选择函数。
from liquidloc.analysis.case_selector_runner import build_empty_selected_cases, select_best_method, select_case_rules  # 导出案例选择编排函数。
from liquidloc.analysis.significance_tests import build_statistics_table, group_metric_values, run_pairwise_tests, run_significance_tests  # 导出统计检验相关函数。
from liquidloc.analysis.statistics_runner import build_smoke_only_statistics_payload, build_statistics_payload  # 导出统计编排函数。
from liquidloc.analysis.summary_builder import build_summary  # 导出 summary 构建函数。

__all__ = (  # 明确告诉外部这个包允许导出的名字。
    "align_ablation_tables",  # 消融表对齐函数。
    "align_traces",  # trace 对齐函数。
    "build_ablation_table",  # 消融汇总表构建函数。
    "build_calibration_report",  # 校准报告构建函数。
    "build_empty_selected_cases",  # 空选例容器构建函数。
    "build_smoke_only_statistics_payload",  # smoke-only 统计载荷构建函数。
    "build_statistics_payload",  # 统计编排载荷构建函数。
    "build_statistics_table",  # 统计表构建函数。
    "build_summary",  # summary 构建函数。
    "compare_one_ablation",  # 单个消融比较函数。
    "compute_trace_correlations",  # trace 相关性计算函数。
    "group_metric_values",  # 指标分组函数。
    "run_pairwise_tests",  # 成对统计检验函数。
    "run_significance_tests",  # 统计检验总入口。
    "select_best_method",  # 最优方法选择函数。
    "select_case_rules",  # 案例分组规则生成函数。
    "select_cases",  # 案例选择函数。
)  # 导出列表结束。
