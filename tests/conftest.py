"""tests/ 全局 pytest 配置。

背景：test_train_pipeline 等重内存测试文件在单进程内连续执行时，
torch 张量与 fixture 数据形成的引用环会让进程 RSS 持续增长，
触发 train_pipeline 的 OOM 保护（系统可用内存 <500MB 即 fail-loud）。
此钩子在每个测试结束后强制 gc 回收并释放 CUDA 缓存，
降低进程常驻内存，使 OOM 保护只在真实内存压力下触发。
"""

from __future__ import annotations

import gc
import os

import pytest

# 全套件单进程运行时，前面 2500+ 个测试的模块导入与张量分配会累积进程 RSS，
# 使 train_pipeline 的 OOM 保护（默认系统可用 <500MB 即 fail-loud）在轮到
# tests/pipelines/test_train_pipeline.py 时必然触发。测试进程在此显式声明
# 更低的保护阈值（仍保留 150MB 硬 OOM 防护）；真实训练进程不设该变量，
# 生产保护行为不变。
os.environ.setdefault("LIQUIDLOC_OOM_MIN_AVAIL_MB", "150")


@pytest.fixture(autouse=True)
def _release_memory_after_test():
    yield  # 先运行测试本身。
    gc.collect()  # 断开张量/fixture 引用环，交还堆内存。
    try:  # CUDA 缓存属于进程级预留，测试结束后显式释放。
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # torch 未安装或 CUDA 初始化失败时静默跳过。
        pass
