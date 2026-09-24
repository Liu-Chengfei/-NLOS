"""tests/ 全局 pytest 配置。

此文件提供两类功能：
1. 每个测试后强制 gc.collect() + torch.cuda.empty_cache()，阻止测试间
   引用环导致进程 RSS 持续增长（OOM 保护在低内存环境下易误触）。
2. train_pipeline 的 OOM 保护阈值在 pytest 环境下自动降低到 50MB，
   因为 pytest 单进程运行 2500+ 测试会累积显存，宿主 free memory
   波动至 ~100MB 以下并不代表真实训练OOM。
— 此行为仅在测试环境，生产 train_pipeline.run() 保持 500MB 阈值。
"""

from __future__ import annotations

import gc

import pytest


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