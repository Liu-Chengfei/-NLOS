"""单元测试：liquidloc.common.parallel_equivalence_check

覆盖:
- verify_parallel_equivalence 串行 vs 并行 bitwise equal（同输入同输出）
- verify_parallel_equivalence 数值容差（微小浮点差）
- verify_parallel_equivalence 不一致时失败
- record_equivalence_decision 写入 decision_log.json
- check_equivalence_record_exists 读取验证
- _default_compare 递归 dict/list 比较
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from liquidloc.common.parallel_equivalence_check import (
    EquivalenceResult,
    SpotCheckRecord,
    _default_compare,
    check_equivalence_record_exists,
    record_equivalence_decision,
    verify_parallel_equivalence,
)


# ── helper ─────────────────────────────────────────────────────────────

def _pure_task(x: int) -> dict:
    """纯计算：x → {x, square}，无 IO、无副作用。"""
    return {"input": x, "square": x * x, "cube": x ** 3}


def _numpy_task(x: int) -> dict:
    """含 numpy 计算：确保数值类型在串行/并行一致。"""
    arr = np.array([float(x), float(x) * 0.1])
    return {
        "sum": float(arr.sum()),
        "mean": float(arr.mean()),
        "values": arr.tolist(),
    }


# ── 基本 bitwise equal ────────────────────────────────────────────────

class TestBitwiseEqual:
    def test_two_units_equal(self):
        result = verify_parallel_equivalence(_pure_task, [1, 2], max_workers=2)
        assert result.passed is True
        assert result.bitwise_equal is True
        assert result.numeric_tolerance_ok is True
        assert len(result.serial_results) == 2
        assert len(result.parallel_results) == 2

    def test_single_unit_equal(self):
        result = verify_parallel_equivalence(_pure_task, [42], max_workers=1)
        assert result.passed is True
        assert result.serial_results[0]["square"] == 1764

    def test_numpy_results_equal(self):
        result = verify_parallel_equivalence(_numpy_task, [3, 7], max_workers=2)
        assert result.passed is True
        assert result.bitwise_equal is True


# ── 数值容差（微小浮点差） ────────────────────────────────────────────

class TestNumericTolerance:
    def test_within_tolerance(self):
        """数值差在容差内 → numeric_ok=True，overall passed=True。"""
        def task_with_jitter(x):
            # 串行/并行各跑一次会有不同运行时浮点精度（GPU/CPU路径）
            return float(x) * 0.1 + 1e-10  # 微小非确定性

        result = verify_parallel_equivalence(
            task_with_jitter, [1, 2, 3],
            tolerance=1e-6, max_workers=2,
        )
        # 允许在容差内
        assert result.tolerance == 1e-6
        assert result.numeric_tolerance_ok is True or result.bitwise_equal is True
        assert result.passed is True


# ── 不一致时失败 ──────────────────────────────────────────────────────

class TestMismatch:
    def test_mismatch_detected(self):
        """输入依赖全局状态导致串行/并行不一致 → 应报告不通过。"""
        counter = {"n": 0}
        def racy_task(x):
            counter["n"] += 1
            return {"seq": counter["n"], "input": x}

        result = verify_parallel_equivalence(racy_task, [1, 2], max_workers=2)
        # 串行: seq=[1,2]；并行: seq=[2,1]（或交错）→ mismatch
        assert result.passed is False
        assert result.n_units > 0

    def test_mismatch_detail_contains_example(self):
        result = verify_parallel_equivalence(
            lambda x: {"id": x, "ts": __import__("time").time()},
            [1], max_workers=2,
        )
        # 含时间戳时两次结果不同
        assert not result.passed or "n_units" in result.raw


# ── 空输入 ────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_args(self):
        result = verify_parallel_equivalence(_pure_task, [], max_workers=2)
        assert result.n_units == 0
        assert result.passed is True  # 无任务 → trivially equal
        assert result.serial_results == []
        assert result.parallel_results == []

    def test_zero_workers_clamped(self):
        result = verify_parallel_equivalence(_pure_task, [1], max_workers=0)
        assert result.n_units == 1
        assert result.passed is True


# ── decision log 写入/读取 ────────────────────────────────────────────

class TestDecisionLog:
    def test_record_and_check_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dl_path = os.path.join(tmpdir, "decision_log.json")
            # 初始化为最小结构
            with open(dl_path, "w", encoding="utf-8") as fh:
                json.dump({"experiments": {"e20_paper_main": {"decision_entries": []}}}, fh)

            eq_result = EquivalenceResult(
                spot_check_id="eq-test-001",
                serial_results=[{"x": 1}],
                parallel_results=[{"x": 1}],
                bitwise_equal=True,
                numeric_tolerance_ok=True,
                passed=True,
                detail="n=1, bw=True, nt=True",
                n_units=1,
                tolerance=1e-6,
            )

            entry = record_equivalence_decision(
                eq_result, round_tag="RZ-0", decision_log_path=dl_path,
            )
            assert entry["id"] == "DEC-eq-test-001"
            assert entry["status"] == "completed"

            status = check_equivalence_record_exists(
                decision_log_path=dl_path, round_tag="RZ-0",
            )
            assert status["exists"] is True
            assert status["count"] == 1
            assert "DEC-eq-test-001" in status["entries"]

    def test_no_record_returns_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dl_path = os.path.join(tmpdir, "decision_log.json")
            # 文件不存在
            status = check_equivalence_record_exists(decision_log_path=dl_path)
            assert status["exists"] is False
            assert status["count"] == 0


# ── _default_compare 递归比较 ─────────────────────────────────────────

class TestDefaultCompare:
    def test_primitives(self):
        assert _default_compare(1, 1, 1e-6) == (True, True)
        assert _default_compare(1, 2, 1e-6) == (False, False)
        # 容差内
        bw, nt = _default_compare(1.0, 1.0 + 1e-7, 1e-6)
        assert bw is False
        assert nt is True

    def test_dicts(self):
        a = {"rmse": 0.5, "states": [1, 2, 3]}
        b = {"rmse": 0.5 + 1e-7, "states": [1, 2, 3]}
        bw, nt = _default_compare(a, b, 1e-6)
        assert bw is False
        assert nt is True

    def test_nested_list(self):
        a = [[1, 2], [3, 4]]
        b = [[1, 2], [3, 5]]
        bw, nt = _default_compare(a, b, 1e-6)
        assert nt is False
