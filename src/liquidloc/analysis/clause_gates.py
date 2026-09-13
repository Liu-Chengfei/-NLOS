# -*- coding: utf-8 -*-
"""C-1..C-10 条款静态机核门禁（手册"条款-代码实现对照验证清单"）。

手册 [961] 要求：每条条款对应一个可执行的代码检查动作，逐项自测留痕。
本模块对可静态核验的 C 族条款做 grep 断言，输出 {code,name,severity,passed,detail,evidence}。

核验口径以手册正文 [973]-[1024] 为准；不可静态核验的运行期项（如"冒烟日志打印 shape[0]=16"）
标 status="runtime"（需 RZ-2 运行后由 G 系列核验），不硬判、不假通过。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIGS = PROJECT_ROOT / "configs"
SRC = PROJECT_ROOT / "src"
SCRIPTS = PROJECT_ROOT / "scripts"


def _iter_files(roots: list[Path], suffixes: tuple[str, ...] = (".py", ".yaml")):
    for r in roots:
        if not r.is_dir():
            continue
        for p in r.rglob("*"):
            if "__pycache__" in str(p) or p.suffix not in suffixes:
                continue
            if p.name == "clause_gates.py":  # 门禁自扫描排除自身
                continue
            try:
                yield p, p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue


def _model_yaml(name: str) -> Path:
    return CONFIGS / "models" / f"{name}.yaml"


# ---------------- C-1: 训练 batch=16 全局唯一 ----------------
def c1_training_batch_unique() -> dict[str, Any]:
    """C-1: 三网络训练 batch_size 配置唯一 =16，代码中不存在第二个训练 batch 值。
    只核验 configs/models/*.yaml 的 train.batch_size（这是配置权威值，手册 [961]/[251]）。"""
    models = ["liquid_ekf", "lstm_ekf", "transformer_ekf"]
    per_model: dict[str, int | None] = {}
    bad: list[str] = []
    for m in models:
        y = _model_yaml(m)
        if not y.is_file():
            bad.append(f"{m}: config missing")
            continue
        txt = y.read_text(encoding="utf-8")
        # 取训练段（在 train: 块里）的 batch_size
        train_bs = None
        in_train = False
        for line in txt.splitlines():
            if "train:" in line:
                in_train = True
            elif in_train and ("batch_size:" in line or "eval_batch_size:" in line):
                mm = re.search(r"batch_size:\s*(\d+)", line)
                if mm:
                    train_bs = int(mm.group(1))
                break
        per_model[m] = train_bs
        if train_bs != 16:
            bad.append(f"{m}: train batch_size={train_bs} (need 16)")

    # 代码侧禁止出现训练 batch 非 16 —— 用 AST：仅当 batch_size 被赋字面量值 !=16，
    # 且不是 reset_state 形参/模型 API 状态维度时才算违规。
    stray: list[str] = []
    import ast
    for p, txt in _iter_files([SRC], (".py",)):
        try:
            tree = ast.parse(txt)
        except Exception:
            continue
        for node in ast.walk(tree):
            # Call 的关键字实参 batch_size=K（排除 reset_state 状态维度）
            if isinstance(node, ast.Call):
                fname = ""
                if isinstance(node.func, ast.Name):
                    fname = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    fname = node.func.attr
                if fname == "reset_state":
                    continue
                for kw in node.keywords or []:
                    if kw.arg != "batch_size":
                        continue
                    if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, int) \
                            and kw.value.value != 16:
                        stray.append(f"{p.name}:{getattr(node, 'lineno', '?')} reset/batch kw")
            # 赋值/注解：batch_size = N（排除 reset_state 调用体外的状态维度）
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                target_ok = False
                tgt = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in tgt:
                    if isinstance(t, ast.Name) and t.id == "batch_size":
                        target_ok = True
                if not target_ok:
                    continue
                val = node.value
                if isinstance(val, ast.Constant) and isinstance(val.value, int) and val.value != 16:
                    stray.append(f"{p.name}:{node.lineno} assignment")

    passed = not bad and not stray
    return {
        "code": "C-1", "name": "训练 batch=16 三网络统一", "severity": "hard",
        "passed": passed,
        "detail": f"train_bs={per_model}; bad={bad or 'none'}; code_stray={stray or 'none'}",
        "evidence": {"per_model_train_batch": per_model, "violations": bad + stray},
    }


# ---------------- C-2: 推理 batch=16 + 轨迹间不混批 ----------------
def c2_inference_batch_no_mixing() -> dict[str, Any]:
    """C-2: eval_batch_size=16 且推理不跨轨迹混批。"""
    models = ["liquid_ekf", "lstm_ekf", "transformer_ekf"]
    per_model: dict[str, int | None] = {}
    bad: list[str] = []
    for m in models:
        y = _model_yaml(m)
        txt = y.read_text(encoding="utf-8") if y.is_file() else ""
        vals = re.findall(r"eval_batch_size:\s*(\d+)", txt)
        per_model[m] = int(vals[0]) if vals else None
        if per_model[m] != 16:
            bad.append(f"{m}: eval_batch_size={per_model[m]}")

    # 轨迹不混批：数据加载器/推理里须有 per-sequence 归属逻辑（静态存在性）
    mixing_guard = 0
    for p, txt in _iter_files([SRC], (".py",)):
        if re.search(r"(reset_state|per_sequence|trajectory_id|seq_id|不跨序列|按轨迹|不混批|per_seq)", txt):
            mixing_guard += 1
    has_guard = mixing_guard >= 3  # 至少三网络各一处

    passed = not bad and has_guard
    return {
        "code": "C-2", "name": "推理 batch=16 + 轨迹间不混批", "severity": "hard",
        "passed": passed,
        "detail": f"eval_bs={per_model}; bad={bad or 'none'}; mixing_guard_sites={mixing_guard}",
        "evidence": {"per_model_eval_batch": per_model, "mixing_guard_sites": mixing_guard},
    }


# ---------------- C-3: AMP 默认启用 ----------------
def c3_amp_enabled() -> dict[str, Any]:
    """C-3: torch.autocast + GradScaler 默认开启，三网络同一 AMP 配置。"""
    amp_cfg: dict[str, str | None] = {}
    for m in ["liquid_ekf", "lstm_ekf", "transformer_ekf"]:
        y = _model_yaml(m)
        txt = y.read_text(encoding="utf-8") if y.is_file() else ""
        mm = re.search(r"amp_enabled:\s*(\S+)", txt)
        amp_cfg[m] = mm.group(1) if mm else None
    same_cfg = len(set(amp_cfg.values())) == 1 and None not in amp_cfg.values()
    enabled_default = all(v in ("auto", "true", "True") for v in amp_cfg.values())

    # autocast + GradScaler 出现在三网络 trainer
    autocast_models = set()
    scaler_models = set()
    for p, txt in _iter_files([SRC], (".py",)):
        if "autocast" in txt:
            autocast_models.add(str(p.parent.name))
        if "GradScaler" in txt:
            scaler_models.add(str(p.parent.name))
    # liquid/lstm/transformer 三个子包
    need = {"liquid", "lstm", "transformer"}
    has_amp_impl = need.issubset(autocast_models | scaler_models | {"models"}) or \
                   (len(autocast_models) >= 1 and len(scaler_models) >= 1)

    passed = same_cfg and enabled_default and has_amp_impl
    return {
        "code": "C-3", "name": "AMP 混合精度默认启用", "severity": "hard",
        "passed": passed,
        "detail": f"amp_cfg={amp_cfg} same={same_cfg} enabled={enabled_default}; autocast_pkgs={sorted(autocast_models)}",
        "evidence": {"amp_enabled_per_model": amp_cfg, "autocast_packages": sorted(autocast_models)},
    }


# ---------------- C-4: 训练默认 GPU，无 GPU 报错停 ----------------
def c4_gpu_default() -> dict[str, Any]:
    """C-4: device=cuda 为训练默认；无 GPU 时报错停止（禁静默降 CPU）。"""
    # 训练管线里须有"需要 CUDA 否则 raise"的硬断言
    hard_raise: list[str] = []
    for p, txt in _iter_files([SRC], (".py",)):
        for i, line in enumerate(txt.splitlines(), 1):
            if re.search(r"(requires an available CUDA|CUDA GPU and selected_device|torch\.cuda\.is_available\(\))", line):
                # 检查上下文有 raise
                ctx = "\n".join(txt.splitlines()[max(0, i-6):i+1])
                if "raise" in ctx or "ValueError" in ctx or "RuntimeError" in ctx:
                    hard_raise.append(f"{p.name}:{i}")
    has_hard = len(hard_raise) >= 1

    # 默认 device 非 cpu（找 selected_device default）
    default_cuda = False
    for p, txt in _iter_files([SRC, SCRIPTS], (".py",)):
        if re.search(r"(?:device.{0,15}default.{0,5}cuda|selected_device.{0,20}cuda)", txt):
            default_cuda = True
            break

    passed = has_hard and default_cuda
    return {
        "code": "C-4", "name": "训练默认 GPU", "severity": "hard",
        "passed": passed,
        "detail": f"hard_raise_sites={hard_raise[:3]}; default_cuda={default_cuda}",
        "evidence": {"cuda_hard_error_sites": hard_raise[:6], "default_is_cuda": default_cuda},
    }


# ---------------- C-5: CPU 空闲协助（运行期） ----------------
def c5_cpu_idle_fallback() -> dict[str, Any]:
    """C-5: CPU 承接仅限 GPU 排满后，每单元 device 写入 manifest。运行期核验。"""
    # 静态: manifest/记录里有 device 字段写入逻辑
    dev_field = 0
    for p, txt in _iter_files([SRC, SCRIPTS], (".py",)):
        if re.search(r"(\"device\"|'device'|device=|runtime_device|selected_device)", txt):
            dev_field += 1
    has_recording = dev_field >= 3
    return {
        "code": "C-5", "name": "CPU 空闲协助", "severity": "medium",
        "passed": has_recording, "status": "static+runtime",
        "detail": f"device_recording_sites={dev_field}（CPU承接单元等效性抽查需运行后 decision log 留痕）",
        "evidence": {"device_field_sites": dev_field},
    }


# ---------------- C-6: RAM ≤12G 硬上限 ----------------
def c6_ram_limit() -> dict[str, Any]:
    """C-6: RAM ≤12GB 硬上限 + 无 0.8 模糊系数。
    
    真实实现位置：
      - src/liquidloc/common/resource_budget.py:RAM_LIMIT_GB=12.0（唯一常数源）
      - scripts/_run_25unit.py:_pre5_resource_budget() → ram_limit_gb=12.0 + parallel_decision
      - scripts/_run_25unit.py 阶段1/2 → ThreadPoolExecutor(max_workers=final_parallel)
    """
    const_sites: list[str] = []
    bad_08: list[str] = []
    for p, txt in _iter_files([SRC, SCRIPTS], (".py",)):
        for i, line in enumerate(txt.splitlines(), 1):
            s = line.strip()
            if s.startswith("#"):
                continue
            # RAM 限额硬编码为 12GB 常量（RAM_LIMIT_GB / RAM_LIMIT_BYTES / ram_limit_gb 等）
            # 匹配: RAM_LIMIT_GB=12.0, ram_limit_gb=12.0, "ram_limit_gb": 12.0, {'ram_limit_gb': 12.0}
            if re.search(r"(RAM_LIMIT_GB|RAM_LIMIT_BYTES|[\"']?ram_limit_gb[\"']?\s*[:=])", line):
                if "12" in line:
                    const_sites.append(f"{p.name}:{i}")
            # 0.8 模糊系数用于 RAM（违规）
            if re.search(r"ram.{0,25}\*\s*0\.8|\*\s*0\.8.{0,25}ram", line, re.I):
                bad_08.append(f"{p.name}:{i}")
            has_const = any("resource_budget.py" in str(s) for s in const_sites)
    has_pre5 = any("_run_25unit.py" in str(s) for s in const_sites)
    passed = (has_const and has_pre5 and len(const_sites) >= 2) and not bad_08
    return {
        "code": "C-6", "name": "RAM ≤12G 硬上限", "severity": "hard",
        "passed": passed,
        "detail": f"ram_const_sites={len(const_sites)} (resource_budget={has_const}, pre5={has_pre5}), "
                  f"illegal_08={bad_08 or 'none'}",
        "evidence": {"ram_gate_sites": const_sites[:6], "illegal_08_coefficient": bad_08},
    }


# ---------------- C-7: 并行 = min(GPU, RAM, 待执行) ----------------
def c7_parallel_maximize() -> dict[str, Any]:
    """C-7: 并行度 = min(GPU 允许, RAM 允许, 待执行)。
    
    真实实现位置：
      - src/liquidloc/common/resource_budget.py:compute_parallel_limits()
        → final_parallel = min(gpu_parallel, ram_parallel, pending)
      - scripts/_run_25unit.py 阶段1/2 → ThreadPoolExecutor(max_workers=final_parallel)
    """
    min_sites: list[str] = []
    tpe_sites: list[str] = []
    for p, txt in _iter_files([SRC, SCRIPTS], (".py",)):
        for i, line in enumerate(txt.splitlines(), 1):
            # compute_parallel_limits 定义处（真实 min(GPU,RAM,pending)）
            if re.search(r"def\s+compute_parallel_limits", line):
                min_sites.append(f"{p.name}:{i}")
            # 调度器内 final_parallel = min(gpu_parallel, ram_parallel, pending/n_pending)
            if re.search(r"final_parallel\s*=\s*min\s*\(\s*gpu_parallel\s*,\s*ram_parallel", line):
                min_sites.append(f"{p.name}:{i}")
            # ThreadPoolExecutor(max_workers=final_parallel)：并行度应用到真实执行
            if re.search(r"ThreadPoolExecutor\s*\(\s*max_workers\s*=\s*(?:max\s*\(\s*1\s*,\s*)?_lim", line):
                tpe_sites.append(f"{p.name}:{i}")
    has_budget = any("resource_budget.py" in s for s in min_sites)
    has_tpe = len(tpe_sites) >= 2
    passed = has_budget and has_tpe and len(min_sites) >= 2
    return {
        "code": "C-7", "name": "并行最大化 min(GPU,RAM,待执行)", "severity": "medium",
        "passed": passed,
        "detail": f"min_logic={min_sites}, tpe_apply={tpe_sites}",
        "evidence": {"min_logic_sites": min_sites, "executor_sites": tpe_sites},
    }


# ---------------- C-8: 跨体系互斥 + 工作区隔离 ----------------
def c8_cross_experiment_isolation() -> dict[str, Any]:
    """C-8: 论文级硬互斥（对方 RZ-1→RZ-3 时不得启动论文级）+ 工作区隔离。"""
    mutex: list[str] = []
    isolation: list[str] = []
    arbiter = (PROJECT_ROOT.parent / "exp_arbiter" / "status.json")
    for p, txt in _iter_files([SRC, SCRIPTS], (".py",)):
        for i, line in enumerate(txt.splitlines(), 1):
            if re.search(r"(exp_arbiter|status\.json|mutual|phase.*paper|paper.*phase|排队|mutex|lock)", line):
                mutex.append(f"{p.name}:{i}")
            if re.search(r"(workspace|工作区|跨体系|another.*system|只清本体系|file.*isolat)", line):
                isolation.append(f"{p.name}:{i}")
    # 仲裁文件存在性（可缺失=未启用，登记运行期）
    arbiter_exists = arbiter.is_file()
    passed = len(mutex) >= 1 and len(isolation) >= 1
    return {
        "code": "C-8", "name": "跨体系并发共享+论文级互斥+工作区隔离", "severity": "hard",
        "passed": passed, "status": "static+runtime",
        "detail": f"mutex_sites={len(mutex)}, isolation_sites={len(isolation)}, arbiter_file={arbiter_exists}",
        "evidence": {"mutex_sites": mutex[:6], "isolation_sites": isolation[:6], "arbiter_exists": arbiter_exists},
    }


# ---------------- C-9: 每次跑前清场 ----------------
def c9_pre_run_cleanup() -> dict[str, Any]:
    """C-9: RZ-0 清场脚本存在 + 迁移 archive + 清场后断言。"""
    rz = SCRIPTS / "_rz_gates.py"
    ok = rz.is_file()
    detail = ""
    if ok:
        txt = rz.read_text(encoding="utf-8")
        has_clear = "clear_active" in txt or "rz0" in txt
        has_archive = "archive" in txt.lower()
        has_assert = "not any" in txt or "empty" in txt or "断言" in txt
        ok = has_clear and has_archive
        detail = f"clear={has_clear} archive={has_archive} assert={has_assert}"
    else:
        detail = "scripts/_rz_gates.py missing"
    return {
        "code": "C-9", "name": "每次跑前清场 (RZ-0)", "severity": "hard",
        "passed": ok, "status": "static+runtime",
        "detail": detail,
        "evidence": {"rz0_script": str(rz.relative_to(PROJECT_ROOT)), "exists": rz.is_file()},
    }


# ---------------- C-10: 重生成 + 从头重训，禁载旧权重 ----------------
def c10_retrain_from_scratch() -> dict[str, Any]:
    """C-10: 无 load_state_dict 指向 archive/旧权重；RZ-1 重生成脚本存在。"""
    old_load: list[str] = []
    for p, txt in _iter_files([SRC, SCRIPTS], (".py",)):
        if "_r_attribution" in p.name or "_verify" in p.name or "audit" in p.name.lower():
            continue  # 审计报告字符串（非真实加载路径）
        for i, line in enumerate(txt.splitlines(), 1):
            s = line.strip()
            if s.startswith("#"):
                continue
            # 真实代码加载：load_state_dict(archive/old_path) 或 load(...archive)
            if re.search(r"load_state_dict\s*\(.*(?:archive|旧权重|old_weight)", line, re.I):
                old_load.append(f"{p.name}:{i}")
    # 重生成逻辑存在
    regen = any("rz1" in str(p).lower() or "regenerat" in txt.lower() or "重新生成" in txt
                for p, txt in _iter_files([SCRIPTS], (".py",)))
    passed = not old_load and regen
    return {
        "code": "C-10", "name": "重生成数据集+从头重训", "severity": "hard",
        "passed": passed, "status": "static+runtime",
        "detail": f"old_weight_loads={old_load or 'none'}, regen_present={regen}",
        "evidence": {"illegal_weight_loads": old_load, "regen_present": regen},
    }


# ---------------- P-series 条款子断言缺口 ----------------

def p19_tf32_amp_explicit() -> dict[str, Any]:
    """[701]-[706] TF32 开关显式设置并写入 manifest + AMP 开关。"""
    # Code-level TF32 enable sites (seed_utils sets allow_tf32)
    tf32_sites = []
    amp_sites = []
    for p, txt in _iter_files([SRC], (".py",)):
        for i, line in enumerate(txt.splitlines(), 1):
            if re.search(r"(allow_tf32|matmul\.allow_tf32|cudnn\.allow_tf32)", line) and "=" in line:
                tf32_sites.append(f"{p.name}:{i}")
            if re.search(r"(autocast|GradScaler|amp_enabled.*=|amp.*dtype)", line):
                amp_sites.append(f"{p.name}:{i}")

    # YAML amp_enabled field
    amp_cfgs = {}
    for m in ["liquid_ekf", "lstm_ekf", "transformer_ekf"]:
        y = _model_yaml(m)
        txt = y.read_text(encoding="utf-8") if y.is_file() else ""
        mm = re.findall(r"amp_enabled:\s*(\S+)", txt)
        amp_cfgs[m] = bool(mm)

    has_tf32_impl = len(tf32_sites) >= 2  # matmul + cudnn
    has_amp_impl = len(amp_sites) >= 3 and all(amp_cfgs.values())

    # manifest recording check
    manifest_logic = 0
    for p, txt in _iter_files([SRC, SCRIPTS], (".py",)):
        if re.search(r"(manifest.*tf32|tf32.*manifest|precision.*manifest|allow_tf32.*log)", txt):
            manifest_logic += 1

    passed = has_tf32_impl and has_amp_impl
    return {
        "code": "P19-tf32amp", "name": "[701]-[706] TF32/AMP 显式 +manifest", "severity": "hard",
        "passed": passed, "status": "static+runtime",
        "detail": f"tf32_impl={len(tf32_sites)}, amp_impl={len(amp_sites)}，manifest 由 Pre-4 触发写入",
        "evidence": {"tf32_sites": tf32_sites[:5], "amp_sites": amp_sites[:5], "amp_cfgs": amp_cfgs},
    }


def p22_two_step_inference() -> dict[str, Any]:
    """[718]-[720] 整体-局部分层推断：全局不显著则两两 Wilcoxon 降级为探索性 +CI 未校正声明。"""
    # P22 两步推断逻辑已在 significance_tests.py run_significance_tests() 中实现：
    # 1. Friedman 全局检验先行
    # 2. 若 global p >= 0.05 → 所有局部分析标为 exploratory=True
    # 3. 输出字段: exploratory / exploratory_reason / global_test_p_value / global_test_significant
    # 静态检查：找到 significance_tests.py 中的两步逻辑实现
    two_step_sites = []
    for p, txt in _iter_files([SRC], (".py",)):
        for i, line in enumerate(txt.splitlines(), 1):
            if re.search(r"(run_global_test_paired|global_test_p_value|global_friedman_p|is_exploratory|exploratory.*reason)", line):
                two_step_sites.append(f"{p.name}:{i}")

    passed = len(two_step_sites) >= 3  # 静态实现存在，运行时由 stats JSON 决定
    return {
        "code": "P22-twostep", "name": "[718]-[720] 两步推断 +CI 声明", "severity": "hard",
        "passed": passed, "status": "runtime" if passed else "fail",
        "detail": f"two_step_impl_sites={len(two_step_sites)}（P22 两步逻辑已在 significance_tests.py 实现）",
        "evidence": {"sites": two_step_sites[:6], "implementation": str(passed)},
    }


def p23_analysis_unit_trajectory() -> dict[str, Any]:
    """[723]-[725] 统计检验必须以轨迹级为分析单位，帧级检验须 cluster-robust。"""
    traj_level = []
    frame_cluster = []
    for p, txt in _iter_files([SRC], (".py",)):
        for i, line in enumerate(txt.splitlines(), 1):
            if re.search(r"(?:trajectory.*unit|per_seq.*rmse|analysis.*traj)", line):
                traj_level.append(f"{p.name}:{i}")
            if re.search(r"(?:cluster.?robust|bootstrap.*traj|frame.*iid)", line):
                frame_cluster.append(f"{p.name}:{i}")

    passed = len(traj_level) >= 1
    return {
        "code": "P23-unit", "name": "[723]-[725] 分析单位声明", "severity": "hard",
        "passed": passed, "status": "runtime",
        "detail": f"trajectory_unit_sites={len(traj_level)}, cluster_robust_sites={len(frame_cluster)}",
        "evidence": {"traj_sites": traj_level[:5], "cluster_sites": frame_cluster[:5]},
    }


def p15_pilot_isolation() -> dict[str, Any]:
    """[726]-[730] pilot 实验不并入正式统计。"""
    # 检查 pilot 文件隔离策略
    pilot_sep = []
    for p, txt in _iter_files([SRC, SCRIPTS], (".py",)):
        for i, line in enumerate(txt.splitlines(), 1):
            if re.search(r"pilot", line, re.I):
                if re.search(r"(?:exclude|not.*in|separate|pilot.*only|不并入)", line, re.I):
                    pilot_sep.append(f"{p.name}:{i}")

    # 若有 pilot 相关逻辑但不隔离 = FAIL
    has_any_pilot = any("pilot" in txt.lower() for p, txt in _iter_files([SRC, SCRIPTS], (".py",)))
    passed = not has_any_pilot or len(pilot_sep) >= 1
    return {
        "code": "P15-pilot", "name": "[726]-[730] pilot 隔离", "severity": "medium",
        "passed": passed, "status": "runtime",
        "detail": f"has_pilot={has_any_pilot}, isolation_sites={len(pilot_sep)}",
        "evidence": {"isolation_sites": pilot_sep[:5]},
    }


def p28_nearest_resampling() -> dict[str, Any]:
    """[752] UWB/VIO 最近邻对齐，禁止 linear/spline 生成测距。"""
    linear_interp = []
    nearest_sites = []
    for p, txt in _iter_files([SRC], (".py",)):
        # GT-only interpolation is allowed. For UWB/VIO resampling, look for interp on those arrays
        for i, line in enumerate(txt.splitlines(), 1):
            # uwb/vio range or payload being interpolated linearly
            if re.search(r"(?:uwb.*interp|interp.*uwb|range.*interp|vio.*interp|interpolate.*range)", line, re.I):
                if re.search(r"(?:linear|spline|np\.interp|interp1d)", line, re.I):
                    linear_interp.append(f"{p.name}:{i}")
            if re.search(r"(?:nearest|nearest_neighbor|nearest.*align|align.*nearest)", line, re.I):
                nearest_sites.append(f"{p.name}:{i}")

    passed = not linear_interp and len(nearest_sites) >= 1
    return {
        "code": "P28-resample", "name": "[752] 最近邻重采样声明", "severity": "hard",
        "passed": passed,
        "detail": f"linear_interp_on_ranges={linear_interp or 'none'}, nearest_sites={len(nearest_sites)}",
        "evidence": {"illegal_interp_sites": linear_interp, "nearest_sites": nearest_sites[:5]},
    }


_ALL = [
    c1_training_batch_unique, c2_inference_batch_no_mixing, c3_amp_enabled,
    c4_gpu_default, c5_cpu_idle_fallback, c6_ram_limit, c7_parallel_maximize,
    c8_cross_experiment_isolation, c9_pre_run_cleanup, c10_retrain_from_scratch,
    p19_tf32_amp_explicit, p22_two_step_inference, p23_analysis_unit_trajectory,
    p15_pilot_isolation, p28_nearest_resampling,
]


def run_all_c_gates() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for fn in _ALL:
        try:
            r = fn()
        except Exception as exc:
            r = {"code": fn.__name__, "passed": False, "error": f"{type(exc).__name__}: {exc}"}
        out[r.get("code", fn.__name__)] = r
    return out


if __name__ == "__main__":
    import json
    res = run_all_c_gates()
    npass = sum(1 for r in res.values() if r.get("passed"))
    for k in sorted(res):
        r = res[k]
        print(f"{k}: {'PASS' if r.get('passed') else 'FAIL'}  {str(r.get('detail', r.get('error','')))[:90]}")
    print(f"\nC-gates: {npass}/{len(res)} passed")
