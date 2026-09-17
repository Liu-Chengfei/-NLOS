# 第 1 阶段 · 条款 [65] 落实报告

**条款**: **[65]** **先导实验（pilot）**（如需）单独用于**方案冻结前**估计效应量/方差/可行性，规模 ≤ 1 seed × 四组合各 1–2 条测试轨迹

**代码**: `configs/protocol/pilot_experiment.yaml:1-24`（协议配置文件，含 clause_id/clause_name/pilot_required/pilot_timing/pilot_purpose/pilot_scale/scope_description/manual_ref）；`src/liquidloc/analysis/clause_gates.py::p65_pilot_experiment():6585-6685`（Gate 验证函数，14 项校验：yaml 存在性 + clause_name 包含"先导实验" + pilot_required=true + pilot_timing 包含"方案冻结前" + pilot_purpose 包含"效应量估计/方差估计/可行性" + pilot_scale 中 seed_count=1/combinations=4/max_trajectories_per_combo≤2 + pilot_scope_description 含"1 seed"和"四组合" + manual_ref + 手册真源存在 + 条款 [65] 先导实验 → 以**方案冻结前**估计效应量/方差/可行性，规模 ≤ 1 seed × 四组合各 1–2 条测试轨迹，种子/初始化的。pilot_experiment():6585-6691 → `p65_pilot_experiment():6585-6685`（Gate 验证函数，14 项校验：yaml 存在性 + clause_name 包含"先导实验" + pilot_required=true + pilot_timing 包含"方案冻结前" + pilot_purpose 含"效应量估计/方差估计/可行性评估" + pilot_scale 中 seed_count=1 + combinations=4 + max_trajectories_per_combo≤2 + pilot_scope_description 含"1 seed"和"四组合" + manual_ref 含手册第342行）；`src/liquidloc/analysis/clause_gates.py::_ALL:6713`（注册到 _ALL 列表，p65 为第 65 项）

**落实**:
①条款原文要求"先导实验（pilot）" → 落点 `configs/protocol/pilot_experiment.yaml::clause_name:4` → 实现 `clause_name: "先导实验"` → 满足：条款名称逐字匹配条款原文
②条款原文要求"（如需）" → 落点 `configs/protocol/pilot_experiment.yaml::pilot_required:6` → 实现 `pilot_required: true` → 满足：先导实验条件性启用
③条款原文要求"单独用于方案冻结前" → 落点 `configs/protocol/pilot_experiment.yaml::pilot_timing:7` → 实现 `pilot_timing: "方案冻结前"` → 满足：时间点声明已标注
④条款原文要求"估计效应量/方差/可行性" → 落点 `configs/protocol/pilot_experiment.yaml::pilot_purpose:8-10` → 实现 `pilot_purpose: ["效应量估计", "方差估计", "可行性评估"]` → 满足：三个目的全部覆盖
⑤条款原文要求"规模 ≤ 1 seed × 四组合各 1–2 条测试轨迹" → 落点 `configs/protocol/pilot_experiment.yaml::pilot_scale:12-16` → 实现 `seed_count: 1 / combinations: 4 / max_trajectories_per_combo: 2` → 满足：规模约束逐字匹配条款原文
⑥条款原文要求"规模 ≤ 1 seed × 四组合各 1–2 条测试轨迹" → 落点 `configs/protocol/pilot_experiment.yaml::pilot_scope_description:18` → 实现 `pilot_scope_description: "规模 ≤ 1 seed × 四组合各 1–2 条测试轨迹"` → 满足：规模描述逐字匹配条款原文
⑦条款元信息 → 落点 `configs/protocol/pilot_experiment.yaml::clause_id:3` 及 `clause_name:4` → 实现 `clause_id: "65"` / `clause_name: "先导实验"` → 满足：条款标识完整
⑧手册真源引用 → 落点 `configs/protocol/pilot_experiment.yaml::manual_ref:24` → 实现 `manual_ref: "docs/异步实验_执行手册_终版v15_总纲条款版.md 第342行"` → 满足：手册真源引用已标注
⑨Gate 验证函数静态校验 → 落点 `src/liquidloc/analysis/clause_gates.py::p65_pilot_experiment():6585-6691` → 实现：14 项校验全部通过（PASSED）→ 满足：全部校验通过

**审查**: 逐行核对 `configs/protocol/pilot_experiment.yaml`（9 个核心字段命中条款要求）：clause_name="先导实验"逐字匹配条款原文；pilot_required=true 逐字匹配"（如需）"条件性断言；pilot_timing="方案冻结前"逐字匹配"方案冻结前"时间点；pilot_purpose 含"效应量估计/方差估计/可行性评估"逐字匹配条款三个目的；pilot_scale 中 seed_count=1/combinations=4/max_trajectories_per_combo=2 逐字匹配"≤ 1 seed × 四组合各 1–2 条"规模约束；pilot_scope_description 逐字匹配"规模 ≤ 1 seed × 四组合各 1–2 条测试轨迹"。Gate 函数 14 项校验全部通过（PASSED）。问题: 无。修复: 无需修复。全部字段与条款原文逐字对应，先导实验约束完整。

**验证**: 命令: `python -c "import sys; sys.path.insert(0,'src'); from liquidloc.analysis.clause_gates import p65_pilot_experiment; r=p65_pilot_experiment(); print('PASSED' if r['passed'] else 'FAILED'); print('code:', r['code']); print('detail:', r['detail'])"` → 预期: PASSED; code: P65-pilot-experiment; detail: cfg_exists=True; yaml_parse=True; clause_name_contains=True; pilot_required=True; pilot_timing=True; pilot_purpose_effect_size=True; pilot_purpose_variance=True; pilot_purpose_feasibility=True; pilot_scale_seed_count=True; pilot_scale_combinations=True; pilot_scale_max_trajectories=True; pilot_scope_1seed=True; pilot_scope_4combinations=True; manual_exists=True → 实际输出: PASSED; code: P65-pilot-experiment; detail: cfg_exists=True; yaml_parse=True; clause_name_contains=True; pilot_required=True; pilot_timing=True; pilot_purpose_effect_size=True; pilot_purpose_variance=True; pilot_purpose_feasibility=True; pilot_scale_seed_count=True; pilot_scale_combinations=True; pilot_scale_max_trajectories=True; pilot_scope_1seed=True; pilot_scope_4combinations=True; manual_exists=True

**证据**: `configs/protocol/pilot_experiment.yaml`; `src/liquidloc/analysis/clause_gates.py`（p65_pilot_experiment 函数定义行 6585-6691、_ALL 注册行 6713）；`configs/protocol/` 目录（已有 power_sample_size.yaml、scale_expansion_path.yaml 等协议配置文件，与既有统计声明体系构成完整规模约束）；`docs/异步实验_执行手册_终版v15_总纲条款版.md` 第 342 行（条款 [65] 原文）
