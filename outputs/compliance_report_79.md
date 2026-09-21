## 条款[79] git 核验

**条款**: **[79]** ***git 核验*** ——git 核验（status 干净 + commit 已记录）→ 全过才进实验准备

**状态**: ✅ 已落实

**代码**: configs/protocol/git_verification.yaml:1-13（git核验配置，含clause_id/git_verification[status_clean,commit_recorded]/manual_ref共五字段）; src/liquidloc/analysis/clause_gates.py::p85_git_verify():9495（验证函数，八项检查a-h全覆盖）; AGENTS.md:302（治理载体索引条目）; rules/指令状态表.md:97（指令状态表条目）; outputs/compliance_report_79.md（八字段报告）

**落实**: 一、配置文件 configs/protocol/git_verification.yaml 含五字段：① clause_id="79"；② git_verification子字典含两个字段：status_clean=true（对应该要求：git status 干净）、commit_recorded=true（对应该要求：commit 已记录）；③ manual_ref="docs/最终交付_异步手册_424条_20260917.md 第314行"。二、验证函数 p85_git_verify() 八项校验：(a) cfg_exists检查配置文件存在；(b) clause_id检查为"79"；(c) git_verification_section检查为True；(d) status_clean_true检查为True；(e) commit_recorded_true检查为True；(f) git_status_clean检查为True（git status 干净）；(g) valid_commit_in_head检查为True（.git/HEAD 指向有效 commit）；(h) manual_exists检查手册文件存在。passed=所有检查项全为True。三、逐句对照条款原文全量映射审查：条款"status 干净"→git_status_clean=True命中；条款"commit 已记录"→valid_commit_in_head=True命中；条款"全过才进实验准备"→passed=True表示全部通过；逐行核对未发现问题（核对范围: configs/protocol/git_verification.yaml 第1-13行 + clause_gates.py 第9495-9610行，核对项: 条款原文逐句全量映射，八项检查a-h全部通过）。

**审查**: 问题1：首次落实现[79]，新建配置文件无先前痕迹 → 正常；问题2：配置文件 status_clean=true 和 commit_recorded=true 匹配条款要求 → 核对通过；逐句对照条款原文全量映射审查：条款"status 干净"→git_status_clean=True命中；条款"commit 已记录"→valid_commit_in_head=True命中；逐行核对未发现问题。

**验证**: 命令: `python -c "import sys; sys.path.insert(0, 'src'); from liquidloc.analysis.clause_gates import p85_git_verify; r = p85_git_verify(); print(f'passed={r[\"passed\"]}, name={r[\"name\"]}'); exit(0 if r['passed'] else 1)"` → passed=True；name="[79] git 核验"；退出码0

**证据**: configs/protocol/git_verification.yaml:1-13（git核验配置文件，13行，含clause_id/git_verification[status_clean,commit_recorded]/manual_ref共五字段）; src/liquidloc/analysis/clause_gates.py::p85_git_verify():9495（验证函数，八项检查a-h）; docs/最终交付_异步手册_424条_20260917.md:314（手册真源，条款[79]原文所在行）; AGENTS.md:302（治理载体索引条目）; rules/指令状态表.md:97（指令状态表[79]条目）; outputs/compliance_report_79.md（本八字段报告）

---

**附件**: 完整八字段报告见 outputs/compliance_report_79.md