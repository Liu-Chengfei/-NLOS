# Script to add p198 function to clause_gates.py
with open('src/liquidloc/analysis/clause_gates.py', 'r', encoding='utf-8') as f:
    content = f.read()

p198_func = '''
def p198_trajectory_count_stats_check() -> dict[str, Any]:
    """[198] 打印各组合实际轨迹数量与 ρ/μ/σ，误差超阈值重生成

    手册原文：
    [198] 打印各组合实际轨迹数量与 ρ/μ/σ（误差超 10% 或任一组合比例偏差超 5% 重生成）。
    """
    cfg_path = PROJECT_ROOT / "configs" / "protocol" / "clause198_trajectory_count_stats.yaml"
    manual_path = PROJECT_ROOT / "docs" / "异步实验_执行手册_终版v15_总纲条款版.md"
    rule_path = PROJECT_ROOT / "rules" / "条款 198_打印轨迹数量rho_mu_sigma.md"
    checks: dict[str, bool | str] = {}
    detail_parts: list[str] = []

    # VS-1: 配置文件存在
    cfg_exists = cfg_path.is_file()
    if not cfg_exists:
        return {"code": "P198-trajectory-count-stats", "name": "[198] 打印轨迹数与rho/mu/sigma误差阈值重生成",
                "severity": "hard", "passed": False,
                "detail": "配置文件 clause198_trajectory_count_stats.yaml 不存在",
                "evidence": {"cfg_path": str(cfg_path.relative_to(PROJECT_ROOT)), "exists": False}}
    checks["vs1_cfg_exists"] = True
    detail_parts.append(f"VS-1 配置文件存在=True, 路径={cfg_path.relative_to(PROJECT_ROOT)}")

    # 解析 YAML
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"code": "P198-trajectory-count-stats", "name": "[198] 打印轨迹数与rho/mu/sigma误差阈值重生成",
                "severity": "hard", "passed": False,
                "detail": f"YAML 解析失败: {exc}",
                "evidence": {"cfg_path": str(cfg_path.relative_to(PROJECT_ROOT)), "parse_error": str(exc)}}

    if not isinstance(cfg, dict):
        return {"code": "P198-trajectory-count-stats", "name": "[198] 打印轨迹数与rho/mu/sigma误差阈值重生成",
                "severity": "hard", "passed": False,
                "detail": "配置文件根节点必须是 dict",
                "evidence": {"cfg_path": str(cfg_path.relative_to(PROJECT_ROOT))}}

    # VS-2: clause_id == "198"
    clause_id = str(cfg.get("clause_id", ""))
    id_ok = clause_id == "198"
    checks["vs2_clause_id"] = id_ok
    detail_parts.append(f"VS-2 clause_id={clause_id!r}, 期望='198', 匹配={id_ok}")

    # VS-3: severity == "hard"
    severity = str(cfg.get("severity", ""))
    sev_ok = severity == "hard"
    checks["vs3_severity_is_hard"] = sev_ok
    detail_parts.append(f"VS-3 severity={severity!r}, 期望=hard, 匹配={sev_ok}")

    # VS-4: four_combinations 包含全部四个组合 A2N2/A2N3/A3N2/A3N3
    four_combos = cfg.get("four_combinations", []) or []
    four_combos_strs = [str(c).strip() for c in four_combos]
    required_combos = {"A2N2", "A2N3", "A3N2", "A3N3"}
    found_combos = set(four_combos_strs)
    combos_ok = required_combos.issubset(found_combos)
    checks["vs4_four_combinations_complete"] = combos_ok
    detail_parts.append(
        f"VS-4 four_combinations 含全部四组合 A2N2/A2N3/A3N2/A3N3={combos_ok}, "
        f"找到={found_combos}"
    )

    # VS-5: stat_metrics 含 rho/mu/sigma 三项
    stat_metrics = cfg.get("stat_metrics", {}) or {}
    stat_keys = list(stat_metrics.keys())
    has_rho = "rho" in stat_keys
    has_mu = "mu" in stat_keys
    has_sigma = "sigma" in stat_keys
    stats_ok = has_rho and has_mu and has_sigma
    checks["vs5_stat_metrics_rho_mu_sigma_present"] = stats_ok
    detail_parts.append(f"VS-5 stat_metrics 含 rho/mu/sigma={stats_ok}, 找到={stat_keys}")

    # VS-6: 误差 10% 阈值明确存在
    error_threshold = cfg.get("error_threshold", {}) or {}
    e10 = error_threshold.get("error_exceeds_10_percent", {}) or {}
    e10_text = str(e10)
    has_10pct = "10" in e10_text or "10%" in e10_text
    checks["vs6_error_exceeds_10_percent_defined"] = has_10pct
    detail_parts.append(f"VS-6 误差超10%阈值明确={has_10pct}")

    # VS-7: 比例偏差 5% 阈值明确存在
    e5pct = error_threshold.get("proportion_deviation_exceeds_5_percent", {}) or {}
    e5pct_text = str(e5pct)
    has_5pct = "5" in e5pct_text or "5%" in e5pct_text
    checks["vs7_proportion_deviation_5_percent_defined"] = has_5pct
    detail_parts.append(f"VS-7 比例偏差超5%阈值明确={has_5pct}")

    # VS-8: disposal_sequence 含重生成步骤
    disposal_seq = cfg.get("disposal_sequence", []) or []
    step_names = [str(s.get("name", "")) for s in disposal_seq]
    step_actions = [s.get("actions", []) or [] for s in disposal_seq]
    all_step_text = " ".join([str(a) for s in step_actions for a in s]) + " " + " ".join(step_names)
    has_regenerate = "重生成" in all_step_text
    checks["vs8_disposal_sequence_has_regenerate"] = has_regenerate
    detail_parts.append(f"VS-8 disposal_sequence含重生成步骤={has_regenerate}, steps={step_names}")

    # VS-9: forbidden 含跳过打印
    boundaries = cfg.get("boundaries", {}) or {}
    forbidden = boundaries.get("forbidden", []) or []
    forbidden_strs = [str(f) for f in forbidden]
    forbidden_full = " ".join(forbidden_strs)
    has_forbidden_skip = "跳过" in forbidden_full and ("打印" in forbidden_full or "统计量" in forbidden_full)
    checks["vs9_forbidden_has_skip_print"] = has_forbidden_skip
    detail_parts.append(f"VS-9 forbidden含'跳过统计量打印'={has_forbidden_skip}")

    # VS-10: required 含必须打印和必须重生成
    required_list = boundaries.get("required", []) or []
    required_strs = [str(r) for r in required_list]
    required_full = " ".join(required_strs)
    has_required_print = "必须打印" in required_full
    has_required_regenerate = "必须重生成" in required_full
    checks["vs10_required_has_print_and_regenerate"] = has_required_print and has_required_regenerate
    detail_parts.append(
        f"VS-10 required含'必须打印'={has_required_print}, 含'必须重生成'={has_required_regenerate}"
    )

    # VS-11: related_clauses 覆盖
    related = cfg.get("related_clauses", []) or []
    related_strs = [str(r) for r in related]
    required_related = {"192", "194", "195", "196", "197", "200", "240", "273", "276", "321", "414"}
    found_related = required_related.intersection(set(related_strs))
    related_ok = found_related == required_related
    checks["vs11_related_clauses_complete"] = related_ok
    detail_parts.append(
        f"VS-11 related_clauses 完整={related_ok}, 找到={found_related}, 期望={required_related}"
    )

    # VS-12: manual_ref 指向正确手册文件
    manual_ref = str(cfg.get("manual_ref", ""))
    manual_ref_ok = "异步实验_执行手册_终版v15_总纲条款版.md" in manual_ref
    checks["vs12_manual_ref_correct"] = manual_ref_ok
    detail_parts.append(f"VS-12 manual_ref={manual_ref!r}, 指向正确={manual_ref_ok}")

    # VS-13: 规则文档存在且内容有效
    rule_exists = rule_path.is_file()
    checks["vs13_rule_doc_exists"] = rule_exists
    detail_parts.append(f"VS-13 规则文档存在={rule_exists}, 路径={rule_path.relative_to(PROJECT_ROOT)}")

    rule_content_ok = False
    if rule_exists:
        rule_text = rule_path.read_text(encoding="utf-8")
        has_original = "**[198]**" in rule_text or "[198]" in rule_text
        has_meaning = "含义" in rule_text or "核心原则" in rule_text or "误差" in rule_text
        has_criteria = "验收" in rule_text or "标准" in rule_text or "阈值" in rule_text
        has_10pct_in_rule = "10%" in rule_text
        has_5pct_in_rule = "5%" in rule_text
        has_regenerate_action = "重生成" in rule_text
        has_four_combos = "A2N2" in rule_text or "A2N3" in rule_text or "A3N2" in rule_text
        has_stats = "rho" in rule_text.lower() or "μ" in rule_text or "mu" in rule_text
        rule_content_ok = (has_original and has_meaning and has_criteria and
                          has_10pct_in_rule and has_5pct_in_rule and has_regenerate_action and
                          has_four_combos and has_stats)
    checks["vs13_rule_doc_content_valid"] = rule_content_ok
    detail_parts.append(f"VS-13 规则文档内容有效={rule_content_ok}")

    # VS-14: 手册真源文件包含 [198]
    manual_exists = manual_path.is_file()
    has_198_in_manual = False
    if manual_exists:
        try:
            manual_content = manual_path.read_text(encoding="utf-8")
            has_198_in_manual = "**[198]**" in manual_content or "[198]" in manual_content
        except Exception:
            pass
    checks["vs14_manual_has_clause_198"] = has_198_in_manual
    checks["vs14_manual_file_exists"] = manual_exists
    detail_parts.append(f"VS-14 手册真源文件存在={manual_exists}, 包含[198]={has_198_in_manual}")

    all_passed = all(v is True for v in checks.values())
    return {
        "code": "P198-trajectory-count-stats",
        "name": "[198] 打印各组合实际轨迹数量与ρ/μ/σ，误差超阈值重生成",
        "severity": "hard",
        "passed": all_passed,
        "detail": "; ".join(detail_parts),
        "evidence": {
            "cfg_path": str(cfg_path.relative_to(PROJECT_ROOT)),
            "rule_path": str(rule_path.relative_to(PROJECT_ROOT)) if rule_exists else None,
            "manual_ref": str(manual_path.relative_to(PROJECT_ROOT)),
            "checks": {k: (str(v) if not isinstance(v, bool) else v) for k, v in checks.items()}
        }
    }


'''

marker = '\n_ALL = ['
idx = content.index(marker)
new_content = content[:idx] + p198_func + '\n' + content[idx:]

new_content = new_content.replace(
    '    p197_four_combination_distribution_check,\n]',
    '    p197_four_combination_distribution_check,\n    p198_trajectory_count_stats_check,\n]'
)

with open('src/liquidloc/analysis/clause_gates.py', 'w', encoding='utf-8') as f:
    f.write(new_content)
print('p198 added successfully')