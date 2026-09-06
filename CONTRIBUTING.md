# CONTRIBUTING

本仓库欢迎维护与补全，但请先理解：

- **中性科研骨架**定义“项目是什么”
- **Codex 施工外壳**定义“施工时怎么做”

## 规则来源优先级
1. `docs/authority_sources.md`
2. 相关 `docs/*.md` 与 `configs/base/*.yaml`
3. 根目录 `AGENTS.md`
4. 目标目录 `AGENTS.md`
5. `.agents/skills/*`
6. 目标测试文件

## 贡献者必须遵守
- 不得让 `AGENTS.md` 成为唯一科学规则来源
- 不得在 `scripts/` 复制业务逻辑
- 不得在 `plotting/` 重算指标
- 不得在 `metrics/` 选择案例或重写统计结论
- 不得为了通过局部测试而改变中性科研协议
- 不得把运行生成物打进源码交付包

## 文档归属
- 科学协议：`docs/` + `configs/` + `src/liquidloc/protocol/*`
- Codex 施工规则：`AGENTS.md` + `.agents/skills/*` + `docs/codex/*`
- 历史审计：`docs/history/`

## 提交前最小检查
- 修改协议/配置：至少跑对应协议测试
- 修改数据链：至少跑 `tests/dataio` 与最小 prepare smoke
- 修改编排链：至少跑 `tests/pipelines` 对应测试与最小 smoke
- 修改文档治理：至少检查 `docs/authority_sources.md` 是否仍自洽

## 交付包要求
- 只包含源码、配置、测试、必要文档
- 不包含运行生成图表、日志、缓存、checkpoint、metrics 输出


## 严格施工约束
- 未要求事项一律不做
- 默认最小修改、最小验证、最小交付
- 若需要扩大范围，必须在任务层明确授权
