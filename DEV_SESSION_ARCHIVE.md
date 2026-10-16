# 开发会话档案（2026-07-15 ~ 2026-08-18）

> 本文档抢救性留档项目主开发会话中**未进入 README / TODO.md / TODO_DSH.md 的信息**。
> 会话原始转录（约 90.9 MB）因 Claude Code 默认 30 天保留期已被自动清理，
> 以下内容整理自留存的会话摘要缓存（226 条任务快照、10 个子 agent 记录）、
> 项目记忆文件与 prompt 历史。已与 git log（至 e725998）交叉核对。

## 会话概况

- 时间跨度：2026-07-15 ~ 2026-08-18，共 191 条用户输入，单一长会话
- 版本轨迹：从"改写 README、实现最小 agent 循环"起步，至 0.2.0（844b81e）及后续 tool-calling 修复
- 代码量：约 +21539 / -4740 行（最后一次统计）
- 模型用量（整个会话累计）：输入约 4.23 亿 tokens、输出约 404 万 tokens、缓存读取约 8.01 亿 tokens；API 花费约 **$584.90**
- 会话中派出的 10 个子 agent：Explore tau3-bench / AgentDojo / BFCL v3 / AgentBoard / tool-calling pipeline / benchmark adapter flow / tool error feedback flow、Resume second explore、Probe gateway truncation cause、Map deepseek-harness architecture

## 基准评测中间结论（其他文档未记录）

### AgentDojo EchoBrain 冒烟基线（2026-07 下旬）

适配器验证可运行时的 EchoBrain 基线：**banking 31.25% utility，其余 suite 0%**。
  当时内网 LLM 服务返回 502，真实评测需待其恢复。
（后续 LLM 全量结果为 93.81% utility / 100% security，见 `benchmarks/agentdojo_failure_analysis.md`。）

### BFCL v3 single-turn（2026-07-27）

| 类别 | 结果 |
|---|---|
| simple_python | 56.25% |
| multiple | 38.00% |
| irrelevance | 65.00% |
| parallel | **0.00%** |

**parallel 为 0 的根因**：当时的 SwayBot server 每轮只输出一个 tool call。
**multi_turn_base 未完成**：200 个 case × 每个最多 20 步太慢，且同样依赖多 tool call 支持。
（分数明细见 `score/*.csv`，但上述两条结论性判断只存在于会话记录中。）

### tau3-bench 进度（截至会话结束）

- 已跑：mock 四域、airline/retail/banking_knowledge 样本（结果在 `tau3_results/`，最后一批 2026-08-12）
- **全量（非 mock）评测始终未跑**——会话结束时仍是 pending
- telecom 无数字型 task-id，格式易错，详见 `benchmarks/tau3bench/TASK_IDS.md`

## 会话结束时仍开放的事项

快照自会话最后状态，已剔除：① 后续 commit 已完成的（如 prose 工具标签恢复=3152dd3、
KB 综合修复=1bdb6cd、非收敛 guard=3a77d25/191b705、calculate-tool 提示=c5e53b7）；
② TODO.md / TODO_DSH.md 已收录的（流式 CLI 接线、CodeAgent 沙箱、语义检索、
外部搜索、反例寻找、PluginProvider/McpProvider、TODO_DSH #4/#5 等）。

### 基准相关（优先）

1. **banking_knowledge user_simulator 侧 tool-calling 失败**未解决（注意是模拟器侧，不是 agent 侧）
2. **retail 工具调用归一化**：`exchange` 工具参数归一化 + numeric-ID schema 问题，两项均未修
3. **banking_knowledge 检索质量**：e725998（KB_search k 提到 20）只解决了 rank-14+ 文档取不到，整体检索质量仍待提升
4. **tau3 adapter behavior guidance** 未更新（对 agent 行为的域级引导）
5. **telecom 参数类型修复验证**（TODO_DSH #4 关联项，未见验证记录）
6. **BFCL multi_turn_base** 与 parallel 多 tool call 输出（见上）
7. **AgentDojo prompt-injection 防御**：fc0dbb5 只加了 guidance，硬性防御未实现
8. **tau3-bench 全量评测**收尾

### 架构级（会话中提出、未排期）

- benchmark adapter 层的**原生函数调用改造**（原生 FC 大版本，区别于已完成的 prompt 修复系列）
- MessageBus 与 EventBus 合并
- 异步/线程边界问题、storage 后端并发问题
- 按 README/SOUL 目标设计实验验证 agent 自进化效果（"Design experiments to validate README/SOUL goals"，从未开始）

## 历史备注

- 曾存在后被移除的模块（结构清理，勿直接恢复原实现，按 `ToolProvider` 协议重接）：
  `plugin_loader.py`、`mcp_client.py`、`subagent.py`、`automation.py`、Python SDK
- 曾规划"分析 nanobot 与 swaybot 对比"，实际以 TODO.md"从 nanobot 借鉴"系列落地
- 会话中早期大量 pending 条目（Create memory module 等）为列表残留，实际均已完成并对应 TODO.md 中已勾选项
