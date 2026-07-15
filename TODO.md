# SwayBot TODO

按优先级排序的后续改进项。完成一项后更新状态，并持续分析下一步最值得做的事。

## P0 — 立刻提升稳定性

### [x] 给 LLM 大脑发送工具签名描述
- **问题**：当前系统提示只列出工具名，LLM 不知道参数，容易传错（如 `done(result=...)`）。
- **方案**：把 `ToolRegistry.describe()` 输出整理进 system prompt，包含参数名和简短说明。
- **文件**：`swaybot/llm_brain.py`, `swaybot/tools.py`, `swaybot/agent.py`
- **验收**：LLM 调用准确率提升，fallback 减少；测试覆盖 describe 拼接。
- **状态**：已完成（2026-07-15）。`Agent` 把 `tool_descriptions` 注入 perception；`LLMBrain` 优先使用签名描述生成 system prompt。

### [x] 增加 `@tool` 装饰器与 JSON Schema 描述
- **问题**：手写工具 schema 容易错，且无法从函数签名/docstring 自动推导。
- **方案**：参考 smolagents，新增 `@tool` 装饰器、`Tool` 数据类、`ToolRegistry.schemas()`，自动从类型注解和 docstring 生成 JSON schema。
- **文件**：`swaybot/tools.py`, `swaybot/agent.py`, `swaybot/llm_brain.py`, `tests/test_tools.py`, `pyproject.toml`
- **验收**：`@tool` 能推断参数类型、默认值和描述；LLM system prompt 包含 JSON schema；pytest 只收集 `tests/` 目录。
- **状态**：已完成（2026-07-15）。46 个测试全部通过。

### [x] Phase 2：YAML/Jinja 提示词模板
- **问题**：系统提示和用户提示硬编码在 Python 中，调试和迭代不直观。
- **方案**：把 prompt 拆成 `swaybot/prompts/*.j2`，用 Jinja2 渲染；`jinja2` 作为 `[llm]` optional dependency；`pyproject.toml` 打包模板文件。
- **文件**：`swaybot/prompts.py`, `swaybot/prompts/system.j2`, `swaybot/prompts/user.j2`, `swaybot/llm_brain.py`, `tests/test_prompts.py`, `pyproject.toml`
- **验收**：`LLMBrain` 使用模板渲染 system/user prompt；测试覆盖变量渲染、条件循环、默认模板存在性；52 个测试全部通过。
- **状态**：已完成（2026-07-15）。

## P1 — 记忆与可维护性

### [ ] 短期记忆自动归档/遗忘
- **问题**：`short_term` 经验永久写入 `~/.swaybot/memory.json`，文件无限增长。
- **方案**：`--reflect` 后将已归档的短期记忆清理，或只保留最近 N 条。
- **文件**：`swaybot/agent.py`, `swaybot/memory.py`
- **验收**：长期运行后内存文件大小可控；测试验证归档后 short_term 数量下降。

### [ ] 改进记忆检索（相关性 > 标签）
- **问题**：`_memory_context` 只按 `tag` 匹配，跨任务相关知识无法复用。
- **方案**：基于关键词重叠或轻量嵌入做相似度检索。
- **文件**：`swaybot/memory.py`
- **验收**：不同任务但语义相关的记忆能被召回。

## P2 — 自我进化

### [ ] 反思结果反馈到行为
- **问题**：反思只生成 `theory` 记忆，不改变 Agent 行为。
- **方案**：根据失败/意外经验，动态调整 system prompt、工具偏好或生成新工具。
- **文件**：`swaybot/agent.py`, `swaybot/reflection.py`, 可能新增 `swaybot/self_improve.py`
- **验收**：同一错误在后续运行中出现频率下降。

### [ ] API 调用重试与退避
- **问题**：LLM 调用失败直接 fallback，没有重试。
- **方案**：对临时网络错误做有限次指数退避重试。
- **文件**：`swaybot/llm_brain.py`
- **验收**：模拟超时/断网时能够重试并最终成功或优雅失败。

## P3 — 体验增强

### [ ] 空闲时主动探索
- **问题**：无人对话时 Agent 什么都不做（SOUL.md 提到应主动探索）。
- **方案**：提供一个 `explore` 模式，让 Agent 自己生成假设、设计实验、记录结果。
- **文件**：新增 `swaybot/explorer.py`, `swaybot/cli.py`
- **验收**：运行 `python -m swaybot --explore` 能自主产生任务并执行。

### [ ] 流式响应支持
- **问题**：LLM 必须等完整响应返回。
- **方案**：支持 SSE/流式输出，便于观察思考过程。
- **文件**：`swaybot/llm_brain.py`
- **验收**：思考过程可实时打印。

---

## 当前聚焦

下一步执行 **Phase 3：Typed Memory Steps**（来自 smolagents 学习计划）——把 `Memory` 重构为带 `to_messages()` 的 step 类型，为后续 PlanningStep 和上下文管理做准备。

同时继续推进 **P1：短期记忆自动归档/遗忘**。
