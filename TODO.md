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

### [x] Phase 3：Typed Memory Steps
- **问题**：`Memory` 是扁平结构，运行时步骤和长期知识混在一起，难以渲染成 LLM 上下文。
- **方案**：引入 `MemoryStep` 基类与 `TaskStep`、`ActionStep`、`ObservationStep`、`ReflectionStep`，每个子类实现 `to_messages()`；`MemoryStore` 通过 `step_kind` 反序列化；`Agent` 用步骤构建 messages 并注入 `LLMBrain`。
- **文件**：`swaybot/memory.py`, `swaybot/agent.py`, `swaybot/reflection.py`, `swaybot/llm_brain.py`, `tests/test_memory.py`, `tests/test_agent.py`, `tests/test_llm_brain.py`
- **验收**：运行时存储为类型化步骤；LLM 收到 system + 相关长期记忆 + 短期步骤 messages；步骤可持久化并原样加载。

### [x] PlanningStep 支持
- **问题**：Agent 直接开始执行，没有显式规划阶段。
- **方案**：新增 `--plan` 标志；`Agent.run` 在任务开始时调用 `_create_plan()`；`EchoBrain` 与 `LLMBrain` 都支持 `planning` 模式并返回 `{"name": "plan", "args": {"steps": [...]}}`；新增 `swaybot/prompts/plan.j2`。
- **文件**：`swaybot/agent.py`, `swaybot/brain.py`, `swaybot/llm_brain.py`, `swaybot/prompts/plan.j2`, `swaybot/cli.py`, `tests/test_agent.py`, `tests/test_llm_brain.py`, `tests/test_cli.py`
- **验收**：`--plan` 产生 `PlanningStep`；`LLMBrain` 能解析 JSON 数组或 `{"steps": [...]}`；63 个测试全部通过。
- **状态**：已完成（2026-07-15）。

## P1 — 记忆与可维护性

### [x] 短期记忆自动归档/遗忘
- **问题**：`short_term` 经验永久写入 `~/.swaybot/memory.json`，文件无限增长。
- **方案**：`--reflect` 后将已归档的短期记忆清理；`MemoryStore` 新增 `prune(scope, tag, keep_last)`。
- **文件**：`swaybot/memory.py`, `swaybot/agent.py`, `tests/test_memory.py`
- **验收**：`--reflect` 后同一任务的 short_term 步骤被清除，只保留长期反思；未启用 `--reflect` 时短期步骤保留；58 个测试全部通过。
- **状态**：已完成（2026-07-15）。

### [x] 改进记忆检索（相关性 > 标签）
- **问题**：`_memory_context` 只按 `tag` 匹配，跨任务相关知识无法复用。
- **方案**：`MemoryStore` 新增 `query_relevant(query, scope, limit)`，基于内容关键词重叠（Jaccard 风格）+ 标签匹配 boost 排序，stdlib 无外部依赖。
- **文件**：`swaybot/memory.py`, `swaybot/agent.py`, `tests/test_memory.py`, `tests/test_agent.py`
- **验收**：不同任务但语义相关的记忆能被召回；Agent 的 memory_context 已改用 `query_relevant`；64 个测试全部通过。
- **状态**：已完成（2026-07-15）。

## P2 — 自我进化

### [x] 反思结果反馈到行为
- **问题**：反思只生成 `theory` 记忆，不改变 Agent 行为。
- **方案**：`Agent` 在每次决策前从长期记忆中检索高可信度 `ReflectionStep`，将其内容作为 `behavior_guidance` 注入 system prompt，影响后续工具选择。
- **文件**：`swaybot/agent.py`, `swaybot/prompts/system.j2`, `swaybot/llm_brain.py`
- **验收**：高可信度反思内容会出现在 system prompt 的 "Lessons learned" 区块；低可信度或无关内容被过滤。
- **状态**：已完成（2026-07-15）。67 个测试全部通过。

### [x] API 调用重试与退避
- **问题**：LLM 调用失败直接 fallback，没有重试。
- **方案**：`LLMBrain._chat()` 对临时错误做最多 `max_retries` 次指数退避重试（`backoff * 2^attempt`），全部失败后返回 `None` 并由上层 fallback。
- **文件**：`swaybot/llm_brain.py`
- **验收**：模拟连续失败后能重试并成功；全部重试耗尽后优雅 fallback。
- **状态**：已完成（2026-07-15）。

## P3 — 体验增强

### [x] 空闲时主动探索
- **问题**：无人对话时 Agent 什么都不做（SOUL.md 提到应主动探索）。
- **方案**：新增 `Explorer` 模块与 `--explore` CLI 标志。`Explorer` 让 brain 自己生成假设/任务（LLM 或 EchoBrain 的默认题库），然后调用 `Agent.run()` 执行并记录反思；长期记忆中会留下新的 reflection。
- **文件**：新增 `swaybot/explorer.py`、`swaybot/prompts/explore.j2`；修改 `swaybot/brain.py`、`swaybot/llm_brain.py`、`swaybot/cli.py`、`tests/test_explorer.py`、`tests/test_cli.py`
- **验收**：`python -m swaybot --explore` 能自主产生任务、执行、打印结果，并生成长期 reflection；74 个测试全部通过。
- **状态**：已完成（2026-07-15）。

### [ ] 流式响应支持
- **问题**：LLM 必须等完整响应返回。
- **方案**：支持 SSE/流式输出，便于观察思考过程。
- **文件**：`swaybot/llm_brain.py`
- **验收**：思考过程可实时打印。

## P4 — 借鉴 smolagents 补齐关键能力

### [x] Final answer 机制
- **问题**：当前 Agent 靠 `done` 工具结束任务，小模型常在 max_steps 边界忘记收尾（如 LLM explore 跑了两步 `add` 都没调 `done`）。
- **方案**：新增 `final_answer` 工具；`Environment.observe()` 将其与 `done` 同样视为终止动作；system prompt 明确 instruct 模型在得到答案时调用 `final_answer(answer=...)`。
- **文件**：`swaybot/prompts/system.j2`, `swaybot/environment.py`, `swaybot/tools.py`, `tests/test_agent.py`, `tests/test_tools.py`
- **验收**：调用 `final_answer` 后 Agent 提前结束并保留答案；`done` 保持兼容；78 个测试全部通过。
- **状态**：已完成（2026-07-16）。

### [ ] 工具输入校验
- **问题**：`ToolRegistry.execute()` 直接执行，不检查参数类型、必填项，LLM 传错参数时可能报错或行为异常。
- **方案**：在 `Tool` / `ToolRegistry` 中加入 `validate_arguments()`，根据 JSON schema 检查类型和 required；失败时返回结构化错误让 LLM 重试。
- **文件**：`swaybot/tools.py`, `swaybot/agent.py`, `tests/test_tools.py`
- **验收**：参数错误时返回可理解的错误信息，而不是抛异常退出。

### [ ] 更详细的 ActionStep 与基础监控
- **问题**：`ActionStep` 只记录动作本身，没有原始模型输入、token 用量、耗时，难以复盘和优化。
- **方案**：扩展 `ActionStep`（或新增 `ModelStep`）记录 `model_input_messages`、`raw_output`、`token_usage`、`duration_ms`；在 `LLMBrain` 中采集这些字段。
- **文件**：`swaybot/memory.py`, `swaybot/llm_brain.py`, `tests/test_memory.py`, `tests/test_llm_brain.py`
- **验收**：每个 action 能追溯到原始 LLM 输出和耗时。

### [ ] 多模型 backend 抽象
- **问题**：`LLMBrain` 直接依赖 `openai` 包，换本地模型或其他 API 时需要重写。
- **方案**：提取 `Model` 基类，把 `LLMBrain` 改名为 `OpenAIModel` 或拆出 `OpenAIModel`；后续可添加 `TransformersModel` / `MockModel` 等。
- **文件**：`swaybot/models.py`（新）、`swaybot/llm_brain.py`（重构）、`tests/test_models.py`（新）
- **验收**：`Agent` 接收任意 `Model` 实现，核心逻辑不依赖 `openai`。

### [ ] 可选的 CodeAgent / 本地 Python 沙箱
- **问题**：JSON action 一次只能调用一个工具，复杂多步组合需要多次 LLM 调用。
- **方案**：参考 smolagents `CodeAgent`，让 LLM 生成 Python 代码块，在受限的 `LocalPythonExecutor` 中执行；默认关闭，通过 `--code-agent` 开启。
- **文件**：新增 `swaybot/code_agent.py`、`swaybot/local_python_executor.py`、`swaybot/prompts/code_agent.j2`
- **验收**：开启后 Agent 能用代码一次组合多个工具，且危险操作被沙箱拦截。

## P5 — 长期进化方向

### [ ] 语义检索（embedding）
- **问题**：关键词检索对同义词、改写不敏感。
- **方案**：可选引入 sentence-transformers 或 openai embedding，把记忆内容编码为向量，按余弦相似度召回。
- **文件**：`swaybot/memory.py`
- **验收**：语义相关但用词不同的记忆能被召回。

### [ ] 从记忆矛盾/问题生成探索假设
- **问题**：`Explorer` 目前靠 LLM 或固定题库生成假设，没有利用已有反思中的矛盾和未解问题。
- **方案**：在 `Reflector` 中把 `question` / `contradiction` 类 reflection 作为候选假设来源；`Explorer` 优先选择这些高价值问题去验证。
- **文件**：`swaybot/explorer.py`, `swaybot/reflection.py`
- **验收**：Agent 会主动验证自己之前提出的疑问或矛盾。

---

## 当前聚焦

P3 的流式响应可先放一放。P4 的 Final answer 机制已完成。下一步做 **工具输入校验**，让错误参数可被优雅处理而不是抛异常退出。
