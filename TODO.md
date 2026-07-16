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

### [x] 工具输入校验
- **问题**：`ToolRegistry.execute()` 直接执行，不检查参数类型、必填项，LLM 传错参数时可能报错或行为异常。
- **方案**：在 `Tool` 中加入 `validate_arguments()`，根据 JSON schema 检查 required 和类型；`ToolRegistry.execute()` 先校验再执行；`Agent.run()` 用 try/except 捕获工具执行错误，把错误信息作为 observation 让 LLM 在下一步纠正。
- **文件**：`swaybot/tools.py`, `swaybot/agent.py`, `tests/test_tools.py`, `tests/test_agent.py`
- **验收**：缺失必填参数或类型错误时返回可理解的错误信息，Agent 不崩溃；81 个测试全部通过。
- **状态**：已完成（2026-07-16）。

### [x] 更详细的 ActionStep 与基础监控
- **问题**：`ActionStep` 只记录动作本身，没有原始模型输入、token 用量、耗时，难以复盘和优化。
- **方案**：扩展 `ActionStep` 记录 `model_input_messages`、`raw_output`、`token_usage`、`duration_ms`；`LLMBrain._chat()` 在每次调用中采集这些字段；`Agent.run()` 通过 `metadata` 字典接收并写入 `ActionStep`。
- **文件**：`swaybot/memory.py`, `swaybot/llm_brain.py`, `swaybot/agent.py`, `swaybot/brain.py`, `tests/test_memory.py`, `tests/test_llm_brain.py`
- **验收**：每个 action 能追溯到原始 LLM 输出、token 用量和耗时；83 个测试全部通过。
- **状态**：已完成（2026-07-16）。

### [x] 运行复盘与 inspect 命令
- **问题**：`ActionStep` 已记录原始输出、token 和耗时，但反思后会清理短期记忆，且 CLI 没有离线查看入口，监控数据无法形成复盘闭环。
- **方案**：新增与 `memory.json` 同级的 `runs.jsonl` 运行日志；`Agent.run()` 在结束时写入任务、假设、final answer、每步 action/result/raw_output/token/duration/error 和反思摘要；新增 `python -m swaybot inspect --last/--task` 离线查看。
- **文件**：`swaybot/run_log.py`, `swaybot/agent.py`, `swaybot/cli.py`, `tests/test_run_log.py`, `tests/test_cli.py`
- **验收**：`--reflect` 清理短期记忆后仍能 inspect；输出包含 raw output、token、耗时和错误；87 个测试全部通过。
- **状态**：已完成（2026-07-16）。

### [x] 验证后的信念更新闭环
- **问题**：`Explorer` 验证了之前的疑问/矛盾后，只生成 `verification` reflection，不会更新原有记忆的置信度，Agent 可能反复验证同一问题。
- **方案**：`Reflector._update_beliefs()` 在 hypothesis 被支持/反驳后，根据 `query_relevant()` 找到相关事实/经验记忆，提升或降低其 `credibility`；同时生成 `belief_update` reflection 进入长期记忆；`Explorer` 挑选候选假设时会跳过已有 `verification` 标签的已验证问题。
- **文件**：`swaybot/reflection.py`, `swaybot/explorer.py`, `tests/test_reflection.py`, `tests/test_explorer.py`
- **验收**：支持的假设让相关记忆置信度上升，反驳的下降；已验证问题不再被 `Explorer` 重复选中；95 个测试全部通过。
- **状态**：已完成（2026-07-16）。

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

### [ ] 语义检索（embedding）【优先级降低】
- **问题**：关键词检索对同义词、改写不敏感。
- **方案**：可选引入 sentence-transformers 或 openai embedding，把记忆内容编码为向量，按余弦相似度召回。
- **文件**：`swaybot/memory.py`
- **验收**：语义相关但用词不同的记忆能被召回。
- **备注**：当前关键词检索已覆盖主要场景，embedding 属于锦上添花；待多模型 backend 抽象和 CodeAgent 等更核心能力稳定后再考虑。

### [ ] 主动搜索外部信息
- **问题**：Agent 只能依赖已有记忆和内置工具，无法主动搜索互联网、本地文件或 API 获取新证据，难以满足 SOUL.md 中“主动探索世界、搜索材料”的要求。
- **方案**：新增通用 `search` 工具族（如 `web_search`、`file_search`、`api_search`）或 MCP/搜索工具接口；`Explorer` 在生成假设时、`Reflector` 在验证假设时均可自动调用。
- **文件**：`swaybot/tools.py`（新增工具）、`swaybot/explorer.py`、`swaybot/reflection.py`
- **验收**：`--explore` 或验证阶段能主动搜索外部信息，并将搜索结果作为 evidence 写入记忆。

### [ ] 主动寻找反例
- **问题**：当前 `find_counterexamples` 只在已有记忆中匹配否定词/惊喜标记，不能主动设计查询或搜索去发现反例。
- **方案**：在 `Reflector` 验证高置信度信念后，触发 `_seek_counterexamples()`：优先使用外部搜索工具（若已实现）查询反例，再回退到现有记忆；`Explorer` 也可生成“寻找 X 的反例”类探索任务。
- **文件**：`swaybot/reflection.py`、`swaybot/explorer.py`
- **验收**：对高置信度命题能主动尝试发现反例，并生成 `contradiction` 或 `verification` 记忆。
- **备注**：可与“主动搜索外部信息”一起实现；当前可先利用 `find_counterexamples` 做轻量版，待搜索工具就绪后再扩展。

### [x] 从记忆矛盾/问题生成探索假设
- **问题**：`Explorer` 目前靠 LLM 或固定题库生成假设，没有利用已有反思中的矛盾和未解问题。
- **方案**：`reflection_to_memory()` 保留 `question` / `contradiction` 等 kind；`Explorer._candidate_hypotheses_from_memory()` 从长期记忆中检索这些条目；生成探索任务时优先把它们作为候选假设传给 LLM 探索提示，EchoBrain 则直接选择第一个候选；无候选时才回退到默认题库或 LLM 自由生成。
- **文件**：`swaybot/explorer.py`, `swaybot/reflection.py`, `swaybot/prompts/explore.j2`, `tests/test_explorer.py`, `tests/test_reflection.py`
- **验收**：长期记忆中存在 question/contradiction 时，`--explore` 会生成对应的验证任务；LLM 探索提示包含候选假设列表；91 个测试全部通过。
- **状态**：已完成（2026-07-16）。

## 从 nanobot 借鉴的长期方向

### [x] 模型 preset 与 fallback 链
- **问题**：单模型失败时无备用；不同任务需要不同模型，目前只能换环境变量或参数。
- **方案**：参考 nanobot `model_runtime.py`，在多模型 backend 抽象上支持 `modelPresets` 与 `fallbackModels`，按名称切换与故障转移。
- **文件**：`swaybot/models.py`
- **验收**：配置中可定义多个 preset 和 fallback；某模型失败时自动切换并继续任务。
- **状态**：已完成（2026-07-16）。

### [x] 统一 ContextBuilder
- **问题**：`Agent` 内部 `_memory_context`、`_behavior_guidance`、`_build_messages` 散落拼装，新增上下文来源时容易改动核心循环。
- **方案**：提取 `ContextBuilder`，统一组装 system prompt、SOUL/identity、长期记忆、行为指导、短期历史、运行时上下文。
- **文件**：新增 `swaybot/context.py`
- **验收**：新增上下文来源只需修改 ContextBuilder，`Agent.run()` 保持不变。
- **状态**：已完成（2026-07-16）。

### [x] AgentHook 生命周期
- **问题**：缺乏迭代级/工具级/运行级回调，难以低侵入地实现日志、指标、审计、流式输出。
- **方案**：参考 nanobot `hook.py`，定义 `AgentHook` 协议与 `CompositeHook`，在 before/after iteration、tool、run 等阶段触发。
- **文件**：新增 `swaybot/hook.py`
- **验收**：可编写独立 hook 记录每次迭代与工具调用，不影响核心逻辑。
- **状态**：已完成（2026-07-16）。

### [x] MCP 动态工具接入
- **问题**：每新增外部能力都要手写 wrapper，扩展成本高。
- **方案**：实现轻量 MCP client（stdio/SSE），让 swaybot 动态发现并加载 MCP server 的工具。
- **文件**：新增 `swaybot/mcp_client.py`、`swaybot/tools.py`
- **验收**：配置 MCP server 后，其工具自动出现在 registry 中并可被调用。
- **状态**：已完成（2026-07-16）。

### [x] WebSearch / WebFetch 工具
- **问题**：Agent 无法联网，难以满足 SOUL.md 主动探索世界、搜索材料的要求。
- **方案**：增加 `web_search` 与 `web_fetch` 工具，支持多 provider 适配与基础 SSRF 防护，供 `Explorer`/`Reflector` 调用。
- **文件**：`swaybot/tools.py`（或新建 `swaybot/tools/web.py`）
- **验收**：验证假设时能搜索网络并引用结果作为 evidence。
- **状态**：已完成（2026-07-17）。

### [x] 工具并发语义
- **问题**：多个工具调用只能串行执行，读-only/安全工具无法并行，效率低。
- **方案**：给 `Tool` 增加 `read_only`、`concurrency_safe`、`exclusive` 元数据；`ToolRegistry.execute_batch` 按语义分组并发执行。
- **文件**：`swaybot/tools/__init__.py`
- **验收**：可并发工具真正并行，exclusive 工具单独顺序执行，测试覆盖分组逻辑。
- **状态**：已完成（2026-07-17）。

### [x] 两阶段记忆（Consolidator + Dream）
- **问题**：`memory.json` 增长后难以审阅、回滚与持续演化，长期知识与原始经历混在一起。
- **方案**：参考 nanobot：短期 `history.jsonl` 自动压缩归档；定期用 LLM 编辑 durable memory 文件（如 `SOUL.md`/`MEMORY.md`）。
- **文件**：`swaybot/memory.py`、`swaybot/reflection.py`
- **验收**：长期记忆可版本化、可回滚、可人工审阅；运行历史与长期知识分离。
- **状态**：已完成（2026-07-17）。

### [x] 子代理与后台自动化
- **问题**：当前单线程顺序执行，空闲时无法并发探索，也缺少定时任务。
- **方案**：后台 `SubagentManager` 支持并行子任务；`Automation` 支持基于时间间隔的后台心跳/定时任务。
- **文件**：新增 `swaybot/subagent.py`、`swaybot/automation.py`
- **验收**：空闲时可并发执行多个探索任务，定时触发心跳/反思。
- **状态**：已完成（2026-07-17）。

### [ ] 目标权限控制（Goal Permission）

### [x] 目标权限控制（Goal Permission）
- **问题**：Agent 可能执行危险操作（文件删除、网络写、调用付费 API），缺少按风险分级校验。
- **方案**：给 `Tool` 增加 `risk_level` 元数据；`Agent` 通过 `permission_level` 在执行前检查，不足时返回权限错误。
- **文件**：`swaybot/tools/__init__.py`、`swaybot/agent.py`
- **验收**：高风险工具在无授权时被拒绝；显式授权后可执行；测试覆盖默认拒绝与授权通过。
- **状态**：已完成（2026-07-17）。

### [ ] AutoCompact 会话压缩

### [x] AutoCompact 会话压缩
- **问题**：长对话后上下文超出模型窗口，且历史未压缩，token 浪费。
- **方案**：当短期历史长度超过阈值时，调用可插拔 brain 将早期步骤压缩为摘要长期记忆，并替换原始步骤。
- **文件**：`swaybot/memory.py`、`swaybot/agent.py`
- **验收**：长对话触发压缩后短期步骤数下降；摘要保留关键决策和最终答案。
- **状态**：已完成（2026-07-17）。

## 当前聚焦

P3 的流式响应可先放一放。P4/P5 中 Final answer、工具校验、监控、inspect、矛盾驱动探索和信念更新闭环已完成。下一步做 **多模型 backend 抽象**，把 `LLMBrain` 拆成通用 `Model` 基类与 `OpenAIModel` 实现，让 Agent 能接入本地模型或其他 API。
