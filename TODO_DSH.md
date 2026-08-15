# SwayBot TODO — 借鉴 deepseek-harness（按性价比排序）

> 基于 `external/deepseek-harness` 架构对比 + τ³-bench 回归实证。
> 核心约束：**不显著增加代码量**。每项标注预估增量 LOC，优先做"高收益 / 低代码"项。
> 关联代码：`swaybot/agent.py`(循环)、`swaybot/hook.py`(钩子)、`swaybot/models.py`(模型)、`swaybot/tools/__init__.py`(工具)、`swaybot/guards.py`(安全)、`swaybot/prompts/system_fc.j2`(提示)。

## 总览

| # | 项 | 来源 | 预估 LOC | 性价比 | 打中的失败模式 |
|---|---|---|---|---|---|
| 1 | banking prompt：检索前不交卷 | banking 回归新发现 | ~5 | ⭐⭐⭐⭐⭐ | banking 0.35→0.05（未检索就交卷） |
| 2 | 防重复调用 guard | dsh repeat-tool-reminder | ~60 | ⭐⭐⭐⭐⭐ | airline task7 step-exhaust（50步重复 get_reservation_details） |
| 3 | finish_reason=length 丢弃残缺 tool_call | dsh BlockAssembler | ~15 | ⭐⭐⭐⭐ | 精化 #3 兜底，防半截工具调用被执行 |
| 4 | 防过早 final_answer guard（banking 加强） | banking 回归 | ~35 | ⭐⭐⭐⭐ | prompt 不足时硬拦截零工具调用收尾 |
| 5 | 结构化工具错误 | dsh toolErrorResult | ~45 | ⭐⭐⭐ | 错误是字符串，LLM 无法区分类型 |
| 6 | 工具超时 policy | dsh timeout-policy | ~30 | ⭐⭐⭐ | 工具可无限挂起 |
| 7 | 递归 JSON schema 校验 | dsh json-schema | ~80(替换) | ⭐⭐ | #4 已在 benchmark 侧绕过，核心校验仍弱 |
| 8 | hook 可干预 waterfall | dsh ctx.on | 高(架构级) | ⭐ | 暂缓 |
| 9 | 会话事件溯源 | dsh session log | 高(架构级) | ⭐ | 暂缓 |

累计 P1+P2 增量预估 **~270 行**，分散在 6 个文件，无新依赖、无新框架。

---

## P1 — 高性价比（低成本 + 直接打中已知失败）

### 1. banking prompt：检索前不交卷 ✅ 已实现（commit 79dcda1）

- **要解决的问题**：banking 域 pass¹ 从 round-5 的 **0.35（7/20）骤降到 #5 回归的 0.05（1/20）**。
  - **根因**：#3 修复把 `_fallback` 从 `raw[:200]` 截断改为全量 prose，`think()` 把 prose 包成 `final_answer`。这修好了截断循环，但暴露了一个更深的倾向——agent 第一轮就反问用户（task001 典型："你的收入？你有 Rho-Bank+ 吗？"），**然后直接 `final_answer` 收尾，全程一次 KB_search 都没调**。
  - **round-5 的 7 个通过里有"水分"**：部分任务是靠"截断→用户说'你被截断了'→被迫继续→阴差阳错调出工具"才通过的。截断循环是 bug，但它客观上阻止了 agent 过早收尾。修掉后真实能力暴露：agent 不会自主检索知识库就交卷。
  - **banking 是 knowledge 域**（纯检索+回答，无 API 调用），正确流程是 KB_search → 综合文档 → final_answer。但 agent 现在跳过了 KB_search。
- **最小方案（纯 prompt）**：在 `swaybot/prompts/system_fc.j2` 的 "Completing your responses" 区块加约束：
  ```
  - Before calling final_answer, ensure you have gathered the information needed to actually answer. If a search/retrieval tool is available and you have not used it yet for this task, do so before answering. Do not end with only clarifying questions if you can retrieve the answer yourself.
  ```
- **文件**：`swaybot/prompts/system_fc.j2`（+5 行）
- **验收**：banking task001 类"只反问不检索"轨迹不再一步收尾；跑 banking 5 个任务，pass¹ 回升；单测覆盖 prompt 渲染含新约束。
- **预估**：~5 行。
- **优先级理由**：**成本最低（5 行）+ 打中当前最大回退点（banking 0.30 降幅）**，应最先做以验证"防过早 final_answer"假设是否成立。若 prompt 够用则无需 #4 guard；若不够再上 #4。

### 2. 防重复调用 guard（repeat-tool-reminder）✅ 已实现（commit 5eb94a2）→ 🔁 重构（接入点 + 检测策略升级）

- **要解决的问题**：airline task7 反复调 `get_reservation_details` 50 步不收敛，无任何机制打断。这是 dsh `repeat-tool-reminder` 的靶心，也是 airline 域当前主要失败模式之一。
- **初版（commit 5eb94a2）**：`swaybot/loop_guards.py` 的 `RepeatCallGuard`，在 `Agent._run` 的 `after_iteration` 里维护 `[(action_name, canonical_args)] -> 连续计数`。阈值 `[3,5,8]`。
- **初版两个缺陷（本轮诊断后修复）**：
  1. **接入点错位**：`SwayBotHalfDuplexAgent`（benchmark 路径）直接调 `brain.think()`，**绕过 `Agent._run`**，所以挂在 `_run` 里的 guard 在 benchmark 里是死代码（v2 回归日志 0 条 notice 证实）。
  2. **检测策略漏检**：task7 的真实失败模式不是"连续相同调用"，而是"**同类工具轮换不同参数**"——每次 `get_reservation_details` 用不同 reservation_id，args 永远不同，exact 计数永远不触发。这是更隐蔽的循环（不收敛到决策）。
- **重构方案**：
  - **接入点下沉到 `LLMBrain`**：guard 挂在 `LLMBrain.think()` 里（`self.repeat_guard` + `self._repeat_notice`），`_with_repeat_notice()` 把 notice 注入下一次 perception 的 `behavior_guidance`。这是 `Agent` 与 benchmark adapter **共用的唯一路径**。`Agent._run` 仅保留"task 切换时 reset guard 链"的职责。
  - **新增 name-only 检测通道**：`RepeatCallGuard` 现在并行跑两条链——(a) exact 链（name+canonical args，`[3,5,8]`），(b) **name-only 链**（只看 name，不看 args，默认 `[5,8]`）。task7 的 8 次 `get_reservation_details`（id 不同）会被 name-only 链在第 5 次抓住。exact 命中时优先返回 exact notice（更具体，含 args 详情）。
- **文件**：`swaybot/loop_guards.py`（+name-only 链 +`_validate_thresholds`）、`swaybot/llm_brain.py`（guard 下沉）、`swaybot/agent.py`（移除 guard 调用，仅留 reset）、`tests/test_loop_guards.py`（+6 name-only 用例）、`tests/test_llm_brain.py`（+2 brain 层 notice 用例）。
- **验收**：单测 373 全绿；mock task7 轨迹（同 name 异 args ×5）→ 第 5 次 think 后 system prompt 含 reminder 文本。
- **下一步实测**：重跑 airline task 5/7/9（2 并发），确认 name-only notice 出现在日志、step-exhaust 下降。

### 3. finish_reason=length 丢弃残缺 tool_call ✅ 已实现（commit c4b9067）

- **要解决的问题**：#3 当前的兜底是 prose→`final_answer`（事后补救）。dsh 的做法更优：max-tokens 时直接丢弃未完成的 tool-call block，不让半截工具调用进入执行路径。SwayBot 的 `OpenAIModel._handle_completion_response` 不区分 finish_reason，可能把 args 被截断的残缺 tool_call 当成合法 action 执行。
- **最小方案**：`swaybot/models.py` 的 `_handle_completion_response`：
  - 已记录 `metadata["finish_reason"]`。
  - 当 `finish_reason == "length"` 且存在 tool_call 时，**不把残缺 tool_call 序列化为 action**，优先信任完整 content（prose），让 `think()` 走 #3 已有的 prose→final_answer 路径，避免执行一个 args 被截断的 JSON。
- **文件**：`swaybot/models.py`（`_handle_completion_response` ~10 行调整）
- **验收**：mock 一个 `finish_reason=length` + 残缺 tool_call 的响应 → 返回 prose 而非执行残缺工具。
- **预估**：~15 行。
- **dsh 对照**：`packages/llm/llm/src/assembler.ts:134-139`（`blocks()` 在 max-tokens 时 filter 掉 tool-call）。

### 4. 防过早 final_answer guard（banking 加强）

- **要解决的问题**：#1 的 prompt 修复可能不够——弱模型（glm-5.1-fp8 量化）即使 prompt 明确要求"检索前不交卷"，仍可能在第一轮就 `final_answer`。需要一个硬性兜底：检测到"final_answer 时全程零工具调用"就拦截。
- **最小方案**：`Agent._run` 里，拿到 action 后、`_execute_action` 前，判断：若 `action.name == "final_answer"` 且 `env.history` 中没有任何工具调用（或没有检索类工具调用），则不执行 final_answer，而是回注一条提示并继续循环——"你还没检索知识库就给出了最终答案。先用 KB_search 检索相关文档，再回答。"
  - 把 final_answer 替换成 echo 提示 + 继续循环（不消耗 final_answer 终止语义）。
  - 设一个上限（如已拦截 2 次仍不检索就放行，避免死循环）。
- **文件**：`swaybot/agent.py`（`_run` 内 ~30 行判断）+ `swaybot/prompts/system_fc.j2`（notice 文本可选）
- **验收**：零工具调用时 final_answer 被拦截并回注提示；已调用过工具的不受影响；拦截上限触发后放行。
- **预估**：~35 行。
- **前置**：先做 #1 prompt 验证；若 prompt 不够再做本项。

---

## P2 — 中性价比（稳健性提升，非当前阻塞）

### 5. 结构化工具错误（isError + code）

- **要解决的问题**：`Agent._execute_action` 把异常变成 `f"Error: {exc}"` 字符串。下游（含 LLM）无法区分超时/权限/参数错误/运行时错误，LLM 对不同错误类型无法采取不同纠正策略。
- **最小方案**：`ToolRegistry.execute` 捕获异常时返回 `{"error": str(exc), "code": _classify(exc)}`，`_classify` 把 `TimeoutError`→`timeout`、`PermissionError/SecurityError`→`permission`、`ValueError`→`invalid_args`、其余→`runtime`。现有"Error: 字符串"行为保持向后兼容（content 仍含原文）。
- **文件**：`swaybot/tools/__init__.py`、`swaybot/agent.py`、`swaybot/violations.py`
- **验收**：工具抛 TimeoutError → observation 含 `code=timeout`；LLM 能看到分类后的错误。
- **预估**：~45 行。
- **dsh 对照**：`toolErrorResult` + `error.info {name, code}`。

### 6. 工具超时 policy

- **要解决的问题**：工具（尤其 shell/web）可无限挂起，无超时。#5 的结构化错误里 `timeout` code 无来源。
- **最小方案**：`Tool` 增加可选 `timeout_ms`；`ToolRegistry.execute` 用 `concurrent.futures.ThreadPoolExecutor` + `future.result(timeout=timeout_ms/1000)` 包裹非并发安全工具，超时抛 `TimeoutError`（由 #5 归类为 `timeout`）。
- **文件**：`swaybot/tools/__init__.py`
- **验收**：声明 `timeout_ms=1000` 的工具 sleep 5s → 1s 后返回 timeout 错误；未声明的不受影响。
- **预估**：~30 行。
- **dsh 对照**：`packages/guard/timeout-policy/src/index.ts`。

### 7. 递归 JSON schema 校验

- **要解决的问题**：`tools/__init__.py` 的 `_check_type` 只做单层 isinstance，不递归、不区分 number/integer、不校验嵌套 required。#4 bug（`gb_amount:"2"`）根因就是 schema 没被正确提取+校验，目前靠 benchmark 侧 `_coerce_args_to_schema` 绕过，核心校验仍弱。
- **最小方案**：把 `_type_to_json_schema` + `_check_type` + `validate_arguments` 替换为递归校验（区分 number/integer、嵌套 object/array、required 深度检查）。可选 `coerce=True` 开关，校验失败时尝试 str→number 强转（对弱模型友好，与 dsh 的严格路线互补）。
- **文件**：`swaybot/tools/__init__.py`（替换 `_type_to_json_schema`/`_check_type`/`validate_arguments`）
- **验收**：`{"amount": "2"}` 对 schema `{"type":"number"}` → 校验失败（或强转后通过，看开关）；嵌套 object 的 required 缺失被检出。
- **预估**：~80 行（替换现有 ~30 行，净增 ~50）。
- **dsh 对照**：`packages/core/tools/src/json-schema.ts`（657 行，强制子集 + 栈安全）。SwayBot 只取"递归校验"核心，不做子集拒绝。

---

## P3 — 暂缓（架构级，成本高）

### 8. hook 升级为可干预 waterfall

- **现状**：`AgentHook` 是观察型（before/after_iteration 只读），不能改写 action、不能 veto。#2/#4 的 guard 目前靠在 `_run` 里内联实现。
- **dsh**：Cordis `ctx.on(event, waterfall)` 让插件能拦截/替换/注入。
- **暂缓理由**：SwayBot 的单体同步循环里，内联 guard 已够用；waterfall 化是架构级改动，收益不抵复杂度。等 guard 积累到 3+ 个、且都需"改写 action"时再升级。
- **预估**：高（重构 hook 协议 + 所有调用点）。

### 9. 会话事件溯源 + 可重放

- **现状**：`run_log.py` 是步级 append-only 记录，非事件流；不支持重放/回放。
- **dsh**：session log 是事件流（turn/start, step/start, assistant/chunk, tool/call, tool/result...），"Model-visible ⟺ logged" 是硬不变量。
- **暂缓理由**：τ³-bench 不需要重放；当前 run_log 够用。等需要时间旅行调试/回放时再做。
- **预估**：高（重构持久化层）。

---

## 建议执行顺序

1. **#1 banking prompt**（5 行）→ banking 5 任务快速验证 pass¹ 是否回升。**最优先**：成本最低、打中最大回退点。
2. **#2 防重复 guard**（~60 行）→ airline 5/7 验证 step-exhaust 是否消除。
3. **#3 finish_reason 残缺丢弃**（~15 行）→ 单测 + 小回归。
4. **#4 banking guard 加强**（~35 行，仅当 #1 prompt 不够）→ banking 全量。
5. P2 按需：#5→#6→#7。

每项做完跑对应域的 5-20 任务快速回归（2 并发），确认无回退再进下一项。
