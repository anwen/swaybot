# 八皇后对照实验：swaybot vs zcode-cli（2026-10-22）

同模型（glm-5.3-flash @ open.bigmodel.cn）、同后端、同用户提示词、同工作区隔离条件下的
单任务对照。zcode-cli 经 `ZCODE_DATA_BASE_DIR` 临时 provider 配置接入同一 bigmodel 凭据
（Anthropic 兼容端点 `open.bigmodel.cn/api/anthropic`）。

## 任务提示词（swaybot 风格，两边逐字相同）

> Write a Python program that solves the eight queens puzzle, save it as
> eight_queens.py, then run it with the shell and verify it reports exactly
> 92 distinct solutions. Finish with a final answer stating how many
> solutions were found and whether verification passed.

## 结果总览

| 维度 | swaybot (v0.3.0) | zcode-cli |
|---|---|---|
| **任务结果** | **未完成**——代码写对，无法运行验证 | **完成**，验证 92 通过 |
| 模型调用数 | 12 | 3 |
| 工具调用 | 2×write_file + 8×run_shell_command + done | Write + Bash |
| tokens (input/output) | 40,743 / 4,294 | 11,950 / 1,194 |
| system prompt | 4,126 字符（guard notice 注入后最高 4,494） | 2,448 字符 |
| 暴露的工具数 | 12 | 17 |

## 决定性差异：沙箱策略，不是模型也不是循环

swaybot 的 `CommandGuard` 允许列表为 9 个只读命令（`ls/cat/grep/find/pwd/echo/wc/head/tail`），
**不含任何解释器**，且与权限模式无关（danger 模式同样拦截）；管道符等元字符亦被禁。
"写代码 → 运行验证"这类任务在当前白名单下结构性不可能完成。

模型的应对其实相当理性：写验证脚本（自带断言）→ 尝试 `python3`/`python`/`python3.12`/`python3.11`/`python3.13`
→ `echo probe` 试探 → 放弃收尾。第 9 次调用起 system 增大 200-370 字符，是
RepeatCallGuard/NonConvergenceGuard 的 notice 注入——**guard 机制在真实失败中正确触发**，
但提醒解决不了白名单硬约束。

zcode-cli 的 yolo 语义下同一模型 2 次工具调用收工：Write（一次成型）→ Bash（运行+内联验证输出）。

## 其他观察

- **Token 经济**：swaybot 花费 3.4× input / 3.6× output，全部消耗在错误重试循环上
- **协议健康度**：swaybot 结构化路径 12 次调用中动作全部走 tool_call 字段（text 仅 guard 相关短文本）；
  zcode 的 Anthropic 流解析同样干净（text+tool_use 混合、thinking 块被正确忽略）
- **质量钩子**：swaybot 的 write_file 自动跑了 ruff format/lint + pyright，全部通过——
  写码质量这道环节 swaybot 反而更讲究
- **上下文增长**：两边 messages 线性增长（3→25 vs 3→7），无异常膨胀

## 结论与后续

1. **swaybot 需要"可执行代码"的沙箱策略**：白名单补 `python`/`pytest` 等解释器，
   或按权限模式分级（danger 放开执行、workspace-write 放开受限执行）。这是"自我进化 agent"
   的核心能力缺口，已记入 TODO。
2. guard 提醒对硬约束无效——notice 文案可补一句"若命令被策略拒绝，换工具或直接收尾"
   （提示词里已有类似句，但针对的是 permission denied 通道，CommandGuard 错误走的是
   工具异常文本，模型未能关联）。
3. 对照实验方法本身可复用：同模型同后端双跑 + 双侧 JSONL 对账，建议作为 BENCHES 的
   附属工具保留。

## 产物

- `swaybot-rollout.jsonl` / `zcode-rollout.jsonl`：两侧逐调用 model-io 记录
- `swaybot-session.ndjson`：swaybot 事件流会话记录（12 步全程）
