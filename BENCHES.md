# BENCHES · 基准分数台账

> 每次基准评测记一行：有数字、有归因。方法论参照 zcode-cli 的 PARITY.md——漂移必须可追溯，
> 被证伪的假设照实记录。历史行由存档结果与 DEV_SESSION_ARCHIVE.md 回填，当时未留档的以 ? 标注。
> 注意：2026-10-16 的历史重写（清除泄露密钥）更改了全部 commit hash，早期行以日期+提交说明定位。

## tau3-bench（τ³，reward = 任务通过率）

| 日期 | 域 | 分数 | 配置 / 归因 | 证据 |
|---|---|---|---|---|
| 2026-08-11 | mock | 0.80 (n=10) | 首轮并行评测（7-27 提交基线） | `tau3_results/parallel_evaluation.json` |
| 2026-08-11 | airline | 0.55 (n=20) | 同上 | 同上 |
| 2026-08-11 | telecom | 0.90 (n=20) | 同上 | 同上 |
| 2026-08-12 | airline | 0.4167 (n=20) | 全量首轮（retail/banking 首次出分） | `tau3_results/parallel_evaluation_all.json` |
| 2026-08-12 | retail | 0.50 (n=20) | 同上 | 同上 |
| 2026-08-12 | telecom | 0.9167 (n=20) | 同上 | 同上 |
| 2026-08-12 | banking_knowledge | 0.5714 (n=20) | 同上 | 同上 |
| 2026-08-12 | banking_knowledge | 0.30 (n=20) | 晚间复跑回落（0.57→0.30），触发 #3 prose 兜底改造排查 | `tau3_results/parallel_evaluation_mock_airline_retail_telecom_banking_knowledge.json` |
| ~2026-08-14 | banking_knowledge | 0.35→0.05 | **round-5 → #3 回归**：`_fallback` 改全量 prose 后 agent 不检索就交卷（详见 TODO_DSH #1） | `tau3_results/logs/`（round 系列日志） |
| 2026-08-15~18 | banking / airline / telecom | 定向回归 | prompt 防过早交卷、RepeatCallGuard、非收敛 guard、calculate 引导、prose 工具标签恢复、KB 综合修复逐项验证（commit 说明见 git log 8-15~8-18） | `tau3_results/logs/reg2~reg5_*.log`、`v*_task*.log` |
| 2026-10-22 | banking_knowledge | task_001 **1.0**（两次复跑一致） | 冒烟回归：新网关（open.bigmodel.cn，glm-5.3-flash，8 月分数为旧网关 glm-5.1-fp8，**不可直接对比**）；验证拒绝引导文案、env_info、RecordingModel rollout（每回合一行、无凭据落盘）均无副作用。task_001 即历史上"反问后零检索交卷"的典型案例，现已通过 | `tau3_results/rollout/model-io-*.jsonl` |
| 2026-10-22 | retail | task 5 **0.0**（修复后正常跑完） | 首跑报 `get_environment() got unexpected keyword 'retrieval_variant'`——tau2 2026-10 签名变化（retrieval_config 仅 banking_knowledge 构造器接受）。已在 tau3bench CLI 按域签名探测转发，修复后跑通；0.0 为新模型单点数据，全量见下一行 | `tau3_results/rollout/` |
| 2026-10-24 | banking_knowledge（bm25） | dev-20 = **0.100**（旧适配器，修复前） | 失败归因：18/20 题命中 bigmodel error 1214（上下文裁剪产生孤儿 tool 消息/连续 assistant），9 题陷入重验证死循环；隐藏工具协议（78/97 题所需）指导为零 | `experiments/tau3-improve-20261024/transcripts/base/` |
| 2026-10-24 | banking_knowledge（bm25，941755f+4f70880） | 同 20 题 **0.150**；干净段 001-027（24 题）**0.125** | 结构修复（裁剪预算 8k→60k、边界安全、程序文档整篇保留）+ 协议/验证指导：1214 清零、验证恰好一次、010 首过。全量 97 题在 task_028 处 **API 余额耗尽**中断（028-101 以 RateLimitError 记 0，非 agent 失败），待充值补跑 | `experiments/tau3-improve-20261024/`（REPORT.md + transcripts/full-lane*） |
| 2026-10-25 | banking_knowledge 97/97 全量（ee5d869，套餐端点 coding/paas/v4） | **0.062**（6 过：001/006/010/034/035/046） | v4 协议编译器（KB 协议语言→点名调用指令）：**程序题零的突破**（035 完整 unlock+call 内部工具链；034 转接型；046）；用户模拟器开始主动尝试用户侧工具（30 次"not given"错误）→ v5（11bcf9c）针对该缺口：指令持续注入至真正调用 + 错误即时触发 | `experiments/tau3-improve-20261024/`（transcripts/full-lane*-v4bak） |
| 2026-10-25 | banking_knowledge 97/97 全量（11bcf9c = v5） | **0.083**（8 过：+002/032，v4 全保） | 持续指令+错误触发生效：give 链路首次走通（018：give→用户提交争议）；剩余瓶颈=检索质量+多卡对账（模型边界）。轨迹：base 0.100(dev-20)→v2 0.150(dev-20)→v4 0.062(全量)→v5 0.083(全量) | `experiments/tau3-improve-20261024/`（transcripts/full-lane*-v5bak，REPORT.md） |
| 2026-10-25 | airline 抽样 n=20（11bcf9c，套餐端点） | **0.800**（16/20） | 同一修复栈跨域泛化：1214 裁剪修复对长对话事务域收益最大（前 10 题全过）；失败 4 题（18/21/23/24）待归因 | `experiments/tau3-improve-20261024/transcripts/airline-sample/` |
| 2026-10-25 | retail 抽样 n=20（11bcf9c，套餐端点） | **0.550**（11/20） | 首题段（1-97 内）9/11 过；失败集中在 100+ 段（100-113，8/9 败）——高分号任务疑似更难/多步，待逐题归因 | `experiments/tau3-improve-20261024/transcripts/retail-sample/` |
| — | **全量（修复后）** | **未跑** | retail 适配器已修复可跑；网关/模型已换（open.bigmodel.cn / glm-5.3-flash，与 8 月不可直接对比），分数重开基线 | — |

## AgentDojo（utility / security）

| 日期 | 配置 | 分数 | 归因 | 证据 |
|---|---|---|---|---|
| 2026-07 下旬 | EchoBrain 冒烟 | banking 31.25%，其余 0% | 适配器可运行性验证；当时内网 LLM 服务 502 | DEV_SESSION_ARCHIVE.md |
| 2026-07-27 | LLM 全量（原生 FC 修复 + 注入引导后） | **93.81% / 100%** | 4 个 utility 失败均为语义理解偏差，非工具调用错误 | `benchmarks/agentdojo_failure_analysis.md` |

## BFCL v3（single-turn）

| 日期 | 类别 | 分数 | 归因 | 证据 |
|---|---|---|---|---|
| 2026-07-27 | simple_python | 56.25% | 单工具调用路径正常 | `score/data_*.csv` |
| 2026-07-27 | multiple | 38.00% | | 同上 |
| 2026-07-27 | irrelevance | 65.00% | | 同上 |
| 2026-07-27 | parallel | 0.00% | server 当时每轮只输出一个 tool call | DEV_SESSION_ARCHIVE.md |
| 2026-07-27 | multi_turn_base | 未完成 | 200 case × 20 步过慢 + 同样依赖多工具调用 | 同上 |

## 如何记新行

1. 跑评测前记下当前 commit（`git rev-parse --short HEAD`）。
2. 结果落盘到 `tau3_results/` / `score/` 等既有位置（已 gitignore），台账只记**数字与归因**。
3. 分数变化必须给一句话归因（改了什么/为什么）；归因不了的变化本身就是待查问题。
4. 被后续证伪的归因不删行，加"证伪"标注。
