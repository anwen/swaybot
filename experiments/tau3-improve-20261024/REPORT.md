# tau3 失败驱动改进环：从 0.000 到全量评测（2026-10-24）

**问题**：banking_knowledge 基线接近 0 分（前次 A/B 实验测试集 0.000）。能否通过失败驱动的
harness 改进逐步提高得分率，然后放大到完整评测？

## 方法

开发集 = banking 前 20 题（base 臂，完整 transcript 落盘：`transcripts/base/`），
程序型任务加测 015-020（base-proc 臂）。每轮：跑 → 读 transcript 归因 → 通用性修复 → 复测。

## 失败分类（base 臂 20 题）

| 根因 | 命中 | 说明 |
|---|---|---|
| **API 1214 错误循环** | 18/20 题（共 39 次） | 会话跨过裁剪边界（旧预算 8k 字符/14 条）即触发；导致记忆清零 → 重验证死循环（9 题 log_verification 2-4 次）→ 50 步耗尽 |
| **隐藏工具协议未知** | 78/97 任务类型 | tau3 银行域 80% 任务要求 unlock/call/give_discoverable_* 内部工具链；行为指导只字未提 |
| **程序文档被摘要器截断** | 检索命中也无效 | KB 摘要只留 3 行+金融关键词行，把 `get_referral_link(user_id, card_name)` 这类工具签名全部剪掉（task_015 连查 4 次同一文档都看不到关键行） |

## 三个结构修复（commit 941755f）

1. **上下文预算 8k→60k 字符 / 14→60 条**：旧预算被两三条 KB 摘要击穿，静默丢弃用户请求与全部
   早前发现（实测 16 条历史被裁到 2 条）。
2. **裁剪边界安全**：丢弃孤儿 tool 消息（assistant tool_call 落在窗口外的结果）与连续纯文本
   assistant 消息——两种形态都被 bigmodel 以 error 1214 拒绝。修复后 fix/fix2 两臂 **1214 = 0**
   （base 臂 39 次）。
3. **程序文档整篇保留（≤1800 字符）**：凡含协议线索（unlock/call_discoverable、`_\d{4}` 工具名、
   "instruct them to run" 等）的文档不再 3 行截断。

## 行为指导 v2（commit 4f70880，均为机制级、非泄题）

- 隐藏工具协议：文档→unlock→call（agent 侧）/ give→用户执行（用户侧）；"has not been given"
  错误的即时恢复；逐项执行（多笔交易逐笔调用）；不得发明文档没写的工具名
- 验证全菜单：四选二，单项不可得/不匹配时换下一项而非放弃；log_verification 七字段逐字复制，
  time_verified 取自 get_current_time()
- 精确标题跟踪检索：结果标题像目标程序文档时，用该标题原样再查一次

## 开发集结果

| 臂 | 配置 | 分数 | 说明 |
|---|---|---|---|
| base（20 题） | 修复前 | **0.100** | 仅 001/006（信用卡申请型）通过 |
| fix-proc（6 题） | 修复 v1 | 0.000 | 1214 清零、验证一次通过、020 完整走通 unlock→call→give 协议（仅漏 3/4 笔争议交易）；残余失败上移至检索质量 |
| fix2-proc（6 题） | 指导 v2 | 0.000 | 程序型任务败因集中在 bm25 检索捞不到程序文档（模型能力边界） |

**判定**：结构修复消除了全部崩溃性失败（1214、验证死循环），验证动作与 golden 完全一致的样本
大量出现；但 glm-5.3-flash + bm25 在程序文档检索上仍是硬瓶颈，6 道程序型开发题未能翻正。
全量 97 题含 71 道未开发过的任务——最终判据看全量。

## 完整评测（进行中/结果）

- banking_knowledge 97/97，双车道，bm25 检索，temperature 0，无学习指导注入（纯 harness 成绩）
- 结果文件：`results-full.jsonl`；transcripts：`transcripts/full-lane{1,2}/`

## 局限与后续

1. bm25 对释义型查询弱（"cash back wrong" 匹配不到 "update_transaction_rewards"）；dense 变体
   依赖 embedding 端点未启用——已知局限，本轮不换测量仪器
2. 程序型任务的检索-连接链路是下一个改进面（模型侧：换 glm-5.3 非_flash；harness 侧：KB 索引
   按标题建立别名）
3. 知识迁移层（learned-guidance 注入）在全量成绩出来后作为独立臂复验
