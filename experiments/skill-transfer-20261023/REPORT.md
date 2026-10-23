# 形态 A 检验：self-evolution skill 移植到 zcode-cli（2026-10-23）

**问题**：swaybot 的自我进化机制能否以纯 prose 协议（skill）的形式让另一个 harness 复现？
**设置**：`skills/self-evolution/SKILL.md`（58 行纪律描述）放入 zcode-cli 工作区的
`.zcode/skills/`，同款 glm-5.3-flash @ bigmodel，种子同生命实验。
**第一次尝试失败**：agent 加载了 skill 然后输出 8192 token 全部被 thinking 吃光——
零可见文本零工具调用，轮结束（加载≠执行；thinking 预算问题跨 harness 存在）。
**第二次（--max-tokens 32768 + 行动强制措辞）成功**：~9 分钟，21 次工具调用。

## 协议遵从度（全部来自纯 prose，无任何代码支撑）

| 协议条目 | 遵从证据 |
|---|---|
| 状态文件 | 5 个文件全部创建并维护 |
| 预注册 | 每个实验先写 T1-T10 / A1-A4 编号探针与判据再执行 |
| 裁决引证 | "T3 HTTP 200、T4 ping 81ms、T8 whoami=askender、出口 IP=武汉电信与主机名自洽" |
| 置信度随证据 | 0.3→0.5（supported）、→0.2（部分反驳） |
| 问题生命周期 | 3 个种子全部 resolved；衍生 3 个新 open 问题（分解自发发生） |
| 冷恢复自觉 | **第三个实验测的是自己的状态文件能否被新会话恢复**——发现 knowledge.md 过期未刷新，裁决"结构化状态+及时沉淀刷新才有留存价值；过期的叙述文件会主动误导后续会话" |

## 两个 harness 的行为差异（同模型同协议）

- **证据风格**：zcode 版的实验更"野"——直接写工作区外（/tmp 根成功）、ping 外网、
  查 IP 归属地做自洽校验；swaybot 版在沙箱内更克制。原因：zcode Bash 是 yolo 无
  工作区约束，swaybot 有 PathGuard
- **认识论亮点**（zcode 版独有）：它把"上下文日期 vs 系统时钟 vs 网络权威时间"
  三方对照（偏差 <1 分钟），主动检验自己观察流的可靠性——这步 swaybot 版没做过
- **对协议弱点的发现**：它用实验证明了 skill 自己的薄弱点（沉淀文件不及时刷新会
  误导续命会话）——协议被它的执行者审计了

## 结论

形态 A 成立：**纪律可以以 prose 迁移**，且执行质量足以支撑跨 harness 复现。
skill 已入 `skills/self-evolution/SKILL.md`（swaybot 侧同样可加载——双宿主验证过
frontmatter 兼容）。留存产物在本目录（已核验零密钥泄漏；zcode 侧 Bash 无脱敏，
其 env 探针未带出密钥值属运气——跨 harness 使用时建议给宿主加输出脱敏）。
