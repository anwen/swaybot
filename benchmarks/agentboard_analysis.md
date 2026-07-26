# AgentBoard 集成价值分析

## 概述

AgentBoard 是一个涵盖 9 个子任务、4 个领域（具身 AI、游戏、Web、工具）的多维度 agent 评测框架，其核心贡献是 **6 维子技能画像**（Memory、Planning、World Modeling、Self-reflection、Grounding、Spatial Navigation）和 **Progress Rate（部分得分）** 指标。

---

## 与已集成的三个 benchmark 的对比

| 维度 | AgentBoard | tau3-bench | AgentDojo | BFCL v3 |
|---|---|---|---|---|
| 多轮任务完成 | ✅ 全部 9 任务 | ✅ | ❌ | ❌ |
| 函数调用准确率 | 🔶 工具任务 | 🔶 部分 | ❌ | ✅ 核心焦点 |
| 对抗鲁棒性 | ❌ | ❌ | ✅ 核心焦点 | ❌ |
| 具身 AI | ✅ 3 任务 | ❌ | ❌ | ❌ |
| 游戏 / 规划 | ✅ 2 任务 | ❌ | ❌ | ❌ |
| Web 导航 | ✅ 2 任务 | 🔶 部分 | ❌ | ❌ |
| 部分得分 (Progress Rate) | ✅ | ❌ | ❌ | ❌ |
| Grounding 准确率 | ✅ | ❌ | ❌ | ❌ |
| 难度分层评估 | ✅ | ❌ | ❌ | ❌ |

---

## AgentBoard 的独特价值

### 1. Progress Rate（部分得分）— 最大的差异化价值
- tau3-bench 和 BFCL 只有二元通过/失败。AgentBoard 对每个交互步骤提供连续奖励信号，可区分"几乎成功"和"完全失败"
- 可绘制 progress-over-steps 曲线，分析 agent 效率和学习趋势

### 2. 6 维子技能画像
- 加权维度评分生成雷达图，揭示 agent **为什么**失败，而不仅仅是**是否**失败
- 例如：AlfWorld 对 World Modeling 权重=3，对 Memory 权重=1；WebArena 对 Memory/Planning/World Modeling 权重均为 3
- 这种诊断能力在已集成的三个 benchmark 中均不存在

### 3. 跨领域统一评测
- 一个框架覆盖具身 AI（AlfWorld、ScienceWorld、BabyAI）、文本游戏（Jericho）、规划（PDDL）、Web 导航（WebShop、WebArena）、真实 API 工具（天气/电影/Todo/表格/学术）
- 已有 benchmark 覆盖范围窄：tau3-bench 主要是知识工作场景，BFCL 是单轮，AgentDojo 只测安全性

### 4. Grounding 准确率
- 独立于任务成功率的度量：agent 输出的 action 有多少比例在环境中是合法可执行的
- 揭示 agent 产生"幻觉 action"的倾向，这在其他 benchmark 中未被单独度量

### 5. 难度分层
- 每个测试实例标注 hard/easy，可分析 scaling 行为

---

## 集成成本

### 接口适配
AgentBoard 的 `BaseAgent` 接口：

```python
class BaseAgent:
    def reset(self, goal, init_obs, init_act=None): ...
    def run(self) -> tuple[bool, str]: ...       # (success, action)
    def update(self, action, state): ...
    @classmethod
    def from_config(cls, llm_model, config): ...
```

SwayBot 适配器需要：
1. 在 `reset()` 中接收 goal + init_obs → 构建 perception
2. 在 `run()` 中调用 `brain.think()` → 将 action dict 转换为环境特定的 action 字符串
3. 在 `update()` 中记录 action + observation → 维护 history

**复杂度：中等**。接口比 tau3-bench 和 AgentDojo 更简单（纯字符串 I/O），但需要为每个环境写 action 格式映射。

### 基础设施开销

| 环境 | 依赖 | 额外要求 |
|---|---|---|
| AlfWorld | alfworld pip 包 | 无 |
| ScienceWorld | scienceworld pip 包 | 无 |
| BabyAI | minigrid pip 包 | 无 |
| Jericho | jericho pip 包 | 无 |
| PDDL | pddlgym + Fast Downward | 系统级安装 |
| WebShop | Flask + faiss + OpenJDK 11 | 搜索索引构建 |
| WebArena | Playwright + Docker 容器 | Xvfb、dbus、4+ Docker 容器 |
| Tool (真实 API) | 外部 API key | Open-Meteo/TMDB/Todoist/Google Sheets/AMiner |

**WebArena 是最大的基础设施瓶颈**：需要运行多个 Docker 容器（购物网站、论坛、CMS、GitLab），配置 Playwright + Xvfb，管理登录会话。WebShop 也需要构建搜索索引。

**Tool 任务的 API 依赖**创建可重复性和稳定性风险（外部 API 可能变更、限流、或下线）。

### 计算成本
- GPT-4 API：约 5.5 小时
- GPT-3.5-Turbo API：约 3 小时
- 本地 LLM（8×V100）：13-28 小时

---

## 建议与结论

### 优先级：**中等偏低**

**推荐集成的部分（高 ROI）：**

1. **AlfWorld + ScienceWorld + BabyAI + Jericho**（纯 pip 包，无外部基础设施）
   - 覆盖具身 AI + 文本游戏
   - 提供 World Modeling 和 Self-reflection 维度评分
   - 集成成本 ≈ AgentDojo（写一个 `SwayBotAgent(BaseAgent)` 适配器 + 环境特定的 action 格式化）

2. **PDDL 任务**（需要 Fast Downward，但可本地安装）
   - 提供纯规划能力的独立度量

**暂不推荐的部分（低 ROI）：**

3. **WebArena** — 基础设施太重（4+ Docker 容器、Playwright、Xvfb），且与 tau3-bench 的 Web 功能部分重叠
4. **WebShop** — 搜索索引构建繁琐，WebArena 已覆盖 Web 导航能力
5. **Tool 任务** — 外部 API 依赖导致可重复性差；BFCL v3 + tau3-bench 已覆盖工具使用评估

### 替代方案：仅提取子技能评分框架

如果不需要运行 AgentBoard 的全部环境，可以仅提取其 **6 维评分权重矩阵**和 **Progress Rate 计算逻辑**，应用于 SwayBot 已有的评测场景。这避免了基础设施开销，同时获得子技能画像的诊断价值。

### 最终建议

**分两步走：**

1. **现在**：集成轻量级子任务（AlfWorld/ScienceWorld/BabyAI/Jericho），获取 AgentBoard 独有的子技能画像和 Progress Rate 指标
2. **之后**：如果子技能画像揭示出 Web 导航或真实 API 使用的显著弱点，再考虑集成 WebArena/Tool 任务

相比已集成的三个 benchmark，AgentBoard 的边际价值主要在 **诊断能力**（子技能画像 + Progress Rate），而非覆盖新的评估维度。如果 SwayBot 的当前优先级是功能调用准确率（BFCL）和安全性（AgentDojo），AgentBoard 可以推迟到需要精细化诊断时再集成。
