# SwayBot

> 一个轻量优雅、会自己成长进化的 Agent。

SwayBot 的设计追求小而美：体积轻巧、结构清晰、能力开放。它不会在一开始就做尽所有事情，而是从一个最小内核出发，通过经验不断扩展自身——习得新技能、优化行为方式、适应新的任务。

## 愿景

我们相信，下一代 Agent 不应该是庞大的单体系统，而应该具备以下特质：

- **轻量** —— 易于运行、易于理解、易于修改。
- **优雅** —— 每个部分都有其存在的理由。
- **自我进化** —— 能够观察、记忆，并随时间不断改进自己的行为。

SwayBot 就是一次让这种 Agent 成真的尝试。

## 核心理念

- **最小内核，生长式边界** —— 从简单开始，通过学习能力而非硬编码来扩展。
- **经验即结构** —— 把学到的东西转化为可复用的模式、工具和工作流。
- **与人类意图对齐的进化** —— 成长由目标、反馈和明确边界共同引导。

## 快速开始

SwayBot 基于 Python 3.10+ 构建，核心无运行时依赖。

```bash
# 克隆仓库
git clone https://github.com/askender/swaybot.git
cd swaybot

# 以可编辑模式安装
pip install -e .

# 运行 Agent
python -m swaybot "count to 3" --max-steps 5
```

### 运行测试

```bash
pip install -e ".[dev]"
pytest
```

## 架构

最小循环为 `感知 → 思考 → 行动 → 观察 → 循环`：

- `Environment` 保存任务、步数计数器和观察历史。
- `Brain` 决定下一步动作。默认的 `EchoBrain` 是确定性大脑，无需 API 密钥。
- `ToolRegistry` 将动作分发给工具（`echo`、`add`、`done`）。
- `Agent` 将它们串联起来，运行到任务完成或步数耗尽。

## 记忆

SwayBot 可选地使用 `MemoryStore` 来记录经验、事实、理论、猜想和灵感。记忆按 `scope` 区分为 `short_term`（单次运行的原始经验）和 `long_term`（经过反思验证的长期知识）。每条记忆都携带来源、证据、可信度、意外程度和标签，以便 Agent 后续检索相关的长期上下文或寻找反例。

```python
from swaybot import Agent, MemoryStore

store = MemoryStore(path="~/.swaybot/memory.json")
agent = Agent(memory=store)
agent.run("explore a topic", max_steps=5)
```

## 反思

一次运行结束后，SwayBot 可以对发生的事情进行反思：总结经历、标记意外事件、检测记忆中的矛盾，并根据已存储的事实验证主张。反思结果会被存为 `long_term` 的 `theory` 记忆，形成一个自我改进的循环——原始经验逐渐转化为结构化知识。

```bash
python -m swaybot "explore colors" --max-steps 5 --reflect
```

输出现在以可读的工具调用形式呈现：

```text
Step 1: echo(message="thinking...") → thinking...
Step 2: done() → finished
```

## 大模型大脑

SwayBot 可以将 OpenAI 兼容的聊天模型作为大脑。安装可选依赖后使用 `--brain llm` 运行：

```bash
pip install -e ".[llm]"
export SWAYBOT_API_KEY="your-key"
export SWAYBOT_API_BASE="https://api.example.com/v1"
export SWAYBOT_MODEL="your-model"
python -m swaybot "2+2 等于多少？" --brain llm --max-steps 3 --reflect
```

或者在项目根目录创建 `.env` 文件（参考 `.env.example`），CLI 会自动加载其中的 `SWAYBOT_API_KEY`、`SWAYBOT_API_BASE` 和 `SWAYBOT_MODEL`。默认情况下，记忆会持久化到 `~/.swaybot/memory.json`；可使用 `--data-dir` 或 `--memory` 自定义位置，或使用 `--no-memory` 禁用持久化。

默认大脑仍是 `EchoBrain`，因此核心包保持无额外依赖。

## 路线图

- [x] 定义最小 Agent 循环
- [x] 构建记忆原语
- [x] 构建反思原语
- [x] 接入大模型推理
- [ ] 增加自我改进机制
- [ ] 记录成长模式与示例

## 许可

MIT © 2026 anwen
