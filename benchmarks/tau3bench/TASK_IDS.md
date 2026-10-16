# τ³-bench task-id 速查

> 来源：2026-07-15 ~ 08-18 开发会话的项目记忆（Claude Code memory 文件 `tau3bench-task-ids.md`），
> 原始转录已过期清理，此处为项目内留档。适配器代码见 `benchmarks/tau3bench/adapter.py`。

`tau3bench --task-ids` 按 `task.id` **精确匹配**过滤任务（见 `tau2.run.get_tasks`），
格式传错会抛 `ValueError: Not all tasks were found`。

## 各 domain 的 task-id 格式

| Domain | 格式 | 示例 |
|---|---|---|
| airline / retail | 纯数字字符串 | `5 7 9` |
| banking_knowledge | `task_001` .. `task_020` | `task_001` |
| telecom | 结构化串，含 `\|` 多问题与 `[PERSONA:None\|Easy\|Hard]` 难度后缀，需逐个加引号 | `"[mobile_data_issue]data_usage_exceeded[PERSONA:Easy]"` |

telecom **没有**数字型 task-id。

## 发现合法 id 的方法

读 `<TAU2_DATA_DIR>/tau2/domains/<domain>/tasks.json`（JSON list，每项的 `id` 字段即 task-id）。

## 本机数据路径（易踩坑）

`TAU2_DATA_DIR`：

```
~/.cache/modelscope/datasets/evalscope--tau3-bench-data/snapshots/master
```

注意目录名是 **modelscope（单数，没有 s）**。写成 `modelscopes` 的话，tau2 会先静默注册组件并
打印 "Data directory does not exist"，直到首个任务加载时才以
`FileNotFoundError: .../tasks.json` 失败——报错位置离真正的拼写错误很远，勿被误导。

`benchmarks/tau3bench/run_parallel.py` 已内置该默认路径。
