"""AgentDojo integration for SwayBot.

Wraps SwayBot's Brain/ToolRegistry to implement AgentDojo's
``BasePipelineElement`` interface for prompt-injection robustness
evaluation across banking, workspace, slack, and travel suites.

Usage::

    python -m benchmarks.agentdojo --suite banking --model gpt-4o
"""
