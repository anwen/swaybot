"""tau3-bench integration for SwayBot.

Wraps SwayBot's Agent/Brain/ToolRegistry to implement tau2's
``HalfDuplexAgent`` interface so that SwayBot can be evaluated on
multi-turn domain tasks (airline, retail, telecom, banking_knowledge, mock).

Usage::

    python -m benchmarks.tau3bench --domain mock --llm-agent gpt-4o
"""
