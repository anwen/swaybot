__version__ = "0.1.0"

from .agent import Agent
from .brain import Brain, EchoBrain
from .memory import Memory, MemoryStore
from .reflection import Reflection, Reflector, reflection_to_memory

try:
    from .llm_brain import LLMBrain
except ImportError:
    LLMBrain = None  # type: ignore[misc, assignment]

__all__ = [
    "Agent",
    "Brain",
    "EchoBrain",
    "Memory",
    "MemoryStore",
    "Reflection",
    "Reflector",
    "reflection_to_memory",
    "LLMBrain",
]
