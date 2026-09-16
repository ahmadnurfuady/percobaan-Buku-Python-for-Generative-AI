# 3.5 modules and packages/llm_package/__init__.py
from .conversation import ConversationHistory
from .config import LLMConfig

__all__ = ["ConversationHistory", "LLMConfig"]
