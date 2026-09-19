"""OpenAI LLM client for the Hack the North 2026 orchestrator."""

from .client import DEFAULT_MODEL, OpenAILLM, load_dotenv
from .errors import (
    AuthError,
    BillingError,
    EmptyToolCallError,
    InvalidToolJSONError,
    LLMError,
    LLMTimeoutError,
    MissingAPIKeyError,
    RateLimitError,
)
from .tools import tool_from_pydantic, tools_from_definitions
from .types import LLMResult, LLMToolResult, ToolCall, msg

__all__ = [
    "DEFAULT_MODEL",
    "OpenAILLM",
    "load_dotenv",
    "LLMResult",
    "LLMToolResult",
    "ToolCall",
    "msg",
    "tool_from_pydantic",
    "tools_from_definitions",
    "LLMError",
    "MissingAPIKeyError",
    "AuthError",
    "BillingError",
    "RateLimitError",
    "LLMTimeoutError",
    "InvalidToolJSONError",
    "EmptyToolCallError",
]
