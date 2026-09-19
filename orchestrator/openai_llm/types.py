"""Pydantic result types for openai_llm."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def msg(role: str, content: str) -> dict[str, str]:
    """Build a Chat Completions message dict."""
    return {"role": role, "content": content}


class LLMResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: str | None = None
    model: str
    latency_ms: int
    raw: Any | None = None


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMToolResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool_calls: list[ToolCall] = Field(default_factory=list)
    content: str | None = None
    model: str
    latency_ms: int
    raw: Any | None = None
