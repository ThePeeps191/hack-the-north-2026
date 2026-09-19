"""Thin typed errors for the OpenAI LLM client."""

from __future__ import annotations


class LLMError(Exception):
    """Base error for openai_llm."""


class MissingAPIKeyError(LLMError):
    """OPENAI_API_KEY is unset, empty, or not found in .env."""


class AuthError(LLMError):
    """API key rejected (401/403)."""


class BillingError(LLMError):
    """Credits exhausted, quota exceeded, or billing inactive."""


class RateLimitError(LLMError):
    """HTTP 429 rate limit (not a billing/credit failure)."""


class LLMTimeoutError(LLMError):
    """Request exceeded the client timeout or the connection timed out."""


class InvalidToolJSONError(LLMError):
    """A tool call's arguments were not valid JSON / not an object."""


class EmptyToolCallError(LLMError):
    """tool_choice was required (or a specific tool) but the model returned none."""
