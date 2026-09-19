"""Sync OpenAI Chat Completions client used as the orchestrator's only LLM."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from openai import OpenAI
from pydantic import BaseModel, ValidationError

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
from .types import LLMResult, LLMToolResult, ToolCall

# Fast, cheap, supports function calling. Override with LLM_MODEL or the constructor.
DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_RETRIES = 2

_BILLING_CODES = {
    "credit_balance_exhausted",
    "insufficient_quota",
    "billing_not_active",
    "billing_hard_limit_reached",
}


def _apply_env_file(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def load_dotenv() -> None:
    """Load the nearest .env files without overriding existing environment variables."""
    seen: set[Path] = set()
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for folder in (start, *start.parents):
            path = (folder / ".env").resolve()
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            _apply_env_file(path)


def _error_code(exc: BaseException) -> str | None:
    code = getattr(exc, "code", None)
    if isinstance(code, str):
        return code
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and isinstance(err.get("code"), str):
            return err["code"]
        if isinstance(body.get("code"), str):
            return body["code"]
    return None


def _is_unsupported_param(exc: BaseException, param: str) -> bool:
    message = str(exc).lower()
    if param not in message:
        return False
    code = _error_code(exc)
    return code == "unsupported_value" or "unsupported" in message or "not supported" in message


def _is_unsupported_temperature(exc: BaseException) -> bool:
    return _is_unsupported_param(exc, "temperature")


def _map_error(exc: BaseException) -> BaseException:
    if isinstance(exc, LLMError):
        return exc
    name = type(exc).__name__.lower()
    message = str(exc) or type(exc).__name__
    status = getattr(exc, "status_code", None)
    code = _error_code(exc)

    if "timeout" in name or "timed out" in message.lower():
        return LLMTimeoutError(message)
    if code in _BILLING_CODES or status == 402:
        return BillingError(message)
    if status in {401, 403} or "authentication" in name:
        return AuthError(message)
    if status == 429 or "ratelimit" in name:
        return RateLimitError(message)
    if isinstance(exc, LLMError):
        return exc
    return LLMError(message)


def _normalize_messages(
    messages: Sequence[Mapping[str, Any] | str],
    system: str | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for item in messages:
        if isinstance(item, str):
            out.append({"role": "user", "content": item})
            continue
        role = item.get("role")
        if not role:
            raise LLMError(f"Message missing role: {item!r}")
        out.append({"role": role, "content": item.get("content")})
    return out


def _tool_choice_payload(tool_choice: str | dict[str, Any]) -> str | dict[str, Any]:
    if isinstance(tool_choice, dict):
        return tool_choice
    if tool_choice in {"auto", "required", "none"}:
        return tool_choice
    return {"type": "function", "function": {"name": tool_choice}}


def _parse_tool_calls(raw: Any) -> list[ToolCall]:
    if not raw:
        return []
    calls: list[ToolCall] = []
    for item in raw:
        if isinstance(item, Mapping):
            fn = item.get("function") or item
            call_id = str(item.get("id") or "")
            name = str((fn.get("name") if isinstance(fn, Mapping) else None) or item.get("name") or "")
            arguments = fn.get("arguments") if isinstance(fn, Mapping) else item.get("arguments")
        else:
            fn = getattr(item, "function", None)
            call_id = str(getattr(item, "id", "") or "")
            if fn is None:
                name = str(getattr(item, "name", "") or "")
                arguments = getattr(item, "arguments", "{}")
            else:
                name = str(getattr(fn, "name", "") or "")
                arguments = getattr(fn, "arguments", "{}")
        if isinstance(arguments, dict):
            args: Any = arguments
        else:
            try:
                args = json.loads(arguments or "{}")
            except json.JSONDecodeError as exc:
                raise InvalidToolJSONError(
                    f"Tool {name!r} arguments were not valid JSON: {arguments!r}"
                ) from exc
        if not isinstance(args, dict):
            raise InvalidToolJSONError(
                f"Tool {name!r} arguments must be a JSON object, got {type(args).__name__}"
            )
        calls.append(ToolCall(id=call_id, name=name, arguments=args))
    return calls


def _schema_payload(schema: dict[str, Any] | type[BaseModel]) -> dict[str, Any]:
    from .tools import _compat_schema, _schema_for_openai

    if isinstance(schema, type) and issubclass(schema, BaseModel):
        name = schema.__name__
        json_schema = _schema_for_openai(schema)
    else:
        name = str(schema.get("title") or schema.get("name") or "response")
        json_schema = _compat_schema(dict(schema))
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": json_schema,
        },
    }


def _log_call(*, model: str, latency_ms: int, tools: str) -> None:
    print(
        f"[openai_llm] model={model} latency_ms={latency_ms} tools={tools}",
        file=sys.stderr,
    )


class OpenAILLM:
    """Thin sync wrapper around OpenAI Chat Completions + tool calling."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        client: Any | None = None,
    ) -> None:
        load_dotenv()
        resolved_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise MissingAPIKeyError(
                "OPENAI_API_KEY is missing. Set it in the environment or a gitignored .env file."
            )
        self.model = model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._client = client or OpenAI(
            api_key=resolved_key,
            timeout=timeout_s,
            max_retries=max_retries,
        )

    def chat(
        self,
        messages: Sequence[Mapping[str, Any] | str],
        *,
        system: str | None = None,
        temperature: float = 0,
        max_tokens: int | None = None,
    ) -> LLMResult:
        payload = _normalize_messages(messages, system)
        resp, latency_ms = self._create(
            messages=payload,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0].message
        content = getattr(choice, "content", None)
        model = getattr(resp, "model", None) or self.model
        _log_call(model=model, latency_ms=latency_ms, tools="-")
        return LLMResult(content=content, model=model, latency_ms=latency_ms, raw=resp)

    def chat_text(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0,
        max_tokens: int | None = None,
    ) -> str:
        result = self.chat(
            [{"role": "user", "content": prompt}],
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return result.content or ""

    def chat_with_tools(
        self,
        messages: Sequence[Mapping[str, Any] | str],
        tools: Sequence[dict[str, Any]],
        *,
        system: str | None = None,
        tool_choice: str | dict[str, Any] = "required",
        temperature: float = 0,
        max_tokens: int | None = None,
        parallel_tool_calls: bool = False,
    ) -> LLMToolResult:
        payload = _normalize_messages(messages, system)
        resp, latency_ms = self._create(
            messages=payload,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=list(tools),
            tool_choice=_tool_choice_payload(tool_choice),
            parallel_tool_calls=parallel_tool_calls,
        )
        choice = resp.choices[0].message
        content = getattr(choice, "content", None)
        tool_calls = _parse_tool_calls(getattr(choice, "tool_calls", None))
        model = getattr(resp, "model", None) or self.model
        names = ",".join(call.name for call in tool_calls) if tool_calls else "-"
        _log_call(model=model, latency_ms=latency_ms, tools=names)

        required = tool_choice == "required" or (
            isinstance(tool_choice, str) and tool_choice not in {"auto", "none"}
        )
        if required and not tool_calls:
            raise EmptyToolCallError(
                f"Model {model!r} returned no tool calls (tool_choice={tool_choice!r})."
            )
        return LLMToolResult(
            tool_calls=tool_calls,
            content=content,
            model=model,
            latency_ms=latency_ms,
            raw=resp,
        )

    def chat_json(
        self,
        messages: Sequence[Mapping[str, Any] | str],
        *,
        schema: dict[str, Any] | type[BaseModel],
        system: str | None = None,
        temperature: float = 0,
        max_tokens: int | None = None,
    ) -> LLMResult:
        payload = _normalize_messages(messages, system)
        resp, latency_ms = self._create(
            messages=payload,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=_schema_payload(schema),
        )
        choice = resp.choices[0].message
        content = getattr(choice, "content", None)
        model = getattr(resp, "model", None) or self.model
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            if not content:
                raise LLMError("Structured output response had empty content.")
            try:
                parsed = schema.model_validate_json(content)
            except ValidationError as exc:
                raise LLMError(f"Structured output failed pydantic validation: {exc}") from exc
            content = parsed.model_dump_json()
        _log_call(model=model, latency_ms=latency_ms, tools="-")
        return LLMResult(content=content, model=model, latency_ms=latency_ms, raw=resp)

    def _create(self, **kwargs: Any) -> tuple[Any, int]:
        kwargs["model"] = self.model
        max_tokens = kwargs.pop("max_tokens", None)
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens
        # Chat Completions + tools on current models requires reasoning off
        # (or the Responses API). None also keeps compiler latency low.
        if kwargs.get("tools"):
            kwargs.setdefault("reasoning_effort", "none")
        started = time.perf_counter()
        try:
            try:
                resp = self._client.chat.completions.create(**kwargs)
            except Exception as exc:
                if "temperature" in kwargs and _is_unsupported_temperature(exc):
                    kwargs.pop("temperature")
                    resp = self._client.chat.completions.create(**kwargs)
                else:
                    raise
        except Exception as exc:
            raise _map_error(exc) from exc
        latency_ms = int((time.perf_counter() - started) * 1000)
        return resp, latency_ms
