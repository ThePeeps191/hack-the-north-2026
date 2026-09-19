"""Helpers to turn pydantic models / JSON schemas into OpenAI tools[] entries."""

from __future__ import annotations

from typing import Any, Sequence

from pydantic import BaseModel


def _schema_for_openai(model: type[BaseModel]) -> dict[str, Any]:
    return _compat_schema(model.model_json_schema())


def _compat_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Trim pydantic metadata and mark objects closed so tool-calling is reliable."""
    out = dict(schema)
    out.pop("title", None)
    out.pop("$schema", None)
    if "properties" in out or out.get("type") == "object":
        out.setdefault("type", "object")
        out.setdefault("additionalProperties", False)
        props = out.get("properties") or {}
        cleaned: dict[str, Any] = {}
        for key, value in props.items():
            cleaned[key] = _compat_schema(value) if isinstance(value, dict) else value
        out["properties"] = cleaned
        if "required" not in out:
            out["required"] = list(cleaned.keys())
    if "$defs" in out and isinstance(out["$defs"], dict):
        out["$defs"] = {
            k: _compat_schema(v) if isinstance(v, dict) else v
            for k, v in out["$defs"].items()
        }
    return out


def tool_from_pydantic(name: str, description: str, model: type[BaseModel]) -> dict[str, Any]:
    """Build one OpenAI Chat Completions `tools[]` function entry from a pydantic model."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": _schema_for_openai(model),
        },
    }


def tools_from_definitions(definitions: Sequence[Any]) -> list[dict[str, Any]]:
    """
    Normalize a mixed list into OpenAI `tools[]`.

    Each item may be:
    - a ready tools[] dict (`{"type": "function", "function": {...}}`)
    - `(name, description, pydantic_model)`
    - `{"name", "description", "model": BaseModel}`
    - `{"name", "description", "parameters": {JSON schema}}`
    """
    tools: list[dict[str, Any]] = []
    for item in definitions:
        if isinstance(item, tuple) and len(item) == 3:
            name, description, model = item
            tools.append(tool_from_pydantic(str(name), str(description), model))
            continue
        if not isinstance(item, dict):
            raise TypeError(f"Unrecognized tool definition: {item!r}")
        if item.get("type") == "function" and "function" in item:
            tools.append(item)
            continue
        name = item.get("name")
        if not name:
            raise TypeError(f"Tool definition missing name: {item!r}")
        description = str(item.get("description") or "")
        if "model" in item:
            tools.append(tool_from_pydantic(str(name), description, item["model"]))
            continue
        if "parameters" in item:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": str(name),
                        "description": description,
                        "parameters": item["parameters"],
                    },
                }
            )
            continue
        raise TypeError(f"Unrecognized tool definition: {item!r}")
    return tools
