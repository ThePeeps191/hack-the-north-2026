"""Smoke test: `python -m openai_llm` from the orchestrator directory."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .client import OpenAILLM
from .errors import MissingAPIKeyError
from .tools import tool_from_pydantic


class ClarifyArgs(BaseModel):
    question: str = Field(description="Question to ask the operator")


def main() -> int:
    try:
        llm = OpenAILLM()
    except MissingAPIKeyError as exc:
        print(f"skip: {exc}")
        return 0

    print("--- chat_text ---")
    print(llm.chat_text("ping"))

    print("--- forced tool call (clarify) ---")
    result = llm.chat_with_tools(
        [{"role": "user", "content": "Do the thing."}],
        tools=[tool_from_pydantic("clarify", "Ask a clarifying question", ClarifyArgs)],
        system="The user request is underspecified. You must call the clarify tool.",
        tool_choice="required",
    )
    for call in result.tool_calls:
        print(f"{call.name}: {call.arguments}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
