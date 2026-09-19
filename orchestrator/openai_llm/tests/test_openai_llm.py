"""Unit tests for openai_llm. No live network — fake the OpenAI client."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import BaseModel, Field

# Allow `python -m unittest` from orchestrator/ or repo root.
_ORCH = Path(__file__).resolve().parents[2]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))


class ClarifyArgs(BaseModel):
    question: str = Field(description="What to ask the operator")


class DummySpec(BaseModel):
    title: str
    steps: list[str]


def _message(*, content: str | None = None, tool_calls: list | None = None):
    return SimpleNamespace(content=content, tool_calls=tool_calls, parsed=None)


def _completion(*, content: str | None = None, tool_calls: list | None = None, model: str = "gpt-5.6-luna"):
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=_message(content=content, tool_calls=tool_calls))],
    )


def _tool_call(name: str, arguments: str, call_id: str = "call_1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    def parse(self, **kwargs):
        return self.create(**kwargs)


class FakeClient:
    def __init__(self, response):
        self.chat = SimpleNamespace(completions=FakeCompletions(response))


class TestTools(unittest.TestCase):
    def test_tool_from_pydantic_is_openai_tools_entry(self):
        from openai_llm import tool_from_pydantic

        tool = tool_from_pydantic("clarify", "Ask a clarifying question", ClarifyArgs)

        self.assertEqual(tool["type"], "function")
        self.assertEqual(tool["function"]["name"], "clarify")
        self.assertEqual(tool["function"]["description"], "Ask a clarifying question")
        params = tool["function"]["parameters"]
        self.assertEqual(params["type"], "object")
        self.assertIn("question", params["properties"])
        self.assertIn("question", params["required"])

    def test_tools_from_definitions_accepts_tuples_and_ready_dicts(self):
        from openai_llm import tool_from_pydantic, tools_from_definitions

        ready = tool_from_pydantic("clarify", "Ask", ClarifyArgs)
        tools = tools_from_definitions(
            [
                ready,
                ("submit_task_spec", "Submit a compiled task", DummySpec),
                {
                    "name": "ping",
                    "description": "Health check",
                    "parameters": {"type": "object", "properties": {}},
                },
            ]
        )
        names = [t["function"]["name"] for t in tools]
        self.assertEqual(names, ["clarify", "submit_task_spec", "ping"])


class TestMissingKey(unittest.TestCase):
    def test_empty_api_key_raises_missing_key(self):
        from openai_llm import OpenAILLM
        from openai_llm.errors import MissingAPIKeyError

        with self.assertRaises(MissingAPIKeyError):
            OpenAILLM(api_key="")


class TestChat(unittest.TestCase):
    def test_chat_returns_content_model_and_latency(self):
        from openai_llm import OpenAILLM

        client = FakeClient(_completion(content="pong"))
        llm = OpenAILLM(api_key="sk-test", client=client)
        result = llm.chat([{"role": "user", "content": "ping"}])

        self.assertEqual(result.content, "pong")
        self.assertEqual(result.model, "gpt-5.6-luna")
        self.assertIsInstance(result.latency_ms, int)
        self.assertGreaterEqual(result.latency_ms, 0)
        sent = client.chat.completions.calls[0]
        self.assertEqual(sent["messages"][0]["role"], "user")
        self.assertEqual(sent["temperature"], 0)

    def test_chat_prepends_system(self):
        from openai_llm import OpenAILLM

        client = FakeClient(_completion(content="ok"))
        llm = OpenAILLM(api_key="sk-test", client=client)
        llm.chat([{"role": "user", "content": "hi"}], system="You are a robot compiler.")
        sent = client.chat.completions.calls[0]["messages"]
        self.assertEqual(sent[0], {"role": "system", "content": "You are a robot compiler."})
        self.assertEqual(sent[1]["content"], "hi")

    def test_chat_text_returns_string(self):
        from openai_llm import OpenAILLM

        llm = OpenAILLM(api_key="sk-test", client=FakeClient(_completion(content="pong")))
        self.assertEqual(llm.chat_text("ping"), "pong")


class TestChatWithTools(unittest.TestCase):
    def test_parses_forced_tool_call_arguments(self):
        from openai_llm import OpenAILLM, tool_from_pydantic

        raw = _completion(
            content=None,
            tool_calls=[_tool_call("clarify", '{"question": "Which room?"}'),],
        )
        client = FakeClient(raw)
        llm = OpenAILLM(api_key="sk-test", client=client)
        tools = [tool_from_pydantic("clarify", "Ask", ClarifyArgs)]
        result = llm.chat_with_tools(
            [{"role": "user", "content": "go there"}],
            tools,
            tool_choice="required",
        )

        self.assertEqual(len(result.tool_calls), 1)
        call = result.tool_calls[0]
        self.assertEqual(call.id, "call_1")
        self.assertEqual(call.name, "clarify")
        self.assertEqual(call.arguments, {"question": "Which room?"})
        sent = client.chat.completions.calls[0]
        self.assertEqual(sent["tool_choice"], "required")
        self.assertIs(sent["parallel_tool_calls"], False)
        self.assertEqual(sent["reasoning_effort"], "none")

    def test_specific_tool_choice_name_becomes_openai_function_choice(self):
        from openai_llm import OpenAILLM, tool_from_pydantic

        client = FakeClient(
            _completion(tool_calls=[_tool_call("clarify", '{"question": "x"}')])
        )
        llm = OpenAILLM(api_key="sk-test", client=client)
        llm.chat_with_tools(
            [{"role": "user", "content": "??"}],
            [tool_from_pydantic("clarify", "Ask", ClarifyArgs)],
            tool_choice="clarify",
        )
        self.assertEqual(
            client.chat.completions.calls[0]["tool_choice"],
            {"type": "function", "function": {"name": "clarify"}},
        )

    def test_required_with_no_tool_calls_raises(self):
        from openai_llm import OpenAILLM
        from openai_llm.errors import EmptyToolCallError

        llm = OpenAILLM(api_key="sk-test", client=FakeClient(_completion(content="no tools")))
        with self.assertRaises(EmptyToolCallError):
            llm.chat_with_tools(
                [{"role": "user", "content": "hi"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "clarify",
                            "description": "Ask",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                tool_choice="required",
            )

    def test_invalid_tool_json_raises(self):
        from openai_llm import OpenAILLM
        from openai_llm.errors import InvalidToolJSONError

        llm = OpenAILLM(
            api_key="sk-test",
            client=FakeClient(_completion(tool_calls=[_tool_call("clarify", "{not json")])),
        )
        with self.assertRaises(InvalidToolJSONError):
            llm.chat_with_tools(
                [{"role": "user", "content": "hi"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "clarify",
                            "description": "Ask",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                tool_choice="required",
            )


class TestChatJson(unittest.TestCase):
    def test_validates_pydantic_schema(self):
        from openai_llm import OpenAILLM

        payload = {"title": "Pick", "steps": ["go to table"]}
        client = FakeClient(_completion(content=json.dumps(payload)))
        llm = OpenAILLM(api_key="sk-test", client=client)
        result = llm.chat_json([{"role": "user", "content": "pick it up"}], schema=DummySpec)
        self.assertIsNotNone(result.content)
        parsed = json.loads(result.content)
        self.assertEqual(parsed["title"], "Pick")
        sent = client.chat.completions.calls[0]
        self.assertEqual(sent["response_format"]["type"], "json_schema")
        self.assertEqual(sent["response_format"]["json_schema"]["name"], "DummySpec")


class TestErrorMapping(unittest.TestCase):
    def test_401_maps_to_auth_error(self):
        from openai_llm import OpenAILLM
        from openai_llm.errors import AuthError

        err = _status_error(401, "invalid_api_key", "Incorrect API key provided")
        llm = OpenAILLM(api_key="sk-test", client=FakeClient(err))
        with self.assertRaises(AuthError):
            llm.chat_text("ping")

    def test_credit_exhausted_maps_to_billing_error(self):
        from openai_llm import OpenAILLM
        from openai_llm.errors import BillingError

        err = _status_error(429, "credit_balance_exhausted", "No credits")
        llm = OpenAILLM(api_key="sk-test", client=FakeClient(err))
        with self.assertRaises(BillingError):
            llm.chat_text("ping")

    def test_429_maps_to_rate_limit(self):
        from openai_llm import OpenAILLM
        from openai_llm.errors import RateLimitError

        err = _status_error(429, "rate_limit_exceeded", "Slow down")
        llm = OpenAILLM(api_key="sk-test", client=FakeClient(err))
        with self.assertRaises(RateLimitError):
            llm.chat_text("ping")

    def test_timeout_maps_to_llm_timeout(self):
        from openai_llm import OpenAILLM
        from openai_llm.errors import LLMTimeoutError

        llm = OpenAILLM(api_key="sk-test", client=FakeClient(TimeoutBoom("timed out")))
        with self.assertRaises(LLMTimeoutError):
            llm.chat_text("ping")


class TimeoutBoom(Exception):
    """Looks like an SDK timeout to the mapper."""


class StatusBoom(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.body = {"error": {"code": code, "message": message}}


def _status_error(status_code: int, code: str, message: str) -> StatusBoom:
    return StatusBoom(status_code, code, message)


class SeqCompletions(FakeCompletions):
    def __init__(self, responses: list):
        super().__init__(None)
        self.queue = list(responses)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class SeqClient:
    def __init__(self, responses: list):
        self.chat = SimpleNamespace(completions=SeqCompletions(responses))


class TestDefaultModel(unittest.TestCase):
    def test_model_env_overrides_default(self):
        from openai_llm import OpenAILLM

        with patch.dict("os.environ", {"LLM_MODEL": "gpt-5.6-terra"}):
            llm = OpenAILLM(api_key="sk-test", client=FakeClient(_completion(content="x")))
            self.assertEqual(llm.model, "gpt-5.6-terra")

    def test_retries_without_temperature_when_model_rejects_it(self):
        from openai_llm import OpenAILLM

        err = _status_error(
            400,
            "unsupported_value",
            "Unsupported value: 'temperature' does not support 0 with this model. Only the default (1) value is supported.",
        )
        client = SeqClient([err, _completion(content="pong")])
        llm = OpenAILLM(api_key="sk-test", client=client)
        self.assertEqual(llm.chat_text("ping"), "pong")
        self.assertIn("temperature", client.chat.completions.calls[0])
        self.assertNotIn("temperature", client.chat.completions.calls[1])


if __name__ == "__main__":
    unittest.main()
