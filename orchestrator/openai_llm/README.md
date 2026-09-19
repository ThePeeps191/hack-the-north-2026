# openai_llm

The only LLM client the Hack the North 2026 orchestrator should talk to.

Wraps the official OpenAI Python SDK (Chat Completions + function/tool calling). Auth is **only** `OPENAI_API_KEY` from the environment or a gitignored `.env`. Never hardcode keys.

Default model: **`gpt-5.6-luna`**. It is a current OpenAI model with tool calling, cheap enough for `$50` of API credits, and fast enough that compiler latency counts toward demo retask time. Override with `LLM_MODEL` or `OpenAILLM(model=...)`.

`temperature=0` is the public default. If the model rejects it (current reasoning models only allow the default), the client retries that call without `temperature`.

Tool calls set `reasoning_effort="none"` so Chat Completions function calling works on `gpt-5.6-luna` and stays fast.

## Install

From `orchestrator/`:

```bash
pip install -r requirements.txt
```

That installs `openai` and `pydantic`.

## Set the key

PowerShell (this session):

```powershell
$env:OPENAI_API_KEY = "sk-proj-..."
```

Or put this in the repo-root `.env` (gitignored):

```
OPENAI_API_KEY=sk-proj-...
LLM_MODEL=gpt-5.6-luna
```

`OpenAILLM()` loads `.env` from the current working directory and parents, so the repo-root file is picked up from `orchestrator/`. Do not commit `.env`.

## Import

Run Python with `orchestrator/` on `PYTHONPATH` (or `cd` into it):

```python
from openai_llm import OpenAILLM, tool_from_pydantic, tools_from_definitions
```

From the repo root:

```python
from orchestrator.openai_llm import OpenAILLM
```

## Examples

### 1. One-shot text

```python
from openai_llm import OpenAILLM

llm = OpenAILLM()  # OPENAI_API_KEY + optional LLM_MODEL
print(llm.chat_text("ping", system="Reply with a single word."))
```

### 2. Chat messages

```python
from openai_llm import OpenAILLM, msg

llm = OpenAILLM()
result = llm.chat(
    [msg("user", "Summarize: pick up the red cup")],
    system="You compile robot instructions.",
    temperature=0,
)
print(result.content, result.latency_ms, result.model)
```

### 3. Forced tool calling (compiler path, <10 lines)

```python
from pydantic import BaseModel, Field
from openai_llm import OpenAILLM, tool_from_pydantic

class ClarifyArgs(BaseModel):
    question: str = Field(description="What to ask the operator")

class SubmitTaskSpecArgs(BaseModel):
    # Compiler owns the real TaskSpec; this package stays generic.
    spec: dict

llm = OpenAILLM()
tools = [
    tool_from_pydantic("clarify", "Ask a clarifying question", ClarifyArgs),
    tool_from_pydantic("submit_task_spec", "Submit the compiled task", SubmitTaskSpecArgs),
]
result = llm.chat_with_tools(
    [{"role": "user", "content": "pick up the cup"}],
    tools,
    system="Compile the instruction. Call submit_task_spec or clarify.",
    tool_choice="required",  # or "auto", or "clarify"
)
call = result.tool_calls[0]
print(call.name, call.arguments, result.latency_ms)
```

Equivalent with `tools_from_definitions`:

```python
tools = tools_from_definitions([
    ("clarify", "Ask a clarifying question", ClarifyArgs),
    ("submit_task_spec", "Submit the compiled task", SubmitTaskSpecArgs),
])
```

Structured JSON without tools:

```python
result = llm.chat_json(
    [{"role": "user", "content": "name the object"}],
    schema=ClarifyArgs,
)
```

## Smoke test

```bash
cd orchestrator
python -m openai_llm
```

If `OPENAI_API_KEY` is missing it prints `skip: ...` and exits 0. Otherwise it runs `chat_text("ping")` and a forced `clarify` tool call.

Each live call prints one line on stderr:

```
[openai_llm] model=gpt-5.6-luna latency_ms=412 tools=clarify
```

## Errors

| Exception | When |
|---|---|
| `MissingAPIKeyError` | No `OPENAI_API_KEY` in env or `.env` |
| `AuthError` | 401 / 403 |
| `BillingError` | credits exhausted / insufficient quota |
| `RateLimitError` | 429 rate limit |
| `LLMTimeoutError` | client timeout (~30s default) |
| `InvalidToolJSONError` | tool arguments are not a JSON object |
| `EmptyToolCallError` | `tool_choice="required"` but no tool call |

The OpenAI SDK retries transient 408/429/5xx (`max_retries=2` by default). Auth and billing failures are not swallowed.

## Constructor

```python
OpenAILLM(
    model=None,       # else LLM_MODEL else gpt-5.6-luna
    api_key=None,     # else OPENAI_API_KEY
    timeout_s=30.0,
    max_retries=2,
)
```
