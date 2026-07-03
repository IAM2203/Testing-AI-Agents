# Financial & Math Agent — From Manual ReAct to a Stateful Framework

A small financial/math tool-using agent, built three times on purpose. Each version is a deliberate step up in abstraction, meant to actually understand what a framework like LangGraph is doing under the hood before relying on it.

> Code and comments are in Spanish (my working language); this README is in English for accessibility.

## Why three versions

Most tutorials start with `create_agent(...)` and skip the part where you understand *why* the agent behaves the way it does. I wanted to build the tool-calling loop by hand first, then swap in native function calling, then a full framework — so that by the time I reached LangGraph, every abstraction it provides (message loop, tool dispatch, memory persistence) mapped to something I'd already implemented myself and hit the limits of.

| # | Version | What it demonstrates |
|---|---------|----------------------|
| 1 | [`react-manual/`](./agente_react.py) | The ReAct loop (Thought → Action → Observation) implemented from raw text parsing, no framework, no native tool support |
| 2 | [`native-function-calling/`](./auto_agent.py) | Same tools, but delegating the reasoning loop to the model's built-in function calling |
| 3 | [`langgraph-agent/`](./framework.py) | A stateful agent with short-term memory (per-conversation) and long-term memory (across conversations) via LangGraph + SQLite |

## 1. Manual ReAct Loop

The agent's "reasoning" is a plain-text scratchpad reprinted to the model every turn. The model must respond in an exact `Thought / Action / Action Input` or `Thought / Final Answer` format, which is parsed with string splitting — fragile by design, since there's no structured output from the API at this stage.

**What this taught me:** why `stop_sequences` matters (to stop the model from hallucinating its own `Observation:`), why single-tool-per-step is a deliberate constraint, and how much scaffolding (retries by error type, format-recovery fallback) a hand-rolled loop needs just to be reliable.

## 2. Native Function Calling

Same four tools, but passed directly to `types.GenerateContentConfig(tools=[...])`. The SDK infers each tool's schema from its Python signature and docstring, and handles calling, waiting for results, and deciding the next step internally — no scratchpad, no manual parsing.

**What this taught me:** how much the ReAct loop's plumbing (dispatch table, format parsing, stop sequences) is exactly what native tool calling replaces, and where the trade-off is — this version has no retry logic and no persistence, since the whole call is a single request/response.

## 3. LangGraph Agent

Built with `langchain.agents.create_agent`, using Groq (`openai/gpt-oss-120b`) as the LLM. Adds two things the earlier versions don't have:

- **Short-term memory** (`SqliteSaver`): conversation state persisted per thread, so a conversation can be closed and resumed.
- **Long-term memory** (`SqliteStore`): durable facts about the user, saved via a `recordar` tool the agent can call explicitly, plus an automatic extraction pass (`extraer_y_guardar`) after every turn so nothing depends on the agent remembering to save it.

**What this taught me:** the real difference between a stateless tool-caller and an agent designed to be talked to across sessions, and why separating the two storage backends (thread state vs. user facts) is the right split rather than one shared store.

## Tools (shared across all three versions)

| Tool | Purpose |
|------|---------|
| `calculadora` | Restricted `eval` for arithmetic (whitelisted `math` functions only, no builtins) |
| `hora_actual` | Current date/time |
| `tipo_cambio` | USD/MXN exchange rate via Twelve Data |
| `precio_accion` | Stock price in USD via Twelve Data |
| `recordar` / `buscar_memoria` | *(LangGraph version only)* Save/retrieve durable facts about the user |

## Setup

```bash
pip install -r requirements.txt  # per-folder, dependencies differ by version
```

Environment variables needed:

| Variable | Used by |
|----------|---------|
| `GEMINI_API_KEY` | Versions 1 and 2 (Gemini via `google-genai`) |
| `GROQ_API_KEY` | Version 3 (Groq via `langchain-groq`) |
| `TWELVEDATA_API_KEY` | All versions (exchange rate + stock price tools) |

## Roadmap

- [ ] Add more tools / data sources (additional market data, macro indicators)
- [ ] Fold this agent into the larger agent I'm building for an investment fund research project, which will use the same short/long-term memory pattern with live financial data

## Background

Built while studying Engineering Physics (quant finance track) — part of a broader effort to go from first-principles ReAct implementations to production-style agent frameworks, rather than starting from the framework and treating the loop as a black box.

## License

MIT
