# Research Agent Architecture

## Graph Structure

```
┌─────────┐
│  START  │
└────┬────┘
     │
     ▼
┌─────────┐
│  agent  │◄─────────────┐
│ (MiMo)  │              │
└────┬────┘              │
     │                   │
     ▼                   │
┌─────────────┐          │
│should_continue│         │
│ (conditional)│         │
└────┬────┬───┘          │
     │    │              │
     │    │ tool_calls   │
     │    ▼              │
     │  ┌───────┐        │
     │  │ tools │────────┘
     │  │(Tavily)│
     │  └───────┘
     │
     │ no tool_calls
     ▼
┌─────────┐
│   END   │
└─────────┘
```

## The 3 Layers

| Layer | Node | Purpose |
|-------|------|---------|
| **1. Agent** | `call_mimo` | LLM thinks and decides if it needs to search |
| **2. Router** | `should_continue` | Checks if LLM requested a tool call |
| **3. Tools** | `tool_node` | Executes Tavily search, returns results |

## Flow Example

1. **User asks:** "What's Nvidia's stock price?"
2. **Agent (MiMo):** Decides it needs real-time data → makes tool call
3. **Router:** Sees `tool_calls` → routes to `tools`
4. **Tools:** Runs Tavily search → returns results
5. **Agent (MiMo):** Receives search results → formulates final answer
6. **Router:** No more tool calls → routes to `END`

The loop (`tools → agent`) allows the agent to make multiple searches if needed before giving a final answer.

## Tech Stack

- **LLM:** Xiaomi MiMo (mimo-v2-flash)
- **Framework:** LangGraph
- **Search Tool:** Tavily Search API
- **Language:** Python 3.9+
