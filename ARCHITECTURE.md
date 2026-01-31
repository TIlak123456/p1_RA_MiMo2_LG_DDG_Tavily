# Research Agent Architecture

A LangGraph-based research agent powered by Xiaomi MiMo model with web search capabilities.

---

## Table of Contents

1. [Overview](#overview)
2. [Tech Stack](#tech-stack)
3. [Graph Structure](#graph-structure)
4. [Components Deep Dive](#components-deep-dive)
5. [Data Flow](#data-flow)
6. [State Management](#state-management)
7. [Error Handling](#error-handling)
8. [Configuration](#configuration)

---

## Overview

This agent follows the **ReAct (Reasoning + Acting)** pattern:
1. **Reason** about what information is needed
2. **Act** by calling tools (web search)
3. **Observe** the results
4. **Repeat** or **Respond** with final answer

---

## Tech Stack

### Core Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **LLM** | Xiaomi MiMo (mimo-v2-flash) | Reasoning and response generation |
| **Framework** | LangGraph | Graph-based agent orchestration |
| **Search Tool** | Tavily Search API | Real-time web search |
| **Backup Search** | DuckDuckGo | Alternative search (available) |
| **Language** | Python 3.9+ | Runtime |
| **Config** | python-dotenv | Environment variable management |

### Design Pattern

| Pattern | Description |
|---------|-------------|
| **ReAct** | Reasoning + Acting - LLM decides when to use tools |
| **Tool-Use Agent** | Agent can call external tools (search) to gather information |
| **Stateful Graph** | Conversation history maintained across turns |

### Dependencies

```
langgraph          - Agent graph framework
langchain_openai   - LLM integration (OpenAI-compatible API)
langchain_community - Tool integrations (Tavily, DuckDuckGo)
tavily-python      - Tavily Search API client
python-dotenv      - Environment variable loading
```

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      USER INPUT                              │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     LANGGRAPH                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    AgentState                        │    │
│  │              (messages: list[BaseMessage])           │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│            ┌─────────────┴─────────────┐                    │
│            ▼                           ▼                    │
│     ┌─────────────┐             ┌─────────────┐             │
│     │    Agent    │◄───────────►│    Tools    │             │
│     │   (MiMo)    │             │  (Tavily)   │             │
│     └─────────────┘             └─────────────┘             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     AGENT RESPONSE                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Graph Structure

```
                    ┌─────────────┐
                    │    START    │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
            ┌──────►│    agent    │◄──────┐
            │       │   (MiMo)    │       │
            │       └──────┬──────┘       │
            │              │              │
            │              ▼              │
            │     ┌────────────────┐      │
            │     │ should_continue │      │
            │     │  (conditional)  │      │
            │     └───┬────────┬───┘      │
            │         │        │          │
            │    no tools   has tools     │
            │         │        │          │
            │         ▼        ▼          │
            │    ┌────────┐ ┌───────┐     │
            │    │  END   │ │ tools │─────┘
            │    └────────┘ │(Tavily)│
            │               └───────┘
            │                   │
            └───────────────────┘
              (max 3 iterations)
```

### Graph Definition (Code)

```python
workflow = StateGraph(AgentState)

# Nodes
workflow.add_node("agent", call_mimo)      # LLM reasoning
workflow.add_node("tools", tool_node)       # Tool execution

# Edges
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")

app = workflow.compile()
```

---

## Components Deep Dive

### 1. Agent Node (`call_mimo`)

The "brain" of the system. Invokes the MiMo LLM with tool-binding.

```python
def call_mimo(state: AgentState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}
```

**Responsibilities:**
- Process conversation history
- Decide if tools are needed
- Generate tool calls OR final response

### 2. Tool Node (`tool_node`)

Executes tool calls made by the agent.

```python
tools = [tavily]
tool_node = ToolNode(tools)
```

**Available Tools:**

| Tool | Description | Parameters |
|------|-------------|------------|
| `TavilySearchResults` | Web search API | `k=3` (returns top 3 results) |
| `DuckDuckGoSearchRun` | Backup search | Available but not bound |

### 3. Router (`should_continue`)

Conditional function that decides the next step.

```python
def should_continue(state: AgentState):
    last_message = state['messages'][-1]
    
    # Check for tool calls (with iteration limit)
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        if tool_call_count < MAX_TOOL_CALLS:  # MAX = 3
            return "tools"
    
    return END
```

**Routing Logic:**

| Condition | Route | Description |
|-----------|-------|-------------|
| `tool_calls` present AND count < 3 | `"tools"` | Execute the tool |
| `tool_calls` present AND count >= 3 | `END` | Force final response |
| No `tool_calls` | `END` | Return response to user |

---

## Data Flow

### Example: "What's Nvidia's stock price?"

```
Step 1: User Input
────────────────────────────────────────────────────────
messages: [HumanMessage("What's Nvidia's stock price?")]

Step 2: Agent (MiMo) - First Pass
────────────────────────────────────────────────────────
LLM decides: "I need real-time data"
Output: AIMessage with tool_calls: [{name: "tavily_search", args: {query: "Nvidia stock price today"}}]

Step 3: Router Check
────────────────────────────────────────────────────────
tool_calls present? YES
tool_call_count < 3? YES (count=1)
Route → "tools"

Step 4: Tool Execution
────────────────────────────────────────────────────────
Tavily searches: "Nvidia stock price today"
Returns: ToolMessage with search results (top 3 URLs + snippets)

Step 5: Agent (MiMo) - Second Pass
────────────────────────────────────────────────────────
LLM receives search results
Synthesizes final answer
Output: AIMessage("Nvidia (NVDA) is currently trading at $134.36...")

Step 6: Router Check
────────────────────────────────────────────────────────
tool_calls present? NO
Route → END

Step 7: Response to User
────────────────────────────────────────────────────────
"Nvidia (NVDA) is currently trading at $134.36..."
```

---

## State Management

### AgentState Schema

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
```

**Message Types in State:**

| Type | Description | Example |
|------|-------------|---------|
| `HumanMessage` | User input | "What's the weather?" |
| `AIMessage` | LLM response (may contain tool_calls) | "Let me search..." |
| `ToolMessage` | Tool execution results | Search results JSON |

**State Accumulation:**

The `operator.add` annotation means messages are **appended**, preserving full conversation history:

```python
# After one exchange:
messages = [
    HumanMessage("What's Nvidia stock?"),
    AIMessage(tool_calls=[...]),
    ToolMessage(content="Search results..."),
    AIMessage("Nvidia is trading at $134...")
]
```

---

## Error Handling

### Recursion Limit Protection

**Problem:** LLM may infinitely request tool calls.

**Solution:** Tool call counter with `MAX_TOOL_CALLS = 3`

```python
tool_call_count = 0
MAX_TOOL_CALLS = 3

def should_continue(state):
    global tool_call_count
    if tool_calls and tool_call_count < MAX_TOOL_CALLS:
        tool_call_count += 1
        return "tools"
    tool_call_count = 0  # Reset for next query
    return END
```

### Additional Safeguard

```python
result = app.invoke({"messages": messages}, config={"recursion_limit": 10})
```

---

## Configuration

### Environment Variables (`.env`)

```bash
# MiMo Model
MIMO_API_KEY=your_mimo_api_key
MIMO_BASE_URL=https://api.xiaomimimo.com/v1

# Search Tool
TAVILY_API_KEY=your_tavily_api_key

# Optional
OPENAI_API_KEY=not_needed
```

### Tunable Parameters

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `MAX_TOOL_CALLS` | `main.py` | 3 | Max tool iterations per query |
| `recursion_limit` | `app.invoke()` | 10 | LangGraph recursion limit |
| `k` | `TavilySearchResults` | 3 | Number of search results |
| `model` | `ChatOpenAI` | mimo-v2-flash | MiMo model variant |

---

## Running the Agent

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Run interactive mode
python main.py
```

### Interactive Commands

| Command | Action |
|---------|--------|
| Any text | Send query to agent |
| `quit` / `exit` / `q` | Exit the agent |

---

## Future Improvements

- [ ] Add system prompt for better tool usage control
- [ ] Implement DuckDuckGo as fallback search
- [ ] Add conversation memory persistence
- [ ] Stream responses for better UX
- [ ] Add more tools (calculator, code execution, etc.)
