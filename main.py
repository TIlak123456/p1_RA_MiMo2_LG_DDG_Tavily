import os
import operator
from typing import Annotated, TypedDict
from dotenv import load_dotenv

# 1. THE BRAIN & LOGIC IMPORTS
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# 2. THE HANDS (TOOLS)
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

# --- INITIALIZATION ---
# This sets up your MiMo model
llm = ChatOpenAI(
    model="mimo-v2-flash", 
    api_key=os.getenv("MIMO_API_KEY"),
    base_url=os.getenv("MIMO_BASE_URL")
)

tavily = TavilySearchResults(k=3)
ddg = DuckDuckGoSearchRun()

llm_with_tools = llm.bind_tools([tavily])

# --- STEP 3: THE STATE ---
class AgentState(TypedDict):
    # This is the 'shared notebook' that keeps history
    messages: Annotated[list[BaseMessage], operator.add]

# --- STEP 4: THE NODES ---
def call_mimo(state: AgentState):
    """This function represents the 'Thinking' node."""
    messages = state['messages']
    # We tell MiMo to look at the messages and decide what to say
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Tool execution node
tools = [tavily]
tool_node = ToolNode(tools)

# Track tool usage to prevent infinite loops
tool_call_count = 0
MAX_TOOL_CALLS = 3

# Conditional function to check if we should use tools
def should_continue(state: AgentState):
    """Decide whether to use tools or end."""
    global tool_call_count
    last_message = state['messages'][-1]
    
    # If the LLM made a tool call, route to tools (but limit iterations)
    if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        if len(last_message.tool_calls) > 0 and tool_call_count < MAX_TOOL_CALLS:
            tool_call_count += 1
            return "tools"
    
    # Reset counter for next query and end
    tool_call_count = 0
    return END

# --- STEP 5: THE GRAPH (FLOW) ---
workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("agent", call_mimo)
workflow.add_node("tools", tool_node)

# Define the flow
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")  # After tools, go back to agent

# 'Compile' turns the map into an actual application
app = workflow.compile()

# --- STEP 6: EXECUTION ---
if __name__ == "__main__":
    print("--- Research Agent (type 'quit' to exit) ---\n")
    
    # Keep conversation history for multi-turn conversations
    messages = []
    
    while True:
        user_text = input("You: ").strip()
        
        if user_text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_text:
            continue
        
        # Add user message to history
        messages.append(HumanMessage(content=user_text))
        
        # Run the agent with recursion limit
        result = app.invoke({"messages": messages}, config={"recursion_limit": 10})
        
        # Update messages with full conversation (includes tool calls and responses)
        messages = result["messages"]
        
        # Print the agent's final response
        print(f"\nAgent: {messages[-1].content}\n")
