import os
import operator
from typing import Annotated, TypedDict
from dotenv import load_dotenv

# 1. THE BRAIN & LOGIC IMPORTS
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
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

# Conditional function to check if we should use tools
def should_continue(state: AgentState):
    """Decide whether to use tools or end."""
    last_message = state['messages'][-1]
    # If the LLM made a tool call, route to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    # Otherwise, end the conversation
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
    # This is the starting message we send to the agent
    user_input = {"messages": [HumanMessage(content="What is the current price of Nvidia stock and what happened in their latest earnings call?")]}
    
    print("--- Running Agent ---")
    result = app.invoke(user_input)
    
    # Print the last message in the notebook (the agent's answer)
    print("Agent Output:", result["messages"][-1].content)
