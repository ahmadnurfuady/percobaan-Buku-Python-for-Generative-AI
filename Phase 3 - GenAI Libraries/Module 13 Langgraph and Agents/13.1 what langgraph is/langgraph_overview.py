# 13.1 What LangGraph Is (Halaman 143-144)
"""
LangGraph is a library for building stateful, multi-step LLM applications modelled as directed graphs.
Each node is a function that reads and writes to a shared state dict.
Edges define control flow - including conditional branches and loops.

LangGraph is the right tool when:
- The number of LLM calls is not fixed in advance
- The agent must decide which tool to call next based on results
- You need persistent state between steps (memory, intermediate results)
- You want human-in-the-loop checkpoints
"""

def print_langgraph_overview():
    print("LangGraph Core: StateGraph + TypedDict State + Nodes + Conditional Edges")

if __name__ == "__main__":
    print_langgraph_overview()
