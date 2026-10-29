# 13.5 Module 13 Exercises (Halaman 152)
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# ----------------------------------------------------------------------
# Exercise 1: Research Agent with 3 Tools (web_search, summarise, save_note)
# ----------------------------------------------------------------------
class ResearchAgentState(TypedDict):
    topic: str
    search_results: str
    summary: str
    saved_notes: list[str]

def search_tool_node(state: ResearchAgentState) -> dict:
    # Mock web search tool
    res = f"Articles for '{state['topic']}': Vector databases enable efficient sub-second similarity search."
    return {"search_results": res}

def summarise_tool_node(state: ResearchAgentState) -> dict:
    # Summarise node
    summary = f"Summary: {state['search_results'][:50]} (Synthesized)"
    return {"summary": summary}

def save_note_tool_node(state: ResearchAgentState) -> dict:
    # Append structured note
    note = f"[NOTE] Topic: {state['topic']} | {state['summary']}"
    notes = state.get("saved_notes", []) + [note]
    return {"saved_notes": notes}

def build_research_agent():
    graph = StateGraph(ResearchAgentState)
    graph.add_node("search", search_tool_node)
    graph.add_node("summarise", summarise_tool_node)
    graph.add_node("save_note", save_note_tool_node)

    graph.add_edge(START, "search")
    graph.add_edge("search", "summarise")
    graph.add_edge("summarise", "save_note")
    graph.add_edge("save_note", END)
    return graph.compile()

# ----------------------------------------------------------------------
# Exercise 2: ReAct Agent with Maximum Step Limit (8 steps)
# ----------------------------------------------------------------------
class StepLimitedReActState(TypedDict):
    step_count: int
    max_steps: int
    has_answer: bool
    status: str

def react_step_node(state: StepLimitedReActState) -> dict:
    new_step = state["step_count"] + 1
    # Simulasi: menghasilkan jawaban pada step 3 jika max_steps cukup
    has_ans = new_step >= 3
    return {"step_count": new_step, "has_answer": has_ans}

def route_step_limit(state: StepLimitedReActState) -> str:
    if state["has_answer"]:
        return "finish"
    if state["step_count"] >= state["max_steps"]:
        return "limit_reached"
    return "continue"

def finish_node(state: StepLimitedReActState) -> dict:
    return {"status": "Successfully answered."}

def limit_reached_node(state: StepLimitedReActState) -> dict:
    return {"status": "Reached step limit."}

def build_step_limited_agent():
    graph = StateGraph(StepLimitedReActState)
    graph.add_node("step", react_step_node)
    graph.add_node("finish", finish_node)
    graph.add_node("limit_reached", limit_reached_node)

    graph.add_edge(START, "step")
    graph.add_conditional_edges(
        "step",
        route_step_limit,
        {"continue": "step", "finish": "finish", "limit_reached": "limit_reached"}
    )
    graph.add_edge("finish", END)
    graph.add_edge("limit_reached", END)
    return graph.compile()

# ----------------------------------------------------------------------
# Exercise 3: PlanAndExecute Agent (Planner + Executor)
# ----------------------------------------------------------------------
class PlanAndExecuteState(TypedDict):
    task: str
    plan: list[str]
    current_step: int
    results: list[str]

def planner_node(state: PlanAndExecuteState) -> dict:
    plan = [
        "1. Retrieve relevant context",
        "2. Analyze key facts",
        "3. Synthesize final response",
    ]
    return {"plan": plan, "current_step": 0, "results": []}

def executor_node(state: PlanAndExecuteState) -> dict:
    idx = state["current_step"]
    step_text = state["plan"][idx]
    res = f"Executed {step_text}"
    return {
        "current_step": idx + 1,
        "results": state["results"] + [res],
    }

def route_plan_executor(state: PlanAndExecuteState) -> str:
    if state["current_step"] < len(state["plan"]):
        return "execute"
    return "done"

def build_plan_and_execute_agent():
    graph = StateGraph(PlanAndExecuteState)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges(
        "executor",
        route_plan_executor,
        {"execute": "executor", "done": END}
    )
    return graph.compile()

# ----------------------------------------------------------------------
# Exercise 4: Multi-Turn Memory Agent with MemorySaver
# ----------------------------------------------------------------------
class ChatMemoryState(TypedDict):
    messages: Annotated[list[str], operator.add]

def chat_node(state: ChatMemoryState) -> dict:
    return {}

def build_memory_agent():
    memory = MemorySaver()
    graph = StateGraph(ChatMemoryState)
    graph.add_node("chat", chat_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    return graph.compile(checkpointer=memory)

# ----------------------------------------------------------------------
# Demonstrasi Eksekusi
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Demo Module 13 Exercises ===\n")

    # 1. Test Exercise 1: Research Agent with 3 Tools
    print("[1] Testing Research Agent (web_search, summarise, save_note):")
    research_agent = build_research_agent()
    r_res = research_agent.invoke({
        "topic": "HNSW Algorithm",
        "search_results": "",
        "summary": "",
        "saved_notes": []
    })
    print(f" - Topic       : {r_res['topic']}")
    print(f" - Summary     : {r_res['summary']}")
    print(f" - Saved Notes : {r_res['saved_notes'][0]}\n")

    # 2. Test Exercise 2: Step-limited ReAct
    print("[2] Testing Step-limited ReAct Agent (Max 8 Steps):")
    step_agent = build_step_limited_agent()
    # Test normal finish (3 steps)
    res_normal = step_agent.invoke({"step_count": 0, "max_steps": 8, "has_answer": False, "status": ""})
    print(f" - With max_steps=8: {res_normal['status']} (Finished at step {res_normal['step_count']})")
    # Test step limit hit (max_steps=2)
    res_limit = step_agent.invoke({"step_count": 0, "max_steps": 2, "has_answer": False, "status": ""})
    print(f" - With max_steps=2: {res_limit['status']} (Stopped at step {res_limit['step_count']})\n")

    # 3. Test Exercise 3: PlanAndExecute Agent
    print("[3] Testing PlanAndExecute Agent:")
    plan_agent = build_plan_and_execute_agent()
    pe_res = plan_agent.invoke({"task": "Generate RAG Report", "plan": [], "current_step": 0, "results": []})
    print(f" - Plan length: {len(pe_res['plan'])} steps")
    for r in pe_res["results"]:
        print(f"   * {r}")
    print()

    # 4. Test Exercise 4: Multi-Turn MemorySaver
    print("[4] Testing Multi-Turn Memory Across Invocations:")
    memory_agent = build_memory_agent()
    config = {"configurable": {"thread_id": "thread-user-42"}}

    # Turn 1
    memory_agent.invoke({"messages": ["Halo, nama saya Budi."]}, config=config)
    # Turn 2
    state_turn2 = memory_agent.invoke({"messages": ["Saya suka AI."]}, config=config)
    print(f" - Stored conversation history ({len(state_turn2['messages'])} turns):")
    for msg in state_turn2["messages"]:
        print(f"   * {msg}")

    print("\nSemua latihan Module 13 berhasil dijalankan.")
