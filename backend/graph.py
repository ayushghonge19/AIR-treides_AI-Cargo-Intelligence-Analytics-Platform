from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from backend.nodes import agent, run_tools, report_writer, should_continue
from backend.state import AgentState

# ---------------------------------------------------------------------------
# Build the cyclic StateGraph workflow
# ---------------------------------------------------------------------------

workflow = StateGraph(AgentState) 
#radhe radhe bol radhe radhe bol , barsane me dol , radhe radhe
workflow.add_node("agent", agent)
workflow.add_node("tools", run_tools)
workflow.add_node("report_writer", report_writer)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "report_writer": "report_writer",
    },
)

workflow.add_edge("tools", "agent")
workflow.add_edge("report_writer", END)

# ---------------------------------------------------------------------------
# Compile with in-memory checkpointer for multi-turn conversation persistence
# ---------------------------------------------------------------------------

memory = MemorySaver()
app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["tools", "report_writer"],
)
