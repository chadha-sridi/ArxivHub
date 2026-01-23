from core.schemas import State, RuntimeContext
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from rag import (
    summarize_conversation_history,
    analyze_query,
    fuzzy_match_papers,
    retrieve, 
    grade_docs, 
    audit_collective_knowledge,
    tavily_search,
    generate, 
    handle_general_talk
)

# Initialize checkpointer
checkpointer = InMemorySaver()

# Routing functions
def dispatch_query(state: State):
    """Decide on general talk vs RAG"""
    if not state.get("questionIsClear", True):
        return "clarify"

    if state["intent"] == "casual":
        return "casual"
    return "research"

def route_by_knowledge_sufficiency(state: State):
    if not state["relevancePassed"] and state["unanswered"] :
        return "tavily"
    return "generation"

# Build workflow graph
graph_builder = StateGraph(State, context_schema=RuntimeContext)

# Nodes
graph_builder.add_node("summarize_conv", summarize_conversation_history)
graph_builder.add_node("analyze_query", analyze_query)
graph_builder.add_node("scope_context", fuzzy_match_papers)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("grade_docs", grade_docs)
graph_builder.add_node("audit_collective_knowledge", audit_collective_knowledge)
graph_builder.add_node("tavily_search", tavily_search)
graph_builder.add_node("generate", generate)
graph_builder.add_node("handle_general_talk", handle_general_talk)

# Edges
graph_builder.add_edge(START, "summarize_conv")
graph_builder.add_edge("summarize_conv", "analyze_query")
graph_builder.add_conditional_edges(
    "analyze_query", # route to RAG or direct generation according to the user query 
    dispatch_query,
    {   
        "clarify": END,
        "casual": "handle_general_talk",
        "research": "scope_context" # RAG path
    }
)
graph_builder.add_edge("scope_context", "retrieve")
graph_builder.add_edge("retrieve", "grade_docs")
graph_builder.add_edge("grade_docs", "audit_collective_knowledge")
graph_builder.add_conditional_edges(
    "audit_collective_knowledge", # Relevance passed ==> generation, else ==> Tavily
    route_by_knowledge_sufficiency,
    {
        "tavily": "tavily_search",
        "generation": "generate" 
    }
)
graph_builder.add_edge("tavily_search", "generate")
graph_builder.add_edge("generate", END)
graph_builder.add_edge("handle_general_talk", END)

# Compile graph
workflow = graph_builder.compile(
    checkpointer=checkpointer
)

img = workflow.get_graph(xray=True).draw_mermaid_png()
with open("assets/workflow.png", "wb") as f:
    f.write(img)

