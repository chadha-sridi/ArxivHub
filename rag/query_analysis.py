import os
from config import llm
from core.schemas import State, QueryAnalysis
from core.prompts import get_query_analysis_prompt
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

def analyze_query(state: State) -> dict:
    last_user_msg = state["messages"][-1].content
    summary = state.get("conversationSummary", "")

    context = f"""
    Conversation summary:
    {summary}

    User question:
    {last_user_msg}
    """.strip()

    llm_structured = (
        llm
        .with_config(temperature=0.1)
        .with_structured_output(QueryAnalysis)
    )

    analysis: QueryAnalysis = llm_structured.invoke([
        SystemMessage(content=get_query_analysis_prompt()),
        HumanMessage(content=context)
    ])

    # Case 1: question is NOT clear → ask for clarification
    if not analysis.is_clear:
        return {
            "questionIsClear": False,
            "messages": [AIMessage(content=analysis.clarification_needed)]
        }

    # Case 2: question is clear → update state
    metadata_present = any(
        getattr(analysis.metadataHints, field)
        for field in analysis.metadataHints.model_fields
    )
    rewritten = analysis.rewrittenQuestion.strip() or last_user_msg
    return {
        "intent": analysis.intent,
        "questionIsClear": True,
        "rewrittenQuestion": rewritten,
        "paperScope": analysis.paperScope,
        "metadataHintPresent": metadata_present,
        "metadataHints": analysis.metadataHints,
        "originalQuestion": state.get("originalQuestion") or last_user_msg
    }

