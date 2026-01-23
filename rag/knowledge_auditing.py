import logging
from config import llm
from core.schemas import State, CollectiveAudit
logger = logging.getLogger(__name__)

async def audit_collective_knowledge(state: State):
    logger.info("Auditing collective knowledge for gaps...")
    
    docs = state.get("retrievedDocs", [])
    question = state.get("rewrittenQuestion") or state.get("originalQuestion")

    if not docs:
        logger.warning("No relevant documents found in local library.")
        return {"relevancePassed": False, "unanswered": question}

    # Concatenate the relevant documents
    full_context = "\n\n".join([f"Doc: {d.page_content}" for d in docs])
    
    auditor_llm = llm.with_structured_output(CollectiveAudit)
    
    prompt = f"""
    Does this set of information contain the answer to ALL aspects of the question?
    QUESTION: {question}
    CONTEXT: {full_context}
    
    If any aspect is missing, formulate a concise question to find that specific info.
    """
    
    report = await auditor_llm.ainvoke(prompt)
    
    return {
        "relevancePassed": report.relevance_passed,
        "unanswered": report.unanswered_aspect if not report.relevance_passed else ""
    }