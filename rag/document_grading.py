import asyncio
import logging
from config import llm
from core.schemas import State, DocRelevance
logger = logging.getLogger(__name__)

async def grade_docs(state: State):
    logger.info("Launching parallel workers for individual document grading...")
    
    docs = state.get("retrievedDocs", [])
    scores = state.get("confidenceScores", []) # Assumes scores are in state
    question = state.get("rewrittenQuestion") or state.get("originalQuestion")
    relevance_threshold = 0.8 # threshold above which we automatically consider the document as relevant 

    if not docs:
        return {"relevancePassed": False}

    grader_llm = llm.with_structured_output(DocRelevance)

    async def check_doc(doc, score):
        # If score is high, we don't call the LLM
        if score >= relevance_threshold:
            return doc, DocRelevance(grade="relevant", reasoning="Bypassed: High confidence score.")
        
        # Else run the LLM grader
        prompt = f"Question: {question}\n\nDocument: {doc.page_content}"
        result = await grader_llm.ainvoke(prompt)
        return doc, result

    tasks = [check_doc(d, s) for d, s in zip(docs, scores)]
    results = await asyncio.gather(*tasks)

    # Filter out only those the LLM (or the bypass) deemed irrelevant
    filtered_docs = [
        doc for doc, report in results 
        if report.grade != "completely irrelevant"
    ]
    
    return {"retrievedDocs": filtered_docs}