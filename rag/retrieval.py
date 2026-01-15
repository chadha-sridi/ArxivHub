from typing import List, Dict
from core.schemas import State, RuntimeContext
from langgraph.runtime import Runtime 
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny


def retrieve(state: State, runtime: Runtime[RuntimeContext]) -> Dict[str, List]:
    """
    Retrieve relevant chunks and update state.
    """
    user_id = runtime.context.user_id
    vectorstore = runtime.context.vectorstore
    settings = runtime.context.settings
    score_threshold = settings.get("retrieval_score_threshold", 0.45)
    top_K = settings.get("retrieval_top_k", 5)
    arxiv_ids = state.get("arxivIDs", [])
    query = state.get("rewrittenQuestion", "")

    # Metadata filtering
    conditions = [
        FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id))
    ]
    if arxiv_ids:
        conditions.append(
            FieldCondition(key="metadata.paper_id", match=MatchAny(any=arxiv_ids))
        )
   
    docs_with_scores = vectorstore.similarity_search_with_score(
        query,
        k=top_K,
        filter=Filter(must=conditions)
    )
   
    retrieved_arxiv_ids = []
    retrieved_chunk_ids = []
    confidence_scores = []

    for doc, score in docs_with_scores:
        if score >= score_threshold: 
            retrieved_arxiv_ids.append(doc.metadata.get("paper_id"))
            retrieved_chunk_ids.append(doc.metadata.get("_id"))
            confidence_scores.append(score)
    
    return {
        "arxivIDs": retrieved_arxiv_ids,
        "retrievedChunkIDs": retrieved_chunk_ids,
        "confidenceScores": confidence_scores,
    }
