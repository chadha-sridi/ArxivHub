import re
from core.schemas import State, RuntimeContext
from rapidfuzz import fuzz
from statistics import mean
from datetime import datetime
from typing import List, Dict, Set
from langgraph.runtime import Runtime 

def extract_year(published):
    m = re.search(r"\d{4}", published)
    return int(m.group(0)) if m else None

def normalize_query_years(q_years: List[str], recent_window: int = 3) -> Set[int]:
    current_year = datetime.now().year
    normalized_q_years = set()
    for y_str in q_years:
        # Explicit year
        years = re.findall(r"\b\d{4}\b", y_str)
        for yr in years:
            normalized_q_years.add(int(yr))
            
        # "recent", "recent years", etc.
        if "recent" in y_str or "last" in y_str:
            normalized_q_years.update(
                range(current_year - recent_window + 1, current_year + 1)
            )
    return normalized_q_years 

def fuzzy_match_papers(state: State, runtime: Runtime[RuntimeContext]) -> Dict[str, List[str]]:
    """
    Select top-N paper IDs based on fuzzy matching between query hints and paper metadata.
    Args:
        LangGraph state to retrieve query_hints: {'titles': [...], 'authors': [...], 'topics': [...], 'publicationYears': [...]}
        Runtime Context to retrieve user's metadata: {paper_id: {Title, Authors, Summary, Published, ...}}
    Returns:
        top-N paper IDs ranked by fuzzy matching score
    """
    query_hints = state.get("metadataHints")
    metadata = runtime.context.metadata
    scope = state.get("paperScope", "multiple") 
    top_n = 2 if scope == "single" else 4 
    # Return empty if no hints provided
    if query_hints is None or not any([
        query_hints.titles, query_hints.authors, query_hints.topics, query_hints.publicationYears
    ]):
        return {"arxivIDs": []}

    # Weights for each field
    weights = {
        "titles": 5.0,
        "topics": 3.0,
        "authors": 1.5,
        "publicationYears": 1.0
    }
    total_weight = sum(weights.values())
    q_titles = [t.lower().strip() for t in query_hints.titles if t.strip()]
    q_topics = [t.lower().strip() for t in query_hints.topics if t.strip()]
    q_authors = [a.lower().strip() for a in query_hints.authors if a.strip()][:3]
    q_years = [y.lower().strip() for y in query_hints.publicationYears if y.strip()]
    normalized_q_years =  normalize_query_years(q_years) 
    
    # Minimum score threshold 
    MIN_THRESHOLD = 0.1
    primary_scores = {} # Preselecting papers based on title and topics 
    scores = {} # Ordering based on total score 
    for paper_id, paper_data in metadata.items():
        primary_score = 0.0
        # --- Fuzzy match titles ---
        if q_titles:
            paper_title = paper_data.get("Title", "").lower()
            if paper_title:
                # Take best match among all query titles
                title_score = max(
                    fuzz.partial_ratio(q_title, paper_title) / 100.0 
                    for q_title in q_titles
                )
                primary_score += weights["titles"] * title_score
            
        # --- Fuzzy match topics against summary ---
        if q_topics:
            paper_summary = paper_data.get("Title", "").lower() + " " + paper_data.get("Summary", "").lower()
            if paper_summary:
                # Check each topic against the full summary text
                topics_score = mean([
                    fuzz.partial_ratio(q_topic, paper_summary) / 100.0
                    for q_topic in q_topics
                ]) 
                primary_score += weights["topics"] * topics_score
        
        total_score = primary_score
        # --- Fuzzy match authors ---
        if q_authors:
            paper_authors = [a.lower().strip() for a in paper_data.get("Authors", [])[:3]]
            if paper_authors:
                best_scores = [
                    max(fuzz.ratio(q, a) / 100.0 for a in paper_authors)
                    for q in q_authors
                ]
                author_score = sum(best_scores) / len(q_authors)  
                total_score += weights["authors"] * author_score
            
        # --- Match publication years ---
        if normalized_q_years:
            paper_year = extract_year(paper_data.get("Published", ""))
            if paper_year and paper_year in normalized_q_years:
                total_score += weights["publicationYears"]

        if primary_score >= MIN_THRESHOLD:
            primary_scores[paper_id] = primary_score
            scores[paper_id] = total_score
        
    # Fallback if no matches
    if not primary_scores:
        return {"arxivIDs": []} 
    preselected_papers = sorted(primary_scores, key=lambda pid: primary_scores[pid], reverse=True)[:top_n]
    # Sort by score descending and return top_n
    top_papers = sorted(preselected_papers, key=lambda pid: scores[pid], reverse=True)[:top_n]
    return {"arxivIDs": top_papers}
