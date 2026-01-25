from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langgraph.graph import MessagesState
from typing import Optional, Literal, List, Dict, Any
from langchain_qdrant import QdrantVectorStore

class MetadataHints(BaseModel):
    titles: List[str] = Field(
        default_factory=list,
        description="Paper titles or partial titles explicitly mentioned or strongly implied in the user query"
    )
    authors: List[str] = Field(
        default_factory=list,
        description="Author names or institutional authors (e.g., Google, OpenAI, Meta) mentioned in the user query"
    )
    topics: List[str] = Field(
        default_factory=list,
        description="Specific research topics, methods, or domains mentioned in the user query (e.g., 'Reinforcement learning', 'LLM reasoning', 'Computer vision')"
    )
    publicationYears: List[str] = Field(
        default_factory=list,
        description="Publication years referenced or implied in the user query (e.g., '2020', 'recent', 'last year')"
    )

class State(MessagesState):
    """State for the workflow"""
    # messages: Annotated[Sequence[BaseMessage], add_messages]
    conversationSummary: str = ""
    originalQuestion: str = ""
    intent: Literal["research", "casual"] = "research"
    questionIsClear: bool = True
    rewrittenQuestion: str = ""
    paperScope: Literal["single", "multiple"] = "multiple"
    chosenDatasource: Literal["vectorstore", "web"] = "vectorstore"
    metadataHintPresent: bool = False
    metadataHints: Optional[MetadataHints] = Field(default_factory=MetadataHints)
    arxivIDs: List[str] = []
    retrievedDocs: List[Document] = []
    confidenceScores: List[float] = []
    relevancePassed: bool = True
    unanswered: str = "" # Stores the specific gap query for Tavily
    finalAnswer: Optional[str]

@dataclass
class RuntimeContext:
    user_id: str
    vectorstore: QdrantVectorStore       
    metadata: Dict[str, Any]  
    settings: Dict[str, Any] = field(default_factory=lambda: {
        "retrieval_score_threshold": 0.45,
        "retrieval_top_k": 5
    })

class QueryAnalysis(BaseModel):
    intent: Literal["research", "casual"] = Field(
        description="Choose research' for questions about papers or science and 'casual' for greetings/thanks and random chat."
    )
    is_clear: bool = Field(
        description="Indicates if the user's question is clear and answerable."
    )
    rewrittenQuestion: str = Field(
        description="Rewritten, self-contained version of the user's question preserving intent."
    )
    paperScope: Literal["single", "multiple"] = Field(
        description="Whether the question refers to one specific paper or multiple papers"
    )
    clarification_needed: str = Field(
        description="Explanation if the question is unclear."
    )
    metadataHints: MetadataHints = Field(
        default_factory=MetadataHints,
        description="metadata (titles, authors, topics, publication years) mentionned in the user query"
    )

class DocRelevance(BaseModel):
    """Grade a single document's relevance to the question."""
    grade: Literal["relevant","fully answers the question", "partially answers the question", "completely irrelevant"]
    reasoning: str = Field(description="Briefly explain why this grade was given.")

class CollectiveAudit(BaseModel):
    """Audit the combined context to find unanswered aspects."""
    relevance_passed: bool = Field(description="True if ALL aspects of the question are answered.")
    unanswered_aspect: Optional[str] = Field(description="A concise question focusing on the missing info.")
