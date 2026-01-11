from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing import Optional, Literal, List

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
    arxivIDs: List[str] = Field(
        default_factory=list,
        description="Arxiv IDs referenced in the user query (e.g., '1706.03762' )"
    )

class State(MessagesState):
    """State for the workflow"""
    # messages: Annotated[Sequence[BaseMessage], add_messages]
    conversationSummary: str = ""
    originalQuestion: str = ""
    questionIsClear: bool = True
    rewrittenQuestion: str = ""
    paperScope: Literal["single", "multiple"] = "multiple"
    chosenDatasource: Literal["vectorstore", "web"] = "vectorstore"
    metadataHintPresent: bool = False
    metadataHints: MetadataHints = Field(default_factory=MetadataHints)
    arxivIDs: List[str] = []
    retrievedChunkIDs: List[str] = []
    confidenceScores: List[float] = []
    relevancePassed: bool = True
    finalAnswer: Optional[str]

class QueryAnalysis(BaseModel):
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
        description="metadata (titles, authors, topics, publication years, arviv ids) mentionned in the user query"
    )
