import os
from dotenv import load_dotenv
from langgraph.graph import MessagesState 
from pydantic import BaseModel, Field
from typing import Optional, Literal, List
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
# ====================== CONFIG ======================
load_dotenv()  
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
llm = ChatNVIDIA(model="meta/llama-3.2-3b-instruct")

class State(MessagesState):
    """State for the workflow"""
    # messages: Annotated[Sequence[BaseMessage], add_messages]
    conversationSummary: str = ""
    originalQuestion: str = ""
    questionIsClear: bool = True
    rewrittenQuestion: str = ""
    chosenDatasource: Literal["vectorstore", "web"] 
    metadataHintPresent: bool = False
    arxivIDs: List[str] = []
    retrievedChunkIDs: List[str] = []
    confidenceScores: List[float] = []
    relevancePassed: bool = True
    finalAnswer: Optional[str]

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
        description="Specific research topics, methods, or domains mentioned in the user query"
    )
    publicationYears: List[str] = Field(
        default_factory=list,
        description="Publication years referenced or implied in the user query (e.g., '2020', 'recent', 'last year')"
    )
    arxivIDs: List[str] = Field(
        default_factory=list,
        description="Arxiv IDs referenced in the user query (e.g., '1706.03762' )"
    )

class QueryAnalysis(BaseModel):
    is_clear: bool = Field(
        description="Indicates if the user's question is clear and answerable."
    )
    rewrittenQuestion: str = Field(
        description="Rewritten, self-contained version of the user's question preserving intent."
    )
    clarification_needed: str = Field(
        description="Explanation if the question is unclear."
    )
    metadataHints: MetadataHints = Field(
        default_factory=MetadataHints,
        description="Predefined metadata fields extracted from the user query"
    )

def get_query_analysis_prompt() -> str:
    return """
        Analyze a user query and prepare it for retrieval.

        Tasks:
        1. Decide if the query is clear enough to answer.
        2. Use the conversation context ONLY if it is needed to understand the query OR to determine the domain when the query itself is ambiguous.
        2. Rewrite the query into one self-contained question that preserves the user's intent (e.g., finding a paper, comparing methods from specific papers, summarizing a topic, etc.).
        3. Carefully analyze the query and populate the predefined metadata fields with any explicit or implied information.
        3. Extract metadata ONLY into the predefined fields: titles, authors, topics, publicationYears, arxivIDs.
        4. If unclear, explain what clarification is needed.

        Rules:
        - Do NOT invent new metadata fields.
        - Leave metadata fields empty if no signal is present.
        - Keep the rewritten query grammatically correct, concise and self-contained.

        Failure:
                - If the intent is unclear or meaningless, mark as unclear.
                
        Example:
        User query: "In recent papers by Smith and Johnson, what deep learning and reinforcement learning methods were used for robotic control?"
        Rewritten: "Which deep learning and reinforcement learning methods are used in robotic control papers authored by Smith and Johnson?"
        MetadataHints:
            titles: []
            authors: ["Smith", "Johnson"]
            topics: ["deep learning", "reinforcement learning", "robotic control"]
            publicationYears: []
            arxivIDs: []
        """

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

    return {
        "questionIsClear": True,
        "rewrittenQuestion": analysis.rewrittenQuestion,
        "metadataHintPresent": metadata_present,
        "metadataHints": analysis.metadataHints,
        "originalQuestion": state.get("originalQuestion") or last_user_msg
    }

if __name__ == "__main__":
    
    examples = [
        "In recent papers by Smith and Johnson, what deep learning and reinforcement learning methods were used for robotic control?",
        "Compare the summarization techniques used in papers from 2021 to 2023 on natural language processing.",
        "Which papers authored by Google researchers discuss diffusion models in image generation?"
    ]

    for i, query in enumerate(examples, start=1):
        # Initialize a fresh state
        state = State(messages=[HumanMessage(content=query)])

        # Run query analysis
        updated_state = analyze_query(state)

        print(f"\n--- Example {i} ---")
        print("Original Query:", query)
        print("Question Is Clear:", updated_state["questionIsClear"])
        print("Rewritten Question:", updated_state.get("rewrittenQuestion"))
        print("Metadata Hint Present:", updated_state.get("metadataHintPresent"))
        if updated_state.get("metadataHints"):
            print("Metadata Hints:", updated_state["metadataHints"].dict())
        if not updated_state["questionIsClear"]:
            print("Clarification Needed:", updated_state["messages"][0].content)
