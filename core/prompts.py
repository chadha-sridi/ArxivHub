def get_conversation_summary_prompt() -> str:
    return """
        Summarize the conversation in a maximum of 4–6 concise sentences.

        Include:
        - Main topics discussed
        - Key facts or entities
        - Key conclusions 
        - Any unresolved questions

        Exclude:
        - Greetings
        - Misunderstandings
        - Off-topic content

        If no meaningful information exists, return an empty string.

        Output only the summary. No explanations.
        """

def get_query_analysis_prompt() -> str:
    return """
        Analyze the user query and extract the following information based ONLY on the query: 

        Tasks: 
        1. Classify the user's intent as 'casual' only for greetings or non-research small talk. If the user mentions papers, topics, or data, classify as 'research'."
        2. Decide if the query is clear enough to answer.
        Guidelines: 
        - Use the conversation context ONLY if it is needed to understand the query OR to determine the domain when the query itself is ambiguous.
        - If the intent is unclear or meaningless, mark as unclear.
        3. Rewrite the query into one concise and self-contained question highlighting the user's intent (e.g., finding a paper, comparing methods from specific papers, summarizing a topic, etc.).
        4. Determine the paper scope of the question.
        Guidelines for paperScope:
        - Set paperScope = "single" if the user explicitly mentions:
            - ONLY one specific paper title,
            - ONLY one arXiv ID,
            - or clearly refers to a SINGLE identifiable paper (e.g. "the BERT paper").
        - Set paperScope = "multiple" if:
            - no specific paper is mentioned,
            - the question is general or exploratory,
            - the user asks for comparisons, trends, surveys, or "recent papers",
            - or multiple papers/authors are implied.
        - When in doubt, choose "multiple".
        5. If the question is unclear, provide a brief clarification message explaining what is missing or ambiguous.
        6. Carefully analyze the query and populate the predefined metadata fields with information from the original query.   
        Guidelines for metadata extarction : 
        - Extract metadata ONLY into the predefined fields: titles, authors, topics, publicationYears.
        - Do NOT invent new metadata fields.
        - For topics, include any methods, models, domains, or research concepts mentioned in the query.
        - For publicationYears, convert explicit or relative references (e.g., '2021', 'after 2020', 'last year') into integers when possible.
        - Leave metadata field empty if no corresponding signal is present.
        - Do NOT extract information not mentionned in the user query.
        
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
        
def get_generation_prompt(context_xml: str, summary: str) -> str:
    return f"""
    You are a world-class Research Scientist. 
    Your goal is to answer the user question precisely includinng all the sub-questions using only the relevant snippets provided in the context below.

    <context>
    {context_xml}
    </context>
    conversation summary from previous messages : {summary} 
    <instructions>
    1. **conversation summary usage** : Use the conversation summary if it helps frame or understand the current user query, or to help resolve references. Disregard it if the new query is a new topic.
    1. **Source Analysis**: Identify which snippets contain the facts needed to answer the user's query.
    2. **Strict Grounding**: Every sentence in your answer MUST be supported by at least one source. Use inline citations [1].
    3. **Mapping Verification**: In your <thinking> block, perform a 'Source-to-Claim' mapping.
    </instructions>

    Your response MUST follow this exact structure:

    <thinking>
    Identify the relevant sources using their [Source Index].
    Plan: "I will use [Source Index] to explain [Claim X] and [Source Index] to support [Claim Y]."
    </thinking>

    <answer>
    [Provide your well-cited, professional answer here. Use inline citations like [1] or [1, 2].]

    Include source citation only if you actually used the sources.
    ---
    **Sources used in this response:**
    - [Source Index] Arxiv ID: Title
    </answer>
    """

def get_casual_generation_prompt(summary: str) -> str:
    return f"""
    You are ArXivHub, a professional and friendly research assistant.
    Current Goal: Respond to the user's small talk, greeting, or general inquiry.
    Context from previous messages : {summary}
    Guidelines:
    1. Be polite and concise.
    2. If the user asks who you are, explain that you help answer questions about research papers added to the user's inventory using Retrieval-Augmented Generation for precise and grounded responses.
    3. If the user asks a general knowledge question (e.g., "What is photosynthesis?" or "how to make tomato sauce?"), answer it briefly using your internal knowledge.
    4. If the user seems to be trying to start a research task but didn't provide enough info, gently guide them to ask about a specific topic or paper.
    5. Use the summary to resolve references (like 'it', 'that', or 'the paper we mentioned'). If the current query is a new topic, prioritize the query over the summary."
    """
