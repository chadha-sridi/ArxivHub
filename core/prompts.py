def get_query_analysis_prompt() -> str:
    return """
        Analyze the user query and extract the following information based ONLY on the query: 

        Tasks: 
        1. Decide if the query is clear enough to answer.
        Guidelines: 
        - Use the conversation context ONLY if it is needed to understand the query OR to determine the domain when the query itself is ambiguous.
        - If the intent is unclear or meaningless, mark as unclear.
        2. Rewrite the query into one concise and self-contained question highlighting the user's intent (e.g., finding a paper, comparing methods from specific papers, summarizing a topic, etc.).
        3. Determine the paper scope of the question.
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
        4. If the question is unclear, provide a brief clarification message explaining what is missing or ambiguous.
        5. Carefully analyze the query and populate the predefined metadata fields with information from the original query.   
        Guidelines for metadata extarction : 
        - Extract metadata ONLY into the predefined fields: titles, authors, topics, publicationYears, arxivIDs.
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

def get_conversation_summary_prompt() -> str:
    return """
        Summarize the conversation in 1–2 concise sentences.

        Include:
        - Main topics discussed
        - Key facts or entities
        - Any unresolved questions

        Exclude:
        - Greetings
        - Misunderstandings
        - Off-topic content

        If no meaningful information exists, return an empty string.

        Output only the summary. No explanations.
        """
