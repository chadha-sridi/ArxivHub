from config import llm
from schemas import State 
from prompts import get_conversation_summary_prompt

def analyze_chat_and_summarize(state: State):
    """
    Summarize conversation history.
    """
    if len(state["messages"]) < 4:  
        return {"conversationSummary": ""}

    # Exclude current query and system messages
    relevant_msgs = [
        msg for msg in state["messages"][:-1]  
        if isinstance(msg, (HumanMessage, AIMessage))
    ]

    if not relevant_msgs:
        return {"conversationSummary": ""}

    conversation = "Conversation history:\n"
    for msg in relevant_msgs[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        conversation += f"{role}: {msg.content}\n"

    summary = llm.invoke([
        SystemMessage(content=get_conversation_summary_prompt()),
        HumanMessage(content=conversation),
    ])
    return {"conversationSummary": summary.content}
