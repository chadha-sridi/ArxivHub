from config import llm
from core.schemas import State
from core.prompts import get_casual_generation_prompt
from langchain_core.messages import HumanMessage, SystemMessage

def handle_general_talk(state: State):

    system_prompt = get_casual_generation_prompt()
    user_query = state.get("rewrittenQuestion") or state.get("originalQuestion")
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ])

    return {
        "messages": [response],          
        "finalAnswer": response.content       
    }
    