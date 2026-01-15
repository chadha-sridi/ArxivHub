import re
from config import llm
from core.schemas import State
from core.prompts import get_generation_prompt
from langchain_core.messages import HumanMessage, SystemMessage

def extract_clean_answer(content: str) -> str:
    # 1. Find the start of the answer tag 
    tag = "<answer>"
    start_idx = content.lower().find(tag)
    
    if start_idx != -1:
        # 2. Extract everything after the tag
        result = content[start_idx + len(tag):].strip()
        
        # 3. Clean up the closing tag if it exists
        result = re.sub(r"</answer>", "", result, flags=re.IGNORECASE).strip()
        return result
    
    # 4. Fallback: If no <answer> tag, remove <thinking> block
    return re.sub(r"<thinking>.*?</thinking>", "", content, flags=re.DOTALL | re.IGNORECASE).strip()

def generate(state: State):
    # Context with XML markers
    docs = state["retrievedDocs"]
    context_blocks = []
    for i, doc in enumerate(docs):
        arxiv_id = doc.metadata.get("paper_id", f"Unknown_{i}")
        title = doc.metadata.get("title", f"Untitled_{i}")
        source = f"{arxiv_id}: {title}"

        context_blocks.append(
            f'<document index="{i+1}">\n'
            f'<source>{source}</source>\n'
            f'<content>{doc.page_content}</content>\n'
            f'</document>'
        )
    
    context_xml = "\n".join(context_blocks)

    system_prompt = get_generation_prompt(context_xml)

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["rewrittenQuestion"])
    ])
    # Extract the clean answer (without <thinking>)
    clean_answer = extract_clean_answer(response.content)

    return {
        "messages": [response],          
        "finalAnswer": clean_answer       
    }
    