import os
from dotenv import load_dotenv
import gradio as gr
from operator import itemgetter
from pathlib import Path
from faiss import IndexFlatL2
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.passthrough import RunnableAssign
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_transformers import LongContextReorder
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings


# ====================== CONFIG ======================
load_dotenv()  
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
EMBEDDING_MODEL = "nvidia/nv-embed-v1"

embedder = NVIDIAEmbeddings(model=EMBEDDING_MODEL, truncate="END")
instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")

# ====================== UTILITIES ======================
def default_FAISS():
    """Create an empty FAISS vectorstore."""
    dims = len(embedder.embed_query("test"))
    return FAISS(
        embedding_function=embedder,
        index=IndexFlatL2(dims),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        normalize_L2=False
    )

def docs2str(docs, title="Document"):
    """Useful utility for making chunks into context string. Optional, but useful"""
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, 'metadata', {}).get('Title', title)
        if doc_name:
            out_str += f"[Quote from {doc_name}] "
        out_str += getattr(doc, 'page_content', str(doc)) + "\n"
    return out_str
# ====================== CONVERSATION AGENT ======================
class ConversationAgent:
    """
    Handles a single conversation session per user:
    - Conversation memory (in FAISS)
    - Document retrieval (RAG)
    - Chat LLM streaming
    """

    def __init__(self, user_id: str, docstore: FAISS):
        self.user_id = user_id
        self.docstore = docstore  #To be loaded
        self.convstore = default_FAISS()  # in-memory conversation memory

        # Chat prompt template
        self.chat_prompt = ChatPromptTemplate.from_messages([
        ("system",
            "You are a document chatbot. Answer questions using only the retrieved context.\n\n"
            "Conversation History:\n{history}\n\n"
            "Retrieved Documents:\n{context}\n\n"
            "Instructions:\n"
            "- Answer ONLY using the information above\n"
            "- Cite sources like [1], [2] when used\n"
            "- If there is no relevant information, answer 'I don't know' or ask for clarification.\n"
            "- Keep responses conversational"
        ),
        ('user', '{input}')
    ])
            
        self.retrieval_chain = (
        {'input' : (lambda x: x)}
        ## retrieve history & context from convstore & docstore, respectively.
        | RunnableAssign({'history' : itemgetter("input")| self.convstore.as_retriever()| LongContextReorder().transform_documents | docs2str })
        | RunnableAssign({'context' : itemgetter("input")| self.docstore.as_retriever() | LongContextReorder().transform_documents | docs2str})
        )

        self.stream_chain = self.chat_prompt | instruct_llm | StrOutputParser()

    def save_message(self, user_input: str, agent_output: str):
        """Save conversation exchange with role metadata."""
        self.convstore.add_texts(
            [user_input, agent_output],
            metadatas=[{"role": "user"}, {"role": "assistant"}]
        )
    
    def chat_gen(self, user_input, return_buffer=True):
        buffer = ""
        ## 1) perform the retrieval based on the input message
        retrieval = self.retrieval_chain.invoke(user_input)
        ## 2) stream the results of the stream_chain
        for token in self.stream_chain.stream(retrieval):
            buffer += token
            yield buffer if return_buffer else token
        ## Save the chat exchange to the conversation memory buffer : convstore
        self.save_message(user_input, buffer) 

# ====================== TEST / MAIN ======================
if __name__ == "__main__":
    user_id = "demo_user"
    docstore_path = Path(f"vectorstores/{user_id}/index")
    docstore = FAISS.load_local(docstore_path, embedder, allow_dangerous_deserialization=True)

    agent = ConversationAgent(user_id=user_id, docstore=docstore)   
    # test_question ="Tell me about attention"
    # for response in agent.chat_gen(test_question, return_buffer=False):
    #     print(response, end='') 
    
    ########## DEMO INTERFACE ########
    initial_msg = (
    "Hello! I am a document chat agent here to help you!"
    # f" I have access to the following documents: {doc_string}\n\nHow can I help you?"
    )
    chatbot = gr.Chatbot(
    value=[{"role": "assistant", "content": initial_msg}],
    type="messages"
    )

    demo = gr.ChatInterface(agent.chat_gen, chatbot=chatbot, type="messages").queue()
    
    try:
        demo.launch(debug=True, share=True, show_api=False)
        demo.close()
    except Exception as e:
        demo.close()
        print(e)
        raise e
