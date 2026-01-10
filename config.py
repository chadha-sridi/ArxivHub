import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
load_dotenv()

# Env variables
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Embedder
embedder = NVIDIAEmbeddings(model=EMBEDDING_MODEL, truncate="END")
EMBED_DIM = len(embedder.embed_query("test"))
# LLM
llm = ChatNVIDIA(model="meta/llama-3.2-3b-instruct")

# Qdrant client
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# === Collection creation (if it does not exist) ===
COLLECTION_NAME = "paperpal_collection"

existing_collections = [c.name for c in qdrant_client.get_collections().collections]

if COLLECTION_NAME not in existing_collections:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE)
    )
    # Indexing payloads 
    qdrant_client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="user_id",  
    field_schema="keyword" 
    )
    qdrant_client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="paper_id",  
    field_schema="keyword"
    )
    qdrant_client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="title",  
    field_schema="text"
    )

BASE_USER_DATA_DIR = Path("user_data")
