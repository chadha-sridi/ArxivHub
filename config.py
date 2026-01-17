import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import PayloadSchemaType
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
load_dotenv()

# Env variables
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
BASE_USER_DATA_DIR = Path("user_data")

# Embedder
embedder = NVIDIAEmbeddings(model=EMBEDDING_MODEL, truncate="END")
EMBED_DIM = len(embedder.embed_query("test"))
# LLM
llm = ChatNVIDIA(model="meta/llama-3.2-3b-instruct")

# Qdrant client
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=300)

# === Collection creation (if it does not exist) ===
COLLECTION_NAME = "ArXivHub_collection"

existing_collections = [c.name for c in qdrant_client.get_collections().collections]

if COLLECTION_NAME not in existing_collections:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE), 
        strict_mode_config=models.StrictModeConfig(
        unindexed_filtering_retrieve=True)
    )
    # Indexing payloads 
    qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="metadata.user_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="metadata.paper_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="metadata.title",
        field_schema=PayloadSchemaType.TEXT,
    )

# ArXivHub Vectorstore 
ArxivHubVectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embedder,
    )