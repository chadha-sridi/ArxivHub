import asyncio
import os
import logging 
from pathlib import Path
from dotenv import load_dotenv
from tavily import AsyncTavilyClient
from qdrant_client import AsyncQdrantClient, models
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import PayloadSchemaType
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
load_dotenv()
logger = logging.getLogger(__name__)

# Env variables
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
BASE_USER_DATA_DIR = Path("user_data")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Embedder
embedder = NVIDIAEmbeddings(model=EMBEDDING_MODEL, truncate="END")

# LLM
llm = ChatNVIDIA(model="meta/llama-3.2-3b-instruct")
research_llm = ChatNVIDIA(model="nvidia/nemotron-3-nano-30b-a3b")

# Initialize the tavily client
tavily = AsyncTavilyClient(api_key=TAVILY_API_KEY)

# Qdrant client
qdrant_client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=300)

# === Collection creation (if it does not exist) ===
COLLECTION_NAME = "ArXivHub_collection"

async def get_vectorstore():
    
    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embedder,
        validate_collection_config=False
    )

async def init_db():
    # Check if collection exists
    collections_response = await qdrant_client.get_collections()
    exists = any(c.name == COLLECTION_NAME for c in collections_response.collections)

    if not exists:
        logger.info(f"Creating collection: {COLLECTION_NAME}")
        sample_embedded = await embedder.aembed_query("health check")
        dim = len(sample_embedded)

        await qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        
        # Indexing payloads 
        payloads = {
            "metadata.user_id": PayloadSchemaType.KEYWORD,
            "metadata.paper_id": PayloadSchemaType.KEYWORD,
            "metadata.title": PayloadSchemaType.TEXT,
        }
        
        for field, schema in payloads.items():
            await qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=schema,
            )
        logger.info("Database initialized.")

if __name__ == "__main__":
    asyncio.run(init_db())