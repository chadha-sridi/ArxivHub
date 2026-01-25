import os
import json
import logging
import asyncio
from pathlib import Path
from filelock import FileLock
from typing import List, Dict, Any
from datetime import datetime
from qdrant_client import models
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from config import BASE_USER_DATA_DIR
from langchain_community.document_loaders import ArxivLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# TEXT SPLITTER
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
MIN_CHUNK_LENGTH = 300
SEPARATORS = ["\n\n", "\n", ".", ";", ",", " "]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=SEPARATORS,
)

# Helper to get lock path for a user
def get_lock_path(user_id: str) -> Path:
    """Get the lock file path for a specific user"""
    user_dir = BASE_USER_DATA_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir / "paper_metadata.json.lock"

async def load_paper_metadata(user_id: str) -> Dict[str, Any]:
    """Load paper metadata from JSON file with file locking."""
    user_dir = BASE_USER_DATA_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    paper_metadata_path = user_dir / "paper_metadata.json"
    lock_path = get_lock_path(user_id)
    
    def _read():
        with FileLock(lock_path, timeout=10):
            if paper_metadata_path.exists():
                try:
                    with open(paper_metadata_path, "r") as f:
                        return json.load(f)
                except json.JSONDecodeError:
                    logging.warning(f"Corrupted metadata file for user {user_id}, returning empty dict")
                    return {}
            return {}
    
    return await asyncio.to_thread(_read)

async def save_paper_metadata(user_id: str, paper_metadata: Dict[str, Any]) -> None:
    """Save paper metadata to JSON file with file locking."""
    user_dir = BASE_USER_DATA_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    paper_metadata_path = user_dir / "paper_metadata.json"
    lock_path = get_lock_path(user_id)
    
    def _write():
        with FileLock(lock_path, timeout=10):
            with open(paper_metadata_path, "w") as f:
                json.dump(paper_metadata, f, indent=2)
    
    await asyncio.to_thread(_write)

async def update_paper_metadata(user_id: str, paper_metadata: Dict[str, Any], doc_metadata: Dict[str, Any], arxiv_id: str, len_chunks: int) -> None:   
    paper_metadata[arxiv_id] = {
        'Title': doc_metadata.get('Title', 'Unknown'),
        'Authors': doc_metadata.get('Authors', []),
        'Published': doc_metadata.get('Published', '')[:4] if doc_metadata.get('Published') else 'Unknown',
        'Summary': doc_metadata.get('Summary', ''),
        'pdf_url': f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        'total_chunks': len_chunks,
        'ingested_at': datetime.now().isoformat()
    }
    await save_paper_metadata(user_id, paper_metadata)

async def save_notes(user_id: str, paper_metadata: Dict[str, Any], paper_id: str, text: str) -> bool:
    if paper_id not in paper_metadata:
        return False
    paper_metadata[paper_id]["notes"] = text
    await save_paper_metadata(user_id, paper_metadata)  
    return True

def preprocess(user_id: str, doc: Document, arxiv_id: str) -> List[Document]:
    """
    Clean a document by removing references and splitting into chunks.
    Returns a list of Document objects (chunks).
    """
    content = doc.page_content
    if "References" in content:
        content = content[:content.index("References")]
    doc.page_content = content
    chunks = []
    for c in text_splitter.split_documents([doc]):
        if len(c.page_content) > MIN_CHUNK_LENGTH: # Filter out tiny chunks
            # assign minimal metadata
            c.metadata = {
            "user_id": user_id,
            "paper_id": arxiv_id,
            "title": c.metadata.get("Title"),
        }
            chunks.append(c)
    return chunks

async def ingest_papers(user_id: str, paper_metadata: Dict[str, Any], vectorstore: QdrantVectorStore, arxiv_ids: List[str]) -> Dict[str, Any]:
    """
    Ingest multiple ArXiv papers, one by one.
    Each paper's chunks are added to Qdrant individually, and metadata is updated only if the add succeeds.
    Returns a dict with success/failure info.
    """
    successful = []
    failed = []

    for arxiv_id in arxiv_ids:
        # Paper already in inventory --> Ignore
        if arxiv_id in paper_metadata: 
            failed.append({"id": arxiv_id, "reason": "Paper already ingested"})
            continue

        try:
            logging.info(f"📥 Loading paper {arxiv_id} from ArXiv")
            docs = await ArxivLoader(query=arxiv_id).aload()
            if not docs:
                failed.append({"id": arxiv_id, "reason": "No content found on ArXiv"})
                continue

            doc = docs[0]
            chunks = preprocess(user_id, doc, arxiv_id)

            try:
                # await vectorstore.aadd_documents(chunks)
                # 1. Generate embeddings for the chunks
                texts = [c.page_content for c in chunks]
                embeddings = await vectorstore.embeddings.aembed_documents(texts)
                # 2. Build Qdrant points
                import uuid
                points = [
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=emb,
                        payload={
                            "page_content": chunk.page_content,
                            "metadata": chunk.metadata
                        }
                    )
                    for chunk, emb in zip(chunks, embeddings)
                ]
                # 3. Upsert
                await vectorstore.client.upsert(
                    collection_name=vectorstore.collection_name,
                    points=points
                )
                # Only update metadata if add succeeds
                await update_paper_metadata(user_id, paper_metadata, doc.metadata, arxiv_id, len(chunks))
                successful.append(arxiv_id)
                logging.info(f"✅ Successfully ingested {arxiv_id}")
            except Exception as e:
                failed.append({"id": arxiv_id, "reason": f"Vector store update failed: {e}"})

        except Exception as e:
            failed.append({"id": arxiv_id, "reason": f"Ingestion failed: {e}"})

    total = len(successful) + len(failed)
    message = (
        f"✅ Ingested {len(successful)} paper(s), ❌ failed {len(failed)} paper(s)"
        if successful and failed
        else f"✅ Successfully ingested {len(successful)} paper(s)" if successful
        else f"❌ Failed to ingest {len(failed)} paper(s)"
    )
    if failed:
        details = "\n".join([f"• {f['id']}: {f['reason']}" for f in failed])
        message += f"\n\nFailed papers:\n{details}"

    return {"successful": successful, "failed": failed, "total_processed": total, "message": message}

async def delete_paper(user_id: str, paper_metadata: Dict[str, Any], vectorstore: QdrantVectorStore, paper_id: str) -> bool:
    """
    Delete all chunks of a given paper from Qdrant and remove its metadata.
    Returns True if deletion succeeded, False otherwise.
    """
    if paper_id not in paper_metadata:
        logging.warning(f"Paper {paper_id} not found in metadata.")
        return False
    try:
        # Create a filter to match user_id and paper_id
        delete_filter = models.Filter(
            must=[
                models.FieldCondition(key="metadata.user_id", match=models.MatchValue(value=user_id)),
                models.FieldCondition(key="metadata.paper_id", match=models.MatchValue(value=paper_id)),
            ]
        )
        await vectorstore.client.delete(
            collection_name=vectorstore.collection_name,
            points_selector=delete_filter
        )        
        # Remove metadata
        del paper_metadata[paper_id]
        await save_paper_metadata(user_id, paper_metadata)
        logging.info(f"✅ Successfully deleted paper {paper_id} and its chunks.")
        return True
    
    except Exception as e:
        logging.error(f"❌ Failed to delete chunks of {paper_id} from vector store: {e}")
        return False

async def get_num_vectors(user_id: str, vectorstore: QdrantVectorStore) -> int:
    """Return total number of vectors belonging to the user."""
    try:  
        result = await vectorstore.client.count(
            collection_name=vectorstore.collection_name,
            count_filter=models.Filter(
                must=[models.FieldCondition(key="metadata.user_id", match=models.MatchValue(value=user_id))]
            )
        )
        return result.count
    except Exception as e:
        return 0