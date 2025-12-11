import os
import json
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from faiss import IndexFlatL2
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

# ====================== CONFIG ======================
load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
EMBEDDING_MODEL = "nvidia/nv-embed-v1"
embedder = NVIDIAEmbeddings(model=EMBEDDING_MODEL, truncate="END")

# TEXT SPLITTER
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100,
    separators=["\n\n", "\n", ".", ";", ",", " "],
)

# FAISS UTILITY
def default_faiss():
    """Create an empty FAISS vectorstore."""
    dims = len(embedder.embed_query("test"))
    return FAISS(
        embedding_function=embedder,
        index=IndexFlatL2(dims),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        normalize_L2=False
    )

# VECTORSTORE CLASS 
class PaperVectorStore:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.path = Path(f"vectorstores/{user_id}")
        self.path.mkdir(parents=True, exist_ok=True)
        self.store_path = self.path / "index"
        self.vectorstore = self.load_or_create_store()
        self.paper_metadata_path = self.path / "paper_metadata.json"
        self.paper_metadata = self.load_paper_metadata()

    
    def load_paper_metadata(self):
        """Load paper metadata from JSON file."""
        if self.paper_metadata_path.exists():
            try:
                with open(self.paper_metadata_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def save_paper_metadata(self):
        with open(self.paper_metadata_path, "w") as f:
            json.dump(self.paper_metadata, f)

    def load_or_create_store(self) -> FAISS:
        """Load existing FAISS vectorstore or create a new one."""
        if self.store_path.exists():
            print(f"🔁 Loading existing FAISS store for {self.user_id}")  #TODO : remove the prints 
            return FAISS.load_local(str(self.store_path), embedder, allow_dangerous_deserialization=True)
        print(f"✨ Creating new FAISS store for {self.user_id}")
        return default_faiss()

    def preprocess(self, doc, arxiv_id) -> List:
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
            if len(c.page_content) > 200: # Filter out tiny chunks
                c.metadata = {"paper_id": arxiv_id, "Title" : c.metadata.get("Title")}   # assign minimal metadata
                chunks.append(c)
        return chunks

    def update_paper_metadata(self, doc, arxiv_id, len_chunks):
        self.paper_metadata[arxiv_id] = {
                    'Title': doc.metadata.get('Title', 'Unknown'),
                    'Authors': doc.metadata.get('Authors', []),
                    'Published': doc.metadata.get('Published', '')[:4] if doc.metadata.get('Published') else 'Unknown',
                    'Summary': doc.metadata.get('Summary', ''),
                    'pdf_url': f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                    'total_chunks': len_chunks,
                    'ingested_at': datetime.now().isoformat()  # Track when ingested
                }
        self.save_paper_metadata() 

    def ingest_papers(self, arxiv_ids: List[str]) -> dict:
        """
        Load multiple ArXiv papers, preprocess, chunk, embed, and add to the vectorstore.
        Returns:
            dict: {
                'successful': list of successfully ingested IDs,
                'failed': list of failed IDs with reasons,
                'total_processed': int,
                'message': str
            }
        """
        successful_ingestions = []
        failed_ingestions = []  # Track failures with reasons
        all_chunks = []
        
        for arxiv_id in arxiv_ids:
            #Ignore already ingested papers 
            if arxiv_id in self.paper_metadata:
                failed_ingestions.append({'id': arxiv_id, 'reason': 'Paper already ingested'})
                continue
            try:
                print(f"📥 Loading paper {arxiv_id} from ArXiv")
                docs = ArxivLoader(query=arxiv_id).load()
                if not docs:
                    failed_ingestions.append({'id': arxiv_id, 'reason': 'No content found on arXiv'})
                    continue
                    
                doc = docs[0]
                chunks = self.preprocess(doc, arxiv_id)
                print(f"🧩 Generated {len(chunks)} chunks for paper {arxiv_id}")
                
                if not chunks:
                    failed_ingestions.append({'id': arxiv_id, 'reason': 'No valid chunks generated after preprocessing'})
                    continue
                    
                all_chunks.extend(chunks)
                successful_ingestions.append(arxiv_id)
                # Update paper metadata
                self.update_paper_metadata(doc, arxiv_id, len(chunks))
                
            except Exception as e:
                error_msg = f"Failed to ingest {arxiv_id}: {str(e)}"
                print(f"❌ {error_msg}")
                failed_ingestions.append({'id': arxiv_id, 'reason': error_msg})
                continue
        
        # Only update vectorstore if we have chunks
        if all_chunks:
            try:
                self.vectorstore.add_documents(all_chunks)
                # self.ingested_ids.extend(successful_ingestions)
                self.save_store()
                # self.save_ingested_ids()
            except Exception as e:
                # If saving fails, mark all as failed
                failed_ingestions.extend([{'id': id, 'reason': f'Vector store update failed: {str(e)}'} 
                                        for id in successful_ingestions])
                successful_ingestions = []
        
        # Build comprehensive result message
        total_processed = len(successful_ingestions) + len(failed_ingestions)
        
        if successful_ingestions and not failed_ingestions:
            count = len(successful_ingestions)
            message = f"✅ Successfully ingested {count} paper{'s' if count > 1 else ''}!"
        elif successful_ingestions and failed_ingestions:
            success_count = len(successful_ingestions)
            fail_count = len(failed_ingestions)
            message = (
                f"✅ Ingested {success_count} paper{'s' if success_count > 1 else ''}, "
                f"❌ failed {fail_count} paper{'s' if fail_count > 1 else ''}"
            )
        elif not successful_ingestions and failed_ingestions:
            fail_count = len(failed_ingestions)
            message = f"❌ Failed to ingest {fail_count} paper{'s' if fail_count > 1 else ''}"
        else:
            message = "No papers processed"

        # Add details about failures
        if failed_ingestions:
            failure_details = "\n".join([f"• {f['id']}: {f['reason']}" for f in failed_ingestions])
            message += f"\n\nFailed papers:\n{failure_details}"
        
        return {
            'successful': successful_ingestions,
            'failed': failed_ingestions,
            'total_processed': total_processed,
            'message': message
        }

    def save_store(self):
        """Save FAISS vectorstore to disk."""
        self.vectorstore.save_local(str(self.store_path))
        print(f"💾 Saved vectorstore at {self.store_path}")

    def get_num_vectors(self) -> int:
        """Return total number of vectors in the store."""
        return self.vectorstore.index.ntotal
    #Note saving
    def save_notes(self, paper_id, text):
        if paper_id not in self.paper_metadata:
            return False
        self.paper_metadata[paper_id]["notes"] = text
        self.save_paper_metadata()  # Your existing JSON save method
        return True

# === TEST ===
if __name__ == "__main__":
    user_store = PaperVectorStore(user_id="demo_user")
    # Ingest multiple papers at once
    user_store.ingest_papers([
        "1706.03762",  # Attention Is All You Need
        "1810.04805",  # BERT
        "2005.11401",  # RAG
    ])
    
    print(f"Total vectors stored: {user_store.get_num_vectors()}")
