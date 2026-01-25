import re
import gradio as gr
from typing import List, Dict, Any, Tuple
from config import get_vectorstore
from ingestion import ingest_papers, save_notes, delete_paper

def get_ingested_papers(user_paper_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    papers = []
    for paper_id, metadata in user_paper_metadata.items():
        papers.append({
            'id': paper_id,
            'Title': metadata.get('Title', 'Unknown'),
            'Authors': metadata.get('Authors', ''),
            'Year': metadata.get('Published', 'Unknown'),
            'Summary': metadata.get('Summary', ''),
            'ingested_at': metadata.get('ingested_at', '')
        })
    # Sort by time: most recent first
    papers.sort(key=lambda x: x.get('ingested_at', ''), reverse=True)
    return papers

def prepare_dataset_samples(user_paper_metadata: Dict[str, Any]) -> Tuple[List[List[str]], List[str]]:
    """Prepare papers for gr.Dataset display"""
    papers = get_ingested_papers(user_paper_metadata)
    samples = []
    paper_ids = []
    
    for p in papers:
        # Format: [display_text, paper_id]
        display = f"{p['Title'][:60]}... • {p['Year']}" if len(p['Title']) > 60 else f"{p['Title']} • {p['Year']}"
        samples.append([display])
        paper_ids.append(p['id'])
    
    return samples, paper_ids

def open_paper_detail_from_dataset(
    evt: gr.SelectData, 
    user_paper_metadata: Dict[str, Any], 
    paper_ids_list: List[str]
    ) -> Tuple[Any, ...]:    
    """Handle paper selection from dataset"""
    selected_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if selected_idx >= len(paper_ids_list):
        return None, gr.update(visible=True), gr.update(visible=False), "", "", gr.update(), ""
    
    paper_id = paper_ids_list[selected_idx]
    metadata = user_paper_metadata.get(paper_id, {})
    
    if not metadata:
        return None, gr.update(visible=True), gr.update(visible=False), "", "", gr.update(), ""
    
    pdf_url = metadata.get('pdf_url', "Couldn't load paper")
    pdf_html = f'<iframe src="{pdf_url}" width="100%" height="800px"></iframe>' if pdf_url else "Couldn't load paper"
    
    return (
        paper_id,
        gr.update(visible=False),                   # main_chat visibility
        gr.update(visible=True),                    # paper_detail visibility
        f"## {metadata.get('Title', 'Unknown')}",   # paper_title
        pdf_html,                                   # paper_content
        gr.update(),                                # paper_chatbot
        metadata.get('notes', "")                   # paper_notes
    )

def back_to_main():
    return 'main_chat', gr.update(visible=True), gr.update(visible=False), "", "", gr.update(), gr.update()

def validate_arxiv_id(arxiv_id: str) -> bool:
    pattern = r'^(\d{4}\.\d{4,5}(v\d+)?|[a-z]+(-[a-z]+)*/\d{7}(v\d+)?)$'
    return bool(re.match(pattern, arxiv_id))

def parse_ids(text: str) -> Tuple[List[str], List[str]]:
    if not text:
        return [], []
    raw = re.split(r"[,\n\s]+", text)
    all_entries = [entry.strip(" '\"") for entry in raw if entry.strip(" '\"")]
    valid_ids, invalid_entries = [], []
    for entry in all_entries:
        if validate_arxiv_id(entry):
            valid_ids.append(entry)
        else:
            invalid_entries.append(entry)
    return valid_ids, invalid_entries

async def submit_papers(
    user_id: str, 
    user_paper_metadata: Dict[str, Any],
    text_input: str
    ) -> Tuple[str, Any, Any, List[str], Dict]:

    valid_ids, invalid_entries = parse_ids(text_input)
    if not valid_ids and not invalid_entries:
        return "Please enter at least one ArXiv ID", gr.update(value=""), gr.update(), gr.update(), user_paper_metadata
    
    validation_message = ""
    if invalid_entries:
        if len(invalid_entries) == 1:
            validation_message = f"❌ Entry not matching arXiv ID format was ignored: {invalid_entries[0]}"
        else:
            validation_message = f"❌ {len(invalid_entries)} entries not matching arXiv ID format were ignored: {', '.join(invalid_entries[:5])}"
            if len(invalid_entries) > 5:
                validation_message += f" ... and {len(invalid_entries) - 5} more"
        validation_message += "\n\n"
    
    if not valid_ids:
        return validation_message + "No valid arXiv IDs to process.", gr.update(value=""), gr.update(), gr.update(), user_paper_metadata
    
    ArxivHubVectorstore = await get_vectorstore()
    unique_ids = list(set(valid_ids))
    result = await ingest_papers(user_id, user_paper_metadata, ArxivHubVectorstore ,unique_ids)
    
    clear_input = gr.update(value="")
    final_message = validation_message + result.get('message', "Ingestion completed.")
    
    # Update dataset
    new_samples, new_ids = prepare_dataset_samples(user_paper_metadata)
    
    return final_message, clear_input, gr.update(samples=new_samples), new_ids, user_paper_metadata

async def save_paper_notes(user_id: str, user_paper_metadata: Dict[str, Any], paper_id: str, notes: str):
    await save_notes(user_id, user_paper_metadata, paper_id, notes)
    return "Notes saved successfully!"

async def handle_delete_paper(user_id: str, user_paper_metadata: Dict[str, Any], paper_id: str):
    if not paper_id or paper_id == 'main_chat':
        return "Select a paper first", gr.update()
    
    ArxivHubVectorstore = await get_vectorstore()
    success = await delete_paper(user_id, user_paper_metadata, ArxivHubVectorstore, paper_id)
    
    if success:
        new_samples, new_ids = prepare_dataset_samples(user_paper_metadata)
        return "🗑️ Paper deleted", gr.update(samples=new_samples), new_ids, 'main_chat'
    return "❌ Delete failed", gr.update(), gr.update(), gr.update()