import re
import gradio as gr
from gradio_modal import Modal
from pathlib import Path
from conversationagent import ConversationAgent, embedder
from langchain_community.vectorstores import FAISS

from paperingestion import PaperVectorStore

# ====== Load user docstore ======
user_id = "demo_user"
user_store = PaperVectorStore(user_id)
user_vectorstore = user_store.vectorstore # user vectorstore
agent = ConversationAgent(user_id=user_id, docstore=user_vectorstore) #TODO: agent should be independent of user ID ? 

# ====== Sample papers ======
papers = [
    {
        "id": 1,
        "title": "Attention Is All You Need", 
        "authors": "Ashish Vaswani et al.",
        "year": 2017, 
        "tags": ["Deep Learning", "NLP"],
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
        "content": "Full paper content would go here...",
        "citations": 85000,
        "conference": "NeurIPS"
    },
    {
        "id": 2,
        "title": "BERT: Pre-training of Deep Bidirectional Transformers", 
        "authors": "Jacob Devlin et al.",
        "year": 2018, 
        "tags": ["NLP", "Transformers"],
        "abstract": "We introduce a new language representation model called BERT...",
        "content": "Full paper content would go here...",
        "citations": 45000,
        "conference": "NAACL"
    },
]

# ====== Callbacks ======
def open_paper_detail(paper_id):
    paper_id = int(paper_id)
    paper = next((p for p in papers if p['id'] == paper_id), None)
    if not paper:
        return gr.update(visible=True), gr.update(visible=False), "", "", gr.update(), gr.update()
    
    return (
        gr.update(visible=False),       # Hide main chat
        gr.update(visible=True),        # Show paper detail
        f"## {paper['title']}",         # Title
        paper["abstract"],              # Content / abstract
        gr.update(),                     # Leave Chatbot as-is for now
        gr.update()                      # Leave Notes as-is
    )

def back_to_main():
    return gr.update(visible=True), gr.update(visible=False)

# ====== Interface ======
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    current_paper = gr.State(value=None)

    # ------------------- Main Chat -------------------
    with gr.Column(scale=2) as main_chat:
        gr.Markdown("## PaperPal\nYour AI Research Assistant")
        initial_msg = "Hello! I am a document chat agent here to help you!\n\nHow can I help you?"
        chatbot = gr.Chatbot(value=[{"role": "assistant", "content": initial_msg}], type="messages")
        generalchat = gr.ChatInterface(agent.chat_gen, chatbot=chatbot).queue()

    # ------------------- Paper Detail View -------------------
    with gr.Column(visible=False, elem_id="paper_detail") as paper_detail:
        with gr.Row():
            back_button = gr.Button("← Back to Chat")
            with gr.Column(scale=10, min_width=0, elem_id="paper_title_col"):
                paper_title = gr.Markdown("## Paper Title", elem_id="paper_title_md")

        with gr.Tabs() as paper_tabs:
            with gr.TabItem("Read"):
                paper_content = gr.Markdown()
            with gr.TabItem("Chat"):
                paper_chatbot = gr.Chatbot(type="messages")
                paper_msg = gr.Textbox(
                    label="Chat about this paper...",
                    placeholder="Ask specific questions about this paper",
                    lines=1
                )
                # TODO: Wire paper_msg to paper_chatbot

            with gr.TabItem("Notes"):
                paper_notes = gr.Textbox(
                    label="Your Notes",
                    placeholder="Add your notes about this paper...",
                    lines=10
                )
                save_notes_btn = gr.Button("Save Notes")
                # TODO: Save notes per paper
        #  Back button 
        back_button.click(fn=back_to_main, outputs=[main_chat, paper_detail])

    # ------------------- Sidebar -------------------
    with gr.Sidebar() as sidebar:
        gr.Markdown("## Research Papers")
        search_box = gr.Textbox(label="Search your paper inventory", placeholder="Search papers...")

        # Each paper as a button
        for paper in papers:
            btn = gr.Button(f"{paper['title']} • {paper['authors']} • {paper['year']}")
            btn.click(
                fn=lambda pid=paper['id']: open_paper_detail(pid),
                inputs=[],
                outputs=[main_chat, paper_detail, paper_title, paper_content, paper_chatbot, paper_notes]
            )

        # Add Paper Modal
        add_papers_button = gr.Button("+ Add Papers")
        with Modal(elem_id="add_paper_modal") as add_paper_modal:
            gr.Markdown("### Add new papers")
            # State to store the list of IDs
            arxiv_ids_state = gr.State(value=[])
            # Markdown to show current IDs
            current_ids_display = gr.Markdown("**Current IDs:** None")
            # Textbox for entering IDs
            arxiv_ids_input = gr.Textbox(
            label="Enter ArXiv IDs",
            placeholder="One per line or comma-separated",
            lines=3)
            # Confirm and ingest the IDs 
            submit_btn = gr.Button("Submit") 
            feedback_markdown = gr.Markdown("") 
  
            def validate_arxiv_id(arxiv_id: str) -> bool:
                """
                Validate if string matches arXiv ID pattern.
                Supports both old and new arXiv ID formats.
                """
                # Enhanced pattern for arXiv IDs
                pattern = r'^(\d{4}\.\d{4,5}(v\d+)?|[a-z]+(-[a-z]+)*/\d{7}(v\d+)?)$'
                return bool(re.match(pattern, arxiv_id))

            def parse_ids(text):
                if not text: 
                    return [], []
                
                # split by comma, newline, space
                raw = re.split(r"[,\n\s]+", text)
                # cleanup + remove empty
                all_entries = [entry.strip(" '\"") for entry in raw if entry.strip(" '\"")]
                
                # Separate valid arXiv IDs from invalid ones
                valid_ids = []
                invalid_entries = []
                
                for entry in all_entries:
                    if validate_arxiv_id(entry):
                        valid_ids.append(entry)
                    else:
                        invalid_entries.append(entry)
                
                return valid_ids, invalid_entries

            def submit_papers(text_input):
                valid_ids, invalid_entries = parse_ids(text_input)
                
                if not valid_ids and not invalid_entries:
                    return "Please enter at least one ArXiv ID", [], gr.update(value="")
                
                # Build validation message
                validation_message = ""
                if invalid_entries:
                    if len(invalid_entries) ==1: 
                        validation_message = f"❌ Entry not matching arXiv ID format was ignored: {', '.join(invalid_entries[0])}"
                    else :
                        validation_message = f"❌ {len(invalid_entries)} entries not matching arXiv ID format were ignored: {', '.join(invalid_entries[:5])}"
                    if len(invalid_entries) > 5:
                        validation_message += f" ... and {len(invalid_entries) - 5} more"
                    validation_message += "\n\n"
                
                if not valid_ids:
                    return validation_message + "No valid arXiv IDs to process.", [], gr.update(value="")
                
                # Remove duplicates from valid IDs
                unique_ids = list(set(valid_ids))

                # Ingest papers 
                result = user_store.ingest_papers(unique_ids)
                
                # Clear the input field after successful submission
                clear_input = gr.update(value="")
                
                # Combine validation message with ingestion results
                final_message = validation_message + result['message']
                
                return final_message, [], clear_input
            
            submit_btn.click(
                fn=submit_papers,
                inputs=[arxiv_ids_input],
                outputs=[feedback_markdown,arxiv_ids_state,arxiv_ids_input]
            )

            def open_add_paper_modal():
                # Reset everything
                return gr.update(visible=True), [], "", "**Current IDs:** None", gr.update(value="")

            add_papers_button.click(
                fn=open_add_paper_modal,
                inputs=[],
                outputs=[add_paper_modal, arxiv_ids_state, arxiv_ids_input, current_ids_display, feedback_markdown]
            )
    

demo.launch()
