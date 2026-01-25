import re
import gradio as gr
from gradio_modal import Modal
from graph import workflow as rag_workflow
from config import get_vectorstore
from core.schemas import RuntimeContext
from ingestion import load_paper_metadata 
from ui import (
    prepare_dataset_samples,
    open_paper_detail_from_dataset,
    back_to_main,
    submit_papers,
    save_paper_notes
)

async def chat_with_agent(message, history, user_id, user_metadata):
    """
    The bridge between Gradio and LangGraph.
    """
    # Runtime context
    runtime_context = RuntimeContext(
        user_id=user_id,
        vectorstore=await get_vectorstore(),
        metadata=user_metadata,
    )
    # Config for LangGraph (Thread isolation)
    config = {"configurable": {"thread_id": user_id}}
    
    inputs = {"messages": [("user", message)]}
    
    async for event in rag_workflow.astream(inputs, config=config, context=runtime_context):
        for node_name, value in event.items():
            if "finalAnswer" in value:
                yield value["finalAnswer"]
            
            elif "messages" in value:
                last_msg = value["messages"][-1]
                # If it's a message object, get .content, else stringify
                content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
                yield content

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    
    # Session States
    user_id = gr.State() 
    user_metadata = gr.State({})
    paper_ids_state = gr.State(value=[])
    current_selection = gr.State(value='main_chat')

    # ------------------- Main Chat -------------------
    with gr.Column(scale=2) as main_chat:
        gr.Markdown("## ArXivHub\nYour AI Research Assistant")
        initial_msg = "Hello! I am ArXivHub your research buddy!\n\nHow can I help you?"
        chatbot = gr.Chatbot(value=[{"role": "assistant", "content": initial_msg}], type="messages")
        generalchat = gr.ChatInterface(
            chat_with_agent, 
            chatbot=chatbot, 
            additional_inputs=[user_id, user_metadata]
            ).queue()

    # ------------------- Paper Detail View -------------------
    with gr.Column(visible=False) as paper_detail:
        with gr.Row():
            back_button = gr.Button("← Back to Chat")
            with gr.Column(scale=10, min_width=0):
                paper_title = gr.Markdown("## Paper Title")
        
        with gr.Tabs():
            with gr.TabItem("Read"):
                paper_content = gr.HTML()
            with gr.TabItem("Chat"):
                paper_chatbot = gr.Chatbot(type="messages")
                paper_msg = gr.Textbox(label="Chat about this paper...", placeholder="Ask questions...", lines=1)
            with gr.TabItem("Notes"):
                paper_notes = gr.Textbox(label="Your Notes", placeholder="Add notes...", lines=10)
                feedback_markdown_for_notes = gr.Markdown() 
                save_notes_btn = gr.Button("Save Notes")
        
            save_notes_btn.click(
            fn=save_paper_notes,
            inputs=[user_id, user_metadata, current_selection, paper_notes],  
            outputs=[feedback_markdown_for_notes]
            )

        back_button.click(
            fn=back_to_main,
            outputs=[current_selection, main_chat, paper_detail, paper_title, paper_content, paper_chatbot, paper_notes]
        )

    # ------------------- Sidebar -------------------
    with gr.Sidebar():
        gr.Markdown("## Research Papers")
        search_box = gr.Textbox(label="Search papers", placeholder="Search...")
        
        # Use Dataset for clickable paper list
        papers_dataset = gr.Dataset(
            components=[gr.Textbox(visible=False)],
            samples=[["Loading papers..."]],  # Placeholder
            label="Click a paper to view",
            samples_per_page=10
        )
        
        # Handle dataset clicks
        papers_dataset.select(
            fn=open_paper_detail_from_dataset,
            inputs=[user_metadata, paper_ids_state],
            outputs=[current_selection, main_chat, paper_detail, paper_title, paper_content, paper_chatbot, paper_notes]
        )

        add_papers_button = gr.Button("+ Add Papers")
        with Modal(visible=False) as add_paper_modal:
            gr.Markdown("### Add new papers")
            arxiv_ids_input = gr.Textbox(
                label="Enter ArXiv IDs",
                placeholder="One per line or comma-separated",
                lines=3
            )
            feedback_markdown = gr.Markdown("")
            submit_btn = gr.Button("Submit")
            
            submit_btn.click(
                fn=submit_papers,
                inputs=[user_id, user_metadata, arxiv_ids_input],
                outputs=[feedback_markdown, arxiv_ids_input, papers_dataset, paper_ids_state, user_metadata]
            )
            
            add_papers_button.click(
                fn=lambda: (gr.update(visible=True), "", ""),
                outputs=[add_paper_modal, arxiv_ids_input, feedback_markdown]
            )
    
    # ------------------- Load papers on startup -------------------
    async def on_start():
        """Load papers when the app starts/refreshes"""
        
        active_id = "demo_user" #TODO Get from request.username or login (in production) 
        
        # JSON metadata for the user
        meta = await load_paper_metadata(active_id) 
        # Paper inventory display 
        samples, ids = prepare_dataset_samples(meta)
        
        return active_id, meta, gr.update(samples=samples if samples else [["No papers yet"]]), ids
    
    # Trigger on demo load
    demo.load(
        fn=on_start,
        outputs=[user_id, user_metadata, papers_dataset, paper_ids_state]
    )

demo.launch()