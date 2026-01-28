import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# -----------------------------
# Load FAISS vectorstore
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "migration_faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

def search_migrations(query, num_results=3):
    """
    Search migration records and return formatted results.
    No LLM needed - just pure retrieval.
    """
    # Retrieve relevant documents
    results = vectorstore.similarity_search(query, k=num_results)
    
    if not results:
        return "No matching migration records found."
    
    # Format the results
    output = f"Found {len(results)} relevant migration record(s):\n\n"
    output += "=" * 80 + "\n\n"
    
    for i, doc in enumerate(results, 1):
        output += f"📋 RESULT {i}\n"
        output += "-" * 80 + "\n"
        output += doc.page_content
        output += "\n"
        
        # Add metadata if available
        if hasattr(doc, 'metadata') and doc.metadata:
            output += f"\nMetadata: {doc.metadata}\n"
        
        output += "\n" + "=" * 80 + "\n\n"
    
    return output

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="Migration Search", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔍 Production Migration Search
    Search through namespace migration records using AI-powered semantic search.
    **No API key required** - this uses local embeddings only.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            query_input = gr.Textbox(
                lines=2,
                placeholder="Example: Show me bluegreen migrations",
                label="Search Query"
            )
            num_results_slider = gr.Slider(
                minimum=1,
                maximum=10,
                value=3,
                step=1,
                label="Number of results to show"
            )
            search_btn = gr.Button("Search", variant="primary")
        
        with gr.Column(scale=3):
            results_output = gr.Textbox(
                lines=20,
                label="Search Results",
                show_copy_button=True
            )
    
    # Example searches
    gr.Examples(
        examples=[
            ["adsp-prod migration"],
            ["bluegreen migration method"],
            ["completed migrations"],
            ["January 22nd migrations"],
            ["Amrutha and Bobby"],
            ["migration notes cssfb-prod"]
        ],
        inputs=query_input
    )
    
    search_btn.click(
        fn=search_migrations,
        inputs=[query_input, num_results_slider],
        outputs=results_output
    )
    
    query_input.submit(
        fn=search_migrations,
        inputs=[query_input, num_results_slider],
        outputs=results_output
    )
    
    gr.Markdown("""
    ---
    ### 💡 Tips
    - Use natural language queries like "show me all completed migrations"
    - Search by namespace name, date, method, or analyst name
    - Increase the number of results to see more matches
    - This uses semantic search, so similar concepts will match even if exact words differ
    """)

demo.launch(server_name="0.0.0.0", server_port=7860)
