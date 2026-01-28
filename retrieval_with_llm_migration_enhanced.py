import os
import gradio as gr
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from google import genai  # new SDK

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
# Make sure GEMINI_API_KEY is set in .env
# GEMINI_API_KEY=your_key_here
client = genai.Client()  # Reads API key from env variable

# -----------------------------
# Load FAISS vectorstore
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "migration_faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

def answer_question(query, num_results=3):
    """
    Answer migration-related questions using RAG.
    
    Args:
        query: User's question
        num_results: Number of relevant documents to retrieve (1-5)
    """
    # Retrieve relevant documents
    results = vectorstore.similarity_search(query, k=num_results)
    
    # Combine retrieved chunks as context
    context = "\n\n".join([doc.page_content for doc in results])
    
    # Show which namespaces were retrieved (for transparency)
    retrieved_namespaces = []
    for doc in results:
        if hasattr(doc, 'metadata') and 'namespace' in doc.metadata:
            retrieved_namespaces.append(doc.metadata['namespace'])
    
    # Construct prompt for Gemini
    prompt = f"""
You are a Migration Assistant helping with namespace migration queries. Answer the user's question strictly based on the Migration Data context below.

Instructions:
- Provide accurate information from the context
- Be specific about namespace names, dates, and statuses
- If migration notes are available, include them
- If the answer is not in the context, say: "The information is not available in the migration records."
- Format your response clearly with proper sections if needed

Context:
{context}

Question: {query}

Answer:
"""
    # Generate answer using Gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    
    answer = response.text.strip()
    
    # Add footer with retrieved namespaces for transparency
    if retrieved_namespaces:
        footer = f"\n\n---\n📌 Sources: {', '.join(retrieved_namespaces)}"
        answer += footer
    
    return answer

# -----------------------------
# Gradio UI with Tabs
# -----------------------------
with gr.Blocks(title="Migration Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔄 Production Migration Q&A Assistant
    Ask questions about namespace migrations. Powered by RAG + Gemini AI.
    """)
    
    with gr.Tab("Ask Questions"):
        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    lines=2,
                    placeholder="Example: What is the status of adsp-prod migration?",
                    label="Your Question"
                )
                num_results_slider = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="Number of sources to retrieve",
                    info="More sources = more context but slower response"
                )
                submit_btn = gr.Button("Get Answer", variant="primary")
            
            with gr.Column(scale=3):
                answer_output = gr.Textbox(
                    lines=15,
                    label="Answer",
                    show_copy_button=True
                )
        
        # Example questions
        gr.Examples(
            examples=[
                ["What is the status of adsp-prod migration?"],
                ["Which namespaces used bluegreen migration method?"],
                ["Show me all completed migrations"],
                ["Who is the contact for ccds-prod?"],
                ["Which migrations happened on January 22nd?"],
                ["What are the migration notes for cssfb-prod?"],
                ["List all namespaces assigned to Amrutha and Bobby"],
                ["Which namespaces are still in progress?"]
            ],
            inputs=query_input
        )
        
        submit_btn.click(
            fn=answer_question,
            inputs=[query_input, num_results_slider],
            outputs=answer_output
        )
        
        # Also allow Enter key to submit
        query_input.submit(
            fn=answer_question,
            inputs=[query_input, num_results_slider],
            outputs=answer_output
        )
    
    with gr.Tab("About"):
        gr.Markdown("""
        ## How it works
        
        1. **Your Question**: Enter a question about namespace migrations
        2. **Retrieval**: The system searches through 57 migration records using AI embeddings
        3. **Context**: Top 3-5 most relevant records are retrieved
        4. **Answer**: Gemini AI generates an answer based only on the retrieved data
        
        ## Sample Questions You Can Ask
        
        ### Status & Progress
        - What is the migration status of [namespace]?
        - Which migrations are completed?
        - Show me in-progress migrations
        
        ### Dates & Timeline
        - When was [namespace] migrated?
        - Which migrations happened in January?
        - Show migrations scheduled for a specific date
        
        ### Methods & Teams
        - Which namespaces used bluegreen migration?
        - Who is the analyst for [namespace]?
        - List all migrations assigned to [analyst name]
        
        ### Details & Notes
        - What are the migration notes for [namespace]?
        - Who is the primary contact for [namespace]?
        - Show me details about [namespace] migration
        
        ## Data Source
        - **Records**: 57 production namespaces
        - **Last Updated**: January 27, 2026
        - **Embedding Model**: all-MiniLM-L6-v2
        - **LLM**: Google Gemini 2.5 Flash
        """)

# Launch the interface
demo.launch(server_name="0.0.0.0", server_port=7860)
