import os
import pandas as pd
import gradio as gr
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from google import genai

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
client = genai.Client()

# -----------------------------
# Load FAISS vectorstore
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "migration_faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# -----------------------------
# Load CSV for aggregation queries
# -----------------------------
# For counting/listing ALL matches, we need direct CSV access
df = pd.read_csv("Production_Migration_COMBINED.csv")

def detect_aggregation_query(query):
    """
    Detect if this is a query that needs ALL results (counting, listing, etc.)
    """
    aggregation_keywords = [
        'all', 'list', 'show me all', 'how many', 'count', 
        'which namespaces', 'what namespaces', 'every', 'total'
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in aggregation_keywords)

def answer_aggregation_query(query):
    """
    Handle queries that need to search ALL records (not just top k)
    """
    query_lower = query.lower()
    
    # Filter based on common query patterns
    results = df.copy()
    
    # Filter by migration method
    if 'bluegreen' in query_lower or 'blue green' in query_lower:
        results = results[results['migration_method'].str.lower().str.contains('bluegreen', na=False)]
    elif 'default' in query_lower:
        results = results[results['migration_method'].str.lower().str.contains('default', na=False)]
    
    # Filter by status
    if 'completed' in query_lower:
        results = results[results['migration_status'].str.lower().str.contains('completed', na=False)]
    elif 'in progress' in query_lower or 'progress' in query_lower:
        results = results[results['migration_status'].str.lower().str.contains('progress', na=False)]
    
    # Filter by analyst
    if 'amrutha' in query_lower or 'bobby' in query_lower:
        results = results[results['assigned_analyst'].str.lower().str.contains('amrutha|bobby', na=False, regex=True)]
    
    # Filter by date (simple matching)
    if 'january 22' in query_lower or 'jan 22' in query_lower or '22nd' in query_lower:
        results = results[results['migration_date'].str.contains('22', na=False)]
    elif 'january 20' in query_lower or 'jan 20' in query_lower or '20th' in query_lower:
        results = results[results['migration_date'].str.contains('20', na=False)]
    elif 'january 21' in query_lower or 'jan 21' in query_lower or '21st' in query_lower:
        results = results[results['migration_date'].str.contains('21', na=False)]
    
    if len(results) == 0:
        return "No matching namespaces found based on your criteria."
    
    # Format the response
    response = f"Found {len(results)} matching namespace(s):\n\n"
    
    for idx, row in results.iterrows():
        response += f"**{row['namespace']}**\n"
        if row['migration_status']:
            response += f"  • Status: {row['migration_status']}\n"
        if row['migration_date']:
            response += f"  • Date: {row['migration_date']}\n"
        if row['migration_method']:
            response += f"  • Method: {row['migration_method']}\n"
        if row['assigned_analyst']:
            response += f"  • Analyst: {row['assigned_analyst']}\n"
        if row['app_team_contact']:
            response += f"  • Contact: {row['app_team_contact']}\n"
        response += "\n"
    
    return response

def answer_question(query, num_results=3):
    """Answer migration-related questions using RAG or direct CSV search."""
    
    # Check if this is an aggregation query
    if detect_aggregation_query(query):
        return answer_aggregation_query(query)
    
    # Regular RAG query for specific questions
    results = vectorstore.similarity_search(query, k=num_results)
    context = "\n\n".join([doc.page_content for doc in results])
    
    retrieved_namespaces = []
    for doc in results:
        if hasattr(doc, 'metadata') and 'namespace' in doc.metadata:
            retrieved_namespaces.append(doc.metadata['namespace'])
    
    prompt = f"""
You are a Migration Assistant helping with namespace migration queries. Answer the user's question based on the Migration Data context below.

IMPORTANT INSTRUCTIONS:
1. If a field is empty or blank in the context, explicitly say "No [field name] recorded" or "[field name] not specified"
2. Do NOT say "information is not available" if you found the namespace - instead specify which fields are present and which are empty
3. Be specific about what information IS available, even if some fields are empty
4. If you find the namespace but a specific field is empty, acknowledge that you found the namespace and state which field is missing

Context:
{context}

Question: {query}

Answer (be specific about what IS available and what ISN'T):
"""
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    
    answer = response.text.strip()
    
    if retrieved_namespaces:
        footer = f"\n\n---\n📌 Sources checked: {', '.join(retrieved_namespaces)}"
        answer += footer
    
    return answer

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="Migration Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔄 Production Migration Q&A Assistant
    Ask questions about namespace migrations. Powered by RAG + Gemini AI.
    
    **NEW:** Now handles counting and listing queries accurately!
    """)
    
    with gr.Tab("Ask Questions"):
        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    lines=2,
                    placeholder="Example: Which namespaces used bluegreen migration method?",
                    label="Your Question"
                )
                num_results_slider = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="Number of sources to retrieve (for specific queries)",
                    info="Ignored for counting/listing queries"
                )
                submit_btn = gr.Button("Get Answer", variant="primary")
            
            with gr.Column(scale=3):
                answer_output = gr.Textbox(
                    lines=20,
                    label="Answer",
                    show_copy_button=True
                )
        
        gr.Examples(
            examples=[
                ["Which namespaces used bluegreen migration method?"],
                ["Show me all completed migrations"],
                ["List all namespaces assigned to Amrutha and Bobby"],
                ["Which migrations happened on January 22nd?"],
                ["How many namespaces used default method?"],
                ["What is the status of adsp-prod migration?"],
                ["Who is the contact for ccds-prod?"],
                ["What are the migration notes for cssfb-prod?"],
            ],
            inputs=query_input
        )
        
        submit_btn.click(
            fn=answer_question,
            inputs=[query_input, num_results_slider],
            outputs=answer_output
        )
        
        query_input.submit(
            fn=answer_question,
            inputs=[query_input, num_results_slider],
            outputs=answer_output
        )
    
    with gr.Tab("Query Types"):
        gr.Markdown("""
        ## Two Types of Queries Supported
        
        ### 1. Aggregation Queries (Searches ALL 57 records)
        These queries need to search the entire dataset:
        - **"Which namespaces used bluegreen method?"** → Lists ALL matching namespaces
        - **"Show me all completed migrations"** → Lists ALL completed
        - **"How many namespaces..."** → Counts ALL matches
        - **"List all namespaces assigned to X"** → Lists ALL matches
        
        **Triggers:** Keywords like "all", "list", "which namespaces", "how many", "show me all"
        
        ### 2. Specific Queries (Uses semantic search)
        These ask about specific namespaces or details:
        - **"What is the status of adsp-prod?"** → Specific namespace query
        - **"Who is the contact for ccds-prod?"** → Specific information
        - **"What are the notes for X?"** → Specific field
        
        ## Supported Filters
        
        When using aggregation queries, you can filter by:
        - **Method:** bluegreen, default
        - **Status:** completed, in progress
        - **Analyst:** Amrutha, Bobby
        - **Date:** January 22nd, Jan 20th, 21st, etc.
        
        ## Examples
        
        ✅ "Which namespaces used bluegreen migration?" → Returns all ~5-10 namespaces  
        ✅ "Show all completed migrations" → Returns all ~18 completed namespaces  
        ✅ "List namespaces migrated on January 22nd" → Returns specific date matches  
        ✅ "What is the status of adsp-prod?" → Returns specific namespace info  
        """)
    
    with gr.Tab("About"):
        gr.Markdown("""
        ## Hybrid Search Approach
        
        This chatbot now uses a **hybrid approach**:
        
        1. **Aggregation queries** → Direct CSV filtering (accurate counts)
        2. **Specific queries** → RAG with semantic search (detailed answers)
        
        ## Data Source
        - **Records**: 57 production namespaces
        - **Completed migrations**: ~18
        - **Methods**: Bluegreen, Default
        - **Last Updated**: January 27, 2026
        """)

demo.launch(server_name="0.0.0.0", server_port=7860)
