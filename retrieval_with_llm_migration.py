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
    "migration_faiss_index_chunked",
    embeddings,
    allow_dangerous_deserialization=True
)

def answer_question(query):
    # Retrieve relevant documents
    results = vectorstore.similarity_search(query, k=3)
    
    # Combine retrieved chunks as context
    context = "\n\n".join([doc.page_content for doc in results])
    
    # Construct prompt for Gemini
    prompt = f"""
You are a Migration Assistant helping with ARO migration queries. Answer the user's question strictly based on the Migration Data context below.
Do NOT include any information that is not present in the provided context.
If the answer is not present in the context, respond with: "The information is not available in the migration records."

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
    
    return response.text.strip()  # remove extra whitespace

# -----------------------------
# Gradio UI
# -----------------------------
iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=1, placeholder="Ask about namespace migrations..."),
    outputs=gr.Textbox(lines=15, label="Answer"),
    title="Production Migration Q&A (RAG + Gemini)",
    description="Ask questions about namespace migrations. Answers are generated strictly from the production migration data.",
    examples=[
        ["What is the status of adsp-prod migration?"],
        ["Which namespaces used bluegreen migration method?"],
        ["Show me all completed migrations"],
        ["Who is the contact for ccds-prod?"],
        ["Which migrations happened on January 22nd?"],
        ["What are the migration notes for cssfb-prod?"]
    ]
)

iface.launch(server_name="0.0.0.0", server_port=7860)
