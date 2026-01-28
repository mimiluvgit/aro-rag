# Production Migration RAG System - Setup Guide

## Overview
This guide helps you create embeddings from the Production Migration CSV data and build a chatbot that can answer questions about namespace migrations.

## Files Provided

### 1. Data File
- **Production_Migration_COMBINED.csv** - Your cleaned migration data (57 namespaces)

### 2. Embedding Creation Scripts

#### Option A: `create_embeddings_migration.py` (RECOMMENDED)
- Creates one document per namespace (57 total documents)
- Preserves complete context for each migration
- Includes structured metadata for better filtering
- Best for: Precise queries about specific namespaces

#### Option B: `create_embeddings_migration_chunked.py`
- Creates smaller chunks by splitting text (similar to your PDF approach)
- May create 20-40 chunks depending on data
- Best for: More granular retrieval across fields

### 3. Query Scripts

#### Terminal-Based
- **query_migration_rag.py** - Command-line testing with sample queries

#### Web UI (Gradio)
- **retrieval_with_llm_migration.py** - Basic UI with Gemini AI (requires API key)
- **retrieval_with_llm_migration_enhanced.py** - Advanced UI with tabs and controls
- **retrieval_no_llm_migration.py** - Search-only UI (no API key needed)

## Setup Instructions

### Step 1: Install Dependencies
```bash
pip install pandas langchain langchain-community faiss-cpu sentence-transformers
```

### Step 2: Copy Files to Your Working Directory
Place these files in the same folder:
- `Production_Migration_COMBINED.csv`
- `create_embeddings_migration.py` (or the chunked version)
- `query_migration_rag.py`

### Step 3: Create Embeddings
Run the embedding script:
```bash
python create_embeddings_migration.py
```

This will:
1. Load the CSV (57 records)
2. Convert each row to a structured text document
3. Create embeddings using HuggingFace model (all-MiniLM-L6-v2)
4. Store in FAISS index: `migration_faiss_index/`

**Expected output:**
```
Step 1: CSV loaded - 57 records found
Step 2: Created 57 documents from CSV records
Step 3: Using 57 documents (one per namespace)
✓ Migration RAG index saved to 'migration_faiss_index' local folder
```

### Step 4: Test Your Chatbot

#### Option A: Terminal-Based Testing
Run the query script:
```bash
python query_migration_rag.py
```

This will:
- Run 6 pre-defined test queries
- Show top 3 results for each query
- Enter interactive mode for custom questions

#### Option B: Web UI (Recommended)

**Without API Key (Search Only):**
```bash
python retrieval_no_llm_migration.py
```
- Opens browser at http://localhost:7860
- Pure semantic search - no LLM needed
- Great for testing retrieval quality

**With Gemini API Key (Full Chatbot):**

1. Create a `.env` file:
```
GEMINI_API_KEY=your_api_key_here
```

2. Install additional dependencies:
```bash
pip install python-dotenv google-genai gradio
```

3. Run the chatbot:
```bash
# Basic version
python retrieval_with_llm_migration.py

# Or enhanced version with better UI
python retrieval_with_llm_migration_enhanced.py
```

Opens browser at http://localhost:7860 with a full Q&A interface.

## Sample Queries You Can Ask

### Status Queries
- "What is the status of adsp-prod migration?"
- "Show me all completed migrations"
- "Which migrations are in progress?"

### Method Queries
- "Which namespaces used bluegreen migration method?"
- "Show me all default migrations"

### Date Queries
- "Which migrations happened on January 20th?"
- "When was ccds-prod migrated?"

### Contact Queries
- "Who is the contact for tenet-prod?"
- "Which analyst handled crs-prod?"

### Detail Queries
- "What are the migration notes for cssfb-prod?"
- "Show me details about dcp-prod migration"

## Key Differences from Your PDF Code

### What Changed:
1. **Input Source**: CSV instead of PDF
   - No `requests.get()` or PDF download
   - Uses `pandas.read_csv()`

2. **Text Extraction**: Structured data parsing
   - No `fitz` (PyMuPDF) needed
   - Each row becomes a formatted document

3. **Metadata**: Added rich metadata
   ```python
   metadata = {
       "namespace": row['namespace'],
       "migration_status": str(row['migration_status']),
       "migration_date": str(row['migration_date']),
       "migration_method": str(row['migration_method'])
   }
   ```

4. **Document Structure**: Better formatting
   - Clear sections for each namespace
   - Organized into categories (Migration Details, Team Info, etc.)

### What Stayed the Same:
- Same embedding model: `all-MiniLM-L6-v2`
- Same vector store: FAISS
- Same retrieval approach: similarity search

## Choosing Between the Two Approaches

### Use `create_embeddings_migration.py` if:
✅ You want exact namespace-level retrieval
✅ Queries like "Tell me about namespace X"
✅ Need complete context per namespace
✅ Working with structured data

### Use `create_embeddings_migration_chunked.py` if:
✅ You have very large text fields
✅ Want more granular field-level retrieval
✅ Prefer the chunking approach you used for PDFs
✅ Need to split long notes/descriptions

## Integrating with Your Chatbot

### Basic Integration
```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load index
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "migration_faiss_index", 
    embeddings,
    allow_dangerous_deserialization=True
)

# Query
def answer_question(question):
    results = vectorstore.similarity_search(question, k=3)
    return results[0].page_content  # Return top result
```

### With LangChain Q&A Chain
```python
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=your_llm,
    chain_type="stuff",
    retriever=retriever
)

# Ask questions
answer = qa_chain.run("What is the status of adsp-prod?")
```

## Troubleshooting

### Error: "No module named 'faiss'"
```bash
pip install faiss-cpu
```

### Error: "allow_dangerous_deserialization=True required"
This is expected for FAISS. The parameter is already included in the query script.

### Poor Results?
Try adjusting:
- `k` value in similarity_search (default is 3)
- chunk_size in the chunked version (default 600)
- Consider using metadata filters

## Performance Notes

- **Index Size**: ~2-3 MB for 57 documents
- **Load Time**: <1 second
- **Query Time**: ~100-200ms per query
- **Model Size**: ~80 MB (downloads once)

## Next Steps

1. **Test different queries** to verify retrieval quality
2. **Integrate with your LLM** (GPT, Claude, Llama, etc.)
3. **Add filtering** by migration_status or date
4. **Deploy** to your chatbot platform

## Need Help?

Common issues:
- Make sure CSV is in the same directory as the script
- Verify all dependencies are installed
- Check Python version (3.8+)
- Ensure enough disk space for FAISS index

Happy building! 🚀
