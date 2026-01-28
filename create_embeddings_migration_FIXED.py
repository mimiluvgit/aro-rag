import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document  # FIXED IMPORT

# Create embeddings locally via HuggingFace Model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Load the CSV file
csv_path = "Production_Migration_COMBINED.csv"
df = pd.read_csv(csv_path)

print(f"Step 1: CSV loaded - {len(df)} records found")

# Convert each row to a formatted text document
documents = []
for idx, row in df.iterrows():
    # Create a structured text representation of each migration record
    text = f"""Namespace: {row['namespace']}

Migration Details:
- Namespace Admins: {row['namespace_admins']}
- Received Response: {row['received_response']}
- Migration Meeting: {row['migration_meeting']}
- Migration Date: {row['migration_date']}
- Migration Method: {row['migration_method']}
- Migration Status: {row['migration_status']}

Team Information:
- Assigned Analyst: {row['assigned_analyst']}
- APP Team Primary Contact: {row['app_team_contact']}

Additional Information:
- Survey Completed: {row['survey_completed']}
- Present in CMDB: {row['present_in_cmdb']}
- Notes: {row['notes']}
"""
    
    # Create metadata for better retrieval
    metadata = {
        "namespace": row['namespace'],
        "migration_status": str(row['migration_status']),
        "migration_date": str(row['migration_date']),
        "migration_method": str(row['migration_method']),
        "row_index": idx
    }
    
    documents.append(Document(page_content=text, metadata=metadata))

print(f"Step 2: Created {len(documents)} documents from CSV records")

# Optional: Split long documents if needed
# For structured data like this, we typically keep each record as one chunk
# But if you want to split further, uncomment below:
"""
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)
docs = splitter.split_documents(documents)
print(f"Step 3: Total chunks created after splitting: {len(docs)}")
"""

# Use documents as-is (one document per namespace)
docs = documents
print(f"Step 3: Using {len(docs)} documents (one per namespace)")

## Inspect sample chunks - optional
if len(docs) >= 2:
    chunk_1 = docs[0].page_content
    chunk_2 = docs[1].page_content
    print("\n========== SAMPLE DOCUMENT 1 ==========")
    print(chunk_1)
    print("\n========== SAMPLE DOCUMENT 2 ==========")
    print(chunk_2)
    
    print("\n========== EMBEDDING FOR DOCUMENT 1 ==========")
    embedding_chunk_1 = embeddings.embed_query(chunk_1)
    print("First 10 values:", embedding_chunk_1[:10])

# Store embeddings in FAISS
vectorstore = FAISS.from_documents(docs, embeddings)

# Save FAISS index
vectorstore.save_local("migration_faiss_index")

print("\n✓ Migration RAG index saved to 'migration_faiss_index' local folder")
print(f"✓ Total documents indexed: {len(docs)}")
