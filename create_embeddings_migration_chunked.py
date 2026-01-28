import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Create embeddings locally via HuggingFace Model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Load the CSV file
csv_path = "Production_Migration_COMBINED.csv"
df = pd.read_csv(csv_path)

print(f"Step 1: CSV loaded - {len(df)} records found")

# Convert entire CSV to a single text corpus (alternative approach)
full_text = ""
for idx, row in df.iterrows():
    # Create a structured text block for each record
    text_block = f"""
=== NAMESPACE: {row['namespace']} ===
Admins: {row['namespace_admins']}
Response: {row['received_response']}
Meeting Status: {row['migration_meeting']}
Migration Date: {row['migration_date']}
Migration Method: {row['migration_method']}
Status: {row['migration_status']}
Assigned Analyst: {row['assigned_analyst']}
Contact: {row['app_team_contact']}
Survey: {row['survey_completed']}
CMDB: {row['present_in_cmdb']}
Notes: {row['notes']}

"""
    full_text += text_block

print("Step 2: Text corpus created from CSV")

# Split text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,           # Similar to your original PDF settings
    chunk_overlap=100,        # Overlap to preserve context
    separators=["\n\n", "\n", " ", ""]
)

docs = splitter.create_documents([full_text])
print(f"Step 3: Total chunks created: {len(docs)}")

## Inspect chunks - optional
if len(docs) >= 2:
    chunk_1 = docs[0].page_content
    chunk_2 = docs[1].page_content
    print("\n========== CHUNK 1 ==========")
    print(chunk_1)
    print("\n========== CHUNK 2 ==========")
    print(chunk_2)
    
    print("\n========== EMBEDDING FOR CHUNK 1 ==========")
    embedding_chunk_1 = embeddings.embed_query(chunk_1)
    print("First 10 values:", embedding_chunk_1[:10])

# Store embeddings in FAISS
vectorstore = FAISS.from_documents(docs, embeddings)

# Save FAISS index
vectorstore.save_local("migration_faiss_index_chunked")

print("\n✓ Migration RAG index saved to 'migration_faiss_index_chunked' local folder")
print(f"✓ Total chunks indexed: {len(docs)}")
