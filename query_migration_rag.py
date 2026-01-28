from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load the same embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Load the FAISS index
vectorstore = FAISS.load_local(
    "migration_faiss_index", 
    embeddings,
    allow_dangerous_deserialization=True
)

print("✓ FAISS index loaded successfully\n")

# Sample queries
queries = [
    "What is the status of adsp-prod migration?",
    "Which namespaces used bluegreen migration method?",
    "Show me all completed migrations",
    "Who is the contact for ccds-prod?",
    "Which migrations are scheduled for January 22nd?",
    "What are the migration notes for cssfb-prod?"
]

print("="*80)
print("TESTING MIGRATION CHATBOT QUERIES")
print("="*80)

for query in queries:
    print(f"\n📝 Query: {query}")
    print("-" * 80)
    
    # Retrieve top 3 most relevant documents
    results = vectorstore.similarity_search(query, k=3)
    
    for i, doc in enumerate(results, 1):
        print(f"\n🔍 Result {i}:")
        print(doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content)
        
        # Show metadata if available
        if doc.metadata:
            print(f"\nMetadata: {doc.metadata}")
    
    print("\n" + "="*80)

# Interactive query mode
print("\n\n💬 INTERACTIVE MODE")
print("Type your questions about migrations (or 'quit' to exit)")
print("-"*80)

while True:
    user_query = input("\nYour question: ").strip()
    
    if user_query.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
    
    if not user_query:
        continue
    
    results = vectorstore.similarity_search(user_query, k=2)
    
    print("\n📋 Answer:")
    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
        if doc.metadata:
            print(f"\nMetadata: {doc.metadata}")
