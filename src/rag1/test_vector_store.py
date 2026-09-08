from src.rag1.data_ingestion import load_all_documents
from src.rag1.vectorstore import FaissVectorStore


# 1. Load documents
documents = load_all_documents("data")

print(f"\nDocuments loaded: {len(documents)}")


# 2. Create FAISS vector store
vector_store = FaissVectorStore(
    persist_dir="faiss_store"
)


# 3. Build the index
vector_store.build_from_documents(documents)


# 4. Check number of vectors
print(f"\nVectors in FAISS: {vector_store.index.ntotal}")


# 5. Search
query = "What is Biological Vision?"

results = vector_store.query(
    query_text=query,
    top_k=5
)


# 6. Display results
print("\n" + "=" * 60)
print("SEARCH RESULTS")
print("=" * 60)

for i, result in enumerate(results, start=1):

    print(f"\nRESULT {i}")
    print("-" * 40)

    print("Distance:", result["distance"])

    print("Metadata:")
    print(result["metadata"])

    print("\nText:")
    print(result["metadata"]["text"][:500])