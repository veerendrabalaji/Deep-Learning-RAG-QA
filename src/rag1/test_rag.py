


from src.rag1.hybrid_retriever import HybridRetriever


from src.rag1.prompt import PromptBuilder

# RAGPipeline connects retrieval -> context -> prompt -> LLM.
from src.rag1.Rag_pipeline import RAGPipeline

from src.rag1.vectorstore import FaissVectorStore

# Import your embedding pipeline if needed.
from src.rag1.embeddings import Embedding_Pipeline

# Import BM25 retriever.
from src.rag1.keyword_retriever import KeywordRetriever



# STEP 1: LOAD YOUR DOCUMENTS / CHUNKS

from src.rag1.data_ingestion import load_all_documents
documents = load_all_documents("data/pdf")

# STEP 2: CREATE CHUNKS

embedding_pipeline = Embedding_Pipeline()
chunks = embedding_pipeline.chunk_documents(documents)



# STEP 3: CREATE VECTOR STORE
vector_store = FaissVectorStore()

vector_store.load()


# STEP 4: CREATE BM25 RETRIEVER
keyword_retriever = KeywordRetriever(chunks)

# STEP 5: CREATE HYBRID RETRIEVER
hybrid_retriever = HybridRetriever(
    vector_store=vector_store,
    keyword_retriever=keyword_retriever
)

# STEP 6: CREATE PROMPT BUILDER
prompt_builder = PromptBuilder()

# STEP 7: CREATE RAG PIPELINE
rag = RAGPipeline(
    retriever=hybrid_retriever,
    prompt_builder=prompt_builder
)

# STEP 8: ASK A QUESTION
query = "What is dataset augmentation?"

# Run the complete RAG pipeline.
answer = rag.run(
    query=query,
    top_k=5
)

# STEP 9: DISPLAY FINAL ANSWER
print("\n" + "=" * 60)
print("RAG RESPONSE")
print("=" * 60)

print(f"\nQuestion: {query}")

print("\nAnswer:")
print(answer)

print("=" * 60)