import json

from src.rag1.data_ingestion import load_all_documents
from src.rag1.embeddings import Embedding_Pipeline
from src.rag1.vectorstore import FaissVectorStore
from src.rag1.keyword_retriever import KeywordRetriever
from src.rag1.hybrid_retriever import HybridRetriever

# Configuration


DATASET_PATH = "evaluation/evaluation_dataset.json"
OUTPUT_PATH = "evaluation/retrieval_candidates.json"

TOP_K = 10
RETRIEVAL_K = 20



# Main

def main():

    # 1. Load evaluation dataset
   

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"[INFO] Loaded {len(questions)} evaluation questions.")

   
    # 2. Load documents
    

    documents = load_all_documents("data/pdf")

   
    # 3. Create chunks


    embedding_pipeline = Embedding_Pipeline()

    chunks = embedding_pipeline.chunk_documents(documents)

    print(f"[INFO] Total chunks: {len(chunks)}")

    # 4. Create BM25 retriever
   

    keyword_retriever = KeywordRetriever(chunks)

   
    # 5. Load FAISS
 

    vector_store = FaissVectorStore()

    vector_store.load()

    print("[INFO] FAISS index loaded.")

    
    # 6. Create Hybrid Retriever


    retriever = HybridRetriever(
        vector_store=vector_store,
        keyword_retriever=keyword_retriever
    )

   
    # 7. Generate candidates
   

    all_candidates = {}

    for q in questions:

        question_id = q["id"]
        question = q["question"]

        # Skip questions where the answer is
        # expected to be outside our documents
        if q.get("expected_source") is None:
            print(f"[SKIP] {question_id}")
            continue

        print(f"[INFO] Processing {question_id}...")

        results = retriever.retrieve(
            query=question,
            top_k=TOP_K,
            retrieval_k=RETRIEVAL_K
        )

        candidates = []

        for rank, result in enumerate(results, start=1):

            metadata = result["metadata"]

            candidate = {
                "rank": rank,
                "chunk_id": metadata.get("chunk_id"),
                "rrf_score": result.get("rrf_score"),
                "semantic_rank": result.get("semantic_rank"),
                "semantic_distance": result.get("semantic_distance"),
                "keyword_rank": result.get("keyword_rank"),
                "keyword_score": result.get("keyword_score"),
                "source": metadata.get("source"),
                "page": metadata.get("page"),
                "text": metadata.get("text", "")
            }

            candidates.append(candidate)

        # Store candidates for this question
        all_candidates[question_id] = {
            "question": question,
            "expected_source": q.get("expected_source"),
            "candidates": candidates
        }


    # 8. Save candidates to JSON
    

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            all_candidates,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 60)
    print("[SUCCESS] Candidate generation completed.")
    print(f"[SUCCESS] Results saved to: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()