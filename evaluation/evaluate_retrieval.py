import json
import sys

from src.rag1.vectorstore import FaissVectorStore
from src.rag1.keyword_retriever import KeywordRetriever
from src.rag1.hybrid_retriever import HybridRetriever
from src.rag1.data_ingestion import load_all_documents
from src.rag1.embeddings import Embedding_Pipeline


def load_evaluation_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_retrievers():
    documents = load_all_documents("data/pdf")

    embedding_pipeline = Embedding_Pipeline()
    chunks = embedding_pipeline.chunk_documents(documents)

    vector_store = FaissVectorStore()
    vector_store.load()

    keyword_retriever = KeywordRetriever(chunks)

    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        keyword_retriever=keyword_retriever
    )

    return vector_store, keyword_retriever, hybrid_retriever


def get_chunk_ids(results):
    return [
        result["metadata"]["chunk_id"]
        for result in results
    ]

def print_results(name, results):
    print(f"\n{name}")

    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]

        print(f"\nRank: {rank}")
        print(f"Chunk: {metadata.get('chunk_id')}")
        print(f"Page: {metadata.get('page')}")
        print(f"Text: {metadata.get('text', '')[:500]}")


def evaluate():
    data = load_evaluation_data(
        "evaluation\evaluation_dataset.json"
    )

    vector_store, keyword_retriever, hybrid_retriever = (
        build_retrievers()
    )

    results = []

    for item in data:
        question = item["question"]

        faiss_results = vector_store.query(
            query_text=question,
            top_k=5
        )

        bm25_results = keyword_retriever.retrieve(
            query=question,
            top_k=5
        )

        hybrid_results = hybrid_retriever.retrieve(
            query=question,
            top_k=5
        )

        results.append({
    "id": item["id"],
    "question": question,
    "expected_answer": item["expected_answer"],
    "expected_source": item["expected_source"],
    "expected_page": item["expected_page"],
    "relevant_chunks": item["relevant_chunks"],
    "faiss_chunks": get_chunk_ids(faiss_results),
    "bm25_chunks": get_chunk_ids(bm25_results),
    "hybrid_chunks": get_chunk_ids(hybrid_results)
})

        print("\n" + "=" * 70)
        print(f"Question {item['id']}: {question}")
        print("-" * 70)
        print(f"Expected page: {item['expected_page']}")
        print_results("FAISS", faiss_results)
        print_results("BM25", bm25_results)
        print_results("HYBRID", hybrid_results)

    with open(
        "evaluation/retrieval_results.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=4)

    print("\nEvaluation results saved to:")
    print("evaluation/retrieval_results.json")



if __name__ == "__main__":
    evaluate()