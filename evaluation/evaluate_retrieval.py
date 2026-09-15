import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

from langsmith import Client, evaluate

from src.rag1.data_ingestion import load_all_documents
from src.rag1.embeddings import Embedding_Pipeline
from src.rag1.vectorstore import FaissVectorStore
from src.rag1.keyword_retriever import KeywordRetriever
from src.rag1.hybrid_retriever import HybridRetriever
from src.rag1.reranker import Reranker


DATASET_NAME = "RAG Retrieval Evaluation"

TOP_K = 10
RETRIEVAL_K = 20


# Build your existing retriever


def build_retriever():

    print("[INFO] Loading documents...")

    documents = load_all_documents("data/pdf")

    embedding_pipeline = Embedding_Pipeline()

    chunks = embedding_pipeline.chunk_documents(documents)

    print(f"[INFO] Total chunks: {len(chunks)}")

    keyword_retriever = KeywordRetriever(chunks)

    vector_store = FaissVectorStore()

    vector_store.load()

    print("[INFO] FAISS index loaded.")

    retriever = HybridRetriever(
        vector_store=vector_store,
        keyword_retriever=keyword_retriever
    )

    return retriever


# ---------------------------------------------------------
# Target function
# LangSmith calls this once for every question
# ---------------------------------------------------------

retriever = build_retriever()
reranker = Reranker()

def target(inputs: dict):
    question = inputs["question"]

    hybrid_results = retriever.retrieve(
        query=question,
        top_k=RETRIEVAL_K,
        retrieval_k=RETRIEVAL_K
    )

    # Reranker disabled for baseline comparison
    reranked_results = reranker.rerank(query=question, results=hybrid_results, top_k=TOP_K)
    reranked_results = hybrid_results[:TOP_K]

    retrieved_chunks = []
    for rank, result in enumerate(reranked_results, start=1):
        metadata = result["metadata"]
        retrieved_chunks.append({
            "rank": rank,
            "chunk_id": metadata.get("chunk_id"),
            "source": metadata.get("source"),
            "page": metadata.get("page"),
            "text": metadata.get("text", ""),
            "rrf_score": result.get("rrf_score"),
        })

    return {"question": question, "retrieved_chunks": retrieved_chunks}

# ---------------------------------------------------------
# Simple source-level evaluator
# ---------------------------------------------------------

from pathlib import Path

def source_hit(outputs: dict, reference_outputs: dict):
    expected_source = Path(
        str(reference_outputs["expected_source"])
    ).name

    retrieved_sources = [
        Path(str(chunk.get("source", ""))).name
        for chunk in outputs.get("retrieved_chunks", [])
    ]

    return {
        "key": "expected_source_found",
        "score": int(expected_source in retrieved_sources)
    }

def hit_at_k(outputs: dict, reference_outputs: dict, k: int):
    expected_source = Path(
        str(reference_outputs["expected_source"])
    ).name

    retrieved_sources = [
        Path(str(chunk.get("source", ""))).name
        for chunk in outputs.get("retrieved_chunks", [])[:k]
    ]

    return {
        "key": f"hit_at_{k}",
        "score": int(expected_source in retrieved_sources)
    }

def hit_at_1(outputs, reference_outputs):
    return hit_at_k(outputs, reference_outputs, 1)


def hit_at_3(outputs, reference_outputs):
    return hit_at_k(outputs, reference_outputs, 3)


def hit_at_5(outputs, reference_outputs):
    return hit_at_k(outputs, reference_outputs, 5)


def hit_at_10(outputs, reference_outputs):
    return hit_at_k(outputs, reference_outputs, 10)

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():

    client = Client()

    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[
            source_hit,
            hit_at_1,
            hit_at_3,
            hit_at_5,
            hit_at_10,
        ],
        experiment_prefix="hybrid-rerank-v1",
        metadata={
            "retriever": "FAISS + BM25 + RRF + CrossEncoder Rerank",
            "top_k": TOP_K,
            "retrieval_k": RETRIEVAL_K
        },
        max_concurrency=1
    )

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 60)

    # Convert LangSmith experiment results to DataFrame
    df = results.to_pandas()

    print("\nDataFrame columns:")
    print(df.columns.tolist())

    print("\nNumber of evaluated questions:", len(df))

    # Calculate aggregate scores
    print("\n" + "=" * 60)
    print("AGGREGATE RETRIEVAL METRICS")
    print("=" * 60)

    for metric in [
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "hit_at_10"
    ]:

        matching_columns = [
            column for column in df.columns
            if metric in column.lower()
        ]

        if matching_columns:

            column = matching_columns[0]

            score = df[column].mean()

            print(
                f"{metric.upper():10} : "
                f"{score:.3f} "
                f"({score * 100:.1f}%)"
            )

    print("=" * 60)


if __name__ == "__main__":
    main()