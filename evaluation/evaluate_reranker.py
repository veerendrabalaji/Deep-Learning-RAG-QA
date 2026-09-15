from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client, evaluate

from src.rag1.data_ingestion import load_all_documents
from src.rag1.embeddings import Embedding_Pipeline
from src.rag1.vectorstore import FaissVectorStore
from src.rag1.keyword_retriever import KeywordRetriever
from src.rag1.hybrid_retriever import HybridRetriever
from src.rag1.reranker import Reranker


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

DATASET_NAME = "RAG Retrieval Evaluation"

# Number of candidates retrieved by the hybrid retriever
RETRIEVAL_K = 20

# Number of chunks returned after reranking
RERANK_TOP_K = 5


# --------------------------------------------------
# Build hybrid retriever
# --------------------------------------------------

def build_retriever():

    print("\n[INFO] Loading documents...")

    documents = load_all_documents("data/pdf")

    print(f"[INFO] Documents loaded: {len(documents)}")

    # Create the same chunks used during indexing
    embedding_pipeline = Embedding_Pipeline()

    chunks = embedding_pipeline.chunk_documents(documents)

    print(f"[INFO] Chunks created: {len(chunks)}")

    # BM25
    keyword_retriever = KeywordRetriever(chunks)

    # FAISS
    vector_store = FaissVectorStore()
    vector_store.load()

    # Hybrid retriever
    retriever = HybridRetriever(
        vector_store=vector_store,
        keyword_retriever=keyword_retriever
    )

    return retriever


# --------------------------------------------------
# Build components
# --------------------------------------------------

retriever = build_retriever()

print("\n[INFO] Loading reranker...")

reranker = Reranker()

print("[INFO] Reranker ready.")


# --------------------------------------------------
# LangSmith target function
# --------------------------------------------------

def target(inputs: dict):

    question = inputs["question"]

    print("\n" + "=" * 60)
    print(f"[QUESTION] {question}")
    print("=" * 60)

    # ----------------------------------------------
    # Stage 1: Hybrid retrieval
    # ----------------------------------------------

    hybrid_results = retriever.retrieve(
        query=question,
        top_k=RETRIEVAL_K,
        retrieval_k=RETRIEVAL_K
    )

    print(
        f"[INFO] Hybrid retrieval returned "
        f"{len(hybrid_results)} candidates"
    )

    # ----------------------------------------------
    # Stage 2: CrossEncoder reranking
    # ----------------------------------------------

    reranked_results = reranker.rerank(
        query=question,
        results=hybrid_results,
        top_k=RERANK_TOP_K
    )

    print(
        f"[INFO] Reranker returned "
        f"{len(reranked_results)} results"
    )

    # ----------------------------------------------
    # Prepare output for LangSmith
    # ----------------------------------------------

    retrieved_chunks = []

    for rank, result in enumerate(
        reranked_results,
        start=1
    ):

        metadata = result["metadata"]

        retrieved_chunks.append({

            "rank": rank,

            "chunk_id": metadata.get(
                "chunk_id"
            ),

            "source": metadata.get(
                "source"
            ),

            "page": metadata.get(
                "page"
            ),

            "text": metadata.get(
                "text",
                ""
            ),

            "rrf_score": result.get(
                "rrf_score"
            ),

            "reranker_score": result.get(
                "reranker_score"
            )
        })

    return {
        "question": question,
        "retrieved_chunks": retrieved_chunks
    }


# --------------------------------------------------
# Hit@K evaluator
# --------------------------------------------------

def hit_at_k(
    outputs: dict,
    reference_outputs: dict,
    k: int
):

    expected_source = Path(
        str(
            reference_outputs["expected_source"]
        )
    ).name

    retrieved_sources = [

        Path(
            str(
                chunk.get(
                    "source",
                    ""
                )
            )
        ).name

        for chunk
        in outputs.get(
            "retrieved_chunks",
            []
        )[:k]
    ]

    return {
        "key": f"hit_at_{k}",
        "score": int(
            expected_source
            in retrieved_sources
        )
    }


# --------------------------------------------------
# Individual evaluators
# --------------------------------------------------

def hit_at_1(outputs, reference_outputs):

    return hit_at_k(
        outputs,
        reference_outputs,
        1
    )


def hit_at_3(outputs, reference_outputs):

    return hit_at_k(
        outputs,
        reference_outputs,
        3
    )


def hit_at_5(outputs, reference_outputs):

    return hit_at_k(
        outputs,
        reference_outputs,
        5
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    client = Client()

    print("\n")
    print("=" * 60)
    print("HYBRID + RERANKER EVALUATION")
    print("=" * 60)

    results = evaluate(

        target,

        data=DATASET_NAME,

        evaluators=[
            hit_at_1,
            hit_at_3,
            hit_at_5,
        ],

        experiment_prefix="hybrid-reranker",

        metadata={

            "retriever":
                "FAISS + BM25 + RRF",

            "reranker":
                "cross-encoder/ms-marco-MiniLM-L6-v2",

            "retrieval_k":
                RETRIEVAL_K,

            "rerank_top_k":
                RERANK_TOP_K
        },

        max_concurrency=1
    )

    # --------------------------------------------------
    # Aggregate results
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("RERANKER EVALUATION RESULTS")
    print("=" * 60)

    df = results.to_pandas()

    print(
        f"\nNumber of evaluated questions: "
        f"{len(df)}"
    )

    print("\nAggregate metrics:")

    for metric in [
        "hit_at_1",
        "hit_at_3",
        "hit_at_5"
    ]:

        matching_columns = [

            column

            for column
            in df.columns

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