import json


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_at_k(results, retriever, k):
    hits = 0

    for item in results:
        relevant = set(item["relevant_chunks"])
        retrieved = item[f"{retriever}_chunks"][:k]

        if relevant.intersection(retrieved):
            hits += 1

    return hits / len(results)


def mrr(results, retriever):
    reciprocal_ranks = []

    for item in results:
        relevant = set(item["relevant_chunks"])
        retrieved = item[f"{retriever}_chunks"]

        rank = 0

        for i, chunk_id in enumerate(retrieved, start=1):
            if chunk_id in relevant:
                rank = i
                break

        if rank:
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def evaluate(results):
    retrievers = ["faiss", "bm25", "hybrid"]

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION")
    print("=" * 60)

    print(
        f"\n{'Retriever':<12}"
        f"{'Recall@1':<12}"
        f"{'Recall@3':<12}"
        f"{'Recall@5':<12}"
        f"{'MRR':<12}"
    )

    print("-" * 60)

    for retriever in retrievers:
        r1 = recall_at_k(results, retriever, 1)
        r3 = recall_at_k(results, retriever, 3)
        r5 = recall_at_k(results, retriever, 5)
        mrr_score = mrr(results, retriever)

        print(
            f"{retriever.upper():<12}"
            f"{r1:<12.3f}"
            f"{r3:<12.3f}"
            f"{r5:<12.3f}"
            f"{mrr_score:<12.3f}"
        )


if __name__ == "__main__":
    results = load_results(
        "evaluation/retrieval_results.json"
    )

    evaluate(results)