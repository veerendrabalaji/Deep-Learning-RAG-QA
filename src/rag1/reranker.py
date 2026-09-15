from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L6-v2"
    ):
        print(f"[INFO] Loading reranker: {model_name}")

        self.model = CrossEncoder(model_name)

        print("[INFO] Reranker loaded successfully.")

    def rerank(self, query: str, results: list, top_k: int = 5):

        if not results:
            return []

        # Create (query, document) pairs
        pairs = [
            (
                query,
                result["metadata"]["text"]
            )
            for result in results
        ]

        # CrossEncoder scores every query-document pair
        scores = self.model.predict(pairs)

        # Attach reranker score
        for result, score in zip(results, scores):
            result["reranker_score"] = float(score)

        # Sort by reranker score
        reranked_results = sorted(
            results,
            key=lambda x: x["reranker_score"],
            reverse=True
        )

        return reranked_results[:top_k]