from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

class KeywordRetriever:
    def __init__(self,chunks:list[Document]):
        self.chunks=chunks
        tokenized_document=[chunk.page_content.lower().split() for chunk in chunks ] ##create chunks
        self.bm25=BM25Okapi(tokenized_document) #conatins rhe chunked documents
        print(
            f"[INFO] BM25 index created for "
            f"{len(chunks)} chunks."
        )

    def retrieve(self,query:str,top_k=5):
        query_tokens=query.lower().split()
        scores=self.bm25.get_scores(query_tokens) #returns one score for each document/chunk in the indexed corpus.
        ranked_indices = scores.argsort()[::-1][:top_k] #indices of the top k highest-scoring chunk
        results = []

        for idx in ranked_indices:

            results.append({
                "index": int(idx),
                "score": float(scores[idx]),
                "metadata": {
                    **self.chunks[idx].metadata,
                    "text": self.chunks[idx].page_content
                }
            })
        return results


if __name__ == "__main__":

    from src.rag1.data_ingestion import load_all_documents
    from src.rag1.embeddings import Embedding_Pipeline

    # 1. Load PDFs
    documents = load_all_documents("data/pdf")

    # 2. Create the same chunks
    embedding_pipeline = Embedding_Pipeline()

    chunks = embedding_pipeline.chunk_documents(documents)

    print(f"[INFO] Total chunks: {len(chunks)}")
    print(chunks[0].metadata)

    # 3. Create BM25 using those chunks
    retriever = KeywordRetriever(chunks)

    # 4. Search
    results = retriever.retrieve(
        "What is Biological Vision?",
        top_k=5
    )

    # 5. Display results
    print("\n" + "=" * 60)
    print("BM25 RESULTS")
    print("=" * 60)

    for i, result in enumerate(results, start=1):

        print(f"\nResult {i}")
        print(f"Score: {result['score']:.4f}")
        print(f"Source: {result['metadata'].get('source')}")
        print(f"Page: {result['metadata'].get('page')}")
        print(f"Text: {result['metadata']['text'][:500]}")