class HybridRetriever:
    def __init__(self,vector_store,keyword_retriever):
        self.vector_store=vector_store
        self.keyword_retriever=keyword_retriever

    def reciprocal_rank_fusion(self,semantic_results,keyword_results,k=60):
        """
    Merge semantic and keyword retrieval results
    using Reciprocal Rank Fusion (RRF).

    RRF formula:

        RRF Score = 1 / (k + rank)

    If the same chunk appears in both retrievers,
    both scores are added together.
        """

    # Stores the final RRF score for each chunk.
        scores = {}

    # Stores ONE combined result for each chunk.
        documents = {}

    # 1. PROCESS SEMANTIC / FAISS RESULTS

        for rank, result in enumerate(semantic_results, start=1):

        # Get the unique ID of the chunk.
            chunk_id = result["metadata"]["chunk_id"]

        # Calculate this chunk's semantic RRF contribution.
            rrf_score = 1 / (k + rank)

        # Add the score to the chunk.
        # If the chunk already exists, add to its existing score.
            scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score

        # If this chunk doesn't exist yet,
        # create a copy of the semantic result.
            if chunk_id not in documents:
             documents[chunk_id] = {**result}

        # Store the semantic ranking.
            documents[chunk_id]["semantic_rank"] = rank

        # Store semantic distance if it exists.
            documents[chunk_id]["semantic_distance"] = result.get("distance")

    # 2. PROCESS KEYWORD / BM25 RESULTS
   
        for rank, result in enumerate(keyword_results, start=1):

        # Get the unique ID of the chunk.
            chunk_id = result["metadata"]["chunk_id"]

        # Calculate this chunk's keyword RRF contribution.
            rrf_score = 1 / (k + rank)

        # Add the keyword contribution to the existing score.
            scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score

        # If this chunk wasn't found by semantic search,
        # create a new result for it.
            if chunk_id not in documents:
                documents[chunk_id] = {**result}

        # Store the keyword ranking.
            documents[chunk_id]["keyword_rank"] = rank

        # Store the BM25 score.
            documents[chunk_id]["keyword_score"] = result.get("score")

    # 3. SORT RESULTS BY FINAL RRF SCORE
    

        ranked_documents = sorted(
            documents.values(),

        # Get the chunk_id and use its RRF score for sorting.
        key=lambda result: scores[result["metadata"]["chunk_id"]],reverse=True)# Highest RRF score first

    # 4. ADD FINAL RRF SCORE TO EACH RESULT
  

        for result in ranked_documents:

        # Get the chunk ID.
            chunk_id = result["metadata"]["chunk_id"]

        # Store the final combined RRF score.
            result["rrf_score"] = scores[chunk_id]

        return ranked_documents


    def retrieve(self,query: str,top_k: int = 5,retrieval_k: int = 5):

        print(f"\n[INFO] Hybrid search: {query}")

        #semantic_retrieval
        semantic_results = self.vector_store.query(
            query_text=query,
            top_k=retrieval_k
        )

        # Keyword retrieval
        keyword_results = self.keyword_retriever.retrieve(
            query=query,
            top_k=retrieval_k
        )
        print("\nSEMANTIC RESULT:")
        print(semantic_results[0])

        print("\nKEYWORD RESULT:")
        print(keyword_results[0])

        # 3. RRF
        hybrid_results = self.reciprocal_rank_fusion(
            semantic_results,
            keyword_results
        )

        return hybrid_results[:top_k]


    def build_context(self,results):
        """
         Build a single context string from the retrieved chunks.

        The top-k chunks returned by hybrid retrieval are combined
        into one piece of text that will later be given to the LLM.

        """
        print(f"\n[DEBUG] build_context received {len(results)} results")
        context_parts=[]
        for i,result in  enumerate(results,start=1):
            text=result['metadata']['text']
            chunk_id=result['metadata'].get('chunk_id')
            source=result['metadata'].get('source')
            page=result['metadata'].get('page')

            # separated section for this chunk

            chunk_context = f"""
--- Retrieved Chunk {i} ---
Chunk ID: {chunk_id}
Source: {source}
Page: {page}

{text}
"""
            context_parts.append(chunk_context)
        context="\n".join(context_parts)
        print(f"[DEBUG] Context contains {len(context_parts)} chunks")

        return context




if __name__ == "__main__":

    from src.rag1.data_ingestion import load_all_documents
    from src.rag1.embeddings import Embedding_Pipeline
    from src.rag1.vectorstore import FaissVectorStore
    from src.rag1.keyword_retriever import KeywordRetriever

    # Load documents

    documents = load_all_documents("data/pdf")

    # Create chunks
   
    embedding_pipeline = Embedding_Pipeline()

    chunks = embedding_pipeline.chunk_documents(
        documents
    )

    print(f"[INFO] Total chunks: {len(chunks)}")
  
    # Create BM25
    
    keyword_retriever = KeywordRetriever(
        chunks
    )
    # Load FAISS
  

    vector_store = FaissVectorStore()

    vector_store.load()

  
    # Create Hybrid Retriever
    

    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        keyword_retriever=keyword_retriever
    )

    # Search
  

    results = hybrid_retriever.retrieve(
        query="What is  Dataset Augmentation",
        top_k=5
    )

    #the context that will eventually be sent to the LLM
    print(f"[INFO] Number of retrieved results: {len(results)}")
    context = hybrid_retriever.build_context(results)

    print("\n")
    print("=" * 60)
    print("CONTEXT FOR LLM")
    print("=" * 60)

    print(context)

   
    # Display results
  

    print("\n")
    print("=" * 60)
    print("HYBRID SEARCH RESULTS")
    print("=" * 60)

    for i, result in enumerate(
    results,
    start=1
):

        print(f"\nResult {i}")

    # Final score after combining FAISS and BM25 rankings.
        print(
        f"RRF Score: "
        f"{result['rrf_score']:.6f}"
    )

    # Unique identifier of the chunk.
        print(
        f"Chunk ID: "
        f"{result['metadata']['chunk_id']}"
    )

    # Rank assigned by semantic/FAISS retrieval.
        print(
        f"Semantic Rank: "
        f"{result.get('semantic_rank')}"
    )

    # FAISS distance/similarity value.
        print(
        f"Semantic Distance: "
        f"{result.get('semantic_distance')}"
    )

    # Rank assigned by BM25 retrieval.
        print(
        f"Keyword Rank: "
        f"{result.get('keyword_rank')}"
    )

    # Original BM25 score.
        print(
        f"Keyword Score: "
        f"{result.get('keyword_score')}"
    )

        print(
        f"Source: "
        f"{result['metadata'].get('source')}"
    )

        print(
        f"Page: "
        f"{result['metadata'].get('page')}"
    )

        print(
        f"Text: "
        f"{result['metadata']['text'][:500]}"
    )