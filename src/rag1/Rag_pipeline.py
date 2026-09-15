from src.rag1.hybrid_retriever import HybridRetriever
from src.rag1.prompt import PromptBuilder
from src.rag1.reranker import Reranker
from langsmith import traceable
from src.rag1.llm import generate_answer


class RAGPipeline:
    """
    Complete RAG pipeline.

    Retrieval:
        FAISS + BM25 + RRF

    Reranking:
        CrossEncoder

    Generation:
        Groq LLM
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        prompt_builder: PromptBuilder,
        reranker: Reranker
    ):
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.reranker = reranker

    @traceable(name="RAG Pipeline")
    def run(self, query: str, top_k: int = 5):
        """
        Run the complete RAG pipeline.
        """

        # STEP 1: HYBRID RETRIEVAL
        # Retrieve a larger set of candidates first.
        hybrid_results = self.retriever.retrieve(
            query=query,
            top_k=20,
            retrieval_k=20
        )

        if not hybrid_results:
            return "No relevant context found.", []

        # STEP 2: RERANK
        # CrossEncoder reorders the retrieved candidates.
        reranked_results = self.reranker.rerank(
            query=query,
            results=hybrid_results,
            top_k=top_k
        )

        if not reranked_results:
            return "No relevant context found.", []

        # STEP 3: BUILD CONTEXT
        context = self.retriever.build_context(
            reranked_results
        )

        if not context.strip():
            return "No relevant context found.", []

        # STEP 4: BUILD PROMPT
        prompt = self.prompt_builder.build_prompt(
            query=query,
            context=context
        )

        # STEP 5: GENERATE ANSWER
        answer = generate_answer(prompt)

        return answer, reranked_results