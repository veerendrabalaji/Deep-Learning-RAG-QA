import os

from dotenv import load_dotenv

# Load environment variables before importing RAG components
load_dotenv()

import streamlit as st

from src.rag1.Rag_pipeline import RAGPipeline
from src.rag1.hybrid_retriever import HybridRetriever
from src.rag1.prompt import PromptBuilder
from src.rag1.reranker import Reranker
from src.rag1.vectorstore import FaissVectorStore
from src.rag1.keyword_retriever import KeywordRetriever




st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide"
)


print("LANGSMITH_TRACING =", os.getenv("LANGSMITH_TRACING"))
print("LANGSMITH_PROJECT =", os.getenv("LANGSMITH_PROJECT"))
print(
    "LANGSMITH_API_KEY loaded =",
    bool(os.getenv("LANGSMITH_API_KEY"))
)



st.title("📚 RAG Document Q&A")

st.write(
    "Ask questions about the information contained "
    "in your documents."
)




with st.sidebar:

    st.header("⚙️ RAG System")

    st.markdown("""
    **Retrieval**
    
    • FAISS Semantic Search  
    • BM25 Keyword Search  
    • Reciprocal Rank Fusion (RRF)

    **Reranking**

    • CrossEncoder

    **Generation**

    • Groq LLM

    **Observability**

    • LangSmith
    """)

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()



if "messages" not in st.session_state:
    st.session_state.messages = []



@st.cache_resource
def initialize_rag():


    from src.rag1.data_ingestion import load_all_documents

    documents = load_all_documents("data/pdf")


    

    from src.rag1.embeddings import Embedding_Pipeline

    embedding_pipeline = Embedding_Pipeline()

    chunks = embedding_pipeline.chunk_documents(documents)


    

    vector_store = FaissVectorStore()

    vector_store.load()



    keyword_retriever = KeywordRetriever(chunks)


    
    # 5. Create hybrid retriever


    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        keyword_retriever=keyword_retriever
    )


    reranker = Reranker()


    prompt_builder = PromptBuilder()




    rag = RAGPipeline(
        retriever=hybrid_retriever,
        prompt_builder=prompt_builder,
        reranker=reranker
    )

    return rag



# DISPLAY PREVIOUS CHAT MESSAGES


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Display sources for assistant messages
        if message["role"] == "assistant":

            sources = message.get("sources", [])

            if sources:

                with st.expander("📚 Sources"):

                    for source in sources:

                        source_name = source["source"]
                        page = source["page"]

                        st.write(
                            f"📄 **{source_name}** — Page {page}"
                        )

                        # Optional retrieval details
                        if "reranker_score" in source:

                            st.caption(
                                f"Reranker score: "
                                f"{source['reranker_score']:.4f}"
                            )



# CHAT INPUT


query = st.chat_input(
    "Ask a question about your documents..."
)



# PROCESS USER QUESTION


if query:

    
    # Display user question
    

    with st.chat_message("user"):

        st.markdown(query)


    # Save user question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )


    # Initialize RAG
   

    rag = initialize_rag()


    

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:

                answer, results = rag.run(
                    query=query,
                    top_k=5
                )

            except Exception as e:

                st.error(
                    "Something went wrong while processing "
                    "your question."
                )

                st.exception(e)

                answer = None
                results = []


      
        # Display answer
     

        if answer:

            st.markdown(answer)


            # Prepare source information
           
            sources = []

            for result in results:

                metadata = result.get(
                    "metadata",
                    {}
                )

                source = metadata.get(
                    "source",
                    "Unknown source"
                )

                page = metadata.get(
                    "page",
                    "Unknown page"
                )

                source_name = os.path.basename(source)

                reranker_score = result.get(
                    "reranker_score"
                )

                sources.append(
                    {
                        "source": source_name,
                        "page": page,
                        "reranker_score": reranker_score
                    }
                )


           
            # Display sources

            if sources:

                with st.expander("📚 Sources"):

                    for source in sources:

                        st.write(
                            f"📄 **{source['source']}** "
                            f"— Page {source['page']}"
                        )

                        if source["reranker_score"] is not None:

                            st.caption(
                                f"Reranker score: "
                                f"{source['reranker_score']:.4f}"
                            )

# ------------------------------------------------
            # Save assistant response
            

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                }
            )