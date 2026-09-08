
# IMPORTS


import streamlit as st

# RAG components
from src.rag1.Rag_pipeline import RAGPipeline
from src.rag1.hybrid_retriever import HybridRetriever
from src.rag1.prompt import PromptBuilder

# Import the components you already use to create
# your vector store and BM25 retriever.
from src.rag1.vectorstore import FaissVectorStore
from src.rag1.keyword_retriever import KeywordRetriever

# STREAMLIT PAGE CONFIGURATION

# This MUST be the first Streamlit command in the file.
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide"
)

# PAGE TITLE

st.title("📚 RAG Document Q&A")

st.write(
    "Ask a question about the information contained "
    "in your documents."
)



# INITIALIZE RAG COMPONENTS
@st.cache_resource
def initialize_rag():

    """
    Create the RAG components once.

    @st.cache_resource prevents Streamlit from rebuilding
    the FAISS index, BM25 index, and other components every
    time the user interacts with the UI.
    """

    # CREATE / LOAD VECTOR STORE

    from src.rag1.data_ingestion import load_all_documents
    documents = load_all_documents("data/pdf")
    from src.rag1.embeddings import Embedding_Pipeline
    embedding_pipeline = Embedding_Pipeline()
    chunks = embedding_pipeline.chunk_documents(documents)

    vector_store = FaissVectorStore()
    vector_store.load()

    # CREATE BM25 RETRIEVER
    keyword_retriever = KeywordRetriever(chunks)

 
    #  HYBRID RETRIEVER
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        keyword_retriever=keyword_retriever
    )


    # PROMPT BUILDER
    prompt_builder = PromptBuilder()

    #  RAG PIPELINE
    rag = RAGPipeline(
    retriever=hybrid_retriever,
    prompt_builder=prompt_builder
)

    return rag

# USER QUERY

query = st.text_input(
    "Ask your question:",
    placeholder="Example: What is dataset augmentation?"
)

# ASK QUESTION BUTTON
if st.button("🔍 Ask Question"):

    # Make sure the user entered a question.
    if not query.strip():
        st.warning("Please enter a question.")

    else:
        st.subheader("Question")
        st.write(query)

        # RUN RAG PIPELINE
        rag = initialize_rag()

        with st.spinner("Searching documents and generating answer..."):
            answer, results = rag.run(
                query=query,
                top_k=5
            )

        
        # DISPLAY ANSWER

        st.subheader("Answer")
        st.write(answer)

        # SOURCE CITATION

        if results:           
            metadata = results[0]["metadata"]
            source = metadata.get("source","Unknown source")
            page = metadata.get("page","Unknown page")

            import os
            source_name = os.path.basename(source)
            st.write(
                f"**Source:** {source_name} | **Page:** {page}"
            )