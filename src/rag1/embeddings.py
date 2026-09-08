from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from src.rag1.data_ingestion import load_all_documents
import numpy as np

class Embedding_Pipeline:
    def __init__(self,model_name:str="all-MiniLM-L6-v2",chunk_size:int=1000,chunk_overlap:int=200):
        self.model_name=model_name
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        self.model=SentenceTransformer(self.model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")


    def chunk_documents(self,documents:List[Document]):
        splitter=RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks=splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def embed_chunks(self,chunks:List[Document])->np.ndarray:
        texts=[chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings=self.model.encode(texts,show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings

if __name__ == "__main__":
    
    docs = load_all_documents("data/pdf")
    emb_pipe = Embedding_Pipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)