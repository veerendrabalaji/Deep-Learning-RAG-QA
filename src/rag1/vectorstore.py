import os
import faiss
import numpy as np
import pickle

from typing import List, Any
from langchain_core.documents import Document
from src.rag1.embeddings import Embedding_Pipeline


class FaissVectorStore:

    def __init__(
        self,persist_dir: str = "faiss_store",embedding_model: str = "all-MiniLM-L6-v2",chunk_size: int = 1000,chunk_overlap: int = 200):

        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index = None
        self.metadata = []

        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embedding_pipeline = Embedding_Pipeline(
            model_name=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def build_from_documents(self, documents: List[Document]):

        print(
            f"[INFO] Building vector store from "
            f"{len(documents)} raw documents..."
        )

        chunks = self.embedding_pipeline.chunk_documents(documents)

        embeddings = self.embedding_pipeline.embed_chunks(chunks)

        metadatas = [
            {
                **chunk.metadata,
                "text": chunk.page_content,
                "chunk_id": i
            }
            for i, chunk in enumerate(chunks)
        ]

        self.add_embeddings(embeddings.astype("float32"),metadatas)

        self.save()

        print(
            f"[INFO] Vector store built and saved "
            f"to {self.persist_dir}"
        )

    def add_embeddings(self,embeddings: np.ndarray,metadatas: List[Any] = None):

        dim = embeddings.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)

        if metadatas:
            self.metadata.extend(metadatas)

        print(
            f"[INFO] Added {embeddings.shape[0]} "
            f"vectors to FAISS index."
        )

    def save(self):
        faiss_path = os.path.join(self.persist_dir,"faiss.index")

        meta_path = os.path.join(self.persist_dir,"metadata.pkl")

        faiss.write_index(self.index,faiss_path)

        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(
            f"[INFO] Saved FAISS index and metadata "
            f"to {self.persist_dir}"
        )

    def load(self):
            
            faiss_path = os.path.join(self.persist_dir,"faiss.index")

            meta_path = os.path.join(self.persist_dir,"metadata.pkl")

            self.index = faiss.read_index(faiss_path)

            with open(meta_path, "rb") as f:
                 self.metadata = pickle.load(f)

            print(
            f"[INFO] Loaded FAISS index and metadata "
            f"from {self.persist_dir}"
        )


    def query(self,query_text: str,top_k: int = 5):

        print(
            f"[INFO] Querying vector store for: "
            f"'{query_text}'"
        )

        query_emb = self.embedding_pipeline.model.encode([query_text]).astype("float32")

        return self.search(query_emb,top_k=top_k)

    def search(self,query_embedding: np.ndarray,top_k: int = 3):
    
            if self.index is None:
                raise ValueError(
                    "FAISS index has not been built or loaded."
                )
    
            D, I = self.index.search(query_embedding,top_k)
    
            results = []
    
            for idx, dist in zip(I[0], D[0]):
                if idx < 0:
                    continue
    
                meta = (
                    self.metadata[idx]
                    if idx < len(self.metadata)
                    else None
                )
    
                results.append(
                    {
                        "index": int(idx),
                        "distance": float(dist),
                        "metadata": meta
                    }
                )
    
            return results


if __name__ == "__main__":
    from src.rag1.data_ingestion import load_all_documents
    docs = load_all_documents("data/pdf")
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.load()
    print(store.query("What is Bilogical Vision?", top_k=3))