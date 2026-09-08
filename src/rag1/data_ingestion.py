from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from pathlib import Path
from typing import List

def load_all_documents(data_dir:str)->List[Document]:
    """
    Load all supported files from the data directory and convert to LangChain document structure.
    Supported: PDF
    """
    data_path=Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    documents:List[Document]=[]

    pdf_files=list(data_path.glob('**/*.pdf'))
    print(f"[DEBUG] Found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}")
    for pdf_file in pdf_files:
           print(f"[DEBUG] Loading PDF: {pdf_file}")
           try:
            loader=PyMuPDFLoader(str(pdf_file))
            loaded=loader.load()
            documents.extend(loaded)
            print(f"[DEBUG] Loaded {len(loaded)} PDF docs from {pdf_file}")
           except Exception as e:
               print(f"[ERROR] Failed to load PDF {pdf_file}: {e}")
           
    return documents

## example usage
if __name__ == "__main__":
    docs = load_all_documents("data/pdf")
    print(f"Loaded {len(docs)} documents.")
    print("Example document:", docs[0] if docs else None)