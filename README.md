# Deep Learning RAG QA

A Retrieval-Augmented Generation (RAG) question-answering system for querying information from PDF documents using hybrid retrieval.

The system combines **FAISS semantic search** and **BM25 keyword search** using **Reciprocal Rank Fusion (RRF)** to retrieve relevant document chunks before generating an answer with a Groq-hosted LLM.

## Features

* PDF document ingestion
* Recursive text chunking
* Semantic retrieval using FAISS
* Keyword retrieval using BM25
* Hybrid retrieval using Reciprocal Rank Fusion (RRF)
* Context-aware answer generation
* Source and page citations
* Streamlit-based user interface
* Retrieval evaluation using Recall@K and MRR

## Architecture

```text
PDF Documents
      ↓
Document Loading
      ↓
Text Chunking
      ↓
Embedding Generation
      ↓
FAISS Semantic Search
      │
      ├──────────────┐
      ↓              ↓
   BM25 Search    Semantic Search
      │              │
      └───────┬──────┘
              ↓
       Reciprocal Rank
          Fusion
              ↓
       Relevant Context
              ↓
        Prompt Builder
              ↓
          Groq LLM
              ↓
        Final Answer
              ↓
      Source + Page Citation
```

## Tech Stack

* Python
* LangChain
* FAISS
* BM25
* Sentence Transformers
* Groq
* Streamlit
* NumPy
* PyPDF

## Project Structure

```text
Deep-Learning-RAG-QA/
│
├── data/
│   └── pdf/
│       └── .gitkeep
│
├── evaluation/
│   ├── evaluation_dataset.json
│   ├── evaluate_retrieval.py
│   ├── calculate_metrics.py
│   └── retrieval_results.json
│
├── src/
│   └── rag1/
│       ├── data_ingestion.py
│       ├── embeddings.py
│       ├── vectorstore.py
│       ├── keyword_retriever.py
│       ├── hybrid_retriever.py
│       ├── prompt.py
│       ├── llm.py
│       └── Rag_pipeline.py
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Retrieval Approach

The project uses two retrieval methods.

### FAISS

FAISS performs semantic similarity search using vector embeddings. This allows the system to retrieve chunks that are conceptually similar to the user's question.

### BM25

BM25 performs keyword-based retrieval and is useful when the query contains specific technical terms or phrases.

### Hybrid Retrieval

The results from FAISS and BM25 are combined using **Reciprocal Rank Fusion (RRF)**.

This allows the system to benefit from both semantic similarity and exact keyword matching.

## Evaluation

The retrieval system was evaluated using a dataset of 20 questions.

| Retriever | Recall@1  | Recall@3  | Recall@5  | MRR       |
| --------- | --------- | --------- | --------- | --------- |
| FAISS     | 0.900     | 1.000     | 1.000     | 0.950     |
| BM25      | 0.900     | 0.950     | 1.000     | 0.929     |
| Hybrid    | **1.000** | **1.000** | **1.000** | **1.000** |

The hybrid retriever achieved **100% Recall@1** and an **MRR of 1.0** on the evaluation dataset.

## Setup

Clone the repository:

```bash
git clone https://github.com/your-username/Deep-Learning-RAG-QA.git
cd Deep-Learning-RAG-QA
```

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key
```

Do not commit your `.env` file to GitHub.

## Add Documents

Place your PDF documents inside:

```text
data/pdf/
```

The PDFs are not included in this repository.

## Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application allows users to enter questions and receive answers based on the uploaded documents.

## Run Retrieval Evaluation

Run:

```bash
python evaluation/evaluate_retrieval.py
```

Then calculate the metrics:

```bash
python evaluation/calculate_metrics.py
```

## Example

**Question:**

```text
What is dataset augmentation?
```

**Answer:**

```text
Dataset augmentation is a technique that creates additional training examples by applying transformations to existing data.

Source: BCS714A-module-2-textbook.pdf
Page: 12
```

## Future Improvements

* Answer-level evaluation
* Faithfulness and hallucination evaluation
* Reranking retrieved documents
* Support for more document formats
* Improved conversational memory
* Better UI and document management
