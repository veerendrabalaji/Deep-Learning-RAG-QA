# 📚 RAG Document Q&A

A document-based **Retrieval-Augmented Generation (RAG)** system that answers questions from PDF documents using **hybrid retrieval, Reciprocal Rank Fusion (RRF), CrossEncoder reranking, and a Groq LLM**.

The application provides answers along with the **source document and page number** used for retrieval.

---

## 🚀 Features

* 📄 PDF document ingestion
* ✂️ Recursive text chunking
* 🧠 Semantic search using FAISS
* 🔎 Keyword search using BM25
* 🔀 Hybrid retrieval using Reciprocal Rank Fusion (RRF)
* 🎯 CrossEncoder-based reranking
* 🤖 Answer generation using Groq LLM
* 📚 Source document and page citations
* 📊 Retrieval and reranking evaluation
* 🔍 LangSmith tracing and observability
* 🖥️ Interactive Streamlit interface

---

## 🏗️ Architecture

```text
                    ┌─────────────────┐
                    │   PDF Documents │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Document Loader │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Text Chunking   │
                    └────────┬────────┘
                             │
                  ┌──────────┴──────────┐
                  ▼                     ▼
          ┌───────────────┐     ┌───────────────┐
          │ FAISS Search  │     │  BM25 Search  │
          │   Semantic    │     │    Keyword    │
          └───────┬───────┘     └───────┬───────┘
                  │                     │
                  └──────────┬──────────┘
                             ▼
                    ┌─────────────────┐
                    │      RRF        │
                    │ Hybrid Fusion   │
                    └────────┬────────┘
                             │
                       Top 20 candidates
                             │
                             ▼
                    ┌─────────────────┐
                    │   CrossEncoder  │
                    │    Reranker     │
                    └────────┬────────┘
                             │
                         Top 5 chunks
                             │
                             ▼
                    ┌─────────────────┐
                    │ Prompt Builder  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    Groq LLM     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Answer + Sources│
                    └─────────────────┘
```

---

## 🧠 Retrieval Strategy

The system uses two complementary retrieval methods.

### 1. Semantic Search — FAISS

The document chunks are converted into embeddings using:

```text
all-MiniLM-L6-v2
```

FAISS is then used to retrieve chunks that are semantically similar to the user's question.

This helps when the query and document use different wording but have similar meaning.

---

### 2. Keyword Search — BM25

BM25 is used for keyword-based retrieval.

It is useful when the query contains important terms, technical terminology, names, or phrases that should match the document directly.

---

### 3. Reciprocal Rank Fusion

The semantic and keyword results are combined using **Reciprocal Rank Fusion (RRF)**.

Instead of directly comparing FAISS distances with BM25 scores, the system combines the **rank positions** of results.

This produces a single ranked list of candidate chunks.

---

## 🎯 Reranking

After hybrid retrieval, the system retrieves **20 candidate chunks**.

A CrossEncoder then evaluates each:

```text
(query, document chunk)
```

pair and assigns a relevance score.

The candidates are sorted using these scores and the **top 5 chunks** are passed to the generation stage.

This improves the ordering of the most relevant retrieved information before sending it to the LLM.

---

## 🤖 Generation

The final context is constructed from the top 5 reranked chunks.

The prompt instructs the LLM to:

* Use only the retrieved context
* Avoid outside knowledge
* Avoid making up information
* Clearly answer the question
* Indicate when the required information is not available

The LLM used in this project is:

```text
openai/gpt-oss-20b
```

through the Groq API.

---

## 📚 Source Citations

The application displays the source document and page number for retrieved information.

Example:

```text
📄 BCS714A-module-2-textbook.pdf — Page 12
```

This makes it easier to verify where the answer came from.

---

## 📊 Evaluation

The retrieval system was evaluated on **48 questions** with known expected source documents, using LangSmith.

### Hybrid Retrieval

| Metric | Score |
| ------ | ----: |
| Hit@1  | 81.2% |
| Hit@3  | 85.4% |
| Hit@5  | 87.5% |
| Hit@10 | 87.5% |

### Hybrid Retrieval + CrossEncoder Reranking

| Metric | Score |
| ------ | ----: |
| Hit@1  | 85.4% |
| Hit@3  | 87.5% |
| Hit@5  | 87.5% |
| Hit@10 | 87.5% |

### Improvement

```text
Hit@1:  81.2% → 85.4%   (+4.2 points)
Hit@3:  85.4% → 87.5%   (+2.1 points)
Hit@5:  87.5% → 87.5%   (unchanged)
Hit@10: 87.5% → 87.5%   (unchanged)
```

Reranking only reorders candidates already retrieved by the hybrid stage — it cannot recover chunks that hybrid retrieval missed entirely. That's why it improves Hit@1/@3 but has no effect at @5/@10. The 87.5% ceiling across all K values indicates a retrieval-recall gap, not a ranking problem — some questions never retrieve the correct source at any depth.

---

## 🔍 Evaluation Approach

The project includes separate scripts for evaluating retrieval and reranking.

```text
evaluation/
│
├── evaluation_dataset.json
├── evaluate_retrieval.py
├── evaluate_reranker.py
└── calculate_metrics.py
```

The evaluation focuses on whether the expected source document appears within the retrieved results.

The project also contains scripts used during evaluation-data preparation:

```text
src/rag1/
├── create_langsmith_dataset.py
└── generate_relevance_candidates.py
```

---

## 🔬 Observability

The RAG pipeline is traced using **LangSmith**.

Tracing helps inspect:

* User queries
* Retrieval execution
* Reranking
* Prompt construction
* LLM generation
* Overall RAG pipeline execution

This makes it easier to debug and understand the behavior of the application.

---

## 🖥️ Streamlit Application

The project includes a Streamlit interface where users can:

1. Enter a question
2. Retrieve relevant document chunks
3. Generate an answer
4. View the supporting source documents and pages

The sidebar also displays the main components of the RAG system.

---

## 📁 Project Structure

```text
Deep-Learning-RAG-QA/
│
├── app.py
├── README.md
├── Requirements.txt
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
│
├── data/
│   └── pdf/
│       └── *.pdf
│
├── evaluation/
│   ├── calculate_metrics.py
│   ├── evaluate_retrieval.py
│   ├── evaluate_reranker.py
│   └── evaluation_dataset.json
│
└── src/
    ├── __init__.py
    │
    ├── test_semantic.py
    │
    └── rag1/
        ├── __init__.py
        ├── data_ingestion.py
        ├── embeddings.py
        ├── hybrid_retriever.py
        ├── keyword_retriever.py
        ├── llm.py
        ├── prompt.py
        ├── Rag_pipeline.py
        ├── reranker.py
        ├── vectorstore.py
        │
        ├── create_langsmith_dataset.py
        └── generate_relevance_candidates.py
```

---

## 🛠️ Tech Stack

| Component        | Technology                      |
| ---------------- | -------------------------------- |
| Language         | Python                          |
| UI               | Streamlit                       |
| PDF Loading      | PyMuPDF                         |
| Text Splitting   | LangChain Text Splitters        |
| Embeddings       | Sentence Transformers           |
| Vector Search    | FAISS                           |
| Keyword Search   | BM25                            |
| Hybrid Retrieval | Reciprocal Rank Fusion          |
| Reranking        | CrossEncoder                    |
| LLM              | Groq                            |
| Observability    | LangSmith                       |
| Environment      | uv / Python virtual environment |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Deep-Learning-RAG-QA
```

### 2. Create the environment

Using `uv`:

```bash
uv sync
```

Or create a standard Python virtual environment and install the dependencies from:

```text
Requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key

LANGSMITH_TRACING=true
LANGSMITH_PROJECT=Deep-Learning-RAG-QA
LANGSMITH_API_KEY=your_langsmith_api_key
```

**Do not commit `.env` to GitHub.**

---

## 📄 Add Documents

Place your PDF documents inside:

```text
data/pdf/
```

Example:

```text
data/
└── pdf/
    ├── document1.pdf
    ├── document2.pdf
    └── document3.pdf
```

The application automatically loads PDF files from this directory.

> Do not upload copyrighted documents to a public repository unless you have permission to redistribute them.

---

## ▶️ Run the Application

Start Streamlit with:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

---

## 🧪 Run Evaluation

Retrieval evaluation:

```bash
python evaluation/evaluate_retrieval.py
```

Reranker evaluation:

```bash
python evaluation/evaluate_reranker.py
```

Calculate metrics:

```bash
python evaluation/calculate_metrics.py
```

---

## 💡 Why Hybrid Retrieval?

Semantic and keyword retrieval have different strengths.

**Semantic search** is useful for understanding meaning and finding conceptually similar content.

**BM25** is useful for exact or important keyword matches.

Combining both using RRF provides a more robust retrieval strategy than relying on only one retrieval method.

---

## 🎯 Project Objective

The goal of this project is to build a practical and explainable RAG system that can:

* Retrieve relevant information from documents
* Reduce irrelevant context before generation
* Improve document ranking using reranking
* Generate answers grounded in retrieved documents
* Provide source references
* Evaluate retrieval performance quantitatively
* Monitor the pipeline using LangSmith

---

## 👨‍💻 Author

**Veerendra Balaji**

Information Science Engineering
2026 Graduate