# 🏭 Ops Copilot — Hybrid RAG for Manufacturing Operations

> Ask factory floor questions in plain English. Get grounded answers from real Fanuc robot SOPs and manufacturing procedures.

Ops Copilot is a **Retrieval-Augmented Generation (RAG) pipeline** that ingests real manufacturing documentation — including 500+ pages of Fanuc robot maintenance manuals — and lets factory operators query it conversationally. Answers are grounded in source documents with full citations.

## What Makes This Not Tutorial-Level

| Feature | Typical RAG Tutorial | Ops Copilot |
|---|---|---|
| **Retrieval** | Dense vector search only | **Hybrid: ChromaDB (semantic) + BM25 (keyword)** |
| **Ranking** | Cosine similarity | **Reciprocal Rank Fusion** — merges dense + sparse results |
| **Chunking** | Fixed `chunk_size=500` | Sentence-boundary chunking with 1-sentence overlap |
| **Data** | Wikipedia paragraphs | **Real Fanuc robot manuals** (500+ pages) + 5 manufacturing SOPs |
| **Prompt** | Generic Q&A | Domain-tuned for manufacturing terminology interpretation |
| **Citations** | None | **Source document + relevance score** per answer |
| **Memory** | Single-turn | **Conversation memory** — retains last 3 turns for follow-up questions |
| **API** | Script | **FastAPI + Pydantic v2** with request/response validation |
| **UI** | None | **Streamlit chat interface** with citation panels |

## Architecture

```
User Question
  → Streamlit Chat UI (conversation memory, source citations)
    → FastAPI /ask endpoint (Pydantic request validation)
      → Hybrid Retrieval:
          ├── ChromaDB (HNSW cosine) → Top 10 semantic matches
          └── BM25Okapi (keyword)    → Top 10 keyword matches
      → Reciprocal Rank Fusion (k=60) → Merged ranking
      → Top 3 chunks selected with source metadata
    → OpenAI GPT-4o-mini (domain-tuned prompt, temperature=0.2)
  → Answer + Source Citations + Relevance Scores
```

## Tech Stack

| Layer | Technology |
|---|---|
| **API** | FastAPI, Pydantic v2, Uvicorn |
| **Vector DB** | ChromaDB (persistent, HNSW cosine similarity) |
| **Keyword Search** | BM25Okapi (rank_bm25) |
| **Fusion** | Reciprocal Rank Fusion (k=60) |
| **LLM** | OpenAI GPT-4o-mini |
| **PDF Parsing** | PyMuPDF (fitz) |
| **UI** | Streamlit |

## Project Structure

```
ops-copilot/
├── src/
│   ├── main.py              # FastAPI app — /health, /ask endpoints
│   ├── app.py               # Streamlit chat UI with citations
│   ├── schemas.py           # Pydantic models (QueryRequest, QueryResponse, SourceCitation)
│   └── rag/
│       ├── retriever.py     # VectorStore: ChromaDB + BM25 + RRF hybrid search
│       └── generator.py     # OpenAI generation with conversation memory
├── data/
│   ├── *.pdf                # Fanuc robot manuals (500+ pages)
│   └── sop_*.txt            # 5 manufacturing SOPs (spindle, coolant, e-stop, FAI, PM)
├── requirements.txt
├── .env.example
└── README.md
```

## Quick Start

```bash
# 1. Clone and enter
git clone https://github.com/siddgawad/ops-copilot.git
cd ops-copilot

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your OpenAI API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Start the FastAPI backend
uvicorn src.main:app --reload

# 6. In a second terminal, start the Streamlit UI
streamlit run src/app.py
```

The API docs will be at `http://127.0.0.1:8000/docs`
The chat UI will be at `http://localhost:8501`

## Sample Questions

- *"What vibration level requires immediate spindle shutdown?"*
- *"What is the maximum torque for spindle bolts?"*
- *"How do I perform a First Article Inspection?"*
- *"What are the E-stop recovery steps for the Fanuc LR Mate?"*
- *"What coolant concentration should I maintain?"*
- *"What is the motion range for axis J1?"*

## How Hybrid Search Works

Most RAG tutorials use **only** semantic (vector) search. This misses exact keyword matches — dangerous in manufacturing where part numbers, torque specs, and model numbers matter.

Ops Copilot uses both:

1. **ChromaDB** (dense vectors, HNSW, cosine similarity) — finds semantically similar chunks
2. **BM25** (term frequency, keyword matching) — finds exact terminology matches
3. **Reciprocal Rank Fusion** (k=60) — merges both ranked lists into a single fused ranking

This means a query like *"Fanuc LR Mate 200iD axis J1 range"* will match even if the semantic embedding doesn't capture the exact model number, because BM25 catches it via keyword overlap.

## Data

| Document | Pages | Type |
|---|---|---|
| Fanuc Robot LR Mate 200iD Operators Manual | 200+ | PDF |
| Fanuc 7066350 Manual | 300+ | PDF |
| SOP 001 — Spindle Vibration Analysis | 3 KB | TXT |
| SOP 002 — Coolant System Maintenance | 3 KB | TXT |
| SOP 003 — Emergency Stop Recovery | 4 KB | TXT |
| SOP 004 — First Article Inspection | 4 KB | TXT |
| SOP 005 — Preventive Maintenance Schedule | 5 KB | TXT |

## Future Roadmap

- [ ] Cross-encoder reranker (top 10 → top 3 with precision improvement)
- [ ] Input guardrails (prompt injection detection)
- [ ] Output guardrails (grounding score — is the answer supported by retrieved chunks?)
- [ ] RAGAS evaluation framework (faithfulness, answer relevancy, context precision)
- [ ] Docker Compose for one-command deployment
- [ ] SSE streaming for real-time token output

## License

MIT
