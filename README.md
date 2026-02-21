# 🤖 Ops Copilot — Production-Grade Agentic RAG for Manufacturing

> An AI operations assistant that answers factory floor questions using Retrieval-Augmented Generation, tool calling, guardrails, and rigorous evaluation. Built to demonstrate $450K-level engineering depth, not tutorial-level API glue.

## Architecture

```
User Question → Input Guardrail (prompt injection detection)
  → Router (RAG question? Tool request? Chitchat?)
    → RAG Path:
        → Hybrid Retrieval (dense embeddings + BM25 keyword search)
        → Reciprocal Rank Fusion (merge dense + sparse results)
        → Cross-Encoder Reranker (top 10 → top 3)
        → LLM Generation (structured JSON output via function calling)
        → Grounding Check (is answer supported by retrieved chunks?)
    → Tool Path:
        → Calculator (torque, unit conversions)
        → Device Status Lookup (mock OPC UA API)
        → Alert History Query (SQL)
    → Chitchat Path:
        → Polite refusal with redirect to ops topics
  → Conversation Memory (last 5 turns)
  → Response + cost metrics + latency → Prometheus → Grafana
```

## What Makes This Elite (Not Tutorial-Level)

| Feature | Tutorial Version | This Version |
|---|---|---|
| Chunking | `chunk_size=500` default | 3 strategies benchmarked (fixed, semantic, parent-child), winner chosen with recall@5 data |
| Embeddings | `text-embedding-ada-002` because default | 3 models benchmarked on recall, latency, and cost |
| Retrieval | Dense search only | Hybrid: dense + BM25 with Reciprocal Rank Fusion |
| Ranking | None | Cross-encoder reranker with precision before/after |
| LLM Output | Free text string | Structured JSON via Pydantic + function calling |
| Latency | Full response only | Token-by-token SSE streaming |
| Hallucination | Hope for the best | Grounding score: is answer in retrieved chunks? |
| Memory | Single-turn Q&A | Conversation buffer (5 turns) with eval comparison |
| Cost | Ignored | Per-request token tracking + $/query on Prometheus |
| Quality | "It looks right" | 100+ golden QA eval (RAGAS: faithfulness, relevance, precision, recall) |
| Failures | Unknown | 20 failure cases root-caused, 10+ fixed with hybrid search |
| Security | None | Input guardrail (prompt injection) + output guardrail (grounding threshold) |

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI, Pydantic v2, SSE streaming |
| Agent | LangGraph (state machine with routing) |
| Vector DB | ChromaDB (dense) + rank_bm25 (sparse) |
| Reranker | sentence-transformers cross-encoder |
| LLM | OpenAI GPT-4o (structured output) |
| Eval | RAGAS, custom golden QA harness |
| Observability | Prometheus + Grafana |
| Infra | Docker Compose, GitHub Actions CI/CD |
| UI | Streamlit |

## Project Structure

```
ops-copilot/
├── src/
│   ├── main.py              # FastAPI app, routes, middleware
│   ├── schemas.py            # Pydantic request/response models
│   ├── agent/
│   │   ├── graph.py          # LangGraph state machine
│   │   ├── nodes.py          # Agent nodes (classify, retrieve, generate)
│   │   └── tools.py          # Calculator, device status, alert query
│   ├── rag/
│   │   ├── loader.py         # Document loading + chunking strategies
│   │   ├── retriever.py      # Hybrid retrieval (dense + BM25 + RRF)
│   │   ├── reranker.py       # Cross-encoder reranking
│   │   └── embeddings.py     # Embedding model abstraction
│   ├── guardrails/
│   │   ├── input_guard.py    # Prompt injection detection
│   │   └── output_guard.py   # Grounding score check
│   ├── eval/
│   │   ├── golden_qa.json    # 100+ question-answer pairs
│   │   ├── run_eval.py       # RAGAS evaluation runner
│   │   └── benchmarks/       # Chunking + embedding comparison results
│   └── observability/
│       └── metrics.py        # Prometheus counters, histograms, gauges
├── data/
│   └── sops/                 # 5+ manufacturing SOPs (PDF/TXT)
├── tests/
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── README.md
```

## Quick Start

```bash
# 1. Clone and enter
git clone https://github.com/siddgawad/Ops-Copilot-Agentic-RAG-for-Manufacturing.git
cd Ops-Copilot-Agentic-RAG-for-Manufacturing

# 2. Create virtual environment
py -m venv .venv
.\.venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
cp .env.example .env
# Add your OPENAI_API_KEY

# 5. Run
uvicorn src.main:app --reload

# 6. Open docs
# http://127.0.0.1:8000/docs
```

## Current Progress

- [x] FastAPI + Pydantic schemas (`/health`, `/ask`)
- [ ] ChromaDB + document loading + 3 chunking strategies
- [ ] Embedding model benchmark
- [ ] Hybrid retrieval (dense + BM25)
- [ ] Cross-encoder reranker
- [ ] LangGraph agent (classify → route → generate)
- [ ] Tools (calculator, device status)
- [ ] Structured LLM output
- [ ] SSE streaming
- [ ] Conversation memory
- [ ] Input/output guardrails
- [ ] Grounding check
- [ ] Cost tracking (Prometheus)
- [ ] 100+ golden QA eval (RAGAS)
- [ ] Failure analysis (20 cases)
- [ ] Streamlit UI
- [ ] Docker Compose
- [ ] CI/CD (GitHub Actions)

## License

MIT
