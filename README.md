# Ops Copilot — Hybrid RAG for Manufacturing Operations

A production-deployed Retrieval-Augmented Generation system that answers manufacturing operations questions using Fanuc robot technical manuals and SOPs. Combines semantic vector search with keyword search via Reciprocal Rank Fusion for accurate, citation-backed responses.

**[Live Demo](https://ops-copilot-agentic-rag-for-manufacturing.onrender.com/docs)** · **[API Docs](https://ops-copilot-agentic-rag-for-manufacturing.onrender.com/docs)**

---

## How It Works

```
User Question
     │
     ▼
┌─────────────────────────────────────────────────┐
│              FastAPI Backend (Render)            │
│                                                 │
│  ┌──────────────┐    ┌──────────────┐           │
│  │  ChromaDB    │    │    BM25      │           │
│  │  (Semantic)  │    │  (Keyword)   │           │
│  └──────┬───────┘    └──────┬───────┘           │
│         │                   │                   │
│         └─────────┬─────────┘                   │
│                   │                             │
│         Reciprocal Rank Fusion                  │
│                   │                             │
│                   ▼                             │
│          Top 3 Ranked Chunks                    │
│                   │                             │
│                   ▼                             │
│          OpenAI GPT-4o-mini                     │
│          (Grounded Generation)                  │
│                   │                             │
│                   ▼                             │
│     Answer + Source Citations + Scores          │
└─────────────────────────────────────────────────┘
     │
     ▼
Next.js Frontend (Vercel)
```

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Backend | Python 3.11, FastAPI | REST API, request handling |
| Vector DB | ChromaDB + OpenAI `text-embedding-3-small` | Semantic similarity search |
| Keyword Search | BM25Okapi (rank-bm25) | Exact term matching |
| Fusion | Reciprocal Rank Fusion (k=60) | Merges semantic + keyword rankings |
| LLM | OpenAI GPT-4o-mini | Answer generation from retrieved context |
| Frontend | Next.js 15, React 19, Tailwind CSS | Chat interface |
| Deployment | Render (backend), Vercel (frontend) | Cloud hosting |

## Project Structure

```
├── src/
│   ├── main.py              # FastAPI app, CORS, routes
│   ├── schemas.py           # Pydantic request/response models
│   └── rag/
│       ├── retriever.py     # VectorStore: chunking, indexing, hybrid search
│       └── generator.py     # OpenAI prompt construction, answer generation
├── tests/
│   ├── test_retriever.py    # Unit tests for chunking and search logic
│   └── test_api.py          # Integration tests for API endpoints
├── frontend/                # Next.js chat UI
├── data/                    # Fanuc manuals + SOPs (PDFs and TXT)
├── requirements.txt
├── render.yaml              # Render deployment config
└── README.md
```

## Data Sources

| Document | Pages | Content |
|----------|-------|---------|
| Fanuc LR Mate 200iD Operators Manual | ~250 | Robot operation, safety, maintenance |
| Fanuc KAREL Language Reference (7066350) | ~500 | Programming reference, error codes |
| 5 Custom SOPs | — | Spindle vibration, coolant, E-stop recovery, inspection, PM |

## API Endpoints

### `POST /ask`
```json
// Request
{
  "question": "What is the motion range of the J1 axis?",
  "history": []
}

// Response
{
  "answer": "The J1 axis motion range is ±170 degrees...",
  "sources": [
    {
      "text": "J1 axis: Motion range ±170°, maximum speed...",
      "source": "Fanuc Robot LR Mate 200iD Operators Manual.pdf",
      "score": 0.033333
    }
  ]
}
```

### `GET /health`
```json
{
  "status": "online",
  "service": "ops-copilot",
  "chunks_indexed": 847
}
```

## Local Development

```bash
# 1. Clone
git clone https://github.com/siddgawad/Ops-Copilot-Agentic-RAG-for-Manufacturing.git
cd Ops-Copilot-Agentic-RAG-for-Manufacturing

# 2. Backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env       # Add your OPENAI_API_KEY
uvicorn src.main:app --reload --port 8000

# 3. Frontend
cd frontend
npm install
npm run dev                 # Opens on localhost:3000

# 4. Run Tests
pytest tests/ -v
```

## Deployment

### Backend (Render)
1. Create a new **Web Service** on [render.com](https://render.com)
2. Connect your GitHub repo
3. **Root Directory:** leave empty
4. **Build Command:** `pip install -r requirements.txt`
5. **Start Command:** `uvicorn src.main:app --host 0.0.0.0 --port $PORT`
6. Add environment variable: `OPENAI_API_KEY`

### Frontend (Vercel)
1. Import the repo on [vercel.com](https://vercel.com)
2. **Root Directory:** `frontend`
3. **Framework Preset:** Next.js
4. Add environment variable: `NEXT_PUBLIC_API_URL` = your Render URL

## Design Decisions

- **Hybrid search over pure vector search:** Manufacturing manuals contain exact part numbers, alarm codes, and axis designations (e.g., "J5", "Alarm 503") that vector embeddings handle poorly. BM25 catches these exact matches. RRF merges both rankings without needing to tune weights.
- **OpenAI embeddings over local models:** Render's free tier has 512MB RAM. The default ChromaDB embedding model (all-MiniLM-L6-v2) is ~400MB. Using the OpenAI API for embeddings keeps server memory under 200MB.
- **100-word chunks with character truncation:** Fanuc manuals contain dense code blocks that tokenize into far more tokens than normal English. Hard-capping chunks at 4000 characters prevents exceeding OpenAI's 8192 token-per-embedding limit.
- **Batch embedding with retry:** Chunks are embedded in batches of 5 with per-batch error handling. If a batch fails, each chunk is retried individually so one bad chunk doesn't kill the entire indexing run.

## Running Tests

```bash
pytest tests/ -v
```

Tests cover:
- Chunk text splitting produces valid output
- Chunk character truncation enforces the 4000-char limit
- API health endpoint returns correct schema
- API ask endpoint returns answer + source citations

## License

MIT
