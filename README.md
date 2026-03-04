# 🏭 Ops Copilot — Hybrid RAG for Manufacturing Operations

> **Live demo:** https://ops-manufacturing-rag.streamlit.app/

## 🎯 Why this project matters
Manufacturing operators need instant, reliable answers from dense technical documentation (PDF manuals, SOPs). **Ops Copilot** demonstrates a production‑ready Retrieval‑Augmented Generation (RAG) system that:
- Retrieves from **real Fanuc robot manuals** (500+ pages) and internal SOPs.
- Provides **source citations** with relevance scores, ensuring traceability.
- Maintains **conversation memory** for multi‑turn interactions.
## 🚀 Architecture: Render + Vercel
The project has been scaled up from a single Streamlit script to a modern, decoupled architecture:
1. **Backend API (Render):** FastAPI powers the core RAG logic (ChromaDB, BM25, Reciprocal Rank Fusion, OpenAI).
2. **Frontend UI (Vercel):** A sleek, dark-mode Next.js 15 application built with React and Tailwind CSS.

## 🛠️ Tech Stack
| Layer | Technology |
|---|---|
| **API** | FastAPI, Pydantic v2, Uvicorn |
| **Vector DB** | ChromaDB (HNSW, cosine) |
| **Keyword Search** | BM25 (rank‑bm25) |
| **Fusion** | Reciprocal Rank Fusion (k=60) |
| **LLM** | OpenAI GPT‑4o‑mini |
| **Frontend** | Next.js 15, React 19, Tailwind CSS, Lucide Icons |

## 📂 Project Structure
```text
ops-copilot/
├─ frontend/           # Next.js UI (deployed to Vercel)
│  ├─ src/app/         # React pages & components
│  └─ package.json     
├─ src/
│  ├─ main.py          # FastAPI endpoints (deployed to Render)
│  ├─ schemas.py       # Pydantic models (incl. SourceCitation & History)
│  └─ rag/
│     ├─ retriever.py  # Hybrid retrieval implementation
│     └─ generator.py  # Prompt construction & OpenAI call
├─ data/                # PDF manuals & SOP txt files
├─ requirements.txt
├─ render.yaml          # Render deployment config
└─ README.md
```

## ▶️ Quick Start (local dev)
```bash
# 1. Start the FastAPI Backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env # Add your OPENAI_API_KEY
uvicorn src.main:app --reload --port 8000

# 2. Start the Next.js Frontend
cd frontend
npm install
npm run dev
# App will run at http://localhost:3000
```

## 🌐 Deployment Instructions

### 1. Deploy the Backend (Render)
Make sure `render.yaml` is committed to your repository.
1. Sign in to [Render](https://render.com/).
2. Click **New** -> **Web Service** -> **Build and deploy from a Git repository**.
3. Connect your GitHub fork. Render will automatically detect the settings from `render.yaml`.
4. In the Render dashboard for your service, go to **Environment** and add `OPENAI_API_KEY` with your actual key.
5. Copy your Render service URL (e.g., `https://ops-copilot-api.onrender.com`).

### 2. Deploy the Frontend (Vercel)
1. Sign in to [Vercel](https://vercel.com/).
2. Click **Add New** -> **Project** and import your GitHub fork.
3. In the "Framework Preset" settings, Vercel should auto-detect Next.js.
4. **Important**: Change the "Root Directory" to `frontend`.
5. Under **Environment Variables**, add `NEXT_PUBLIC_API_URL` and set it to your Render service URL (no trailing slash).
6. Click **Deploy**.

## 📊 Sample Queries
- *"What vibration level requires immediate spindle shutdown?"*
- *"What is the maximum torque for spindle bolts?"*
- *"How do I perform a First Article Inspection?"*
- *"What are the E‑stop recovery steps for the Fanuc LR Mate?"*

## 📈 Impact & Next Steps
- **Potential employer showcase**: Demonstrates a full-stack, enterprise-grade AI architecture (decoupled frontend/backend) preferred by Tier-1 employers.

## 📜 License
MIT © 2024‑2026 Siddharth Gawad
