from src.schemas import QueryRequest, QueryResponse, SourceCitation
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.rag.retriever import VectorStore
from src.rag.generator import generate_answer

app = FastAPI(
    title="Ops Copilot",
    description="Hybrid RAG pipeline for manufacturing operations — BM25 + Vector Search + Reciprocal Rank Fusion",
    version="1.0.0"
)

# Allow requests from Vercel frontend (and localhost for dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten to your Vercel URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the vector store once when the app starts
db = VectorStore()
db.load_documents_from_folder("data")


@app.get('/health')
async def health_check():
    return {
        "status": "online",
        "service": "ops-copilot",
        "chunks_indexed": len(db.raw_chunks)
    }


@app.post('/ask', response_model=QueryResponse)
async def ask_agent(request: QueryRequest):
    # 1. Hybrid retrieval: ChromaDB (semantic) + BM25 (keyword) → RRF fusion
    results = db.search(query=request.question, n_results=3)

    # 2. Extract chunk texts for the generator
    chunks = [r["text"] for r in results]

    # 3. Generate the answer using OpenAI
    answer = generate_answer(request.question, chunks, history=request.history or [])

    # 4. Build source citations
    sources = [
        SourceCitation(text=r["text"][:200] + "...", source=r["source"], score=r["score"])
        for r in results
    ]

    return QueryResponse(answer=answer, sources=sources)
