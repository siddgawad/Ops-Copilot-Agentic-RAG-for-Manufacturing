from src.schemas import QueryRequest, QueryResponse
from fastapi import FastAPI
from src.rag.retriever import VectorStore
from src.rag.generator import generate_answer

# Initialize the FastAPI application 
app = FastAPI(
    title="Ops Copilot",
    description="Agentic RAG for manufacturing operations",
    version="0.1.0"
)

# Initialize the vector store once when the app starts
db = VectorStore()

@app.get('/health')
async def health_check():
    return {
        "status": "online",
        "service": "ops-copilot"
    }

@app.post('/ask', response_model=QueryResponse)
async def ask_agent(request: QueryRequest):
    # 1. Retrieve the top 3 most relevant chunks from ChromaDB
    results = db.search(query=request.question, n_results=3)
    chunks = results['documents'][0]
    
    # 2. Generate the answer using OpenAI
    answer = generate_answer(request.question, chunks)

    # (Optional) Append machine ID info if the user provided it
    if request.machine_id:
        answer += f"\n\n[Context applied for machine: {request.machine_id}]"

    # 3. Return the exact response schema expected by the frontend
    return QueryResponse(answer=answer)
