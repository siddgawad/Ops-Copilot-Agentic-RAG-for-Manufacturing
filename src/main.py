from src.schemas import QueryRequest, QueryResponse
from fastapi import FastAPI

#initialise the fastapi application 

app = FastAPI(
    title = "Ops Copilot",
    description="Agentic RAG for manufacturing operations",
    version="0.1.0"
)

#this is a decorator which tells FastAPI when someone sends a GET request to the root URL '/' - at that point we should run this function 

@app.get('/health')
async def health_check():
    return{
        "status":"online",
        "service":"ops-copilot"
    }


@app.post('/ask',
response_model=QueryResponse)
async def ask_agent(request:QueryRequest):
    #for now we just echo back a fake answer 
    #by week 2 we will trigger the langraph ai to actually process it 
    fake_answer=f"You asked about `{request.question}`. The AI is not awake yet!"

    if request.machine_id:
        fake_answer+= f"(Checking machine : {request.machine_id})"

    return QueryResponse(answer=fake_answer)

