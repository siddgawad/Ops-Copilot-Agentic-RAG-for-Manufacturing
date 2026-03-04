from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


def generate_answer(question: str, chunks: list[str], history: list[dict] = None) -> str:
    """Generate a grounded answer using OpenAI GPT-4o-mini with optional conversation history."""
    context = "\n\n".join(chunks)
    
    system_prompt = """You are a Senior Mechanical Engineer assisting factory operators.
You must answer their question using ONLY the provided technical context. 
RULES:
1. The context may contain raw tables, disjointed PDF text, or fragmented sentences. 
2. You are expected to interpret manufacturing terminology. If they ask about "motion range", and the context provides "motion limit" or "degrees", synthesize that data for them.
3. If the context contains a partial answer, provide what you know and explicitly state what is missing.
4. If the context is completely irrelevant to the question, ONLY then reply: "I don't have that information in my SOPs."
5. Be concise and professional. Factory operators need quick, clear answers."""

    user_prompt = f"""CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    # Build messages with optional conversation history
    messages = [{"role": "system", "content": system_prompt}]
    
    if history:
        for turn in history[-3:]:  # Keep last 3 turns for context
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})
    
    messages.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2
    )
    
    return response.choices[0].message.content