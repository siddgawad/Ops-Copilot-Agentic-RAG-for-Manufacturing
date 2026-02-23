from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

def generate_answer(question: str, chunks: list[str]) -> str:
    context = "\n\n".join(chunks)
    
    prompt = f"""You are an expert manufacturing operations assistant.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have that information in my SOPs."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    
    return response.choices[0].message.content