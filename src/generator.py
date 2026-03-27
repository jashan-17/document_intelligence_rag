import os

import requests
from openai import OpenAI


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


def use_openai_generation() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_answer(query: str, retrieved_chunks: list[tuple[str, float]], model: str = OLLAMA_MODEL) -> str:
    context = "\n\n".join(
        [f"[Source {i+1}]\n{chunk}" for i, (chunk, _) in enumerate(retrieved_chunks)]
    )

    prompt = f"""
You are a document question-answering assistant.
Answer the question using only the provided context.
If the answer is not in the context, say you could not find it in the documents.

Context:
{context}

Question:
{query}

Give a concise answer and cite the source numbers used.
"""

    if use_openai_generation():
        client = get_openai_client()
        response = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You answer questions using only the supplied document context.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        return response.choices[0].message.content or "No answer returned."

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()
    return response.json()["response"]
