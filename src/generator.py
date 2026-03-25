import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:mini"


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