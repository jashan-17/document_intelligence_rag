import os

import requests
from openai import OpenAI


OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL", "http://localhost:11434/api/embeddings")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")


def use_openai_embeddings() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embedding(text: str) -> list[float]:
    if use_openai_embeddings():
        client = get_openai_client()
        response = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=text)
        return response.data[0].embedding

    response = requests.post(
        OLLAMA_EMBED_URL,
        json={
            "model": OLLAMA_EMBED_MODEL,
            "prompt": text,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def get_embeddings(texts: list[str]) -> list[list[float]]:
    return [get_embedding(text) for text in texts]
