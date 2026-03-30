import os
import re
from urllib.parse import urlparse

from openai import OpenAI


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9]{3,}", text.lower()))


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _extractive_fallback(query: str, retrieved_chunks: list[tuple[str, float]]) -> str:
    if not retrieved_chunks:
        return "I could not find relevant information in the uploaded documents."

    query_terms = _tokenize(query)
    scored_sentences: list[tuple[int, int, str]] = []

    for source_index, (chunk, _) in enumerate(retrieved_chunks, start=1):
        for sentence in _split_sentences(chunk):
            sentence_terms = _tokenize(sentence)
            overlap = len(query_terms & sentence_terms)
            if overlap > 0:
                scored_sentences.append((overlap, source_index, sentence))

    if not scored_sentences:
        fallback_chunk = retrieved_chunks[0][0]
        fallback_sentences = _split_sentences(fallback_chunk)[:2]
        if not fallback_sentences:
            return "I could not find enough readable text in the retrieved document sections."
        return f"{' '.join(fallback_sentences)} [Source 1]"

    scored_sentences.sort(key=lambda item: (-item[0], item[1], len(item[2])))
    chosen: list[tuple[int, str]] = []
    seen_sentences = set()

    for _, source_index, sentence in scored_sentences:
        if sentence in seen_sentences:
            continue
        chosen.append((source_index, sentence))
        seen_sentences.add(sentence)
        if len(chosen) == 3:
            break

    answer_text = " ".join(sentence for _, sentence in chosen)
    cited_sources = ", ".join(str(source_index) for source_index, _ in chosen)
    return f"{answer_text} [Sources: {cited_sources}]"


def _normalized_base_url() -> str | None:
    base_url = os.getenv("LLM_BASE_URL", "").strip()
    if not base_url:
        return None

    normalized_base_url = base_url.rstrip("/")
    if not normalized_base_url.endswith("/v1"):
        normalized_base_url = f"{normalized_base_url}/v1"
    return normalized_base_url


def get_llm_settings() -> dict[str, str | None]:
    normalized_base_url = _normalized_base_url()
    parsed = urlparse(normalized_base_url) if normalized_base_url else None

    return {
        "base_url": normalized_base_url,
        "display_host": parsed.netloc if parsed else None,
        "model": os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        "api_key": os.getenv("LLM_API_KEY", "local-inference-key"),
    }


def remote_llm_configured() -> bool:
    return bool(_normalized_base_url())


def check_remote_llm_health() -> tuple[bool, str]:
    settings = get_llm_settings()
    base_url = settings["base_url"]

    if not base_url:
        return False, "No public LLM endpoint is configured."

    try:
        client = OpenAI(
            api_key=settings["api_key"],
            base_url=base_url,
            timeout=10.0,
        )
        client.models.list()
        return True, f"Connected to {settings['display_host']}."
    except Exception as exc:
        return False, f"Could not reach the public LLM endpoint: {exc}"


def generate_answer(query: str, retrieved_chunks: list[tuple[str, float]]) -> str:
    settings = get_llm_settings()
    base_url = settings["base_url"]
    if not base_url:
        return _extractive_fallback(query, retrieved_chunks)

    if not retrieved_chunks:
        return "I could not find relevant information in the uploaded documents."

    context = "\n\n".join(
        [f"[Source {i + 1}]\n{chunk}" for i, (chunk, _) in enumerate(retrieved_chunks)]
    )

    prompt = f"""
You are a document question-answering assistant.
Answer the user's question using only the provided context.
If the answer is not present in the context, say you could not find it in the documents.
Keep the answer concise and cite the source numbers you used.

Context:
{context}

Question:
{query}
"""

    try:
        client = OpenAI(
            api_key=settings["api_key"],
            base_url=base_url,
            timeout=60.0,
        )

        response = client.chat.completions.create(
            model=settings["model"] or "Qwen/Qwen2.5-7B-Instruct",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a document question-answering assistant. "
                        "Answer only from the provided context and cite source numbers."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or _extractive_fallback(query, retrieved_chunks)
    except Exception:
        return _extractive_fallback(query, retrieved_chunks)
