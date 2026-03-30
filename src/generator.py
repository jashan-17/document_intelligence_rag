import re


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9]{3,}", text.lower()))


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def generate_answer(query: str, retrieved_chunks: list[tuple[str, float]]) -> str:
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
