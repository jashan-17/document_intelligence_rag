from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import streamlit as st
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.loader import load_document


UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

REFUSAL_MESSAGE = "I could not find that in the uploaded documents."

METADATA_LABELS = {
    "title": ["title", "document title", "name of the document"],
    "author": ["author", "written by", "who wrote", "creator"],
    "published": ["published", "publication date", "publish date", "date"],
}
METADATA_FIELDS = tuple(METADATA_LABELS.keys())

CONTENT_INTENT_KEYWORDS = {
    "uses": ["use", "uses", "application", "applications", "example", "examples"],
    "challenges": ["challenge", "challenges", "risk", "risks", "problem", "problems"],
    "benefits": ["help", "helps", "benefit", "benefits", "improve", "improves"],
    "summary": ["summarize", "summary", "overview", "main point", "what is this about"],
}

KNOWN_USE_CASES = [
    "medical imaging",
    "predictive analytics",
    "drug discovery",
    "patient monitoring",
    "clinical decision support",
    "automating administrative tasks",
]

KNOWN_CHALLENGES = [
    "privacy concerns",
    "incorrect diagnoses",
    "data privacy",
    "bias",
    "security concerns",
    "limited training data",
]

APP_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(60, 130, 246, 0.12), transparent 28%),
            linear-gradient(180deg, #f8fbff 0%, #eef4fb 100%);
        color: #0f172a;
    }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at top left, rgba(60, 130, 246, 0.12), transparent 28%),
            linear-gradient(180deg, #f8fbff 0%, #eef4fb 100%);
    }
    [data-testid="stHeader"] {
        background: #1f2937;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #3f4256 0%, #4b4e63 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    [data-testid="stSidebar"] * {
        color: #f8fafc;
    }
    section[data-testid="stSidebar"] .stButton > button,
    section[data-testid="stSidebar"] .stCheckbox label,
    section[data-testid="stSidebar"] .stCaption {
        color: #f8fafc !important;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4, h5, h6, p, label, div, span {
        color: inherit;
    }
    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 60%, #38bdf8 100%);
        padding: 1.5rem 1.75rem;
        border-radius: 20px;
        color: white;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.18);
        margin-bottom: 1rem;
    }
    .hero-card h1 {
        color: white;
        margin: 0 0 0.35rem 0;
        font-size: 2.1rem;
    }
    .hero-card p {
        margin: 0;
        color: rgba(255, 255, 255, 0.9);
        font-size: 1rem;
        max-width: 52rem;
    }
    .stat-card {
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
    }
    .stat-label {
        font-size: 0.85rem;
        color: #475569;
        margin-bottom: 0.15rem;
    }
    .stat-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #0f172a;
    }
    .chip-row {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin: 0.5rem 0 0.75rem 0;
    }
    .mode-pill {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(59, 130, 246, 0.12);
        color: #1d4ed8;
        margin-right: 0.45rem;
    }
    .stButton > button {
        background: #1f2937;
        color: #ffffff;
        border: 1px solid rgba(15, 23, 42, 0.15);
        border-radius: 12px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background: #0f172a;
        color: #ffffff;
        border-color: rgba(15, 23, 42, 0.25);
    }
    .stTextInput > div > div > input,
    [data-testid="stChatInputTextArea"] textarea,
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255, 255, 255, 0.92) !important;
        color: #0f172a !important;
        border: 1px solid rgba(148, 163, 184, 0.35) !important;
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: #0f172a !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: #1f2937 !important;
        color: #ffffff !important;
    }
    [data-testid="stChatInput"] {
        background: transparent;
    }
    [data-testid="stChatInput"] textarea {
        background: rgba(255, 255, 255, 0.96) !important;
        color: #0f172a !important;
    }
    [data-testid="stInfo"],
    [data-testid="stSuccess"],
    [data-testid="stWarning"],
    [data-testid="stError"] {
        background: rgba(255, 255, 255, 0.84);
        border: 1px solid rgba(148, 163, 184, 0.28);
        color: #0f172a;
    }
    [data-testid="stInfo"] *,
    [data-testid="stSuccess"] *,
    [data-testid="stWarning"] *,
    [data-testid="stError"] * {
        color: #0f172a !important;
    }
    [data-testid="stMarkdownContainer"] p {
        color: #0f172a;
    }
    .stCaption {
        color: #475569 !important;
    }
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.86);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 14px;
    }
</style>
"""


def init_session_state() -> None:
    defaults: dict[str, Any] = {
        "documents": [],
        "body_chunks": [],
        "vectorizer": None,
        "chunk_matrix": None,
        "indexed_files": [],
        "document_fingerprints": [],
        "current_question": "",
        "answer_history": [],
        "retrieved_sources": [],
        "debug_mode": False,
        "last_debug": {},
        "last_prompt": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_chat() -> None:
    st.session_state.current_question = ""
    st.session_state.answer_history = []
    st.session_state.retrieved_sources = []
    st.session_state.last_debug = {}
    st.session_state.last_prompt = ""


def reset_documents() -> None:
    st.session_state.documents = []
    st.session_state.body_chunks = []
    st.session_state.vectorizer = None
    st.session_state.chunk_matrix = None
    st.session_state.indexed_files = []
    st.session_state.document_fingerprints = []
    clear_chat()


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]{2,}", text.lower())


def split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []
    return [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", normalized) if piece.strip()]


def lexical_overlap_ratio(query: str, text: str) -> float:
    query_tokens = set(tokenize(query))
    text_tokens = set(tokenize(text))
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def clean_answer_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return text


def extract_text_from_file(file_path: str) -> str:
    return normalize_text(load_document(file_path))


def extract_value_after_label(line: str, field: str) -> str | None:
    for label in METADATA_LABELS[field]:
        match = re.match(rf"^{re.escape(label)}\s*:\s*(.+)$", line.strip(), flags=re.IGNORECASE)
        if match:
            return clean_answer_text(match.group(1))
    return None


def extract_document_metadata(text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    candidate_lines = [line.strip() for line in text.splitlines() if line.strip()][:60]

    for line in candidate_lines:
        for field in METADATA_FIELDS:
            value = extract_value_after_label(line, field)
            if value and field not in metadata:
                metadata[field] = value

    if "title" not in metadata and candidate_lines:
        first_line = candidate_lines[0]
        if len(first_line) <= 120 and ":" not in first_line:
            metadata["title"] = clean_answer_text(first_line)

    if "published" not in metadata:
        year_match = re.search(r"\b(19|20)\d{2}\b", text[:1600])
        if year_match:
            metadata["published"] = year_match.group(0)

    return metadata


def split_document_content(text: str, metadata: dict[str, str]) -> tuple[list[str], str]:
    lines = [line.strip() for line in text.splitlines()]
    body_lines: list[str] = []
    removed_metadata_lines: list[str] = []

    metadata_patterns = []
    for field in METADATA_FIELDS:
        for label in METADATA_LABELS[field]:
            metadata_patterns.append(re.compile(rf"^{re.escape(label)}\s*:\s*.+$", re.IGNORECASE))

    metadata_values = {value.strip().lower() for value in metadata.values() if value.strip()}
    skipped_leading_title = False

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            if body_lines and body_lines[-1] != "":
                body_lines.append("")
            continue

        is_metadata_line = any(pattern.match(stripped) for pattern in metadata_patterns)
        if not is_metadata_line and not skipped_leading_title and index == 0:
            if metadata.get("title") and stripped.lower() == metadata["title"].strip().lower():
                is_metadata_line = True
                skipped_leading_title = True

        if not is_metadata_line and stripped.lower() in metadata_values and len(stripped.split()) <= 8:
            is_metadata_line = True

        if is_metadata_line:
            removed_metadata_lines.append(stripped)
            continue

        body_lines.append(stripped)

    body_text = "\n".join(body_lines)
    body_text = re.sub(r"\n{3,}", "\n\n", body_text).strip()
    return removed_metadata_lines, body_text


def chunk_text(text: str, chunk_size: int = 700, overlap_sentences: int = 1) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n+", text) if paragraph.strip()]
    if not paragraphs:
        paragraphs = split_sentences(text)

    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
            continue

        current = ""
        for sentence in split_sentences(paragraph):
            candidate = f"{current} {sentence}".strip()
            if current and len(candidate) > chunk_size:
                units.append(current.strip())
                current = sentence
            else:
                current = candidate
        if current:
            units.append(current.strip())

    chunks: list[str] = []
    current_sentences: list[str] = []
    for unit in units:
        sentences = split_sentences(unit) or [unit]
        prospective = " ".join(current_sentences + sentences).strip()
        if current_sentences and len(prospective) > chunk_size:
            chunks.append(" ".join(current_sentences).strip())
            current_sentences = current_sentences[-overlap_sentences:] + sentences
        else:
            current_sentences.extend(sentences)

    if current_sentences:
        chunks.append(" ".join(current_sentences).strip())

    return [chunk for chunk in chunks if chunk]


def fingerprint_uploaded_files(uploaded_files: list[Any]) -> list[str]:
    fingerprints: list[str] = []
    for uploaded_file in uploaded_files:
        payload = uploaded_file.getvalue()
        digest = hashlib.md5(payload).hexdigest()
        fingerprints.append(f"{uploaded_file.name}:{len(payload)}:{digest}")
    return sorted(fingerprints)


def classify_question(question: str) -> str:
    lowered = question.lower()
    metadata_signals = [
        "author",
        "written by",
        "who wrote",
        "title",
        "name of the document",
        "published",
        "publication date",
    ]
    if any(signal in lowered for signal in metadata_signals):
        return "metadata"
    if "when was" in lowered and "document" in lowered:
        return "metadata"
    return "content"


def infer_content_intent(question: str) -> str:
    lowered = question.lower()
    for intent, keywords in CONTENT_INTENT_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return intent
    return "general"


def build_index(chunks: list[dict[str, Any]]) -> tuple[TfidfVectorizer | None, Any]:
    if not chunks:
        return None, None

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )
    matrix = vectorizer.fit_transform([chunk["search_text"] for chunk in chunks])
    return vectorizer, matrix


def try_metadata_answer(query: str, documents: list[dict[str, Any]]) -> tuple[str | None, dict[str, Any]]:
    lowered = query.lower()
    if "author" in lowered or "written by" in lowered or "who wrote" in lowered:
        field = "author"
    elif "title" in lowered or "name of the document" in lowered:
        field = "title"
    else:
        field = "published"

    debug: dict[str, Any] = {"mode": "metadata", "field": field, "metadata_candidates": []}
    for document in documents:
        value = document["metadata"].get(field)
        if value:
            debug["metadata_candidates"].append(
                {"doc_name": document["name"], "field": field, "value": value}
            )
            return value, debug

    return None, debug


def retrieve_chunks(
    query: str,
    vectorizer: TfidfVectorizer | None,
    chunk_matrix: Any,
    chunks: list[dict[str, Any]],
    top_k: int = 6,
) -> list[dict[str, Any]]:
    if not vectorizer or chunk_matrix is None or not chunks:
        return []

    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, chunk_matrix).flatten()
    intent = infer_content_intent(query)

    ranked: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        similarity = float(similarities[idx])
        overlap = lexical_overlap_ratio(query, chunk["search_text"])
        keyword_boost = 0.0
        chunk_lower = chunk["text"].lower()

        if intent == "uses":
            keyword_boost += 0.12 * sum(phrase in chunk_lower for phrase in KNOWN_USE_CASES)
        elif intent == "challenges":
            keyword_boost += 0.11 * sum(phrase in chunk_lower for phrase in KNOWN_CHALLENGES)
        elif intent == "benefits":
            keyword_boost += 0.05 * sum(word in chunk_lower for word in ["help", "helps", "improve", "support"])

        sentence_bonus = min(len(split_sentences(chunk["text"])), 5) / 20
        combined = (0.72 * similarity) + (0.22 * overlap) + keyword_boost + sentence_bonus

        ranked.append(
            {
                **chunk,
                "similarity": round(similarity, 4),
                "overlap": round(overlap, 4),
                "combined_score": round(combined, 4),
            }
        )

    ranked.sort(key=lambda item: item["combined_score"], reverse=True)
    return ranked[:top_k]


def extract_candidate_units(chunk_text_value: str) -> list[str]:
    sentences = split_sentences(chunk_text_value)
    lines = [line.strip() for line in chunk_text_value.splitlines() if line.strip()]
    candidates: list[str] = []
    seen: set[str] = set()
    for item in sentences + lines:
        item = clean_answer_text(item)
        if item and item not in seen:
            candidates.append(item)
            seen.add(item)
    return candidates


def find_phrase_in_text(text: str, phrase: str) -> str | None:
    match = re.search(re.escape(phrase), text, flags=re.IGNORECASE)
    if not match:
        return None
    return clean_answer_text(text[match.start() : match.end()])


def extract_list_item(text: str) -> str | None:
    patterns = [
        r"(?:such as|including|include|includes|like)\s+([^.;]+)",
        r"(?:used for|used in|helps with|supports)\s+([^.;]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            parts = [clean_answer_text(part.strip(" .")) for part in re.split(r",| and ", match.group(1)) if part.strip()]
            if parts:
                return parts[0]
    return None


def summarize_known_phrases(retrieved_chunks: list[dict[str, Any]], phrases: list[str], limit: int = 3) -> list[str]:
    found: list[str] = []
    for chunk in retrieved_chunks:
        for phrase in phrases:
            matched = find_phrase_in_text(chunk["text"], phrase)
            if matched and matched.lower() not in {item.lower() for item in found}:
                found.append(matched)
                if len(found) == limit:
                    return found
    return found


def score_candidate_answer(query: str, candidate: str, parent_chunk: dict[str, Any], intent: str) -> float:
    overlap = lexical_overlap_ratio(query, candidate)
    score = (0.62 * parent_chunk["combined_score"]) + (0.3 * overlap)
    candidate_lower = candidate.lower()

    if intent == "uses" and any(phrase in candidate_lower for phrase in KNOWN_USE_CASES):
        score += 0.22
    if intent == "challenges" and any(phrase in candidate_lower for phrase in KNOWN_CHALLENGES):
        score += 0.2
    if intent == "benefits" and any(term in candidate_lower for term in ["help", "improve", "support"]):
        score += 0.12
    if len(candidate) > 260:
        score -= 0.18
    if len(candidate.split()) < 3:
        score -= 0.12
    return round(score, 4)


def extract_answer_span(query: str, candidate: str, intent: str) -> str:
    if intent == "uses":
        for phrase in KNOWN_USE_CASES:
            matched = find_phrase_in_text(candidate, phrase)
            if matched:
                return matched
        list_item = extract_list_item(candidate)
        if list_item:
            return list_item

    if intent == "challenges":
        found = []
        for phrase in KNOWN_CHALLENGES:
            matched = find_phrase_in_text(candidate, phrase)
            if matched and matched.lower() not in {item.lower() for item in found}:
                found.append(matched)
        if found:
            return ", ".join(found[:3])

    return clean_answer_text(candidate)


def synthesize_local_answer(query: str, retrieved_chunks: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    intent = infer_content_intent(query)
    debug: dict[str, Any] = {
        "mode": "local-extractive",
        "intent": intent,
        "selected_span": None,
        "candidate_answers": [],
    }

    if not retrieved_chunks:
        debug["refusal_reason"] = "no_retrieved_chunks"
        return REFUSAL_MESSAGE, debug

    if retrieved_chunks[0]["combined_score"] < 0.18:
        debug["refusal_reason"] = f"low_retrieval_score:{retrieved_chunks[0]['combined_score']}"
        return REFUSAL_MESSAGE, debug

    if intent == "uses":
        known_matches = summarize_known_phrases(retrieved_chunks, KNOWN_USE_CASES, limit=3)
        if known_matches:
            debug["selected_span"] = known_matches[0]
            debug["known_phrase_matches"] = known_matches
            return known_matches[0], debug

    if intent == "challenges":
        known_matches = summarize_known_phrases(retrieved_chunks, KNOWN_CHALLENGES, limit=3)
        if known_matches:
            answer = ", ".join(known_matches[:3])
            debug["selected_span"] = answer
            debug["known_phrase_matches"] = known_matches
            return answer, debug

    candidates: list[dict[str, Any]] = []
    for chunk in retrieved_chunks:
        for candidate in extract_candidate_units(chunk["text"]):
            answer_span = extract_answer_span(query, candidate, intent)
            candidate_score = score_candidate_answer(query, candidate, chunk, intent)
            candidates.append(
                {
                    "doc_name": chunk["doc_name"],
                    "chunk_id": chunk["chunk_id"],
                    "candidate": candidate,
                    "answer_span": answer_span,
                    "score": candidate_score,
                }
            )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    debug["candidate_answers"] = candidates[:8]

    if not candidates or candidates[0]["score"] < 0.2:
        debug["refusal_reason"] = "weak_candidate_score"
        return REFUSAL_MESSAGE, debug

    top_candidates = candidates[:2]
    selected = top_candidates[0]["answer_span"]
    if intent in {"benefits", "summary", "general"} and len(top_candidates) > 1:
        secondary = top_candidates[1]["answer_span"]
        if secondary.lower() != selected.lower() and lexical_overlap_ratio(query, secondary) > 0.18:
            selected = f"{selected} {secondary}"

    selected = clean_answer_text(selected)
    debug["selected_span"] = selected
    return selected, debug


def calculate_confidence(sources: list[dict[str, Any]], debug: dict[str, Any]) -> float:
    if not sources:
        return 0.98 if debug.get("mode") == "metadata" else 0.0
    top_score = sources[0]["combined_score"]
    if debug.get("mode") == "remote-llm":
        return round(min(0.95, 0.45 + top_score), 2)
    if debug.get("mode") == "local-extractive":
        candidate_answers = debug.get("candidate_answers", [])
        best_candidate_score = candidate_answers[0]["score"] if candidate_answers else top_score
        return round(min(0.94, 0.35 + top_score + (best_candidate_score / 2.5)), 2)
    return round(min(0.98, 0.6 + top_score), 2)


def get_llm_settings() -> dict[str, str | None]:
    base_url = os.getenv("LLM_BASE_URL", "").strip()
    if base_url:
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
    parsed = urlparse(base_url) if base_url else None

    return {
        "base_url": base_url or None,
        "display_host": parsed.netloc if parsed else None,
        "api_key": os.getenv("LLM_API_KEY", "local-inference-key"),
        "model": os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    }


def remote_llm_configured() -> bool:
    return bool(get_llm_settings()["base_url"])


def build_remote_prompt(query: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    sources = "\n\n".join(
        f"[Source {index}] {chunk['doc_name']} (score={chunk['combined_score']})\n{chunk['text']}"
        for index, chunk in enumerate(retrieved_chunks, start=1)
    )
    return (
        "You are a careful document question-answering assistant.\n"
        "Answer only from the provided sources.\n"
        "If the answer is not present, reply exactly: I could not find that in the uploaded documents.\n"
        "Be concise. Prefer short direct answers for factual questions.\n\n"
        f"Retrieved sources:\n{sources}\n\n"
        f"Question: {query}"
    )


def answer_with_remote_llm(query: str, retrieved_chunks: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    settings = get_llm_settings()
    prompt = build_remote_prompt(query, retrieved_chunks)
    debug: dict[str, Any] = {
        "mode": "remote-llm",
        "prompt": prompt,
        "model": settings["model"],
        "host": settings["display_host"],
    }

    if not settings["base_url"]:
        debug["fallback_reason"] = "no_llm_configured"
        answer, local_debug = synthesize_local_answer(query, retrieved_chunks)
        debug["local_fallback"] = local_debug
        return answer, debug

    try:
        client = OpenAI(
            api_key=settings["api_key"],
            base_url=settings["base_url"],
            timeout=45.0,
        )
        response = client.chat.completions.create(
            model=settings["model"] or "Qwen/Qwen2.5-7B-Instruct",
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer only from the provided document sources. "
                        "If the answer is not present, say you could not find it in the uploaded documents."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        answer = clean_answer_text(response.choices[0].message.content or "")
        if not answer:
            raise ValueError("Empty LLM response")
        return answer, debug
    except Exception as exc:
        debug["fallback_reason"] = str(exc)
        answer, local_debug = synthesize_local_answer(query, retrieved_chunks)
        debug["local_fallback"] = local_debug
        return answer, debug


def answer_question(query: str) -> dict[str, Any]:
    question_type = classify_question(query)
    debug_payload: dict[str, Any] = {"question_type": question_type}

    if question_type == "metadata":
        metadata_answer, metadata_debug = try_metadata_answer(query, st.session_state.documents)
        debug_payload["metadata_debug"] = metadata_debug
        if metadata_answer:
            debug_payload["mode"] = "metadata"
            debug_payload["selected_span"] = metadata_answer
            return {
                "answer": metadata_answer,
                "sources": [],
                "debug": debug_payload,
                "confidence": calculate_confidence([], debug_payload),
            }
        return {
            "answer": REFUSAL_MESSAGE,
            "sources": [],
            "debug": debug_payload,
            "confidence": 0.0,
        }

    retrieved = retrieve_chunks(
        query=query,
        vectorizer=st.session_state.vectorizer,
        chunk_matrix=st.session_state.chunk_matrix,
        chunks=st.session_state.body_chunks,
        top_k=6,
    )
    debug_payload["retrieved_chunks"] = retrieved

    if not retrieved or retrieved[0]["combined_score"] < 0.18:
        debug_payload["mode"] = "content-refusal"
        debug_payload["refusal_reason"] = "retrieval_below_threshold"
        return {
            "answer": REFUSAL_MESSAGE,
            "sources": retrieved[:3],
            "debug": debug_payload,
            "confidence": 0.0,
        }

    if remote_llm_configured():
        answer, mode_debug = answer_with_remote_llm(query, retrieved[:4])
    else:
        answer, mode_debug = synthesize_local_answer(query, retrieved[:4])

    debug_payload.update(mode_debug)
    return {
        "answer": answer,
        "sources": retrieved[:4],
        "debug": debug_payload,
        "confidence": calculate_confidence(retrieved[:4], debug_payload),
    }


def process_uploaded_documents(uploaded_files: list[Any]) -> None:
    fingerprints = fingerprint_uploaded_files(uploaded_files)
    if fingerprints == st.session_state.document_fingerprints:
        return

    documents: list[dict[str, Any]] = []
    body_chunks: list[dict[str, Any]] = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        digest = hashlib.md5(file_bytes).hexdigest()[:10]
        file_path = UPLOAD_DIR / f"{digest}_{uploaded_file.name}"
        with open(file_path, "wb") as target_file:
            target_file.write(file_bytes)

        raw_text = extract_text_from_file(str(file_path))
        metadata = extract_document_metadata(raw_text)
        removed_metadata_lines, body_text = split_document_content(raw_text, metadata)
        body_lines = [line.strip() for line in body_text.splitlines() if line.strip()]

        document = {
            "name": uploaded_file.name,
            "stored_path": str(file_path),
            "text": raw_text,
            "metadata": metadata,
            "body_text": body_text,
            "body_lines": body_lines,
            "removed_metadata_lines": removed_metadata_lines,
            "preview": raw_text[:1400],
        }
        documents.append(document)

        for chunk_index, chunk_value in enumerate(chunk_text(body_text)):
            body_chunks.append(
                {
                    "doc_name": uploaded_file.name,
                    "chunk_id": chunk_index,
                    "text": chunk_value,
                    "lines": [line.strip() for line in chunk_value.splitlines() if line.strip()],
                    "search_text": chunk_value,
                }
            )

    vectorizer, chunk_matrix = build_index(body_chunks)
    st.session_state.documents = documents
    st.session_state.body_chunks = body_chunks
    st.session_state.vectorizer = vectorizer
    st.session_state.chunk_matrix = chunk_matrix
    st.session_state.indexed_files = [document["name"] for document in documents]
    st.session_state.document_fingerprints = fingerprints
    st.session_state.retrieved_sources = []
    st.session_state.last_debug = {}
    st.session_state.last_prompt = ""
    st.session_state.answer_history = []


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <h1>Document Intelligence RAG Assistant</h1>
            <p>
                Upload PDF or DOCX files once, keep them indexed in session, and ask grounded
                follow-up questions with metadata-aware QA, body-only retrieval, and guarded local fallback.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stats() -> None:
    total_docs = len(st.session_state.documents)
    total_chunks = len(st.session_state.body_chunks)
    total_answers = len(st.session_state.answer_history)
    mode_label = "Remote LLM" if remote_llm_configured() else "Local Extractive"

    columns = st.columns(4)
    stats = [
        ("Indexed Files", str(total_docs)),
        ("Body Chunks", str(total_chunks)),
        ("Answers This Session", str(total_answers)),
        ("Active Mode", mode_label),
    ]
    for column, (label, value) in zip(columns, stats):
        with column:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">{label}</div>
                    <div class="stat-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_sidebar() -> None:
    llm_settings = get_llm_settings()
    with st.sidebar:
        st.header("Session Controls")
        st.checkbox("Debug mode", key="debug_mode")
        if st.button("Clear chat", use_container_width=True):
            clear_chat()
            st.rerun()
        if st.button("Reset documents", use_container_width=True):
            reset_documents()
            st.rerun()

        st.markdown("---")
        st.subheader("Runtime")
        if remote_llm_configured():
            st.markdown(
                f"<span class='mode-pill'>Remote LLM</span><span class='mode-pill'>{llm_settings['display_host']}</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"Model: {llm_settings['model']}")
        else:
            st.markdown("<span class='mode-pill'>Local Extractive</span>", unsafe_allow_html=True)
            st.caption("No public LLM endpoint configured.")

        st.markdown("---")
        st.subheader("Indexed Files")
        if st.session_state.indexed_files:
            for name in st.session_state.indexed_files:
                st.write(f"- {name}")
        else:
            st.caption("No documents indexed yet.")


def render_debug_panels() -> None:
    if not st.session_state.debug_mode:
        return

    st.subheader("Debug Information")

    with st.expander("Extracted raw text preview", expanded=False):
        for document in st.session_state.documents:
            st.markdown(f"**{document['name']}**")
            st.text(document["preview"] or "(empty)")

    with st.expander("Extracted metadata", expanded=False):
        st.json({document["name"]: document["metadata"] for document in st.session_state.documents})

    with st.expander("Body text preview", expanded=False):
        for document in st.session_state.documents:
            st.markdown(f"**{document['name']}**")
            st.text(document["body_text"][:1400] or "(empty body text)")

    with st.expander("Removed metadata lines", expanded=False):
        st.json({document["name"]: document["removed_metadata_lines"] for document in st.session_state.documents})

    with st.expander("Body chunk list", expanded=False):
        st.write(
            [
                {
                    "doc_name": chunk["doc_name"],
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                }
                for chunk in st.session_state.body_chunks
            ]
        )

    if st.session_state.last_debug:
        with st.expander("Last answer debug", expanded=True):
            st.json(st.session_state.last_debug)


def render_suggested_questions() -> None:
    if not st.session_state.documents:
        return

    st.caption("Quick prompts")
    suggestions = [
        "Who is the author of the document?",
        "What is one use of AI in healthcare mentioned in the document?",
        "What are the challenges of using AI in healthcare?",
    ]
    columns = st.columns(len(suggestions))
    for column, suggestion in zip(columns, suggestions):
        with column:
            if st.button(suggestion, use_container_width=True):
                result = answer_question(suggestion)
                st.session_state.current_question = suggestion
                st.session_state.retrieved_sources = result["sources"]
                st.session_state.last_debug = result["debug"]
                st.session_state.last_prompt = result["debug"].get("prompt", "")
                st.session_state.answer_history.append(
                    {
                        "question": suggestion,
                        "answer": result["answer"],
                        "sources": result["sources"],
                        "debug": result["debug"],
                        "confidence": result["confidence"],
                    }
                )
                st.rerun()


def render_history() -> None:
    for index, item in enumerate(st.session_state.answer_history, start=1):
        with st.chat_message("user"):
            st.write(item["question"])

        with st.chat_message("assistant"):
            st.write(item["answer"])
            confidence = item.get("confidence", 0.0)
            mode = item["debug"].get("mode", item["debug"].get("question_type", "unknown"))
            st.caption(f"Mode: {mode} | Confidence: {confidence:.2f}")

            with st.expander("Sources", expanded=False):
                if not item["sources"]:
                    st.write("No sources shown for metadata-only answer.")
                else:
                    for source_index, source in enumerate(item["sources"], start=1):
                        st.markdown(
                            f"**Source {source_index}** | `{source['doc_name']}` | "
                            f"similarity={source['similarity']}, overlap={source['overlap']}, "
                            f"combined={source['combined_score']}"
                        )
                        st.write(source["text"])

            if st.session_state.debug_mode:
                with st.expander(f"Answer debug #{index}", expanded=False):
                    st.json(item["debug"])


def handle_new_question(question: str) -> None:
    result = answer_question(question)
    st.session_state.current_question = question
    st.session_state.retrieved_sources = result["sources"]
    st.session_state.last_debug = result["debug"]
    st.session_state.last_prompt = result["debug"].get("prompt", "")
    st.session_state.answer_history.append(
        {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "debug": result["debug"],
            "confidence": result["confidence"],
        }
    )


st.set_page_config(page_title="Document Intelligence RAG Assistant", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)
init_session_state()
render_sidebar()
render_hero()
render_stats()

upload_col, info_col = st.columns([1.2, 1])
with upload_col:
    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )
with info_col:
    if remote_llm_configured():
        llm_settings = get_llm_settings()
        st.info(
            "Remote answer generation is enabled. "
            f"Requests will be sent to `{llm_settings['display_host']}` using `{llm_settings['model']}`."
        )
    else:
        st.info("Local production mode is enabled with metadata-aware QA and guarded body-content retrieval.")

if uploaded_files:
    if st.button("Process Documents", use_container_width=True):
        with st.spinner("Extracting metadata, separating body content, and building the retrieval index..."):
            process_uploaded_documents(uploaded_files)
        st.success(
            f"Indexed {len(st.session_state.body_chunks)} body chunks from "
            f"{len(st.session_state.documents)} file(s)."
        )
else:
    st.caption("Upload files and click `Process Documents` to start a new indexed session.")

st.subheader("Ask a Question")
render_suggested_questions()

prompt = st.chat_input("Ask about the indexed documents...", disabled=not bool(st.session_state.documents))
if prompt:
    handle_new_question(prompt)

if not st.session_state.documents:
    st.caption("Upload and process documents to start asking questions.")
else:
    render_history()
    render_debug_panels()
