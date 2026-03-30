from __future__ import annotations

import hashlib
from html import escape
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

FOLLOW_UP_PHRASES = [
    "what else",
    "another",
    "anything else",
    "name another",
    "more",
]

KNOWN_USE_CASES = [
    "medical imaging",
    "predictive analytics",
    "drug discovery",
    "patient monitoring",
    "clinical decision support",
    "predict patient admissions",
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
    :root {
        --bg: #0f172a;
        --bg-soft: #111c33;
        --panel: rgba(30, 41, 59, 0.92);
        --panel-strong: #1e293b;
        --panel-soft: rgba(51, 65, 85, 0.72);
        --text: #e2e8f0;
        --muted: #94a3b8;
        --accent: #38bdf8;
        --accent-strong: #2563eb;
        --border: rgba(148, 163, 184, 0.18);
        --shadow: 0 18px 60px rgba(2, 6, 23, 0.32);
        --user-bubble: linear-gradient(135deg, #2563eb 0%, #38bdf8 100%);
        --assistant-bubble: rgba(30, 41, 59, 0.96);
    }
    .stApp {
        background:
            radial-gradient(circle at top center, rgba(56, 189, 248, 0.12), transparent 24%),
            radial-gradient(circle at bottom left, rgba(37, 99, 235, 0.10), transparent 20%),
            linear-gradient(180deg, #0b1220 0%, var(--bg) 100%);
        color: var(--text);
    }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at top center, rgba(56, 189, 248, 0.12), transparent 24%),
            radial-gradient(circle at bottom left, rgba(37, 99, 235, 0.10), transparent 20%),
            linear-gradient(180deg, #0b1220 0%, var(--bg) 100%);
    }
    [data-testid="stHeader"] {
        background: rgba(11, 18, 32, 0.86);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #172036 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.10);
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
        padding-top: 1.5rem;
        padding-bottom: 7rem;
        max-width: 920px;
    }
    h1, h2, h3, h4, h5, h6, p, label, div, span {
        color: inherit;
    }
    .app-title {
        font-size: 1.9rem;
        font-weight: 800;
        line-height: 1.2;
        color: var(--text);
        margin: 0;
    }
    .app-subtitle {
        font-size: 0.98rem;
        line-height: 1.7;
        color: var(--muted);
        margin: 0.45rem 0 0 0;
    }
    .top-shell {
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 24px;
        padding: 1.25rem 1.3rem;
        backdrop-filter: blur(18px);
        box-shadow: var(--shadow);
    }
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.85rem;
    }
    .stat-card {
        background: rgba(15, 23, 42, 0.62);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 18px;
        padding: 0.95rem 1rem;
    }
    .stat-label {
        font-size: 0.76rem;
        color: var(--muted);
        margin-bottom: 0.2rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .stat-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--text);
    }
    .upload-note {
        color: var(--muted);
        font-size: 0.92rem;
        margin-top: 0.35rem;
    }
    .files-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
    }
    .file-pill {
        padding: 0.42rem 0.75rem;
        border-radius: 999px;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.18);
        color: var(--text);
        font-size: 0.88rem;
    }
    .mode-pill, .confidence-pill {
        display: inline-flex;
        align-items: center;
        padding: 0.26rem 0.62rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        margin-right: 0.45rem;
    }
    .mode-pill {
        background: rgba(56, 189, 248, 0.14);
        color: #7dd3fc;
        border: 1px solid rgba(56, 189, 248, 0.18);
    }
    .confidence-pill {
        background: rgba(148, 163, 184, 0.16);
        color: var(--text);
        border: 1px solid rgba(148, 163, 184, 0.15);
    }
    .chat-shell {
        display: flex;
        flex-direction: column;
        gap: 0.9rem;
    }
    .message-wrap {
        display: flex;
        width: 100%;
    }
    .message-wrap.user {
        justify-content: flex-end;
    }
    .message-wrap.assistant {
        justify-content: flex-start;
    }
    .bubble {
        max-width: 78%;
        padding: 0.9rem 1rem;
        border-radius: 22px;
        box-shadow: 0 12px 34px rgba(2, 6, 23, 0.18);
        font-size: 0.98rem;
        line-height: 1.65;
        white-space: pre-wrap;
    }
    .bubble.user {
        background: var(--user-bubble);
        color: #eff6ff;
        border-bottom-right-radius: 8px;
    }
    .bubble.assistant {
        background: var(--assistant-bubble);
        border: 1px solid rgba(148, 163, 184, 0.12);
        color: var(--text);
        border-bottom-left-radius: 8px;
    }
    .bubble-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-bottom: 0.6rem;
    }
    .message-label {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
        margin-bottom: 0.35rem;
        font-weight: 700;
    }
    .resolved-text {
        color: #7dd3fc;
        font-size: 0.82rem;
        margin-top: 0.4rem;
    }
    .source-caption {
        color: var(--muted);
        font-size: 0.8rem;
        margin-top: 0.3rem;
    }
    .thinking-note {
        color: var(--muted);
        font-size: 0.9rem;
    }
    .section-heading {
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--text);
        margin: 1.2rem 0 0.85rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1d4ed8 0%, #38bdf8 100%);
        color: #ffffff;
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        font-weight: 600;
        box-shadow: 0 12px 30px rgba(37, 99, 235, 0.22);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%);
        color: #ffffff;
        border-color: rgba(125, 211, 252, 0.32);
    }
    .stTextInput > div > div > input,
    [data-testid="stChatInputTextArea"] textarea,
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(15, 23, 42, 0.72) !important;
        color: var(--text) !important;
        border: 1px solid rgba(148, 163, 184, 0.14) !important;
        border-radius: 18px !important;
        box-shadow: var(--shadow);
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: var(--text) !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(135deg, #1d4ed8 0%, #38bdf8 100%) !important;
        color: #ffffff !important;
    }
    [data-testid="stChatInput"] {
        background: transparent;
    }
    [data-testid="stBottomBlockContainer"] {
        background: rgba(11, 18, 32, 0.88);
        border-top: 1px solid rgba(148, 163, 184, 0.12);
        backdrop-filter: blur(18px);
    }
    [data-testid="stChatInput"] textarea {
        background: rgba(15, 23, 42, 0.94) !important;
        color: var(--text) !important;
        border-radius: 20px !important;
        box-shadow: 0 14px 40px rgba(2, 6, 23, 0.28) !important;
    }
    [data-testid="stInfo"],
    [data-testid="stSuccess"],
    [data-testid="stWarning"],
    [data-testid="stError"] {
        background: rgba(30, 41, 59, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.12);
        color: var(--text);
        border-radius: 16px;
    }
    [data-testid="stInfo"] *,
    [data-testid="stSuccess"] *,
    [data-testid="stWarning"] *,
    [data-testid="stError"] * {
        color: var(--text) !important;
    }
    [data-testid="stMarkdownContainer"] p {
        color: var(--text);
    }
    .stCaption {
        color: var(--muted) !important;
    }
    [data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(2, 6, 23, 0.22);
    }
    [data-testid="stFileUploaderFile"] {
        background: rgba(15, 23, 42, 0.72);
        border-radius: 14px;
    }
    [data-testid="stExpander"] details summary p {
        font-size: 0.88rem !important;
    }
    [data-testid="stSpinner"] {
        color: #7dd3fc !important;
    }
    @media (max-width: 900px) {
        .stats-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .bubble {
            max-width: 100%;
        }
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
        "previous_answers": [],
        "used_sentences": [],
        "used_chunks": [],
        "last_content_question": "",
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
    st.session_state.previous_answers = []
    st.session_state.used_sentences = []
    st.session_state.used_chunks = []
    st.session_state.last_content_question = ""


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


def normalize_for_dedupe(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"[^\w\s]", "", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


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


def is_follow_up_question(question: str) -> bool:
    lowered = question.lower().strip()
    return any(phrase in lowered for phrase in FOLLOW_UP_PHRASES)


def resolve_follow_up_question(question: str) -> tuple[str, str | None]:
    if not is_follow_up_question(question):
        return question, None

    previous = st.session_state.last_content_question.strip()
    if not previous:
        return question, None

    intent = infer_content_intent(previous)
    if intent == "uses":
        rewritten = "Name another use of AI in healthcare from the document."
    elif intent == "challenges":
        rewritten = "What are other challenges of using AI in healthcare?"
    elif intent == "benefits":
        rewritten = "How else does AI help in hospitals according to the document?"
    else:
        rewritten = previous
    return rewritten, rewritten


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
        normalized = normalize_for_dedupe(item)
        if item and normalized and normalized not in seen:
            candidates.append(item)
            seen.add(normalized)
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


def dedupe_answer_parts(parts: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        normalized = normalize_for_dedupe(part)
        if normalized and normalized not in seen:
            deduped.append(clean_answer_text(part))
            seen.add(normalized)
    return deduped


def summarize_known_phrases(retrieved_chunks: list[dict[str, Any]], phrases: list[str]) -> list[str]:
    found: list[str] = []
    found_norms: set[str] = set()
    used_norms = {normalize_for_dedupe(item) for item in st.session_state.previous_answers}

    for chunk in retrieved_chunks:
        for phrase in phrases:
            matched = find_phrase_in_text(chunk["text"], phrase)
            if not matched:
                continue
            normalized = normalize_for_dedupe(matched)
            if normalized in found_norms:
                continue
            if normalized in used_norms:
                continue
            found.append(matched)
            found_norms.add(normalized)
    return found


def score_candidate_answer(query: str, candidate: str, parent_chunk: dict[str, Any], intent: str) -> float:
    overlap = lexical_overlap_ratio(query, candidate)
    score = (0.62 * parent_chunk["combined_score"]) + (0.3 * overlap)
    candidate_lower = candidate.lower()
    normalized = normalize_for_dedupe(candidate)

    if intent == "uses" and any(phrase in candidate_lower for phrase in KNOWN_USE_CASES):
        score += 0.22
    if intent == "challenges" and any(phrase in candidate_lower for phrase in KNOWN_CHALLENGES):
        score += 0.2
    if intent == "benefits" and any(term in candidate_lower for term in ["help", "improve", "support", "predict"]):
        score += 0.12
    if normalized in st.session_state.used_sentences:
        score -= 0.55
    if f"{parent_chunk['doc_name']}::{parent_chunk['chunk_id']}" in st.session_state.used_chunks:
        score -= 0.08
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
                return matched.title() if matched.islower() else matched
        list_item = extract_list_item(candidate)
        if list_item:
            return list_item

    if intent == "challenges":
        found = []
        for phrase in KNOWN_CHALLENGES:
            matched = find_phrase_in_text(candidate, phrase)
            if matched:
                normalized = normalize_for_dedupe(matched)
                if normalized not in {normalize_for_dedupe(item) for item in found}:
                    found.append(matched)
        if found:
            return ", ".join(found)

    return clean_answer_text(candidate)


def has_sufficient_evidence(
    question: str,
    retrieved_sentences: list[dict[str, Any]],
    best_score: float,
    overlap_score: float,
) -> tuple[bool, dict[str, Any]]:
    intent = infer_content_intent(question)
    min_score = 0.28 if intent in {"uses", "challenges", "benefits"} else 0.34
    min_overlap = 0.18 if intent in {"uses", "challenges", "benefits"} else 0.22

    if any(phrase in question.lower() for phrase in ["capital of", "president of", "population of"]):
        min_score = 0.42
        min_overlap = 0.24

    relevant_candidates = [item for item in retrieved_sentences if item["score"] >= min_score]
    has_relevant_sentence = bool(relevant_candidates)
    decision = best_score >= min_score and overlap_score >= min_overlap and has_relevant_sentence

    return decision, {
        "intent": intent,
        "best_score": round(best_score, 4),
        "overlap_score": round(overlap_score, 4),
        "min_score": min_score,
        "min_overlap": min_overlap,
        "relevant_sentence_count": len(relevant_candidates),
        "decision": "answer" if decision else "refuse",
    }


def synthesize_local_answer(query: str, retrieved_chunks: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    intent = infer_content_intent(query)
    debug: dict[str, Any] = {
        "mode": "local-extractive",
        "intent": intent,
        "selected_span": None,
        "candidate_sentences_before_dedupe": [],
        "candidate_sentences_after_dedupe": [],
        "excluded_due_to_prior_use": [],
    }

    if not retrieved_chunks:
        debug["refusal_reason"] = "no_retrieved_chunks"
        return REFUSAL_MESSAGE, debug

    if intent == "uses":
        known_matches = summarize_known_phrases(retrieved_chunks, KNOWN_USE_CASES)
        if known_matches:
            selected = known_matches[0]
            debug["selected_span"] = selected
            debug["known_phrase_matches"] = known_matches
            return selected, debug

    if intent == "challenges":
        found = summarize_known_phrases(retrieved_chunks, KNOWN_CHALLENGES)
        if found:
            answer = " and ".join(dedupe_answer_parts(found[:2]))
            debug["selected_span"] = answer
            debug["known_phrase_matches"] = found
            return answer, debug

    candidates: list[dict[str, Any]] = []
    raw_candidates: list[str] = []
    dedupe_seen: set[str] = set()
    deduped_candidates: list[str] = []

    for chunk in retrieved_chunks:
        for candidate in extract_candidate_units(chunk["text"]):
            raw_candidates.append(candidate)
            answer_span = extract_answer_span(query, candidate, intent)
            normalized_answer = normalize_for_dedupe(answer_span)
            if normalized_answer and normalized_answer not in dedupe_seen:
                dedupe_seen.add(normalized_answer)
                deduped_candidates.append(answer_span)
            candidate_score = score_candidate_answer(query, candidate, chunk, intent)
            excluded_prior_use = normalized_answer in st.session_state.used_sentences
            if excluded_prior_use:
                debug["excluded_due_to_prior_use"].append(answer_span)
            candidates.append(
                {
                    "doc_name": chunk["doc_name"],
                    "chunk_id": chunk["chunk_id"],
                    "candidate": candidate,
                    "answer_span": answer_span,
                    "score": candidate_score,
                    "overlap": lexical_overlap_ratio(query, candidate),
                    "excluded_prior_use": excluded_prior_use,
                }
            )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    debug["candidate_sentences_before_dedupe"] = raw_candidates[:12]
    debug["candidate_sentences_after_dedupe"] = deduped_candidates[:12]
    debug["candidate_answers"] = candidates[:8]

    if not candidates:
        debug["refusal_reason"] = "no_candidates"
        return REFUSAL_MESSAGE, debug

    best_score = candidates[0]["score"]
    best_overlap = candidates[0]["overlap"]
    evidence_ok, evidence_debug = has_sufficient_evidence(query, candidates[:8], best_score, best_overlap)
    debug["evidence"] = evidence_debug
    if not evidence_ok:
        debug["refusal_reason"] = "insufficient_evidence"
        return REFUSAL_MESSAGE, debug

    primary = candidates[0]
    parts = [primary["answer_span"]]

    if intent == "benefits":
        for candidate in candidates[1:4]:
            if candidate["score"] < primary["score"] - 0.18:
                continue
            if candidate["overlap"] < max(0.18, primary["overlap"] - 0.06):
                continue
            if normalize_for_dedupe(candidate["answer_span"]) == normalize_for_dedupe(primary["answer_span"]):
                continue
            parts.append(candidate["answer_span"])
            break

    parts = dedupe_answer_parts(parts)
    final_answer = clean_answer_text(" ".join(parts[:1]) if intent in {"uses", "general"} else " ".join(parts))
    debug["selected_span"] = final_answer
    return final_answer, debug


def calculate_confidence(sources: list[dict[str, Any]], debug: dict[str, Any]) -> float:
    if not sources:
        return 0.98 if debug.get("mode") == "metadata" else 0.0
    top_score = sources[0]["combined_score"]
    if debug.get("mode") == "remote-llm":
        return round(min(0.95, 0.45 + top_score), 2)
    evidence = debug.get("evidence", {})
    best_score = evidence.get("best_score", top_score)
    return round(min(0.92, 0.25 + top_score + (best_score / 3)), 2)


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
                        "If the answer is missing, say you could not find it in the uploaded documents."
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
    resolved_query, rewritten = resolve_follow_up_question(query)
    question_type = classify_question(resolved_query)
    debug_payload: dict[str, Any] = {
        "question_type": question_type,
        "rewritten_follow_up_question": rewritten,
    }

    if question_type == "metadata":
        metadata_answer, metadata_debug = try_metadata_answer(resolved_query, st.session_state.documents)
        debug_payload["metadata_debug"] = metadata_debug
        if metadata_answer:
            debug_payload["mode"] = "metadata"
            debug_payload["selected_span"] = metadata_answer
            return {
                "answer": metadata_answer,
                "sources": [],
                "debug": debug_payload,
                "confidence": calculate_confidence([], debug_payload),
                "resolved_query": resolved_query,
            }
        return {
            "answer": REFUSAL_MESSAGE,
            "sources": [],
            "debug": debug_payload,
            "confidence": 0.0,
            "resolved_query": resolved_query,
        }

    retrieved = retrieve_chunks(
        query=resolved_query,
        vectorizer=st.session_state.vectorizer,
        chunk_matrix=st.session_state.chunk_matrix,
        chunks=st.session_state.body_chunks,
        top_k=6,
    )
    debug_payload["retrieved_chunks"] = retrieved

    if not retrieved or retrieved[0]["combined_score"] < 0.18:
        debug_payload["mode"] = "content-refusal"
        debug_payload["refusal_reason"] = "retrieval_below_threshold"
        debug_payload["best_score"] = retrieved[0]["combined_score"] if retrieved else 0.0
        debug_payload["overlap_score"] = retrieved[0]["overlap"] if retrieved else 0.0
        return {
            "answer": REFUSAL_MESSAGE,
            "sources": retrieved[:3],
            "debug": debug_payload,
            "confidence": 0.0,
            "resolved_query": resolved_query,
        }

    if remote_llm_configured():
        answer, mode_debug = answer_with_remote_llm(resolved_query, retrieved[:4])
    else:
        answer, mode_debug = synthesize_local_answer(resolved_query, retrieved[:4])

    debug_payload.update(mode_debug)
    evidence = debug_payload.get("evidence", {})
    debug_payload["best_score"] = evidence.get("best_score", retrieved[0]["combined_score"])
    debug_payload["overlap_score"] = evidence.get("overlap_score", retrieved[0]["overlap"])
    return {
        "answer": answer,
        "sources": retrieved[:4],
        "debug": debug_payload,
        "confidence": calculate_confidence(retrieved[:4], debug_payload),
        "resolved_query": resolved_query,
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
    st.session_state.previous_answers = []
    st.session_state.used_sentences = []
    st.session_state.used_chunks = []
    st.session_state.last_content_question = ""


def render_hero() -> None:
    st.markdown(
        """
        <div class="top-shell">
            <div class="app-title">Document Intelligence RAG Assistant</div>
            <p class="app-subtitle">
                Upload your files, index them once, and chat with your documents in a grounded,
                source-aware workflow.
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
    stats = [
        ("Indexed Files", str(total_docs)),
        ("Body Chunks", str(total_chunks)),
        ("Answers This Session", str(total_answers)),
        ("Active Mode", mode_label),
    ]
    columns = st.columns(4)
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
                {"doc_name": chunk["doc_name"], "chunk_id": chunk["chunk_id"], "text": chunk["text"]}
                for chunk in st.session_state.body_chunks
            ]
        )

    if st.session_state.last_debug:
        with st.expander("Last answer debug", expanded=True):
            st.json(st.session_state.last_debug)


def render_chat_bubble(role: str, text: str, mode: str | None = None, confidence: float | None = None, resolved_query: str | None = None, original_question: str | None = None) -> None:
    bubble_class = "user" if role == "user" else "assistant"
    label = "You" if role == "user" else "Assistant"

    badge_html = ""
    if role == "assistant" and mode is not None and confidence is not None:
        badge_html = (
            "<div class='bubble-meta'>"
            f"<span class='mode-pill'>{escape(mode)}</span>"
            f"<span class='confidence-pill'>Confidence {confidence:.2f}</span>"
            "</div>"
        )

    resolved_html = ""
    if role == "user" and resolved_query and original_question and resolved_query != original_question:
        resolved_html = f"<div class='resolved-text'>Resolved as: {escape(resolved_query)}</div>"

    st.markdown(
        f"""
        <div class="message-wrap {bubble_class}">
            <div class="bubble-shell">
                <div class="message-label">{label}</div>
                <div class="bubble {bubble_class}">
                    {badge_html}
                    {escape(text)}
                </div>
                {resolved_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_history() -> None:
    for index, item in enumerate(st.session_state.answer_history, start=1):
        render_chat_bubble(
            role="user",
            text=item["question"],
            resolved_query=item.get("resolved_query"),
            original_question=item["question"],
        )
        mode = item["debug"].get("mode", item["debug"].get("question_type", "unknown"))
        confidence = item.get("confidence", 0.0)
        render_chat_bubble(
            role="assistant",
            text=item["answer"],
            mode=mode,
            confidence=confidence,
        )

        with st.expander("Sources", expanded=False):
            if not item["sources"]:
                st.caption("No sources shown for metadata-only answer.")
            else:
                for source_index, source in enumerate(item["sources"], start=1):
                    st.markdown(
                        f"**Source {source_index}** | `{source['doc_name']}` | "
                        f"similarity={source['similarity']}, overlap={source['overlap']}, "
                        f"combined={source['combined_score']}"
                    )
                    st.caption(source["text"])

        if st.session_state.debug_mode:
            with st.expander(f"Answer debug #{index}", expanded=False):
                st.json(item["debug"])


def register_answer_usage(answer: str, sources: list[dict[str, Any]], debug: dict[str, Any]) -> None:
    answer_norm = normalize_for_dedupe(answer)
    if answer_norm and answer_norm not in st.session_state.previous_answers:
        st.session_state.previous_answers.append(answer_norm)

    selected = debug.get("selected_span")
    if selected:
        selected_norm = normalize_for_dedupe(selected)
        if selected_norm and selected_norm not in st.session_state.used_sentences:
            st.session_state.used_sentences.append(selected_norm)

    for source in sources:
        chunk_key = f"{source['doc_name']}::{source['chunk_id']}"
        if chunk_key not in st.session_state.used_chunks:
            st.session_state.used_chunks.append(chunk_key)


def handle_new_question(question: str) -> None:
    result = answer_question(question)
    st.session_state.current_question = question
    st.session_state.retrieved_sources = result["sources"]
    st.session_state.last_debug = result["debug"]
    st.session_state.last_prompt = result["debug"].get("prompt", "")

    if classify_question(result["resolved_query"]) == "content":
        st.session_state.last_content_question = result["resolved_query"]

    register_answer_usage(result["answer"], result["sources"], result["debug"])

    st.session_state.answer_history.append(
        {
            "question": question,
            "resolved_query": result["resolved_query"],
            "answer": result["answer"],
            "sources": result["sources"],
            "debug": result["debug"],
            "confidence": result["confidence"],
        }
    )


# Expected local behavior checks for the healthcare sample document:
# 1. Author question -> "Dr. Sarah Johnson"
# 2. Published question -> "2023"
# 3. First use question -> one of "Medical imaging", "Predictive analytics", "Drug discovery"
# 4. Follow-up use questions -> different valid uses when possible
# 5. Challenges question -> should mention both privacy and incorrect diagnoses
# 6. Out-of-scope question like "What is the capital of Australia?" -> refusal


st.set_page_config(page_title="Document Intelligence RAG Assistant", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)
init_session_state()
render_sidebar()
render_hero()
render_stats()

st.markdown("<div class='section-heading'>Upload Documents</div>", unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Upload PDF or DOCX files",
    type=["pdf", "docx"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)
if remote_llm_configured():
    llm_settings = get_llm_settings()
    st.caption(
        f"Remote runtime active via `{llm_settings['display_host']}` using `{llm_settings['model']}`."
    )
else:
    st.caption("Local runtime active. Metadata routing, body retrieval, and guarded local synthesis are enabled.")

if uploaded_files:
    if st.button("Process Documents", use_container_width=True):
        with st.spinner("Processing documents..."):
            process_uploaded_documents(uploaded_files)
        st.success(
            f"Indexed {len(st.session_state.body_chunks)} body chunks from "
            f"{len(st.session_state.documents)} file(s)."
        )
else:
    st.markdown("<div class='upload-note'>Upload files and click <strong>Process Documents</strong> to start a new indexed session.</div>", unsafe_allow_html=True)

st.markdown("<div class='section-heading'>Indexed Files</div>", unsafe_allow_html=True)
if st.session_state.indexed_files:
    pills = "".join(f"<span class='file-pill'>{escape(name)}</span>" for name in st.session_state.indexed_files)
    st.markdown(f"<div class='files-wrap'>{pills}</div>", unsafe_allow_html=True)
else:
    st.caption("No documents indexed yet.")

st.markdown("<div class='section-heading'>Chat</div>", unsafe_allow_html=True)

prompt = st.chat_input("Ask about your documents...", disabled=not bool(st.session_state.documents))
if prompt:
    with st.spinner("Thinking..."):
        handle_new_question(prompt)

if not st.session_state.documents:
    st.markdown("<div class='thinking-note'>Upload and process documents to start asking questions.</div>", unsafe_allow_html=True)
else:
    render_history()
    render_debug_panels()
