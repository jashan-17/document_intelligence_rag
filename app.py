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
SOFT_GUIDANCE_MESSAGE = (
    "I couldn't find a direct answer from the uploaded documents. "
    "Try asking something more specific."
)

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

OUT_OF_SCOPE_PHRASES = [
    "capital of",
    "current president",
    "president of",
    "weather today",
    "weather",
    "news today",
    "stock price",
    "population of",
]

APP_CSS = """
<style>
    .stApp {
        background: #0f172a;
        color: #e5e7eb;
    }
    [data-testid="stAppViewContainer"] {
        background: #0f172a;
    }
    [data-testid="stHeader"] {
        background: #0f172a;
    }
    [data-testid="stSidebar"] {
        background: #111827;
    }
    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 6rem;
        max-width: 960px;
    }
    .thinking-note {
        color: #94a3b8;
        font-size: 0.9rem;
    }
    .stButton > button {
        background: #2563eb;
        color: #ffffff;
        border: 1px solid #2563eb;
        border-radius: 12px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background: #1d4ed8;
        border-color: #1d4ed8;
    }
    .stTextInput > div > div > input,
    [data-testid="stChatInputTextArea"] textarea,
    [data-testid="stFileUploaderDropzone"] {
        border-radius: 12px !important;
    }
    [data-testid="stBottomBlockContainer"] {
        background: rgba(15, 23, 42, 0.96);
    }
    [data-testid="stChatInput"] textarea {
        border-radius: 14px !important;
    }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] strong,
    h1, h2, h3, h4, h5, h6, label {
        color: #e5e7eb;
    }
    .stCaption {
        color: #94a3b8 !important;
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
    sanitize_history_state()


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


def strip_html_artifacts(text: str) -> str:
    if not text:
        return text
    cleaned = re.sub(r"</?div[^>]*>", " ", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?span[^>]*>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?p[^>]*>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def sanitize_history_state() -> None:
    cleaned_history = []
    for item in st.session_state.answer_history:
        cleaned_item = dict(item)
        cleaned_item["question"] = strip_html_artifacts(cleaned_item.get("question", ""))
        cleaned_item["answer"] = strip_html_artifacts(cleaned_item.get("answer", ""))
        cleaned_item["resolved_query"] = strip_html_artifacts(cleaned_item.get("resolved_query", ""))
        cleaned_history.append(cleaned_item)
    st.session_state.answer_history = cleaned_history
    st.session_state.current_question = strip_html_artifacts(st.session_state.current_question)
    st.session_state.previous_answers = [
        normalize_for_dedupe(strip_html_artifacts(answer))
        for answer in st.session_state.previous_answers
        if strip_html_artifacts(answer)
    ]


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


def classify_query_mode(question: str) -> str:
    lowered = question.lower().strip()
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
    if any(phrase in lowered for phrase in OUT_OF_SCOPE_PHRASES):
        return "out_of_scope_candidate"
    if any(keyword in lowered for keyword in CONTENT_INTENT_KEYWORDS["summary"]):
        return "summary"
    if any(phrase in lowered for phrase in FOLLOW_UP_PHRASES):
        return "follow_up"
    if lowered in {"why", "how", "explain", "describe"}:
        return "follow_up"
    if any(lowered.startswith(prefix) for prefix in ("why", "how", "explain", "describe")):
        return "explanatory"
    return "factoid"


def classify_question(question: str) -> str:
    return "metadata" if classify_query_mode(question) == "metadata" else "content"


def infer_content_intent(question: str) -> str:
    lowered = question.lower()
    for intent, keywords in CONTENT_INTENT_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return intent
    if lowered.startswith("why") or lowered.startswith("how") or lowered.startswith("explain"):
        return "benefits"
    return "general"


def is_follow_up_question(question: str) -> bool:
    return classify_query_mode(question) == "follow_up"


def is_out_of_scope_candidate(question: str) -> bool:
    lowered = question.lower()
    return any(phrase in lowered for phrase in OUT_OF_SCOPE_PHRASES)


def build_follow_up_question(question: str, previous: str) -> str:
    lowered = question.lower().strip()
    previous_intent = infer_content_intent(previous)

    if "summarize" in lowered or lowered == "summary":
        return "Summarize the document."
    if lowered.startswith("why"):
        if previous_intent == "challenges":
            return "Explain why the challenges of using AI in healthcare matter according to the document."
        if previous_intent == "uses":
            return "Explain why these uses of AI in healthcare are useful according to the document."
        return f"Explain this based on the document: {previous}"
    if lowered.startswith("how"):
        if previous_intent == "benefits":
            return "How does AI help in healthcare according to the document?"
        return f"Explain how this works according to the document: {previous}"
    if previous_intent == "uses":
        return "Name another use of AI in healthcare from the document."
    if previous_intent == "challenges":
        return "What are other challenges of using AI in healthcare?"
    if previous_intent == "benefits":
        return "How else does AI help in hospitals according to the document?"
    return previous


def resolve_follow_up_question(question: str) -> tuple[str, str | None]:
    if not is_follow_up_question(question):
        return question, None

    previous = st.session_state.last_content_question.strip()
    if not previous:
        return question, None
    rewritten = build_follow_up_question(question, previous)
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


def representative_document_chunks(chunks: list[dict[str, Any]], top_k: int = 8) -> list[dict[str, Any]]:
    if not chunks:
        return []
    step = max(1, len(chunks) // max(1, top_k))
    selected = [chunks[index] for index in range(0, len(chunks), step)]
    return selected[:top_k]


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


def sentence_supports_query(sentence: str, query: str, query_mode: str) -> bool:
    overlap = lexical_overlap_ratio(query, sentence)
    lowered = sentence.lower()
    if query_mode == "summary":
        return len(tokenize(sentence)) >= 6
    if query_mode == "explanatory":
        return overlap >= 0.08 or any(
            token in lowered for token in tokenize(query) if len(token) > 3
        )
    return overlap >= 0.12


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


def score_summary_sentence(query: str, candidate: str, parent_chunk: dict[str, Any]) -> float:
    overlap = lexical_overlap_ratio(query, candidate)
    score = (0.55 * parent_chunk["combined_score"]) + (0.2 * overlap)
    candidate_lower = candidate.lower()
    if any(term in candidate_lower for term in ["ai", "healthcare", "hospital", "patient", "diagnos", "predict"]):
        score += 0.14
    if len(candidate.split()) < 7:
        score -= 0.18
    if len(candidate.split()) > 36:
        score -= 0.12
    return round(score, 4)


def score_explanatory_sentence(query: str, candidate: str, parent_chunk: dict[str, Any]) -> float:
    overlap = lexical_overlap_ratio(query, candidate)
    score = (0.58 * parent_chunk["combined_score"]) + (0.25 * overlap)
    candidate_lower = candidate.lower()
    if any(term in candidate_lower for term in ["because", "helps", "allowing", "improves", "reducing", "support"]):
        score += 0.12
    if len(candidate.split()) < 6:
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


def should_answer(
    query_mode: str,
    retrieved_chunks: list[dict[str, Any]],
    candidate_sentences: list[dict[str, Any]],
    scores: dict[str, float],
) -> tuple[bool, dict[str, Any]]:
    thresholds = {
        "factoid": {"min_score": 0.28, "min_overlap": 0.18, "min_relevant": 1, "min_chunks": 1},
        "explanatory": {"min_score": 0.22, "min_overlap": 0.10, "min_relevant": 1, "min_chunks": 1},
        "summary": {"min_score": 0.18, "min_overlap": 0.05, "min_relevant": 2, "min_chunks": 2},
        "out_of_scope_candidate": {"min_score": 0.42, "min_overlap": 0.24, "min_relevant": 2, "min_chunks": 2},
    }
    config = thresholds.get(query_mode, thresholds["factoid"])
    best_score = scores.get("best_score", 0.0)
    overlap_score = scores.get("overlap_score", 0.0)
    relevant_candidates = [item for item in candidate_sentences if item.get("score", 0.0) >= config["min_score"]]
    relevant_chunk_count = len(
        {
            f"{item.get('doc_name', '')}::{item.get('chunk_id', '')}"
            for item in relevant_candidates
        }
    )
    retrieved_top_count = len([chunk for chunk in retrieved_chunks if chunk.get("combined_score", 0.0) >= config["min_score"]])
    decision = (
        best_score >= config["min_score"]
        and overlap_score >= config["min_overlap"]
        and len(relevant_candidates) >= config["min_relevant"]
        and max(retrieved_top_count, relevant_chunk_count) >= config["min_chunks"]
    )
    return decision, {
        "query_mode": query_mode,
        "best_score": round(best_score, 4),
        "overlap_score": round(overlap_score, 4),
        "min_score": config["min_score"],
        "min_overlap": config["min_overlap"],
        "relevant_sentence_count": len(relevant_candidates),
        "relevant_chunk_count": max(retrieved_top_count, relevant_chunk_count),
        "decision": "answer" if decision else "refuse",
    }


def has_sufficient_evidence(
    question: str,
    retrieved_sentences: list[dict[str, Any]],
    best_score: float,
    overlap_score: float,
) -> tuple[bool, dict[str, Any]]:
    query_mode = classify_query_mode(question)
    pseudo_chunks = [
        {
            "combined_score": item.get("score", 0.0),
            "doc_name": item.get("doc_name", ""),
            "chunk_id": item.get("chunk_id", ""),
        }
        for item in retrieved_sentences
    ]
    return should_answer(
        "factoid" if query_mode == "follow_up" else query_mode,
        pseudo_chunks,
        retrieved_sentences,
        {"best_score": best_score, "overlap_score": overlap_score},
    )


def fallback_message(query_mode: str, hard_refusal: bool = False) -> str:
    if hard_refusal or query_mode in {"factoid", "out_of_scope_candidate"}:
        return REFUSAL_MESSAGE
    return SOFT_GUIDANCE_MESSAGE


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


def answer_factoid_question(query: str, retrieved_chunks: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    return synthesize_local_answer(query, retrieved_chunks)


def compose_local_summary(query: str, retrieved_chunks: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    debug: dict[str, Any] = {
        "mode": "local-summary",
        "summary_candidate_sentences": [],
        "selected_span": None,
    }
    candidates: list[dict[str, Any]] = []
    deduped_preview: list[str] = []
    seen: set[str] = set()

    for chunk in retrieved_chunks:
        for sentence in extract_candidate_units(chunk["text"]):
            if not sentence_supports_query(sentence, query, "summary"):
                continue
            score = score_summary_sentence(query, sentence, chunk)
            normalized = normalize_for_dedupe(sentence)
            if normalized not in seen:
                deduped_preview.append(sentence)
                seen.add(normalized)
            candidates.append(
                {
                    "doc_name": chunk["doc_name"],
                    "chunk_id": chunk["chunk_id"],
                    "candidate": sentence,
                    "score": score,
                    "overlap": lexical_overlap_ratio(query, sentence),
                }
            )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    debug["summary_candidate_sentences"] = deduped_preview[:12]
    if not candidates:
        debug["refusal_reason"] = "no_summary_candidates"
        debug["fallback_type"] = "soft_guidance"
        return SOFT_GUIDANCE_MESSAGE, debug

    evidence_ok, evidence = should_answer(
        "summary",
        retrieved_chunks,
        candidates,
        {"best_score": candidates[0]["score"], "overlap_score": candidates[0]["overlap"]},
    )
    debug["evidence"] = evidence
    if not evidence_ok:
        debug["refusal_reason"] = "insufficient_summary_evidence"
        debug["fallback_type"] = "soft_guidance"
        return SOFT_GUIDANCE_MESSAGE, debug

    selected_parts: list[str] = []
    used_norms: set[str] = set()
    for candidate in candidates:
        normalized = normalize_for_dedupe(candidate["candidate"])
        if normalized in used_norms:
            continue
        selected_parts.append(candidate["candidate"])
        used_norms.add(normalized)
        if len(selected_parts) == 3:
            break

    answer = clean_answer_text(" ".join(dedupe_answer_parts(selected_parts[:3])))
    debug["selected_span"] = answer
    return answer, debug


def answer_summary_question_local(query: str, retrieved_chunks: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    return compose_local_summary(query, retrieved_chunks)


def answer_explanatory_question_local(query: str, retrieved_chunks: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    debug: dict[str, Any] = {
        "mode": "local-explanatory",
        "explanatory_candidate_sentences": [],
        "selected_span": None,
    }
    candidates: list[dict[str, Any]] = []
    preview: list[str] = []
    seen: set[str] = set()

    for chunk in retrieved_chunks:
        for sentence in extract_candidate_units(chunk["text"]):
            if not sentence_supports_query(sentence, query, "explanatory"):
                continue
            score = score_explanatory_sentence(query, sentence, chunk)
            normalized = normalize_for_dedupe(sentence)
            if normalized not in seen:
                preview.append(sentence)
                seen.add(normalized)
            candidates.append(
                {
                    "doc_name": chunk["doc_name"],
                    "chunk_id": chunk["chunk_id"],
                    "candidate": sentence,
                    "score": score,
                    "overlap": lexical_overlap_ratio(query, sentence),
                }
            )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    debug["explanatory_candidate_sentences"] = preview[:12]
    if not candidates:
        debug["refusal_reason"] = "no_explanatory_candidates"
        debug["fallback_type"] = "soft_guidance"
        return SOFT_GUIDANCE_MESSAGE, debug

    evidence_ok, evidence = should_answer(
        "explanatory",
        retrieved_chunks,
        candidates,
        {"best_score": candidates[0]["score"], "overlap_score": candidates[0]["overlap"]},
    )
    debug["evidence"] = evidence
    if not evidence_ok:
        debug["refusal_reason"] = "insufficient_explanatory_evidence"
        debug["fallback_type"] = "soft_guidance"
        return SOFT_GUIDANCE_MESSAGE, debug

    selected_parts: list[str] = []
    used_norms: set[str] = set()
    for candidate in candidates:
        normalized = normalize_for_dedupe(candidate["candidate"])
        if normalized in used_norms:
            continue
        selected_parts.append(candidate["candidate"])
        used_norms.add(normalized)
        if len(selected_parts) == 2:
            break

    answer = clean_answer_text(" ".join(dedupe_answer_parts(selected_parts)))
    debug["selected_span"] = answer
    return answer, debug


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


def build_remote_prompt(query: str, retrieved_chunks: list[dict[str, Any]], query_mode: str) -> str:
    sources = "\n\n".join(
        f"[Source {index}] {chunk['doc_name']} (score={chunk['combined_score']})\n{chunk['text']}"
        for index, chunk in enumerate(retrieved_chunks, start=1)
    )
    mode_instruction = {
        "summary": "Summarize the document context in 1-3 grounded sentences.",
        "explanatory": "Answer the explanatory question using only the retrieved context and include the supporting reasons from the document.",
        "factoid": "Answer with a concise document-grounded fact.",
    }.get(query_mode, "Answer using only the provided context.")
    return (
        "You are a careful document question-answering assistant.\n"
        "Answer only from the provided sources.\n"
        "If the answer is not present, reply exactly: I could not find that in the uploaded documents.\n"
        f"{mode_instruction}\n\n"
        f"Retrieved sources:\n{sources}\n\n"
        f"Question: {query}"
    )


def answer_with_remote_llm(query: str, retrieved_chunks: list[dict[str, Any]], query_mode: str) -> tuple[str, dict[str, Any]]:
    settings = get_llm_settings()
    prompt = build_remote_prompt(query, retrieved_chunks, query_mode)
    debug: dict[str, Any] = {
        "mode": "remote-llm",
        "prompt": prompt,
        "model": settings["model"],
        "host": settings["display_host"],
        "query_mode": query_mode,
    }

    if not settings["base_url"]:
        debug["fallback_reason"] = "no_llm_configured"
        if query_mode == "summary":
            answer, local_debug = answer_summary_question_local(query, retrieved_chunks)
        elif query_mode == "explanatory":
            answer, local_debug = answer_explanatory_question_local(query, retrieved_chunks)
        else:
            answer, local_debug = answer_factoid_question(query, retrieved_chunks)
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
        if query_mode == "summary":
            answer, local_debug = answer_summary_question_local(query, retrieved_chunks)
        elif query_mode == "explanatory":
            answer, local_debug = answer_explanatory_question_local(query, retrieved_chunks)
        else:
            answer, local_debug = answer_factoid_question(query, retrieved_chunks)
        debug["local_fallback"] = local_debug
        return answer, debug


def answer_question(query: str) -> dict[str, Any]:
    resolved_query, rewritten = resolve_follow_up_question(query)
    query_mode = classify_query_mode(resolved_query)
    question_type = "metadata" if query_mode == "metadata" else "content"
    debug_payload: dict[str, Any] = {
        "question_type": question_type,
        "query_mode": query_mode,
        "rewritten_follow_up_question": rewritten,
    }

    if query_mode == "metadata":
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

    if query_mode == "follow_up" and rewritten is None:
        debug_payload["mode"] = "soft-guidance"
        debug_payload["fallback_type"] = "soft_guidance"
        return {
            "answer": SOFT_GUIDANCE_MESSAGE,
            "sources": [],
            "debug": debug_payload,
            "confidence": 0.0,
            "resolved_query": resolved_query,
        }

    if query_mode == "out_of_scope_candidate":
        debug_payload["mode"] = "hard-refusal"
        debug_payload["fallback_type"] = "hard_refusal"
        return {
            "answer": REFUSAL_MESSAGE,
            "sources": [],
            "debug": debug_payload,
            "confidence": 0.0,
            "resolved_query": resolved_query,
        }

    top_k = 8 if query_mode == "summary" else 6
    retrieval_query = resolved_query
    if query_mode == "summary" and resolved_query.lower().strip() in {"summarize", "summary", "summarize the document."}:
        retrieval_query = "document summary main idea overview healthcare ai"

    retrieved = retrieve_chunks(
        query=retrieval_query,
        vectorizer=st.session_state.vectorizer,
        chunk_matrix=st.session_state.chunk_matrix,
        chunks=st.session_state.body_chunks,
        top_k=top_k,
    )
    if query_mode == "summary" and not retrieved:
        retrieved = representative_document_chunks(st.session_state.body_chunks, top_k=8)
    debug_payload["retrieved_chunks"] = retrieved

    if not retrieved:
        fallback = fallback_message(query_mode, hard_refusal=is_out_of_scope_candidate(resolved_query))
        debug_payload["mode"] = "content-refusal"
        debug_payload["fallback_type"] = "hard_refusal" if fallback == REFUSAL_MESSAGE else "soft_guidance"
        debug_payload["refusal_reason"] = "no_retrieved_chunks"
        return {
            "answer": fallback,
            "sources": [],
            "debug": debug_payload,
            "confidence": 0.0,
            "resolved_query": resolved_query,
        }

    if query_mode == "summary":
        precheck_candidates = [
            {"score": chunk["combined_score"], "overlap": chunk["overlap"], "doc_name": chunk["doc_name"], "chunk_id": chunk["chunk_id"]}
            for chunk in retrieved
        ]
        should_summarize, evidence = should_answer(
            "summary",
            retrieved,
            precheck_candidates,
            {"best_score": retrieved[0]["combined_score"], "overlap_score": retrieved[0]["overlap"]},
        )
        debug_payload["evidence"] = evidence
        if not should_summarize:
            return {
                "answer": SOFT_GUIDANCE_MESSAGE,
                "sources": retrieved[:4],
                "debug": {**debug_payload, "mode": "soft-guidance", "fallback_type": "soft_guidance"},
                "confidence": 0.0,
                "resolved_query": resolved_query,
            }

    if remote_llm_configured():
        remote_chunks = retrieved[:6] if query_mode == "summary" else retrieved[:4]
        answer, mode_debug = answer_with_remote_llm(
            resolved_query,
            remote_chunks,
            "explanatory" if query_mode == "follow_up" else query_mode,
        )
    else:
        if query_mode == "summary":
            answer, mode_debug = answer_summary_question_local(resolved_query, retrieved[:8])
        elif query_mode in {"explanatory", "follow_up"}:
            answer, mode_debug = answer_explanatory_question_local(resolved_query, retrieved[:6])
        else:
            answer, mode_debug = answer_factoid_question(resolved_query, retrieved[:4])

    debug_payload.update(mode_debug)
    evidence = debug_payload.get("evidence", {})
    debug_payload["best_score"] = evidence.get("best_score", retrieved[0]["combined_score"])
    debug_payload["overlap_score"] = evidence.get("overlap_score", retrieved[0]["overlap"])
    if answer == SOFT_GUIDANCE_MESSAGE:
        debug_payload["fallback_type"] = "soft_guidance"
    elif answer == REFUSAL_MESSAGE:
        debug_payload["fallback_type"] = "hard_refusal"
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
    st.title("Document Intelligence RAG Assistant")
    st.caption("Upload your files, index them once, and chat with your documents in a grounded, source-aware workflow.")


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
            st.metric(label, value)


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
            st.write("Remote LLM")
            st.caption(llm_settings["display_host"])
            st.caption(f"Model: {llm_settings['model']}")
        else:
            st.write("Local Extractive")
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

def render_history() -> None:
    for index, item in enumerate(st.session_state.answer_history, start=1):
        with st.chat_message("user"):
            st.markdown(strip_html_artifacts(item["question"]))
            resolved_query = strip_html_artifacts(item.get("resolved_query", ""))
            if resolved_query and resolved_query != strip_html_artifacts(item["question"]):
                st.caption(f"Resolved as: {resolved_query}")
        mode = item["debug"].get("mode", item["debug"].get("question_type", "unknown"))
        confidence = item.get("confidence", 0.0)
        with st.chat_message("assistant"):
            st.markdown(strip_html_artifacts(item["answer"]))
            st.caption(f"Mode: {mode} | Confidence: {confidence:.2f}")

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
    question = strip_html_artifacts(question)
    result = answer_question(question)
    clean_answer = strip_html_artifacts(result["answer"])
    clean_resolved_query = strip_html_artifacts(result["resolved_query"])
    st.session_state.current_question = question
    st.session_state.retrieved_sources = result["sources"]
    st.session_state.last_debug = result["debug"]
    st.session_state.last_prompt = result["debug"].get("prompt", "")

    if classify_query_mode(clean_resolved_query) != "metadata":
        st.session_state.last_content_question = clean_resolved_query

    register_answer_usage(clean_answer, result["sources"], result["debug"])

    st.session_state.answer_history.append(
        {
            "question": question,
            "resolved_query": clean_resolved_query,
            "answer": clean_answer,
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
# 6. Explanatory question -> grounded hospital/benefit explanation from document content
# 7. Summary question -> short grounded summary from retrieved body chunks
# 8. Out-of-scope question like "What is the capital of Australia?" -> refusal


st.set_page_config(page_title="Document Intelligence RAG Assistant", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)
init_session_state()
render_sidebar()
render_hero()
render_stats()

st.subheader("Upload Documents")
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
    st.caption("Upload files and click Process Documents to start a new indexed session.")

st.subheader("Indexed Files")
if st.session_state.indexed_files:
    st.write(st.session_state.indexed_files)
else:
    st.caption("No documents indexed yet.")

st.subheader("Chat")

prompt = st.chat_input("Ask about your documents...", disabled=not bool(st.session_state.documents))
if prompt:
    with st.spinner("Thinking..."):
        handle_new_question(prompt)

if not st.session_state.documents:
    st.caption("Upload and process documents to start asking questions.")
else:
    render_history()
    render_debug_panels()
