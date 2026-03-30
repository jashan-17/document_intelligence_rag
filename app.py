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

METADATA_LABELS = {
    "title": ["title", "document title", "name of the document"],
    "author": ["author", "written by", "who wrote", "creator"],
    "published": ["published", "publication date", "publish date", "date"],
}

METADATA_FIELDS = tuple(METADATA_LABELS.keys())

CONTENT_KEYWORDS = {
    "uses": ["use", "uses", "application", "applications", "example", "examples"],
    "challenges": ["challenge", "challenges", "risk", "risks", "problem", "problems"],
    "benefits": ["help", "helps", "benefit", "benefits", "improve", "improves"],
}

KNOWN_USE_CASES = [
    "medical imaging",
    "predictive analytics",
    "drug discovery",
    "patient monitoring",
    "clinical decision support",
]

KNOWN_CHALLENGES = [
    "privacy concerns",
    "incorrect diagnoses",
    "data privacy",
    "bias",
    "security concerns",
    "limited training data",
]

REFUSAL_MESSAGE = "I could not find that in the uploaded documents."


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
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]


def extract_text_from_file(file_path: str) -> str:
    return normalize_text(load_document(file_path))


def extract_value_after_label(line: str, field: str) -> str | None:
    for label in METADATA_LABELS[field]:
        pattern = rf"^{re.escape(label)}\s*:\s*(.+)$"
        match = re.match(pattern, line.strip(), flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
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
            metadata["title"] = first_line

    if "published" not in metadata:
        year_match = re.search(r"\b(19|20)\d{2}\b", text[:1500])
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

        sentences = split_sentences(paragraph)
        current = ""
        for sentence in sentences:
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
        unit_sentences = split_sentences(unit) or [unit]
        prospective = " ".join(current_sentences + unit_sentences).strip()
        if current_sentences and len(prospective) > chunk_size:
            chunks.append(" ".join(current_sentences).strip())
            current_sentences = current_sentences[-overlap_sentences:] + unit_sentences
        else:
            current_sentences.extend(unit_sentences)

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
    if (
        "author" in lowered
        or "written by" in lowered
        or "who wrote" in lowered
        or "title" in lowered
        or "name of the document" in lowered
        or "published" in lowered
        or "publication date" in lowered
        or "when was" in lowered and "document" in lowered
    ):
        return "metadata"
    return "content"


def build_index(chunks: list[dict[str, Any]]) -> tuple[TfidfVectorizer | None, Any]:
    if not chunks:
        return None, None

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform([chunk["search_text"] for chunk in chunks])
    return vectorizer, matrix


def lexical_overlap_ratio(query: str, text: str) -> float:
    query_tokens = set(tokenize(query))
    text_tokens = set(tokenize(text))
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def try_metadata_answer(query: str, documents: list[dict[str, Any]]) -> tuple[str | None, dict[str, Any]]:
    debug: dict[str, Any] = {"classification": "metadata", "metadata_candidates": []}

    if "author" in query.lower() or "written by" in query.lower() or "who wrote" in query.lower():
        field = "author"
    elif "title" in query.lower() or "name of the document" in query.lower():
        field = "title"
    else:
        field = "published"

    debug["field"] = field
    for document in documents:
        value = document["metadata"].get(field)
        if value:
            debug["metadata_candidates"].append(
                {
                    "doc_name": document["name"],
                    "field": field,
                    "value": value,
                }
            )
            return value, debug

    return None, debug


def retrieve_chunks(
    query: str,
    vectorizer: TfidfVectorizer | None,
    chunk_matrix: Any,
    chunks: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    if not vectorizer or chunk_matrix is None or not chunks:
        return []

    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, chunk_matrix).flatten()
    lowered = query.lower()

    ranked: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        similarity = float(similarities[idx])
        overlap = lexical_overlap_ratio(query, chunk["search_text"])

        keyword_boost = 0.0
        if any(keyword in lowered for keyword in CONTENT_KEYWORDS["uses"]):
            for phrase in KNOWN_USE_CASES:
                if phrase in chunk["text"].lower():
                    keyword_boost += 0.15
        if any(keyword in lowered for keyword in CONTENT_KEYWORDS["challenges"]):
            for phrase in KNOWN_CHALLENGES:
                if phrase in chunk["text"].lower():
                    keyword_boost += 0.12
        if any(keyword in lowered for keyword in CONTENT_KEYWORDS["benefits"]):
            if any(term in chunk["text"].lower() for term in ["help", "improve", "faster", "better", "support"]):
                keyword_boost += 0.08

        sentence_density = min(len(split_sentences(chunk["text"])), 6) / 10
        combined = (0.68 * similarity) + (0.22 * overlap) + keyword_boost + sentence_density

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
    lines = [line.strip() for line in chunk_text_value.splitlines() if line.strip()]
    sentences = split_sentences(chunk_text_value)
    seen: set[str] = set()
    ordered: list[str] = []
    for item in sentences + lines:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def find_phrase_in_text(text: str, phrase: str) -> str | None:
    match = re.search(re.escape(phrase), text, flags=re.IGNORECASE)
    if not match:
        return None
    return text[match.start() : match.end()]


def extract_list_item(text: str) -> str | None:
    patterns = [
        r"(?:such as|including|include|includes|like)\s+([^.;]+)",
        r"(?:used for|used in|helps with|supports)\s+([^.;]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            parts = [part.strip(" .") for part in re.split(r",| and ", match.group(1)) if part.strip()]
            if parts:
                return parts[0]
    return None


def extract_answer_span(query: str, candidate: str) -> str:
    lowered = query.lower()

    if any(keyword in lowered for keyword in CONTENT_KEYWORDS["uses"]):
        for phrase in KNOWN_USE_CASES:
            matched = find_phrase_in_text(candidate, phrase)
            if matched:
                return matched
        list_item = extract_list_item(candidate)
        if list_item:
            return list_item

    if any(keyword in lowered for keyword in CONTENT_KEYWORDS["challenges"]):
        found: list[str] = []
        for phrase in KNOWN_CHALLENGES:
            matched = find_phrase_in_text(candidate, phrase)
            if matched:
                found.append(matched)
        if found:
            return ", ".join(dict.fromkeys(found))

    return candidate.strip()


def score_candidate_answer(query: str, candidate: str, parent_chunk: dict[str, Any]) -> float:
    overlap = lexical_overlap_ratio(query, candidate)
    score = (0.62 * parent_chunk["combined_score"]) + (0.28 * overlap)

    lowered = query.lower()
    candidate_lower = candidate.lower()

    if any(keyword in lowered for keyword in CONTENT_KEYWORDS["uses"]) and any(
        phrase in candidate_lower for phrase in KNOWN_USE_CASES
    ):
        score += 0.25
    if any(keyword in lowered for keyword in CONTENT_KEYWORDS["challenges"]) and any(
        phrase in candidate_lower for phrase in KNOWN_CHALLENGES
    ):
        score += 0.22
    if len(candidate) > 240:
        score -= 0.18
    if len(candidate.split()) < 3:
        score -= 0.15

    return round(score, 4)


def answer_with_local_extractive_mode(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    debug: dict[str, Any] = {
        "mode": "local-extractive",
        "selected_span": None,
        "candidate_answers": [],
    }

    if not retrieved_chunks:
        debug["refusal_reason"] = "no_retrieved_chunks"
        return REFUSAL_MESSAGE, debug

    if retrieved_chunks[0]["combined_score"] < 0.18:
        debug["refusal_reason"] = f"low_retrieval_score:{retrieved_chunks[0]['combined_score']}"
        return REFUSAL_MESSAGE, debug

    candidates: list[dict[str, Any]] = []
    for chunk in retrieved_chunks:
        for candidate in extract_candidate_units(chunk["text"]):
            answer_span = extract_answer_span(query, candidate)
            candidate_score = score_candidate_answer(query, candidate, chunk)
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

    selected = candidates[0]
    debug["selected_span"] = selected["answer_span"]
    return selected["answer_span"], debug


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
        "Answer using only the provided retrieved sources.\n"
        "If the answer is not present, reply exactly: I could not find that in the uploaded documents.\n"
        "For direct factual questions, answer concisely.\n\n"
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
        answer, local_debug = answer_with_local_extractive_mode(query, retrieved_chunks)
        debug["local_fallback"] = local_debug
        return answer, debug

    try:
        client = OpenAI(
            api_key=settings["api_key"],
            base_url=settings["base_url"],
            timeout=60.0,
        )
        response = client.chat.completions.create(
            model=settings["model"] or "Qwen/Qwen2.5-7B-Instruct",
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer only from the supplied document body context. "
                        "If the answer is missing, say you could not find it in the uploaded documents."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            raise ValueError("Empty LLM response")
        return answer, debug
    except Exception as exc:
        debug["fallback_reason"] = str(exc)
        answer, local_debug = answer_with_local_extractive_mode(query, retrieved_chunks)
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
            return {"answer": metadata_answer, "sources": [], "debug": debug_payload}
        debug_payload["mode"] = "metadata-refusal"
        return {"answer": REFUSAL_MESSAGE, "sources": [], "debug": debug_payload}

    retrieved = retrieve_chunks(
        query=query,
        vectorizer=st.session_state.vectorizer,
        chunk_matrix=st.session_state.chunk_matrix,
        chunks=st.session_state.body_chunks,
        top_k=5,
    )
    debug_payload["retrieved_chunks"] = retrieved

    if not retrieved or retrieved[0]["combined_score"] < 0.18:
        debug_payload["mode"] = "content-refusal"
        debug_payload["refusal_reason"] = "retrieval_below_threshold"
        return {"answer": REFUSAL_MESSAGE, "sources": retrieved[:3], "debug": debug_payload}

    if remote_llm_configured():
        answer, mode_debug = answer_with_remote_llm(query, retrieved[:4])
    else:
        answer, mode_debug = answer_with_local_extractive_mode(query, retrieved[:4])

    debug_payload.update(mode_debug)
    return {"answer": answer, "sources": retrieved[:4], "debug": debug_payload}


def process_uploaded_documents(uploaded_files: list[Any]) -> None:
    fingerprints = fingerprint_uploaded_files(uploaded_files)
    if fingerprints == st.session_state.document_fingerprints:
        return

    documents: list[dict[str, Any]] = []
    body_chunks: list[dict[str, Any]] = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        file_path = UPLOAD_DIR / uploaded_file.name
        with open(file_path, "wb") as target_file:
            target_file.write(file_bytes)

        raw_text = extract_text_from_file(str(file_path))
        metadata = extract_document_metadata(raw_text)
        removed_metadata_lines, body_text = split_document_content(raw_text, metadata)
        body_lines = [line.strip() for line in body_text.splitlines() if line.strip()]

        document = {
            "name": uploaded_file.name,
            "text": raw_text,
            "metadata": metadata,
            "body_text": body_text,
            "body_lines": body_lines,
            "removed_metadata_lines": removed_metadata_lines,
            "preview": raw_text[:1200],
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
            st.text(document["body_text"][:1200] or "(empty body text)")

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
            st.write(item["question"])

        with st.chat_message("assistant"):
            st.write(item["answer"])
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


st.set_page_config(page_title="Document Intelligence RAG Assistant", layout="wide")
init_session_state()

st.title("Document Intelligence RAG Assistant")
st.caption("Upload PDF or DOCX files, index them once, and ask grounded follow-up questions in the same session.")

llm_settings = get_llm_settings()
if remote_llm_configured():
    st.info(
        "Remote LLM mode is enabled. Retrieved body-content chunks will be sent to your public "
        f"OpenAI-compatible endpoint at `{llm_settings['display_host']}`."
    )
else:
    st.info("Local extractive mode is enabled. Answers will be selected deterministically from document metadata or body content.")

toolbar_left, toolbar_middle, toolbar_right = st.columns([1, 1, 2])
with toolbar_left:
    st.checkbox("Debug mode", key="debug_mode")
with toolbar_middle:
    if st.button("Clear chat", use_container_width=True):
        clear_chat()
        st.rerun()
with toolbar_right:
    if st.button("Reset documents", use_container_width=True):
        reset_documents()
        st.rerun()

st.subheader("Indexed Files")
if st.session_state.indexed_files:
    st.write(st.session_state.indexed_files)
else:
    st.caption("No documents indexed yet.")

uploaded_files = st.file_uploader(
    "Upload PDF or DOCX files",
    type=["pdf", "docx"],
    accept_multiple_files=True,
)

if uploaded_files:
    process_col, info_col = st.columns([1, 2])
    with process_col:
        if st.button("Process Documents", use_container_width=True):
            with st.spinner("Extracting metadata, separating body text, and building the body-content index..."):
                process_uploaded_documents(uploaded_files)
            st.success(
                f"Indexed {len(st.session_state.body_chunks)} body chunks from "
                f"{len(st.session_state.documents)} file(s)."
            )
    with info_col:
        st.caption("Reprocess only when your uploaded files change.")

st.subheader("Ask a Question")
prompt = st.chat_input("Ask about the indexed documents...", disabled=not bool(st.session_state.documents))

if prompt:
    st.session_state.current_question = prompt
    result = answer_question(prompt)
    st.session_state.retrieved_sources = result["sources"]
    st.session_state.last_debug = result["debug"]
    st.session_state.last_prompt = result["debug"].get("prompt", "")
    st.session_state.answer_history.append(
        {
            "question": prompt,
            "answer": result["answer"],
            "sources": result["sources"],
            "debug": result["debug"],
        }
    )

if not st.session_state.documents:
    st.caption("Upload and process documents to start asking questions.")
else:
    render_history()
    render_debug_panels()
