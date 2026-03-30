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
        "chunks": [],
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


def reset_documents() -> None:
    st.session_state.documents = []
    st.session_state.chunks = []
    st.session_state.vectorizer = None
    st.session_state.chunk_matrix = None
    st.session_state.indexed_files = []
    st.session_state.document_fingerprints = []
    st.session_state.retrieved_sources = []
    st.session_state.last_debug = {}
    st.session_state.last_prompt = ""
    clear_chat()


def clear_chat() -> None:
    st.session_state.current_question = ""
    st.session_state.answer_history = []
    st.session_state.retrieved_sources = []
    st.session_state.last_debug = {}
    st.session_state.last_prompt = ""


def normalize_text(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\r", "")).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]{2,}", text.lower())


def extract_text_from_file(file_path: str) -> str:
    return normalize_text(load_document(file_path))


def extract_document_metadata(text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    candidate_lines = [line.strip() for line in text.splitlines() if line.strip()][:60]

    for line in candidate_lines:
        line_lower = line.lower()
        for field, labels in METADATA_LABELS.items():
            for label in labels:
                pattern = rf"^{re.escape(label)}\s*:\s*(.+)$"
                match = re.match(pattern, line_lower, flags=0)
                if match:
                    original_value = line.split(":", 1)[1].strip()
                    if original_value and field not in metadata:
                        metadata[field] = original_value
                        break
            if field in metadata:
                continue

    if "title" not in metadata and candidate_lines:
        first_line = candidate_lines[0]
        if len(first_line) <= 120 and ":" not in first_line:
            metadata["title"] = first_line

    year_match = re.search(r"\b(19|20)\d{2}\b", text[:1500])
    if "published" not in metadata and year_match:
        metadata["published"] = year_match.group(0)

    return metadata


def split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return [part.strip() for part in parts if part.strip()]


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


def build_index(chunks: list[dict[str, Any]]) -> tuple[TfidfVectorizer | None, Any]:
    if not chunks:
        return None, None

    search_corpus = [chunk["search_text"] for chunk in chunks]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(search_corpus)
    return vectorizer, matrix


def metadata_question_type(query: str) -> str | None:
    lowered = query.lower()
    if "author" in lowered or "written by" in lowered or "who wrote" in lowered:
        return "author"
    if "title" in lowered or "name of the document" in lowered:
        return "title"
    if "published" in lowered or "publication date" in lowered or "when was" in lowered:
        return "published"
    return None


def extract_value_after_label(line: str, field: str) -> str | None:
    labels = METADATA_LABELS[field]
    for label in labels:
        pattern = rf"^{re.escape(label)}\s*:\s*(.+)$"
        match = re.match(pattern, line.strip(), flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def try_metadata_answer(query: str, documents: list[dict[str, Any]]) -> tuple[str | None, dict[str, Any]]:
    field = metadata_question_type(query)
    debug: dict[str, Any] = {"metadata_field": field, "metadata_candidates": []}
    if not field:
        return None, debug

    for document in documents:
        metadata = document["metadata"]
        if metadata.get(field):
            debug["metadata_candidates"].append(
                {
                    "doc_name": document["name"],
                    "field": field,
                    "value": metadata[field],
                    "source": "document_metadata",
                }
            )
            return metadata[field], debug

        for line in document["lines"][:80]:
            value = extract_value_after_label(line, field)
            if value:
                debug["metadata_candidates"].append(
                    {
                        "doc_name": document["name"],
                        "field": field,
                        "value": value,
                        "source": "line_scan",
                    }
                )
                return value, debug

    return None, debug


def lexical_overlap_ratio(query: str, text: str) -> float:
    query_tokens = set(tokenize(query))
    text_tokens = set(tokenize(text))
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


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
    field_intent = metadata_question_type(query)
    query_lower = query.lower()

    ranked: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        similarity = float(similarities[idx])
        overlap = lexical_overlap_ratio(query, chunk["search_text"])
        label_boost = 0.0
        if field_intent:
            for line in chunk["lines"]:
                if extract_value_after_label(line, field_intent):
                    label_boost = 0.35
                    break
        short_boost = 0.05 if len(chunk["text"]) < 220 else 0.0
        keyword_boost = 0.05 if any(term in chunk["text"].lower() for term in tokenize(query_lower)) else 0.0
        combined = (0.6 * similarity) + (0.25 * overlap) + label_boost + short_boost + keyword_boost

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
    candidates: list[str] = []
    for item in lines + sentences:
        if item not in seen:
            candidates.append(item)
            seen.add(item)
    return candidates


def find_phrase_in_text(text: str, phrase: str) -> str | None:
    match = re.search(re.escape(phrase), text, flags=re.IGNORECASE)
    if not match:
        return None
    return text[match.start() : match.end()]


def extract_list_item(text: str) -> str | None:
    patterns = [
        r"(?:such as|including|include|includes|like)\s+([^.;]+)",
        r"(?:used in|helps with|supports)\s+([^.;]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        fragment = match.group(1)
        parts = [part.strip(" .") for part in re.split(r",| and ", fragment) if part.strip()]
        if parts:
            return parts[0]
    return None


def extract_answer_span(query: str, candidate: str, field_intent: str | None) -> str:
    if field_intent:
        exact_value = extract_value_after_label(candidate, field_intent)
        if exact_value:
            return exact_value

    lowered_query = query.lower()

    if "one use" in lowered_query or "use of ai" in lowered_query or "example" in lowered_query:
        for known_use in KNOWN_USE_CASES:
            phrase = find_phrase_in_text(candidate, known_use)
            if phrase:
                return phrase
        item = extract_list_item(candidate)
        if item:
            return item

    if "challenge" in lowered_query or "risk" in lowered_query or "problem" in lowered_query:
        found = []
        for phrase in KNOWN_CHALLENGES:
            matched = find_phrase_in_text(candidate, phrase)
            if matched:
                found.append(matched)
        if found:
            return ", ".join(dict.fromkeys(found))

    if ":" in candidate and len(candidate) <= 120:
        value = candidate.split(":", 1)[1].strip()
        if value and len(value.split()) <= 14:
            return value

    return candidate.strip()


def score_candidate_answer(
    query: str,
    candidate: str,
    parent_chunk: dict[str, Any],
    field_intent: str | None,
) -> float:
    overlap = lexical_overlap_ratio(query, candidate)
    score = (0.55 * parent_chunk["combined_score"]) + (0.35 * overlap)

    if field_intent and extract_value_after_label(candidate, field_intent):
        score += 0.45
    if ":" in candidate and len(candidate) <= 120:
        score += 0.15
    if len(candidate.split()) <= 2:
        score -= 0.2
    if len(candidate) > 260:
        score -= 0.18
    return round(score, 4)


def answer_with_local_extractive_mode(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    field_intent = metadata_question_type(query)
    debug: dict[str, Any] = {
        "mode": "local-extractive",
        "field_intent": field_intent,
        "selected_span": None,
        "candidate_answers": [],
    }

    if not retrieved_chunks:
        debug["refusal_reason"] = "no_retrieved_chunks"
        return REFUSAL_MESSAGE, debug

    top_score = retrieved_chunks[0]["combined_score"]
    if top_score < 0.14:
        debug["refusal_reason"] = f"low_retrieval_score:{top_score}"
        return REFUSAL_MESSAGE, debug

    candidates: list[dict[str, Any]] = []
    for chunk in retrieved_chunks:
        for candidate in extract_candidate_units(chunk["text"]):
            answer_span = extract_answer_span(query, candidate, field_intent)
            candidate_score = score_candidate_answer(query, candidate, chunk, field_intent)
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

    if not candidates or candidates[0]["score"] < 0.16:
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


def build_remote_prompt(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
    documents: list[dict[str, Any]],
) -> str:
    metadata_lines = []
    for document in documents:
        if document["metadata"]:
            metadata_lines.append(f"{document['name']}: {document['metadata']}")

    sources = "\n\n".join(
        f"[Source {index}] {chunk['doc_name']} (score={chunk['combined_score']})\n{chunk['text']}"
        for index, chunk in enumerate(retrieved_chunks, start=1)
    )

    metadata_block = "\n".join(metadata_lines) if metadata_lines else "No metadata extracted."
    return (
        "You are a careful document question-answering assistant.\n"
        "Answer using only the provided metadata and retrieved document sources.\n"
        "If the answer is not present, reply exactly: I could not find that in the uploaded documents.\n"
        "For direct metadata questions, return only the exact value.\n"
        "Be concise and include source numbers when helpful.\n\n"
        f"Metadata:\n{metadata_block}\n\n"
        f"Retrieved sources:\n{sources}\n\n"
        f"Question: {query}"
    )


def answer_with_remote_llm(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
    documents: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    settings = get_llm_settings()
    prompt = build_remote_prompt(query, retrieved_chunks, documents)
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
                        "Answer only from the supplied document context. "
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
    documents = st.session_state.documents
    chunks = st.session_state.chunks
    vectorizer = st.session_state.vectorizer
    chunk_matrix = st.session_state.chunk_matrix

    metadata_answer, metadata_debug = try_metadata_answer(query, documents)
    retrieved = retrieve_chunks(query, vectorizer, chunk_matrix, chunks, top_k=5)

    debug_payload: dict[str, Any] = {
        "metadata_debug": metadata_debug,
        "retrieved_chunks": retrieved,
    }

    if metadata_answer:
        debug_payload["mode"] = "metadata"
        debug_payload["selected_span"] = metadata_answer
        return {
            "answer": metadata_answer,
            "sources": retrieved[:3],
            "debug": debug_payload,
        }

    if not retrieved or retrieved[0]["combined_score"] < 0.14:
        debug_payload["mode"] = "refusal"
        debug_payload["refusal_reason"] = "retrieval_below_threshold"
        return {
            "answer": REFUSAL_MESSAGE,
            "sources": retrieved[:3],
            "debug": debug_payload,
        }

    if remote_llm_configured():
        answer, mode_debug = answer_with_remote_llm(query, retrieved[:4], documents)
    else:
        answer, mode_debug = answer_with_local_extractive_mode(query, retrieved[:4])

    debug_payload.update(mode_debug)
    return {
        "answer": answer,
        "sources": retrieved[:4],
        "debug": debug_payload,
    }


def process_uploaded_documents(uploaded_files: list[Any]) -> None:
    fingerprints = fingerprint_uploaded_files(uploaded_files)
    if fingerprints == st.session_state.document_fingerprints:
        return

    documents: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        file_path = UPLOAD_DIR / uploaded_file.name
        with open(file_path, "wb") as target_file:
            target_file.write(file_bytes)

        raw_text = extract_text_from_file(str(file_path))
        metadata = extract_document_metadata(raw_text)
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

        document = {
            "name": uploaded_file.name,
            "text": raw_text,
            "metadata": metadata,
            "lines": lines,
            "preview": raw_text[:1200],
        }
        documents.append(document)

        document_chunks = chunk_text(raw_text)
        for chunk_index, chunk_value in enumerate(document_chunks):
            metadata_prefix = " ".join(f"{key}: {value}" for key, value in metadata.items())
            search_text = f"{metadata_prefix} {chunk_value}".strip()
            chunks.append(
                {
                    "doc_name": uploaded_file.name,
                    "chunk_id": chunk_index,
                    "text": chunk_value,
                    "lines": [line.strip() for line in chunk_value.splitlines() if line.strip()],
                    "search_text": search_text,
                }
            )

    vectorizer, chunk_matrix = build_index(chunks)

    st.session_state.documents = documents
    st.session_state.chunks = chunks
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
        metadata_map = {document["name"]: document["metadata"] for document in st.session_state.documents}
        st.json(metadata_map)

    with st.expander("Chunk list", expanded=False):
        st.write(
            [
                {
                    "doc_name": chunk["doc_name"],
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                }
                for chunk in st.session_state.chunks
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
                    st.write("No sources retrieved.")
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
        "Remote LLM mode is enabled. Retrieved document chunks will be sent to your public "
        f"OpenAI-compatible endpoint at `{llm_settings['display_host']}`."
    )
else:
    st.info("Local extractive mode is enabled. Answers will be selected deterministically from the uploaded documents.")

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
            with st.spinner("Extracting text, metadata, and retrieval chunks..."):
                process_uploaded_documents(uploaded_files)
            st.success(f"Indexed {len(st.session_state.chunks)} chunks from {len(st.session_state.documents)} file(s).")
    with info_col:
        st.caption("Reprocess only when your uploaded files change.")

st.subheader("Ask a Question")

prompt = st.chat_input(
    "Ask about the indexed documents...",
    disabled=not bool(st.session_state.chunks),
)

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

if not st.session_state.chunks:
    st.caption("Upload and process documents to start asking questions.")
else:
    render_history()
    render_debug_panels()
