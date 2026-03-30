from pathlib import Path
import os
import streamlit as st

from src.loader import load_document
from src.chunker import chunk_text
from src.vector_store import LocalVectorStore
from src.retriever import retrieve_relevant_chunks
from src.generator import check_remote_llm_health, generate_answer, get_llm_settings, remote_llm_configured


UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Document Intelligence RAG Assistant", layout="wide")
st.title("Document Intelligence RAG Assistant")
st.caption("Upload PDF or DOCX files, then ask questions grounded in those documents.")

llm_settings = get_llm_settings()

if os.getenv("GEMINI_API_KEY"):
    st.warning(
        "Deprecated Gemini setting detected. This app now uses a public OpenAI-compatible LLM "
        "endpoint configured with LLM_BASE_URL."
    )
elif remote_llm_configured():
    st.info(
        "Using a public LLM server for answer generation with local document retrieval. "
        "The endpoint is configured through Streamlit secrets or environment variables."
    )
else:
    st.info(
        "No public LLM endpoint detected. The app will use free local extractive answering instead of an LLM."
    )

with st.expander("Runtime configuration", expanded=False):
    st.write(
        {
            "mode": "remote-llm" if remote_llm_configured() else "extractive-fallback",
            "llm_host": llm_settings["display_host"] or "not configured",
            "llm_model": llm_settings["model"],
        }
    )
    if remote_llm_configured():
        if st.button("Test public LLM connection"):
            with st.spinner("Checking remote model server..."):
                healthy, message = check_remote_llm_health()
            if healthy:
                st.success(message)
            else:
                st.error(message)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Process Documents"):
        all_chunks = []

        with st.spinner("Loading and chunking documents..."):
            for uploaded_file in uploaded_files:
                file_path = UPLOAD_DIR / uploaded_file.name

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())

                raw_text = load_document(str(file_path))
                chunks = chunk_text(raw_text)
                all_chunks.extend(chunks)

                if uploaded_file.name not in st.session_state.indexed_files:
                    st.session_state.indexed_files.append(uploaded_file.name)

        if all_chunks:
            with st.spinner("Building searchable document index..."):
                vector_store = LocalVectorStore()
                vector_store.add_documents(all_chunks)

                st.session_state.vector_store = vector_store

            st.success(f"Indexed {len(all_chunks)} chunks from {len(st.session_state.indexed_files)} file(s).")
        else:
            st.warning("No text could be extracted from the uploaded files.")

if st.session_state.indexed_files:
    st.subheader("Indexed Files")
    for name in st.session_state.indexed_files:
        st.write(f"- {name}")

st.subheader("Ask a Question")
query = st.text_input("Enter your question")

if st.button("Get Answer"):
    if not st.session_state.vector_store:
        st.error("Please upload and process documents first.")
    elif not query.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("Retrieving relevant chunks..."):
            retrieved_chunks = retrieve_relevant_chunks(
                query=query,
                vector_store=st.session_state.vector_store,
                top_k=3
            )

        with st.spinner("Generating answer..."):
            answer = generate_answer(query, retrieved_chunks)

        st.markdown("### Answer")
        st.write(answer)

        st.markdown("### Retrieved Sources")
        for i, (chunk, score) in enumerate(retrieved_chunks, start=1):
            with st.expander(f"Source {i} (distance={score:.4f})"):
                st.write(chunk)
