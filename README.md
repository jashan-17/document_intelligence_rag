# Document RAG

Document RAG is a Retrieval-Augmented Generation style project for asking questions over uploaded PDF and DOCX files. It uses a Streamlit frontend, TF-IDF retrieval, and your own public OpenAI-compatible LLM server over HTTPS for grounded answer generation, with a no-cost extractive fallback when no server is configured.

## Overview

The project lives in:

```text
Portfolio/document_intelligence_rag
```

It allows a user to:

- upload one or more documents
- extract text from PDF and DOCX files
- split the text into overlapping chunks
- build a searchable TF-IDF index over those chunks
- retrieve the most relevant chunks for a question
- generate a grounded answer using retrieved context and a remote LLM

## Project Structure

```text
document_intelligence_rag/
├── app.py
├── requirements.txt
├── README.md
├── data/
├── src/
│   ├── chunker.py
│   ├── generator.py
│   ├── loader.py
│   ├── retriever.py
│   ├── utils.py
│   └── vector_store.py
└── venv/
```

## How It Works

### 1. Document Loading

The loader supports:

- PDF via `PyMuPDF`
- DOCX via `python-docx`

Source: [loader.py](/mnt/c/Users/HP/Desktop/Portfolio/document_intelligence_rag/src/loader.py)

### 2. Chunking

Documents are split into overlapping text chunks to improve retrieval quality.

- default chunk size: `800`
- default overlap: `150`

Source: [chunker.py](/mnt/c/Users/HP/Desktop/Portfolio/document_intelligence_rag/src/chunker.py)

### 3. Retrieval Index

Chunks are indexed using TF-IDF so the app can retrieve relevant sections without calling external embedding APIs.

Source: [vector_store.py](/mnt/c/Users/HP/Desktop/Portfolio/document_intelligence_rag/src/vector_store.py)

### 4. Search

Queries are scored against indexed chunks using cosine similarity over TF-IDF vectors.

Source: [retriever.py](/mnt/c/Users/HP/Desktop/Portfolio/document_intelligence_rag/src/retriever.py)

### 5. Answer Generation

The app supports two answer modes:

- Remote LLM generation when `LLM_BASE_URL` is set
- Extractive fallback when no endpoint is configured

This keeps the app usable even when your model server is offline.

Source: [generator.py](/mnt/c/Users/HP/Desktop/Portfolio/document_intelligence_rag/src/generator.py)

### 6. Frontend

The user interface is built with Streamlit and supports:

- multiple file upload
- document processing on demand
- file indexing status display
- question input
- grounded answer generation
- expandable retrieved source display

Source: [app.py](/mnt/c/Users/HP/Desktop/Portfolio/document_intelligence_rag/app.py)

## Requirements

This project depends on:

- Python
- Streamlit
- scikit-learn
- OpenAI SDK for calling your own compatible inference API
- PyMuPDF
- python-docx
- built-in extractive answer generation

There is an existing `requirements.txt` in the app folder:

[requirements.txt](/mnt/c/Users/HP/Desktop/Portfolio/document_intelligence_rag/requirements.txt)

The dependency list is intentionally lightweight so the app can deploy cleanly on Streamlit Community Cloud without heavyweight ML packages.

## Deployment

This version is designed for:

- Streamlit Community Cloud for the UI
- your own HTTPS-hosted LLM server for generation

Recommended configuration:

- Repository: `document_intelligence_rag`
- Branch: `main`
- Main file path: `app.py`
- Secrets:
  - optional `LLM_BASE_URL`
  - optional `LLM_API_KEY`
  - optional `LLM_MODEL`

If you add `LLM_BASE_URL`, the app uses a real LLM for grounded answer generation.
If you do not add it, the app still works using extractive answering only.

### Streamlit Secrets Example

Use the sample file at [secrets.toml.example](/mnt/c/Users/HP/Desktop/Portfolio/document_intelligence_rag/.streamlit/secrets.toml.example#L1) and set:

```toml
LLM_BASE_URL = "https://your-llm-domain.example.com/v1"
LLM_API_KEY = "replace-with-your-server-key"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
```

### Public LLM Server

This repo now includes a starter public vLLM deployment in:

- [docker-compose.yml](/mnt/c/Users/HP/Desktop/Portfolio/document_intelligence_rag/deploy/vllm/docker-compose.yml#L1)
- [Caddyfile](/mnt/c/Users/HP/Desktop/Portfolio/document_intelligence_rag/deploy/vllm/Caddyfile#L1)
- [.env.example](/mnt/c/Users/HP/Desktop/Portfolio/document_intelligence_rag/deploy/vllm/.env.example#L1)

The stack is:

- `vLLM` serving an OpenAI-compatible API on port `8000`
- `Caddy` providing public HTTPS in front of it

Recommended starter models:

- `Qwen/Qwen2.5-7B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `google/gemma-2-9b-it`

### End-to-End Deployment Steps

#### 1. Deploy the model server on a GPU VM

Use a Linux VM with Docker and a public DNS record such as `llm.your-domain.com`.

```bash
cd deploy/vllm
cp .env.example .env
docker compose up -d
```

Update `.env` with:

- your real domain
- your email for HTTPS certificates
- a strong `VLLM_API_KEY`
- the Hugging Face model name you want to serve
- `HUGGING_FACE_HUB_TOKEN` if the model requires one

Once running, your server should expose an HTTPS OpenAI-compatible API such as:

```text
https://llm.your-domain.com/v1
```

#### 2. Point Streamlit Community Cloud at the app

Deploy this repo on Streamlit Community Cloud with:

- Repository: `document_intelligence_rag`
- Branch: `main`
- Main file: `app.py`

#### 3. Add Streamlit secrets

In Streamlit Community Cloud, add:

```toml
LLM_BASE_URL = "https://llm.your-domain.com/v1"
LLM_API_KEY = "your-vllm-api-key"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
```

#### 4. Verify the connection in the app

After deploy:

1. Open the app.
2. Expand `Runtime configuration`.
3. Click `Test public LLM connection`.
4. Upload documents and ask a question.

## Run the App

From the project folder:

```bash
cd /mnt/c/Users/HP/Desktop/Portfolio/document_intelligence_rag
venv/Scripts/python.exe -m streamlit run app.py
```

If you are not using a project virtual environment, you can run:

```bash
streamlit run app.py
```

## End-to-End Flow

1. Upload one or more PDF or DOCX files.
2. Click `Process Documents`.
3. The app extracts and chunks the text.
4. Chunks are indexed with TF-IDF.
5. The app searches the most relevant chunks for your question.
6. Ask a question in the input box.
7. The app retrieves relevant chunks and either sends them to your public HTTPS LLM API for generation or falls back to extractive answering if no endpoint is set.
8. Retrieved source chunks are shown below the answer.

## Notes

- The vector store is kept in Streamlit session state, so indexing is session-based.
- Uploaded files are written to `data/uploads`.
- `utils.py` currently exists but is empty.
- The app uses the `openai` Python SDK only as a client for your own OpenAI-compatible endpoint.
- The remote LLM endpoint should be OpenAI-compatible, such as `vLLM`.
- Without `LLM_BASE_URL`, the app still works in extractive mode.
- Streamlit Community Cloud cannot call `localhost` on your laptop. The model server must be reachable over the public internet with HTTPS.

## Security Note

There is a `.env` file in the project folder. If it contains a real API key or secret, it should be rotated and removed from version control if it is not needed.

## Future Improvements

- Persist the TF-IDF index to disk
- Add metadata-aware retrieval by file and page
- Improve chunking with sentence-aware splitting
- Add citation spans instead of whole-chunk citations
- Support more document types
- Add conversation history and follow-up questions
- Add stronger ranking heuristics for answer sentence selection
- Add a UI toggle between remote-LLM mode and extractive-only mode

## Summary

This project is a solid document-question-answering pipeline that combines document ingestion, chunking, TF-IDF retrieval, and answer generation in a simple Streamlit interface. It is especially useful as a portfolio project because the pipeline is clear, modular, and easy to extend from extractive search to a real hosted LLM backend.
