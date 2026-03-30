# Document RAG

Document RAG is a free Retrieval-Augmented Generation style project for asking questions over uploaded PDF and DOCX files. It uses a Streamlit frontend, TF-IDF retrieval, and extractive answer generation that runs directly inside the app without paid APIs.

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
- generate a grounded answer using retrieved context

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

The app generates answers extractively by selecting the most relevant sentences from the retrieved chunks and citing the matching source numbers.

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
- PyMuPDF
- python-docx
- built-in extractive answer generation

There is an existing `requirements.txt` in the app folder:

[requirements.txt](/mnt/c/Users/HP/Desktop/Portfolio/document_intelligence_rag/requirements.txt)

The dependency list is intentionally lightweight so the app can deploy cleanly on Streamlit Community Cloud without paid APIs or heavyweight ML packages.

## Deployment

This version is designed to deploy directly on Streamlit Community Cloud without extra secrets or external model services.

Recommended configuration:

- Repository: `document_intelligence_rag`
- Branch: `main`
- Main file path: `app.py`
- Secrets: none required

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
7. The app retrieves relevant chunks and generates an extractive final answer.
8. Retrieved source chunks are shown below the answer.

## Notes

- The vector store is kept in Streamlit session state, so indexing is session-based.
- Uploaded files are written to `data/uploads`.
- `utils.py` currently exists but is empty.
- The app does not depend on OpenAI, Ollama, or any paid hosted model API.
- This is a retrieval-plus-extraction workflow rather than a full generative LLM pipeline.

## Security Note

There is a `.env` file in the project folder. If it contains a real API key or secret, it should be rotated and removed from version control if it is not needed.

## Future Improvements

- Persist the FAISS index to disk
- Add metadata-aware retrieval by file and page
- Improve chunking with sentence-aware splitting
- Add citation spans instead of whole-chunk citations
- Support more document types
- Add conversation history and follow-up questions
- Add stronger ranking heuristics for answer sentence selection

## Summary

This project is a solid local document-question-answering pipeline that combines document ingestion, chunking, embedding, retrieval, and answer generation in a simple Streamlit interface. It is especially useful as a portfolio project because the pipeline is clear, modular, and easy to extend.
