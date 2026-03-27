# Document RAG

Document RAG is a local Retrieval-Augmented Generation project for asking questions over uploaded PDF and DOCX files. It uses a Streamlit frontend, local embeddings and generation through Ollama, and FAISS for vector search.

## Overview

The project lives in:

```text
Document RAG/document_intelligence_rag
```

It allows a user to:

- upload one or more documents
- extract text from PDF and DOCX files
- split the text into overlapping chunks
- generate embeddings for those chunks
- store embeddings in a FAISS vector index
- retrieve the most relevant chunks for a question
- generate a grounded answer using retrieved context

## Project Structure

```text
Document RAG/
├── README.md
└── document_intelligence_rag/
    ├── app.py
    ├── requirements.txt
    ├── .env
    └── src/
        ├── chunker.py
        ├── embedder.py
        ├── generator.py
        ├── loader.py
        ├── retriever.py
        ├── utils.py
        └── vector_store.py
```

## How It Works

### 1. Document Loading

The loader supports:

- PDF via `PyMuPDF`
- DOCX via `python-docx`

Source: [loader.py](/mnt/c/Users/HP/Desktop/Portfolio/Document%20RAG/document_intelligence_rag/src/loader.py)

### 2. Chunking

Documents are split into overlapping text chunks to improve retrieval quality.

- default chunk size: `800`
- default overlap: `150`

Source: [chunker.py](/mnt/c/Users/HP/Desktop/Portfolio/Document%20RAG/document_intelligence_rag/src/chunker.py)

### 3. Embeddings

Each chunk is embedded through a local Ollama embeddings endpoint:

- endpoint: `http://localhost:11434/api/embeddings`
- model: `nomic-embed-text`

Source: [embedder.py](/mnt/c/Users/HP/Desktop/Portfolio/Document%20RAG/document_intelligence_rag/src/embedder.py)

### 4. Vector Search

Embeddings are stored in an in-memory FAISS `IndexFlatL2` vector index for similarity search.

Source: [vector_store.py](/mnt/c/Users/HP/Desktop/Portfolio/Document%20RAG/document_intelligence_rag/src/vector_store.py)

### 5. Answer Generation

Retrieved chunks are combined into a prompt and sent to a local Ollama generation endpoint:

- endpoint: `http://localhost:11434/api/generate`
- default model: `phi3:mini`

The generator instructs the model to answer only from the supplied context and cite source numbers.

Source: [generator.py](/mnt/c/Users/HP/Desktop/Portfolio/Document%20RAG/document_intelligence_rag/src/generator.py)

### 6. Frontend

The user interface is built with Streamlit and supports:

- multiple file upload
- document processing on demand
- file indexing status display
- question input
- grounded answer generation
- expandable retrieved source display

Source: [app.py](/mnt/c/Users/HP/Desktop/Portfolio/Document%20RAG/document_intelligence_rag/app.py)

## Requirements

This project depends on:

- Python
- Streamlit
- FAISS
- PyMuPDF
- python-docx
- requests
- Ollama

There is an existing `requirements.txt` in the app folder:

[requirements.txt](/mnt/c/Users/HP/Desktop/Portfolio/Document%20RAG/document_intelligence_rag/requirements.txt)

## Ollama Setup

This app depends on a local Ollama server running at:

```text
http://localhost:11434
```

You should have these models available locally:

```bash
ollama pull nomic-embed-text
ollama pull phi3:mini
```

Then start Ollama before launching the app.

## Run the App

From the project app folder:

```bash
cd "/mnt/c/Users/HP/Desktop/Portfolio/Document RAG/document_intelligence_rag"
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
4. Chunk embeddings are created with Ollama.
5. Embeddings are stored in FAISS.
6. Ask a question in the input box.
7. The app retrieves relevant chunks and generates a final answer.
8. Retrieved source chunks are shown below the answer.

## Notes

- The vector store is kept in Streamlit session state, so indexing is session-based.
- Uploaded files are written to `data/uploads`.
- `utils.py` currently exists but is empty.
- The current implementation uses local Ollama endpoints directly and does not rely on OpenAI in the app code.

## Security Note

There is a `.env` file in the project folder. If it contains a real API key or secret, it should be rotated and removed from version control if it is not needed.

## Future Improvements

- Persist the FAISS index to disk
- Add metadata-aware retrieval by file and page
- Improve chunking with sentence-aware splitting
- Add citation spans instead of whole-chunk citations
- Support more document types
- Add conversation history and follow-up questions
- Add deployment-ready environment configuration

## Summary

This project is a solid local document-question-answering pipeline that combines document ingestion, chunking, embedding, retrieval, and answer generation in a simple Streamlit interface. It is especially useful as a portfolio project because the pipeline is clear, modular, and easy to extend.