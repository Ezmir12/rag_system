# RAG System

This repository implements a document-backed chat application that ingests files, builds a persistent vector index, retrieves relevant passages, and answers questions through a LangChain chain.

## Purpose

The application is organized around a single data flow:

1. Upload one or more supported documents through the Streamlit UI.
2. The FastAPI backend loads each file, extracts text, and creates semantic chunks.
3. The chunks are embedded and stored in a persistent Chroma vector store.
4. The retriever expands the user question into multiple queries, fuses the results, and passes the best passages to the answer-generation chain.
5. The Streamlit interface streams the answer back to the user and exposes basic conversation state.

## Architecture and data flow

- Backend entry point: [main.py](main.py)
  - Exposes the /initialize, /ask, /reset, and /health endpoints.
  - Owns the in-memory chain state for the current session.
- User interface: [app.py](app.py)
  - Sends uploaded documents to the backend and streams chat answers back to the user.
  - Includes a reset action for clearing the knowledge base.
- Document loading: [loaders/document_loader.py](loaders/document_loader.py)
  - Supports PDF, image, and text files. Image OCR relies on the available OCR stack; otherwise it falls back to a placeholder document describing the missing dependency.
- Chunking and embedding: [processing/ingestion_service.py](processing/ingestion_service.py) and [vectorstore/vector_service.py](vectorstore/vector_service.py)
  - Uses semantic chunking and stores embeddings in Chroma.
  - The vector store directory is persisted under ./chroma_db by default.
- Retrieval and answer generation: [retrieval/retrieval_service.py](retrieval/retrieval_service.py) and [chains/agent_logic.py](chains/agent_logic.py)
  - Generates multiple search queries, fuses them with reciprocal rank fusion, and builds a history-aware QA chain.

## Tech stack

- Python 3.10+
- FastAPI for the backend API
- Streamlit for the UI
- LangChain and LangChain LCEL for prompt orchestration and retrieval
- Chroma for vector storage
- OpenAI embeddings and chat models
- python-dotenv for environment loading

## Environment variables

The application reads the following variables from a .env file:

- OPENAI_API_KEY: required for OpenAI embeddings and chat completion.
- API_URL: optional override for the Streamlit backend URL. Defaults to http://127.0.0.1:8001.
- CHAT_MODEL: optional override for the chat model. Defaults to gpt-4o-mini.
- EMBEDDING_MODEL: optional override for the embedding model. Defaults to text-embedding-3-small.
- PERSIST_DIRECTORY: optional override for the Chroma persistence path. Defaults to ./chroma_db.
- LOG_LEVEL: optional Python logging level. Defaults to INFO.

Example:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=your-key
API_URL=http://127.0.0.1:8001
CHAT_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
PERSIST_DIRECTORY=./chroma_db
LOG_LEVEL=INFO
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the application

Start the FastAPI backend:

```bash
uvicorn main:app --reload --port 8001
```

Start the Streamlit UI in a separate terminal:

```bash
streamlit run app.py
```

## Project tree

```text
.
├── app.py
├── main.py
├── config.py
├── requirements.txt
├── README.md
├── assets/
│   └── style.css
├── chains/
│   ├── __init__.py
│   ├── agent_logic.py
│   └── README.md
├── loaders/
│   ├── __init__.py
│   ├── document_loader.py
│   └── README.md
├── processing/
│   ├── __init__.py
│   ├── ingestion_service.py
│   └── README.md
├── retrieval/
│   ├── __init__.py
│   ├── retrieval_service.py
│   └── README.md
├── tools/
│   └── __init__.py
├── vectorstore/
│   ├── __init__.py
│   ├── vector_service.py
│   └── README.md
├── tests/
│   └── test_config.py
└── .streamlit/
    └── config.toml
```
