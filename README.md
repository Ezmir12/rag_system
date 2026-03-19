# 🚀  RAG System
An end-to-end Retrieval-Augmented Generation system designed for the iCog Labs technical assessment.

## 🏗️ Architecture
This system follows a modular micro-service inspired architecture:
- **FastAPI Backend:** Handles asynchronous requests and document streaming.
- **Streamlit Frontend:** A modern, reactive chat interface.
- **Advanced RAG:** Features Semantic Chunking, Multi-Query expansion, and RRF reranking.

## 🛠️ Tech Stack
- **Orchestration:** LangChain (LCEL)
- **Database:** ChromaDB
- **Models:** OpenAI GPT-4o-mini & Embeddings
- **APIs:** FastAPI & Uvicorn

## 🚦 Quick Start
1. **Environment:** Create a `.env` file with `OPENAI_API_KEY`.
2. **Install:** `pip install -r requirements.txt`
3. **Run Backend:** `uvicorn main:app --reload --port 8000`
4. **Run Frontend:** `streamlit run app.py`

## 📂 Project Structure
- `loaders/`: Document ingestion.
- `processing/`: Semantic chunking.
- `vectorstore/`: Persistent database.
- `retrieval/`: Multi-query & RRF.
- `chains/`: Conversation logic.
