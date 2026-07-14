import os
import shutil
import tempfile
from typing import Any, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from chains.agent_logic import build_advanced_chain
from config import get_chat_model, get_logger
from loaders.document_loader import load_documents
from processing.ingestion_service import process_documents
from vectorstore.vector_service import clear_vectorstore, get_embeddings, get_vectorstore

logger = get_logger(__name__)
app = FastAPI(title="RAG System API")


class GlobalState:
    chain: Any = None


state = GlobalState()


class QuestionRequest(BaseModel):
    question: str
    history: List[dict] = Field(default_factory=list)


def _history_to_langchain(history: List[dict]) -> List[Any]:
    """Convert frontend message format to LangChain message objects."""
    messages: List[Any] = []
    for msg in history:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg.get("content", "")))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content=msg.get("content", "")))
    return messages


@app.post("/initialize")
async def initialize(files: List[UploadFile] = File(..., description="PDF and image files")) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    all_chunks: List[Any] = []
    try:
        embeddings = get_embeddings()
    except Exception as exc:
        logger.exception("Embedding initialization failed")
        raise HTTPException(
            status_code=500,
            detail=f"Embeddings failed (check OPENAI_API_KEY in .env): {exc}",
        ) from exc

    for file in files:
        await file.seek(0)
        suffix = os.path.splitext(file.filename or "file")[1] or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        try:
            docs = load_documents(temp_path, file.filename or "document")
            if not docs:
                raise HTTPException(
                    status_code=400,
                    detail=f"No content extracted from {file.filename}. Try a different file.",
                )
            chunks = process_documents(docs, embeddings)
            all_chunks.extend(chunks)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Failed to process uploaded file %s", file.filename)
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process '{file.filename}': {exc}",
            ) from exc
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    if not all_chunks:
        raise HTTPException(status_code=400, detail="No content was extracted from the uploaded files.")

    try:
        vectorstore = get_vectorstore(all_chunks)
        llm = ChatOpenAI(model=get_chat_model(), streaming=True)
        state.chain = build_advanced_chain(llm, vectorstore)
    except Exception as exc:
        logger.exception("Vector store or chain setup failed")
        raise HTTPException(
            status_code=500,
            detail=f"Vector store / chain setup failed: {exc}",
        ) from exc

    return {"status": "initialized", "chunks": len(all_chunks)}


@app.post("/ask")
async def ask(request: QuestionRequest) -> StreamingResponse:
    if state.chain is None:
        persist_dir = os.path.abspath("./chroma_db")
        if os.path.isdir(persist_dir) and os.listdir(persist_dir):
            try:
                vectorstore = get_vectorstore()
                llm = ChatOpenAI(model=get_chat_model(), streaming=True)
                state.chain = build_advanced_chain(llm, vectorstore)
            except Exception as exc:
                logger.warning("Failed to recover chain from persisted vector store: %s", exc)

    if state.chain is None:
        raise HTTPException(
            status_code=503,
            detail="System not initialized. Please upload documents first.",
        )

    chat_history = _history_to_langchain(request.history)

    async def generate():
        async for chunk in state.chain.astream({
            "input": request.question,
            "chat_history": chat_history,
        }):
            if isinstance(chunk, str) and chunk:
                yield chunk
            elif isinstance(chunk, dict):
                for value in chunk.values():
                    if isinstance(value, str) and value:
                        yield value
                        break

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


@app.post("/reset")
async def reset() -> dict:
    state.chain = None
    clear_vectorstore()
    return {"status": "reset"}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "ready": state.chain is not None}

