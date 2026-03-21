from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from typing import List
import shutil
import tempfile
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

from loaders.document_loader import load_documents
from processing.ingestion_service import process_documents
from vectorstore.vector_service import get_vectorstore, get_embeddings
from chains.agent_logic import build_advanced_chain

app = FastAPI()

# Global state to keep the chain in memory
class GlobalState:
    chain = None

state = GlobalState()

class QuestionRequest(BaseModel):
    question: str
    history: List[dict] = []

def _history_to_langchain(history: List[dict]) -> List:
    """Convert frontend message format to LangChain message objects."""
    messages = []
    for msg in history:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg.get("content", "")))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content=msg.get("content", "")))
    return messages

@app.post("/initialize")
async def initialize(files: List[UploadFile] = File(..., description="PDF and image files")):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    all_chunks = []
    try:
        embeddings = get_embeddings()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Embeddings failed (check OPENAI_API_KEY in .env): {str(e)}",
        )

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
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process '{file.filename}': {str(e)}",
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    try:
        vectorstore = get_vectorstore(all_chunks)
        llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
        state.chain = build_advanced_chain(llm, vectorstore)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Vector store / chain setup failed: {str(e)}",
        )

    return {"status": "initialized", "chunks": len(all_chunks)}

@app.post("/ask")
async def ask(request: QuestionRequest):
    if state.chain is None:
        raise HTTPException(
            status_code=503,
            detail="System not initialized. Please upload documents first.",
        )

    chat_history = _history_to_langchain(request.history)

    async def generate():
        # astream yields string chunks from StrOutputParser; ensure we send text only
        async for chunk in state.chain.astream({
            "input": request.question,
            "chat_history": chat_history
        }):
            if isinstance(chunk, str) and chunk:
                yield chunk
            elif isinstance(chunk, dict):
                # Handle per-node streaming (e.g. {"StrOutputParser": "token"})
                for v in chunk.values():
                    if isinstance(v, str) and v:
                        yield v
                        break

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")



