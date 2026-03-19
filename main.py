from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from typing import List
import shutil
import tempfile
import os
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from loaders.document_loader import load_pdf
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

@app.post("/initialize")
async def initialize(files: List[UploadFile] = File(...)):
    all_chunks = []
    embeddings = get_embeddings()

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name
        
        docs = load_pdf(temp_path)
        chunks = process_documents(docs, embeddings)
        all_chunks.extend(chunks)
        os.remove(temp_path)

    vectorstore = get_vectorstore(all_chunks)
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
    state.chain = build_advanced_chain(llm, vectorstore)

    return {"status": "initialized", "chunks": len(all_chunks)}

@app.post("/ask")
async def ask(request: QuestionRequest):
    if state.chain is None:
        return {"error": "System not initialized. Please upload PDFs first."}

    async def generate():
        # This uses the 'astream' method for the typing effect
        async for chunk in state.chain.astream({"input": request.question, "chat_history": []}):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")



"""from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import StreamingResponse
from typing import List
import shutil
import os
import tempfile
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from loaders.document_loader import load_pdf
from processing.ingestion_service import process_documents # Corrected function name
from vectorstore.vector_service import get_vectorstore, get_embeddings
from chains.agent_logic import build_advanced_chain



app = FastAPI()

# Global variable to store the initialized chain
state = {"chain": None}

class ChatRequest(BaseModel):
    question: str
    history: List[dict] = []

@app.post("/initialize")
async def initialize(files: List[UploadFile] = File(...)):
    all_chunks = []
    embeddings = get_embeddings()

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name
        
        docs = load_pdf(temp_path)
        chunks = process_documents(docs, embeddings) # Ensure this matches your ingestion_service
        all_chunks.extend(chunks)
        os.remove(temp_path)

    vectorstore = get_vectorstore(all_chunks)
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True) # Enabled Streaming
    state["chain"] = build_advanced_chain(llm, vectorstore)

    return {"status": "Success", "message": f"Indexed {len(all_chunks)} semantic chunks."}

@app.post("/ask")
async def ask(request: ChatRequest):
    if not state["chain"]:
        return {"error": "System not initialized"}

    # Convert dictionary history to LangChain message objects if needed
    # (Simplified for this example)
    
    async def event_generator():
        async for chunk in state["chain"].astream({
            "input": request.question,
            "chat_history": [] # Pass your history here
        }):
            yield chunk

    return StreamingResponse(event_generator(), media_type="text/plain")


"""
