import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv() # Ensures OPENAI_API_KEY is loaded from your .env file

def get_embeddings():
    """Centralized embedding model configuration."""
    return OpenAIEmbeddings(model="text-embedding-3-small")

def get_vectorstore(chunks=None):
    """
    Retrieves or creates a Chroma vector store.
    """
    persist_dir = "./chroma_db"
    embeddings = get_embeddings()
    
    if chunks:
        # Create new store from documents [cite: 95]
        vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            persist_directory=persist_dir
        )
        return vectorstore
    
    # Load existing store 
    return Chroma(
        persist_directory=persist_dir, 
        embedding_function=embeddings
    )



