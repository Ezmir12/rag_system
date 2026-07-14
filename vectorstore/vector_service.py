import os
import shutil
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from config import get_embedding_model, get_logger

logger = get_logger(__name__)


def get_embeddings() -> OpenAIEmbeddings:
    """Centralized embedding model configuration."""
    return OpenAIEmbeddings(model=get_embedding_model())


def get_vectorstore(chunks: Optional[list[Any]] = None) -> Chroma:
    """Retrieve or create the persisted Chroma vector store."""
    persist_dir = os.path.abspath("./chroma_db")
    embeddings = get_embeddings()

    if chunks:
        clear_vectorstore(persist_dir)
        logger.info("Creating a new Chroma vector store at %s", persist_dir)
        return Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir,
        )

    logger.info("Loading existing Chroma vector store from %s", persist_dir)
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )


def clear_vectorstore(persist_dir: Optional[str] = None) -> None:
    """Remove the persisted vector store directory if it exists."""
    target_dir = persist_dir or os.path.abspath("./chroma_db")
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir, ignore_errors=True)
        logger.info("Removed vector store directory at %s", target_dir)

