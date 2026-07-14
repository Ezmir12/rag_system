import logging
import os
from dotenv import load_dotenv

load_dotenv()


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    return logging.getLogger(name)


def get_api_url() -> str:
    return os.getenv("API_URL", "http://127.0.0.1:8001")


def get_chat_model() -> str:
    return os.getenv("CHAT_MODEL", "gpt-4o-mini")


def get_embedding_model() -> str:
    return os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def get_persist_dir() -> str:
    return os.getenv("PERSIST_DIRECTORY", "./chroma_db")
