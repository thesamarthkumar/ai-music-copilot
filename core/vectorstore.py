# core/vectorstore.py
import os
from typing import Optional
from langchain_community.vectorstores import Chroma
from core.llm import get_embeddings

DEFAULT_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

def get_vectorstore(collection_name: str, persist_directory: Optional[str] = None):
    if persist_directory is None:
        persist_directory = DEFAULT_DB_DIR

    embeddings = get_embeddings()

    vs = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    return vs


