# core/llm.py
from functools import lru_cache
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings


@lru_cache(maxsize=1)
def get_llm():
    """Return a single shared chat LLM instance backed by Ollama."""
    return ChatOllama(model="llama3", temperature=0.4)


@lru_cache(maxsize=1)
def get_embeddings():
    """Reuse the embedding model so it is not reloaded on every call."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
