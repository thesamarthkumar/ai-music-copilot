# core/llm.py
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_llm():
    # Uses your local Ollama model
    return ChatOllama(model="llama3", temperature=0.4)

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

