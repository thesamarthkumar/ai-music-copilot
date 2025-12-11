# core/vectorstore.py
from pathlib import Path
from typing import Optional, Union
from langchain_chroma import Chroma
from core.llm import get_embeddings

DEFAULT_DB_DIR = (Path(__file__).resolve().parent.parent / "chroma_db").resolve()

def get_vectorstore(
    collection_name: str,
    persist_directory: Optional[Union[str, Path]] = None,
):
    if persist_directory is None:
        persist_path = DEFAULT_DB_DIR
    else:
        persist_path = Path(persist_directory).expanduser().resolve()

    persist_path.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings()

    vs = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_path),
    )
    return vs
