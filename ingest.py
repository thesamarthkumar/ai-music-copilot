import shutil
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from core.vectorstore import DEFAULT_DB_DIR, get_vectorstore

DATA_DIR = Path(__file__).resolve().parent / "data" / "music_knowledge"

def ingest_music_knowledge(force_recreate: bool = True):
    if not DATA_DIR.exists():
        raise ValueError(f"Data directory not found: {DATA_DIR}")

    file_paths = sorted(DATA_DIR.glob("*.md"))

    if not file_paths:
        raise ValueError(f"No .md files found in {DATA_DIR}")

    documents = []
    for path in file_paths:
        loader = TextLoader(str(path), encoding="utf-8")
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks were produced from the source files.")

    if force_recreate and DEFAULT_DB_DIR.exists():
        shutil.rmtree(DEFAULT_DB_DIR)

    vs = get_vectorstore(
        collection_name="music_knowledge",
        persist_directory=DEFAULT_DB_DIR,
    )
    vs.add_documents(chunks)

    persist_method = getattr(vs, "persist", None)
    if callable(persist_method):
        persist_method()

    print(f"Ingested {len(chunks)} chunks from {len(file_paths)} files.")

if __name__ == "__main__":
    ingest_music_knowledge()
