import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from core.vectorstore import get_vectorstore

DATA_DIR = os.path.join("data", "music_knowledge")

def ingest_music_knowledge():
    file_paths = glob.glob(os.path.join(DATA_DIR, "*.md"))

    if not file_paths:
        raise ValueError("No .md files found in data/music_knowledge")

    documents = []
    for path in file_paths:
        loader = TextLoader(path, encoding="utf-8")
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    vs = get_vectorstore(collection_name="music_knowledge")
    vs.add_documents(chunks)
    vs.persist()

    print(f"Ingested {len(chunks)} chunks from {len(file_paths)} files.")

if __name__ == "__main__":
    ingest_music_knowledge()


