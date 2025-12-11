from langchain_core.tools import tool
from langchain.chains import RetrievalQA
from core.llm import get_llm
from core.vectorstore import get_vectorstore

@tool("music_knowledge_qa", return_direct=False)
def music_knowledge_qa(query: str) -> str:
    """
    Answer questions about artists, albums, and songs
    using the music knowledge base (RAG).
    """
    llm = get_llm()
    vs = get_vectorstore("music_knowledge")

    retriever = vs.as_retriever(search_kwargs={"k": 4})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    response = qa_chain.invoke({"query": query})
    if isinstance(response, dict):
        return str(response.get("result") or response.get("output_text") or response)
    return str(response)


@tool("music_similarity_recommender", return_direct=False)
def music_similarity_recommender(description: str) -> str:
    """
    Recommend music from the knowledge base
    based on a natural language description.
    """
    vs = get_vectorstore("music_knowledge")
    docs = vs.similarity_search(description, k=5)

    if not docs:
        return "No matching songs found."

    results = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        results.append(
            f"Source: {src}\nSnippet: {d.page_content[:200]}..."
        )

    return "\n\n".join(results)

