import warnings
from langchain.agents import AgentExecutor, AgentType, initialize_agent
from core.llm import get_llm
from tools.music_rag_tools import (
    music_knowledge_qa,
    music_similarity_recommender,
)

try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning
except ImportError:  # pragma: no cover - fallback for older LangChain versions
    class LangChainDeprecationWarning(DeprecationWarning):
        pass


warnings.filterwarnings(
    "ignore",
    category=LangChainDeprecationWarning,
    message="LangChain agents will continue to be supported.*",
)


def get_music_agent() -> AgentExecutor:
    """Construct the agent that orchestrates the RAG tools."""
    llm = get_llm()
    tools = [
        music_knowledge_qa,
        music_similarity_recommender,
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        max_iterations=5,
        handle_parsing_errors=True,
    )
    return agent
