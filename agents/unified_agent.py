# agents/unified_agent.py
from langchain.agents import initialize_agent, AgentType
from core.llm import get_llm
from tools.music_rag_tools import (
   music_knowledge_qa, music_similarity_recommender,
)

def get_music_agent():
   llm = get_llm()
   tools = [
      music_knowledge_qa,
      music_similarity_recommender,
   ]

   agent = initialize_agent(
      tools = tools,
      llm = llm,
      agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
      verbose=False,
   )

   return agent
	
