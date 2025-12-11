# 🎵 AI Music Copilot

A LangChain-based agent that performs RAG-powered music analysis, semantic song recommendations, and (soon) lyric generation & critique.

🚀 Overview

AI Music Copilot is a local, agentic AI application built with:

* LangChain

* ChromaDB (vector store)

* HuggingFace sentence-transformers (embeddings)

* Ollama (LLaMA 3) as the local LLM

* Python CLI interface

## It ingests a custom knowledge base of music metadata (artist notes, album summaries, themes), embeds it, and enables an AI agent to:

* Answer questions about artists, albums, and genres using RAG

* Recommend music based on mood, sound, and style

* Reason over music descriptions using embeddings

## (Coming soon) Generate, analyze, and improve song lyrics

All computation runs locally, with no API keys or cloud costs.

## ✨ Features (Current)
✅ Retrieval-Augmented Music Q&A

* Ask questions like: “Describe the mood and themes of The Dark Side of the Moon.”

The agent retrieves relevant chunks from your knowledge base and generates grounded answers.

## ✅ Semantic Music Recommendation Tool

* Ask for similarity-based recommendations: “Find artists with melancholic indie rock vibes.”

* Uses vector similarity search to surface the closest matches.

🎤 To be added later: Lyric Intelligence

* Generate lyrics in various artist styles

* Analyze tone, rhyme, imagery, and structure

* Suggest improvements or rewrite lyrics

## 🧱 Tech Stack

* Python 3.9+

* LangChain: agent orchestration + tools

* ChromaDB: vector store for RAG

* HuggingFace Embeddings: all-MiniLM-L6-v2

* Ollama running llama3 locally

* Sentence splitting using RecursiveCharacterTextSplitter

* Terminal-based CLI for interaction (for now -- will add front-end UI in the future).
