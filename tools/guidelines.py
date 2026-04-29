"""
tools/guidelines.py
 
Wraps the RAG pipeline as a callable tool for the LangGraph agent.
The agent calls this when it needs to ground a recommendation in NICE guidance.
"""

import os
from rag.ingest import load_vector_store, configure_settings, build_vector_store

configure_settings()

# Build vector store on first run if it doesn't exist or is empty
if not os.path.exists("./chroma_db") or not any(os.scandir("./chroma_db")):
    print("[guidelines] Building vector store for first time...")
    build_vector_store()

_index = load_vector_store()
_query_engine = _index.as_query_engine(similarity_top_k=3)
 
 
def retrieve_guideline(query: str) -> str:
    """
    Retrieves relevant NICE guideline content for a clinical query.
 
    Args:
        query: A clinical question or topic to search for.
               e.g. "troponin threshold for ACS admission"
 
    Returns:
        Relevant guideline text as a string.
    """
    response = _query_engine.query(query)
    result = str(response).strip()
 
    if not result or result == "Empty Response":
        return "No relevant guideline content found for this query."
 
    return result
 