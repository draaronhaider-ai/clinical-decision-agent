"""
rag/ingest.py

Ingests NICE guideline PDFs into a ChromaDB vector store.
Run this script once to build the vector store, then the agent
queries it at runtime without needing to re-ingest.

Uses:
- Anthropic Claude as the LLM
- HuggingFace sentence-transformers for embeddings (free, runs locally)

Usage:
    python rag/ingest.py
"""

import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings
import chromadb

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GUIDELINES_PATH = os.getenv("GUIDELINES_PATH", "./guidelines")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

# HuggingFace embedding model — runs locally, no API key needed
# all-MiniLM-L6-v2 is fast, lightweight, and works well for medical text
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Anthropic model for query synthesis
ANTHROPIC_MODEL = "claude-sonnet-4-6"


def configure_settings():
    """Configure LlamaIndex to use Anthropic + HuggingFace."""
    Settings.llm = Anthropic(model=ANTHROPIC_MODEL)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)


def build_vector_store():
    """
    Reads all PDFs from the guidelines folder, splits them into chunks,
    embeds them using HuggingFace, and stores them in ChromaDB.

    This only needs to be run once — or again whenever you add new guidelines.
    """
    configure_settings()

    print(f"Loading guidelines from: {GUIDELINES_PATH}")

    # --- Step 1: Load PDFs ---
    documents = SimpleDirectoryReader(GUIDELINES_PATH).load_data()
    print(f"Loaded {len(documents)} document pages")

    # --- Step 2: Split into chunks ---
    # 512 tokens per chunk with 50 token overlap works well for dense
    # clinical guidelines — enough context per chunk, overlap prevents
    # losing meaning at chunk boundaries
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Split into {len(nodes)} chunks")

    # --- Step 3: Set up ChromaDB ---
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection("nice_guidelines")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # --- Step 4: Build the index (embeds and stores all chunks) ---
    print("Embedding chunks and building vector store — this may take a minute...")
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
    )

    print("Vector store built successfully")
    print(f"Stored at: {CHROMA_DB_PATH}")
    return index


def load_vector_store():
    """
    Loads an existing vector store from disk.
    Called by the agent at runtime — much faster than rebuilding.
    """
    configure_settings()

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection("nice_guidelines")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
    )

    return index


if __name__ == "__main__":
    build_vector_store()

    # --- Test retrieval immediately after building ---
    print("\nTesting retrieval...")
    index = load_vector_store()
    query_engine = index.as_query_engine(similarity_top_k=3)

    test_queries = [
        "What does NICE recommend for suspected ACS?",
        "When should a patient with chest pain be admitted?",
        "What is the role of troponin in chest pain assessment?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        response = query_engine.query(query)
        print(f"Response: {str(response)[:300]}...")
