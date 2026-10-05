"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# LM Studio (OpenAI-compatible server). BOTH vars are required:
#   - the openai SDK reads OPENAI_BASE_URL
#   - langchain-openai 0.1.x (used internally by RAGAS) reads OPENAI_API_BASE only
# Missing either one makes RAGAS target the real OpenAI API and fail auth.
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "")
LLM_MODEL = os.getenv("LLM_MODEL", "ternary-bonsai-8b")

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
# Dense search (M2) embeds locally via sentence-transformers — this is a
# HuggingFace model id, NOT an LM Studio model id.
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
# RAGAS (M4) scores through the LM Studio /v1/embeddings endpoint instead,
# which serves a different model id. Kept separate on purpose.
RAGAS_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-bge-m3")

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
