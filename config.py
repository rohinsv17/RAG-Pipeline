"""
config.py — Central configuration for the RAG pipeline
--------------------------------------------------------
All tunable settings live here. Neither ingestion.py nor query.py
should contain hardcoded values — they import from this file instead.

To experiment (different model, chunk size, etc.), change it here
and the rest of the pipeline picks it up automatically.
"""

import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

# Folder where your source PDFs live
PDF_FOLDER = "."

# Where ChromaDB writes its database files to disk
# ChromaDB creates this folder automatically if it doesn't exist
CHROMA_DB_PATH = "./chroma_db"


# ─────────────────────────────────────────────
# CHROMADB
# ─────────────────────────────────────────────

# The name of the collection inside ChromaDB that holds your chunks.
# Think of it like a table name in a regular database.
# If you ingest multiple unrelated document sets, give each its own
# collection name so their chunks don't get mixed together.
CHROMA_COLLECTION_NAME = "documents"


# ─────────────────────────────────────────────
# OLLAMA
# ─────────────────────────────────────────────

# The URL where Ollama's local server listens.
# Ollama exposes an OpenAI-compatible API here, which is why
# the OpenAI SDK can talk to it without any other changes.
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# Ollama requires *some* value in the api_key field of the SDK,
# but it doesn't actually validate it — any non-empty string works.
OLLAMA_API_KEY = "ollama"


# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────

# The embedding model — used by BOTH ingestion and querying.
# Both phases must use the same model. If you change this after
# ingesting, you must wipe ChromaDB and re-ingest, because the
# new model produces vectors in a different space.
# Pull with: ollama pull bge-large
EMBEDDING_MODEL = "bge-large"

# The LLM used to generate the final answer.
# Only used in query.py. Safe to swap without re-ingesting.
LLM_MODEL = "gemma4:31b-cloud"


# ─────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────

# Target number of words per chunk.
# ~400 words ≈ ~500 tokens, which fits comfortably inside most
# LLM context windows while being specific enough to retrieve well.
# Smaller = more precise retrieval, less context per chunk.
# Larger = more context per chunk, less precise retrieval.
CHUNK_SIZE = 400

# Number of words to repeat at the start of each new chunk.
# Prevents ideas that straddle a chunk boundary from being lost.
# Rule of thumb: ~10-15% of CHUNK_SIZE.
CHUNK_OVERLAP = 50

# Minimum words a chunk must have to be stored.
# Filters out near-empty chunks from pages that only had a
# header, page number, or a few words of content.
CHUNK_MIN_WORDS = 20


# ─────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────

# How many chunks to retrieve per query (the "k" in kNN).
# These retrieved chunks are what gets sent to the LLM as context.
# More chunks = more context, but also more noise and a longer prompt.
# 3-5 is a good starting range for most documents.
TOP_K = 3

# Minimum similarity score (0.0 to 1.0) a chunk must have to be included.
# ChromaDB returns cosine distance (lower = more similar), so this is
# converted at retrieval time. Set to 0.0 to disable filtering entirely.
# Useful for preventing the LLM from receiving completely irrelevant chunks
# when the question has no good match in the document.
MIN_SIMILARITY = 0.0

# ─────────────────────────────────────────────
# RAG EVALUATION
# ─────────────────────────────────────────────

# Golden Dataset file path for evaluation
GOLDEN_DATASET_PATH = "./golden_dataset.json"

# Evaluation record-holder file path
EVAL_RECORDS_PATH = "./eval_records.json"


# ─────────────────────────────────────────────
# LLM GENERATION
# ─────────────────────────────────────────────

# Maximum tokens the LLM can generate in its response.
# 1024 is enough for a thorough paragraph-length answer.
# Raise it if you want longer, more detailed responses.
MAX_TOKENS = 1024

# Controls randomness in the LLM's output.
# 0.0 = deterministic, always picks the most likely next token (factual, consistent).
# 1.0 = more creative and varied, but can drift or hallucinate more.
# For RAG (grounded, factual answers), keep this low — 0.1 to 0.3.
TEMPERATURE = 0.1

# The system prompt sets the LLM's behaviour and constraints.
# This is the most impactful thing to tune for answer quality.
# Key instructions:
#   - Only use the provided context (prevents hallucination)
#   - Admit when the answer isn't in the context
#   - Keep answers concise and grounded
SYSTEM_PROMPT = """You are a precise question-answering assistant.
Answer the user's question using ONLY the context provided below.
If the answer is not contained in the context, say "I don't have enough information in the provided document to answer that."
Do not use any outside knowledge. Answer in a complete sentence by restating the question as closely as possible, followed by your answer.
Keep your answer concise, within 1-2 sentences, and directly answer the question. 
Do not say, "Based on the provided text".
"""