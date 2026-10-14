"""
Rohin Vijay
Configuration file for RAG
--------------------------------
Defines paths, values, and settings for configuration of the RAG pipeline in different areas
Usage: Exists in same directory as RAG pipeline, values imported to other files
"""

# Folder where your source PDFs live
PDF_FOLDER = "."


#  == Qdrant ==

# The name of the collection inside Vector DB that holds files
QDRANT_COLLECTION_NAME = "documents"

# URL where Vector DB writes files and retrieves data
QDRANT_URL = "http://localhost:6333"

# The number of dimensions in dense embedding vectors
QDRANT_VECTOR_SIZE = 1024


# == Ollama ==

# URL where Ollama runs locally
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# Unvalidated API key for Ollama, required by client
OLLAMA_API_KEY = "ollama"

# Embedding model accessed through Ollama for converting text to dense vectors
EMBEDDING_MODEL = "bge-large"

# LLM model accessed through Ollama for generating answers
LLM_MODEL = "gemma4:31b-cloud"


# == Chunking ==

# Maximum number of words per chunk retrieved
CHUNK_SIZE = 400

# Overlap size between chunks to maintain context properly
CHUNK_OVERLAP = 50

# Minimum number of words a chunk can have
CHUNK_MIN_WORDS = 20

# Threshold for a large document to be chunked at a higher size
CHUNK_LARGE_THRESHOLD = 500

# Maximum chunk size used for larger documents
LARGE_CHUNK_SIZE = 800


# == Retrieval ==

# Number of chunks retrieved initally for queries
TOP_K = 50

# Number of chunks kept after reranking
RERANK_K = 10

# Minimum similarity score a relevant chunk can have
MIN_SIMILARITY = 0.0


# == Evaluation ==

# Golden Dataset file path for evaluation
GOLDEN_DATASET_PATH = "./golden_dataset.json"

# Evaluation record-holder file path
EVAL_RECORDS_PATH = "./eval_records.json"


# == LLM ==

# Maximum tokens the LLM can use in a generated response
MAX_TOKENS = 1024

# Controls randomness in LLM output
TEMPERATURE = 0.1

# Prompt given to LLM for proper answering and behavior
SYSTEM_PROMPT = """You are a precise question-answering assistant.
Answer the user's question using ONLY the context provided below.
If the answer is not contained in the context, say "I don't have enough information in the provided document to answer that."
Do not use any outside knowledge. Answer in a complete sentence by restating the question as closely as possible, followed by your answer.
Keep your answer concise, within 1-2 sentences, and directly answer the question. 
Do not say, "Based on the provided text".
"""