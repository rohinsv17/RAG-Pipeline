"""
query.py — Retriever class for the RAG pipeline
-------------------------------------------------
Responsible for everything that happens at question time:
embedding the query, searching ChromaDB for relevant chunks,
assembling a prompt, and generating an answer with Gemma.

Assumes ingestion.py has already been run and ChromaDB is populated.

Usage:
    from query import Retriever
    retriever = Retriever()
    answer = retriever.ask("What is the main argument of the document?")
    print(answer)
"""

import chromadb
from openai import OpenAI
from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues
from config import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    OLLAMA_BASE_URL,
    OLLAMA_API_KEY,
    EMBEDDING_MODEL,
    LLM_MODEL,
    TOP_K,
    MIN_SIMILARITY,
    MAX_TOKENS,
    TEMPERATURE,
    SYSTEM_PROMPT,
)

# Retrieval and generation class
class Retriever:

    # Constructor sets up connections to ChromaDB and Ollama, and prints config summary
    def __init__(self):
        print("[Retriever] Initializing...")

        # Initializes ChromaDB client in folder
        self.chroma_client = chromadb.PersistentClient(path = CHROMA_DB_PATH)

        # Loads existing collection with cosine similarity
        self.collection = self.chroma_client.get_or_create_collection(name = CHROMA_COLLECTION_NAME, metadata = {"hnsw:space": "cosine"})

        # Initializes OpenAI client using Ollama as localhost
        self.ollama_client = OpenAI(base_url = OLLAMA_BASE_URL, api_key = OLLAMA_API_KEY)

        # Gets a named tracer for this module — used to create custom spans
        self.tracer = trace.get_tracer("rag.retriever")

        chunk_count = self.collection.count()
        print(f"[Retriever] Collection '{CHROMA_COLLECTION_NAME}' loaded — {chunk_count} chunks available")

        # Warns if collection is empty meaning ingestion hasn't been run yet
        if(chunk_count == 0):
            print("[Retriever] WARNING: Collection is empty. Run ingestion.py first.")

        print(f"[Retriever] Embedding model: {EMBEDDING_MODEL}")
        print(f"[Retriever] LLM: {LLM_MODEL}\n")


    # ─────────────────────────────────────────────
    # STEP 1: EMBED THE QUERY
    # ─────────────────────────────────────────────

    # Embeds the user's question into a vector using the same model as ingestion
    def embedQuery(self, question: str) -> list[float]:

        # Sends question to BGE-Large via Ollama and retrieves embedding
        response = self.ollama_client.embeddings.create(model = EMBEDDING_MODEL, input = [question])

        # Extracts the single embedding vector from the response
        embedding = response.data[0].embedding

        # Attaches custom attributes to the auto-instrumented CreateEmbeddings span
        current_span = trace.get_current_span()
        current_span.set_attribute("embedding.model", EMBEDDING_MODEL)
        current_span.set_attribute("embedding.dimensions", len(embedding))

        return embedding


    # ─────────────────────────────────────────────
    # STEP 2: RETRIEVE RELEVANT CHUNKS
    # ─────────────────────────────────────────────

    # Searches ChromaDB for the most similar chunks to the query vector
    def retrieveChunks(self, query_embedding: list[float]) -> list[dict]:

        # Opens a RETRIEVER custom span
        with self.tracer.start_as_current_span("Vector Search") as span:

            # Marks this span as a retrieval operation in Phoenix's UI
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.RETRIEVER.value)

            # Records the top_k config so you can see retrieval settings per trace
            span.set_attribute("retrieval.top_k", TOP_K)

            # Records the minimum similarity threshold used to filter results
            span.set_attribute("retrieval.min_similarity", MIN_SIMILARITY)

            # Queries ChromaDB with Vector Search for top K most similar chunks with their text, metadata and distances
            vector_results = self.collection.query(query_embeddings = [query_embedding], n_results = TOP_K, include = ["documents", "metadatas", "distances"])

            # Queries ChromaDB with Keyword Search using BM25
            #keyword_results = self.collection.query(query_texts = )

            # Unpacks results from outer list since ChromaDB supports multiple queries at once
            documents = vector_results["documents"][0]
            metadatas  = vector_results["metadatas"][0]
            distances  = vector_results["distances"][0]

            # Initializes list to hold retrieved chunks
            chunks = []

            for doc, meta, dist in zip(documents, metadatas, distances):
                # Converts cosine distance to similarity (0 = identical distance becomes 1 = identical similarity)
                similarity = 1 - dist

                # Skips chunks below the minimum similarity threshold
                if(similarity < MIN_SIMILARITY):
                    continue

                # Appends chunk text, source, page and similarity score to list
                chunks.append({
                    "text":doc,
                    "source":meta.get("source", "unknown"),
                    "page":meta.get("page", "?"),
                    "similarity": round(similarity, 4),
                })

            # Records how many chunks passed the similarity filter
            span.set_attribute("retrieval.chunks_returned", len(chunks))

            # Records the top similarity score so you can spot weak retrievals at a glance
            if(chunks):
                span.set_attribute("retrieval.top_similarity", chunks[0]["similarity"])

            return chunks


    # ─────────────────────────────────────────────
    # STEP 3: BUILD THE PROMPT
    # ─────────────────────────────────────────────

    # Assembles retrieved chunks and the question into a prompt string for Gemma
    def buildPrompt(self, question: str, chunks: list[dict]) -> str:

        # Tells Gemma no context was found if retrieval returned nothing
        if(not chunks):
            context_block = "No relevant context was found in the document."
        else:
            # Builds numbered context block with source and page label per chunk
            context_parts = []
            for i, chunk in enumerate(chunks, start = 1):
                header = f"[Chunk {i} — {chunk['source']}, page {chunk['page']}]"
                context_parts.append(f"{header}\n{chunk['text']}")
            # Joins all chunks into one context block separated by blank lines
            context_block = "\n\n".join(context_parts)

        # Assembles final prompt with context block and question
        prompt = (f"Context:\n{context_block}\n\nQuestion: {question}")

        return prompt


    # ─────────────────────────────────────────────
    # STEP 4: GENERATE THE ANSWER
    # ─────────────────────────────────────────────

    # Sends the assembled prompt to Gemma via Ollama and returns its response
    # No manual span needed here — auto_instrument=True in tracing.py captures
    # the OpenAI SDK call automatically as an LLM span with request and response
    def generateAnswer(self, prompt: str) -> str:

        # Calls Gemma with system prompt for grounding and user prompt with context and question
        response = self.ollama_client.chat.completions.create(model = LLM_MODEL, messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}], max_tokens = MAX_TOKENS, temperature = TEMPERATURE)

        # Returns the text content of Gemma's response
        return response.choices[0].message.content


    # ─────────────────────────────────────────────
    # ORCHESTRATOR
    # ─────────────────────────────────────────────

    # Orchestrates the full query pipeline and returns a natural language answer
    def ragOrchestrator(self, question: str, verbose: bool = False) -> str:

        # Opens a root CHAIN span that wraps the entire query pipeline
        # All child spans (embedQuery, retrieveChunks, generateAnswer) nest inside this
        with self.tracer.start_as_current_span("Root") as span:

            # Marks this as the top-level chain span in Phoenix's UI
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)

            # Records the user's question on the root span for easy inspection
            span.set_attribute(SpanAttributes.INPUT_VALUE, question)

            # Returns early if no documents have been ingested yet
            if(self.collection.count() == 0):
                return "No documents have been ingested yet. Please run ingestion.py first."

            # Step 1 — embed the question into a vector
            query_embedding = self.embedQuery(question)

            # Step 2 — find the most relevant chunks from ChromaDB
            chunks = self.retrieveChunks(query_embedding)

            # Returns early if no relevant chunks were found
            if(not chunks):
                return "No relevant content found in the document for that question."

            # Prints retrieved chunks and similarity scores if verbose mode is on
            if(verbose):
                print(f"\n--- Retrieved {len(chunks)} chunk(s) ---")
                for i, chunk in enumerate(chunks, start = 1):
                    print(f"\n[{i}] {chunk['source']} p.{chunk['page']} (similarity: {chunk['similarity']})")
                    print(f"    {chunk['text'][:200]}...")
                print(f"{'─'*50}\n")

            # Step 3 — assemble the prompt from chunks and question
            prompt = self.buildPrompt(question, chunks)

            # Step 4 — generate and return the answer from Gemma
            answer = self.generateAnswer(prompt)

            # Records the final answer on the root span
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, answer)

            return answer


# ─────────────────────────────────────────────
# ENTRY POINT — simple interactive loop
# ─────────────────────────────────────────────

if(__name__ == "__main__"):
    # Must import and call initTracing before creating Retriever
    # so the tracer provider is registered before any spans are opened
    from tracing import initTracing
    initTracing()

    retriever = Retriever()

    print("RAG Pipeline — ready. Type 'quit' to exit.\n")

    while True:
        question = input("Question: ").strip()

        if(not question):
            continue

        if(question.lower() in ("quit", "exit", "q")):
            print("Goodbye.")
            break

        answer = retriever.ragOrchestrator(question, verbose = True)
        print(f"\nAnswer: {answer}\n")
        print("=" * 50 + "\n")