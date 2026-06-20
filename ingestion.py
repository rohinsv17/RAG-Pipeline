"""
ingestion.py — Ingester class for the RAG pipeline
----------------------------------------------------
Responsible for everything that happens before a question is asked:
reading PDFs, cleaning and chunking text, embedding chunks, and
storing them in ChromaDB.

Run this once per document set (or re-run if your PDFs change).

Usage:
    from ingestion import Ingester
    ingester = Ingester()
    ingester.ingest_folder()
"""

import os
import re
import pdfplumber
from docx import Document
import chromadb
from openai import OpenAI
from config import (
    PDF_FOLDER,
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    OLLAMA_BASE_URL,
    OLLAMA_API_KEY,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_MIN_WORDS,
)

# Ingestion and embedding class
class Ingester:

    # ─────────────────────────────────────────────
    # STEP 1: EXTRACTION
    # ─────────────────────────────────────────────

    # Determines whether to call pdf or docx text extraction
    def extractText(self, file_path: str) -> list[dict]:
        
        # Conditional to determine file type being ingested
        if(file_path.lower().endswith(".pdf")):
            return self.extractPDF(file_path)
        elif(file_path.lower().endswith(".docx")):
            return self.extractDocx(file_path)
        else:
            print(f"Unsupported file type: {file_path}. Skipping.")
            return []

    # Extracts PDF text with pdfplumber and returns list of dictionaries with page number and text
    def extractPDF(self, file_path: str) -> list[dict]:

        # Initialise empty list to hold pages
        pages = []

        # Opens each pdf using pdfplumber
        with pdfplumber.open(file_path) as pdf:
            # Iterates through each page
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()

                # Marks pages without text for skipping
                if(not text):
                    print(f"  [!] Page {i + 1}: no extractable text (scanned?). Skipping.")
                    continue

                # Adds scanned text as a dictionary to the pages list
                pages.append({
                    "page": i + 1,
                    "text": text,
                })

        return pages
    
    # Extracts docx file text with python-docx and returns list of dictionaries
    def extractDocx(self, file_path: str) -> list[dict]:

        # Initialize empty list to hold pages
        pages = []

        # Loads docx file and initializes list to hold text
        doc = Document(file_path)
        full_text_temp = []

        # Appends text to list without any whitespace
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                full_text_temp.append(paragraph.text)

        # Joins all paragraphs into large text
        full_text = "\n".join(full_text_temp)

        if(full_text.strip()):
            pages.append({
                "page": 1,
                "text": full_text,
            })
        
        return pages



    # ─────────────────────────────────────────────
    # STEP 2: CLEANING
    # ─────────────────────────────────────────────

    # Cleans extracted text
    def cleanText(self, text: str) -> str:

        # Rejoin words broken with a hyphen at a line break
        text = re.sub(r"-\n", "", text)

        # Collapse 3+ consecutive newlines into one blank line
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Collapse multiple spaces and tabs into a single space
        text = re.sub(r"[ \t]+", " ", text)

        # Removes whitespace
        text = text.strip()

        return text


    # ─────────────────────────────────────────────
    # STEP 3: CHUNKING
    # ─────────────────────────────────────────────

    # Chunks cleaned text using pre-configured settings for chunking
    def chunkText(self, text: str) -> list[str]:

        # Splits text into words and intializes list to hold chunks
        words = text.split()
        chunks = []
        start = 0

        # Iterates through words and creates chunks according to settings
        while start < len(words):
            end = start + CHUNK_SIZE
            # Adds a section of words to the chunks list
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += CHUNK_SIZE - CHUNK_OVERLAP

        return chunks


    # ─────────────────────────────────────────────
    # STEP 4: EMBEDDING
    # ─────────────────────────────────────────────

    # Sends chunks to BGE-Large and retrieves vectors
    def embedChunks(self, chunks: list[str]) -> list[list[float]]:

        # Initializes list to hold embeddings with batch size for efficiency
        all_embeddings = []
        batch_size = 10

        # Iterates each chunk in batches by page number
        for i in range(0, len(chunks), batch_size):
            # Gets current 
            batch = chunks[i : i + batch_size]

            response = self.ollama_client.embeddings.create(model = EMBEDDING_MODEL, input = batch)

            # response.data is a list of Embedding objects, each with a .embedding attribute
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

            print(f"  Embedded chunks {i + 1}-{min(i + batch_size, len(chunks))} of {len(chunks)}")

        return all_embeddings


    # ─────────────────────────────────────────────
    # STEP 5: STORING
    # ─────────────────────────────────────────────

    # Stores chunks into ChromaDB with embeddings and metadata
    def storeChunks(self, chunk_ids: list[str], chunk_texts: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None:

        # Adds data into ChromaDB collection
        self.collection.add(ids = chunk_ids, documents = chunk_texts, embeddings = embeddings, metadatas = metadatas)
        print(f"  Stored {len(chunk_ids)} chunks in ChromaDB")


    # ─────────────────────────────────────────────
    # ORCHESTRATOR
    # ─────────────────────────────────────────────

    # Orchestrates entire ingestion process for PDFs and docx files
    def ingestFolder(self, folder: str = PDF_FOLDER) -> None:

        supported_files = [
            file for file in os.listdir(folder)
            if(file.lower().endswith(".pdf") or file.lower().endswith(".docx"))
        ]

        if(not supported_files):
            print(f"[Ingester] No supported files found in '{folder}'.")
            return

        # Fetch IDs already stored so we can skip duplicates
        existing = self.collection.get(include = [])
        existing_ids = set(existing["ids"])

        total_chunks_stored = 0

        for filename in supported_files:
            file_path = os.path.join(folder, filename)
            print(f"\n{'─'*50}")
            print(f"Processing: {filename}")
            print(f"{'─'*50}")

            # Step 1 — extract
            pages = self.extractText(file_path)
            if(not pages):
                print("  No extractable text found. Skipping.")
                continue
            print(f"  Extracted {len(pages)} pages")

            # Collect all chunks across all pages for this PDF
            chunk_ids    = []
            chunk_texts  = []
            metadatas    = []

            for page in pages:
                # Step 2 — clean
                cleaned = self.cleanText(page["text"])

                # Step 3 — chunk
                raw_chunks = self.chunkText(cleaned)

                for i, chunk_text in enumerate(raw_chunks):
                    # Filter out near-empty chunks
                    if(len(chunk_text.split()) < CHUNK_MIN_WORDS):
                        continue

                    # Build a unique, traceable ID for this chunk
                    chunk_id = f"{filename}_p{page['page']}_c{i}"

                    # Skip if already ingested
                    if(chunk_id in existing_ids):
                        continue

                    chunk_ids.append(chunk_id)
                    chunk_texts.append(chunk_text)
                    metadatas.append({
                        "source": filename,
                        "page":   page["page"],
                        "chunk":  i,
                    })

            if(not chunk_ids):
                print(f"  All chunks already in ChromaDB. Skipping.")
                continue

            print(f"  Created {len(chunk_ids)} new chunks")

            # Step 4 — embed
            print(f"  Embedding chunks (this is the slow part)...")
            embeddings = self.embedChunks(chunk_texts)

            # Step 5 — store
            self.storeChunks(chunk_ids, chunk_texts, embeddings, metadatas)
            total_chunks_stored += len(chunk_ids)

        # Summary
        print(f"\n{'='*50}")
        print(f"Ingestion complete.")
        print(f"New chunks stored this run: {total_chunks_stored}")
        print(f"Total chunks in collection: {self.collection.count()}")
        print(f"{'='*50}\n")

    # Constructor sets up connections to ChromaDB and Ollama, and prints config summary
    def __init__(self):
        print("[Ingester] Initializing...")

        # Initializes ChromaDB client in folder
        self.chroma_client = chromadb.PersistentClient(path = CHROMA_DB_PATH)

        # Initializes or retrieves data collection with cosine similarity
        self.collection = self.chroma_client.get_or_create_collection(name = CHROMA_COLLECTION_NAME, metadata = {"hnsw:space": "cosine"})

        # Initializes OpenAI client using Ollama as localhost
        self.ollama_client = OpenAI(base_url = OLLAMA_BASE_URL, api_key = OLLAMA_API_KEY)

        print(f"[Ingester] Connected to ChromaDB at '{CHROMA_DB_PATH}'")
        print(f"[Ingester] Collection: '{CHROMA_COLLECTION_NAME}' ({self.collection.count()} chunks already stored)")
        print(f"[Ingester] Embedding model: {EMBEDDING_MODEL}\n")



# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if(__name__ == "__main__"):
    ingester = Ingester()
    ingester.ingestFolder()