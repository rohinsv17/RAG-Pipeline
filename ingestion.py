"""
Rohin Vijay
Ingestion file for RAG
--------------------------------
Ingests PDF and Docx files in chunks to the Vector DB
Usage: Run ingestion.py in CLI, checks for any uningested files and ingests to Vector DB
Have Docker Desktop application running for Qdrant container
"docker run -p 6333:6333 -p 6334:6334 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant" in open terminal to access Qdrant container
"""

import os
import re
import uuid
import pdfplumber
from docx import Document
from openai import OpenAI
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from config import (
    PDF_FOLDER,
    QDRANT_URL,
    QDRANT_COLLECTION_NAME,
    QDRANT_VECTOR_SIZE,
    OLLAMA_BASE_URL,
    OLLAMA_API_KEY,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_MIN_WORDS,
    CHUNK_LARGE_THRESHOLD,
    LARGE_CHUNK_SIZE
)

# Ingestion and embedding class
class Ingester:


    # == EXTRACTION AND PREPROCESSING ==

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


    # == CHUNKING ==

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

    # Chunks text across pages for large documents
    def chunkAcrossPages(self, pages: list[dict], chunk_size: int) -> list[dict]:
    
        # Initializes lists to hold all words per page and 
        page_words = []
        page_end_index = []

        # Iterates through each page
        for page in pages:
            # Splits page text into words
            words = page["text"].split()

            # Skips pages with no words
            if(not words):
                continue

            # Appends words to list and records page number and end index of words as a tuple
            page_words.extend(words)
            page_end_index.append((page["page"], len(page_words) - 1))

        # Initializes list to hold chunks and indices for chunking
        chunks = []
        start = 0
        chunk_index = 0

        # Iterates through all words and creates chunks according to settings
        while(start < len(page_words)):
            end = start + chunk_size

            # Adds a section of words to the chunks list
            chunk_text = " ".join(page_words[start:end])

            # Records end index of chunk
            chunk_end = min(end, len(page_words)) - 1

            # Initializes variables to hold start and end page numbers
            start_page = None
            end_page = None

            # Iterates through list of page end indices
            for page_num, page_end in page_end_index:
                # Assigns start and end page numbers for the chunk based on the end index of the chunk
                if(start_page is None and page_end >= start):
                    start_page = page_num
                if(page_end >= chunk_end):
                    end_page = page_num
                    break

            # Builds a single page number or range string for the payload
            if(start_page == end_page):
                page_label = start_page
            else:
                page_label = f"{start_page}-{end_page}"
                
            chunks.append({
                "text":chunk_text,
                "page":page_label,
                "chunk_index":chunk_index,
            })

            start += chunk_size - CHUNK_OVERLAP
            chunk_index += 1

        return chunks


    # == EMBEDDING ==

    # Sends chunks to BGE-Large and retrieves dense vectors
    def denseEmbedChunks(self, chunks: list[str]) -> list[list[float]]:

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
    
    # Uses fastembed to retrieve sparse vectors from chunks
    def sparseEmbedChunks(self, chunks: list[str]) -> list[models.SparseVector]:

        # Creates a list to hold sparsely embedded chunks
        sparse_embeds = list(self.sparse_model.embed(chunks, batch_size = 32))

        # Initializes list to hold sparse vectors
        sparse_vectors = []

        # Iterates through list of SparseEmbedding objects and appends to list as a vector
        for embed in sparse_embeds:
            sparse_vectors.append(models.SparseVector(
                indices = embed.indices.tolist(),
                values = embed.values.tolist(),
            ))

        return sparse_vectors


    # == STORING VECTORS ==

    # Stores dense and sparse vectors into Qdrant db as PointStructs
    def storeChunks(self, chunk_ids: list[str], chunk_texts: list[str], dense_embeddings: list[list[float]], sparse_embeddings: list[models.SparseVector], metadatas: list[dict]) -> None:

        # Initializes a list to hold a PointStruct
        points = []

        # Iterates through all chunks and creates a PointStruct to store vectors
        for id, text, dense, sparse, meta in zip(chunk_ids, chunk_texts, dense_embeddings, sparse_embeddings, metadatas):

            # Creates a uuid from the chunk id with the first 25 chars of text so that it can be fed into PointStructs
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, id + text[:25]))

            # Creates PointStruct for unique chunk id and appends to list
            points.append(models.PointStruct(
                id = point_id,
                vector = {
                    "dense": dense,
                    "sparse": sparse,
                },
                payload = {
                    "id": id,
                    "text": text,
                    "source": meta["source"],
                    "page": meta["page"],
                    "chunk": meta["chunk"],
                }
            ))

        # Uploads batch of PointStructs to collection
        self.qdrant_client.upload_points(
            collection_name = QDRANT_COLLECTION_NAME,
            points = points,
            parallel = 4,
            max_retries = 3,
        )
        print(f"[Ingester] Stored {len(points)} chunks in Qdrant")

    # Orchestrates entire ingestion process for PDFs and docx files
    def ingestFolder(self, folder: str = PDF_FOLDER) -> None:

        supported_files = [
            file for file in os.listdir(folder)
            if(file.lower().endswith(".pdf") or file.lower().endswith(".docx"))
        ]

        if(not supported_files):
            print(f"[Ingester] No supported files found in '{folder}'.")
            return

        # Fetch existing ids by scrolling all points in collection
        existing_ids = set()
        next_offset = 0
        while next_offset is not None:
            scroll_result, next_offset = self.qdrant_client.scroll(
                collection_name = QDRANT_COLLECTION_NAME,
                limit = 1000,
                offset = next_offset,
                with_vectors = False,
            )

            for point in scroll_result:
                existing_ids.add(point.payload.get("id"))


        total_chunks_stored = 0

        for filename in supported_files:
            file_path = os.path.join(folder, filename)
            print(f"\n{'-'*50}")
            print(f"Processing: {filename}")
            print(f"{'-'*50}")

            # Step 1 - extract
            pages = self.extractText(file_path)
            if(not pages):
                print("  No extractable text found. Skipping.")
                continue
            print(f"  Extracted {len(pages)} pages")

            # Collect all chunks across all pages for this PDF
            chunk_ids = []
            chunk_texts = []
            metadatas = []

            # Chunks by connecting
            if(len(pages) >= CHUNK_LARGE_THRESHOLD):
                print(f"  Large document detected: ({len(pages)}), using chunk size of {LARGE_CHUNK_SIZE}")

                # Step 2 - cleans each page's text before passing to chunker
                cleaned_pages = []

                for page in pages:
                    cleaned_pages.append({
                        "page": page["page"],
                        "text": self.cleanText(page["text"]),
                    })

                # Step 3 - chunks across all pages as one combined word list
                cross_page_chunks = self.chunkAcrossPages(cleaned_pages, LARGE_CHUNK_SIZE)

                for chunk in cross_page_chunks:
                    # Filters out near-empty chunks
                    if(len(chunk["text"].split()) < CHUNK_MIN_WORDS):
                        continue

                    # Builds chunk ID using chunk index since there is no single page
                    chunk_id = f"{filename}_c{chunk['chunk_index']}"

                    # Skips chunk if already ingested
                    if(chunk_id in existing_ids):
                        continue

                    chunk_ids.append(chunk_id)
                    chunk_texts.append(chunk["text"])
                    metadatas.append({
                        "source":filename,
                        "page":chunk["page"],
                        "chunk":chunk["chunk_index"],
                    })
            else:
                print(f"  Standard size document: ({len(pages)}), using chunk size of {CHUNK_SIZE}")

                for page in pages:
                    
                    # Step 2 - clean
                    cleaned = self.cleanText(page["text"])

                    # Step 3 - chunk
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
                            "page": page["page"],
                            "chunk": i,
                        })

            if(not chunk_ids):
                print(f"  All chunks already in Qdrant. Skipping.")
                continue

            print(f"  Created {len(chunk_ids)} new chunks")

            # Step 4 - embeds chunks into dense and sparse vectors
            print(f"  Embedding chunks (this is the slow part)...")
            dense_embeddings = self.denseEmbedChunks(chunk_texts)
            sparse_embeddings = self.sparseEmbedChunks(chunk_texts)

            # Step 5 - store
            self.storeChunks(chunk_ids, chunk_texts, dense_embeddings, sparse_embeddings, metadatas)
            total_chunks_stored += len(chunk_ids)

        # Prints summary
        print(f"\n{'='*50}")
        print(f"Ingestion complete.")
        print(f"New chunks stored this run: {total_chunks_stored}")
        print(f"Total chunks in collection: {self.qdrant_client.count(collection_name = QDRANT_COLLECTION_NAME).count}")
        print(f"{'='*50}\n")

    # Constructor sets up connections to Qdrant and Ollama, and prints config summary
    def __init__(self):
        print("[Ingester] Initializing...")

        # Initializes Qdrant client in Docker
        self.qdrant_client = QdrantClient(url = QDRANT_URL)

        # Initializes Qdrant fastembed Sparse Embedding model
        self.sparse_model = SparseTextEmbedding(model_name = "Qdrant/bm25")

        # If collection does not exist, creates a Qdrant collection with dense and sparse vector configs
        if(not self.qdrant_client.collection_exists(collection_name = QDRANT_COLLECTION_NAME)):
            self.qdrant_client.create_collection(
                collection_name = QDRANT_COLLECTION_NAME,
                vectors_config = {
                    "dense": models.VectorParams(
                        distance = models.Distance.COSINE,
                        size = QDRANT_VECTOR_SIZE,
                    )
                },
                sparse_vectors_config = {
                    "sparse": models.SparseVectorParams(
                        modifier = models.Modifier.IDF
                    )
                }
            )
            print(f"[Ingester] Created Qdrant Collection '{QDRANT_COLLECTION_NAME}'")
        else:
            print(f"[Ingester] Collection '{QDRANT_COLLECTION_NAME} already exists'")

        # Initializes OpenAI client using Ollama as localhost
        self.ollama_client = OpenAI(base_url = OLLAMA_BASE_URL, api_key = OLLAMA_API_KEY)

        print(f"[Ingester] Connected to Qdrant at '{QDRANT_URL}'")
        print(f"[Ingester] Collection: '{QDRANT_COLLECTION_NAME}' ({self.qdrant_client.count(collection_name = QDRANT_COLLECTION_NAME).count} chunks already stored)")
        print(f"[Ingester] Embedding model: {EMBEDDING_MODEL}\n")

# Entry point runs program
if(__name__ == "__main__"):
    ingester = Ingester()
    ingester.ingestFolder()