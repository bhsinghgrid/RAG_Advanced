from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Qdrant, FAISS
from qdrant_client import QdrantClient
import fitz # PyMuPDF
import pdfplumber
import os
import gc

from app.ingestion.image_extractor_v2 import ImageExtractorV2
from app.llm.image_describer import ChartImageDescriber
from app.config import (
    PDF_PATH, 
    CHUNK_SIZE, 
    CHUNK_OVERLAP, 
    QDRANT_HOST, 
    QDRANT_PORT, 
    QDRANT_COLLECTION,
    VECTOR_DIM
)

def run_ingestion(pdf_path, embedder, vectorstores_list=None, start_page_limit=None, end_page_limit=None):
    print(f"[*] Step 1: Loading structured documents from {pdf_path} (Hybrid Loader)...")
    docs = []
    
    # 1.1 Extract Text using PyMuPDF (Fast)
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    start_at = start_page_limit if start_page_limit is not None else 0
    end_at = end_page_limit if end_page_limit is not None else total_pages
    
    for i in range(start_at, end_at):
        page = doc[i]
        text = page.get_text()
        if text.strip():
            # Enrich metadata
            metadata = {
                "source": pdf_path,
                "page": i,
                "content_type": "text"
            }
            
            content_lower = text.lower()
            if any(k in content_lower for k in ["climate", "commitment", "target", "paris agreement"]):
                metadata["is_climate_related"] = True
                if any(k in content_lower for k in ["percent", "%", "target", "committed"]):
                    metadata["content_type"] = "table" # Promote to table search for visibility
            
            docs.append(Document(page_content=text, metadata=metadata))
    doc.close()
    print(f"    - Extracted text from {len(docs)} pages.")

    # 1.2 Extract Tables using pdfplumber (Accurate)
    print("    - Extracting tables using pdfplumber...")
    with pdfplumber.open(pdf_path) as pdf:
        for i in range(start_at, end_at):
            page = pdf.pages[i]
            tables = page.extract_tables()
            for table in tables:
                # Convert table to markdown-like text
                table_text = ""
                for row in table:
                    clean_row = [str(cell).replace("\n", " ") if cell else "" for cell in row]
                    table_text += "| " + " | ".join(clean_row) + " |\n"
                
                if table_text.strip():
                    docs.append(Document(
                        page_content=table_text,
                        metadata={
                            "source": pdf_path,
                            "page": i,
                            "content_type": "table",
                            "is_climate_related": "climate" in table_text.lower()
                        }
                    ))
    print(f"    - Total document elements: {len(docs)}")

    # 2. Extract Images (Phase 6)
    print("[*] Step 2: Extracting images and generating descriptions...")
    image_extractor = ImageExtractorV2(output_dir="data/ingested_images")
    extracted_images = image_extractor.extract(pdf_path)
    image_describer = ChartImageDescriber()
    
    image_docs = []
    # Only describe images in the requested range
    for img in extracted_images:
        if start_at <= img.page < end_at:
            print(f"      - Describing image on page {img.page}...")
            desc = image_describer.describe(img.path)
            context_text = f"[IMAGE DESCRIPTION]\n{desc.text}\n\n[STRUCTURED DATA]\n{desc.structured_data}"
            
            image_docs.append(Document(
                page_content=context_text,
                metadata={
                    "source": pdf_path,
                    "page": img.page,
                    "content_type": "image",
                    "image_path": img.path,
                    "image_type": desc.image_type,
                    "is_climate_related": "climate" in context_text.lower()
                }
            ))
    
    # 3. Chunking & Indexing
    all_docs = docs + image_docs
    print("[*] Step 3: Chunking and Indexing...")
    
    # We split text but keep tables and images whole to preserve structure
    text_docs = [d for d in all_docs if d.metadata.get("content_type") == "text"]
    structured_docs = [d for d in all_docs if d.metadata.get("content_type") in ["table", "image"]]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    text_chunks = splitter.split_documents(text_docs)
    final_chunks = text_chunks + structured_docs # Tables/Images are NOT split
    
    print(f"    - Created {len(final_chunks)} total chunks ({len(text_chunks)} text, {len(structured_docs)} structured).")
    
    # 4. Indexing using QdrantStore and FAISS
    print(f"[*] Step 4: Indexing {len(final_chunks)} chunks...")
    
    from app.vectorstores.qdrant_store import QdrantStore
    
    # Truncate extremely large chunks to avoid embedding model limits
    MAX_CHARS = 10000 
    texts = []
    for c in final_chunks:
        content = c.page_content
        if len(content) > MAX_CHARS:
            content = content[:MAX_CHARS] + "\n...[Truncated due to size]..."
        texts.append(content)
        
    metadatas = [c.metadata for c in final_chunks]
    
    try:
        embeddings = embedder.embed_documents(texts)
        
        # Qdrant Indexing
        print("    - Adding to Qdrant...")
        q_store = QdrantStore()
        q_store.add(embeddings, texts, metadatas)
        
        # FAISS Indexing
        print("    - Adding to FAISS...")
        faiss_store = FAISS.from_documents(final_chunks, embedder)
        faiss_store.save_local("faiss_index")
    except Exception as e:
        print(f"    [!] Error during indexing: {e}")
        print("    [!] Attempting individual indexing for tables...")
        # Fallback logic if needed or just re-raise
        raise e
    
    return len(final_chunks)

if __name__ == "__main__":
    from app.embeddings.embedder import Embedder
    embedder = Embedder()
    run_ingestion(PDF_PATH, embedder)