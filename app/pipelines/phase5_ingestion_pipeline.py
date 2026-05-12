from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from app.ingestion.text_extractor import extract_text
from app.ingestion.chunker import chunk_documents
from app.ingestion.metadata import MetadataEnricher
from app.ingestion.table_extractor_v2 import TableExtractorV2
from app.ingestion.table_chunker_v2 import TableChunkerV2, TableChunkingConfig
from app.ingestion.image_extractor_v2 import ImageExtractorV2
from app.llm.image_describer import ChartImageDescriber


def run_phase5_ingestion(
    pdf_path: str,
    embedder: Any,
    vectorstores: Sequence[Any],
    *,
    table_vertical_strategy: str = "lines",
    table_horizontal_strategy: str = "lines",
    table_rows_per_chunk: int = 20,
    include_images: bool = False,
    image_output_dir: str = "data/phase5_images",
    min_image_pixels: int = 40_000,
    max_images: Optional[int] = None,
    skip_non_chart_images: bool = True,
) -> Dict[str, int]:
    """Phase 5 ingestion: index text chunks, table chunks, and optional image chunks.

    This is intentionally implemented as a new pipeline so Phases 1–4 remain unchanged.

    Returns counts: {"text_chunks": ..., "table_chunks": ..., "image_chunks": ..., "total_chunks": ...}
    """
    if not isinstance(vectorstores, (list, tuple)):
        vectorstores = [vectorstores]

    print(f"[*] Phase 5.1 Step 1: Extracting text from {pdf_path}...")
    docs = extract_text(pdf_path)
    print(f"    - Extracted {len(docs)} pages.")

    page_text_by_num = {d.get("metadata", {}).get("page"): d.get("text", "") for d in docs}

    print("[*] Phase 5.1 Step 2: Chunking text documents...")
    text_chunks = chunk_documents(docs)
    for c in text_chunks:
        meta = c.get("metadata", {}).copy()
        meta["content_type"] = "text"
        c["metadata"] = meta
        # Make provenance visible to the LLM without relying on metadata display.
        page = meta.get("page")
        if page is not None:
            c["text"] = f"[TEXT]\npage={page}\n{c.get('text','')}"
    print(f"    - Created {len(text_chunks)} text chunks.")

    print("[*] Phase 5.1 Step 3: Extracting tables...")
    table_extractor = TableExtractorV2(
        vertical_strategy=table_vertical_strategy,
        horizontal_strategy=table_horizontal_strategy,
    )
    extracted_tables = table_extractor.extract(pdf_path)
    print(f"    - Extracted {len(extracted_tables)} tables (best-effort).")

    print("[*] Phase 5.1 Step 4: Chunking tables for retrieval...")
    table_chunker = TableChunkerV2(
        TableChunkingConfig(rows_per_chunk=table_rows_per_chunk)
    )
    table_chunks: List[Dict[str, Any]] = []
    for t in extracted_tables:
        raw_page_text = page_text_by_num.get(t.page, "")
        table_chunks.extend(table_chunker.to_chunks(t, raw_page_text=raw_page_text))

    print(f"    - Created {len(table_chunks)} table chunks.")

    image_chunks: List[Dict[str, Any]] = []
    if include_images:
        print("[*] Phase 5.2 Step 4.5: Extracting images (charts/graphs)...")
        img_extractor = ImageExtractorV2(
            output_dir=image_output_dir,
            min_pixels=min_image_pixels,
            max_images=max_images,
            deduplicate=True,
        )
        images = img_extractor.extract(pdf_path)
        print(f"    - Extracted {len(images)} embedded images.")

        if images:
            print("[*] Phase 5.2 Step 4.6: Describing images with the LLM (cached)...")
            describer = ChartImageDescriber(cache_dir=f"{image_output_dir}/descriptions")
            seen_sha: set[str] = set()
            for img in images:
                if img.sha256 in seen_sha:
                    continue
                seen_sha.add(img.sha256)
                desc = describer.describe(img.path, cache_key=img.sha256)
                if skip_non_chart_images and desc.image_type in {"other", "logo", "photo"}:
                    continue

                image_chunks.append(
                    {
                        "text": (
                            f"[IMAGE]\npage={img.page} image_sha={img.sha256}\n{desc.text}"
                        ),
                        "metadata": {
                            "page": img.page,
                            "content_type": "image",
                            "image_sha": img.sha256,
                            "image_type": desc.image_type,
                            "image_path": img.path,
                            "width": img.width,
                            "height": img.height,
                        },
                    }
                )

            print(f"    - Created {len(image_chunks)} image chunks.")

    print("[*] Phase 5.1 Step 5: Enriching metadata...")
    enricher = MetadataEnricher(document_type="financial_report")
    all_chunks = enricher.enrich(text_chunks + table_chunks + image_chunks)
    print("    - Enrichment complete.")

    texts = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]

    print(f"[*] Phase 5.1 Step 6: Generating embeddings for {len(texts)} chunks...")
    embeddings = embedder.embed(texts)
    print("    - Done.")

    for vs in vectorstores:
        vs_name = vs.__class__.__name__
        print(f"[*] Phase 5.1 Step 7: Adding to vector store ({vs_name})...")
        vs.add(embeddings, texts, metadatas)
        print(f"    - {vs_name} indexing complete.")

    return {
        "text_chunks": len(text_chunks),
        "table_chunks": len(table_chunks),
        "image_chunks": len(image_chunks),
        "total_chunks": len(all_chunks),
    }


if __name__ == "__main__":
    from app.embeddings.embedder import Embedder
    from app.vectorstores.qdrant_store import QdrantStore
    from app.config import PDF_PATH
    import os

    if os.path.exists(PDF_PATH):
        embedder = Embedder()
        q_store = QdrantStore(collection_name="ifc_annual_report_phase5")
        counts = run_phase5_ingestion(PDF_PATH, embedder, [q_store])
        print(f"Phase 5 ingestion complete: {counts}")
    else:
        print(f"PDF not found at {PDF_PATH}")
