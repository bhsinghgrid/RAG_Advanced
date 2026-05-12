from typing import List, Dict, Any
import numpy as np
from app.embeddings.multimodal_embedder import MultimodalEmbedder
from app.vectorstores.faiss_store import FAISSStore


class VisualRetriever:
    """
    Simple visual patch retriever: embeds query text and searches FAISS for
    nearest visual patches. Designed to be replaced with a late-interaction
    / MaxSim approach when ColPali/ColBERT style models are available.
    """

    def __init__(self, store: FAISSStore = None, embedder: MultimodalEmbedder = None, dim: int = 768, save_path: str = "visual_faiss"):
        self.store = store if store is not None else FAISSStore(dim=dim, save_path=save_path)
        self.embedder = embedder if embedder is not None else MultimodalEmbedder()

    def add_patches(self, patch_texts: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        """Add patches to FAISS store.
        patch_texts are optional human-readable descriptions used for inspection.
        """
        self.store.add(embeddings, patch_texts, metadatas)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        # Embed the query with the embedding backend (text embedding)
        q_emb = None
        try:
            q_emb = self.embedder.embed_text([query])[0]
        except Exception:
            # As a fallback, try to embed the raw query string with the text embedder
            q_emb = self.embedder.embed_text([query])[0]

        results = self.store.search(q_emb, k=k)
        # Convert FAISS distances (L2) to similarity-like scores if desired
        for r in results:
            r["similarity"] = 1.0 / (1.0 + r.get("score", 1.0))
        return results


if __name__ == "__main__":
    vr = VisualRetriever()
    print("VisualRetriever ready. FAISS index entries:", len(vr.store.texts))
