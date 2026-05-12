from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


def _metadata_matches(doc_meta: Dict[str, Any], metadata_filter: Optional[Dict[str, Any]]) -> bool:
    if not metadata_filter:
        return True

    for key, expected in metadata_filter.items():
        actual = doc_meta.get(key)
        if isinstance(expected, dict):
            if "gte" in expected and not (actual is not None and actual >= expected["gte"]):
                return False
            if "lte" in expected and not (actual is not None and actual <= expected["lte"]):
                return False
        else:
            if actual != expected:
                return False
    return True


def _safe_search(vectorstore: Any, query_embedding: Sequence[float], *, k: int, metadata_filter: Optional[Dict[str, Any]] = None):
    """Call vectorstore.search with/without metadata_filter depending on backend support."""
    try:
        return vectorstore.search(query_embedding, k=k, metadata_filter=metadata_filter)
    except TypeError:
        # Backend doesn't accept metadata_filter, so do best-effort client-side filtering.
        results = vectorstore.search(query_embedding, k=k)
        if not metadata_filter:
            return results
        return [r for r in results if _metadata_matches(r.get("metadata", {}), metadata_filter)]


def _dedup_keep_order(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in results:
        meta = r.get("metadata", {}) or {}
        doc_id = meta.get("chunk_id") or r.get("text")
        if doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(r)
    return out


def _rrf_fuse(results_list: List[List[Dict[str, Any]]], *, rrf_k: int = 60) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion on rank position only (ignores raw scores)."""
    fused: Dict[str, Dict[str, Any]] = {}
    for results in results_list:
        for rank, doc in enumerate(results, start=1):
            meta = doc.get("metadata", {}) or {}
            doc_id = meta.get("chunk_id") or doc.get("text")
            if not doc_id:
                continue
            entry = fused.get(doc_id)
            if entry is None:
                fused[doc_id] = {"doc": doc, "score": 0.0}
                entry = fused[doc_id]
            entry["score"] += 1.0 / (rrf_k + rank)

    ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
    out = [e["doc"] for e in ranked]
    return _dedup_keep_order(out)


@dataclass(frozen=True)
class MultimodalRetrievalConfig:
    k_text: int = 3
    k_table: int = 3
    k_image: int = 0
    fetch_multiplier: int = 6
    rrf_k: int = 60
    use_reranker: bool = False


class MultimodalRetriever:
    """Integrated retriever over BOTH text chunks and table chunks.

    Assumes ingestion stored:
    - `content_type`: "text" or "table" or "image"
    """

    def __init__(self, vectorstore: Any, embedder: Any, *, config: Optional[MultimodalRetrievalConfig] = None):
        self.vectorstore = vectorstore
        self.embedder = embedder
        self.config = config or MultimodalRetrievalConfig()
        self.reranker = None
        if self.config.use_reranker:
            from app.retriever.reranker import ReRanker

            self.reranker = ReRanker()

    def retrieve(
        self,
        query: str,
        *,
        k: int = 5,
        threshold: Optional[float] = None,
        hybrid: bool = False,  # kept for signature compatibility; unused
        rerank: bool = True,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        # 1) Embed query
        query_embedding = self.embedder.embed([query])[0]

        # 2) Retrieve separately from text and table chunks
        base_fetch = max(k * self.config.fetch_multiplier, k)
        fetch_text = max(self.config.k_text * self.config.fetch_multiplier, base_fetch)
        fetch_table = max(self.config.k_table * self.config.fetch_multiplier, base_fetch)
        fetch_image = max(self.config.k_image * self.config.fetch_multiplier, base_fetch)

        text_filter = {"content_type": "text"}
        table_filter = {"content_type": "table"}
        image_filter = {"content_type": "image"}
        if metadata_filter:
            # Merge: user-provided keys win if overlapping (rare).
            text_filter = {**text_filter, **metadata_filter}
            table_filter = {**table_filter, **metadata_filter}
            image_filter = {**image_filter, **metadata_filter}

        text_results = _safe_search(self.vectorstore, query_embedding, k=fetch_text, metadata_filter=text_filter)
        table_results = _safe_search(self.vectorstore, query_embedding, k=fetch_table, metadata_filter=table_filter)
        image_results: List[Dict[str, Any]] = []
        if self.config.k_image > 0:
            image_results = _safe_search(
                self.vectorstore, query_embedding, k=fetch_image, metadata_filter=image_filter
            )

        # 3) Apply threshold (best-effort)
        if threshold is not None:
            def passes(res: Dict[str, Any]) -> bool:
                score = res.get("score")
                if score is None:
                    return True
                # Heuristic: if threshold < 0 treat as max distance; else as min similarity
                if threshold < 0:
                    return score <= abs(threshold)
                return score >= threshold

            text_results = [r for r in text_results if passes(r)]
            table_results = [r for r in table_results if passes(r)]
            image_results = [r for r in image_results if passes(r)]

        # 4) Fuse
        lists = [text_results, table_results]
        if image_results:
            lists.append(image_results)
        fused = _rrf_fuse(lists, rrf_k=self.config.rrf_k)

        # 5) Optional rerank
        if rerank and self.reranker and fused:
            fused = self.reranker.rerank(query, fused, top_n=k)

        return fused[:k]


if __name__ == "__main__":
    # Minimal smoke test with dummy store/embedder.
    class DummyEmbedder:
        def embed(self, texts):
            return [[0.0, 0.0, 0.0] for _ in texts]

    class DummyStore:
        def search(self, query_embedding, k=5, metadata_filter=None):
            docs = [
                {"text": "text chunk about net income", "metadata": {"chunk_id": "t1", "content_type": "text"}, "score": 0.9},
                {"text": "table rows: net income 2024 = 1485", "metadata": {"chunk_id": "tb1", "content_type": "table"}, "score": 0.92},
            ]
            if metadata_filter:
                docs = [d for d in docs if _metadata_matches(d["metadata"], metadata_filter)]
            return docs[:k]

    retriever = MultimodalRetriever(DummyStore(), DummyEmbedder())
    print(retriever.retrieve("What is net income in 2024?", k=2))
