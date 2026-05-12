import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_google_vertexai import VertexAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
import numpy as np

from app.config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, EMBEDDING_MODEL, GCP_PROJECT_ID, GCP_LOCATION
from app.retriever.reranker import ReRanker

class BM25Retriever:
    """
    Sparse retriever using BM25.
    """
    def __init__(self, documents):
        self.documents = documents
        self.corpus = [doc["text"] for doc in documents]
        self.tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query, k=5, metadata_filter=None):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get all candidates that match the filter first
        if metadata_filter:
            filtered_indices = []
            for i, doc in enumerate(self.documents):
                match = True
                for key, val in metadata_filter.items():
                    doc_val = doc["metadata"].get(key)
                    if isinstance(val, dict):
                        if "gte" in val and not (doc_val >= val["gte"]): match = False
                        if "lte" in val and not (doc_val <= val["lte"]): match = False
                    elif doc_val != val:
                        match = False
                if match:
                    filtered_indices.append(i)
            
            # Only score the filtered documents
            if not filtered_indices:
                return []
            
            filtered_scores = [(i, scores[i]) for i in filtered_indices if scores[i] > 0]
            filtered_scores.sort(key=lambda x: x[1], reverse=True)
            top_n = [x[0] for x in filtered_scores[:k]]
        else:
            top_n = np.argsort(scores)[::-1][:k]
        
        results = []
        for i in top_n:
            if scores[i] > 0:
                doc = self.documents[i].copy()
                doc["score"] = float(scores[i])
                results.append(doc)
        return results

def reciprocal_rank_fusion(results_list, k=60):
    """
    Combines multiple ranked lists using Reciprocal Rank Fusion.
    """
    fused_scores = {}
    for results in results_list:
        for rank, doc in enumerate(results):
            doc_id = doc["text"] # Using text as ID for deduplication
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"doc": doc, "score": 0.0}
            fused_scores[doc_id]["score"] += 1.0 / (rank + k)
    
    # Sort and return
    fused_results = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in fused_results]

class CombinedRetriever:
    """
    Combines search results from multiple vector stores into a single ranked list.
    """
    def __init__(self, vectorstores, embedder, generator=None, use_reranker=True):
        self.vectorstores = vectorstores
        self.embedder = embedder
        self.generator = generator
        self.reranker = ReRanker() if use_reranker else None
        self.retrievers = [Retriever(store, embedder, generator=generator, use_reranker=False) for store in vectorstores]

    def retrieve(self, query, k=5, threshold=None, hybrid=True, rerank=True, metadata_filter=None):
        results_list = []
        for retriever in self.retrievers:
            results = retriever.retrieve(
                query,
                k=k,
                threshold=threshold,
                hybrid=hybrid,
                rerank=False,
                metadata_filter=metadata_filter,
            )
            results_list.append(results)

        fused_results = reciprocal_rank_fusion(results_list, k=k)

        if rerank and self.reranker and fused_results:
            fused_results = self.reranker.rerank(query, fused_results, top_n=k)

        return fused_results[:k]

class LangChainRetriever:
    """
    Retriever implementation using LangChain's built-in Qdrant integration.
    Strictly uses Vertex AI Enterprise backend.
    """
    def __init__(self, collection_name=QDRANT_COLLECTION):
        self.embeddings = VertexAIEmbeddings(
            model_name=EMBEDDING_MODEL,
            project=GCP_PROJECT_ID,
            location=GCP_LOCATION
        )
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.6}
        )

    def retrieve(self, query):
        docs = self.retriever.invoke(query)
        return [
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
                # LangChain doesn't always expose score in the same way, 
                # but we can get it if needed.
                "score": 0.0 
            } for doc in docs
        ]

class Retriever:
    """
    Advanced Retriever that handles semantic search, 
    multi-query expansion, and result filtering.
    """
    def __init__(self, vectorstore, embedder, generator=None, use_reranker=True):
        self.vectorstore = vectorstore
        self.embedder = embedder
        self.generator = generator
        self.reranker = ReRanker() if use_reranker else None
        self.bm25_retriever = None
        
        # Lazy initialization of BM25 if supported by vectorstore
        if hasattr(self.vectorstore, "get_all_documents"):
            print("[*] Initializing BM25 index from vector store...")
            all_docs = self.vectorstore.get_all_documents()
            if all_docs:
                self.bm25_retriever = BM25Retriever(all_docs)

    def retrieve(self, query, k=5, threshold=None, hybrid=True, rerank=True, metadata_filter=None):
        """
        Retrieves relevant documents using semantic search, hybrid search, 
        and optional re-ranking.
        """
        # Auto-detect threshold if not provided
        if threshold is None:
            from app.vectorstores.faiss_store import FAISSStore
            if isinstance(self.vectorstore, FAISSStore):
                threshold = -1000.0 # Default Distance Threshold
            else:
                threshold = 0.5 # Default Similarity Threshold

        # 1. Vector Search (Dense)
        query_embedding = self.embedder.embed([query])[0]
        # Some vectorstore implementations (e.g., FAISSStore) do not accept
        # a `metadata_filter` kwarg. Try calling with the filter first,
        # fall back to calling without it and then apply client-side filtering.
        try:
            dense_results = self.vectorstore.search(query_embedding, k=k*2, metadata_filter=metadata_filter)
        except TypeError:
            # Fallback: call without metadata_filter and then filter results
            dense_results = self.vectorstore.search(query_embedding, k=k*2)
            if metadata_filter:
                dense_results = self._apply_metadata_filter(dense_results, metadata_filter)
        
        # 2. BM25 Search (Sparse)
        if hybrid and self.bm25_retriever:
            sparse_results = self.bm25_retriever.retrieve(query, k=k*2, metadata_filter=metadata_filter)
            # Combine using RRF
            results = reciprocal_rank_fusion([dense_results, sparse_results])
        else:
            results = dense_results

        # 3. Deduplicate and Filter
        unique_results = self._deduplicate_and_filter(results, threshold)
        
        # 4. Re-ranking
        if rerank and self.reranker and unique_results:
            # print(f"[*] Re-ranking {len(unique_results)} candidates...")
            unique_results = self.reranker.rerank(query, unique_results, top_n=k)
        
        return unique_results[:k]

    def _apply_metadata_filter(self, results, metadata_filter):
        if not metadata_filter:
            return results

        filtered = []
        for r in results:
            meta = r.get("metadata", {}) or {}
            match = True
            for key, val in metadata_filter.items():
                doc_val = meta.get(key)
                if isinstance(val, dict):
                    if "gte" in val and not (doc_val >= val["gte"]):
                        match = False
                        break
                    if "lte" in val and not (doc_val <= val["lte"]):
                        match = False
                        break
                else:
                    if doc_val != val:
                        match = False
                        break
            if match:
                filtered.append(r)
        return filtered

    def _deduplicate_and_filter(self, results, threshold):
        seen_ids = set()
        filtered = []
        
        for res in results:
            cid = res["metadata"].get("chunk_id", hash(res["text"]))
            
            # Logic: 
            # If threshold > 0 (Similarity): keep if score >= threshold
            # If threshold < 0 (Distance): keep if score <= abs(threshold)
            if threshold >= 0:
                is_valid = res["score"] >= threshold
            else:
                is_valid = res["score"] <= abs(threshold)

            if cid not in seen_ids and is_valid:
                seen_ids.add(cid)
                filtered.append(res)
                
        return filtered

if __name__ == "__main__":
    from app.embeddings.embedder import Embedder
    from app.vectorstores.qdrant_store import QdrantStore
    import os

    query = "What is IFC's net income in 2024?"
    print(f"Query: {query}")

    # 1. Test Custom Retriever
    print("\n--- [1] Testing Custom Retriever ---")
    embedder = Embedder()
    vectorstore = QdrantStore()
    retriever = Retriever(vectorstore, embedder)
    results = retriever.retrieve(query, k=2, min_score=0.7)
    for i, res in enumerate(results):
        print(f"Result {i+1} (Score: {res['score']:.4f}): {res['text'][:100]}...")

    # 2. Test LangChain Retriever
    print("\n--- [2] Testing LangChain Retriever ---")
    lc_retriever = LangChainRetriever()
    lc_results = lc_retriever.retrieve(query)
    for i, res in enumerate(lc_results):
        print(f"Result {i+1}: {res['text'][:100]}...")