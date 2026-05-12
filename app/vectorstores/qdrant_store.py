import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, VECTOR_DIM

class QdrantStore:
    def __init__(self, collection_name=QDRANT_COLLECTION):
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            print(f"[*] Creating Qdrant collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=VECTOR_DIM, 
                    distance=models.Distance.COSINE
                )
            )

    def add(self, embeddings, texts, metadatas):
        points = []
        for i, (emb, text, meta) in enumerate(zip(embeddings, texts, metadatas)):
            # Combine text into metadata (payload)
            payload = {**meta, "text": text}
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb,
                    payload=payload
                )
            )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query_embedding, k=5, metadata_filter=None):
        query_filter = None
        if metadata_filter:
            # Simple conversion of dict filter to Qdrant filter
            # Example: {"page": 1} -> FieldCondition(key="page", match=MatchValue(value=1))
            conditions = []
            for key, value in metadata_filter.items():
                if isinstance(value, dict):
                    # Handle ranges if needed (e.g., {"page": {"gte": 1, "lte": 5}})
                    if "gte" in value or "lte" in value:
                        conditions.append(models.FieldCondition(
                            key=key, 
                            range=models.Range(gte=value.get("gte"), lte=value.get("lte"))
                        ))
                else:
                    conditions.append(models.FieldCondition(
                        key=key, 
                        match=models.MatchValue(value=value)
                    ))
            
            if conditions:
                query_filter = models.Filter(must=conditions)

        res = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=k,
            query_filter=query_filter,
            with_payload=True
        )
        
        results = []
        for hit in res.points:
            results.append({
                "text": hit.payload.get("text", ""),
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
                "score": hit.score
            })
            
        return results

    def get_all_documents(self):
        """
        Retrieves all documents from the collection for BM25 initialization.
        """
        all_docs = []
        next_page_offset = None
        
        while True:
            res, next_page_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=next_page_offset,
                with_payload=True,
                with_vectors=False
            )
            
            for hit in res:
                all_docs.append({
                    "text": hit.payload.get("text", ""),
                    "metadata": {k: v for k, v in hit.payload.items() if k != "text"}
                })
            
            if next_page_offset is None:
                break
                
        return all_docs

if __name__ == "__main__":
    # Test block
    store = QdrantStore(collection_name="test_collection")
    emb = [0.1] * VECTOR_DIM
    store.add([emb], ["Test text"], [{"page": 0}])
    results = store.search(emb, k=1)
    print(f"Qdrant Search Result: {results}")
