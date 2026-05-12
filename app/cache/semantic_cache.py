from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from app.config import (
    GCP_LOCATION,
    GCP_PROJECT_ID,
    EMBEDDING_MODEL,
    QDRANT_CACHE_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
    VECTOR_DIM,
)


class SemanticCache:
    def __init__(self, collection_name=QDRANT_CACHE_COLLECTION, embedding_model=EMBEDDING_MODEL):
        self.collection_name = collection_name
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.embeddings = VertexAIEmbeddings(
            model_name=embedding_model,
            project=GCP_PROJECT_ID,
            location=GCP_LOCATION,
        )
        self._ensure_collection()
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=VECTOR_DIM,
                    distance=models.Distance.COSINE,
                ),
            )

    def lookup(self, query, threshold=0.8, k=1):
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        if not results:
            return None

        document, score = results[0]
        if score < threshold:
            return None

        return {
            "query": document.page_content,
            "answer": document.metadata.get("answer"),
            "contexts": document.metadata.get("contexts", []),
            "score": score,
        }

    def save(self, query, answer, contexts, metadata=None):
        payload = {"answer": answer, "contexts": contexts}
        if metadata:
            payload.update(metadata)

        document = Document(page_content=query, metadata=payload)
        self.vectorstore.add_documents([document])
