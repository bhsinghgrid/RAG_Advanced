from google import genai
from app.config import GCP_PROJECT_ID, GCP_LOCATION, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE
import numpy as np
from langchain_core.embeddings import Embeddings

class Embedder(Embeddings):
    def __init__(self, model_name=EMBEDDING_MODEL):
        self.client = genai.Client(
            vertexai=True,
            project=GCP_PROJECT_ID,
            location=GCP_LOCATION
        )
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i : i + EMBEDDING_BATCH_SIZE]
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=batch
            )
            all_embeddings.extend([e.values for e in response.embeddings])
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed(self, texts, batch_size=EMBEDDING_BATCH_SIZE):
        # Legacy support
        return self.embed_documents(texts)

if __name__ == "__main__":
    embedder = Embedder()
    test_text = ["Hello world", "RAG is awesome"]
    vectors = embedder.embed(test_text)
    print(f"Generated {len(vectors)} embeddings.")
    print(f"First vector dimension: {len(vectors[0])}")
    print(f"Embedding for {test_text[0]}: {vectors[0][:10]}...")
    print(f"Embedding for {test_text[1]}: {vectors[1][:10]}...")
    print(f"Dot product of embeddings: {np.dot(vectors[0], vectors[1])}")
    print(f"Cosine similarity of embeddings: {np.dot(vectors[0], vectors[1]) / (np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1]))}")
