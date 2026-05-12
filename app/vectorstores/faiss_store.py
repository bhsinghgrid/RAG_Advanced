import faiss
import numpy as np
import pickle
import os

class FAISSStore:
    def __init__(self, dim=768, save_path="faiss_index"):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []
        self.metadata = []
        self.save_path = save_path
        
        # Try to load if exists
        if os.path.exists(f"{self.save_path}.index"):
            self.load()

    def add(self, embeddings, texts, metadatas):
        self.index.add(np.array(embeddings).astype("float32"))
        self.texts.extend(texts)
        self.metadata.extend(metadatas)
        self.save() # Auto-save on add

    def search(self, query_embedding, k=5):
        if self.index.ntotal == 0:
            return []
            
        D, I = self.index.search(
            np.array([query_embedding]).astype("float32"), k
        )

        results = []
        for idx, i in enumerate(I[0]):
            if i != -1:
                results.append({
                    "text": self.texts[i],
                    "metadata": self.metadata[i],
                    "score": float(D[0][idx])
                })
        return results

    def save(self):
        faiss.write_index(self.index, f"{self.save_path}.index")
        with open(f"{self.save_path}.pkl", "wb") as f:
            pickle.dump({"texts": self.texts, "metadata": self.metadata}, f)

    def load(self):
        self.index = faiss.read_index(f"{self.save_path}.index")
        with open(f"{self.save_path}.pkl", "rb") as f:
            data = pickle.load(f)
            self.texts = data["texts"]
            self.metadata = data["metadata"]

if __name__ == "__main__":
    store = FAISSStore(dim=4, save_path="test_faiss")
    embeddings = [[1, 0, 0, 0], [0, 1, 0, 0]]
    store.add(embeddings, ["Text 1", "Text 2"], [{"id": 1}, {"id": 2}])
    print("Saved and loaded test successful.")