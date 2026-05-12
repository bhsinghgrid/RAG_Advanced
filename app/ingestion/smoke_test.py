from app.pipelines.ingestion_pipeline import run_ingestion
from app.embeddings.embedder import Embedder
from app.vectorstores.faiss_store import FAISSStore
from app.config import PDF_PATH, VECTOR_DIM
import os

def test_ingestion():
    if not os.path.exists(PDF_PATH):
        print(f"PDF not found at {PDF_PATH}")
        return

    embedder = Embedder()
    # Use a temporary FAISS index for testing
    f_store = FAISSStore(dim=VECTOR_DIM, save_path="data/test_faiss_index")
    
    print(f"[*] Starting smoke test for {PDF_PATH}...")
    try:
        num_chunks = run_ingestion(PDF_PATH, embedder, [f_store])
        print(f"[✅] Success! Ingested {num_chunks} chunks.")
    except Exception as e:
        print(f"[❌] Ingestion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ingestion()
