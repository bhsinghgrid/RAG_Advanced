from app.embeddings.embedder import Embedder
from app.vectorstores.qdrant_store import QdrantStore
from app.vectorstores.faiss_store import FAISSStore
from app.retriever.retriever import Retriever
from app.llm.generator import Generator
from app.pipelines.rag_pipeline import RAGPipeline
from app.pipelines.ingestion_pipeline import run_ingestion
from app.config import PDF_PATH, VECTOR_DIM

def main():
    print("--- Initializing RAG System (Dual Backend) ---")
    embedder = Embedder()
    
    # Initialize both stores
    q_store = QdrantStore()
    f_store = FAISSStore(dim=VECTOR_DIM, save_path="faiss_index")
    vectorstores = [q_store, f_store]

    print(f"--- Running Ingestion for {PDF_PATH} ---")
    num_chunks = run_ingestion(PDF_PATH, embedder, vectorstores)
    print(f"Done. Ingested {num_chunks} chunks into both Qdrant and FAISS.")

    retriever = Retriever(q_store, embedder)
    generator = Generator()
    pipeline = RAGPipeline(retriever, generator)

    query = "What are the main financial highlights for 2024?"
    print(f"\n--- Querying: {query} ---")
    print("Answer: ", end="")
    for chunk in pipeline.run(query):
        print(chunk, end="", flush=True)
    print("\n\n--- System Ready ---")

if __name__ == "__main__":
    main()