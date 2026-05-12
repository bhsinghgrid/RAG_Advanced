from app.embeddings.embedder import Embedder
from app.vectorstores.qdrant_store import QdrantStore
from app.retriever.retriever import Retriever
from app.config import VECTOR_DIM

def test_filtering():
    print("--- Testing Metadata Filtering ---")
    embedder = Embedder()
    vectorstore = QdrantStore()
    retriever = Retriever(vectorstore, embedder)
    
    query = "What are the financial highlights?"
    
    # Test 1: No filter
    print("\n[1] No Filter:")
    results = retriever.retrieve(query, k=3, rerank=False)
    for r in results:
        print(f"Page: {r['metadata'].get('page')} | Text: {r['text'][:50]}...")
        
    # Test 2: Filter by Page 4
    print("\n[2] Filter by Page 4 only:")
    results = retriever.retrieve(query, k=3, rerank=False, metadata_filter={"page": 4})
    for r in results:
        print(f"Page: {r['metadata'].get('page')} | Text: {r['text'][:50]}...")

    # Test 3: Range Filter (Pages 1-5)
    print("\n[3] Filter by Pages 1-5:")
    results = retriever.retrieve(query, k=3, rerank=False, metadata_filter={"page": {"gte": 1, "lte": 5}})
    for r in results:
        print(f"Page: {r['metadata'].get('page')} | Text: {r['text'][:50]}...")

if __name__ == "__main__":
    test_filtering()
