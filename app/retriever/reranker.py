from sentence_transformers import CrossEncoder
import numpy as np

class ReRanker:
    """
    Re-ranks retrieved documents using a Cross-Encoder model 
    for higher accuracy.
    """
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"[*] Loading Re-ranker model: {model_name}...")
        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents, top_n=5):
        """
        Re-ranks a list of documents based on the query.
        """
        if not documents:
            return []

        # Prepare pairs: (query, doc_text)
        pairs = [[query, doc["text"]] for doc in documents]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Attach scores to documents
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        # Sort by rerank_score in descending order
        reranked_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        
        return reranked_docs[:top_n]

if __name__ == "__main__":
    # Test
    query = "What is the capital of France?"
    docs = [
        {"text": "Berlin is the capital of Germany."},
        {"text": "Paris is the capital of France."},
        {"text": "The Eiffel Tower is in Paris."}
    ]
    
    reranker = ReRanker()
    results = reranker.rerank(query, docs)
    for res in results:
        print(f"Score: {res['rerank_score']:.4f} | Text: {res['text']}")
