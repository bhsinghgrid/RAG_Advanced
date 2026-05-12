from langchain_google_vertexai import VertexAI
from langchain_core.prompts import PromptTemplate

from app.config import GCP_PROJECT_ID, GCP_LOCATION, LLM_MODEL


class MultiHopRetriever:
    """Performs a second retrieval pass after refining the query with an LLM."""

    def __init__(self, base_retriever, llm_model=LLM_MODEL):
        self.base_retriever = base_retriever
        self.llm = VertexAI(model_name=llm_model, project=GCP_PROJECT_ID, location=GCP_LOCATION)
        self.prompt_template = (
            "You are a retrieval assistant. A user asked: {query}\n"
            "You retrieved these initial contexts:\n{contexts}\n\n"
            "Based on the user question and the retrieved paragraphs, create a refined follow-up query that would retrieve more specific or missing information."
            "Return only the refined query text."
        )
        # Expose commonly-used attributes from the wrapped retriever so
        # external code (e.g., RAGPipeline) can inspect them.
        # This keeps the MultiHopRetriever as a transparent wrapper.
        self.vectorstore = getattr(base_retriever, "vectorstore", None)
        self.embedder = getattr(base_retriever, "embedder", None)

    def refine_query(self, query, contexts):
        if not contexts:
            return query

        contexts_text = "\n\n".join(contexts)
        prompt_text = self.prompt_template.format(query=query, contexts=contexts_text)
        refined = self.llm.invoke(prompt_text)
        return refined.strip()

    def retrieve(self, query, k=5, threshold=None, hybrid=True, rerank=True, metadata_filter=None):
        # First pass retrieval
        first_pass = self.base_retriever.retrieve(
            query,
            k=k,
            threshold=threshold,
            hybrid=hybrid,
            rerank=rerank,
            metadata_filter=metadata_filter,
        )

        if not first_pass:
            return first_pass

        context_texts = [item["text"] for item in first_pass[:3]]
        refined_query = self.refine_query(query, context_texts)

        if not refined_query or refined_query.lower() == query.lower():
            return first_pass

        second_pass = self.base_retriever.retrieve(
            refined_query,
            k=k,
            threshold=threshold,
            hybrid=hybrid,
            rerank=rerank,
            metadata_filter=metadata_filter,
        )

        merged = first_pass + [doc for doc in second_pass if doc["text"] not in {d["text"] for d in first_pass}]
        return merged[:k]


if __name__ == "__main__":
    class DummyRetriever:
        def retrieve(self, query, k=5, threshold=None, hybrid=True, rerank=True, metadata_filter=None):
            return [
                {"text": "IFC's 2024 financial outlook is focused on sustainable investments.", "metadata": {}, "score": 0.92},
                {"text": "The annual report highlights strong growth in renewable energy financing.", "metadata": {}, "score": 0.87},
            ]

    print("--- MultiHopRetriever Test ---")
    base_retriever = DummyRetriever()
    multi_hop = MultiHopRetriever(base_retriever)
    query = "What is IFC planning for sustainable finance in 2024?"

    print(f"Query: {query}")
    results = multi_hop.retrieve(query, k=3)

    print("\nRetrieved Results:")
    for i, doc in enumerate(results, start=1):
        print(f"{i}. {doc['text']}")
