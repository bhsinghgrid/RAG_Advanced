from app.observability.langfuse_tracer import get_langfuse

class RAGPipeline:
    def __init__(self, retriever, generator, semantic_cache=None, cache_threshold=0.85):
        self.retriever = retriever
        self.generator = generator
        self.semantic_cache = semantic_cache
        self.cache_threshold = cache_threshold
        self.langfuse = get_langfuse()

    def run(self, query, **kwargs):
        # 0. Tracing Initialization (Safe backend detection)
        backend_name = "Unknown"
        if hasattr(self.retriever, "vectorstore") and self.retriever.vectorstore:
            backend_name = self.retriever.vectorstore.__class__.__name__
        else:
            backend_name = self.retriever.__class__.__name__

        cache_hit = None
        if self.semantic_cache:
            cache_hit = self.semantic_cache.lookup(query, threshold=self.cache_threshold)
            if cache_hit:
                trace = self.langfuse.trace(
                    name="rag-pipeline",
                    input=query,
                    metadata={
                        "backend": backend_name,
                        "cache_hit": True,
                        "params": kwargs
                    }
                )
                trace.update(output=cache_hit["answer"])
                yield cache_hit["answer"]
                return

        trace = self.langfuse.trace(
            name="rag-pipeline",
            input=query,
            metadata={
                "backend": backend_name,
                "cache_hit": False,
                "params": kwargs
            }
        )
        
        # 2. Retrieval Step
        span_retrieval = trace.span(name="retrieval", input=query)
        contexts = self.retriever.retrieve(query, **kwargs)
        span_retrieval.end(output=contexts)

        # 3. Generation Step (Streaming)
        full_response = ""
        span_generation = trace.span(name="generation", input=query)
        for chunk in self.generator.generate(query, contexts):
            full_response += chunk
            yield chunk
        
        span_generation.end(output=full_response)

        if self.semantic_cache:
            context_texts = [c["text"] for c in contexts]
            self.semantic_cache.save(query, full_response, context_texts)
        
        # 4. Finalize Trace
        trace.update(output=full_response)

if __name__ == "__main__":
    # Mock classes for testing
    class MockStore: pass
    class MockRetriever:
        def __init__(self): self.vectorstore = MockStore()
        def retrieve(self, q): return [{"text": "Sample context about IFC."}]
    class MockGenerator:
        def generate(self, q, c): yield "This is a mock answer."
        
    pipeline = RAGPipeline(MockRetriever(), MockGenerator())
    print("Running pipeline with tracing...")
    for chunk in pipeline.run("test query"):
        print(chunk, end="")
    print("\nTrace sent to Langfuse.")