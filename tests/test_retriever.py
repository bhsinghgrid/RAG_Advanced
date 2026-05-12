def test_retrieval(retriever):
    query = "What is IFC investment strategy?"
    results = retriever.retrieve(query)

    assert len(results) > 0
    assert "IFC" in results[0]["text"]