import os
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from app.embeddings.embedder import Embedder
from app.vectorstores.qdrant_store import QdrantStore
from app.vectorstores.faiss_store import FAISSStore
from app.retriever.retriever import CombinedRetriever
from app.retriever.multi_hop_retriever import MultiHopRetriever
from app.llm.generator import Generator
from app.pipelines.rag_pipeline import RAGPipeline
from app.cache.semantic_cache import SemanticCache
from app.config import GCP_PROJECT_ID, GCP_LOCATION, LLM_MODEL, VECTOR_DIM
import vertexai
from vertexai.generative_models import GenerativeModel
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

# 1. Configuration & Setup
DATASET_PATH = "RAG_evaluation_dataset - convertcsv (2).csv"
OUTPUT_PATH = "evaluation_results_phase4.csv"


def get_rag_response(pipeline, query):
    full_answer = ""
    contexts = pipeline.retriever.retrieve(query)
    for chunk in pipeline.run(query):
        full_answer += chunk

    context_texts = [c["text"] for c in contexts]
    return full_answer, context_texts


def llm_as_judge_score(query, answer, ground_truth):
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
    model = GenerativeModel(LLM_MODEL)

    prompt = f"""
    You are an expert evaluator of RAG systems. Rate the following answer compared to the ground truth on a scale of 1 to 5.

    Criteria:
    1: Completely wrong or irrelevant.
    2: Partially correct but missing key details.
    3: Correct but poorly structured or slightly vague.
    4: Correct and clear, matching ground truth closely.
    5: Exceptional clarity, nuance, and accuracy.

    Question: {query}
    Ground Truth: {ground_truth}
    Generated Answer: {answer}

    Return ONLY a single integer from 1 to 5.
    Score:
    """

    try:
        response = model.generate_content(prompt)
        score_text = response.text.strip()
        return int(score_text[0])
    except Exception as e:
        print(f"Error in LLM Judge: {e}")
        return None


def main():
    print(f"--- Loading Dataset from {DATASET_PATH} ---")
    df = pd.read_csv(DATASET_PATH)

    print("--- Initializing RAG Pipeline (Phase 4) ---")
    embedder = Embedder()
    q_store = QdrantStore()
    f_store = FAISSStore(dim=VECTOR_DIM, save_path="faiss_index")
    combined_retriever = CombinedRetriever([q_store, f_store], embedder)
    multi_hop_retriever = MultiHopRetriever(combined_retriever)
    generator = Generator()
    semantic_cache = SemanticCache()
    pipeline = RAGPipeline(multi_hop_retriever, generator, semantic_cache=semantic_cache)

    eval_llm = ChatVertexAI(model_name=LLM_MODEL, project=GCP_PROJECT_ID, location=GCP_LOCATION)
    eval_embeddings = VertexAIEmbeddings(model_name="text-embedding-004", project=GCP_PROJECT_ID, location=GCP_LOCATION)

    ragas_llm = LangchainLLMWrapper(eval_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(eval_embeddings)

    results = []

    print(f"--- Running Pipeline for {len(df)} questions ---")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        question = row['Question']
        ground_truth = row['Ground_Truth_Answer']

        answer, contexts = get_rag_response(pipeline, question)
        judge_score = llm_as_judge_score(question, answer, ground_truth)

        results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
            "llm_judge_score": judge_score,
        })

    print("--- Running RAGAS Evaluation ---")
    eval_df = pd.DataFrame(results)
    dataset = Dataset.from_pandas(eval_df[['question', 'answer', 'contexts', 'ground_truth']])

    ragas_result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    print("\n--- RAGAS Results ---")
    print(ragas_result)

    ragas_scores_df = ragas_result.to_pandas()
    final_df = pd.concat([eval_df, ragas_scores_df.drop(columns=['question', 'answer', 'contexts', 'ground_truth'], errors='ignore')], axis=1)

    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n--- Evaluation Complete. Results saved to {OUTPUT_PATH} ---")


if __name__ == "__main__":
    main()
