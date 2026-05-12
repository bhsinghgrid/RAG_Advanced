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
from app.retriever.retriever import Retriever
from app.llm.generator import Generator
from app.pipelines.rag_pipeline import RAGPipeline
from app.config import GCP_PROJECT_ID, GCP_LOCATION, LLM_MODEL
import vertexai
from vertexai.generative_models import GenerativeModel

# 1. Configuration & Setup
# Ensure GOOGLE_APPLICATION_CREDENTIALS is set in your environment
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/your/key.json"

DATASET_PATH = "RAG_evaluation_dataset - convertcsv (2).csv"
OUTPUT_PATH = "evaluation_results_phase3.csv"

def get_rag_response(pipeline, query):
    """
    Runs the pipeline and collects the full response.
    """
    full_answer = ""
    contexts = pipeline.retriever.retrieve(query, hybrid=True, rerank=True)
    # Note: RAGPipeline.run returns a generator of chunks
    # We enable hybrid search and reranking for Phase 3
    for chunk in pipeline.run(query, hybrid=True, rerank=True):
        full_answer += chunk
    
    # Ragas expects contexts as a list of strings
    context_texts = [c["text"] for c in contexts]
    return full_answer, context_texts

def llm_as_judge_score(query, answer, ground_truth):
    """
    Custom LLM-based evaluation for nuance and clarity.
    """
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
        return int(score_text[0]) # Extract first digit
    except Exception as e:
        print(f"Error in LLM Judge: {e}")
        return None

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

def main():
    print(f"--- Loading Dataset from {DATASET_PATH} ---")
    df = pd.read_csv(DATASET_PATH)
    
    # Initialize RAG Components
    print("--- Initializing RAG Pipeline ---")
    embedder = Embedder()
    vectorstore = QdrantStore()
    retriever = Retriever(vectorstore, embedder)
    generator = Generator()
    pipeline = RAGPipeline(retriever, generator)
    
    # Initialize Ragas LLM and Embeddings for evaluation
    # We use ChatVertexAI for evaluation as it's better for judging
    eval_llm = ChatVertexAI(model_name=LLM_MODEL, project=GCP_PROJECT_ID, location=GCP_LOCATION)
    eval_embeddings = VertexAIEmbeddings(model_name="text-embedding-004", project=GCP_PROJECT_ID, location=GCP_LOCATION)
    
    ragas_llm = LangchainLLMWrapper(eval_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(eval_embeddings)
    
    results = []
    
    print(f"--- Running Pipeline for {len(df)} questions ---")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        question = row['Question']
        ground_truth = row['Ground_Truth_Answer']
        
        # Generate Answer
        answer, contexts = get_rag_response(pipeline, question)
        
        # LLM as Judge Score
        judge_score = llm_as_judge_score(question, answer, ground_truth)
        
        results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
            "llm_judge_score": judge_score
        })
    
    # Prepare for RAGAS
    print("--- Running RAGAS Evaluation ---")
    eval_df = pd.DataFrame(results)
    dataset = Dataset.from_pandas(eval_df[['question', 'answer', 'contexts', 'ground_truth']])
    
    # Run evaluation with Vertex AI backends
    ragas_result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )
    
    print("\n--- RAGAS Results ---")
    print(ragas_result)
    
    # Merge RAGAS scores back to our results
    ragas_scores_df = ragas_result.to_pandas()
    final_df = pd.concat([eval_df, ragas_scores_df.drop(columns=['question', 'answer', 'contexts', 'ground_truth'], errors='ignore')], axis=1)
    
    # Save Results
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n--- Evaluation Complete. Results saved to {OUTPUT_PATH} ---")

if __name__ == "__main__":
    main()
