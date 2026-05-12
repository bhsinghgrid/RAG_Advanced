import streamlit as st
import os
import sys
import pandas as pd

# Add the project root to sys.path to allow importing from 'app'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.embeddings.embedder import Embedder
from app.vectorstores.qdrant_store import QdrantStore
from app.vectorstores.faiss_store import FAISSStore
from app.retriever.retriever import Retriever, CombinedRetriever, reciprocal_rank_fusion
from app.retriever.multi_hop_retriever import MultiHopRetriever
from app.cache.semantic_cache import SemanticCache
from app.llm.generator import Generator
from app.pipelines.rag_pipeline import RAGPipeline
from app.visualization.plotting import plot_from_csv_string
from app.visualization.attribution import render_bbox_overlay_from_image
from app.config import PDF_PATH, VECTOR_DIM, QDRANT_COLLECTION, GCP_PROJECT_ID, GCP_LOCATION, LLM_MODEL
from phase2_evaluation import llm_as_judge_score
from phase6_colpali_like import (
    build_phase6_index,
    retrieve_patches_phase6,
    annotate_sources,
    answer_with_visual_context,
)
import hashlib
from pathlib import Path

def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]

def _index_present(index_dir: str) -> bool:
    idx = Path(index_dir)
    return (idx / "faiss_pooled.index").exists() and (idx / "faiss_pooled.pkl").exists()

def _get_img_base64(img_path):
    import base64
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Page configuration
st.set_page_config(
    page_title="RAG Database Comparison",
    page_icon="⚖️",
    layout="wide"
)

# Premium UI Enhancements
st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    
    <style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 100%);
        color: #e0e0e0;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header Banner */
    .main-header {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeIn 1.5s ease-in;
    }
    
    .sub-header {
        color: #8892b0;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: rgba(0, 210, 255, 0.5);
    }
    
    /* Search Input Styling */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        padding: 12px 15px !important;
        font-size: 1.1rem !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #00d2ff !important;
        box-shadow: 0 0 10px rgba(0, 210, 255, 0.2) !important;
    }
    
    /* Sidebar Aesthetics */
    [data-testid="stSidebar"] {
        background-color: #0a0a0a;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .sidebar-title {
        color: #00d2ff;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Metric Card Styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #00d2ff !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        opacity: 0.9;
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(0, 210, 255, 0.3);
    }
    
    /* Source Box */
    .source-box {
        background: rgba(0, 210, 255, 0.05);
        border-left: 4px solid #00d2ff;
        padding: 15px;
        border-radius: 0 10px 10px 0;
        margin: 10px 0;
        font-size: 0.9rem;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Hide Default Streamlit Menu/Footer for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_resources():
    embedder = Embedder()
    generator = Generator()
    q_store = QdrantStore()
    f_store = FAISSStore(dim=VECTOR_DIM, save_path="faiss_index")
    # Force reload FAISS
    f_store.load()
    return embedder, generator, q_store, f_store

def get_ragas_scores(question, answer, contexts, ground_truth):
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
        import vertexai

        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        eval_llm = ChatVertexAI(model_name=LLM_MODEL, project=GCP_PROJECT_ID, location=GCP_LOCATION)
        eval_embeddings = VertexAIEmbeddings(model_name="text-embedding-004", project=GCP_PROJECT_ID, location=GCP_LOCATION)

        ragas_llm = LangchainLLMWrapper(eval_llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(eval_embeddings)

        eval_df = pd.DataFrame([{
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth
        }])
        dataset = Dataset.from_pandas(eval_df)
        ragas_result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ragas_llm,
            embeddings=ragas_embeddings
        )
        scores = ragas_result.to_pandas().iloc[0].to_dict()
        return scores
    except Exception as e:
        st.warning(f"RAGAS evaluation unavailable: {e}")
        return None

def render_phase2_scorecard(score, reasoning, phase_name=""):
    """
    Renders a descriptive scorecard for the Phase 2 LLM Judge evaluation.
    """
    if score is None:
        return
    
    score_map = {
        5: {"label": "Exceptional", "color": "#2e7d32", "desc": "Matches ground truth perfectly with nuance."},
        4: {"label": "Strong", "color": "#4caf50", "desc": "Correct and clear, matches ground truth closely."},
        3: {"label": "Satisfactory", "color": "#ff9800", "desc": "Correct but slightly vague or poorly structured."},
        2: {"label": "Weak", "color": "#f57c00", "desc": "Partially correct but missing key details."},
        1: {"label": "Fail", "color": "#d32f2f", "desc": "Completely wrong or irrelevant."}
    }
    
    data = score_map.get(score, {"label": "Unknown", "color": "#757575", "desc": "Evaluation inconclusive."})
    
    st.markdown(f"""
        <div style="background-color: {data['color']}22; border-left: 5px solid {data['color']}; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <h4 style="margin: 0; color: {data['color']};">Phase 2 Score: {score}/5 ({data['label']})</h4>
            <p style="margin: 5px 0 10px 0; font-size: 0.95rem; color: #eee;"><b>Reasoning:</b> {reasoning}</p>
            <p style="margin: 0; font-size: 0.8rem; color: #aaa; font-style: italic;">{data['desc']}</p>
        </div>
    """, unsafe_allow_html=True)

def render_ragas_metrics(scores):
    """
    Renders RAGAS metrics as a grid of progress bars and numbers.
    """
    if not scores:
        return
    
    col1, col2 = st.columns(2)
    
    metrics = [
        ("Faithfulness", "faithfulness", "How well the answer is grounded in the retrieved chunks."),
        ("Answer Relevancy", "answer_relevancy", "How relevant the answer is to the original question."),
        ("Context Precision", "context_precision", "How precise the retrieved chunks are relative to ground truth."),
        ("Context Recall", "context_recall", "How much of the ground truth is covered by the chunks.")
    ]
    
    for i, (label, key, desc) in enumerate(metrics):
        val = scores.get(key, 0.0)
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            st.metric(label, f"{val:.4f}")
            st.progress(min(max(float(val), 0.0), 1.0))
            st.caption(desc)


def run_db_query(name, query, retriever, generator, col, semantic_cache=None, evaluation_enabled=False, ground_truth=None, show_patches=False, visual_extractor=None, enable_plotting=False):
    with col:
        header_class = "qdrant-header" if name == "Qdrant" else "faiss-header"
        pipeline = RAGPipeline(retriever, generator, semantic_cache=semantic_cache)
        
        with st.spinner(f"Querying {name}..."):
            results = retriever.retrieve(query, k=3)
            
            if not results:
                st.error(f"No relevant chunks found in {name} database.")
            else:
                st.write("### 🤖 Answer")
                answer_container = st.empty()
                full_answer = ""
                for chunk in pipeline.run(query):
                    full_answer += chunk
                    answer_container.markdown(f'<div class="glass-card"><div class="answer-text">{full_answer}</div></div>', unsafe_allow_html=True)

                st.write("### 📚 Sources")
                for r in results:
                    meta = r.get("metadata", {}) or {}
                    is_visual = meta.get("is_visual_patch") or meta.get("content_type") == "image"
                    thumb_html = ""
                    if is_visual and visual_extractor is not None and show_patches:
                        try:
                            pdf_path = meta.get('pdf_path')
                            page = meta.get('page')
                            bbox = meta.get('bbox')
                            if pdf_path and page is not None and bbox:
                                page_img = visual_extractor.page_to_image(pdf_path, page)
                                crop = page_img.crop(bbox)
                                from io import BytesIO
                                buf = BytesIO()
                                crop.thumbnail((200, 200))
                                crop.save(buf, format='PNG')
                                import base64
                                data = base64.b64encode(buf.getvalue()).decode('utf-8')
                                thumb_html = f'<div><img src="data:image/png;base64,{data}" style="max-width:200px; border-radius:4px;"/></div>'
                        except Exception:
                            thumb_html = ""

                    source_html = f"""
                    <div class="source-box">
                    <b>Page {meta.get('page', '?')}</b> | Content: {meta.get('content_type', 'text')} | Score: {r.get('score', 0.0):.2f}<br>
                    """
                    
                    if meta.get('content_type') == 'table':
                        st.markdown(source_html, unsafe_allow_html=True)
                        st.markdown(r['text']) # This is the markdown table
                    elif meta.get('content_type') == 'image':
                        st.markdown(source_html, unsafe_allow_html=True)
                        st.info(f"🖼️ Image: {meta.get('image_type', 'unknown')}")
                        st.write(r['text']) # Image summary
                        if st.checkbox("Show structured data", key=f"struct_{abs(hash(r['text']))}"):
                            st.json(meta.get('structured_data', {}))
                    else:
                        source_html += f"<small>{r['text'][:300]}...</small></div>"
                        st.markdown(source_html, unsafe_allow_html=True)

                    if thumb_html:
                        st.markdown(thumb_html, unsafe_allow_html=True)
                    # Provide a button to view the region with overlay for visual patches
                    if is_visual and visual_extractor is not None:
                        try:
                            btn_key = f"show_region_{meta.get('page','?')}_{meta.get('table_index', meta.get('chunk_id',''))}_{abs(hash(r.get('text','')))% (10**8)}"
                            if st.button("Show region", key=btn_key):
                                pdf_path = meta.get('pdf_path')
                                page = meta.get('page')
                                bbox = meta.get('bbox')
                                if pdf_path and page is not None and bbox:
                                    page_img = visual_extractor.page_to_image(pdf_path, page)
                                    uri = render_bbox_overlay_from_image(page_img, tuple(bbox))
                                    st.image(uri, use_column_width=False)
                                else:
                                    st.info("No patch coordinates available for this source.")
                        except Exception:
                            pass
                    # If table CSV present and plotting enabled, render a chart
                    if enable_plotting and meta.get('csv'):
                        try:
                            uri = plot_from_csv_string(meta.get('csv'))
                            if uri:
                                st.image(uri, use_column_width=False)
                        except Exception:
                            pass

                if evaluation_enabled:
                    with st.expander("Evaluation Scores"):
                        if not ground_truth:
                            st.info("Enter ground truth text in the sidebar to compute LLM judge and RAGAS scores.")
                        else:
                            st.write("#### ⚖️ Phase 2: LLM Judge Evaluation")
                            llm_score, reasoning = llm_as_judge_score(query, full_answer, ground_truth)
                            if llm_score is not None:
                                render_phase2_scorecard(llm_score, reasoning)
                                # Store score for cross-DB comparison
                                if 'scores' not in st.session_state:
                                    st.session_state.scores = {}
                                st.session_state.scores[name] = llm_score
                            
                            st.write("#### 📊 Phase 3: RAGAS Metrics")
                            contexts = [r["text"] for r in results]
                            ragas_scores = get_ragas_scores(query, full_answer, contexts, ground_truth)
                            if ragas_scores:
                                render_ragas_metrics(ragas_scores)
                                # Store Ragas avg for comparison
                                avg_ragas = sum(ragas_scores.values()) / len(ragas_scores)
                                if 'ragas_avg' not in st.session_state:
                                    st.session_state.ragas_avg = {}
                                st.session_state.ragas_avg[name] = avg_ragas
                                
                            # Phase-specific insights
                            st.divider()
                            st.caption(f"Evaluation Context: Metrics optimized for current phase.")

def main():
    embedder, generator, q_store, f_store = init_resources()
    
    st.sidebar.markdown('<p class="sidebar-title">⚖️ DB Control Center</p>', unsafe_allow_html=True)
    phase = st.sidebar.radio(
        "Select Active Pipeline:",
        [
            "Phase 1: Retrieval Only",
            "Phase 2: Evaluation Dashboard",
            "Phase 3: Full Evaluation",
            "Phase 4: Advanced RAG",
            "Phase 5.1: Tables",
            "Phase 5.2: Tables + Plotting",
            "Phase 6: Multimodal RAG",
            "All Phases (Show All Results)"
        ],
        index=2
    )
    mode = st.sidebar.radio("Select Mode:", ["Qdrant Only", "FAISS Only", "Compare Both"], index=2)
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"📄 **Document:** {PDF_PATH}")
    st.sidebar.markdown(f"**Selected Phase:** {phase}")
    
    # FAISS Health Check
    if _index_present("faiss_index") or os.path.exists("faiss_index/faiss_pooled.index"):
        st.sidebar.success("✅ FAISS Index: Loaded")
    else:
        st.sidebar.warning("⚠️ FAISS Index: Not Found")
        if st.sidebar.button("Build Text Index (Hybrid)"):
            from app.pipelines.ingestion_pipeline import run_ingestion
            with st.sidebar:
                with st.spinner("Indexing..."):
                    run_ingestion(PDF_PATH, embedder)
            st.rerun()

    eval_enabled = st.sidebar.checkbox("Show evaluation scores", value=False)
    ground_truth = None
    if eval_enabled:
        st.sidebar.write("Provide the expected answer below to compute LLM judge and RAGAS scores.")
        ground_truth = st.sidebar.text_area(
            "Ground truth answer",
            placeholder="Enter the expected answer text for evaluation...",
            height=160
        )

    semantic_cache_enabled = False
    multi_hop_enabled = False
    enable_visual = False
    show_patches = False
    enable_table_retrieval = False
    enable_plotting = False
    if phase == "Phase 4: Advanced RAG":
        semantic_cache_enabled = st.sidebar.checkbox("Enable semantic cache", value=False)
        multi_hop_enabled = st.sidebar.checkbox("Enable multi-hop retrieval", value=True)
        st.sidebar.info("Phase 4 combines advanced retrieval with cache and optional multi-hop refinement.")
    elif phase == "Phase 6: Multimodal RAG":
        st.sidebar.subheader("🧩 Phase 6 (ColPali-like) Settings")
        index_dir = st.sidebar.text_input("Index dir", value="phase6_index")
        encoder = st.sidebar.selectbox("Encoder", ["hash (offline)", "siglip (multimodal)"], index=0)
        encoder_key = "hash" if encoder.startswith("hash") else "siglip"
        
        st.sidebar.markdown("---")
        page_dpi = st.sidebar.number_input("Page DPI", min_value=72, max_value=300, value=150, step=10)
        patch_size = st.sidebar.number_input("Patch size", min_value=128, max_value=1024, value=512, step=64)
        patch_overlap = st.sidebar.number_input("Patch overlap", min_value=0, max_value=512, value=0, step=32)
        max_pages_p6 = st.sidebar.number_input("Max pages (0 = all)", min_value=0, max_value=2000, value=2, step=1)
        proj_dim_p6 = st.sidebar.number_input("proj_dim", min_value=32, max_value=512, value=128, step=32)

        model_name_p6 = "google/siglip-base-patch16-224"
        hf_local_only_p6 = True
        hash_seed_p6 = 0
        hash_max_tokens_p6 = 128

        if encoder_key == "siglip":
            model_name_p6 = st.sidebar.text_input("SigLIP model", value=model_name_p6)
            hf_local_only_p6 = st.sidebar.checkbox("HF local only (no downloads)", value=True)
        else:
            hash_seed_p6 = st.sidebar.number_input("Hash seed", min_value=0, max_value=10_000, value=0, step=1)
            hash_max_tokens_p6 = st.sidebar.number_input("Hash max tokens/patch", min_value=16, max_value=512, value=128, step=16)

        if st.sidebar.button("Build / Rebuild Phase 6 Index"):
            with st.spinner("Building Phase 6 index..."):
                build_phase6_index(
                    pdf_path=PDF_PATH,
                    index_dir=index_dir,
                    model_name=model_name_p6,
                    encoder=encoder_key,
                    hf_local_only=hf_local_only_p6,
                    hash_seed=int(hash_seed_p6),
                    hash_max_tokens=int(hash_max_tokens_p6),
                    page_dpi=int(page_dpi),
                    patch_size=int(patch_size),
                    patch_overlap=int(patch_overlap),
                    max_pages=None if int(max_pages_p6) <= 0 else int(max_pages_p6),
                    proj_dim=int(proj_dim_p6),
                )
            st.sidebar.success("Index built.")

        st.sidebar.markdown("---")
        st.sidebar.subheader("Retrieval")
        top_k_p6 = st.sidebar.slider("Top-k patches", min_value=1, max_value=10, value=5, step=1)
        rerank_k_p6 = st.sidebar.slider("Rerank candidates", min_value=5, max_value=100, value=30, step=5)
        show_patch_text_p6 = st.sidebar.checkbox("Show patch text (if available)", value=True)
        annotate_p6 = st.sidebar.checkbox("Draw attribution boxes", value=True)
        
        st.sidebar.info("Phase 6 enables ColPali-like patch-level multimodal retrieval.")
    elif phase == "Phase 5.1: Tables":
        enable_table_retrieval = st.sidebar.checkbox("Enable table retrieval", value=True)
        st.sidebar.info("Phase 5.1 indexes tables and enables table-aware retrieval.")
        if st.sidebar.button("Index tables from PDF"):
            from app.pipelines.ingestion_pipeline import run_ingestion
            with st.sidebar:
                with st.spinner("Re-indexing PDF (Hybrid Mode)..."):
                    try:
                        count = run_ingestion(PDF_PATH, embedder)
                        st.success(f"Indexed {count} chunks (Text + Tables).")
                    except Exception as e:
                        st.error(f"Indexing failed: {e}")
    elif phase == "Phase 5.2: Tables + Plotting":
        enable_table_retrieval = st.sidebar.checkbox("Enable table retrieval", value=True)
        enable_plotting = st.sidebar.checkbox("Enable plotting of table results", value=True)
        st.sidebar.info("Phase 5.2 includes plotting of table data when CSV is available.")
    elif phase == "Phase 1: Retrieval Only":
        st.sidebar.info("Phase 1 supports retrieval results with optional evaluation.")
    elif phase == "Phase 2: Evaluation Dashboard":
        st.sidebar.info("Run full batch evaluation against your CSV dataset using LLM-as-a-judge and Ragas metrics.")
        dataset_path = st.sidebar.text_input("Dataset Path:", value="RAG_evaluation_dataset - convertcsv (2).csv")
        sample_size = st.sidebar.number_input("Number of rows to evaluate (0 for all)", min_value=0, max_value=100, value=3)
        
        if st.sidebar.button("🚀 Run Batch Evaluation"):
            try:
                df = pd.read_csv(dataset_path)
                if sample_size > 0:
                    df = df.head(sample_size)
                    
                st.write(f"### 📊 Running Batch Evaluation on {len(df)} Questions")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                # Use Qdrant as default for batch
                retriever = Retriever(q_store, embedder)
                pipeline = RAGPipeline(retriever, generator)
                
                for index, row in df.iterrows():
                    question = row['Question']
                    gt = row['Ground_Truth_Answer']
                    status_text.text(f"Processing Q{index+1}: {question[:50]}...")
                    
                    # Generate Answer
                    full_answer = ""
                    retrieved_docs = retriever.retrieve(question)
                    for chunk in pipeline.run(question):
                        full_answer += chunk
                    
                    # LLM Judge
                    score, reasoning = llm_as_judge_score(question, full_answer, gt)
                    
                    # RAGAS Evaluation
                    contexts = [d["text"] for d in retrieved_docs]
                    status_text.text(f"Running Ragas metrics for Q{index+1}...")
                    ragas_scores = get_ragas_scores(question, full_answer, contexts, gt) or {}
                    
                    results.append({
                        "Question": question,
                        "Answer": full_answer,
                        "LLM Judge Score": score,
                        "Reasoning": reasoning,
                        "Faithfulness": ragas_scores.get("faithfulness", None),
                        "Answer Relevancy": ragas_scores.get("answer_relevancy", None),
                        "Context Precision": ragas_scores.get("context_precision", None),
                        "Context Recall": ragas_scores.get("context_recall", None)
                    })
                    progress_bar.progress((index + 1) / len(df))
                
                status_text.success("✅ Batch processing complete!")
                res_df = pd.DataFrame(results)
                st.dataframe(res_df)
                
                # Summary
                st.write("#### 📈 Average Scores")
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("LLM Judge", f"{res_df['LLM Judge Score'].mean():.2f}/5")
                col2.metric("Faithfulness", f"{res_df['Faithfulness'].mean():.2f}")
                col3.metric("Relevancy", f"{res_df['Answer Relevancy'].mean():.2f}")
                col4.metric("Precision", f"{res_df['Context Precision'].mean():.2f}")
                col5.metric("Recall", f"{res_df['Context Recall'].mean():.2f}")
                
                csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download results as CSV",
                    data=csv,
                    file_name="evaluation_results_phase2.csv",
                    mime="text/csv",
                )
                
            except Exception as e:
                st.error(f"Batch Evaluation failed: {e}")
    else:
        st.sidebar.info("Select a phase to explore the features.")
    
    st.markdown('<h1 class="main-header">⚖️ RAG Intelligence Hub</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Multimodal Retrieval & Evaluation Dashboard</p>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown(f"""
            <div style="text-align: center; margin-bottom: 2rem;">
                <span style="background: rgba(0, 210, 255, 0.1); color: #00d2ff; padding: 5px 15px; border-radius: 20px; font-size: 0.9rem; font-weight: 600; border: 1px solid rgba(0, 210, 255, 0.2);">
                    Active Workflow: {phase}
                </span>
            </div>
        """, unsafe_allow_html=True)

    semantic_cache = SemanticCache() if semantic_cache_enabled else None

    query = st.text_input("Enter your financial query:", 
                         placeholder="What is IFC's net income in 2024?",
                         key="search_input")

    if query:
        if phase == "All Phases":
            # Run all phases sequentially and display results in expanders
            all_phases = [
                "Phase 1: Retrieval Only",
                "Phase 3: Full Evaluation",
                "Phase 4: Advanced RAG",
                "Phase 5.1: Tables",
                "Phase 5.2: Tables + Plotting",
                "Phase 6: Multimodal RAG",
            ]
            for p in all_phases:
                with st.expander(p):
                    # Determine retriever and config per phase (basic defaults)
                    if p == "Phase 1: Retrieval Only":
                        retr = Retriever(q_store, embedder)
                        run_db_query("Qdrant", query, retr, generator, st.container(), evaluation_enabled=False)
                    elif p == "Phase 3: Full Evaluation":
                        retr = Retriever(q_store, embedder)
                        run_db_query("Qdrant", query, retr, generator, st.container(), evaluation_enabled=True, ground_truth=ground_truth)
                    elif p == "Phase 4: Advanced RAG":
                        base_ret = Retriever(q_store, embedder)
                        ret = MultiHopRetriever(base_ret)
                        run_db_query("Phase 4 (Combined)", query, ret, generator, st.container(), evaluation_enabled=eval_enabled, ground_truth=ground_truth)
                    elif p == "Phase 5.1: Tables":
                        from app.retriever.multimodal_retriever import MultimodalRetriever, MultimodalRetrievalConfig
                        cfg = MultimodalRetrievalConfig(k_text=0, k_table=5, k_image=0)
                        mm_ret = MultimodalRetriever(q_store, embedder, config=cfg)
                        run_db_query("Phase 5 (Tables)", query, mm_ret, generator, st.container(), evaluation_enabled=eval_enabled, ground_truth=ground_truth)
                    elif p == "Phase 5.2: Tables + Plotting":
                        from app.retriever.multimodal_retriever import MultimodalRetriever, MultimodalRetrievalConfig
                        cfg = MultimodalRetrievalConfig(k_text=0, k_table=5, k_image=0)
                        mm_ret = MultimodalRetriever(q_store, embedder, config=cfg)
                        run_db_query("Phase 5 (Tables+Plot)", query, mm_ret, generator, st.container(), evaluation_enabled=eval_enabled, ground_truth=ground_truth, enable_plotting=True)
                        else:
                            st.warning("Phase 6 index not found. Please build it in the single-phase view.")
            
            # --- Cross-Database Verdict ---
            if mode == "Compare Both" and 'scores' in st.session_state and len(st.session_state.scores) >= 2:
                st.divider()
                st.subheader("🏁 Performance Verdict")
                s_q = st.session_state.scores.get("Qdrant", 0)
                s_f = st.session_state.scores.get("FAISS", 0)
                
                winner = "Tie"
                if s_q > s_f: winner = "Qdrant"
                elif s_f > s_q: winner = "FAISS"
                
                cols = st.columns(3)
                with cols[1]:
                    color = "#00d2ff" if winner != "Tie" else "#888"
                    st.markdown(f"""
                        <div class="glass-card" style="text-align: center; border-color: {color};">
                            <h2 style="color: {color}; margin: 0;">🏆 Winner: {winner}</h2>
                            <p style="color: #aaa; margin: 10px 0 0 0;">
                                Based on LLM Judge scores (Q: {s_q}/5 vs F: {s_f}/5)
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                # Reset scores for next query
                del st.session_state.scores
        else:
            if phase == "Phase 6: Multimodal RAG":
                st.subheader("🧩 Phase 6: Multimodal Patch RAG")
                
                query_image_file = st.file_uploader("Upload query image (optional)", type=["png", "jpg", "jpeg"])
                query_image_path = None
                if query_image_file:
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(query_image_file.name).suffix) as f:
                        f.write(query_image_file.getbuffer())
                        query_image_path = f.name

                if (query or query_image_path):
                    if query_image_path and encoder_key == "hash":
                        st.error("The 'hash' encoder does not support image queries. Please select 'siglip (multimodal)' in the sidebar.")
                    elif not _index_present(index_dir):
                        st.warning(f"Index not found at {index_dir}. Build it in the sidebar.")
                    else:
                        try:
                            with st.spinner("Retrieving patches..."):
                                results_p6 = retrieve_patches_phase6(
                                    question=query,
                                    query_image_path=query_image_path,
                                    index_dir=index_dir,
                                    model_name=model_name_p6,
                                    encoder=encoder_key,
                                    hf_local_only=hf_local_only_p6,
                                    hash_seed=int(hash_seed_p6),
                                    hash_max_tokens=int(hash_max_tokens_p6),
                                    top_k=int(top_k_p6),
                                    rerank_k=int(rerank_k_p6),
                                    proj_dim=int(proj_dim_p6),
                                )
                            
                            if results_p6:
                                st.write("### 🔎 Retrieved patches")
                                if query_image_path:
                                    st.write("**Your Query Image:**")
                                    st.markdown(f'<div class="glass-card"><img src="data:image/png;base64,{_get_img_base64(query_image_path)}" style="width: 100%; border-radius: 8px;"/></div>', unsafe_allow_html=True)

                                for rank, (p, score) in enumerate(results_p6, start=1):
                                    with st.container():
                                        st.markdown(f"""
                                            <div class="glass-card">
                                                <div style="color: #00d2ff; font-weight: 600; margin-bottom: 10px;">Rank #{rank} | Page {p.page} | Score {score:.4f}</div>
                                                <div style="display: flex; gap: 15px;">
                                                    <div style="flex: 1;"><img src="data:image/png;base64,{_get_img_base64(p.patch_path)}" style="width: 100%; border-radius: 8px;"/></div>
                                                    <div style="flex: 2;"><img src="data:image/png;base64,{_get_img_base64(p.page_image_path)}" style="width: 100%; border-radius: 8px;"/></div>
                                                </div>
                                            </div>
                                        """, unsafe_allow_html=True)
                                        if show_patch_text_p6 and p.text:
                                            with st.expander(f"Patch text (#{rank})"):
                                                st.text(p.text[:2000])
                                                # Check for tabular data in patch text (Phase 5 feature)
                                                if "csv" in p.text.lower() or "|" in p.text:
                                                    try:
                                                        # Best-effort CSV extraction for plotting
                                                        lines = p.text.split('\n')
                                                        csv_lines = [l for l in lines if '|' in l or ',' in l]
                                                        if len(csv_lines) > 2:
                                                            uri = plot_from_csv_string("\n".join(csv_lines))
                                                            if uri:
                                                                st.image(uri, caption="Table detected in patch")
                                                    except Exception:
                                                        pass

                                if annotate_p6:
                                    out_dir = str(Path(index_dir) / "attribution" / f"q_{_sha1(query or 'img_only')}")
                                    written = annotate_sources(results_p6, out_dir=out_dir)
                                    if written:
                                        st.write("### 🧾 Attribution")
                                        for img_path in written:
                                            st.image(img_path, use_container_width=True)

                                st.divider()
                                st.write("### 🤖 Answer & Evaluation")
                                
                                # Use Gemini to answer
                                use_gemini = st.checkbox("Generate answer with Gemini", key="p6_gemini", value=True)
                                full_answer_p6 = ""
                                if use_gemini:
                                    n_patches = st.slider("Number of patches to send", 1, min(5, len(results_p6)), 3)
                                    selected = [p for (p, _s) in results_p6[:n_patches]]
                                    with st.spinner("Calling Gemini (multimodal)..."):
                                        full_answer_p6 = answer_with_visual_context(query, selected, query_image_path=query_image_path)
                                    st.markdown(f"<div class='answer-text'>{full_answer_p6}</div>", unsafe_allow_html=True)

                                # Evaluation (Phase 2/3 feature integrated into Phase 6)
                                if eval_enabled and full_answer_p6:
                                    if not ground_truth:
                                        st.info("Enter ground truth in sidebar to see evaluation.")
                                    else:
                                        with st.expander("⚖️ Evaluation Scores (Phase 6)"):
                                            score, reasoning = llm_as_judge_score(query or "Image Query", full_answer_p6, ground_truth)
                                            if score:
                                                render_phase2_scorecard(score, reasoning)
                                            
                                            # Ragas (using patch text as context)
                                            contexts = [p.text for p, _ in results_p6 if p.text]
                                            if contexts:
                                                ragas_scores = get_ragas_scores(query or "Image Query", full_answer_p6, contexts, ground_truth)
                                                if ragas_scores:
                                                    render_ragas_metrics(ragas_scores)
                            else:
                                st.info("No patches retrieved.")
                        except Exception as e:
                            st.error(f"Phase 6 failed: {e}")

            elif phase in ("Phase 5.1: Tables", "Phase 5.2: Tables + Plotting"):
                # Use the QdrantStore as the primary vectorstore for table chunks
                from app.retriever.multimodal_retriever import MultimodalRetriever, MultimodalRetrievalConfig
                # Strictly fetch only tables for Phase 5 to avoid text noise
                config = MultimodalRetrievalConfig(k_text=0, k_table=5, k_image=0)
                mm_retriever = MultimodalRetriever(q_store, embedder, config=config)
                # Run the pipeline
                run_db_query("Phase 5 (Tables)", query, mm_retriever, generator, st.container(), semantic_cache=semantic_cache, evaluation_enabled=eval_enabled, ground_truth=ground_truth, enable_plotting=enable_plotting)

            elif phase == "Phase 4: Advanced RAG":
                if mode == "Compare Both":
                    base_retriever = CombinedRetriever([q_store, f_store], embedder)
                    retriever = MultiHopRetriever(base_retriever) if multi_hop_enabled else base_retriever
                    run_db_query("Phase 4 (Combined)", query, retriever, generator, st.container(), semantic_cache=semantic_cache, evaluation_enabled=eval_enabled, ground_truth=ground_truth)
                elif mode == "Qdrant Only":
                    base_retriever = Retriever(q_store, embedder)
                    retriever = MultiHopRetriever(base_retriever) if multi_hop_enabled else base_retriever
                    run_db_query("Qdrant", query, retriever, generator, st.container(), semantic_cache=semantic_cache, evaluation_enabled=eval_enabled, ground_truth=ground_truth)
                else:
                    base_retriever = Retriever(f_store, embedder)
                    retriever = MultiHopRetriever(base_retriever) if multi_hop_enabled else base_retriever
                    run_db_query("FAISS", query, retriever, generator, st.container(), semantic_cache=semantic_cache, evaluation_enabled=eval_enabled, ground_truth=ground_truth)
            else:
                if mode == "Compare Both":
                    col1, col2 = st.columns(2)
                    run_db_query("Qdrant", query, Retriever(q_store, embedder), generator, col1, evaluation_enabled=eval_enabled, ground_truth=ground_truth)
                    run_db_query("FAISS", query, Retriever(f_store, embedder), generator, col2, evaluation_enabled=eval_enabled, ground_truth=ground_truth)
                elif mode == "Qdrant Only":
                    run_db_query("Qdrant", query, Retriever(q_store, embedder), generator, st.container(), evaluation_enabled=eval_enabled, ground_truth=ground_truth)
                else:
                    run_db_query("FAISS", query, Retriever(f_store, embedder), generator, st.container(), evaluation_enabled=eval_enabled, ground_truth=ground_truth)

if __name__ == "__main__":
    main()