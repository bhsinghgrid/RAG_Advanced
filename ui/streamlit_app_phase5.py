import streamlit as st
import os
import sys
import re
import pandas as pd

# Add the project root to sys.path to allow importing from 'app'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import PDF_PATH, VECTOR_DIM
from app.embeddings.embedder import Embedder
from app.vectorstores.qdrant_store import QdrantStore
from app.vectorstores.faiss_store import FAISSStore
from app.pipelines.phase5_ingestion_pipeline import run_phase5_ingestion
from app.retriever.multimodal_retriever import MultimodalRetriever, MultimodalRetrievalConfig
from app.llm.table_generator import TableAwareGenerator
from app.pipelines.rag_pipeline import RAGPipeline


PHASE5_QDRANT_COLLECTION = "ifc_annual_report_phase5"
PHASE5_FAISS_INDEX = "faiss_index_phase5"


st.set_page_config(
    page_title="Phase 5.1: Table-Aware RAG",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #121212; color: #ffffff; }
    .stTextInput input { color: #111111 !important; background-color: #ffffff !important; font-weight: 500 !important; border-radius: 5px; }
    .answer-text { color: #ffffff !important; font-size: 1rem; line-height: 1.6; margin-bottom: 20px; }
    [data-testid="stSidebar"] { background-color: #1e1e1e; }
    .source-box { background-color: #262626; border: 1px solid #444; padding: 10px; border-radius: 5px; margin-top: 5px; color: #ddd; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def init_resources():
    embedder = Embedder()
    generator = TableAwareGenerator()
    q_store = QdrantStore(collection_name=PHASE5_QDRANT_COLLECTION)
    f_store = FAISSStore(dim=VECTOR_DIM, save_path=PHASE5_FAISS_INDEX)
    if os.path.exists(f"{PHASE5_FAISS_INDEX}.index"):
        f_store.load()
    return embedder, generator, q_store, f_store


def _parse_table_rows_kv(chunk_text: str) -> pd.DataFrame | None:
    # Only parse the key=value table row format.
    if "[TABLE ROWS]" not in chunk_text:
        return None
    row_lines = [ln.strip() for ln in chunk_text.splitlines() if ln.strip().startswith("row_")]
    if not row_lines:
        return None

    records = []
    for ln in row_lines:
        try:
            _, rest = ln.split(":", 1)
        except ValueError:
            continue
        rest = rest.strip()
        if rest == "(empty)":
            continue
        pairs = [p.strip() for p in rest.split(";") if p.strip()]
        rec = {}
        for p in pairs:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            rec[k.strip()] = v.strip()
        if rec:
            records.append(rec)
    if not records:
        return None
    return pd.DataFrame(records)


def run_backend(name: str, retriever: MultimodalRetriever, pipeline: RAGPipeline, query: str, *, show_contexts: bool):
    st.subheader(name)
    results = retriever.retrieve(query, k=5, rerank=False)
    if not results:
        st.error("No relevant chunks found.")
        return

    st.write("### 🤖 Answer")
    answer_container = st.empty()
    full_answer = ""
    for piece in pipeline.run(query, k=5):
        full_answer += piece
        answer_container.markdown(f'<div class="answer-text">{full_answer}</div>', unsafe_allow_html=True)

    if show_contexts:
        st.write("### 📚 Retrieved Contexts")
        for r in results:
            meta = r.get("metadata", {}) or {}
            st.markdown(
                f"<div class='source-box'><b>type</b>: {meta.get('content_type')} | <b>page</b>: {meta.get('page')}<br><pre>{r.get('text','')[:2000]}</pre></div>",
                unsafe_allow_html=True,
            )

        # Optional plotting: if we can parse any table chunk into a DataFrame, allow simple charting.
        dfs = []
        for r in results:
            df = _parse_table_rows_kv(r.get("text", ""))
            if df is not None and not df.empty:
                dfs.append(df)
        if dfs:
            st.write("### 📈 Plot (optional)")
            df = pd.concat(dfs, ignore_index=True).dropna(axis=1, how="all")
            st.dataframe(df, use_container_width=True)

            cols = df.columns.tolist()
            if cols:
                x_col = st.selectbox("X column", cols, index=0)
                y_candidates = [c for c in cols if c != x_col]
                y_col = st.selectbox("Y column", y_candidates, index=0) if y_candidates else None
                if y_col:
                    # Attempt numeric conversion on y
                    tmp = df.copy()
                    tmp[y_col] = (
                        tmp[y_col]
                        .astype(str)
                        .str.replace(",", "", regex=False)
                        .str.replace(r"[^0-9.\\-+]", "", regex=True)
                    )
                    tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce")
                    st.line_chart(tmp[[x_col, y_col]].dropna(), x=x_col, y=y_col)


def main():
    st.sidebar.title("📊 Phase 5.1 Settings")
    mode = st.sidebar.radio("Backend", ["Qdrant Only", "FAISS Only", "Compare Both"], index=2)
    show_contexts = st.sidebar.checkbox("Show retrieved contexts", value=True)
    ingest = st.sidebar.checkbox("Run ingestion now (slow)", value=False)
    include_images = st.sidebar.checkbox("Include image descriptions (Phase 5.2)", value=False)
    k_image = st.sidebar.slider("Retrieve image chunks (k_image)", min_value=0, max_value=5, value=0, step=1)

    st.sidebar.markdown("---")
    st.sidebar.info(f"📄 **Document:** {PDF_PATH}")

    st.title("📊 Phase 5.1: Table-Aware RAG (Text + Tables)")
    st.markdown("Retrieves across **text chunks** and **table row chunks**, then answers with a table-aware prompt.")
    st.markdown("---")

    embedder, generator, q_store, f_store = init_resources()

    if ingest:
        with st.spinner("Running Phase 5 ingestion (text + tables)..."):
            run_phase5_ingestion(PDF_PATH, embedder, [q_store, f_store], include_images=include_images)
        st.success("Ingestion complete.")

    query = st.text_input(
        "Enter your query",
        placeholder="What is net income (loss) in 2024?",
        key="phase5_query",
    )

    if not query:
        return

    if mode == "Compare Both":
        col1, col2 = st.columns(2)
        with col1:
            retriever_q = MultimodalRetriever(q_store, embedder, config=MultimodalRetrievalConfig(k_image=k_image))
            pipeline_q = RAGPipeline(retriever_q, generator)
            run_backend("Qdrant", retriever_q, pipeline_q, query, show_contexts=show_contexts)
        with col2:
            retriever_f = MultimodalRetriever(f_store, embedder, config=MultimodalRetrievalConfig(k_image=k_image))
            pipeline_f = RAGPipeline(retriever_f, generator)
            run_backend("FAISS", retriever_f, pipeline_f, query, show_contexts=show_contexts)
    elif mode == "Qdrant Only":
        retriever_q = MultimodalRetriever(q_store, embedder, config=MultimodalRetrievalConfig(k_image=k_image))
        pipeline_q = RAGPipeline(retriever_q, generator)
        run_backend("Qdrant", retriever_q, pipeline_q, query, show_contexts=show_contexts)
    else:
        retriever_f = MultimodalRetriever(f_store, embedder, config=MultimodalRetrievalConfig(k_image=k_image))
        pipeline_f = RAGPipeline(retriever_f, generator)
        run_backend("FAISS", retriever_f, pipeline_f, query, show_contexts=show_contexts)


if __name__ == "__main__":
    main()
