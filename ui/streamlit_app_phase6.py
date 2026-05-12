import os
import sys
import hashlib
from pathlib import Path

import streamlit as st

# Add the project root to sys.path to allow importing from 'app' and phase runners.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import PDF_PATH
from phase6_colpali_like import (
    build_phase6_index,
    retrieve_patches_phase6,
    annotate_sources,
    answer_with_visual_context,
)


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _index_present(index_dir: str) -> bool:
    idx = Path(index_dir)
    return (idx / "faiss_pooled.index").exists() and (idx / "faiss_pooled.pkl").exists()


st.set_page_config(
    page_title="Phase 6: ColPali-like Patch RAG",
    page_icon="🧩",
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


def main():
    st.sidebar.title("🧩 Phase 6 Settings")

    pdf_path = st.sidebar.text_input("PDF path", value=PDF_PATH)
    index_dir = st.sidebar.text_input("Index dir", value="phase6_index")

    encoder = st.sidebar.selectbox("Encoder", ["hash (offline)", "siglip (multimodal)"], index=0)
    encoder_key = "hash" if encoder.startswith("hash") else "siglip"

    st.sidebar.markdown("---")
    # Sidebar setting for auto-building index if missing
    auto_build = st.sidebar.checkbox("Auto‑build index if missing", value=False)
    page_dpi = st.sidebar.number_input("Page DPI", min_value=72, max_value=300, value=150, step=10)
    patch_size = st.sidebar.number_input("Patch size", min_value=128, max_value=1024, value=512, step=64)
    patch_overlap = st.sidebar.number_input("Patch overlap", min_value=0, max_value=512, value=0, step=32)
    max_pages = st.sidebar.number_input("Max pages (0 = all)", min_value=0, max_value=2000, value=2, step=1)
    proj_dim = st.sidebar.number_input("proj_dim", min_value=32, max_value=512, value=128, step=32)

    model_name = "google/siglip-base-patch16-224"
    hf_local_only = True
    hash_seed = 0
    hash_max_tokens = 128

    if encoder_key == "siglip":
        model_name = st.sidebar.text_input("SigLIP model", value=model_name)
        hf_local_only = st.sidebar.checkbox("HF local only (no downloads)", value=True)
    else:
        hash_seed = st.sidebar.number_input("Hash seed", min_value=0, max_value=10_000, value=0, step=1)
        hash_max_tokens = st.sidebar.number_input("Hash max tokens/patch", min_value=16, max_value=512, value=128, step=16)

    do_build = st.sidebar.button("Build / Rebuild index")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Query")
    top_k = st.sidebar.slider("Top-k patches", min_value=1, max_value=10, value=5, step=1)
    rerank_k = st.sidebar.slider("Rerank candidates", min_value=5, max_value=100, value=30, step=5)
    show_patch_text = st.sidebar.checkbox("Show patch text (if available)", value=True)
    annotate = st.sidebar.checkbox("Draw attribution boxes", value=True)

    st.sidebar.markdown("---")
    st.sidebar.caption("Phase 6 index files are stored under the index dir (pages/, patches/, faiss_pooled.*).")

    st.title("🧩 Phase 6: Multimodal Patch RAG (ColPali-like)")
    st.markdown(
        "Builds a **patch-level index** from rendered PDF pages, retrieves relevant patches, and highlights sources. "
        "Use `hash` for offline demo; use `siglip` for true multimodal embeddings (requires model files)."
    )

    if do_build:
        if not os.path.exists(pdf_path):
            st.error(f"PDF not found: {pdf_path}")
        else:
            with st.spinner("Building Phase 6 index (this can take a while)..."):
                build_phase6_index(
                    pdf_path=pdf_path,
                    index_dir=index_dir,
                    model_name=model_name,
                    encoder=encoder_key,
                    hf_local_only=hf_local_only,
                    hash_seed=int(hash_seed),
                    hash_max_tokens=int(hash_max_tokens),
                    page_dpi=int(page_dpi),
                    patch_size=int(patch_size),
                    patch_overlap=int(patch_overlap),
                    max_pages=None if int(max_pages) <= 0 else int(max_pages),
                    proj_dim=int(proj_dim),
                )
            st.success("Index built.")

    # Determine if index is present; optionally auto‑build
    present = _index_present(index_dir)
    if not present and auto_build:
        st.info("Index not found – building now (auto‑build enabled).")
        if not os.path.exists(pdf_path):
            st.error(f"PDF not found: {pdf_path}")
        else:
            with st.spinner("Building Phase 6 index (auto‑build)..."):
                build_phase6_index(
                    pdf_path=pdf_path,
                    index_dir=index_dir,
                    model_name=model_name,
                    encoder=encoder_key,
                    hf_local_only=hf_local_only,
                    hash_seed=int(hash_seed),
                    hash_max_tokens=int(hash_max_tokens),
                    page_dpi=int(page_dpi),
                    patch_size=int(patch_size),
                    patch_overlap=int(patch_overlap),
                    max_pages=None if int(max_pages) <= 0 else int(max_pages),
                    proj_dim=int(proj_dim),
                )
            st.success("Index built (auto‑build).")
            present = True
    # Original existing check (now using updated present variable)
    if present:
        st.success(f"Index ready: {index_dir}")
    else:
        st.warning("Index not found. Build it from the sidebar first.")

    question = st.text_input(
        "Enter your question (optional if image is provided)",
        placeholder="What is IFC's net income (loss) in 2024?",
        key="phase6_question",
    )
    query_image_file = st.file_uploader("Or upload an image to search", type=["png", "jpg", "jpeg"])

    if not present:
        return

    if not question and not query_image_file:
        return

    query_image_path = None
    if query_image_file:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(query_image_file.name).suffix) as f:
            f.write(query_image_file.getbuffer())
            query_image_path = f.name
            
    if query_image_path and encoder_key == "hash":
        st.error("The 'hash' encoder does not support image queries. Please select 'siglip (multimodal)' in the sidebar.")
        return

    try:
        with st.spinner("Retrieving patches..."):
            results = retrieve_patches_phase6(
                question=question,
                query_image_path=query_image_path,
                index_dir=index_dir,
                model_name=model_name,
                encoder=encoder_key,
                hf_local_only=hf_local_only,
                hash_seed=int(hash_seed),
                hash_max_tokens=int(hash_max_tokens),
                top_k=int(top_k),
                rerank_k=int(rerank_k),
                proj_dim=int(proj_dim),
            )
    except Exception as e:
        st.error(f"Retrieval failed: {e}")
        return

    if not results:
        st.error("No patches retrieved.")
        return

    st.write("### 🔎 Retrieved patches")
    if query_image_path:
        st.write("**Your Query Image:**")
        st.image(query_image_path, use_container_width=True)
        
    for rank, (p, score) in enumerate(results, start=1):
        st.markdown(
            f"<div class='source-box'><b>#{rank}</b> score={score:.4f} | page={p.page} | patch_id={p.patch_id}<br>"
            f"<small>bbox={p.bbox}</small></div>",
            unsafe_allow_html=True,
        )
        cols = st.columns([1, 2])
        with cols[0]:
            st.image(p.patch_path, caption=f"Patch #{rank}", use_container_width=True)
        with cols[1]:
            st.image(p.page_image_path, caption=f"Source page (page={p.page})", use_container_width=True)
            if show_patch_text and p.text:
                with st.expander(f"Patch text (#{rank})", expanded=False):
                    st.text(p.text[:8000])

    if annotate:
        out_dir = str(Path(index_dir) / "attribution" / f"q_{_sha1(question)}")
        written = annotate_sources(results, out_dir=out_dir)
        if written:
            st.write("### 🧾 Attribution (highlighted pages)")
            for img_path in written:
                st.image(img_path, caption=Path(img_path).name, use_container_width=True)

    st.write("### 🤖 Answer (optional)")
    st.caption("This requires outbound network access + Vertex AI credentials (Gemini multimodal).")
    use_gemini = st.checkbox("Generate answer with Gemini using patch images", value=False)
    if use_gemini:
        n_patches = st.slider("Number of patch images to send", min_value=1, max_value=min(5, len(results)), value=3)
        selected = [p for (p, _s) in results[: int(n_patches)]]
        try:
            with st.spinner("Calling Gemini (multimodal)..."):
                ans = answer_with_visual_context(question, selected, model_name=None, query_image_path=query_image_path)
            st.markdown(f"<div class='answer-text'>{ans}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Gemini call failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

