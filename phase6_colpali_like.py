"""
Phase 6 — Multimodal RAG with a ColPali-like (late interaction / MaxSim) approach
───────────────────────────────────────────────────────────────────────────────
This phase builds a patch-level index from PDF page images and retrieves relevant
patches using:

1) A pooled vector FAISS index (fast candidate search)
2) Late-interaction reranking (MaxSim) over query-token × patch-token embeddings

Backends:
  - hash (default): offline patch-text hashing encoder (no downloads / no network)
  - siglip: SigLIP dual-encoder (requires local model files or internet access)

Commands:
  - Build index:
      python phase6_colpali_like.py build --encoder hash
  - Query (retrieve + highlight sources):
      python phase6_colpali_like.py query --encoder hash -q "What are total assets in 2024?"
  - Ask (retrieve + send images to multimodal Gemini on Vertex AI):
      python phase6_colpali_like.py ask --encoder siglip -q "..." --n-patches 3
  - Compare (Phase 5 vs Phase 6 retrieval):
      python phase6_colpali_like.py compare -q "..."
"""

from __future__ import annotations

import os
import sys
import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from PIL import Image, ImageDraw

from app.config import PDF_PATH, GCP_LOCATION, GCP_PROJECT_ID, LLM_MODEL
# FAISS (and some ML libs) can trigger OpenMP runtime clashes on macOS.
# This env var is an unsafe workaround, but prevents hard crashes for many setups.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from app.vectorstores.faiss_store import FAISSStore


@dataclass(frozen=True)
class Patch:
    patch_id: str
    page: int
    bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1) in page-image pixels
    patch_path: str
    page_image_path: str
    text: Optional[str] = None

    def to_metadata(self) -> dict:
        meta = {
            "phase": 6,
            "content_type": "image_patch",
            "patch_id": self.patch_id,
            "page": self.page,
            "bbox": list(self.bbox),
            "patch_path": self.patch_path,
            "page_image_path": self.page_image_path,
        }
        if self.text:
            # Store a compact version; retrieval can still use tokenization.
            meta["patch_text"] = self.text
        return meta


def _batched(items: list, batch_size: int) -> Iterator[list]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _default_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def init_vertex() -> None:
    import vertexai

    if not GCP_PROJECT_ID:
        raise RuntimeError("GCP_PROJECT_ID is not set. Add it to `.env` or your environment.")
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)


def pdf_pages_to_images(
    pdf_path: str,
    out_dir: str,
    dpi: int = 150,
    max_pages: int | None = None,
) -> list[dict]:
    """
    Render PDF pages to PNG images using PyMuPDF.
    Returns list of {"page": int, "image_path": str, "width": int, "height": int}.
    """
    import fitz  # PyMuPDF

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    try:
        pages: list[dict] = []
        n_pages = len(doc) if max_pages is None else min(len(doc), max_pages)
        for i in range(n_pages):
            page_num = i + 1
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=dpi)
            image_path = out / f"page_{page_num:04d}.png"
            pix.save(str(image_path))
            pages.append(
                {
                    "page": page_num,
                    "image_path": str(image_path),
                    "width": pix.width,
                    "height": pix.height,
                }
            )
        return pages
    finally:
        doc.close()


def image_to_patches(
    page_image_path: str,
    page: int,
    out_dir: str,
    patch_size: int = 512,
    overlap: int = 0,
) -> list[Patch]:
    """
    Segment a page image into a grid of (possibly overlapping) patches.
    Patches are saved as PNG to `out_dir`.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with Image.open(page_image_path) as opened:
        img = opened.convert("RGB")
        w, h = img.size
        stride = max(1, patch_size - max(0, overlap))

        patches: list[Patch] = []
        patch_idx = 0
        for y0 in range(0, h, stride):
            for x0 in range(0, w, stride):
                x1 = min(x0 + patch_size, w)
                y1 = min(y0 + patch_size, h)
                if x1 <= x0 or y1 <= y0:
                    continue

                patch = img.crop((x0, y0, x1, y1))
                patch_id = f"p{page:04d}_{patch_idx:05d}_x{x0}_y{y0}_x1{x1}_y1{y1}"
                patch_path = out / f"{patch_id}.png"
                patch.save(patch_path)

                patches.append(
                    Patch(
                        patch_id=patch_id,
                        page=page,
                        bbox=(x0, y0, x1, y1),
                        patch_path=str(patch_path),
                        page_image_path=str(page_image_path),
                    )
                )
                patch_idx += 1

        return patches


_DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}


def _simple_tokenize(text: str) -> list[str]:
    toks = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
    # Keep 1-char tokens only if they are digits.
    toks = [t for t in toks if (len(t) > 1) or t.isdigit()]
    return toks


def _dedup_limit(tokens: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        if t in _DEFAULT_STOPWORDS:
            continue
        out.append(t)
        if len(out) >= limit:
            break
    return out


class HashTextEncoder:
    """Offline, deterministic text hashing encoder.

    This is a *fallback* encoder that enables Phase 6 to run without any model downloads
    or network access. It uses patch-region text extracted from the PDF (no OCR),
    tokenizes it, then maps tokens to unit vectors via a hash-seeded RNG.
    """

    def __init__(self, *, dim: int = 128, seed: int = 0, max_tokens: int = 128) -> None:
        self.dim = int(dim)
        self.seed = int(seed)
        self.max_tokens = int(max_tokens)
        self._cache: dict[str, np.ndarray] = {}

    def _token_vec(self, token: str) -> np.ndarray:
        v = self._cache.get(token)
        if v is not None:
            return v
        # Deterministic per-token vector.
        digest = hashlib.sha1(f"{self.seed}:{token}".encode("utf-8")).digest()
        token_seed = int.from_bytes(digest, byteorder="big", signed=False)
        rng = np.random.default_rng(token_seed)
        v = rng.standard_normal(self.dim).astype("float32", copy=False)
        v = _l2_normalize(v, axis=0)
        self._cache[token] = v
        return v

    def encode_text(self, text: str, *, max_tokens: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Returns (tokens, pooled)."""
        limit = self.max_tokens if max_tokens is None else int(max_tokens)
        toks = _dedup_limit(_simple_tokenize(text), limit=limit)
        if not toks:
            # Degenerate case: return a single zero-ish token so MaxSim is defined.
            z = np.zeros((1, self.dim), dtype="float32")
            return z, z
        mat = np.stack([self._token_vec(t) for t in toks], axis=0).astype("float32", copy=False)
        mat = _l2_normalize(mat, axis=-1)
        pooled = mat.mean(axis=0, keepdims=True)
        pooled = _l2_normalize(pooled, axis=-1)
        return mat, pooled


def _load_siglip(model_name: str, device: str, *, local_files_only: bool = False):
    """
    Returns (vision_model, text_model, processor, tokenizer, vision_config)
    """
    import torch
    from transformers import AutoProcessor, AutoTokenizer, SiglipModel

    processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = SiglipModel.from_pretrained(model_name, local_files_only=local_files_only)

    model.eval()
    model.to(device)

    # Access submodules to get token-level states
    vision_model = model.vision_model
    text_model = model.text_model

    return vision_model, text_model, processor, tokenizer, model.config.vision_config


def _load_siglip_or_raise(model_name: str, device: str, *, local_files_only: bool):
    try:
        return _load_siglip(model_name, device=device, local_files_only=local_files_only)
    except Exception as e:
        if local_files_only:
            raise RuntimeError(
                "SigLIP backend requested, but model files are not available locally. "
                "Either (1) download the model weights on a machine with internet and pass a local path via `--model`, "
                "or (2) run with `--encoder hash` (offline fallback)."
            ) from e
        raise


def _ensure_projection(index_dir: str, in_dim: int, out_dim: int, seed: int = 0) -> np.ndarray:
    """
    Create/load a fixed random projection matrix for storage + MaxSim rerank.
    Stored at {index_dir}/projection.npy
    """
    index = Path(index_dir)
    index.mkdir(parents=True, exist_ok=True)
    path = index / "projection.npy"
    if path.exists():
        w = np.load(path)
        if w.shape != (in_dim, out_dim):
            raise ValueError(f"Projection shape mismatch: expected {(in_dim, out_dim)}, got {w.shape}")
        return w

    rng = np.random.default_rng(seed)
    w = rng.standard_normal(size=(in_dim, out_dim)).astype("float32")
    np.save(path, w)
    return w


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def _attach_patch_texts(pdf_path: str, pages: list[dict], patches: list[Patch]) -> list[Patch]:
    """Populate Patch.text by extracting the PDF text within each patch bbox.

    Uses PyMuPDF's `Page.get_textbox(rect)` on the patch rectangle mapped from
    pixmap (pixel) coords back into page coords.
    """
    import fitz  # PyMuPDF

    pages_by_num = {int(p["page"]): p for p in pages}
    by_page: dict[int, list[Patch]] = {}
    for p in patches:
        by_page.setdefault(int(p.page), []).append(p)

    doc = fitz.open(pdf_path)
    try:
        out: list[Patch] = []
        for page_num, ps in by_page.items():
            page_info = pages_by_num.get(page_num)
            if not page_info:
                out.extend(ps)
                continue

            page = doc.load_page(page_num - 1)  # Patch.page is 1-based
            pix_irect = fitz.IRect(0, 0, int(page_info["width"]), int(page_info["height"]))
            # matrix mapping page coords -> pixmap coords
            mat = page.rect.torect(pix_irect)
            inv = ~mat  # pixmap -> page coords

            for p in ps:
                rect_pix = fitz.Rect(*p.bbox)
                rect_page = rect_pix * inv
                try:
                    txt = (page.get_textbox(rect_page) or "").strip()
                except Exception:
                    txt = ""
                out.append(
                    Patch(
                        patch_id=p.patch_id,
                        page=p.page,
                        bbox=p.bbox,
                        patch_path=p.patch_path,
                        page_image_path=p.page_image_path,
                        text=txt,
                    )
                )
        # preserve original order where possible
        patch_by_id = {p.patch_id: p for p in out}
        return [patch_by_id.get(p.patch_id, p) for p in patches]
    finally:
        doc.close()


def _encode_image_tokens(
    vision_model,
    processor,
    image_paths: list[str],
    device: str,
    proj: np.ndarray,
    drop_cls: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (tokens, pooled) where:
      tokens: (batch, n_tokens, d_proj) float32 L2-normalized per token
      pooled: (batch, d_proj) float32 L2-normalized
    """
    import torch

    images: list[Image.Image] = []
    for p in image_paths:
        with Image.open(p) as opened:
            images.append(opened.convert("RGB"))
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.inference_mode():
        out = vision_model(pixel_values=pixel_values)
        h = out.last_hidden_state  # (b, seq, hidden)
        h = h.detach().float().cpu().numpy()

    if drop_cls and h.shape[1] > 1:
        h = h[:, 1:, :]

    # project + normalize per token
    tokens = h @ proj  # (b, n_tokens, d_proj)
    tokens = _l2_normalize(tokens, axis=-1)

    pooled = tokens.mean(axis=1)
    pooled = _l2_normalize(pooled, axis=-1)
    return tokens.astype("float32"), pooled.astype("float32")


def _encode_query_tokens(
    text_model,
    tokenizer,
    query: str,
    device: str,
    proj: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (tokens, pooled) where:
      tokens: (q_tokens, d_proj) float32 L2-normalized
      pooled: (1, d_proj) float32 L2-normalized
    """
    import torch

    inputs = tokenizer([query], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        out = text_model(**inputs)
        h = out.last_hidden_state  # (b, seq, hidden)

    h0 = h[0].detach().float().cpu().numpy()  # (seq, hidden)
    attn = inputs.get("attention_mask")
    if attn is not None:
        mask = attn[0].detach().cpu().numpy().astype(bool)
        h0 = h0[mask]

    # drop common special tokens if tokenizer exposes ids
    special_ids = set()
    for attr in ("pad_token_id", "bos_token_id", "eos_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            special_ids.add(int(token_id))
    input_ids = inputs.get("input_ids")
    if input_ids is not None and special_ids:
        ids0 = input_ids[0].detach().cpu().numpy()
        # align ids0 with mask-filtered h0
        if attn is not None:
            ids0 = ids0[attn[0].detach().cpu().numpy().astype(bool)]
        keep = np.array([int(i) not in special_ids for i in ids0], dtype=bool)
        if keep.any():
            h0 = h0[keep]

    tokens = h0 @ proj  # (q_tokens, d_proj)
    tokens = _l2_normalize(tokens, axis=-1)
    pooled = tokens.mean(axis=0, keepdims=True)
    pooled = _l2_normalize(pooled, axis=-1)
    return tokens.astype("float32"), pooled.astype("float32")


def maxsim_score(q_tokens: np.ndarray, d_tokens: np.ndarray) -> float:
    """
    ColBERT-style late interaction:
      score(q, d) = sum_j max_i dot(q_j, d_i)
    Assumes tokens are L2-normalized.
    """
    sims = q_tokens @ d_tokens.T  # (q, d)
    return float(sims.max(axis=1).sum())


def build_phase6_index(
    pdf_path: str,
    index_dir: str = "phase6_index",
    model_name: str = "google/siglip-base-patch16-224",
    *,
    encoder: str = "hash",  # "hash" (offline) or "siglip" (requires model files)
    hf_local_only: bool = True,
    hash_seed: int = 0,
    hash_max_tokens: int = 128,
    page_dpi: int = 150,
    patch_size: int = 512,
    patch_overlap: int = 0,
    max_pages: int | None = None,
    batch_size: int = 8,
    proj_dim: int = 128,
    device: str | None = None,
) -> None:
    """
    Build a patch-level FAISS index:
      - Renders pages to images
      - Segments into patches
      - Embeds each patch (pooled) and stores in FAISS + metadata
    """
    device = device or _default_device()

    index = Path(index_dir)
    pages_dir = index / "pages"
    patches_dir = index / "patches"
    faiss_dir = index / "faiss_pooled"
    meta_path = index / "patches.jsonl"

    pages_dir.mkdir(parents=True, exist_ok=True)
    patches_dir.mkdir(parents=True, exist_ok=True)

    # 1) Render pages
    pages = pdf_pages_to_images(pdf_path, str(pages_dir), dpi=page_dpi, max_pages=max_pages)

    # 2) Segment pages into patches
    all_patches: list[Patch] = []
    for p in pages:
        all_patches.extend(
            image_to_patches(
                p["image_path"],
                page=p["page"],
                out_dir=str(patches_dir),
                patch_size=patch_size,
                overlap=patch_overlap,
            )
        )

    if not all_patches:
        raise RuntimeError("No patches created. Check PDF render and patch parameters.")

    if encoder.lower() == "hash":
        # Attach patch-region text from PDF for offline embedding + provenance.
        all_patches = _attach_patch_texts(pdf_path, pages, all_patches)

    # Persist metadata
    with meta_path.open("w", encoding="utf-8") as f:
        for p in all_patches:
            f.write(
                json.dumps(
                    {
                        "patch_id": p.patch_id,
                        "page": p.page,
                        "bbox": list(p.bbox),
                        "patch_path": p.patch_path,
                        "page_image_path": p.page_image_path,
                        "patch_text": p.text or "",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # 3) Embed patches (pooled) and build FAISS index
    encoder = encoder.lower().strip()
    pooled_vecs: list[np.ndarray] = []
    texts: list[str] = []
    metadatas: list[dict] = []

    if encoder == "siglip":
        vision_model, text_model, processor, tokenizer, vision_cfg = _load_siglip_or_raise(
            model_name,
            device=device,
            local_files_only=hf_local_only,
        )

        # Determine if vision model includes CLS token
        expected_patches = (vision_cfg.image_size // vision_cfg.patch_size) ** 2
        # Probe one patch quickly to detect token count
        dummy_proj = np.zeros((int(vision_cfg.hidden_size), 1), dtype="float32")
        probe_tokens, _ = _encode_image_tokens(
            vision_model,
            processor,
            [all_patches[0].patch_path],
            device=device,
            proj=dummy_proj,
            drop_cls=False,
        )
        # probe_tokens was projected with a dummy proj; only shape matters.
        seq_len = probe_tokens.shape[1]
        drop_cls = seq_len == expected_patches + 1

        # Real projection matrix
        proj = _ensure_projection(index_dir, in_dim=int(vision_cfg.hidden_size), out_dim=int(proj_dim))

        patch_paths = [p.patch_path for p in all_patches]
        for batch_paths, batch_patches in zip(_batched(patch_paths, batch_size), _batched(all_patches, batch_size)):
            _, pooled = _encode_image_tokens(
                vision_model,
                processor,
                batch_paths,
                device=device,
                proj=proj,
                drop_cls=drop_cls,
            )
            pooled_vecs.append(pooled)
            texts.extend([p.patch_id for p in batch_patches])
            metadatas.extend([p.to_metadata() for p in batch_patches])

        pooled_mat = np.vstack(pooled_vecs).astype("float32", copy=False)
    elif encoder == "hash":
        hash_enc = HashTextEncoder(dim=proj_dim, seed=hash_seed, max_tokens=hash_max_tokens)
        for batch_patches in _batched(all_patches, batch_size):
            pooled = []
            for p in batch_patches:
                _toks, pool = hash_enc.encode_text(p.text or "")
                pooled.append(pool[0])
            pooled_vecs.append(np.vstack(pooled).astype("float32", copy=False))
            texts.extend([p.patch_id for p in batch_patches])
            metadatas.extend([p.to_metadata() for p in batch_patches])
        pooled_mat = np.vstack(pooled_vecs).astype("float32", copy=False)
    else:
        raise ValueError(f"Unknown encoder: {encoder}. Use 'hash' or 'siglip'.")

    store = FAISSStore(dim=pooled_mat.shape[1], save_path=str(faiss_dir))
    # Reset to avoid accidentally appending to an existing index on rebuild.
    import faiss

    store.index = faiss.IndexFlatL2(pooled_mat.shape[1])
    store.texts = []
    store.metadata = []
    store.add(pooled_mat, texts, metadatas)

    # Store config for reproducibility
    with (index / "index_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "pdf_path": pdf_path,
                "encoder": encoder,
                "model_name": model_name,
                "hf_local_only": hf_local_only,
                "hash_seed": hash_seed,
                "hash_max_tokens": hash_max_tokens,
                "device": device,
                "page_dpi": page_dpi,
                "patch_size": patch_size,
                "patch_overlap": patch_overlap,
                "max_pages": max_pages,
                "proj_dim": proj_dim,
            },
            f,
            indent=2,
        )


def retrieve_patches_phase6(
    question: str | None = None,
    index_dir: str = "phase6_index",
    model_name: str = "google/siglip-base-patch16-224",
    *,
    query_image_path: str | None = None,
    encoder: str = "hash",
    hf_local_only: bool = True,
    hash_seed: int = 0,
    hash_max_tokens: int = 128,
    top_k: int = 5,
    rerank_k: int = 30,
    proj_dim: int = 128,
    device: str | None = None,
) -> list[tuple[Patch, float]]:
    """
    Two-stage retrieval:
      1) pooled FAISS search → candidates
      2) MaxSim rerank over token embeddings computed on-the-fly
    """
    device = device or _default_device()
    index = Path(index_dir)

    store = FAISSStore(dim=proj_dim, save_path=str(index / "faiss_pooled"))
    if not getattr(store, "metadata", None):
        raise FileNotFoundError(f"Phase 6 index not found or empty at {index/'faiss_pooled'}. Run `build` first.")

    encoder = encoder.lower().strip()
    if encoder == "siglip":
        vision_model, text_model, processor, tokenizer, vision_cfg = _load_siglip_or_raise(
            model_name, device=device, local_files_only=hf_local_only
        )
        proj = _ensure_projection(index_dir, in_dim=int(vision_cfg.hidden_size), out_dim=int(proj_dim))

        # Determine CLS presence deterministically from config
        expected_patches = (vision_cfg.image_size // vision_cfg.patch_size) ** 2
        # If model uses CLS, seq_len typically = expected_patches + 1
        # We'll assume CLS exists if vision_cfg has add_pooling_layer-like behavior;
        # for robustness, we probe one patch from stored metadata.
        probe_patch_path = (store.metadata[0] or {}).get("patch_path")
        if not probe_patch_path:
            raise RuntimeError("Phase 6 index metadata missing `patch_path` (rebuild the index).")
        dummy_proj = np.zeros((int(vision_cfg.hidden_size), 1), dtype="float32")
        probe_tokens, _ = _encode_image_tokens(
            vision_model,
            processor,
            [probe_patch_path],
            device=device,
            proj=dummy_proj,
            drop_cls=False,
        )
        drop_cls = probe_tokens.shape[1] == expected_patches + 1

        if query_image_path:
            b_tokens, b_pool = _encode_image_tokens(
                vision_model,
                processor,
                [query_image_path],
                device=device,
                proj=proj,
                drop_cls=drop_cls,
            )
            q_tokens = b_tokens[0]
            q_pool = b_pool
        else:
            q_tokens, q_pool = _encode_query_tokens(text_model, tokenizer, question or "", device=device, proj=proj)
    elif encoder == "hash":
        if query_image_path:
            raise ValueError("The 'hash' encoder does not support image queries. Please use 'siglip'.")
        hash_enc = HashTextEncoder(dim=proj_dim, seed=hash_seed, max_tokens=hash_max_tokens)
        q_tokens, q_pool = hash_enc.encode_text(question or "", max_tokens=64)
        vision_model = text_model = processor = tokenizer = vision_cfg = proj = drop_cls = None  # type: ignore[assignment]
    else:
        raise ValueError(f"Unknown encoder: {encoder}. Use 'hash' or 'siglip'.")

    # Stage 1: pooled FAISS
    pooled_hits = store.search(q_pool[0], k=rerank_k)
    candidates: list[Patch] = []
    for hit in pooled_hits:
        meta = hit.get("metadata", {}) or {}
        try:
            candidates.append(
                Patch(
                    patch_id=meta["patch_id"],
                    page=int(meta.get("page", 0)),
                    bbox=tuple(meta.get("bbox") or (0, 0, 0, 0)),
                    patch_path=meta["patch_path"],
                    page_image_path=meta["page_image_path"],
                    text=meta.get("patch_text") or None,
                )
            )
        except Exception:
            continue

    # Stage 2: MaxSim rerank
    scored: list[tuple[Patch, float]] = []
    if encoder == "siglip":
        rerank_batch = 8
        for batch in _batched(candidates, rerank_batch):
            d_tokens, _ = _encode_image_tokens(
                vision_model,
                processor,
                [p.patch_path for p in batch],
                device=device,
                proj=proj,
                drop_cls=drop_cls,
            )
            for p, toks in zip(batch, d_tokens):
                scored.append((p, maxsim_score(q_tokens, toks)))
    else:
        # Offline hash rerank uses patch text.
        for p in candidates:
            d_tokens, _d_pool = hash_enc.encode_text(p.text or "")
            scored.append((p, maxsim_score(q_tokens, d_tokens)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def annotate_sources(
    results: list[tuple[Patch, float]],
    out_dir: str,
    color: str = "red",
    width: int = 5,
) -> list[str]:
    """
    Draw bounding boxes for retrieved patches on their source page images.
    Returns list of written image paths.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    # Group by page for clearer attribution images
    by_page: dict[str, list[Patch]] = {}
    for p, _s in results:
        by_page.setdefault(p.page_image_path, []).append(p)

    for page_img_path, patches in by_page.items():
        with Image.open(page_img_path) as opened:
            img = opened.convert("RGB")
            draw = ImageDraw.Draw(img)
            for p in patches:
                draw.rectangle(p.bbox, outline=color, width=width)
            out_path = out / (Path(page_img_path).stem + "_attribution.png")
            img.save(out_path)
            written.append(str(out_path))

    return written


def answer_with_visual_context(
    question: str | None,
    patches: list[Patch],
    model_name: str | None = None,
    query_image_path: str | None = None,
) -> str:
    """
    Retrieve→Generate step: sends patch images as context to Vertex AI Gemini.
    Requires:
      - Working outbound network access
      - ADC credentials (gcloud auth application-default login)
      - GCP_PROJECT_ID set
    """
    from vertexai.generative_models import GenerativeModel, Part

    init_vertex()
    model = GenerativeModel(model_name or LLM_MODEL)

    parts: list[object] = []
    parts.append(
        "You are a financial analyst assistant. Use ONLY the provided image patches as evidence. "
        "If the answer is not present, say you don't know. "
        "Cite sources as (page N, patch_id)."
    )

    if query_image_path:
        parts.append("User uploaded query image:")
        img_bytes = Path(query_image_path).read_bytes()
        ext = Path(query_image_path).suffix.lower()
        mime_type = "image/png" if ext == ".png" else "image/jpeg"
        parts.append(Part.from_data(data=img_bytes, mime_type=mime_type))

    for i, p in enumerate(patches, start=1):
        img_bytes = Path(p.patch_path).read_bytes()
        # patches are saved as PNG
        parts.append(Part.from_data(data=img_bytes, mime_type="image/png"))
        parts.append(f"Patch {i}: page={p.page}, patch_id={p.patch_id}, bbox={p.bbox}")

    if question:
        parts.append(f"Question: {question}")
    resp = model.generate_content(parts)
    return resp.text


def compare_phase5_vs_phase6(question: str, phase6_index_dir: str) -> dict:
    """
    Returns a dict with best-effort retrieval outputs from Phase 5 and Phase 6.
    Phase 5 requires an existing Phase 5 index (recommended: FAISS backend).
    """
    out: dict = {"question": question, "phase5": None, "phase6": None}

    # Phase 6
    try:
        p6 = retrieve_patches_phase6(question, index_dir=phase6_index_dir, top_k=5)
        out["phase6"] = [
            {
                "page": p.page,
                "patch_id": p.patch_id,
                "bbox": list(p.bbox),
                "score": s,
            }
            for p, s in p6
        ]
    except Exception as e:
        out["phase6"] = {"error": f"{type(e).__name__}: {e}"}

    # Phase 5
    # If network/DNS is unavailable, Vertex AI calls can hang; fail fast.
    try:
        import socket

        socket.gethostbyname("us-central1-aiplatform.googleapis.com")
    except Exception:
        out["phase5"] = {"error": "Network/DNS unavailable; skipping Phase 5 compare."}
        return out

    try:
        from phase5_multimodal_rag import build_pipeline

        _pipeline, retriever, _store = build_pipeline("faiss", use_reranker=False)
        hits = retriever.retrieve(question, k=5, rerank=False)
        out["phase5"] = [
            {
                "content_type": h.get("metadata", {}).get("content_type"),
                "page": h.get("metadata", {}).get("page"),
                "score": h.get("score"),
                "snippet": (h.get("text") or "")[:200],
            }
            for h in hits
        ]
    except Exception as e:
        out["phase5"] = {"error": f"{type(e).__name__}: {e}"}

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6: ColPali-like multimodal RAG (patch-level).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build the Phase 6 patch index.")
    p_build.add_argument("--pdf", default=PDF_PATH)
    p_build.add_argument("--index-dir", default="phase6_index")
    p_build.add_argument("--encoder", choices=["hash", "siglip"], default="hash", help="Embedding backend")
    p_build.add_argument("--model", default="google/siglip-base-patch16-224")
    p_build.add_argument("--hf-local-only", action="store_true", help="SigLIP: do not try to download model files")
    p_build.add_argument("--hash-seed", type=int, default=0, help="Hash encoder: deterministic seed")
    p_build.add_argument("--hash-max-tokens", type=int, default=128, help="Hash encoder: max tokens per patch")
    p_build.add_argument("--page-dpi", type=int, default=150)
    p_build.add_argument("--patch-size", type=int, default=512)
    p_build.add_argument("--patch-overlap", type=int, default=0)
    p_build.add_argument("--max-pages", type=int, default=None)
    p_build.add_argument("--batch-size", type=int, default=8)
    p_build.add_argument("--proj-dim", type=int, default=128)
    p_build.add_argument("--device", default=None)

    p_query = sub.add_parser("query", help="Retrieve patches with MaxSim rerank + optional attribution images.")
    p_query.add_argument("-q", "--question", required=True)
    p_query.add_argument("--index-dir", default="phase6_index")
    p_query.add_argument("--encoder", choices=["hash", "siglip"], default="hash", help="Embedding backend")
    p_query.add_argument("--model", default="google/siglip-base-patch16-224")
    p_query.add_argument("--hf-local-only", action="store_true", help="SigLIP: do not try to download model files")
    p_query.add_argument("--hash-seed", type=int, default=0, help="Hash encoder: deterministic seed")
    p_query.add_argument("--hash-max-tokens", type=int, default=128, help="Hash encoder: max tokens per patch")
    p_query.add_argument("--top-k", type=int, default=5)
    p_query.add_argument("--rerank-k", type=int, default=30)
    p_query.add_argument("--proj-dim", type=int, default=128)
    p_query.add_argument("--device", default=None)
    p_query.add_argument("--annotate", action="store_true", help="Write page images with highlighted patch boxes.")

    p_ask = sub.add_parser("ask", help="Retrieve patches and ask Gemini using patch images as context.")
    p_ask.add_argument("-q", "--question", required=True)
    p_ask.add_argument("--index-dir", default="phase6_index")
    p_ask.add_argument("--encoder", choices=["hash", "siglip"], default="hash", help="Embedding backend")
    p_ask.add_argument("--model", default="google/siglip-base-patch16-224")
    p_ask.add_argument("--hf-local-only", action="store_true", help="SigLIP: do not try to download model files")
    p_ask.add_argument("--hash-seed", type=int, default=0, help="Hash encoder: deterministic seed")
    p_ask.add_argument("--hash-max-tokens", type=int, default=128, help="Hash encoder: max tokens per patch")
    p_ask.add_argument("--top-k", type=int, default=5)
    p_ask.add_argument("--rerank-k", type=int, default=30)
    p_ask.add_argument("--proj-dim", type=int, default=128)
    p_ask.add_argument("--device", default=None)
    p_ask.add_argument("--n-patches", type=int, default=3)
    p_ask.add_argument("--gemini-model", default=None)

    p_cmp = sub.add_parser("compare", help="Compare Phase 5 retrieval vs Phase 6 retrieval.")
    p_cmp.add_argument("-q", "--question", required=True)
    p_cmp.add_argument("--index-dir", default="phase6_index")

    args = parser.parse_args()

    if args.cmd == "build":
        build_phase6_index(
            pdf_path=args.pdf,
            index_dir=args.index_dir,
            model_name=args.model,
            encoder=args.encoder,
            hf_local_only=bool(args.hf_local_only),
            hash_seed=int(args.hash_seed),
            hash_max_tokens=int(args.hash_max_tokens),
            page_dpi=args.page_dpi,
            patch_size=args.patch_size,
            patch_overlap=args.patch_overlap,
            max_pages=args.max_pages,
            batch_size=args.batch_size,
            proj_dim=args.proj_dim,
            device=args.device,
        )
        print(f"[Phase 6] Index built at: {args.index_dir}")
        return

    if args.cmd == "query":
        results = retrieve_patches_phase6(
            args.question,
            index_dir=args.index_dir,
            model_name=args.model,
            encoder=args.encoder,
            hf_local_only=bool(args.hf_local_only),
            hash_seed=int(args.hash_seed),
            hash_max_tokens=int(args.hash_max_tokens),
            top_k=args.top_k,
            rerank_k=args.rerank_k,
            proj_dim=args.proj_dim,
            device=args.device,
        )
        for rank, (p, s) in enumerate(results, start=1):
            print(f"#{rank} score={s:.4f} page={p.page} patch_id={p.patch_id} bbox={p.bbox}")

        if args.annotate:
            out_dir = str(Path(args.index_dir) / "attribution" / f"q_{_sha1(args.question)}")
            written = annotate_sources(results, out_dir=out_dir)
            for w in written:
                print(f"[attribution] wrote {w}")
        return

    if args.cmd == "ask":
        results = retrieve_patches_phase6(
            args.question,
            index_dir=args.index_dir,
            model_name=args.model,
            encoder=args.encoder,
            hf_local_only=bool(args.hf_local_only),
            hash_seed=int(args.hash_seed),
            hash_max_tokens=int(args.hash_max_tokens),
            top_k=max(args.top_k, args.n_patches),
            rerank_k=args.rerank_k,
            proj_dim=args.proj_dim,
            device=args.device,
        )
        selected = [p for (p, _s) in results[: args.n_patches]]
        answer = answer_with_visual_context(args.question, selected, model_name=args.gemini_model)
        print(answer)
        return

    if args.cmd == "compare":
        comp = compare_phase5_vs_phase6(args.question, phase6_index_dir=args.index_dir)
        print(json.dumps(comp, indent=2))
        return


if __name__ == "__main__":
    main()
