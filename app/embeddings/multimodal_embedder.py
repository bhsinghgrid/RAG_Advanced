from typing import List, Any
from PIL import Image
import numpy as np
import io

# Lazy imports for optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBM = True
except Exception:
    _HAS_SBM = False

# Fallback to text embedder if available
try:
    from app.embeddings.embedder import Embedder as TextEmbedder
    _HAS_TEXT_EMBEDDER = True
except Exception:
    _HAS_TEXT_EMBEDDER = False

# Optional OCR support
try:
    import pytesseract
    _HAS_PYTESSERACT = True
except Exception:
    pytesseract = None
    _HAS_PYTESSERACT = False


class MultimodalEmbedder:
    """
    Adapter to produce embeddings for images and text. Attempts to use a CLIP-like
    model via `sentence-transformers` when available; otherwise falls back to the
    existing text embedder by running OCR or textual summaries.

    The `use_ocr` flag enables OCR-based text extraction for image patches when
    `pytesseract` and a text embedder are available.
    """

    def __init__(self, model_name: str = "clip-ViT-B-32", use_ocr: bool = True):
        self.model_name = model_name
        self.model = None
        self.use_ocr = use_ocr and _HAS_PYTESSERACT

        if _HAS_SBM:
            try:
                # sentence-transformers supports several CLIP variants
                self.model = SentenceTransformer(self.model_name)
            except Exception:
                self.model = None

        if self.model is None and _HAS_TEXT_EMBEDDER:
            # We'll use the text embedder for fallback text-based embeddings
            self.text_embedder = TextEmbedder()
        else:
            self.text_embedder = None

    def embed_image(self, pil_image: Image.Image) -> List[float]:
        """Return an embedding vector for a PIL image."""
        if self.model is not None:
            try:
                # SentenceTransformer can accept PIL images for some models
                emb = self.model.encode(pil_image, convert_to_numpy=True)
                return emb.tolist()
            except Exception:
                pass

        # If OCR available and enabled, extract text and embed that
        if self.use_ocr and pytesseract is not None and self.text_embedder is not None:
            try:
                if pil_image.mode != "RGB":
                    img = pil_image.convert("RGB")
                else:
                    img = pil_image
                text = pytesseract.image_to_string(img)
                if text and text.strip():
                    return self.text_embedder.embed([text])[0]
            except Exception:
                pass

        # Fallback: convert image to text via simple heuristic or raise
        if self.text_embedder is not None:
            desc = "Image patch containing visual document content"
            emb = self.text_embedder.embed([desc])[0]
            return emb

        raise RuntimeError("No image embedding backend available. Install sentence-transformers, pytesseract + text embedder, or provide a text embedder.")

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts using whichever backend is available."""
        if self.model is not None:
            emb = self.model.encode(texts, convert_to_numpy=False)
            # SentenceTransformer may return a numpy array or a Python list
            if hasattr(emb, "tolist"):
                return emb.tolist()
            return list(emb)

        if self.text_embedder is not None:
            return self.text_embedder.embed(texts)

        raise RuntimeError("No text embedding backend available.")

    def embed_patch(self, patch_dict: dict) -> List[float]:
        """Convenience: embed a patch dictionary produced by VisualExtractor."""
        pil_img = patch_dict.get("patch_image")
        if pil_img is None:
            raise ValueError("patch_dict missing 'patch_image'")
        return self.embed_image(pil_img)


if __name__ == "__main__":
    # Quick smoke test (no heavy model load to keep import safe)
    me = MultimodalEmbedder()
    print("MultimodalEmbedder initialized. Model available:", me.model is not None)
    
