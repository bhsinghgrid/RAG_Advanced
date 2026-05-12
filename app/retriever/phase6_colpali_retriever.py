from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from phase6_colpali_like import retrieve_patches_phase6


class Phase6PatchRetriever(BaseRetriever):
    """LangChain wrapper around the Phase 6 patch retriever.

    Notes:
    - Uses the Phase 6 on-disk index built by `phase6_colpali_like.py build`.
    - Returns `langchain_core.documents.Document` with patch metadata attached.
    """

    index_dir: str = "phase6_index"
    encoder: str = "hash"
    model_name: str = "google/siglip-base-patch16-224"
    hf_local_only: bool = True
    hash_seed: int = 0
    hash_max_tokens: int = 128
    top_k: int = 5
    rerank_k: int = 30
    proj_dim: int = 128

    def _get_relevant_documents(self, query: str) -> List[Document]:
        results = retrieve_patches_phase6(
            query,
            index_dir=self.index_dir,
            model_name=self.model_name,
            encoder=self.encoder,
            hf_local_only=self.hf_local_only,
            hash_seed=self.hash_seed,
            hash_max_tokens=self.hash_max_tokens,
            top_k=self.top_k,
            rerank_k=self.rerank_k,
            proj_dim=self.proj_dim,
        )

        docs: List[Document] = []
        for patch, score in results:
            meta = {
                "phase": 6,
                "content_type": "image_patch",
                "patch_id": patch.patch_id,
                "page": patch.page,
                "bbox": patch.bbox,
                "patch_path": patch.patch_path,
                "page_image_path": patch.page_image_path,
                "score": score,
            }
            if patch.text:
                meta["patch_text"] = patch.text

            docs.append(Document(page_content=patch.text or patch.patch_id, metadata=meta))

        return docs

