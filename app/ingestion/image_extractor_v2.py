from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import hashlib


@dataclass(frozen=True)
class ExtractedImage:
    page: int
    image_index: int
    xref: int
    bbox: Optional[Tuple[float, float, float, float]]
    width: int
    height: int
    sha256: str
    path: str


class ImageExtractorV2:
    """Extract raster images from a PDF using PyMuPDF.

    Notes:
    - This extracts embedded raster images. Vector charts/graphs may not be extractable
      as standalone images; handle those via page rendering if needed.
    """

    def __init__(
        self,
        *,
        output_dir: str = "data/phase5_images",
        min_pixels: int = 40_000,
        max_images: Optional[int] = None,
        deduplicate: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.min_pixels = int(min_pixels)
        self.max_images = max_images
        self.deduplicate = deduplicate

    def extract(self, pdf_path: str) -> List[ExtractedImage]:
        import fitz  # PyMuPDF

        self.output_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(pdf_path)
        extracted: List[ExtractedImage] = []
        sha_to_path: dict[str, str] = {}

        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            infos = page.get_image_info(hashes=True, xrefs=True) or []

            for idx, info in enumerate(infos):
                xref = int(info.get("xref") or 0)
                if xref <= 0:
                    continue

                width = int(info.get("width") or 0)
                height = int(info.get("height") or 0)
                if width * height < self.min_pixels:
                    continue

                img = doc.extract_image(xref)
                if not img or not img.get("image"):
                    continue

                data: bytes = img["image"]
                ext = (img.get("ext") or "png").lower()
                sha = hashlib.sha256(data).hexdigest()

                if self.deduplicate and sha in sha_to_path:
                    path = sha_to_path[sha]
                else:
                    filename = f"p{page_num:03d}_img{idx:02d}_{sha[:10]}.{ext}"
                    out_path = self.output_dir / filename
                    path = str(out_path)
                    with open(out_path, "wb") as f:
                        f.write(data)
                    sha_to_path[sha] = path

                bbox = info.get("bbox")
                try:
                    bbox = tuple(bbox) if bbox is not None else None
                except Exception:
                    bbox = None

                # Capture surrounding text (e.g., text within 50 pixels of the image bbox)
                surrounding_text = ""
                if bbox:
                    # Expand bbox slightly to find nearby text
                    expanded_bbox = (bbox[0]-20, bbox[1]-20, bbox[2]+20, bbox[3]+20)
                    surrounding_text = page.get_text("text", clip=expanded_bbox).strip()

                extracted.append(
                    ExtractedImage(
                        page=page_num,
                        image_index=idx,
                        xref=xref,
                        bbox=bbox,
                        width=width,
                        height=height,
                        sha256=sha,
                        path=path,
                    )
                )
                # We'll store the surrounding text in a separate field in the result dict if needed
                # For now, let's just make sure it's accessible.
                extracted[-1].__dict__["surrounding_text"] = surrounding_text

                if self.max_images is not None and len(extracted) >= self.max_images:
                    return extracted

        return extracted


if __name__ == "__main__":
    extractor = ImageExtractorV2()
    imgs = extractor.extract("ifc-annual-report-2024-financials.pdf")
    print(f"Extracted {len(imgs)} images")
    if imgs:
        print(imgs[0])
