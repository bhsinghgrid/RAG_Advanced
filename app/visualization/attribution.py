from PIL import Image, ImageDraw
import io
import base64
from typing import Tuple


def render_bbox_overlay_from_image(pil_image: Image.Image, bbox: Tuple[int,int,int,int], outline_color: str = "red", width: int = 4) -> str:
    """
    Draws a bounding box on the given PIL image and returns a base64 PNG data URI.

    Args:
        pil_image: Source PIL Image (page image)
        bbox: (x0,y0,x1,y1) bounding box in image coordinates
        outline_color: Color for the box
        width: Line width

    Returns:
        data URI string containing PNG image with overlay
    """
    img = pil_image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255,255,255,0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(bbox, outline=outline_color, width=width)
    composed = Image.alpha_composite(img, overlay).convert("RGB")

    buf = io.BytesIO()
    composed.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def render_bbox_overlay_from_pdf(pdf_page_image: Image.Image, bbox: Tuple[int,int,int,int], outline_color: str = "red", width: int = 4) -> str:
    """Alias for clarity when using page images."""
    return render_bbox_overlay_from_image(pdf_page_image, bbox, outline_color=outline_color, width=width)


if __name__ == "__main__":
    # Quick local test requires VisualExtractor to produce an image; not run here.
    print("attribution utilities loaded")
