from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

from google import genai
from google.genai import types
from pydantic import BaseModel
from app.config import GCP_PROJECT_ID, GCP_LOCATION, LLM_MODEL

@dataclass(frozen=True)
class ImageDescription:
    image_type: str
    text: str
    structured_data: Dict[str, Any]

class ChartImageDescriber:
    """Generates a retrieval-friendly text description of a chart/graph image.
    Uses Gemini 2.0 Flash via google-genai SDK with structured output.
    """

    def __init__(
        self,
        model_name: str = LLM_MODEL,
        *,
        cache_dir: str = "data/phase5_images/descriptions",
    ) -> None:
        self.client = genai.Client(
            vertexai=True,
            project=GCP_PROJECT_ID,
            location=GCP_LOCATION
        )
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def describe(self, image_path: str, *, cache_key: Optional[str] = None) -> ImageDescription:
        cache_key = cache_key or Path(image_path).stem
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        import json
        if cache_file.exists():
            with open(cache_file, "r") as f:
                data = json.load(f)
                return ImageDescription(
                    image_type=data.get("image_type", "unknown"),
                    text=data.get("summary", ""),
                    structured_data=data
                )

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        prompt = (
            "You are extracting visual data from a PDF image. "
            "If the image is a chart/graph, extract the key information needed to answer questions. "
            "If it is a table, extract the data. If it is a photo, describe it."
        )

        class ImageAnalysis(BaseModel):
            image_type: str # chart, graph, diagram, table, photo, logo, other
            title: Optional[str]
            x_axis: Optional[str]
            y_axis: Optional[str]
            legend: Optional[str]
            key_values: Dict[str, str]
            summary: str

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png")
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ImageAnalysis,
                )
            )
            
            result = response.parsed
            if not result:
                # Fallback if parsing failed
                result_dict = json.loads(response.text)
            else:
                result_dict = result.model_dump()
                
            with open(cache_file, "w") as f:
                json.dump(result_dict, f, indent=2)

            return ImageDescription(
                image_type=result_dict.get("image_type", "unknown"),
                text=result_dict.get("summary", ""),
                structured_data=result_dict
            )

        except Exception as e:
            error_data = {"image_type": "unknown", "summary": f"Failed to analyze image ({e})"}
            return ImageDescription(image_type="unknown", text=error_data["summary"], structured_data=error_data)
