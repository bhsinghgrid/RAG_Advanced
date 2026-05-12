import os
import yaml
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env
load_dotenv()

# Path to config.yaml
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

# Default configuration
config = {}
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

# PDF Configuration
PDF_PATH = os.getenv("PDF_PATH", config.get("pdf", {}).get("path", "ifc-annual-report-2024-financials.pdf"))

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", config.get("chunking", {}).get("size", 1000)))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", config.get("chunking", {}).get("overlap", 200)))

# GCP Configuration
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", config.get("gcp", {}).get("project_id", ""))
GCP_LOCATION = os.getenv("GCP_LOCATION", config.get("gcp", {}).get("location", "us-central1"))

# Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", config.get("embeddings", {}).get("model_name", "text-embedding-004"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", config.get("embeddings", {}).get("batch_size", 50)))
LLM_MODEL = os.getenv("LLM_MODEL", config.get("llm", {}).get("model_name", "gemini-2.0-flash"))

# Vector Store Configuration
VECTOR_DIM = int(os.getenv("VECTOR_DIM", config.get("vectorstore", {}).get("dimension", 768)))
QDRANT_HOST = os.getenv("QDRANT_HOST", config.get("qdrant", {}).get("host", "localhost"))
QDRANT_PORT = int(os.getenv("QDRANT_PORT", config.get("qdrant", {}).get("port", 6333)))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", config.get("qdrant", {}).get("collection", "ifc_annual_report"))
QDRANT_CACHE_COLLECTION = os.getenv("QDRANT_CACHE_COLLECTION", config.get("qdrant", {}).get("cache_collection", "semantic_cache"))

# Langfuse Configuration
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

if __name__ == "__main__":
    print(f"Loaded Configuration:")
    print(f"PDF_PATH: {PDF_PATH}")
    print(f"CHUNK_SIZE: {CHUNK_SIZE}")
    print(f"GCP_PROJECT_ID: {GCP_PROJECT_ID}")
    print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
