from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


DOCUMENTS_DIR = BASE_DIR / "Dokumente"
CHROMA_DIR = BASE_DIR / "chroma_db"

EMBEDDING_MODEL = "nomic-embed-text"

LLM_MODEL = "qwen3.6:latest"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls", ".txt", ".md"}