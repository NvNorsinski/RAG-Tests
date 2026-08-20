from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


DOCUMENTS_DIR = BASE_DIR / "Dokumente"
CHROMA_DIR = BASE_DIR / "chroma_db"

EMBEDDING_MODEL = "nomic-embed-text"

LLM_MODEL = "gemma4:e2b"

# Judge model for DeepEval metrics (separate from the main LLM)
DEEPEVAL_JUDGE_MODEL = "gemma4:e2b"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls", ".txt", ".md"}


VECTOR_WEIGHT = 0.6 
BM25_WEIGHT = 0.4
RETRIEVER_K = 6
RETRIEVER_FETCH_K = 30
MMR_LAMBDA = 0.5

