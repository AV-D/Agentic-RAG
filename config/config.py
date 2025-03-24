import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DIALOGUE_MANAGER_MODEL = os.getenv("DIALOGUE_MANAGER_MODEL", "deepseek-r1:latest")
RETRIEVER_MODEL = os.getenv("RETRIEVER_MODEL", "deepseek-r1:latest")
EVALUATION_MODEL = os.getenv("EVALUATION_MODEL", "deepseek-r1:latest")
ANSWER_GENERATION_MODEL = os.getenv("ANSWER_GENERATION_MODEL", "deepseek-r1:latest")
HALLUCINATION_CHECK_MODEL = os.getenv("HALLUCINATION_CHECK_MODEL", "deepseek-r1:latest")

# Vector DB settings
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "./data/vector_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large:latest")

# Web search settings
WEB_SEARCH_ENABLED = os.getenv("WEB_SEARCH_ENABLED", "True").lower() == "true"
MAX_WEB_RESULTS = int(os.getenv("MAX_WEB_RESULTS", "3"))

# Retrieval settings
RETRIEVAL_CHUNK_SIZE = int(os.getenv("RETRIEVAL_CHUNK_SIZE", "1000"))
RETRIEVAL_CHUNK_OVERLAP = int(os.getenv("RETRIEVAL_CHUNK_OVERLAP", "200"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))

# Evaluation thresholds
RETRIEVAL_QUALITY_THRESHOLD = float(os.getenv("RETRIEVAL_QUALITY_THRESHOLD", "0.7"))
ANSWER_QUALITY_THRESHOLD = float(os.getenv("ANSWER_QUALITY_THRESHOLD", "0.7"))
HALLUCINATION_THRESHOLD = float(os.getenv("HALLUCINATION_THRESHOLD", "0.3")) 