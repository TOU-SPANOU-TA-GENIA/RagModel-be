import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
INSTRUCTIONS_DIR = DATA_DIR / "instructions"
INDEX_DIR = BASE_DIR / "faiss_index"
OFFLOAD_DIR = BASE_DIR / "offload"

KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
INSTRUCTIONS_DIR.mkdir(parents=True, exist_ok=True)
OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Greek-specific
# LLM_MODEL_NAME = "ilsp/Llama-Krikri-3B-Instruct"

# LLM_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

# Only for testing
LLM_MODEL_NAME = "google/gemma-2b-it"

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

LLM_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.7, # we control randomeness on the answers
    "top_p": 0.9, # similar with temoerature but is percentage
    "do_sample": True,
    "repetition_penalty": 1.1,
}

RAG_CONFIG = {
    "top_k": 3,  # Number of documents to retrieve
    "chunk_size": 500,  # Characters per chunk
    "chunk_overlap": 50,  # Overlap between chunks
    "min_relevance_score": 0.5,  # Minimum similarity score (0-1)
}

# for front requests to back
CORS_ORIGINS = [
    "http://localhost:4200",
    "http://127.0.0.1:4200",
]

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def load_system_instructions():
    persona_file = INSTRUCTIONS_DIR / "persona.txt"
    rules_file = INSTRUCTIONS_DIR / "rules.txt"
    
    persona = ""
    rules = ""
    
    if persona_file.exists():
        persona = persona_file.read_text(encoding="utf-8")
    
    if rules_file.exists():
        rules = rules_file.read_text(encoding="utf-8")
    
    return f"{persona}\n\n{rules}".strip()

SYSTEM_INSTRUCTION = load_system_instructions() or """
Your Name is Panos and you are a random 28 years old dude
"""