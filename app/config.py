import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
INSTRUCTIONS_DIR = DATA_DIR / "instructions"
INDEX_DIR = BASE_DIR / "faiss_index"
OFFLOAD_DIR = BASE_DIR / "offload"

# Offline models directory
OFFLINE_MODELS_DIR = BASE_DIR / "offline_models"

KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
INSTRUCTIONS_DIR.mkdir(parents=True, exist_ok=True)
OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Use local model paths instead of HuggingFace names
LLM_MODEL_NAME = str(OFFLINE_MODELS_DIR / "llm" / "meta-llama--Llama-3.2-3B-Instruct")
EMBEDDING_MODEL_NAME = str(OFFLINE_MODELS_DIR / "embeddings" / "sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2")

# Verify models exist (optional check - comment out if not needed)
if not Path(LLM_MODEL_NAME).exists():
    # Use fallback to HF model name if local not found
    LLM_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

if not Path(EMBEDDING_MODEL_NAME).exists():
    # Use fallback to HF model name if local not found
    EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

LLM_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.1,
}

RAG_CONFIG = {
    "top_k": 3,
    "chunk_size": 500,
    "chunk_overlap": 50,
    "min_relevance_score": 0.2,
}

# ============================================================================
# NEW AGENT CONFIGURATION
# ============================================================================

AGENT_MODE = os.getenv("AGENT_MODE", "production")  # production, development, military
AGENT_USE_CACHE = os.getenv("AGENT_USE_CACHE", "false").lower() == "true"
AGENT_DEBUG_MODE = os.getenv("AGENT_DEBUG", "false").lower() == "true"

# IMPORTANT: Use in-memory storage for performance
USE_IN_MEMORY_STORAGE = True  # Set to False only if you need persistence
CACHE_MODELS_IN_MEMORY = True  # Keep models loaded in memory
CACHE_EMBEDDINGS = True  # Cache computed embeddings

# Agent tool configuration
AGENT_ALLOWED_DIRECTORIES = [
    DATA_DIR,
    KNOWLEDGE_DIR,
    INSTRUCTIONS_DIR,
    BASE_DIR / "logs",
    BASE_DIR / "config",
]

AGENT_MAX_FILE_SIZE_MB = 10

AGENT_ALLOWED_EXTENSIONS = {
    '.txt', '.md', '.text',
    '.json', '.yaml', '.yml',
    '.conf', '.config', '.cfg',
    '.log',
    '.py', '.sh', '.bash',
    '.xml', '.html', '.csv',
}

AGENT_ALLOWED_COMMANDS = ["ls", "pwd", "echo", "date", "whoami", "df", "free"]

AGENT_SHOW_FILE_CONTENT = True
AGENT_FILE_CONTENT_FORMAT = "pretty"  # Options: "pretty", "raw", "minimal"
AGENT_MAX_CONTENT_DISPLAY_LINES = None

# ============================================================================
# API CONFIGURATION
# ============================================================================

CORS_ORIGINS = [
    "http://localhost:4200",
    "http://127.0.0.1:4200",
]

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================================================
# SYSTEM INSTRUCTIONS
# ============================================================================

def load_system_instructions():
    persona_file = INSTRUCTIONS_DIR / "persona.txt"
    rules_file = INSTRUCTIONS_DIR / "rules.txt"
    
    persona = ""
    rules = ""
    
    if persona_file.exists():
        persona = persona_file.read_text(encoding="utf-8")
    
    if rules_file.exists():
        rules = rules_file.read_text(encoding="utf-8")
    
    knowledge_instruction = """

## IMPORTANT: YOUR KNOWLEDGE BASE

You have access to a knowledge base containing information about:
- Panos (the user)
- System documentation
- Configuration files
- Procedures and guides

When asked about information, you should:
1. Use the context provided in <context> tags
2. Answer based on the knowledge base first
3. If information is not in the knowledge base, say so clearly

You are NOT limited to general knowledge - you have specific information 
about Panos and this system in your knowledge base.
"""
    
    combined = f"{persona}\n\n{rules}\n\n{knowledge_instruction}".strip()
    
    return combined if combined else """
Your Name is Panos and you are a random 28 years old dude

## IMPORTANT: YOUR KNOWLEDGE BASE

You have access to a knowledge base with specific information.
When context is provided in <context> tags, use it to answer questions.
"""

SYSTEM_INSTRUCTION = load_system_instructions()