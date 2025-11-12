import os

# LLM_MODEL_NAME = "ilsp/Llama-Krikri-3B-Instruct"

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
INSTRUCTIONS_DIR = DATA_DIR / "instructions"
INDEX_DIR = BASE_DIR / "faiss_index"
OFFLOAD_DIR = BASE_DIR / "offload"

# NEW: Offline models directory
OFFLINE_MODELS_DIR = BASE_DIR / "offline_models"  # Or your custom path

KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
INSTRUCTIONS_DIR.mkdir(parents=True, exist_ok=True)
OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MODEL CONFIGURATION - OFFLINE MODE
# ============================================================================

# Use local model paths instead of HuggingFace names
LLM_MODEL_NAME = str(OFFLINE_MODELS_DIR / "llm" / "meta-llama--Llama-3.2-3B-Instruct")

EMBEDDING_MODEL_NAME = str(OFFLINE_MODELS_DIR / "embeddings" / "sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2")

# Verify models exist
if not Path(LLM_MODEL_NAME).exists():
    raise FileNotFoundError(
        f"LLM model not found at: {LLM_MODEL_NAME}\n"
        f"Please download models first using download_models.py"
    )

if not Path(EMBEDDING_MODEL_NAME).exists():
    raise FileNotFoundError(
        f"Embedding model not found at: {EMBEDDING_MODEL_NAME}\n"
        f"Please download models first using download_models.py"
    )

# Rest of your config...
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
    "min_relevance_score": 0.2,  # Lowered for better recall
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
    
    # Add knowledge base instruction
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

# ============================================================================
# AGENT CONFIGURATION
# ============================================================================

# Directories where the agent can read files
# Add your military network paths here
AGENT_ALLOWED_DIRECTORIES = [
    DATA_DIR,  # Already defined: BASE_DIR / "data"
    KNOWLEDGE_DIR,  # Already defined: DATA_DIR / "knowledge"
    INSTRUCTIONS_DIR,  # Already defined: DATA_DIR / "instructions"
    BASE_DIR / "logs",  # If you have logs directory
    BASE_DIR / "config",  # If you have config directory
    # Add more directories as needed for your military network:
    # Path("/opt/military_app/data"),
    # Path("/var/log/military_app"),
]

# Maximum file size the agent can read (in MB)
AGENT_MAX_FILE_SIZE_MB = 10

# Allowed file extensions for reading
AGENT_ALLOWED_EXTENSIONS = {
    '.txt', '.md', '.text',  # Text files
    '.json', '.yaml', '.yml',  # Data files
    '.conf', '.config', '.cfg',  # Config files
    '.log',  # Log files
    '.py', '.sh', '.bash',  # Scripts (read-only)
    '.xml', '.html', '.csv',  # Markup and data
}

AGENT_SHOW_FILE_CONTENT = True  # Include file content in response
AGENT_FILE_CONTENT_FORMAT = "pretty"  # Options: "pretty", "raw", "minimal"
AGENT_MAX_CONTENT_DISPLAY_LINES = None  # None = show all, or set a number like 100