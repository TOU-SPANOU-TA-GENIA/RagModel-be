# AI Agent RAG System

A modular AI agent with Retrieval-Augmented Generation (RAG) capabilities, designed for offline deployment.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the System](#running-the-system)
- [Testing](#testing)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. setup
cd ragmodel-be

# 2. Create virtual environment
conda create -n ragmodel_10 python=3.10
conda activate ragmodel_10

# 3. Install dependencies
pip install -r requirements.txt
pip install pytest

# 4. Download models (requires internet)
python scripts/download_models.py

# 5. Add documents to knowledge base
cp your_documents/*.txt data/knowledge/

# 6. Start the system
python scripts/run.py --full

# 7. Test it
python tests/cli/chat_client.py "Hello, what can you do?"
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ RAM (16GB recommended)
- 6GB+ GPU VRAM (for local LLM)

### Step 1: Environment Setup

**Using Conda (recommended):**
```bash
conda create -n ragmodel_10 python=3.10
conda activate ragmodel_10
```

### Step 2: Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# For GPU support (if not already installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Download Models

Models must be downloaded before first use:

```bash
# Download default models from config
python scripts/download_models.py

# Or specify custom models
python scripts/download_models.py --llm "meta-llama/Llama-3.2-1B-Instruct"
```

### Step 4: Verify Installation

```bash
# Run basic test
python -m pytest tests/unit/test_config.py -v
```

---

## Configuration

Configuration is managed through `config.json`. You can modify settings via:

### 1. Edit config.json directly

```json
{
  "llm": {
    "model_name": "meta-llama/Llama-3.2-3B-Instruct",
    "temperature": 0.7,
    "max_new_tokens": 2048
  },
  "rag": {
    "top_k": 3,
    "chunk_size": 500
  }
}
```

### 3. Use environment variables

```bash
export AGENT_MODE=development
export AGENT_DEBUG=true
```

### Key Configuration Options

| Category | Setting | Default | Description |
|----------|---------|---------|-------------|
| llm | model_name | Llama-3.2-3B | LLM model to use |
| llm | temperature | 0.7 | Response creativity (0-2) |
| llm | max_new_tokens | 2048 | Max response length |
| llm | quantization | 4bit | Model quantization |
| rag | top_k | 3 | Number of documents to retrieve |
| rag | chunk_size | 500 | Document chunk size |
| agent | mode | production | Agent mode |
| agent | debug_mode | false | Enable debug output |

---

## Running the System

### Development Mode

```bash
# Full setup: ingest documents + start server
python scripts/run.py --full

# Or step by step:
python scripts/run.py --ingest   # Build knowledge base
python scripts/run.py --run      # Start server
```

### Production Mode

```bash
# With custom host/port
python scripts/run.py --run --host 0.0.0.0 --port 8080
```


## Testing

### Run All Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=app --cov-report=html

# Verbose output
pytest -v
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# End-to-end tests
pytest tests/e2e/ -v

# Skip GPU tests
pytest -m "not gpu"
```

### Interactive Testing

```bash
# CLI chat client
python tests/cli/chat_client.py

# Single query
python tests/cli/chat_client.py "What is Python?"

# Interactive mode
python tests/cli/chat_client.py -i
```

### Test Configuration

```bash
# Test with mock LLM (no GPU needed)
AGENT_MODE=development pytest tests/unit/

# Test specific component
pytest tests/unit/test_agent.py::TestIntentClassifier -v
```

## Examples

### Example 1: Simple Q&A

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={"role": "user", "content": "What is machine learning?"}
)

print(response.json()["answer"])
```

### Example 2: File Operations

```python
# Ask to read a file
response = requests.post(
    "http://localhost:8000/chat",
    json={"role": "user", "content": "Read the file config.txt"}
)

print(response.json()["answer"])
```

### Example 3: Document Generation

```python
# Create a document
response = requests.post(
    "http://localhost:8000/chat",
    json={
        "role": "user",
        "content": "Create a Word document about Python basics"
    }
)

# Download the created document
print(response.json()["tool_result"])
```

### Example 4: Conversation with Memory

```python
import requests

session_id = None

# First message
r1 = requests.post(
    "http://localhost:8000/chat",
    json={"role": "user", "content": "My name is Alice"}
)
session_id = r1.json()["session_id"]

# Second message (remembers context)
r2 = requests.post(
    "http://localhost:8000/chat",
    params={"session_id": session_id},
    json={"role": "user", "content": "What is my name?"}
)

print(r2.json()["answer"])  # Should mention "Alice"
```

### Example 5: Custom Instructions

```python
# Set a custom instruction
r1 = requests.post(
    "http://localhost:8000/chat",
    json={
        "role": "user",
        "content": "When I say 'weather', respond with 'Check the sky!'"
    }
)
session_id = r1.json()["session_id"]

# Test the instruction
r2 = requests.post(
    "http://localhost:8000/chat",
    params={"session_id": session_id},
    json={"role": "user", "content": "weather"}
)

print(r2.json()["answer"])  # Should say "Check the sky!"
```

---

## Troubleshooting

### Common Issues

**1. CUDA not available**
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Install correct PyTorch version
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**2. Out of memory**
```bash
# Use smaller model
python scripts/download_models.py --llm "meta-llama/Llama-3.2-1B-Instruct"

# Or increase quantization
# In config.json: "quantization": "8bit"
```

**3. Model not found**
```bash
# Re-download models
python scripts/download_models.py --force

# Check model path in config
cat config.json | grep model_name
```

**4. Server won't start**
```bash
# Check port availability
lsof -i :8000

# Check logs
python scripts/run.py --run 2>&1 | tee server.log
```

**5. Slow responses**
- Enable GPU acceleration
- Use 4-bit quantization
- Reduce max_new_tokens
- Pre-warm LLM (enabled by default)

### Getting Help

1. Check server health: `curl http://localhost:8000/health`
2. Check startup status: `curl http://localhost:8000/startup-status`
3. Enable debug mode: Set `AGENT_DEBUG=true`
4. Check logs in console output

---
