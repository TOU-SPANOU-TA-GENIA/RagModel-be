# RagModel-be

AI Agent with RAG, Greek language support, authentication, and persistent chats.

---

## Quick Start

```bash
# 1. Setup environment
conda create -n ragmodel_10 python=3.10
conda activate ragmodel_10
pip install -r requirements.txt

# 2. Install Redis
sudo apt install redis-server  # WSL/Linux
sudo systemctl start redis

# 3. Download AI model (one-time)
huggingface-cli download Qwen/Qwen3-4B  --local-dir ./offline_models/qwen3-4b  --local-dir-use-symlinks False  --resume-download

# 4. Initialize database
python scripts/setup_auth.py

# 5. Start server
python scripts/run.py

# 6. Chat!
python tests/cli/authenticated_chat_client.py
```

**Test credentials:** `test` / `test`

---

## Features

- 🇬🇷 **Greek Language** - Native support via Qwen3-4B
- 🤖 **AI Agent** - Conversational AI with tool execution
- 📚 **RAG** - Answer questions from your documents
- 🔐 **Authentication** - Multi-user with JWT tokens
- 💬 **Persistent Chats** - All conversations saved
- 🛠️ **Tools** - File operations, document generation

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU)
- 8GB RAM, 6GB+ VRAM recommended
- Redis server

### 1. Clone and Setup

```bash
git clone <your-repo>
cd RagModel-be

conda create -n ragmodel_10 python=3.10
conda activate ragmodel_10
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Redis

**WSL/Linux:**
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis
redis-cli ping  # Should return PONG
```

### 4. Download AI Model

**Option A: Quick (Recommended)**

```bash
huggingface-cli download Qwen/Qwen3-4B  --local-dir ./offline_models/qwen3-4b  --local-dir-use-symlinks False  --resume-download
```

Downloads Qwen3-4B (~4.5GB) to `./offline_models/qwen3-4b/`

**Option B: Manual**

1. Download from: https://huggingface.co/Qwen/Qwen3-4B/tree/main
2. Place files in `./offline_models/qwen3-4b/`

### 5. Initialize Database

```bash
python scripts/setup_auth.py
```

Creates test user: `testuser` / `testpass123`

---

## Usage

### Start Server

```bash
# Full startup (ingest documents + start server)
python scripts/run.py --full

# Just start server
python scripts/run.py --run

# Just ingest documents
python scripts/run.py --ingest
```

**Server ready when you see:**
```
✅ Application ready
INFO: Uvicorn running on http://localhost:8000
```

### Interactive CLI Chat (Easiest)

```bash
python tests/cli/authenticated_chat_client.py
```

**Commands:**
- `login` - Login to account
- `register` - Create new account
- `new <title>` - Create new chat
- `list` - List all chats
- `select <#>` - Switch to chat
- `history` - View messages
- `<message>` - Send message to AI
- `exit` - Quit

**Example:**
```
[guest] > login
Username: testuser
Password: testpass123
✅ Login successful!

[testuser] > new "Productivity Chat"
✅ Created chat

[testuser@Productivity Chat] > Γεια σου! Πώς μπορώ να βελτιώσω την παραγωγικότητά μου;
🤖 Assistant: Γεια σου! Για να βελτιώσεις την παραγωγικότητα...
```


## Configuration
```

**Common adjustments:**
- Longer responses: Increase `max_new_tokens` (256-1024)
- More creative: Increase `temperature` (0.7-1.0)
- More documents: Increase `top_k` (3-10)
- Faster responses: Decrease `max_new_tokens`

```


## API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | No | Health check |
| `/auth/register` | POST | No | Register user |
| `/auth/login` | POST | No | Get JWT token |
| `/chats/` | POST | Yes | Create chat |
| `/chats/` | GET | Yes | List user chats |
| `/chats/{id}` | GET | Yes | Get chat details |
| `/chats/{id}` | DELETE | Yes | Delete chat |
| `/chats/{id}/messages` | GET | Yes | Get messages |
| `/chats/{id}/messages` | POST | Yes | Send message |
| `/config/` | GET | No | Get config |
| `/config/{category}/{field}` | PUT | No | Update config |
| `/docs` | GET | No | API documentation |

---

## Model Information

**Qwen3-4B** (Alibaba Cloud)
- **Size:** 4 billion parameters (~4.5GB on disk)
- **Languages:** 119 including Greek (native, not translated)
- **Context:** 32K tokens
- **License:** Apache 2.0 (fully permissive, no restrictions)
- **VRAM:** 4-5GB with 4-bit quantization (perfect for RTX 4050)
- **Speed:** 20-40 tokens/second on RTX 4050
- **Strengths:** Greek language, multilingual, agent tasks, RAG


**You're ready to go!**

Quick start: `python scripts/run.py --full` then `python tests/cli/authenticated_chat_client.py`