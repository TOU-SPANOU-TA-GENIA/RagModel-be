# RagModel-be

AI Agent with RAG capabilities, user authentication, and persistent chat storage.

---

## Features

- 🤖 **AI Agent** with GPT-style conversations
- 📚 **RAG** - Answer questions based on your documents
- 🔐 **Authentication** - User accounts with JWT tokens
- 💬 **Persistent Chats** - All conversations saved to database
- ⚡ **Fast** - GPU acceleration + Redis caching
- 🛠️ **Tools** - File operations, document generation

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU)
- 8GB RAM minimum

### Step 1: Clone & Setup

```bash
git clone <your-repo>
cd RagModel-be

# Create environment
conda create -n ragmodel_10 python=3.10
conda activate ragmodel_10
```

### Step 2: Install Dependencies

```bash
# Core packages
pip install -r requirements.txt
```

### Step 3: Install Redis

**WSL:**
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis
```

**Verify:**
```bash
redis-cli ping  # Should respond: PONG
```

### Step 4: Initialize Database

```bash
python scripts/setup_auth.py
```

This creates:
- SQLite database at `data/app.db`
- Test user (username: `testuser`, password: `testpass123`)

---

## Setup

### 1. Download AI Models

```bash
python scripts/download_models.py
```

This downloads:
- LLM model (3GB)
- Embedding model (500MB)

### 2. Add Your Documents

```bash
# Copy your files to knowledge directory
cp your_files/*.txt data/knowledge/
cp your_files/*.pdf data/knowledge/
```

### 3. Start the Server

```bash
python scripts/run.py --full
```

This will:
- Ingest your documents
- Start the API server at http://localhost:8000

**Server is ready when you see:**
```
✅ Application ready
INFO: Uvicorn running on http://localhost:8000
```

---

## Usage

### Option 1: Interactive CLI (Recommended)

```bash
python tests/cli/authenticated_chat_client.py
```

**Example session:**
```
[guest] > login
Username: testuser
Password: testpass123
✅ Login successful!

[testuser] > new "My First Chat"
✅ Created new chat: 'My First Chat'

[testuser@My First Chat] > hello, how are you?
🤔 Thinking...
🤖 Assistant: I'm doing well! How can I help you today?

[testuser@My First Chat] > list
💬 Your Chats
→ 1. My First Chat | Messages: 2

[testuser@My First Chat] > exit
```

**Available commands:**
- `register` - Create new account
- `login` - Login
- `new <title>` - Create chat
- `list` - List all chats
- `select <#>` - Switch chat
- `history` - View messages
- `<message>` - Send message
- `exit` - Quit

### Option 2: API (Python)

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Register
response = requests.post(f"{BASE_URL}/auth/register", json={
    "username": "alice",
    "email": "alice@example.com",
    "password": "secret123"
})

# 2. Login
response = requests.post(f"{BASE_URL}/auth/login", json={
    "username": "alice",
    "password": "secret123"
})
token = response.json()["access_token"]

# 3. Create chat
headers = {"Authorization": f"Bearer {token}"}
response = requests.post(
    f"{BASE_URL}/chats/",
    headers=headers,
    json={"title": "Python Help"}
)
chat_id = response.json()["id"]

# 4. Send message
response = requests.post(
    f"{BASE_URL}/chats/{chat_id}/messages",
    headers=headers,
    json={"content": "Explain Python decorators"}
)
print(response.json()["answer"])

# 5. List all chats
response = requests.get(f"{BASE_URL}/chats/", headers=headers)
chats = response.json()
for chat in chats:
    print(f"{chat['title']}: {chat['message_count']} messages")
```

### Option 3: REST API

**Interactive docs:** http://localhost:8000/docs

**Example requests:**

```bash
# Register
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"bob","email":"bob@example.com","password":"pass123"}'

# Login
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"bob","password":"pass123"}'
# Returns: {"access_token": "eyJ...", "user": {...}}

# Create chat
curl -X POST http://localhost:8000/chats/ \
  -H "Authorization: Bearer <your_token>" \
  -H "Content-Type: application/json" \
  -d '{"title":"AI Discussion"}'

# Send message
curl -X POST http://localhost:8000/chats/<chat_id>/messages \
  -H "Authorization: Bearer <your_token>" \
  -H "Content-Type: application/json" \
  -d '{"content":"What is machine learning?"}'
```

---

## Testing

### 1. Test Authentication

```bash
python tests/cli/authenticated_chat_client.py
```

Try:
- Register a new account
- Login
- Create a chat
- Send messages
- List chats
- View history

### 2. Test API Endpoints

```bash
# Start server
python scripts/run.py --run

# In another terminal, test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/
```

### 3. Run Unit Tests

```bash
pytest tests/unit/ -v
```

### 4. Database Inspection

**SQLite (view users, chats, messages):**
```bash
python scripts/inspect_sqlite.py
```

**Redis (view cached data):**
```bash
python scripts/inspect_redis.py
```

**Or use GUI tools:**
- SQLite: Download [DB Browser](https://sqlitebrowser.org/) → Open `data/app.db`
- Redis: Download [RedisInsight](https://redis.com/redis-enterprise/redis-insight/) → Connect to `localhost:6379`

### 5. Quick Functionality Test

```python
# test_basic.py
import requests

def test_full_flow():
    base = "http://localhost:8000"
    
    # Register
    r = requests.post(f"{base}/auth/register", json={
        "username": "test123",
        "email": "test@test.com",
        "password": "pass123"
    })
    assert r.status_code == 201
    
    # Login
    r = requests.post(f"{base}/auth/login", json={
        "username": "test123",
        "password": "pass123"
    })
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Create chat
    r = requests.post(f"{base}/chats/", headers=headers, json={"title": "Test"})
    chat_id = r.json()["id"]
    
    # Send message
    r = requests.post(
        f"{base}/chats/{chat_id}/messages",
        headers=headers,
        json={"content": "Hello"}
    )
    assert r.status_code == 200
    assert "answer" in r.json()
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    test_full_flow()
```

Run it:
```bash
python test_basic.py
```

---

## Configuration

Edit `config.json` to change settings:

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

**Common changes:**
- Use smaller model: Change `model_name` to `"meta-llama/Llama-3.2-1B-Instruct"`
- Adjust response length: Change `max_new_tokens`
- More context: Increase `top_k` (more documents retrieved)

---

## Troubleshooting

### Redis not available
```
⚠️ Redis is not available
```
**Fix:**
```bash
sudo systemctl start redis  # Linux
brew services start redis   # macOS
```

### CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**Fix:** Use smaller model
```bash
python scripts/download_models.py --llm "meta-llama/Llama-3.2-1B-Instruct"
```

### Token expired
```
401 Unauthorized: Invalid token
```
**Fix:** Login again to get a new token (tokens expire after 24 hours)

### Port already in use
```
Address already in use
```
**Fix:**
```bash
lsof -i :8000  # Find process
kill -9 <PID>  # Kill it
```

### Can't find models
```
OSError: Model not found
```
**Fix:**
```bash
python scripts/download_models.py --force
```

---

## Quick Reference

### File Structure
```
RagModel-be/
├── data/
│   ├── knowledge/      # Your documents (add files here)
│   └── app.db         # SQLite database
├── config.json        # Configuration
├── scripts/
│   ├── setup_auth.py  # Initialize database
│   └── run.py         # Start server
└── tests/cli/
    └── authenticated_chat_client.py  # Interactive CLI
```

### Common Commands
```bash
# Start everything
python scripts/run.py --full

# Just start server
python scripts/run.py --run

# Initialize database
python scripts/setup_auth.py

# Interactive chat
python tests/cli/authenticated_chat_client.py

# View database
python scripts/inspect_sqlite.py

# View cache
python scripts/inspect_redis.py

# Run tests
pytest tests/unit/
```

### API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/auth/register` | POST | No | Register user |
| `/auth/login` | POST | No | Get token |
| `/chats/` | POST | Yes | Create chat |
| `/chats/` | GET | Yes | List chats |
| `/chats/{id}/messages` | POST | Yes | Send message |
| `/health` | GET | No | Health check |
| `/docs` | GET | No | API docs |

---

## Support

- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health
- **View Logs:** Check console output when running server

---

**That's it! You're ready to use the AI agent.** 🚀

Start with: `python scripts/run.py --full` then `python tests/cli/authenticated_chat_client.py`