# tests/cli/authenticated_chat_client.py
"""
Authenticated Chat Client with TRUE real-time streaming.

Features:
- No timeout (waits for complete response)
- Shows status updates (searching, generating)
- Real token-by-token display
- Number-based chat operations
- Configurable max_tokens
"""

import requests
import json
from typing import Optional, Dict, List
from pathlib import Path


class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    GRAY = '\033[90m'
    END = '\033[0m'


def cprint(text: str, color: str = "", end: str = "\n"):
    print(f"{color}{text}{Colors.END}", end=end, flush=True)


class AuthenticatedChatClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.token: Optional[str] = None
        self.user: Optional[Dict] = None
        self.current_chat_id: Optional[str] = None
        self.current_chat_title: Optional[str] = None
        self.use_streaming: bool = True
        self.show_thinking: bool = False
        self.max_tokens: int = 256
        self._chat_cache: List[Dict] = []
        
        self.session_file = Path("data/.session.json")
        self.load_session()
    
    def save_session(self):
        if self.token and self.user:
            self.session_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.session_file, 'w') as f:
                json.dump({
                    "token": self.token,
                    "user": self.user,
                    "current_chat_id": self.current_chat_id,
                    "current_chat_title": self.current_chat_title,
                    "use_streaming": self.use_streaming,
                    "show_thinking": self.show_thinking,
                    "max_tokens": self.max_tokens
                }, f)
    
    def load_session(self):
        if self.session_file.exists():
            try:
                with open(self.session_file, 'r') as f:
                    d = json.load(f)
                self.token = d.get("token")
                self.user = d.get("user")
                self.current_chat_id = d.get("current_chat_id")
                self.current_chat_title = d.get("current_chat_title")
                self.use_streaming = d.get("use_streaming", True)
                self.show_thinking = d.get("show_thinking", False)
                self.max_tokens = d.get("max_tokens", 256)
            except:
                pass
    
    def clear_session(self):
        if self.session_file.exists():
            self.session_file.unlink()
        self.token = self.user = self.current_chat_id = self.current_chat_title = None
    
    def get_headers(self) -> Dict:
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h
    
    def _refresh_chats(self) -> List[Dict]:
        if not self.token:
            return []
        try:
            r = requests.get(f"{self.base_url}/chats", headers=self.get_headers(), timeout=10)
            if r.status_code == 200:
                self._chat_cache = r.json()
        except:
            pass
        return self._chat_cache
    
    def _get_chat(self, id_or_num: str) -> Optional[Dict]:
        chats = self._refresh_chats()
        if id_or_num.isdigit():
            idx = int(id_or_num) - 1
            return chats[idx] if 0 <= idx < len(chats) else None
        for c in chats:
            if c["id"].startswith(id_or_num):
                return c
        return None
    
    # === Auth ===
    def register(self, u, e, p):
        try:
            r = requests.post(f"{self.base_url}/auth/register", json={"username":u,"email":e,"password":p})
            cprint("✅ Registered!" if r.ok else f"❌ {r.json().get('detail','Error')}", 
                   Colors.GREEN if r.ok else Colors.RED)
        except Exception as ex:
            cprint(f"❌ {ex}", Colors.RED)
    
    def login(self, u, p):
        try:
            r = requests.post(f"{self.base_url}/auth/login", data={"username":u,"password":p})
            if r.ok:
                d = r.json()
                self.token = d["access_token"]
                self.user = {"username": u, "id": d.get("user_id")}
                self.save_session()
                self._refresh_chats()
                cprint(f"✅ Welcome {u}", Colors.GREEN)
            else:
                cprint(f"❌ {r.json().get('detail','Error')}", Colors.RED)
        except Exception as ex:
            cprint(f"❌ {ex}", Colors.RED)
    
    def logout(self):
        self.clear_session()
        cprint("✅ Logged out", Colors.GREEN)
    
    # === Chats ===
    def new_chat(self, title="Νέα Συνομιλία"):
        if not self.token:
            return cprint("❌ Login first", Colors.RED)
        try:
            r = requests.post(f"{self.base_url}/chats", headers=self.get_headers(), json={"title":title})
            if r.ok:
                d = r.json()
                self.current_chat_id = d["id"]
                self.current_chat_title = d["title"]
                self.save_session()
                cprint(f"✅ {d['title']}", Colors.GREEN)
        except Exception as ex:
            cprint(f"❌ {ex}", Colors.RED)
    
    def list_chats(self):
        if not self.token:
            return cprint("❌ Login first", Colors.RED)
        chats = self._refresh_chats()
        if not chats:
            return print("📭 No chats")
        cprint("\n📋 Chats:", Colors.CYAN)
        for i, c in enumerate(chats, 1):
            m = "→" if c["id"] == self.current_chat_id else " "
            print(f" {m} {i}. {c['title'][:40]} ({c.get('message_count',0)})")
    
    def select_chat(self, x):
        c = self._get_chat(x)
        if c:
            self.current_chat_id = c["id"]
            self.current_chat_title = c["title"]
            self.save_session()
            cprint(f"✅ Selected: {c['title']}", Colors.GREEN)
        else:
            cprint(f"❌ Not found: {x}", Colors.RED)
    
    def delete_chat(self, x=None):
        if not self.token:
            return cprint("❌ Login first", Colors.RED)
        if x:
            c = self._get_chat(x)
            if not c:
                return cprint("❌ Not found", Colors.RED)
            tid, tname = c["id"], c["title"]
        elif self.current_chat_id:
            tid, tname = self.current_chat_id, self.current_chat_title
        else:
            return cprint("❌ No chat selected", Colors.RED)
        
        if input(f"Delete '{tname}'? (y/n): ").lower() != 'y':
            return print("Cancelled")
        
        try:
            r = requests.delete(f"{self.base_url}/chats/{tid}", headers=self.get_headers())
            if r.status_code in (200, 204):
                cprint("✅ Deleted", Colors.GREEN)
                if tid == self.current_chat_id:
                    self.current_chat_id = self.current_chat_title = None
                    self.save_session()
        except Exception as ex:
            cprint(f"❌ {ex}", Colors.RED)
    
    # === Messaging ===
    def send(self, msg):
        if not self.token:
            return cprint("❌ Login first", Colors.RED)
        if not self.current_chat_id:
            return cprint("❌ Select a chat first (new/select)", Colors.RED)
        
        if self.use_streaming:
            self._send_streaming(msg)
        else:
            self._send_regular(msg)
    
    def _send_streaming(self, msg):
        """TRUE real-time streaming with no timeout."""
        try:
            # NO TIMEOUT
            with requests.post(
                f"{self.base_url}/stream/chat",
                json={
                    "content": msg,
                    "chat_id": self.current_chat_id,
                    "include_thinking": self.show_thinking,
                    "max_tokens": self.max_tokens
                },
                stream=True,
                headers={"Content-Type": "application/json"},
                timeout=None
            ) as resp:
                
                if not resp.ok:
                    return cprint(f"❌ {resp.status_code}", Colors.RED)
                
                buffer = ""
                in_thinking = False
                response_started = False
                
                for chunk in resp.iter_content(chunk_size=1, decode_unicode=True):
                    if not chunk:
                        continue
                    
                    buffer += chunk
                    
                    while "\n\n" in buffer:
                        event_str, buffer = buffer.split("\n\n", 1)
                        
                        for line in event_str.split("\n"):
                            if not line.startswith("data: "):
                                continue
                            
                            try:
                                d = json.loads(line[6:])
                                evt = d.get('type', '')
                                data = d.get('data', '')
                                
                                # Debug: uncomment to show all events
                                # print(f"[DEBUG] {evt}: {data[:50] if data else '(empty)'}") 
                                
                                if evt == 'status':
                                    cprint(f"\n{data}", Colors.YELLOW)
                                
                                elif evt == 'heartbeat':
                                    print(".", end="", flush=True)
                                
                                elif evt == 'thinking_start':
                                    in_thinking = True
                                    if self.show_thinking:
                                        cprint("\n[Thinking: ", Colors.GRAY, end="")
                                
                                elif evt == 'thinking_end':
                                    in_thinking = False
                                    if self.show_thinking:
                                        cprint("]", Colors.GRAY)
                                
                                elif evt == 'response_start':
                                    if not response_started:
                                        print("\n", end="")
                                        cprint("🤖 ", Colors.GREEN, end="")
                                        response_started = True
                                
                                elif evt == 'token':
                                    if in_thinking:
                                        if self.show_thinking:
                                            print(data, end="", flush=True)
                                    else:
                                        if not response_started:
                                            cprint("🤖 ", Colors.GREEN, end="")
                                            response_started = True
                                        print(data, end="", flush=True)
                                
                                elif evt == 'error':
                                    cprint(f"\n❌ {data}", Colors.RED)
                            
                            except json.JSONDecodeError:
                                continue
            
            print("\n")
            
        except Exception as ex:
            cprint(f"\n❌ {ex}", Colors.RED)
    
    def _send_regular(self, msg):
        try:
            cprint("🤔 ...", Colors.YELLOW)
            r = requests.post(
                f"{self.base_url}/chats/{self.current_chat_id}/messages",
                headers=self.get_headers(),
                json={"content": msg},
                timeout=600
            )
            if r.ok:
                cprint(f"\n🤖 {r.json()['answer']}\n", Colors.GREEN)
            else:
                cprint(f"❌ {r.json().get('detail','Error')}", Colors.RED)
        except Exception as ex:
            cprint(f"❌ {ex}", Colors.RED)
    
    # === Settings ===
    def toggle_streaming(self):
        self.use_streaming = not self.use_streaming
        self.save_session()
        cprint(f"✅ Streaming: {'ON' if self.use_streaming else 'OFF'}", Colors.GREEN)
    
    def toggle_thinking(self):
        self.show_thinking = not self.show_thinking
        self.save_session()
        cprint(f"✅ Thinking: {'ON' if self.show_thinking else 'OFF'}", Colors.GREEN)
    
    def set_max_tokens(self, n: int):
        self.max_tokens = max(50, min(2048, n))
        self.save_session()
        cprint(f"✅ Max tokens: {self.max_tokens}", Colors.GREEN)
    
    def show_status(self):
        cprint(f"\n📊 Status:", Colors.CYAN)
        print(f"   User: {self.user['username'] if self.user else 'Not logged in'}")
        print(f"   Chat: {self.current_chat_title or 'None'}")
        print(f"   Streaming: {'ON' if self.use_streaming else 'OFF'}")
        print(f"   Thinking: {'ON' if self.show_thinking else 'OFF'}")
        print(f"   Max tokens: {self.max_tokens}")
    
    def check_health(self):
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            if r.ok:
                d = r.json()
                cprint(f"\n✅ API: {d.get('status','ok')}", Colors.GREEN)
        except Exception as ex:
            cprint(f"❌ {ex}", Colors.RED)
    
    def check_rag(self):
        try:
            r = requests.get(f"{self.base_url}/rag/status", timeout=5)
            if r.ok:
                d = r.json()
                cprint(f"\n📚 RAG: {d.get('indexed_files',0)}/{d.get('total_network_files',0)} files", Colors.CYAN)
        except Exception as ex:
            cprint(f"❌ {ex}", Colors.RED)


def main():
    c = AuthenticatedChatClient()
    
    print("╔════════════════════════════════════════════╗")
    print("║  🤖 AI Chat - True Real-Time Streaming     ║")
    print("╚════════════════════════════════════════════╝")
    
    if c.user:
        cprint(f"✅ {c.user['username']}", Colors.GREEN)
        if c.current_chat_title:
            print(f"   Chat: {c.current_chat_title}")
    else:
        print("⚠️  Use 'login' or 'register'")
    
    print("\nCommands: help, new, list, select N, delete N")
    print("Settings: streaming, thinking, tokens N, status\n")
    
    while True:
        try:
            if c.current_chat_id:
                p = f"[{c.user['username']}@{c.current_chat_title[:15]}] > "
            elif c.user:
                p = f"[{c.user['username']}] > "
            else:
                p = "[guest] > "
            
            inp = input(p).strip()
            if not inp:
                continue
            
            parts = inp.split(maxsplit=1)
            cmd, args = parts[0].lower(), parts[1] if len(parts) > 1 else ""
            
            if cmd in ("exit", "quit", "q"): 
                break
            elif cmd == "help":
                print("""
Commands:
  register, login, logout     - Auth
  new [title], list           - Chats  
  select N, delete N          - Chat by number
  
  streaming                   - Toggle streaming
  thinking                    - Toggle thinking display
  tokens N                    - Set max tokens (50-2048)
  status                      - Show settings
  health, rag                 - Check API
  
  [message]                   - Send message
""")
            elif cmd == "register":
                c.register(input("User: "), input("Email: "), input("Pass: "))
            elif cmd == "login":
                c.login(input("User: "), input("Pass: "))
            elif cmd == "logout":
                c.logout()
            elif cmd == "new":
                c.new_chat(args or "Νέα Συνομιλία")
            elif cmd == "list":
                c.list_chats()
            elif cmd == "select":
                c.select_chat(args) if args else print("select N")
            elif cmd == "delete":
                c.delete_chat(args if args else None)
            elif cmd == "streaming":
                c.toggle_streaming()
            elif cmd == "thinking":
                c.toggle_thinking()
            elif cmd == "tokens":
                c.set_max_tokens(int(args)) if args.isdigit() else print("tokens N")
            elif cmd == "status":
                c.show_status()
            elif cmd == "health":
                c.check_health()
            elif cmd == "rag":
                c.check_rag()
            else:
                c.send(inp)
        
        except KeyboardInterrupt:
            print("\n👋 Bye!")
            break
        except Exception as ex:
            cprint(f"❌ {ex}", Colors.RED)


if __name__ == "__main__":
    main()