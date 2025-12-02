# tests/cli/authenticated_chat_client.py
"""
Authenticated Chat Client - Full featured CLI for testing the authenticated API.
Supports login, registration, chat management, conversations, and streaming.

Updated with:
- Real-time streaming support
- Greek language UI
- Thinking display option
"""

import requests
import json
import os
import sys
from typing import Optional, Dict, List
from datetime import datetime
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    GRAY = '\033[90m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'


def print_colored(text: str, color: str = ""):
    """Print colored text."""
    print(f"{color}{text}{Colors.END}")


class AuthenticatedChatClient:
    """Client for interacting with the authenticated chat API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.token: Optional[str] = None
        self.user: Optional[Dict] = None
        self.current_chat_id: Optional[str] = None
        self.current_chat_title: Optional[str] = None
        self.use_streaming: bool = True  # Enable streaming by default
        self.show_thinking: bool = False  # Hide thinking by default
        
        # Load saved session if exists
        self.session_file = Path("data/.session.json")
        self.load_session()
    
    def save_session(self):
        """Save current session to file."""
        if self.token and self.user:
            session_data = {
                "token": self.token,
                "user": self.user,
                "current_chat_id": self.current_chat_id,
                "current_chat_title": self.current_chat_title,
                "use_streaming": self.use_streaming,
                "show_thinking": self.show_thinking
            }
            self.session_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.session_file, 'w') as f:
                json.dump(session_data, f)
    
    def load_session(self):
        """Load saved session from file."""
        if self.session_file.exists():
            try:
                with open(self.session_file, 'r') as f:
                    session_data = json.load(f)
                self.token = session_data.get("token")
                self.user = session_data.get("user")
                self.current_chat_id = session_data.get("current_chat_id")
                self.current_chat_title = session_data.get("current_chat_title")
                self.use_streaming = session_data.get("use_streaming", True)
                self.show_thinking = session_data.get("show_thinking", False)
            except Exception as e:
                print(f"⚠️  Could not load session: {e}")
    
    def clear_session(self):
        """Clear saved session."""
        if self.session_file.exists():
            self.session_file.unlink()
        self.token = None
        self.user = None
        self.current_chat_id = None
        self.current_chat_title = None
    
    def get_headers(self) -> Dict[str, str]:
        """Get headers with authentication token."""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers
    
    # =========================================================================
    # Authentication
    # =========================================================================
    
    def register(self, username: str, email: str, password: str) -> bool:
        """Register a new user."""
        try:
            response = requests.post(
                f"{self.base_url}/auth/register",
                json={
                    "username": username,
                    "email": email,
                    "password": password
                }
            )
            
            if response.status_code == 201:
                print_colored("✅ Επιτυχής εγγραφή!", Colors.GREEN)
                return self.login(username, password)
            else:
                error = response.json().get("detail", "Unknown error")
                print_colored(f"❌ Αποτυχία εγγραφής: {error}", Colors.RED)
                return False
        
        except Exception as e:
            print_colored(f"❌ Σφάλμα εγγραφής: {e}", Colors.RED)
            return False
    
    def login(self, username: str, password: str) -> bool:
        """Login with username and password."""
        try:
            response = requests.post(
                f"{self.base_url}/auth/login",
                json={
                    "username": username,
                    "password": password
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                self.token = data["access_token"]
                self.user = data["user"]
                self.save_session()
                print_colored(f"✅ Επιτυχής σύνδεση! Καλώς ήρθες, {self.user['username']}!", Colors.GREEN)
                return True
            else:
                error = response.json().get("detail", "Invalid credentials")
                print_colored(f"❌ Αποτυχία σύνδεσης: {error}", Colors.RED)
                return False
        
        except Exception as e:
            print_colored(f"❌ Σφάλμα σύνδεσης: {e}", Colors.RED)
            return False
    
    def logout(self):
        """Logout and clear session."""
        self.clear_session()
        print_colored("✅ Αποσυνδεθήκατε επιτυχώς!", Colors.GREEN)
    
    def whoami(self):
        """Display current user info."""
        if self.user:
            print(f"\n👤 Χρήστης: {self.user['username']}")
            print(f"   Email: {self.user['email']}")
            print(f"   Streaming: {'✅' if self.use_streaming else '❌'}")
            print(f"   Show thinking: {'✅' if self.show_thinking else '❌'}")
        else:
            print("❌ Δεν έχετε συνδεθεί")
    
    # =========================================================================
    # Chat Management
    # =========================================================================
    
    def create_chat(self, title: str = "Νέα Συνομιλία"):
        """Create a new chat."""
        if not self.token:
            print_colored("❌ Πρέπει να συνδεθείτε πρώτα", Colors.RED)
            return
        
        try:
            response = requests.post(
                f"{self.base_url}/chats/",
                headers=self.get_headers(),
                json={"title": title}
            )
            
            if response.status_code == 201:
                data = response.json()
                self.current_chat_id = data["id"]
                self.current_chat_title = data["title"]
                self.save_session()
                print_colored(f"✅ Δημιουργήθηκε: \"{self.current_chat_title}\"", Colors.GREEN)
            else:
                print_colored(f"❌ Αποτυχία δημιουργίας: {response.status_code}", Colors.RED)
        
        except Exception as e:
            print_colored(f"❌ Σφάλμα: {e}", Colors.RED)
    
    def list_chats(self):
        """List all chats for current user."""
        if not self.token:
            print_colored("❌ Πρέπει να συνδεθείτε πρώτα", Colors.RED)
            return
        
        try:
            response = requests.get(
                f"{self.base_url}/chats/",
                headers=self.get_headers()
            )
            
            if response.status_code == 200:
                chats = response.json()
                
                if not chats:
                    print("\n📭 Δεν υπάρχουν συνομιλίες. Δημιουργήστε μία με 'new'\n")
                    return
                
                print("\n" + "="*50)
                print_colored("💬 Οι Συνομιλίες σας", Colors.CYAN)
                print("="*50)
                
                for i, chat in enumerate(chats, 1):
                    marker = "→" if chat["id"] == self.current_chat_id else " "
                    print(f" {marker} {i}. {chat['title']}")
                    print(f"      Μηνύματα: {chat['message_count']} | Τελ. ενημέρωση: {chat['updated_at'][:10]}")
                
                print("="*50)
                print("Χρησιμοποιήστε 'select <αριθμός>' για επιλογή\n")
            else:
                print_colored(f"❌ Αποτυχία: {response.status_code}", Colors.RED)
        
        except Exception as e:
            print_colored(f"❌ Σφάλμα: {e}", Colors.RED)
    
    def select_chat(self, identifier: str):
        """Select a chat by index or ID."""
        if not self.token:
            print_colored("❌ Πρέπει να συνδεθείτε πρώτα", Colors.RED)
            return
        
        try:
            # Get chat list
            response = requests.get(
                f"{self.base_url}/chats/",
                headers=self.get_headers()
            )
            
            if response.status_code != 200:
                print_colored("❌ Αποτυχία λήψης συνομιλιών", Colors.RED)
                return
            
            chats = response.json()
            
            # Try to find chat
            selected_chat = None
            
            if identifier.isdigit():
                index = int(identifier) - 1
                if 0 <= index < len(chats):
                    selected_chat = chats[index]
            else:
                for chat in chats:
                    if chat["id"] == identifier or chat["id"].startswith(identifier):
                        selected_chat = chat
                        break
            
            if selected_chat:
                self.current_chat_id = selected_chat["id"]
                self.current_chat_title = selected_chat["title"]
                self.save_session()
                print_colored(f"✅ Επιλέχθηκε: \"{self.current_chat_title}\"", Colors.GREEN)
            else:
                print_colored("❌ Η συνομιλία δεν βρέθηκε", Colors.RED)
        
        except Exception as e:
            print_colored(f"❌ Σφάλμα: {e}", Colors.RED)
    
    def delete_chat(self, chat_id: Optional[str] = None):
        """Delete a chat."""
        if not self.token:
            print_colored("❌ Πρέπει να συνδεθείτε πρώτα", Colors.RED)
            return
        
        target_id = chat_id or self.current_chat_id
        if not target_id:
            print_colored("❌ Δεν υπάρχει επιλεγμένη συνομιλία", Colors.RED)
            return
        
        try:
            response = requests.delete(
                f"{self.base_url}/chats/{target_id}",
                headers=self.get_headers()
            )
            
            if response.status_code == 204:
                print_colored("✅ Η συνομιλία διαγράφηκε", Colors.GREEN)
                if target_id == self.current_chat_id:
                    self.current_chat_id = None
                    self.current_chat_title = None
                    self.save_session()
            else:
                print_colored(f"❌ Αποτυχία διαγραφής: {response.status_code}", Colors.RED)
        
        except Exception as e:
            print_colored(f"❌ Σφάλμα: {e}", Colors.RED)
    
    def show_chat_history(self):
        """Show message history for current chat."""
        if not self.current_chat_id:
            print_colored("❌ Δεν υπάρχει επιλεγμένη συνομιλία", Colors.RED)
            return
        
        try:
            response = requests.get(
                f"{self.base_url}/chats/{self.current_chat_id}",
                headers=self.get_headers()
            )
            
            if response.status_code == 200:
                data = response.json()
                chat = data['chat']
                messages = data['messages']
                
                print("\n" + "="*60)
                print_colored(f"💬 {chat['title']}", Colors.CYAN)
                print("="*60)
                
                if not messages:
                    print("(Δεν υπάρχουν μηνύματα)")
                else:
                    for msg in messages:
                        role_emoji = "👤" if msg['role'] == 'user' else "🤖"
                        role_color = Colors.BLUE if msg['role'] == 'user' else Colors.GREEN
                        timestamp = msg['timestamp'][:19]
                        print(f"\n{role_emoji} ", end="")
                        print_colored(f"{msg['role'].upper()} [{timestamp}]", role_color)
                        print(f"   {msg['content']}")
                
                print("="*60 + "\n")
            else:
                print_colored(f"❌ Αποτυχία: {response.status_code}", Colors.RED)
        
        except Exception as e:
            print_colored(f"❌ Σφάλμα: {e}", Colors.RED)
    
    # =========================================================================
    # Messaging
    # =========================================================================
    
    def send_message(self, content: str):
        """Send a message - uses streaming or regular API based on setting."""
        if not self.token:
            print_colored("❌ Πρέπει να συνδεθείτε πρώτα", Colors.RED)
            return
        
        if not self.current_chat_id:
            print_colored("❌ Δεν υπάρχει επιλεγμένη συνομιλία. Δημιουργήστε με 'new'", Colors.RED)
            return
        
        if self.use_streaming:
            self._send_message_streaming(content)
        else:
            self._send_message_regular(content)
    
    def _send_message_streaming(self, content: str):
        """Send message with streaming response."""
        try:
            print_colored("\n🤖 Απάντηση:", Colors.GREEN)
            print("   ", end="", flush=True)
            
            # Use streaming endpoint
            response = requests.post(
                f"{self.base_url}/stream/chat",
                json={
                    "content": content,
                    "chat_id": self.current_chat_id,
                    "include_thinking": self.show_thinking
                },
                stream=True,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                print_colored(f"\n❌ Σφάλμα: {response.status_code}", Colors.RED)
                return
            
            full_response = ""
            in_thinking = False
            
            for line in response.iter_lines():
                if not line:
                    continue
                
                line_str = line.decode('utf-8')
                if not line_str.startswith('data: '):
                    continue
                
                try:
                    data = json.loads(line_str[6:])
                    event_type = data.get('type', '')
                    event_data = data.get('data', '')
                    
                    if event_type == 'thinking_start':
                        in_thinking = True
                        if self.show_thinking:
                            print_colored("\n   [Σκέψη: ", Colors.GRAY, end="")
                    
                    elif event_type == 'thinking_end':
                        in_thinking = False
                        if self.show_thinking:
                            print_colored("]", Colors.GRAY)
                            print("   ", end="", flush=True)
                    
                    elif event_type == 'token':
                        if in_thinking and self.show_thinking:
                            print_colored(event_data, Colors.GRAY, end="", flush=True)
                        elif not in_thinking:
                            print(event_data, end="", flush=True)
                            full_response += event_data
                    
                    elif event_type == 'done':
                        break
                    
                    elif event_type == 'error':
                        print_colored(f"\n❌ Σφάλμα: {event_data}", Colors.RED)
                        break
                
                except json.JSONDecodeError:
                    continue
            
            print("\n")
            
            # Save message to chat history via regular API
            self._save_message_to_chat(content, full_response)
            
        except Exception as e:
            print_colored(f"\n❌ Σφάλμα streaming: {e}", Colors.RED)
            # Fallback to regular
            self._send_message_regular(content)
    
    def _send_message_regular(self, content: str):
        """Send message with regular (non-streaming) response."""
        try:
            print_colored("🤔 Σκέφτομαι...", Colors.YELLOW)
            
            response = requests.post(
                f"{self.base_url}/chats/{self.current_chat_id}/messages",
                headers=self.get_headers(),
                json={"content": content}
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data['answer']
                
                print_colored(f"\n🤖 Απάντηση:", Colors.GREEN)
                print(f"   {answer}\n")
            else:
                error = response.json().get("detail", "Unknown error")
                print_colored(f"❌ Σφάλμα: {error}", Colors.RED)
        
        except Exception as e:
            print_colored(f"❌ Σφάλμα: {e}", Colors.RED)
    
    def _save_message_to_chat(self, user_content: str, assistant_content: str):
        """Save streamed messages to chat history."""
        try:
            # This is a workaround since streaming doesn't auto-save
            # You may want to add a dedicated endpoint for this
            pass
        except Exception:
            pass
    
    # =========================================================================
    # Settings
    # =========================================================================
    
    def toggle_streaming(self):
        """Toggle streaming mode."""
        self.use_streaming = not self.use_streaming
        self.save_session()
        status = "ενεργό" if self.use_streaming else "ανενεργό"
        print_colored(f"✅ Streaming: {status}", Colors.GREEN)
    
    def toggle_thinking(self):
        """Toggle thinking display."""
        self.show_thinking = not self.show_thinking
        self.save_session()
        status = "ενεργό" if self.show_thinking else "ανενεργό"
        print_colored(f"✅ Εμφάνιση σκέψης: {status}", Colors.GREEN)
    
    # =========================================================================
    # System
    # =========================================================================
    
    def check_health(self):
        """Check API health."""
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print_colored(f"\n✅ API Status: {data['status']}", Colors.GREEN)
                print(f"   Database: {data.get('database', 'unknown')}")
                print(f"   Redis: {'✅' if data.get('redis_available') else '❌'}")
                print(f"   Language: {data.get('language', 'unknown')}")
                print(f"   Streaming: {'✅' if data.get('streaming') else '❌'}")
            else:
                print_colored(f"❌ API unhealthy: {response.status_code}", Colors.RED)
        except Exception as e:
            print_colored(f"❌ Cannot reach API: {e}", Colors.RED)


def print_help():
    """Print help message in Greek."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║        🤖 AI Chat Client - Εντολές (Commands)                ║
╚══════════════════════════════════════════════════════════════╝

Αυθεντικοποίηση:
  register                  - Δημιουργία νέου λογαριασμού
  login                     - Σύνδεση σε υπάρχοντα λογαριασμό
  logout                    - Αποσύνδεση
  whoami                    - Εμφάνιση τρέχοντος χρήστη

Διαχείριση Συνομιλιών:
  new [τίτλος]             - Δημιουργία νέας συνομιλίας
  list                     - Λίστα συνομιλιών
  select <αριθμός|id>      - Επιλογή συνομιλίας
  delete [id]              - Διαγραφή συνομιλίας
  history                  - Ιστορικό μηνυμάτων

Μηνύματα:
  <μήνυμα>                 - Αποστολή μηνύματος (απλά γράψτε)

Ρυθμίσεις:
  streaming                - Εναλλαγή streaming mode
  thinking                 - Εναλλαγή εμφάνισης σκέψης

Σύστημα:
  health                   - Έλεγχος API
  help                     - Εμφάνιση βοήθειας
  exit                     - Έξοδος

Παραδείγματα:
  login                    → Σύνδεση
  new "Python Help"        → Νέα συνομιλία με τίτλο
  Γεια σου!                → Αποστολή μηνύματος
  streaming                → Εναλλαγή real-time streaming
  
╚══════════════════════════════════════════════════════════════╝
    """)


def main():
    """Main interactive loop."""
    client = AuthenticatedChatClient()
    
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       🤖 AI Agent Chat - Διαδραστικό CLI                     ║")
    print("║       Με υποστήριξη Ελληνικών και Streaming                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    if client.token:
        print_colored(f"\n✅ Συνδεδεμένος ως: {client.user['username']}", Colors.GREEN)
        if client.current_chat_id:
            print(f"   Τρέχουσα συνομιλία: {client.current_chat_title or client.current_chat_id[:8]}")
        print(f"   Streaming: {'✅' if client.use_streaming else '❌'}")
        print("\nΓράψτε 'help' για εντολές\n")
    else:
        print("\n⚠️  Δεν έχετε συνδεθεί. Χρησιμοποιήστε 'login' ή 'register'")
        print("Γράψτε 'help' για εντολές\n")
    
    while True:
        try:
            # Build prompt
            if client.current_chat_id:
                chat_name = client.current_chat_title or client.current_chat_id[:8]
                prompt = f"[{client.user['username']}@{chat_name}] > "
            elif client.user:
                prompt = f"[{client.user['username']}] > "
            else:
                prompt = "[guest] > "
            
            user_input = input(prompt).strip()
            
            if not user_input:
                continue
            
            # Parse command
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            # Handle commands
            if command in ("exit", "quit", "έξοδος"):
                print("👋 Αντίο!")
                break
            
            elif command in ("help", "βοήθεια"):
                print_help()
            
            elif command == "register":
                print("\n📝 Εγγραφή Νέου Λογαριασμού")
                username = input("Username: ").strip()
                email = input("Email: ").strip()
                password = input("Password: ").strip()
                client.register(username, email, password)
            
            elif command == "login":
                print("\n🔐 Σύνδεση")
                username = input("Username: ").strip()
                password = input("Password: ").strip()
                client.login(username, password)
            
            elif command == "logout":
                client.logout()
            
            elif command == "whoami":
                client.whoami()
            
            elif command == "new":
                title = args or "Νέα Συνομιλία"
                client.create_chat(title)
            
            elif command == "list":
                client.list_chats()
            
            elif command == "select":
                if args:
                    client.select_chat(args)
                else:
                    print_colored("❌ Χρήση: select <αριθμός|id>", Colors.RED)
            
            elif command == "delete":
                client.delete_chat(args if args else None)
            
            elif command == "history":
                client.show_chat_history()
            
            elif command == "health":
                client.check_health()
            
            elif command == "streaming":
                client.toggle_streaming()
            
            elif command == "thinking":
                client.toggle_thinking()
            
            else:
                # Treat as message
                client.send_message(user_input)
        
        except KeyboardInterrupt:
            print("\n\n👋 Αντίο!")
            break
        except Exception as e:
            print_colored(f"❌ Σφάλμα: {e}", Colors.RED)


if __name__ == "__main__":
    main()