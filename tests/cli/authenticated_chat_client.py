# tests/cli/authenticated_chat_client.py
"""
Authenticated Chat Client - Full featured CLI for testing the authenticated API.
Supports login, registration, chat management, and conversations.
"""

import requests
import json
import os
from typing import Optional, Dict, List
from datetime import datetime
from pathlib import Path


class AuthenticatedChatClient:
    """Client for interacting with the authenticated chat API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.token: Optional[str] = None
        self.user: Optional[Dict] = None
        self.current_chat_id: Optional[str] = None
        self.current_chat_title: Optional[str] = None
        
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
                "current_chat_title": self.current_chat_title
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
                print(f"✅ Registration successful!")
                # Auto-login after registration
                return self.login(username, password)
            else:
                error = response.json().get("detail", "Unknown error")
                print(f"❌ Registration failed: {error}")
                return False
        
        except Exception as e:
            print(f"❌ Registration error: {e}")
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
                print(f"✅ Login successful! Welcome, {self.user['username']}!")
                return True
            else:
                error = response.json().get("detail", "Invalid credentials")
                print(f"❌ Login failed: {error}")
                return False
        
        except Exception as e:
            print(f"❌ Login error: {e}")
            return False
    
    def logout(self):
        """Logout and clear session."""
        self.clear_session()
        print("✅ Logged out successfully!")
    
    def whoami(self):
        """Display current user info."""
        if not self.user:
            print("❌ Not logged in")
            return
        
        print("\n" + "="*60)
        print("👤 Current User")
        print("="*60)
        print(f"Username: {self.user['username']}")
        print(f"Email: {self.user['email']}")
        print(f"User ID: {self.user['id']}")
        if self.current_chat_id:
            print(f"Current Chat: {self.current_chat_title or self.current_chat_id}")
        print("="*60 + "\n")
    
    # =========================================================================
    # Chat Management
    # =========================================================================
    
    def list_chats(self) -> List[Dict]:
        """List all user's chats."""
        if not self.token:
            print("❌ Please login first")
            return []
        
        try:
            response = requests.get(
                f"{self.base_url}/chats/",
                headers=self.get_headers()
            )
            
            if response.status_code == 200:
                chats = response.json()
                
                if not chats:
                    print("\n📭 No chats yet. Create one with 'new <title>'")
                    return []
                
                print("\n" + "="*60)
                print("💬 Your Chats")
                print("="*60)
                
                for i, chat in enumerate(chats, 1):
                    chat_id_short = chat['id'][:8]
                    marker = "→" if chat['id'] == self.current_chat_id else " "
                    print(f"{marker} {i}. {chat['title']}")
                    print(f"     ID: {chat_id_short}... | Messages: {chat['message_count']} | Updated: {chat['updated_at'][:19]}")
                
                print("="*60 + "\n")
                return chats
            else:
                print(f"❌ Failed to list chats: {response.status_code}")
                return []
        
        except Exception as e:
            print(f"❌ Error listing chats: {e}")
            return []
    
    def create_chat(self, title: str = "New Chat") -> bool:
        """Create a new chat."""
        if not self.token:
            print("❌ Please login first")
            return False
        
        try:
            response = requests.post(
                f"{self.base_url}/chats/",
                headers=self.get_headers(),
                json={"title": title}
            )
            
            if response.status_code == 201:
                chat = response.json()
                self.current_chat_id = chat['id']
                self.current_chat_title = chat['title']
                self.save_session()
                print(f"✅ Created new chat: '{title}'")
                return True
            else:
                print(f"❌ Failed to create chat: {response.status_code}")
                return False
        
        except Exception as e:
            print(f"❌ Error creating chat: {e}")
            return False
    
    def select_chat(self, chat_id_or_index: str):
        """Select a chat by ID or list index."""
        if not self.token:
            print("❌ Please login first")
            return
        
        # If it's a number, treat it as an index
        if chat_id_or_index.isdigit():
            chats = self.list_chats()
            index = int(chat_id_or_index) - 1
            
            if 0 <= index < len(chats):
                chat = chats[index]
                self.current_chat_id = chat['id']
                self.current_chat_title = chat['title']
                self.save_session()
                print(f"✅ Selected chat: '{chat['title']}'")
                self.show_chat_history()
            else:
                print(f"❌ Invalid chat index: {chat_id_or_index}")
        else:
            # Treat as chat ID
            self.current_chat_id = chat_id_or_index
            self.save_session()
            print(f"✅ Selected chat: {chat_id_or_index[:8]}...")
            self.show_chat_history()
    
    def delete_chat(self, chat_id_or_index: str = None):
        """Delete a chat."""
        if not self.token:
            print("❌ Please login first")
            return
        
        chat_to_delete = None
        
        # If argument provided, check if it's an index
        if chat_id_or_index and chat_id_or_index.isdigit():
            chats = self.list_chats()
            index = int(chat_id_or_index) - 1
            
            if 0 <= index < len(chats):
                chat_to_delete = chats[index]['id']
                chat_title = chats[index]['title']
            else:
                print(f"❌ Invalid chat index: {chat_id_or_index}")
                return
        elif chat_id_or_index:
            # Treat as chat ID
            chat_to_delete = chat_id_or_index
            chat_title = chat_id_or_index[:8]
        else:
            # No argument, use current chat
            chat_to_delete = self.current_chat_id
            chat_title = self.current_chat_title or self.current_chat_id[:8]
        
        if not chat_to_delete:
            print("❌ No chat selected")
            return
        
        confirm = input(f"⚠️  Delete '{chat_title}'? (yes/no): ")
        if confirm.lower() != 'yes':
            print("❌ Cancelled")
            return
        
        try:
            response = requests.delete(
                f"{self.base_url}/chats/{chat_to_delete}",
                headers=self.get_headers()
            )
            
            if response.status_code == 204:
                print(f"✅ Chat deleted")
                if chat_to_delete == self.current_chat_id:
                    self.current_chat_id = None
                    self.current_chat_title = None
                    self.save_session()
            else:
                print(f"❌ Failed to delete chat: {response.status_code}")
        
        except Exception as e:
            print(f"❌ Error deleting chat: {e}")
    
    def show_chat_history(self):
        """Show current chat's message history."""
        if not self.current_chat_id:
            print("❌ No chat selected")
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
                print(f"💬 {chat['title']}")
                print("="*60)
                
                if not messages:
                    print("(No messages yet)")
                else:
                    for msg in messages:
                        role_emoji = "👤" if msg['role'] == 'user' else "🤖"
                        timestamp = msg['timestamp'][:19]
                        print(f"\n{role_emoji} {msg['role'].upper()} [{timestamp}]")
                        print(f"   {msg['content']}")
                
                print("="*60 + "\n")
            else:
                print(f"❌ Failed to get chat: {response.status_code}")
        
        except Exception as e:
            print(f"❌ Error getting chat: {e}")
    
    # =========================================================================
    # Messaging
    # =========================================================================
    
    def send_message(self, content: str):
        """Send a message in the current chat."""
        if not self.token:
            print("❌ Please login first")
            return
        
        if not self.current_chat_id:
            print("❌ No chat selected. Create one with 'new' or select with 'select'")
            return
        
        try:
            print("🤔 Thinking...")
            
            response = requests.post(
                f"{self.base_url}/chats/{self.current_chat_id}/messages",
                headers=self.get_headers(),
                json={"content": content}
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data['answer']
                
                print(f"\n🤖 Assistant:")
                print(f"   {answer}\n")
            else:
                error = response.json().get("detail", "Unknown error")
                print(f"❌ Error: {error}")
        
        except Exception as e:
            print(f"❌ Error sending message: {e}")
    
    # =========================================================================
    # System
    # =========================================================================
    
    def check_health(self):
        """Check API health."""
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"\n✅ API Status: {data['status']}")
                print(f"   Services: {data.get('services', {})}")
            else:
                print(f"❌ API unhealthy: {response.status_code}")
        except Exception as e:
            print(f"❌ Cannot reach API: {e}")


def print_help():
    """Print help message."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║           🤖 Authenticated Chat Client - Commands            ║
╚══════════════════════════════════════════════════════════════╝

Authentication:
  register                  - Register a new account
  login                     - Login to existing account
  logout                    - Logout and clear session
  whoami                    - Show current user info

Chat Management:
  new [title]              - Create new chat (default: "New Chat")
  list                     - List all your chats
  select <index|id>        - Select a chat by number or ID
  delete [id]              - Delete current or specified chat
  history                  - Show current chat's messages

Messaging:
  <your message>           - Send a message (just type normally)
  
System:
  health                   - Check API health
  help                     - Show this help
  exit                     - Exit the client

Examples:
  register                 → Create account
  login                    → Login to account
  new "Python Help"        → Create chat titled "Python Help"
  list                     → See all chats
  select 1                 → Select first chat from list
  hello                    → Send "hello" message
  history                  → View conversation
  
╚══════════════════════════════════════════════════════════════╝
    """)


def main():
    """Main interactive loop."""
    client = AuthenticatedChatClient()
    
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       🤖 Authenticated AI Agent Chat - Interactive CLI       ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    # Check if already logged in
    if client.token:
        print(f"\n✅ Resumed session as: {client.user['username']}")
        if client.current_chat_id:
            print(f"   Current chat: {client.current_chat_title or client.current_chat_id[:8]}")
        print("\nType 'help' for commands\n")
    else:
        print("\n⚠️  Not logged in. Use 'login' or 'register' to start")
        print("Type 'help' for commands\n")
    
    while True:
        try:
            # Show prompt
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
            if command == "exit" or command == "quit":
                print("👋 Goodbye!")
                break
            
            elif command == "help":
                print_help()
            
            elif command == "register":
                print("\n📝 Register New Account")
                username = input("Username: ").strip()
                email = input("Email: ").strip()
                password = input("Password: ").strip()
                client.register(username, email, password)
            
            elif command == "login":
                print("\n🔐 Login")
                username = input("Username: ").strip()
                password = input("Password: ").strip()
                client.login(username, password)
            
            elif command == "logout":
                client.logout()
            
            elif command == "whoami":
                client.whoami()
            
            elif command == "new":
                title = args or "New Chat"
                client.create_chat(title)
            
            elif command == "list":
                client.list_chats()
            
            elif command == "select":
                if args:
                    client.select_chat(args)
                else:
                    print("❌ Usage: select <chat_index|chat_id>")
            
            elif command == "delete":
                client.delete_chat(args if args else None)
            
            elif command == "history":
                client.show_chat_history()
            
            elif command == "health":
                client.check_health()
            
            else:
                # Treat as message
                client.send_message(user_input)
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()