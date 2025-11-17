#!/usr/bin/env python3
"""
Usage:
    # Simple one-shot query
    python chat_cli.py "Πώς κάνω restart το PostgreSQL;"
    
    # Interactive mode
    python chat_cli.py --interactive
    
    # With specific chat ID
    python chat_cli.py --chat-id abc-123 "Next question"
    
    # Create new chat session
    python chat_cli.py --new-chat "PostgreSQL Help"
"""
import requests
import argparse
import sys
import json
from typing import Optional

try:
    from app.config import AGENT_DEBUG_MODE
except ImportError:
    # Fallback if not available
    AGENT_DEBUG_MODE = False
    
BASE_URL = "http://localhost:8000"

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class ChatSession:
    """Manages chat session with conversation memory."""
    
    def __init__(self):
        self.session_id = None
        self.base_url = BASE_URL
    
    def send_message(self, message: str) -> dict:
        """Send message with session context."""
        try:
            url = f"{self.base_url}/chat"
            params = {}
            
            if self.session_id:
                params["session_id"] = self.session_id
            
            response = requests.post(
                url,
                json={"role": "user", "content": message},
                params=params
            )
            
            if response.status_code == 200:
                result = response.json()
                # Update session ID if provided
                if "session_id" in result:
                    self.session_id = result["session_id"]
                return result
            else:
                print_error(f"Server error: {response.status_code}")
                return None
                
        except Exception as e:
            print_error(f"Failed to send message: {e}")
            return None
    
    def get_session_id(self) -> str:
        """Get current session ID."""
        return self.session_id or "no-session"

def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")

def print_success(text: str):
    print(f"{Colors.GREEN}{text}{Colors.END}")

def print_error(text: str):
    print(f"{Colors.RED}Error: {text}{Colors.END}", file=sys.stderr)

def print_warning(text: str):
    print(f"{Colors.YELLOW}{text}{Colors.END}")

def print_answer(text: str):
    print(f"\n{Colors.CYAN}Assistant:{Colors.END}")
    print(text)

def print_sources(sources: list):
    if not sources:
        return
    
    print(f"\n{Colors.YELLOW}Sources:{Colors.END}")
    for i, src in enumerate(sources, 1):
        score = src.get('relevance_score', 0)
        source_name = src.get('source', 'Unknown')
        print(f"  {i}. {source_name} (relevance: {score:.2f})")

def check_server() -> bool:
    """Check server with startup awareness."""
    try:
        # First, check if server is responding at all
        response = requests.get(f"{BASE_URL}/startup-status", timeout=5)
        
        if response.status_code == 200:
            status_data = response.json()
            
            if status_data.get("status") == "ready":
                return True
            else:
                print_warning("Server is starting up...")
                print(f"Status: {status_data.get('message')}")
                return True  # Still accept connections during startup
        
        return False
        
    except requests.exceptions.ConnectionError:
        return False
    except Exception as e:
        print_warning(f"Server check: {e}")
        return False

def create_chat(title: str = "CLI Chat") -> Optional[str]:
    try:
        response = requests.post(
            f"{BASE_URL}/chats",
            json={"title": title},
            timeout=10
        )
        
        if response.status_code == 201:
            chat_id = response.text.strip('"')
            print_success(f"Created new chat: {chat_id}")
            return chat_id
        else:
            print_error(f"Failed to create chat: {response.status_code}")
            return None
    
    except Exception as e:
        print_error(f"Failed to create chat: {e}")
        return None

def send_message(message: str, chat_id: Optional[str] = None) -> dict:
    try:
        if chat_id:
            url = f"{BASE_URL}/chats/{chat_id}/message"
        else:
            url = f"{BASE_URL}/chat"
        
        print("message: ", message)
        print("url: ", url)
        response = requests.post(
            url,
            json={"role": "user", "content": message},
            #timeout=60000  # LLM can take time
        )
        
        if response.status_code == 200:
            print('response:', response)
            return response.json()
        else:
            print_error(f"Server returned error: {response.status_code}")
            print_error(response.text)
            return None
    
    except requests.exceptions.Timeout:
        print_error("Request timed out. Server may be processing or overloaded.")
        return None
    
    except Exception as e:
        print_error(f"Failed to send message: {e}")
        return None

def list_chats():
    try:
        response = requests.get(f"{BASE_URL}/chats", timeout=10)
        
        if response.status_code == 200:
            chats = response.json()
            
            if not chats:
                print_warning("No chats found.")
                return
            
            print_header("Available Chats:")
            for chat in chats:
                print(f"  ID: {chat['id']}")
                print(f"  Title: {chat['title']}")
                print(f"  Messages: {chat['message_count']}")
                print(f"  Updated: {chat['last_updated']}")
                print()
        else:
            print_error(f"Failed to list chats: {response.status_code}")
    
    except Exception as e:
        print_error(f"Failed to list chats: {e}")

# test/chat_cli.py - Complete fixed interactive_mode function

def interactive_mode(chat_id: Optional[str] = None):
    print_header("🤖 RAG Chat - Interactive Mode (Context-Aware)")
    print("💬 Type 'exit' or 'quit' to end the session")
    print("🆕 Type 'new' to start a new chat")
    print("📋 Type 'list' to see all chats")
    print("🧹 Type 'clear' to clear screen")
    print("ℹ️  Type 'session' to show session info")
    print("🆔 Type 'reset' to reset session and start fresh")
    print("❓ Type 'help' to show this help")
    print("-" * 60)
    
    # Initialize chat session
    chat_session = ChatSession()
    
    # If chat_id provided, use it as session
    if chat_id:
        chat_session.session_id = chat_id
        print_success(f"Using existing chat session: {chat_session.get_session_id()}")
    
    while True:
        try:
            # Create dynamic prompt with session info
            session_display = chat_session.get_session_id()
            prompt = f"{Colors.GREEN}You [{session_display}]:{Colors.END} "
            
            user_input = input(prompt).strip()
            
            # Handle commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print_success("👋 Goodbye!")
                break
            
            elif user_input.lower() == 'new':
                title = input("Enter chat title (or press Enter for default): ").strip()
                new_chat_id = create_chat(title or "CLI Chat")
                if new_chat_id:
                    chat_session.session_id = new_chat_id
                    print_success(f"🆕 New chat created: {chat_session.get_session_id()}")
                continue
            
            elif user_input.lower() == 'list':
                list_chats()
                continue
            
            elif user_input.lower() == 'clear':
                import os
                os.system('cls' if os.name == 'nt' else 'clear')
                # Re-print header after clear
                print_header("🤖 RAG Chat - Interactive Mode (Context-Aware)")
                print("Session continues...")
                print("-" * 60)
                continue
            
            elif user_input.lower() == 'session':
                print(f"🔑 Session ID: {chat_session.session_id or 'No active session'}")
                print(f"📝 Display: {chat_session.get_session_id()}")
                if chat_session.session_id:
                    print_success("✅ Session is active - conversation context is being maintained")
                else:
                    print_warning("⚠️  No session - each message is treated independently")
                continue
            
            elif user_input.lower() == 'reset':
                chat_session.session_id = None
                print_success("🔄 Session reset - starting fresh conversation")
                continue
            
            elif user_input.lower() == 'help':
                print("\n📚 Available Commands:")
                print("  exit/quit/q   - Exit the program")
                print("  new           - Create new chat session")
                print("  list          - List all available chats")
                print("  clear         - Clear the screen")
                print("  session       - Show current session information")
                print("  reset         - Reset session and start fresh")
                print("  help          - Show this help message")
                print("\n💡 Tips:")
                print("  - The AI will remember your conversation across messages")
                print("  - Use 'when I say X, you answer Y' to give specific instructions")
                print("  - Session ID ensures context is maintained")
                continue
            
            elif not user_input:
                continue
            
            # Send message to AI
            print(f"{Colors.YELLOW}🤔 Thinking...{Colors.END}")
            
            result = chat_session.send_message(user_input)
            
            if result:
                print_answer(result.get('answer', 'No answer received'))
                
                # Show sources if available
                sources = result.get('sources', [])
                if sources:
                    print_sources(sources)
                
                # Show intent and execution time
                intent = result.get('intent', 'unknown')
                exec_time = result.get('execution_time', 0)
                tool_used = result.get('tool_used')
                
                print(f"{Colors.CYAN}📊 Metadata:{Colors.END}")
                print(f"  Intent: {intent}")
                print(f"  Time: {exec_time:.2f}s")
                if tool_used:
                    print(f"  Tool: {tool_used}")
                
                # Show session info if changed
                if 'session_id' in result and result['session_id'] != chat_session.session_id:
                    chat_session.session_id = result['session_id']
                    print(f"{Colors.GREEN}🔗 New session: {chat_session.get_session_id()}{Colors.END}")
                
                # Show debug info if available and debug mode is enabled
                debug_info = result.get('debug_info', [])
                if debug_info and AGENT_DEBUG_MODE:
                    print(f"{Colors.YELLOW}🐛 Debug Info:{Colors.END}")
                    for debug_msg in debug_info:
                        print(f"  - {debug_msg}")
            
            else:
                print_error("❌ Failed to get response from server")
                print_warning("💡 Try again or check server status")
        
        except KeyboardInterrupt:
            print("\n")
            print_success("👋 Goodbye!")
            break
        
        except EOFError:
            print("\n")
            print_success("👋 Goodbye!")
            break
        
        except Exception as e:
            print_error(f"Unexpected error: {e}")
            print_warning("💡 The session will continue...")

def one_shot_query(message: str, chat_id: Optional[str] = None, show_sources: bool = True):
    print_header("Sending query to server...")
    print(f"{Colors.GREEN}You:{Colors.END} {message}")
    print(f"{Colors.YELLOW} Waiting for response...{Colors.END}")
    
    result = send_message(message, chat_id)
    
    if result:
        print_answer(result.get('answer', 'No answer received'))
        
        if show_sources:
            print_sources(result.get('sources', []))
        
        # Show metadata if available
        metadata = result.get('metadata', {})
        if metadata:
            print(f"\n{Colors.YELLOW}Metadata:{Colors.END}")
            for key, value in metadata.items():
                print(f"  {key}: {value}")

def main():
    global BASE_URL
    parser = argparse.ArgumentParser(
        description='Command line client for RAG Chat System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick question
  python chat_cli.py "Πώς κάνω restart το PostgreSQL;"
  
  # Interactive mode
  python chat_cli.py -i
  
  # Use specific chat
  python chat_cli.py --chat-id abc-123 -i
  
  # Create new chat and ask question
  python chat_cli.py --new-chat "DB Help" "How do I backup?"
  
  # List all chats
  python chat_cli.py --list
        """
    )
    
    parser.add_argument(
        'message',
        nargs='?',
        help='Message to send (if not in interactive mode)'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Start interactive mode'
    )
    
    parser.add_argument(
        '--chat-id',
        type=str,
        help='Use specific chat ID'
    )
    
    parser.add_argument(
        '--new-chat',
        type=str,
        metavar='TITLE',
        help='Create a new chat with given title'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available chats'
    )
    
    parser.add_argument(
        '--no-sources',
        action='store_true',
        help='Hide source documents in output'
    )
    
    parser.add_argument(
        '--url',
        type=str,
        default=BASE_URL,
        help=f'Server URL (default: {BASE_URL})'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output raw JSON response'
    )
    
    args = parser.parse_args()
    
    # global BASE_URL
    BASE_URL = args.url.rstrip('/')
    
    if not check_server():
        print_error(f"Cannot connect to server at {BASE_URL}")
        print_error("Make sure the server is running:")
        print("  python setup_and_run.py --run")
        sys.exit(1)
    
    if args.list:
        list_chats()
        sys.exit(0)
    
    chat_id = args.chat_id
    if args.new_chat:
        chat_id = create_chat(args.new_chat)
        if not chat_id:
            sys.exit(1)
    
    # Interactive mode
    if args.interactive:
        interactive_mode(chat_id)
    
    elif args.message:
        result = send_message(args.message, chat_id)
        
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            if result:
                print_answer(result.get('answer', 'No answer received'))
                
                if not args.no_sources:
                    print_sources(result.get('sources', []))
            else:
                sys.exit(1)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()