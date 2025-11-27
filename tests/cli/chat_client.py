#!/usr/bin/env python3
# tests/cli/chat_client.py
"""
Interactive CLI client for testing the chat system.

Usage:
    python tests/cli/chat_client.py                    # Interactive mode
    python tests/cli/chat_client.py "Your question"   # Single query
    python tests/cli/chat_client.py --help            # Show help
"""

import requests
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

BASE_URL = "http://localhost:8000"


class Colors:
    """ANSI color codes."""
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_colored(text: str, color: str = ""):
    """Print colored text."""
    print(f"{color}{text}{Colors.END}")


def check_server() -> bool:
    """Check if server is running."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def send_message(message: str, session_id: str = None) -> dict:
    """Send message to chat endpoint."""
    try:
        params = {"session_id": session_id} if session_id else {}
        response = requests.post(
            f"{BASE_URL}/chat",
            json={"role": "user", "content": message},
            params=params,
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print_colored(f"Error: {response.status_code}", Colors.RED)
            return None
    except requests.exceptions.Timeout:
        print_colored("Request timed out", Colors.RED)
        return None
    except Exception as e:
        print_colored(f"Error: {e}", Colors.RED)
        return None


def interactive_mode():
    """Run interactive chat session."""
    print_colored("🤖 AI Agent Chat - Interactive Mode", Colors.BOLD)
    print_colored("Commands: 'exit', 'new', 'help'", Colors.YELLOW)
    print("-" * 50)
    
    session_id = None
    
    while True:
        try:
            prompt = f"{Colors.GREEN}You:{Colors.END} "
            user_input = input(prompt).strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print_colored("Goodbye! 👋", Colors.CYAN)
                break
            
            if user_input.lower() == 'new':
                session_id = None
                print_colored("🆕 New session started", Colors.YELLOW)
                continue
            
            if user_input.lower() == 'help':
                print_help()
                continue
            
            print_colored("🤔 Thinking...", Colors.YELLOW)
            result = send_message(user_input, session_id)
            
            if result:
                session_id = result.get("session_id")
                
                # Print answer
                print_colored(f"\n{Colors.CYAN}Assistant:{Colors.END}")
                print(result.get("answer", "No response"))
                
                # Print metadata
                print_colored(f"\n[Intent: {result.get('intent')} | "
                            f"Time: {result.get('execution_time', 0):.2f}s]", 
                            Colors.YELLOW)
                
                # Print sources if any
                sources = result.get("sources", [])
                if sources:
                    print_colored(f"Sources: {len(sources)}", Colors.BLUE)
                
                print()
            
        except KeyboardInterrupt:
            print_colored("\nGoodbye! 👋", Colors.CYAN)
            break
        except EOFError:
            break


def print_help():
    """Print help information."""
    print("""
Commands:
  exit, quit, q  - Exit the program
  new           - Start a new session
  help          - Show this help

Tips:
  - The AI remembers context within a session
  - Use 'new' to reset context
  - Set instructions like: "when I say X, respond Y"
""")


def single_query(message: str):
    """Execute single query and exit."""
    result = send_message(message)
    
    if result:
        print(result.get("answer", "No response"))
        return 0
    return 1


def main():
    global BASE_URL
    
    parser = argparse.ArgumentParser(
        description='CLI client for AI Agent chat',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Interactive mode
  %(prog)s "What is Python?"        # Single query
  %(prog)s --url http://server:8000 # Custom server
        """
    )
    
    parser.add_argument('message', nargs='?', help='Message to send')
    parser.add_argument('--url', default=BASE_URL, help='Server URL')
    parser.add_argument('-i', '--interactive', action='store_true',
                       help='Force interactive mode')
    
    args = parser.parse_args()
    
    BASE_URL = args.url.rstrip('/')
    
    if not check_server():
        print_colored(f"❌ Cannot connect to server at {BASE_URL}", Colors.RED)
        print("Start the server with: python scripts/run.py --run")
        sys.exit(1)
    
    if args.message and not args.interactive:
        sys.exit(single_query(args.message))
    else:
        interactive_mode()


if __name__ == "__main__":
    main()