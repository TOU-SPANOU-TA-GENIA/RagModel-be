#!/usr/bin/env python3
# test/agent_demo.py
"""
Interactive Agent Demo

This script demonstrates the agent's capabilities with
pre-defined scenarios and interactive mode.

Usage:
    # Run all demos
    python test/test_agent.py
    
    # Interactive mode
    python test/test_agent.py --interactive
    
    # Specific demo
    python test/test_agent.py --demo file_reading
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tempfile
import shutil
from typing import Dict, Any
import json

from app.agent import create_agent, ReadFileTool
from app.logger import setup_logger

logger = setup_logger(__name__)


class Colors:
    """ANSI color codes for pretty output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(70)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")


def print_query(text: str):
    """Print user query."""
    print(f"{Colors.GREEN}User:{Colors.END} {text}")


def print_response(response: Dict[str, Any]):
    """Print agent response with formatting."""
    print(f"\n{Colors.CYAN}Agent:{Colors.END}")
    print(response["answer"])
    
    # Show metadata
    metadata = response.get("metadata", {})
    print(f"\n{Colors.YELLOW}[Metadata]{Colors.END}")
    print(f"  Intent: {metadata.get('intent', 'unknown')}")
    print(f"  Used Tool: {response.get('tool_used', 'None')}")
    print(f"  Used RAG: {metadata.get('used_rag', False)}")
    
    # Show tool result if available
    if response.get("tool_result"):
        tool_result = response["tool_result"]
        print(f"\n{Colors.YELLOW}[Tool Execution]{Colors.END}")
        print(f"  Success: {tool_result.get('success')}")
        if tool_result.get("error"):
            print(f"  Error: {tool_result['error']}")
        elif tool_result.get("data"):
            data = tool_result["data"]
            if isinstance(data, dict) and "file_path" in data:
                print(f"  File: {data['file_path']}")
                print(f"  Size: {data.get('file_size_bytes', 0)} bytes")
                print(f"  Lines: {data.get('lines', 0)}")
    
    # Show sources
    sources = response.get("sources", [])
    if sources:
        print(f"\n{Colors.YELLOW}[RAG Sources]{Colors.END}")
        for i, src in enumerate(sources, 1):
            print(f"  {i}. {src['source']} (score: {src['relevance_score']:.3f})")
    
    print()


class AgentDemo:
    """Demo scenarios for the agent."""
    
    def __init__(self):
        """Initialize demo environment."""
        self.agent = create_agent()
        # self.test_dir = Path(tempfile.mkdtemp())
        self.test_dir = Path("./data/knowledge")
        
        # Add test directory to allowed paths
        read_tool = self.agent.tool_registry.get_tool("read_file")
        if read_tool:
            read_tool.ALLOWED_DIRECTORIES.append(self.test_dir)
        
        logger.info(f"Demo environment created: {self.test_dir}")
        
        # Create sample files
        self._setup_sample_files()
    
    def _setup_sample_files(self):
        """Create sample files for demonstration."""
        # Config file
        config_file = self.test_dir / "test.txt"
        config_file.write_text("""# Server Configuration
host=192.168.1.100
port=8080
database=production_db
max_connections=100
timeout=30
""")
        
        # Log file
        log_file = self.test_dir / "application.log"
        log_file.write_text("""2024-01-15 10:00:00 [INFO] Application started
2024-01-15 10:00:05 [INFO] Database connection established
2024-01-15 10:15:23 [WARNING] High memory usage detected: 85%
2024-01-15 10:15:30 [ERROR] Database query timeout
2024-01-15 10:15:31 [INFO] Retrying database connection
2024-01-15 10:15:32 [INFO] Database connection restored
""")
        
        # Data file
        data_file = self.test_dir / "users.json"
        data_file.write_text("""{
    "users": [
        {"id": 1, "name": "Admin", "role": "administrator"},
        {"id": 2, "name": "Operator", "role": "operator"},
        {"id": 3, "name": "Viewer", "role": "viewer"}
    ],
    "total": 3
}
""")
        
        logger.info("Sample files created")
    
    def cleanup(self):
        """Clean up demo environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            logger.info("Demo environment cleaned up")
    
    def demo_file_reading(self):
        """Demonstrate file reading capability."""
        print_header("Demo 1: File Reading")
        
        config_path = self.test_dir / "test.txt"
        
        query = f"Read the test file at {config_path}"
        print_query(query)
        
        response = self.agent.process_query(
            user_query=query,
            chat_history=[],
            use_rag=False
        )
        
        print_response(response)
    
    def demo_question_answering(self):
        """Demonstrate question answering with RAG."""
        print_header("Demo 2: Question Answering")
        
        query = "What is Panos's favorite food?"
        print_query(query)
        
        response = self.agent.process_query(
            user_query=query,
            chat_history=[],
            use_rag=True
        )
        
        print_response(response)
    
    def demo_conversation(self):
        """Demonstrate conversational ability."""
        print_header("Demo 3: Conversation")
        
        queries = [
            "Hello, how are you?",
            "What can you help me with?",
            "Thanks for the information!"
        ]
        
        chat_history = []
        
        for query in queries:
            print_query(query)
            
            response = self.agent.process_query(
                user_query=query,
                chat_history=chat_history,
                use_rag=False
            )
            
            print_response(response)
            
            # Update history
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": response["answer"]})
    
    def demo_intent_classification(self):
        """Demonstrate intent classification."""
        print_header("Demo 4: Intent Classification")
        
        test_cases = [
            ("Read the log file", "ACTION"),
            ("What is PostgreSQL?", "QUESTION"),
            ("Hello there", "CONVERSATION"),
            ("Show me the config", "ACTION"),
            ("How do I restart the server?", "QUESTION"),
        ]
        
        for query, expected_intent in test_cases:
            print_query(query)
            
            # Get actual intent (this would normally be internal)
            detected_intent = self.agent._analyze_intent(query)
            
            print(f"  Expected: {expected_intent}")
            print(f"  Detected: {detected_intent.value.upper()}")
            
            match = "✓" if detected_intent.value.upper() == expected_intent else "✗"
            color = Colors.GREEN if match == "✓" else Colors.RED
            print(f"  {color}{match}{Colors.END}\n")
    
    def demo_error_handling(self):
        """Demonstrate error handling."""
        print_header("Demo 5: Error Handling")
        
        # Try to read non-existent file
        query = f"Read the file at {self.test_dir}/nonexistent.txt"
        print_query(query)
        
        response = self.agent.process_query(
            user_query=query,
            chat_history=[],
            use_rag=False
        )
        
        print_response(response)
        
        # Try to read file outside allowed directory
        print_query("Read the file at /etc/passwd")
        
        response = self.agent.process_query(
            user_query="Read the file at /etc/passwd",
            chat_history=[],
            use_rag=False
        )
        
        print_response(response)
    
    def demo_multi_turn_conversation(self):
        """Demonstrate multi-turn conversation with context."""
        print_header("Demo 6: Multi-Turn Conversation")
        
        conversation = [
            f"Read the log file at {self.test_dir}/application.log",
            "What errors did you see in that log?",
            "When did those errors occur?",
        ]
        
        chat_history = []
        
        for query in conversation:
            print_query(query)
            
            response = self.agent.process_query(
                user_query=query,
                chat_history=chat_history,
                use_rag=False
            )
            
            print_response(response)
            
            # Update history
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": response["answer"]})
            
            input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.END}")


def interactive_mode(demo: AgentDemo):
    """Interactive mode for testing agent."""
    print_header("Interactive Agent Mode")
    print("Commands:")
    print("  'exit' or 'quit' - Exit interactive mode")
    print("  'files' - List available demo files")
    print("  'history' - Show conversation history")
    print("  'clear' - Clear conversation history")
    print("-" * 70)
    
    chat_history = []
    
    while True:
        try:
            query = input(f"\n{Colors.GREEN}You:{Colors.END} ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['exit', 'quit', 'q']:
                print(f"{Colors.YELLOW}Goodbye!{Colors.END}")
                break
            
            if query.lower() == 'files':
                print(f"\n{Colors.CYAN}Available demo files:{Colors.END}")
                for file in demo.test_dir.glob("*"):
                    print(f"  - {file}")
                continue
            
            if query.lower() == 'history':
                print(f"\n{Colors.CYAN}Conversation history:{Colors.END}")
                for i, msg in enumerate(chat_history, 1):
                    role = msg['role'].capitalize()
                    content = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
                    print(f"  {i}. {role}: {content}")
                continue
            
            if query.lower() == 'clear':
                chat_history = []
                print(f"{Colors.YELLOW}History cleared{Colors.END}")
                continue
            
            # Process query
            response = demo.agent.process_query(
                user_query=query,
                chat_history=chat_history,
                use_rag=True
            )
            
            print_response(response)
            
            # Update history
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": response["answer"]})
            
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Goodbye!{Colors.END}")
            break
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.END}")
            logger.error("Error in interactive mode", exc_info=True)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Agent Demo")
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--demo",
        type=str,
        choices=['file_reading', 'question', 'conversation', 'intent', 'error', 'multi_turn', 'all'],
        default='all',
        help="Run specific demo"
    )
    
    args = parser.parse_args()
    
    demo = AgentDemo()
    
    try:
        if args.interactive:
            interactive_mode(demo)
        else:
            print_header("AI Agent Demonstration")
            print(f"Test directory: {demo.test_dir}\n")
            
            if args.demo in ['file_reading', 'all']:
                demo.demo_file_reading()
                if args.demo == 'all':
                    input(f"\n{Colors.YELLOW}Press Enter for next demo...{Colors.END}")
            
            if args.demo in ['question', 'all']:
                demo.demo_question_answering()
                if args.demo == 'all':
                    input(f"\n{Colors.YELLOW}Press Enter for next demo...{Colors.END}")
            
            if args.demo in ['conversation', 'all']:
                demo.demo_conversation()
                if args.demo == 'all':
                    input(f"\n{Colors.YELLOW}Press Enter for next demo...{Colors.END}")
            
            if args.demo in ['intent', 'all']:
                demo.demo_intent_classification()
                if args.demo == 'all':
                    input(f"\n{Colors.YELLOW}Press Enter for next demo...{Colors.END}")
            
            if args.demo in ['error', 'all']:
                demo.demo_error_handling()
                if args.demo == 'all':
                    input(f"\n{Colors.YELLOW}Press Enter for next demo...{Colors.END}")
            
            if args.demo in ['multi_turn', 'all']:
                demo.demo_multi_turn_conversation()
            
            print_header("Demo Complete")
    
    finally:
        demo.cleanup()


if __name__ == "__main__":
    main()