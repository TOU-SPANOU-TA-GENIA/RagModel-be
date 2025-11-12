# app/agent/core.py
"""
Agent Core - The Reasoning Engine

This module implements the core agent logic that:
1. Analyzes user queries to determine intent
2. Decides whether to use tools or just answer
3. Plans and executes multi-step actions
4. Integrates tool results with LLM responses

Architecture:
    User Query -> Intent Analysis -> Decision:
                                      ├─> Use Tool(s) -> Format Result -> LLM Response
                                      └─> Direct Answer (with RAG if needed)

The agent uses a "chain of thought" approach where it:
- First decides IF action is needed
- Then decides WHICH tool to use
- Executes the tool
- Synthesizes results into a natural response
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import re

from .tools import get_tool_registry, ToolResult, ToolRegistry
from ..llm import generate_answer, build_prompt
from ..vectorstore import retrieve
from ..config import (
    SYSTEM_INSTRUCTION,
    AGENT_SHOW_FILE_CONTENT,
    AGENT_FILE_CONTENT_FORMAT,
    AGENT_MAX_CONTENT_DISPLAY_LINES
)
from ..logger import setup_logger

logger = setup_logger(__name__)


class AgentIntent(Enum):
    """
    Classification of user intent.
    
    This helps the agent decide how to respond:
    - QUESTION: User wants information (use RAG + knowledge)
    - ACTION: User wants something done (use tools)
    - CONVERSATION: Casual chat (direct response)
    """
    QUESTION = "question"
    ACTION = "action"
    CONVERSATION = "conversation"


@dataclass
class AgentDecision:
    """
    Represents the agent's decision about how to handle a query.
    
    Attributes:
        intent: What type of request this is
        use_tool: Whether to use a tool
        tool_name: Which tool to use (if use_tool=True)
        tool_params: Parameters for the tool
        reasoning: Why the agent made this decision (for debugging)
    """
    intent: AgentIntent
    use_tool: bool
    tool_name: Optional[str] = None
    tool_params: Optional[Dict[str, Any]] = None
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "intent": self.intent.value,
            "use_tool": self.use_tool,
            "tool_name": self.tool_name,
            "tool_params": self.tool_params,
            "reasoning": self.reasoning
        }


class ActionParser:
    """
    Parses LLM output to extract structured action commands.
    
    The LLM is prompted to output actions in a specific format:
    <ACTION>
    {
        "tool": "tool_name",
        "parameters": {...}
    }
    </ACTION>
    
    This parser extracts and validates that format.
    """
    
    @staticmethod
    def extract_action(text: str) -> Optional[Dict[str, Any]]:
        """
        Extract action JSON from LLM response.
        
        Args:
            text: LLM output text
            
        Returns:
            Dictionary with tool and parameters, or None if no action found
        """
        # Look for <ACTION>...</ACTION> tags
        print(f"ActionParser.extract_action: text={text}")
        pattern = r'<ACTION>(.*?)</ACTION>'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        print(f"ActionParser.extract_action: match={match}")
        if not match:
            return None
        
        action_text = match.group(1).strip()
        
        try:
            # Parse JSON
            action_data = json.loads(action_text)
            
            # Validate structure
            print(f"Parsed action data: {action_data}")
            if "tool" not in action_data:
                logger.warning("Action missing 'tool' field")
                return None
            
            return action_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse action JSON: {e}")
            logger.debug(f"Action text was: {action_text}")
            return None
    
    @staticmethod
    def remove_action_tags(text: str) -> str:
        """Remove <ACTION>...</ACTION> tags from text."""
        pattern = r'<ACTION>.*?</ACTION>'
        return re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE).strip()


class Agent:
    """
    Main Agent class that orchestrates all agent capabilities.
    
    The agent workflow:
    1. Receive user query
    2. Analyze intent (is this a question or action request?)
    3. If action -> Use decision prompt to determine which tool
    4. Execute tool and get results
    5. Generate final response incorporating tool results
    6. Return to user
    
    Usage:
        agent = Agent()
        response = agent.process_query(
            user_query="Read the file at /data/config.txt",
            chat_history=[]
        )
    """
    
    def __init__(self, tool_registry: Optional[ToolRegistry] = None):
        """
        Initialize the agent.
        
        Args:
            tool_registry: Optional custom tool registry. If None, uses global registry.
        """
        self.tool_registry = tool_registry or get_tool_registry()
        logger.info(f"Agent initialized with {len(self.tool_registry.list_tools())} tools")
    
    # app/agent/core.py - Replace the existing method (around line 160)

    # app/agent/core.py - Replace _build_system_prompt_with_tools method

    def _build_system_prompt_with_tools(self) -> str:
        """
        Build enhanced system prompt that includes tool descriptions.
        
        This teaches the LLM about available tools and how to use them.
        """
        base_instruction = SYSTEM_INSTRUCTION
        
        # Get all tool schemas
        tool_schemas = self.tool_registry.get_all_schemas()
        
        if not tool_schemas:
            return base_instruction
        
        # Build tools section with VERY CLEAR instructions
        tools_description = """

    ==========================================================================
    YOU ARE AN AI AGENT WITH TWO CAPABILITIES
    ==========================================================================

    1. ANSWERING QUESTIONS using your knowledge base (context provided)
    2. PERFORMING ACTIONS using tools (when user asks you to DO something)

    ## CRITICAL RULES - READ CAREFULLY

    ### WHEN TO ANSWER DIRECTLY (WITHOUT TOOLS):
    - User asks a QUESTION (what, who, where, why, how, when)
    - Information is available in the <context> section
    - User wants to know/learn something
    - Examples:
    ✓ "What is Panos favorite food?" → Answer from context
    ✓ "Where does Panos live?" → Answer from context  
    ✓ "Tell me about X" → Answer from context
    ✓ "What does the config say?" → Answer from context

    ### WHEN TO USE TOOLS:
    - User asks you to DO/PERFORM an action
    - User wants you to READ/OPEN/SHOW a FILE
    - Information is NOT in the context
    - Examples:
    ✓ "Read the file at /data/config.txt" → Use tool
    ✓ "Show me what's in test.txt" → Use tool
    ✓ "Open the log file" → Use tool

    ==========================================================================
    YOUR AVAILABLE TOOLS
    ==========================================================================
    """
        
        for tool in tool_schemas:
            tools_description += f"\n**{tool['name']}** - {tool['description']}\n"
            tools_description += "   Parameters:\n"
            
            for param_name, param_info in tool['parameters'].items():
                required = "REQUIRED" if param_info.get('required') else "optional"
                tools_description += f"      • {param_name} ({param_info['type']}, {required}): "
                tools_description += f"{param_info['description']}\n"
        
        tools_description += """

    ==========================================================================
    HOW TO USE TOOLS
    ==========================================================================

    ONLY when the user explicitly asks you to perform an action, output:

    <ACTION>
    {
        "tool": "tool_name",
        "parameters": {
            "param": "value"
        }
    }
    </ACTION>

    Then add explanation text.

    ==========================================================================
    EXAMPLES - STUDY THESE
    ==========================================================================

    Example 1 - QUESTION (Answer directly, NO tool):
    Context: [backup_guide.txt] panos favorite food is: souvlaki
    User: "What is Panos favorite food?"
    You: "Panos's favorite food is souvlaki."

    Example 2 - QUESTION (Answer directly, NO tool):
    Context: [linux_user_guide.txt] he lives in pefki
    User: "Where does Panos live?"
    You: "Panos lives in Pefki."

    Example 3 - ACTION (Use tool):
    User: "Read the file at /data/config.txt"
    You:
    <ACTION>
    {
        "tool": "read_file",
        "parameters": {
            "file_path": "/data/config.txt"
        }
    }
    </ACTION>

    I'll read that file for you.

    Example 4 - ACTION (Use tool):
    User: "Show me what's in notes.txt"
    You:
    <ACTION>
    {
        "tool": "read_file",
        "parameters": {
            "file_path": "notes.txt"
        }
    }
    </ACTION>

    Let me open that file.

    ==========================================================================
    REMEMBER
    ==========================================================================

    IF context is provided → Answer from context (NO TOOL)
    IF user asks to read/open a file → Use read_file tool
    IF user asks a question and NO context → Say you don't know

    DO NOT use tools to answer questions when context is already provided!

    ==========================================================================
    """
        
        return base_instruction + tools_description
    # app/agent/core.py - Replace the _analyze_intent method (around line 180)

    # app/agent/core.py - Replace _analyze_intent method

    def _analyze_intent(self, user_query: str) -> AgentIntent:
        """
        Analyze the user's query to determine intent.
        
        Args:
            user_query: The user's input text
            
        Returns:
            AgentIntent classification
        """
        query_lower = user_query.lower().strip()
        
        # File-related action keywords - EXPANDED LIST
        file_action_keywords = [
            'read', 'open', 'show', 'display', 'view', 'check', 'see',
            'get', 'fetch', 'look at', 'look in', 'what\'s in',
            'content of', 'contents of', 'file at', 'file called',
            'cat', 'show me', 'display the', 'read the'
        ]
        
        # File indicators
        file_indicators = [
            '.txt', '.log', '.conf', '.json', '.py', '.md',
            'file', 'document', 'config', 'configuration'
        ]
        
        # Other action keywords
        other_action_keywords = [
            'execute', 'run', 'create', 'write', 'update', 
            'delete', 'modify', 'start', 'stop'
        ]
        
        # Question keywords - EXPANDED
        question_keywords = [
            'what', 'how', 'why', 'when', 'where', 'who', 'which',
            'explain', 'describe', 'tell me', 'what is', 'who is',
            'can you explain', 'do you know', 'what does',
            'tell me about', 'information about', 'details about'
        ]
        
        # Information-seeking patterns (even without explicit question words)
        info_patterns = [
            'favorite', 'live', 'lives', 'location', 'address',
            'like', 'prefer', 'enjoy', 'loves', 'hates',
            'works at', 'job', 'age', 'name', 'email',
            'phone', 'contact', 'birthday', 'from'
        ]
        
        # Conversational keywords
        conversation_keywords = [
            'hello', 'hi', 'hey', 'thanks', 'thank you',
            'goodbye', 'bye', 'ok', 'okay', 'good'
        ]
        
        # PRIORITY 1: Check for file reading intent
        # If they mention file actions + file indicators → ACTION
        has_file_action = any(keyword in query_lower for keyword in file_action_keywords)
        has_file_indicator = any(indicator in query_lower for indicator in file_indicators)
        
        if has_file_action or has_file_indicator:
            # Make sure it's not a question about HOW to do something
            if not any(q in query_lower for q in ['how to', 'how do i', 'how can i', 'how should i']):
                logger.info(f"Detected ACTION intent (file operation): {user_query[:50]}...")
                return AgentIntent.ACTION
        
        # PRIORITY 2: Check for other action keywords
        for keyword in other_action_keywords:
            if keyword in query_lower:
                if not any(q in query_lower for q in ['how to', 'how do i', 'how can i']):
                    logger.info(f"Detected ACTION intent: {user_query[:50]}...")
                    return AgentIntent.ACTION
        
        # PRIORITY 3: Check for short conversational greetings ONLY
        # Only classify as conversation if it's JUST a greeting (short query)
        if len(query_lower.split()) <= 3:  # Short queries only
            if any(keyword == query_lower or query_lower.startswith(keyword) for keyword in conversation_keywords):
                logger.info(f"Detected CONVERSATION intent: {user_query[:50]}...")
                return AgentIntent.CONVERSATION
        
        # PRIORITY 4: Check for information-seeking patterns
        # This catches "Panos favorite food" style queries
        if any(pattern in query_lower for pattern in info_patterns):
            logger.info(f"Detected QUESTION intent (info pattern): {user_query[:50]}...")
            return AgentIntent.QUESTION
        
        # PRIORITY 5: Check for explicit question keywords
        for keyword in question_keywords:
            if keyword in query_lower or query_lower.endswith('?'):
                logger.info(f"Detected QUESTION intent: {user_query[:50]}...")
                return AgentIntent.QUESTION
        
        # Default to QUESTION if unsure (better to search than to ignore)
        logger.info(f"Defaulting to QUESTION intent: {user_query[:50]}...")
        return AgentIntent.QUESTION
    def _make_decision(self, user_query: str, intent: AgentIntent) -> AgentDecision:
        """
        Decide how to handle the query based on intent analysis.
        
        This is where the agent's "thinking" happens. We use a simple
        rule-based approach here, but you could enhance this with
        an LLM reasoning step for complex cases.
        
        Args:
            user_query: The user's input
            intent: Classified intent
            
        Returns:
            AgentDecision with action plan
        """
        logger.info(f"Making decision for intent: {intent.value}")
        
        if intent == AgentIntent.CONVERSATION:
            return AgentDecision(
                intent=intent,
                use_tool=False,
                reasoning="Conversational query, no tools needed"
            )
        
        if intent == AgentIntent.QUESTION:
            # For questions, we'll use RAG but not tools
            return AgentDecision(
                intent=intent,
                use_tool=False,
                reasoning="Information query, will use RAG knowledge base"
            )
        
        if intent == AgentIntent.ACTION:
            # For actions, we need to ask the LLM which tool to use
            # This is handled by the LLM itself via the prompt
            return AgentDecision(
                intent=intent,
                use_tool=True,
                reasoning="Action requested, will let LLM choose appropriate tool"
            )
        
        return AgentDecision(
            intent=intent,
            use_tool=False,
            reasoning="Unable to determine clear action"
        )
    
    def _execute_tool_from_llm_output(self, llm_output: str) -> Tuple[Optional[ToolResult], str]:
        """
        Parse LLM output for action commands and execute them.
        
        Args:
            llm_output: Raw output from LLM
            
        Returns:
            (ToolResult if action was executed, cleaned text without action tags)
        """
        # Extract action from output
        print(f"llm_output: {llm_output}")
        action = ActionParser.extract_action(llm_output)
        print(f"Extracted action: {action}")
        
        if action is None:
            # No action found, return original text
            return None, llm_output
        
        logger.info(f"Extracted action: {action}")
        
        # Execute the tool
        tool_name = action.get("tool")
        parameters = action.get("parameters", {})
        
        result = self.tool_registry.execute_tool(tool_name, **parameters)
        
        # Remove action tags from text
        cleaned_text = ActionParser.remove_action_tags(llm_output)
        
        return result, cleaned_text
    
    def _format_tool_result_for_llm(self, tool_result: ToolResult) -> str:
        """
        Format tool execution result for inclusion in LLM context.
        
        This converts the structured ToolResult into natural text
        that the LLM can use to formulate its response.
        
        Args:
            tool_result: Result from tool execution
            
        Returns:
            Formatted string for LLM context
        """
        if not tool_result.success:
            return f"Tool execution failed: {tool_result.error}"
        
        # Format based on tool type
        if tool_result.tool_name == "read_file":
            data = tool_result.data
            content = data.get("content", "")
            file_path = data.get("file_path", "")
            lines = data.get("lines", 0)
            
            result_text = f"Successfully read file: {file_path}\n"
            result_text += f"File contains {lines} lines.\n\n"
            result_text += "File contents:\n"
            result_text += "```\n"
            result_text += content
            result_text += "\n```"
            
            return result_text
        
        # Generic format for unknown tools
        return f"Tool result: {json.dumps(tool_result.data, indent=2)}"
    
    # app/agent/core.py - Replace the section around line 400-450

    def process_query(
        self,
        user_query: str,
        chat_history: List[Dict[str, str]] = None,
        use_rag: bool = True
    ) -> Dict[str, Any]:
        """
        Main entry point - process a user query and return response.
        """
        if chat_history is None:
            chat_history = []
        
        logger.info(f"Agent processing query: {user_query[:100]}...")
        
        # Step 1: Analyze intent
        intent = self._analyze_intent(user_query)
        logger.info(f"Detected intent: {intent.value}")
        
        print(f"\n{'='*70}")
        print(f"DEBUG: Query: {user_query}")
        print(f"DEBUG: Intent: {intent.value}")
        print(f"DEBUG: Will use RAG: {use_rag and intent in [AgentIntent.QUESTION, AgentIntent.ACTION]}")
        print(f"{'='*70}\n")
        
        # Step 2: Make decision
        decision = self._make_decision(user_query, intent)
        logger.info(f"Decision: {decision.to_dict()}")
        
        # Step 3: Retrieve RAG context if needed
        rag_sources = []
        context_text = ""
        
        if use_rag and intent in [AgentIntent.QUESTION, AgentIntent.ACTION]:
            try:
                retrieved_docs = retrieve(user_query, k=3)
                context_parts = []
                
                for content, metadata, score in retrieved_docs:
                    source_name = metadata.get("source", "Unknown")
                    context_parts.append(f"[{source_name}]\n{content}")
                    
                    rag_sources.append({
                        "content": content[:200] + "..." if len(content) > 200 else content,
                        "source": source_name,
                        "relevance_score": round(score, 3)
                    })
                
                context_text = "\n\n".join(context_parts)
                logger.info(f"Retrieved {len(rag_sources)} RAG sources")
                
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
        
        # Step 4: Build prompt with tool awareness
        system_prompt = self._build_system_prompt_with_tools()
        
        prompt = build_prompt(
            system_instruction=system_prompt,
            context=context_text,
            history=chat_history,
            user_query=user_query
        )
        
        # Step 5: Get initial LLM response
        logger.info("Generating LLM response...")
        llm_output = generate_answer(prompt)
        
        # Step 6: Check if LLM wants to use a tool
        tool_result = None
        tool_used = None
        final_answer = llm_output  # Default to LLM output
        
        if decision.use_tool:
            tool_result, llm_output = self._execute_tool_from_llm_output(llm_output)
            
            if tool_result:
                tool_used = tool_result.tool_name
                logger.info(f"Tool executed: {tool_used}, success: {tool_result.success}")
                
                # NEW: Format the final answer with file content
                if tool_result.success:
                    if tool_used == "read_file" and AGENT_SHOW_FILE_CONTENT:
                        data = tool_result.data
                        file_path = data.get("file_path", "")
                        file_content = data.get("content", "")
                        lines = data.get("lines", 0)
                        file_size = data.get("file_size_bytes", 0)
                        
                        # Truncate if needed
                        display_content = file_content
                        truncated = False
                        if AGENT_MAX_CONTENT_DISPLAY_LINES:
                            content_lines = file_content.split('\n')
                            if len(content_lines) > AGENT_MAX_CONTENT_DISPLAY_LINES:
                                display_content = '\n'.join(content_lines[:AGENT_MAX_CONTENT_DISPLAY_LINES])
                                display_content += f"\n\n... (showing first {AGENT_MAX_CONTENT_DISPLAY_LINES} of {lines} lines)"
                                truncated = True
                        
                        # Format based on config
                        if AGENT_FILE_CONTENT_FORMAT == "pretty":
                            final_answer = f"{llm_output.strip()}\n\n"
                            final_answer += f"{'='*70}\n"
                            final_answer += f"File: {file_path}\n"
                            final_answer += f"Size: {file_size} bytes | Lines: {lines}\n"
                            final_answer += f"{'='*70}\n\n"
                            final_answer += display_content
                            if truncated:
                                final_answer += f"\n\n{'='*70}\n"
                                final_answer += f"(File truncated for display)\n"
                        
                        elif AGENT_FILE_CONTENT_FORMAT == "minimal":
                            final_answer = f"{llm_output.strip()}\n\n"
                            final_answer += f"[{file_path}]\n"
                            final_answer += display_content
                        
                        else:  # raw
                            final_answer = f"{llm_output.strip()}\n\n{display_content}"
                        
                        logger.info(f"Added file content to answer ({len(display_content)} chars)")
                    else:
                        # Other tools or file content display disabled
                        tool_context = self._format_tool_result_for_llm(tool_result)
                        final_answer = f"{llm_output.strip()}\n\n{tool_context}"
                else:
                    # Tool failed
                    final_answer = f"I tried to perform that action, but encountered an error: {tool_result.error}\n\n{llm_output}"
        
        # Step 7: Return complete response
        response = {
            "answer": final_answer.strip(),
            "tool_used": tool_used,
            "tool_result": tool_result.to_dict() if tool_result else None,
            "sources": rag_sources,
            "decision": decision.to_dict(),
            "metadata": {
                "intent": intent.value,
                "used_rag": len(rag_sources) > 0,
                "used_tool": tool_used is not None
            }
        }
        
        logger.info(f"Agent response complete. Tool used: {tool_used}, RAG sources: {len(rag_sources)}")
        
        return response

def create_agent() -> Agent:
    """
    Factory function to create a configured agent instance.
    
    This is the main way to get an agent in your application.
    """
    agent = Agent()
    logger.info("Agent created and ready")
    return agent