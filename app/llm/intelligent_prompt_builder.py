# app/llm/intelligent_prompt_builder.py
"""
Intelligent prompt builder with context relevance filtering.
Uses dynamic context selection instead of including everything.
"""

from typing import Dict, Any, Optional, List
from app.core.interfaces import PromptBuilder, Context
from app.core.conversation_memory import conversation_memory
from app.core.context_relevance_filter import context_filter
from app.logger import setup_logger

logger = setup_logger(__name__)

class IntelligentPromptBuilder(PromptBuilder):
    """
    Prompt builder that intelligently selects relevant context.
    Prevents context bleeding and improves response quality.
    """
    
    def __init__(self, system_instruction: str, tools: Optional[Dict[str, Any]] = None):
        self.system_instruction = system_instruction
        self.tools = tools or {}
    
    def build(self, context: Context, **kwargs) -> str:
        """Build prompt with intelligently filtered context."""
        # Get or create conversation session
        session_id = context.metadata.get("session_id")
        session = conversation_memory.get_or_create_session(session_id)
        
        # Don't add current message here - it's already in context.query
        # session.add_message("user", context.query)  # REMOVED
        
        # Filter context based on query relevance
        filtered = context_filter.filter_context(
            query=context.query,
            messages=session.messages,
            user_facts=session.metadata.get("user_facts", {}),
            instructions=session.user_instructions
        )
        
        # Build prompt sections
        sections = []
        
        # 1. System instruction
        sections.append(f"<system>\n{self.system_instruction}\n</system>")
        
        # 2. Active instructions (if any)
        if filtered["active_instructions"]:
            instructions_text = self._format_instructions(filtered["active_instructions"])
            sections.append(f"<active_instructions>\n{instructions_text}\n</active_instructions>")
        
        # 3. Relevant user facts (if any)
        if filtered["relevant_facts"]:
            facts_text = self._format_facts(filtered["relevant_facts"])
            sections.append(f"<user_info>\n{facts_text}\n</user_info>")
        
        # 4. Relevant conversation history (if any)
        if filtered["should_use_history"] and filtered["relevant_messages"]:
            history_text = self._format_history(filtered["relevant_messages"])
            sections.append(f"<conversation_history>\n{history_text}\n</conversation_history>")
        
        # 5. RAG context (if provided)
        rag_context = kwargs.get("rag_context", "")
        if rag_context:
            sections.append(f"<knowledge_base>\n{rag_context}\n</knowledge_base>")
        
        # 6. Tool results (if any)
        tool_result = kwargs.get("tool_result")
        if tool_result:
            tool_text = self._format_tool_result(tool_result)
            sections.append(f"<tool_result>\n{tool_text}\n</tool_result>")
        
        # 7. Current query
        sections.append(f"<current_query>\nUser: {context.query}\n</current_query>")
        
        # 8. Response instruction
        sections.append("\nAssistant:")
        
        # Build final prompt
        full_prompt = "\n\n".join(sections)
        
        # Store session ID for response tracking
        context.metadata["session_id"] = session.session_id
        
        logger.debug(f"Built intelligent prompt - Query type: {filtered['query_type']}, "
                    f"History: {len(filtered['relevant_messages'])}, "
                    f"Facts: {len(filtered['relevant_facts'])}")
        
        return full_prompt
    
    def _format_instructions(self, instructions: Dict[str, Any]) -> str:
        """Format active instructions."""
        lines = []
        
        for key, instruction in instructions.items():
            # New format: instructions are dicts with 'description'
            if isinstance(instruction, dict):
                desc = instruction.get('description', '')
                if desc:
                    lines.append(f"- {desc}")
            # Old format: plain strings
            elif isinstance(instruction, str):
                lines.append(f"- {instruction}")
        
        return "\n".join(lines) if lines else "No active instructions"
    
    def _format_facts(self, facts: Dict[str, Any]) -> str:
        """Format relevant user facts."""
        lines = []
        for key, value in facts.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines) if lines else "No user information"
    
    def _format_history(self, messages: List[Dict[str, str]]) -> str:
        """Format relevant conversation history."""
        lines = []
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines) if lines else "No conversation history"
    
    # Complete _format_tool_result method for app/llm/intelligent_prompt_builder.py

    def _format_tool_result(self, tool_result: Dict[str, Any]) -> str:
        """
        Format tool execution result for LLM context.
        Handles all tool result types with clear instructions.
        """
        if tool_result.get("success"):
            data = tool_result.get("data", {})
            
            # Case 1: Auto-selected file (NEW - handles multiple matches automatically)
            if data.get("auto_selected"):
                content = data.get("content", "")
                file_name = data.get("file_name", "file")
                selected_path = data.get("selected_path", "unknown")
                other_versions = data.get("other_versions", [])
                lines = data.get("lines", 0)
                size = data.get("size_bytes", 0)
                
                result_text = f"""Successfully read file: {file_name}
    Location: {selected_path}
    Size: {size} bytes, {lines} lines

    COMPLETE FILE CONTENT:
    {content}

    INSTRUCTIONS FOR YOUR RESPONSE:
    1. Tell the user you successfully read {file_name}
    2. Provide the complete content or answer questions based on it
    3. DO NOT truncate or summarize unless specifically asked
    4. DO NOT ask "which file" - this IS the file you read"""
                
                # Add info about other versions if they exist
                if other_versions:
                    result_text += f"\n\nNote: Found other versions at:"
                    for path in other_versions[:3]:  # Show max 3
                        result_text += f"\n  - {path}"
                    result_text += "\n\n(I automatically selected the one from the knowledge directory)"
                
                return result_text
            
            # Case 2: Multiple file matches requiring user choice (FALLBACK - shouldn't happen with auto-select)
            elif data.get("action_required") == "choose_file":
                matches = data.get("matches", [])
                query = data.get("query", "the query")
                
                if not matches:
                    return "No files found matching the search criteria."
                
                # Format matches list with full paths
                matches_list = []
                for i, match in enumerate(matches, 1):
                    matches_list.append(f"{i}. {match['name']} - {match['path']}")
                
                matches_text = "\n".join(matches_list)
                
                return f"""Multiple files found matching '{query}':

    {matches_text}

    CRITICAL INSTRUCTIONS:
    You MUST show the user this EXACT numbered list above.
    Ask them to choose by number (1-{len(matches)}) or provide the full path.

    Format your response EXACTLY like this:
    "I found {len(matches)} files matching '{query}':

    {matches_text}

    Which file would you like me to read? Please respond with a number (1-{len(matches)}) or the full path."

    DO NOT give a vague response. Show the ACTUAL list."""
            
            # Case 3: Successful file read (standard single match)
            elif "content" in data:
                file_name = data.get("file_name", "file")
                file_path = data.get("file_path", "unknown")
                content = data.get("content", "")
                lines = data.get("lines", 0)
                size = data.get("size_bytes", 0)
                
                result_text = f"""Successfully read file: {file_name}
    Path: {file_path}
    Size: {size} bytes, {lines} lines

    COMPLETE FILE CONTENT:
    {content}

    INSTRUCTIONS FOR YOUR RESPONSE:
    1. Confirm you read "{file_name}"
    2. Provide the content or answer questions using it
    3. DO NOT truncate or summarize unless asked
    4. The complete content is shown above - use it"""
                
                # Check if matched from search
                if data.get("matched_from_search"):
                    result_text += f"\n\n(Found this file by searching for: {data.get('search_query', 'file')})"
                
                return result_text
            
            # Case 4: File listing results
            elif "files" in data:
                files = data.get("files", [])
                directory = data.get("directory", "directory")
                
                if not files:
                    return f"No files found in {directory}"
                
                files_list = "\n".join([
                    f"- {f['name']} ({f['size_bytes']} bytes)"
                    for f in files
                ])
                
                return f"""Files in {directory}:

    {files_list}

    Total: {len(files)} files

    INSTRUCTIONS: List these files to the user in a clear format."""
            
            # Case 5: File write success
            elif "bytes_written" in data:
                file_path = data.get("file_path", "unknown")
                bytes_written = data.get("bytes_written", 0)
                
                return f"""Successfully wrote file: {file_path}
    Bytes written: {bytes_written}

    INSTRUCTIONS: Confirm to the user that the file was created successfully."""
            
            # Case 6: Command execution results
            elif "stdout" in data:
                command = data.get("command", "command")
                stdout = data.get("stdout", "")
                stderr = data.get("stderr", "")
                return_code = data.get("return_code", 0)
                
                result_text = f"""Command executed: {command}
    Return code: {return_code}

    Output:
    {stdout}"""
                
                if stderr:
                    result_text += f"\n\nErrors/Warnings:\n{stderr}"
                
                return result_text
            
            # Case 7: Search results
            elif "matches" in data and data.get("action_required") != "choose_file":
                matches = data.get("matches", [])
                query = data.get("query", "query")
                
                if not matches:
                    return f"No files found matching '{query}'"
                
                matches_list = "\n".join([
                    f"- {m['name']} ({m['size_mb']} MB) - {m['path']}"
                    for m in matches
                ])
                
                return f"""Found {len(matches)} files matching '{query}':

    {matches_list}

    INSTRUCTIONS: Show this list to the user."""
            
            # Case 8: Generic successful tool execution
            else:
                # Try to format data nicely
                if isinstance(data, dict):
                    formatted_data = "\n".join([f"{k}: {v}" for k, v in data.items()])
                    return f"Tool executed successfully:\n\n{formatted_data}"
                else:
                    return f"Tool executed successfully:\n\n{data}"
        
        else:
            # Tool execution failed
            error = tool_result.get("error", "Unknown error occurred")
            
            return f"""Tool execution FAILED.

    Error: {error}

    CRITICAL INSTRUCTIONS:
    1. Tell the user the operation failed
    2. Explain the error clearly: "{error}"
    3. Suggest what they should try instead
    4. DO NOT ask them to provide information manually
    5. DO NOT pretend the operation succeeded

    Example response:
    "I tried to read the file but encountered an error: {error}

    This might mean [explain likely cause]. You could try [suggest solution]."
    """
    
    
    