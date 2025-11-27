# app/llm/tool_result_formatter.py
"""
Formats tool results for LLM context.
"""

from typing import Dict, Any
from pathlib import Path

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class ToolResultFormatter:
    """Formats tool execution results for prompt inclusion."""
    
    def format(self, tool_result: Dict[str, Any]) -> str:
        """Format tool result based on type."""
        if not tool_result.get("success"):
            return self._format_error(tool_result)
        
        data = tool_result.get("data", {})
        
        # Route to appropriate formatter
        if data.get("auto_selected"):
            return self._format_auto_selected_file(data)
        elif data.get("action_required") == "choose_file":
            return self._format_file_choice(data)
        elif data.get("download_ready"):
            return self._format_download_ready(data)
        elif "content" in data:
            return self._format_file_content(data)
        elif "files" in data:
            return self._format_file_list(data)
        elif "bytes_written" in data:
            return self._format_write_result(data)
        elif "stdout" in data:
            return self._format_command_result(data)
        elif "matches" in data:
            return self._format_search_results(data)
        else:
            return self._format_generic(data)
    
    def _format_auto_selected_file(self, data: Dict) -> str:
        content = data.get("content", "")
        file_name = data.get("file_name", "file")
        selected_path = data.get("selected_path", "unknown")
        other_versions = data.get("other_versions", [])
        
        result = f"""Successfully read file: {file_name}
Location: {selected_path}
Size: {data.get('size_bytes', 0)} bytes, {data.get('lines', 0)} lines

COMPLETE FILE CONTENT:
{content}

INSTRUCTIONS:
1. Tell user you read {file_name}
2. Provide content or answer questions using it
3. DO NOT truncate unless asked"""
        
        if other_versions:
            result += "\n\nOther versions found at:"
            for path in other_versions[:3]:
                result += f"\n  - {path}"
        
        return result
    
    def _format_file_choice(self, data: Dict) -> str:
        matches = data.get("matches", [])
        query = data.get("query", "the query")
        
        if not matches:
            return "No files found."
        
        matches_text = "\n".join([
            f"{i+1}. {m['name']} - {m['path']}"
            for i, m in enumerate(matches)
        ])
        
        return f"""Multiple files found matching '{query}':

{matches_text}

INSTRUCTIONS:
Show user this list and ask them to choose by number (1-{len(matches)})."""
    
    def _format_download_ready(self, data: Dict) -> str:
        file_name = data.get("file_name", "document")
        file_type = data.get("file_type", "file")
        file_path = data.get("file_path", "")
        filename_only = Path(file_path).name
        
        return f"""Document created successfully!

File: {file_name}
Type: {file_type.upper()}
Download: /download/{filename_only}

INSTRUCTIONS:
Tell user document is ready and provide download link."""
    
    def _format_file_content(self, data: Dict) -> str:
        return f"""Successfully read file: {data.get('file_name', 'file')}
Path: {data.get('file_path', 'unknown')}
Size: {data.get('size_bytes', 0)} bytes, {data.get('lines', 0)} lines

COMPLETE FILE CONTENT:
{data.get('content', '')}

INSTRUCTIONS:
Confirm you read the file and provide content or answer questions."""
    
    def _format_file_list(self, data: Dict) -> str:
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

Total: {len(files)} files"""
    
    def _format_write_result(self, data: Dict) -> str:
        return f"""Successfully wrote file: {data.get('file_path', 'unknown')}
Bytes written: {data.get('bytes_written', 0)}"""
    
    def _format_command_result(self, data: Dict) -> str:
        result = f"""Command: {data.get('command', 'command')}
Return code: {data.get('return_code', 0)}

Output:
{data.get('stdout', '')}"""
        
        if data.get('stderr'):
            result += f"\n\nErrors:\n{data['stderr']}"
        
        return result
    
    def _format_search_results(self, data: Dict) -> str:
        matches = data.get("matches", [])
        query = data.get("query", "query")
        
        if not matches:
            return f"No files found matching '{query}'"
        
        matches_list = "\n".join([
            f"- {m['name']} ({m.get('size_mb', 0)} MB) - {m['path']}"
            for m in matches
        ])
        
        return f"""Found {len(matches)} files matching '{query}':

{matches_list}"""
    
    def _format_generic(self, data: Dict) -> str:
        if isinstance(data, dict):
            formatted = "\n".join([f"{k}: {v}" for k, v in data.items()])
            return f"Tool executed successfully:\n\n{formatted}"
        return f"Tool executed successfully:\n\n{data}"
    
    def _format_error(self, tool_result: Dict) -> str:
        error = tool_result.get("error", "Unknown error")
        
        return f"""Tool execution FAILED.

Error: {error}

INSTRUCTIONS:
1. Tell user the operation failed
2. Explain the error: "{error}"
3. Suggest alternatives
4. DO NOT pretend it succeeded"""


# Global instance
tool_result_formatter = ToolResultFormatter()