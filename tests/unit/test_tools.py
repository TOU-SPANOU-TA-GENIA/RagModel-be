# tests/unit/test_tools.py
"""
Unit tests for tool system.
"""

import pytest
from pathlib import Path


class TestToolResult:
    """Tests for ToolResult model."""
    
    def test_success_result(self):
        from app.tools.base import ToolResult
        
        result = ToolResult(success=True, data={"key": "value"})
        
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.error is None
    
    def test_failure_result(self):
        from app.tools.base import ToolResult
        
        result = ToolResult(success=False, data=None, error="Something failed")
        
        assert result.success is False
        assert result.error == "Something failed"
    
    def test_to_dict(self):
        from app.tools.base import ToolResult
        
        result = ToolResult(success=True, data={"test": 123})
        d = result.to_dict()
        
        assert d["success"] is True
        assert d["data"]["test"] == 123


class TestReadFileTool:
    """Tests for file reading tool."""
    
    @pytest.fixture
    def read_tool(self, tmp_path):
        from app.tools.base import ReadFileTool
        return ReadFileTool(allowed_dirs=[tmp_path])
    
    @pytest.fixture
    def test_file(self, tmp_path):
        file_path = tmp_path / "test.txt"
        file_path.write_text("Hello, World!")
        return file_path
    
    def test_read_existing_file(self, read_tool, test_file):
        """Test reading an existing file."""
        result = read_tool.execute(file_path=str(test_file))
        
        assert result["success"] is True
        assert result["data"]["content"] == "Hello, World!"
    
    def test_read_nonexistent_file(self, read_tool, tmp_path):
        """Test reading a file that doesn't exist."""
        result = read_tool.execute(file_path=str(tmp_path / "nonexistent.txt"))
        
        assert result["success"] is False
        assert "not found" in result["error"].lower()
    
    def test_read_outside_allowed_dirs(self, tmp_path):
        from app.tools.base import ReadFileTool
        
        # Create tool with restricted directory
        tool = ReadFileTool(allowed_dirs=[tmp_path / "allowed"])
        
        # Try to read from disallowed location
        other_file = tmp_path / "other" / "file.txt"
        other_file.parent.mkdir(parents=True, exist_ok=True)
        other_file.write_text("secret")
        
        result = tool.execute(file_path=str(other_file))
        
        assert result["success"] is False
        assert "denied" in result["error"].lower()


class TestWriteFileTool:
    """Tests for file writing tool."""
    
    @pytest.fixture
    def write_tool(self, tmp_path):
        from app.tools.base import WriteFileTool
        return WriteFileTool(allowed_dirs=[tmp_path])
    
    def test_write_new_file(self, write_tool, tmp_path):
        """Test writing a new file."""
        file_path = tmp_path / "new_file.txt"
        
        result = write_tool.execute(
            file_path=str(file_path),
            content="Test content"
        )
        
        assert result["success"] is True
        assert file_path.exists()
        assert file_path.read_text() == "Test content"


class TestToolRegistry:
    """Tests for tool registry."""
    
    def test_register_tool(self):
        from app.tools.base import SimpleToolRegistry, ReadFileTool
        
        registry = SimpleToolRegistry()
        tool = ReadFileTool()
        
        registry.register(tool)
        
        assert "read_file" in registry.list_tools()
        assert registry.get("read_file") is tool
    
    def test_execute_registered_tool(self, tmp_path):
        from app.tools.base import SimpleToolRegistry, ReadFileTool
        
        # Setup
        registry = SimpleToolRegistry()
        tool = ReadFileTool(allowed_dirs=[tmp_path])
        registry.register(tool)
        
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        
        # Execute through registry
        result = registry.execute("read_file", file_path=str(test_file))
        
        assert result["success"] is True
    
    def test_execute_unknown_tool(self):
        from app.tools.base import SimpleToolRegistry
        
        registry = SimpleToolRegistry()
        result = registry.execute("unknown_tool")
        
        assert result["success"] is False
        assert "unknown" in result["error"].lower()