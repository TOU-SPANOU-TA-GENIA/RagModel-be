from typing import Type
from pydantic import BaseModel, Field

from app.tools.base import BaseTool, ToolResult
from app.services.filesystem import filesystem_service
from app.services.document import document_service
from app.rag.retriever import rag_retriever
from app.diagnostics.service import diagnostic_service
from app.analysis.factory import AnalyzerFactory, AnalysisType
from app.analysis.schemas import ExtractedContent
from app.localization import get_status, get_error, get_success

# --- File Tools ---

class ReadFileSchema(BaseModel):
    filename: str = Field(..., description="Name of the file to read (must be in allowed directory)")

class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Reads the content of a specific file. Use this to analyze logs, reports, or documents."
    args_schema = ReadFileSchema

    def run(self, filename: str) -> ToolResult:
        try:
            content = filesystem_service.read_file(filename)
            return ToolResult(success=True, output=content)
        except Exception as e:
            return ToolResult(success=False, output=str(e), error=str(e))

class WriteFileSchema(BaseModel):
    filename: str = Field(..., description="Name of the file to save")
    content: str = Field(..., description="Text content to write")

class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Saves text to a file. Use this to save reports, summaries, or code."
    args_schema = WriteFileSchema

    def run(self, filename: str, content: str) -> ToolResult:
        try:
            path = filesystem_service.save_file(filename, content.encode('utf-8'))
            return ToolResult(success=True, output=f"File saved successfully at {path}")
        except Exception as e:
            return ToolResult(success=False, output=str(e), error=str(e))

class ListFilesSchema(BaseModel):
    subdir: str = Field("uploads", description="Subdirectory to list")

class ListFilesTool(BaseTool):
    name = "list_files"
    description = "Lists available files in the storage. Use this to see what documents are available."
    args_schema = ListFilesSchema

    def run(self, subdir: str = "uploads") -> ToolResult:
        try:
            files = filesystem_service.list_files(subdir)
            file_list = "\n".join([f"{f.filename} ({f.size} bytes)" for f in files])
            return ToolResult(success=True, output=file_list or "No files found.")
        except Exception as e:
            return ToolResult(success=False, output=str(e), error=str(e))

# --- RAG / Knowledge Tools ---

class SearchKnowledgeBaseSchema(BaseModel):
    query: str = Field(..., description="The query to search for in the knowledge base")

class SearchKnowledgeBaseTool(BaseTool):
    name = "search_knowledge_base"
    description = "Searches the vector database for relevant information (laws, manuals, history)."
    args_schema = SearchKnowledgeBaseSchema

    def run(self, query: str) -> ToolResult:
        try:
            results = rag_retriever.retrieve(query)
            if not results:
                return ToolResult(success=True, output="No relevant information found.")
            
            # Format nicely
            output = ""
            for idx, res in enumerate(results, 1):
                output += f"[{idx}] Source: {res['metadata'].get('source', 'Unknown')}\n{res['content'][:500]}...\n\n"
            
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, output=str(e), error=str(e))

# --- System / Diagnostics Tools ---

class SystemDiagnosticsSchema(BaseModel):
    check_type: str = Field("all", description="Type of check: 'gpu', 'system', or 'all'")

class SystemDiagnosticsTool(BaseTool):
    name = "system_diagnostics"
    description = "Runs a system health check (GPU status, disk space). Use this if the user asks about system status."
    args_schema = SystemDiagnosticsSchema

    def run(self, check_type: str = "all") -> ToolResult:
        try:
            report = diagnostic_service.run_diagnostics()
            # Convert report to string summary
            summary = f"Overall Status: {report.overall_status.value.upper()}\n"
            for check in report.checks:
                icon = "✅" if check.status == "pass" else "❌"
                summary += f"{icon} {check.name}: {check.message}\n"
            
            return ToolResult(success=True, output=summary, data=report.dict())
        except Exception as e:
            return ToolResult(success=False, output=str(e), error=str(e))