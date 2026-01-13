from typing import Literal
from pydantic import BaseModel, Field

from app.tools.base import BaseTool, ToolResult
from app.analysis.factory import AnalyzerFactory, AnalysisType
from app.analysis.schemas import ExtractedContent
from app.services.filesystem import filesystem_service

class AnalyzeDocumentSchema(BaseModel):
    filename: str = Field(..., description="The file to analyze")
    analysis_type: Literal["intelligence", "logistics"] = Field(..., description="Type of analysis to perform")

class AnalyzeDocumentTool(BaseTool):
    name = "analyze_document"
    description = "Performs deep analysis (Intelligence or Logistics) on a document. Extracts entities, anomalies, and patterns."
    args_schema = AnalyzeDocumentSchema

    def run(self, filename: str, analysis_type: str) -> ToolResult:
        try:
            # 1. Read File
            content = filesystem_service.read_file(filename)
            
            # 2. Prepare Input
            doc = ExtractedContent(
                source_name=filename,
                text_content=content,
                content_type="text/plain"
            )
            
            # 3. Create Analyzer
            analyzer = AnalyzerFactory.create(analysis_type)
            
            # 4. Run Analysis
            result = analyzer.analyze([doc])
            
            # 5. Format Output
            summary = f"Analysis Report ({analysis_type.upper()})\n"
            summary += f"Found {len(result.patterns)} patterns and {len(result.findings)} findings.\n\n"
            
            if result.findings:
                summary += "Key Findings:\n"
                for f in result.findings:
                    summary += f"- [{f.severity.upper()}] {f.title}: {f.description}\n"
            
            return ToolResult(success=True, output=summary, data=result.__dict__)
            
        except Exception as e:
            return ToolResult(success=False, output=f"Analysis failed: {e}", error=str(e))