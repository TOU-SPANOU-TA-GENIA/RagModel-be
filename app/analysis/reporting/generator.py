from pathlib import Path
from typing import Dict, Any
from app.analysis.schemas import AnalysisResult
from app.analysis.reporting.renderers import DocxRenderer, ReportRenderer

class ReportGenerator:
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.renderers = {
            "docx": DocxRenderer()
        }

    def generate(self, result: AnalysisResult, format: str = "docx") -> str:
        renderer = self.renderers.get(format)
        if not renderer:
            raise ValueError(f"Format {format} not supported")

        # Transform AnalysisResult to Generic Content Dict
        content = {
            "title": f"Analysis Report - {result.timestamp}",
            "summary": result.summary,
            "sections": [
                {"title": "Findings", "content": "\n".join([f.title for f in result.findings])}
            ]
        }
        
        filename = f"report_{result.timestamp}.{format}"
        path = self.output_dir / filename
        return str(renderer.render(content, path))