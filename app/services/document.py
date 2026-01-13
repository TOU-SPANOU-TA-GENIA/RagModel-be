from pathlib import Path
from typing import Dict, Any, Literal
from app.services.filesystem import filesystem_service
from app.analysis.reporting.renderers import DocxRenderer # Reusing our previous refactor
# We assume we might add PDF renderer here later

class DocumentService:
    """
    Service to generate physical files (Reports, Briefings) from data.
    """
    
    def __init__(self):
        self.docx_renderer = DocxRenderer()

    def create_report(self, title: str, content: str, format: Literal["docx", "txt"] = "docx") -> str:
        """
        Creates a report file in the 'outputs' directory.
        Returns the filename.
        """
        filename = f"{title.replace(' ', '_')}_{int(time.time())}.{format}"
        output_dir = filesystem_service.output_path
        file_path = output_dir / filename
        
        if format == "docx":
            # Structure data for the renderer
            data = {
                "title": title,
                "sections": [
                    {"title": "Content", "content": content}
                ]
            }
            self.docx_renderer.render(data, file_path)
        else:
            # Text fallback
            file_path.write_text(f"{title}\n\n{content}", encoding="utf-8")
            
        return str(file_path)

import time

document_service = DocumentService()