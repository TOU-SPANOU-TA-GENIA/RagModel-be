from abc import ABC, abstractmethod
from typing import List
from app.analysis.schemas import ExtractedContent, AnalysisResult

class BaseAnalyzer(ABC):
    @abstractmethod
    def analyze(self, documents: List[ExtractedContent]) -> AnalysisResult:
        pass