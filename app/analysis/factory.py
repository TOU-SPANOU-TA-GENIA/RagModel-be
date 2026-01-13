from app.analysis.schemas import AnalysisType
from app.analysis.engines.intelligence import IntelligenceAnalyzer
from app.analysis.engines.logistics import LogisticsAnalyzer
from app.analysis.patterns import PatternService
from app.config import get_config

class AnalyzerFactory:
    @staticmethod
    def create(analyzer_type: str):
        config = get_config()
        patterns = PatternService(config)
        
        if analyzer_type == AnalysisType.INTELLIGENCE:
            return IntelligenceAnalyzer(patterns)
        elif analyzer_type == AnalysisType.LOGISTICS:
            return LogisticsAnalyzer(patterns)
        else:
            raise ValueError(f"Unknown analyzer type: {analyzer_type}")