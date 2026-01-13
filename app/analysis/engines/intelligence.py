from typing import List, Dict
from collections import defaultdict
from app.analysis.engines.base import BaseAnalyzer
from app.analysis.schemas import ExtractedContent, AnalysisResult, AnalysisFinding
from app.analysis.patterns import PatternService

class IntelligenceAnalyzer(BaseAnalyzer):
    def __init__(self, pattern_service: PatternService):
        self.patterns = pattern_service

    def analyze(self, documents: List[ExtractedContent]) -> AnalysisResult:
        all_patterns = []
        for doc in documents:
            all_patterns.extend(self.patterns.scan(doc))

        # Core Intelligence Logic: Entity Graph
        graph = self._build_entity_graph(all_patterns)
        
        # Core Intelligence Logic: Cross-referencing
        findings = self._generate_cross_reference_findings(all_patterns)

        return AnalysisResult(
            timestamp="", # Fill in orchestrator
            doc_count=len(documents),
            patterns=all_patterns,
            findings=findings,
            graph=graph,
            summary=f"Analyzed {len(documents)} documents for intelligence patterns."
        )

    def _build_entity_graph(self, patterns) -> Dict[str, List[str]]:
        """Link entities appearing in the same context/source."""
        graph = defaultdict(list)
        # Simplified graph logic for brevity
        entities = [p for p in patterns if p.category == 'entity']
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                if e1.source == e2.source:
                    graph[e1.value].append(e2.value)
        return dict(graph)

    def _generate_cross_reference_findings(self, patterns) -> List[AnalysisFinding]:
        findings = []
        # Example logic: Find entities appearing in multiple docs
        counts = defaultdict(set)
        for p in patterns:
            counts[p.value].add(p.source)
        
        for value, sources in counts.items():
            if len(sources) > 1:
                findings.append(AnalysisFinding(
                    title=f"Cross-Document Entity: {value}",
                    description=f"Entity '{value}' appears in {len(sources)} documents.",
                    evidence=list(sources),
                    severity="medium"
                ))
        return findings