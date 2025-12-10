# app/analysis/intelligence_analyzer.py
"""
Intelligence pattern analysis across multiple documents.
Identifies key patterns, cross-references, and generates insights.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict
import re

from app.utils.logger import setup_logger
from app.analysis.content_extractors import ExtractedContent

logger = setup_logger(__name__)


@dataclass
class IntelligencePattern:
    """A detected intelligence pattern."""
    pattern_type: str  # entity, location, date, keyword, relationship
    value: str
    confidence: float  # 0.0 - 1.0
    sources: List[str] = field(default_factory=list)  # Document names where found
    context_snippets: List[str] = field(default_factory=list)
    frequency: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_occurrence(self, source: str, context: str):
        """Add another occurrence of this pattern."""
        if source not in self.sources:
            self.sources.append(source)
        self.context_snippets.append(context[:200])  # Limit context length
        self.frequency += 1


@dataclass
class CrossReference:
    """A cross-reference between documents."""
    source_doc: str
    target_doc: str
    reference_type: str  # mentions, contradicts, confirms, extends
    common_elements: List[str]
    relevance_score: float
    notes: str = ""


@dataclass
class AnalysisResult:
    """Complete analysis result from intelligence analyzer."""
    documents_analyzed: int
    analysis_timestamp: str
    patterns: List[IntelligencePattern]
    cross_references: List[CrossReference]
    key_findings: List[str]
    summary: str
    entity_graph: Dict[str, List[str]]  # Entity relationships
    timeline: List[Dict[str, Any]]  # Chronological events
    confidence_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "documents_analyzed": self.documents_analyzed,
            "analysis_timestamp": self.analysis_timestamp,
            "patterns": [
                {
                    "type": p.pattern_type,
                    "value": p.value,
                    "confidence": p.confidence,
                    "sources": p.sources,
                    "frequency": p.frequency
                }
                for p in self.patterns
            ],
            "cross_references": [
                {
                    "source": cr.source_doc,
                    "target": cr.target_doc,
                    "type": cr.reference_type,
                    "common_elements": cr.common_elements,
                    "score": cr.relevance_score
                }
                for cr in self.cross_references
            ],
            "key_findings": self.key_findings,
            "summary": self.summary,
            "entity_graph": self.entity_graph,
            "timeline": self.timeline,
            "confidence_score": self.confidence_score
        }


class PatternDetector:
    """Detects various patterns in text content."""
    
    # Greek/English date patterns
    DATE_PATTERNS = [
        r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}',  # DD/MM/YYYY
        r'\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}',  # YYYY/MM/DD
        r'\d{1,2}\s+(Ιαν|Φεβ|Μαρ|Απρ|Μαϊ|Ιουν|Ιουλ|Αυγ|Σεπ|Οκτ|Νοε|Δεκ)\w*\s+\d{4}',  # Greek months
        r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}',  # English months
    ]
    
    # Location patterns (coordinates, place names)
    LOCATION_PATTERNS = [
        r'\d{1,3}°\s*\d{1,2}[\'\′]\s*\d{1,2}[\"\″]?\s*[NSEW]',  # Coordinates
        r'(latitude|longitude|lat|lon|πλάτος|μήκος)[\s:]+[\d\.\-]+',
        r'(περιοχή|τοποθεσία|θέση|location|area|position)[\s:]+([Α-Ωα-ω\w\s]+)',
    ]
    
    # Military/intelligence keywords (Greek + English)
    INTELLIGENCE_KEYWORDS = {
        'high_priority': [
            'επείγον', 'άμεσο', 'κρίσιμο', 'urgent', 'critical', 'immediate',
            'απόρρητο', 'εμπιστευτικό', 'classified', 'confidential', 'secret',
            'απειλή', 'κίνδυνος', 'threat', 'danger', 'risk'
        ],
        'entities': [
            'μονάδα', 'unit', 'τμήμα', 'section', 'ομάδα', 'team', 'group',
            'οργάνωση', 'organization', 'δύναμη', 'force'
        ],
        'actions': [
            'κίνηση', 'movement', 'μεταφορά', 'transfer', 'ανάπτυξη', 'deployment',
            'επιχείρηση', 'operation', 'αποστολή', 'mission'
        ],
        'equipment': [
            'όπλο', 'weapon', 'όχημα', 'vehicle', 'εξοπλισμός', 'equipment',
            'σύστημα', 'system', 'radar', 'ραντάρ'
        ]
    }
    
    def __init__(self):
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        self._date_regex = [re.compile(p, re.IGNORECASE) for p in self.DATE_PATTERNS]
        self._location_regex = [re.compile(p, re.IGNORECASE) for p in self.LOCATION_PATTERNS]
    
    def detect_dates(self, text: str, source: str) -> List[IntelligencePattern]:
        """Detect date patterns in text."""
        patterns = []
        
        for regex in self._date_regex:
            for match in regex.finditer(text):
                # Get surrounding context
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                pattern = IntelligencePattern(
                    pattern_type="date",
                    value=match.group(),
                    confidence=0.9,
                    sources=[source],
                    context_snippets=[context]
                )
                patterns.append(pattern)
        
        return patterns
    
    def detect_locations(self, text: str, source: str) -> List[IntelligencePattern]:
        """Detect location patterns in text."""
        patterns = []
        
        for regex in self._location_regex:
            for match in regex.finditer(text):
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                pattern = IntelligencePattern(
                    pattern_type="location",
                    value=match.group(),
                    confidence=0.85,
                    sources=[source],
                    context_snippets=[context]
                )
                patterns.append(pattern)
        
        return patterns
    
    def detect_keywords(self, text: str, source: str) -> List[IntelligencePattern]:
        """Detect intelligence-related keywords."""
        patterns = []
        text_lower = text.lower()
        
        for category, keywords in self.INTELLIGENCE_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    # Find all occurrences
                    for match in re.finditer(re.escape(keyword), text_lower):
                        start = max(0, match.start() - 50)
                        end = min(len(text), match.end() + 50)
                        context = text[start:end]
                        
                        confidence = 0.95 if category == 'high_priority' else 0.75
                        
                        pattern = IntelligencePattern(
                            pattern_type=f"keyword_{category}",
                            value=keyword,
                            confidence=confidence,
                            sources=[source],
                            context_snippets=[context],
                            metadata={"category": category}
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def detect_entities(self, text: str, source: str) -> List[IntelligencePattern]:
        """Detect named entities (simplified NER)."""
        patterns = []
        
        # Pattern for Greek/English proper nouns (capitalized words)
        entity_pattern = re.compile(r'\b[Α-ΩA-Z][α-ωa-z]+(?:\s+[Α-ΩA-Z][α-ωa-z]+)*\b')
        
        # Common non-entity words to filter
        stopwords = {'The', 'This', 'That', 'Ο', 'Η', 'Το', 'Αυτός', 'Αυτή', 'Αυτό'}
        
        for match in entity_pattern.finditer(text):
            value = match.group()
            if value not in stopwords and len(value) > 2:
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                pattern = IntelligencePattern(
                    pattern_type="entity",
                    value=value,
                    confidence=0.6,  # Lower confidence for simple NER
                    sources=[source],
                    context_snippets=[context]
                )
                patterns.append(pattern)
        
        return patterns


class IntelligenceAnalyzer:
    """
    Analyzes multiple documents for intelligence patterns and insights.
    """
    
    def __init__(self, llm_provider=None):
        self.pattern_detector = PatternDetector()
        self.llm = llm_provider  # Optional LLM for advanced analysis
    
    def analyze(self, documents: List[ExtractedContent]) -> AnalysisResult:
        """
        Perform comprehensive analysis on multiple documents.
        """
        logger.info(f"Starting intelligence analysis on {len(documents)} documents")
        
        # Step 1: Detect patterns in each document
        all_patterns = self._detect_all_patterns(documents)
        
        # Step 2: Merge and deduplicate patterns
        merged_patterns = self._merge_patterns(all_patterns)
        
        # Step 3: Find cross-references between documents
        cross_refs = self._find_cross_references(documents, merged_patterns)
        
        # Step 4: Build entity relationship graph
        entity_graph = self._build_entity_graph(merged_patterns)
        
        # Step 5: Extract timeline
        timeline = self._build_timeline(merged_patterns, documents)
        
        # Step 6: Generate key findings
        key_findings = self._generate_findings(merged_patterns, cross_refs, documents)
        
        # Step 7: Generate summary
        summary = self._generate_summary(documents, merged_patterns, key_findings)
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(merged_patterns, cross_refs)
        
        return AnalysisResult(
            documents_analyzed=len(documents),
            analysis_timestamp=datetime.now().isoformat(),
            patterns=merged_patterns,
            cross_references=cross_refs,
            key_findings=key_findings,
            summary=summary,
            entity_graph=entity_graph,
            timeline=timeline,
            confidence_score=confidence
        )
    
    def _detect_all_patterns(self, documents: List[ExtractedContent]) -> List[IntelligencePattern]:
        """Detect patterns in all documents."""
        patterns = []
        
        for doc in documents:
            text = doc.text_content
            source = doc.source_name
            
            patterns.extend(self.pattern_detector.detect_dates(text, source))
            patterns.extend(self.pattern_detector.detect_locations(text, source))
            patterns.extend(self.pattern_detector.detect_keywords(text, source))
            patterns.extend(self.pattern_detector.detect_entities(text, source))
        
        return patterns
    
    def _merge_patterns(self, patterns: List[IntelligencePattern]) -> List[IntelligencePattern]:
        """Merge duplicate patterns and aggregate sources."""
        merged: Dict[Tuple[str, str], IntelligencePattern] = {}
        
        for pattern in patterns:
            key = (pattern.pattern_type, pattern.value.lower())
            
            if key in merged:
                existing = merged[key]
                for source in pattern.sources:
                    existing.add_occurrence(source, pattern.context_snippets[0] if pattern.context_snippets else "")
            else:
                merged[key] = pattern
        
        # Sort by frequency and confidence
        result = list(merged.values())
        result.sort(key=lambda p: (p.frequency, p.confidence), reverse=True)
        
        return result
    
    def _find_cross_references(
        self, 
        documents: List[ExtractedContent],
        patterns: List[IntelligencePattern]
    ) -> List[CrossReference]:
        """Find relationships between documents."""
        cross_refs = []
        doc_names = [d.source_name for d in documents]
        
        # Find patterns that appear in multiple documents
        for pattern in patterns:
            if len(pattern.sources) > 1:
                # Create cross-references between all pairs
                for i, source1 in enumerate(pattern.sources):
                    for source2 in pattern.sources[i+1:]:
                        cross_ref = CrossReference(
                            source_doc=source1,
                            target_doc=source2,
                            reference_type="confirms",  # Same pattern in both
                            common_elements=[pattern.value],
                            relevance_score=pattern.confidence,
                            notes=f"Κοινό στοιχείο: {pattern.pattern_type}"
                        )
                        cross_refs.append(cross_ref)
        
        # Deduplicate cross-references
        seen = set()
        unique_refs = []
        for cr in cross_refs:
            key = (cr.source_doc, cr.target_doc)
            if key not in seen:
                seen.add(key)
                seen.add((cr.target_doc, cr.source_doc))
                unique_refs.append(cr)
        
        return unique_refs
    
    def _build_entity_graph(self, patterns: List[IntelligencePattern]) -> Dict[str, List[str]]:
        """Build a graph of entity relationships."""
        graph = defaultdict(list)
        
        entities = [p for p in patterns if p.pattern_type == "entity"]
        
        # Connect entities that appear in the same documents
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                # Check if they share any sources
                common_sources = set(e1.sources) & set(e2.sources)
                if common_sources:
                    graph[e1.value].append(e2.value)
                    graph[e2.value].append(e1.value)
        
        return dict(graph)
    
    def _build_timeline(
        self, 
        patterns: List[IntelligencePattern],
        documents: List[ExtractedContent]
    ) -> List[Dict[str, Any]]:
        """Build chronological timeline from date patterns."""
        timeline = []
        
        date_patterns = [p for p in patterns if p.pattern_type == "date"]
        
        for dp in date_patterns:
            timeline.append({
                "date": dp.value,
                "sources": dp.sources,
                "context": dp.context_snippets[0] if dp.context_snippets else "",
                "confidence": dp.confidence
            })
        
        # Sort chronologically (basic sorting, may need parsing for accuracy)
        timeline.sort(key=lambda x: x["date"])
        
        return timeline
    
    def _generate_findings(
        self,
        patterns: List[IntelligencePattern],
        cross_refs: List[CrossReference],
        documents: List[ExtractedContent]
    ) -> List[str]:
        """Generate key findings from the analysis."""
        findings = []
        
        # Finding 1: Document overview
        doc_types = defaultdict(int)
        for doc in documents:
            doc_types[doc.content_type] += 1
        type_summary = ", ".join(f"{count} {dtype}" for dtype, count in doc_types.items())
        findings.append(f"Αναλύθηκαν {len(documents)} έγγραφα ({type_summary})")
        
        # Finding 2: High-priority items
        high_priority = [p for p in patterns if 'high_priority' in p.pattern_type]
        if high_priority:
            hp_values = list(set(p.value for p in high_priority[:5]))
            findings.append(f"Εντοπίστηκαν {len(high_priority)} στοιχεία υψηλής προτεραιότητας: {', '.join(hp_values)}")
        
        # Finding 3: Most frequent entities
        entities = [p for p in patterns if p.pattern_type == "entity" and p.frequency > 1]
        if entities:
            top_entities = [f"{e.value} ({e.frequency}x)" for e in entities[:5]]
            findings.append(f"Συχνότερες οντότητες: {', '.join(top_entities)}")
        
        # Finding 4: Cross-document connections
        if cross_refs:
            findings.append(f"Βρέθηκαν {len(cross_refs)} διασυνδέσεις μεταξύ εγγράφων")
        
        # Finding 5: Location intelligence
        locations = [p for p in patterns if p.pattern_type == "location"]
        if locations:
            loc_values = list(set(p.value for p in locations[:3]))
            findings.append(f"Εντοπίστηκαν {len(locations)} αναφορές τοποθεσίας: {', '.join(loc_values)}")
        
        return findings
    
    def _generate_summary(
        self,
        documents: List[ExtractedContent],
        patterns: List[IntelligencePattern],
        findings: List[str]
    ) -> str:
        """Generate executive summary."""
        # If LLM is available, use it for better summary
        if self.llm:
            return self._llm_summary(documents, patterns, findings)
        
        # Fallback to template-based summary
        summary_parts = [
            f"Η ανάλυση περιλαμβάνει {len(documents)} έγγραφα.",
            f"Εντοπίστηκαν {len(patterns)} μοτίβα πληροφοριών.",
        ]
        
        high_priority_count = len([p for p in patterns if 'high_priority' in p.pattern_type])
        if high_priority_count > 0:
            summary_parts.append(f"ΠΡΟΣΟΧΗ: {high_priority_count} στοιχεία υψηλής προτεραιότητας.")
        
        summary_parts.append("Κύρια ευρήματα:")
        for finding in findings[:3]:
            summary_parts.append(f"• {finding}")
        
        return "\n".join(summary_parts)
    
    def _llm_summary(
        self,
        documents: List[ExtractedContent],
        patterns: List[IntelligencePattern],
        findings: List[str]
    ) -> str:
        """Use LLM to generate intelligent summary."""
        prompt = f"""Δημιούργησε μια σύντομη σύνοψη (3-5 προτάσεις) για την παρακάτω ανάλυση πληροφοριών:

Έγγραφα: {len(documents)}
Εντοπισμένα μοτίβα: {len(patterns)}
Κύρια ευρήματα:
{chr(10).join('- ' + f for f in findings)}

Η σύνοψη πρέπει να είναι στα Ελληνικά και να επικεντρώνεται στα πιο σημαντικά σημεία."""

        try:
            return self.llm.generate(prompt, max_tokens=300)
        except Exception as e:
            logger.error(f"LLM summary failed: {e}")
            return self._generate_summary.__doc__ or "Σφάλμα δημιουργίας σύνοψης"
    
    def _calculate_confidence(
        self,
        patterns: List[IntelligencePattern],
        cross_refs: List[CrossReference]
    ) -> float:
        """Calculate overall analysis confidence score."""
        if not patterns:
            return 0.0
        
        # Average pattern confidence
        pattern_conf = sum(p.confidence for p in patterns) / len(patterns)
        
        # Bonus for cross-references (indicates corroboration)
        cross_ref_bonus = min(len(cross_refs) * 0.05, 0.2)
        
        # Bonus for high-frequency patterns
        high_freq = len([p for p in patterns if p.frequency > 2])
        freq_bonus = min(high_freq * 0.02, 0.1)
        
        return min(pattern_conf + cross_ref_bonus + freq_bonus, 1.0)


def analyze_documents(documents: List[ExtractedContent], llm_provider=None) -> AnalysisResult:
    """Convenience function for document analysis."""
    analyzer = IntelligenceAnalyzer(llm_provider)
    return analyzer.analyze(documents)