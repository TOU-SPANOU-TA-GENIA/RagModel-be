# app/agent/intelligence_patterns.py
"""
Pattern detection for intelligence analysis requests.

Extends the decision maker to recognize commands for:
- Multi-document analysis
- Intelligence report generation
- Pattern extraction
"""

import re
from typing import Dict, Any, Optional, List

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class IntelligencePatternDetector:
    """
    Detects intelligence analysis and report generation patterns in queries.
    """
    
    # Greek and English patterns for intelligence operations
    ANALYSIS_PATTERNS = [
        # Greek patterns
        r'αναλ[υύ](?:σε|σου)\s+(?:τα\s+)?(?:έγγραφα|αρχεία|documents?)',
        r'ανάλυση\s+(?:πληροφοριών|εγγράφων|δεδομένων)',
        r'εντόπισε\s+(?:μοτίβα|patterns?|σχέσεις)',
        r'(?:εξ[αέ]γαγε|extract)\s+(?:πληροφορίες|στοιχεία|data)',
        
        # English patterns
        r'analyz?e\s+(?:the\s+)?(?:documents?|files?|reports?)',
        r'(?:extract|identify)\s+(?:patterns?|intelligence|insights?)',
        r'(?:cross[- ]?reference|correlate)\s+(?:the\s+)?(?:documents?|data)',
    ]
    
    REPORT_PATTERNS = [
        # Greek patterns
        r'(?:δημιούργησε|φτιάξε|κάνε)\s+(?:μια\s+)?(?:αναφορά|briefing|report)',
        r'(?:αναφορά|briefing)\s+(?:πληροφοριών|intelligence)',
        r'(?:συνοψ[ιί]σ[ετ]|summarize)\s+(?:τα\s+)?(?:έγγραφα|ευρήματα)',
        r'(?:ετοίμασε|prepare)\s+(?:ενημέρωση|briefing)',
        
        # English patterns
        r'(?:generate|create|make)\s+(?:an?\s+)?(?:intelligence\s+)?(?:report|briefing)',
        r'(?:prepare|compile)\s+(?:a\s+)?briefing\s+(?:document|report)?',
        r'summarize\s+(?:the\s+)?(?:documents?|findings?|analysis)',
    ]
    
    BATCH_UPLOAD_PATTERNS = [
        # Greek
        r'(?:ανέβασα|φόρτωσα|uploaded?)\s+(?:πολλά\s+)?(?:αρχεία|έγγραφα)',
        r'(?:τα\s+)?αρχεία\s+που\s+(?:ανέβασα|έστειλα)',
        r'από\s+(?:τα\s+)?(?:συνημμένα|uploads?)',
        
        # English
        r'(?:uploaded?|attached?)\s+(?:multiple\s+)?(?:files?|documents?)',
        r'(?:the\s+)?(?:files?|documents?)\s+I\s+(?:uploaded?|sent|attached)',
        r'(?:from|using)\s+(?:the\s+)?(?:uploads?|attachments?)',
    ]
    
    CLASSIFICATION_KEYWORDS = {
        'ΑΔΙΑΒΑΘΜΗΤΟ': ['αδιαβάθμητο', 'unclassified', 'δημόσιο'],
        'ΠΕΡΙΟΡΙΣΜΕΝΗΣ ΧΡΗΣΗΣ': ['περιορισμένης', 'restricted', 'internal'],
        'ΕΜΠΙΣΤΕΥΤΙΚΟ': ['εμπιστευτικό', 'confidential'],
        'ΑΠΟΡΡΗΤΟ': ['απόρρητο', 'secret', 'classified']
    }
    
    def __init__(self):
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        self._analysis_regex = [re.compile(p, re.IGNORECASE) for p in self.ANALYSIS_PATTERNS]
        self._report_regex = [re.compile(p, re.IGNORECASE) for p in self.REPORT_PATTERNS]
        self._batch_regex = [re.compile(p, re.IGNORECASE) for p in self.BATCH_UPLOAD_PATTERNS]
    
    def detect_intent(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Detect if query is requesting intelligence analysis or report generation.
        
        Returns tool info if detected, None otherwise.
        """
        query_lower = query.lower()
        
        # Check for report generation first (more specific)
        if self._matches_patterns(query, self._report_regex):
            return self._build_report_params(query)
        
        # Check for analysis request
        if self._matches_patterns(query, self._analysis_regex):
            return self._build_analysis_params(query)
        
        # Check for batch upload context
        if self._matches_patterns(query, self._batch_regex):
            # Could be either analysis or report, check for additional context
            if any(word in query_lower for word in ['αναφορά', 'report', 'briefing', 'document']):
                return self._build_report_params(query)
            return self._build_analysis_params(query)
        
        return None
    
    def _matches_patterns(self, query: str, patterns: List[re.Pattern]) -> bool:
        """Check if query matches any pattern."""
        return any(p.search(query) for p in patterns)
    
    def _build_report_params(self, query: str) -> Dict[str, Any]:
        """Build parameters for report generation."""
        params = {
            'title': self._extract_title(query) or 'Αναφορά Πληροφοριών',
            'output_format': self._extract_format(query) or 'docx',
            'classification': self._extract_classification(query),
            'include_timeline': True,
            'include_entity_graph': True
        }
        
        return {
            'name': 'generate_intelligence_report',
            'params': params
        }
    
    def _build_analysis_params(self, query: str) -> Dict[str, Any]:
        """Build parameters for document analysis."""
        return {
            'name': 'analyze_documents',
            'params': {}
        }
    
    def _extract_title(self, query: str) -> Optional[str]:
        """Extract report title from query."""
        # Look for quoted titles
        quoted = re.search(r'["\']([^"\']+)["\']', query)
        if quoted:
            return quoted.group(1)
        
        # Look for "titled/called/named X" patterns
        title_match = re.search(
            r'(?:τίτλο|titled?|called?|named?)\s+["\']?([^"\'.,]+)["\']?',
            query, re.IGNORECASE
        )
        if title_match:
            return title_match.group(1).strip()
        
        return None
    
    def _extract_format(self, query: str) -> Optional[str]:
        """Extract output format from query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['pdf']):
            return 'pdf'
        elif any(word in query_lower for word in ['markdown', 'md']):
            return 'md'
        elif any(word in query_lower for word in ['word', 'docx', 'document']):
            return 'docx'
        
        return None
    
    def _extract_classification(self, query: str) -> str:
        """Extract classification level from query."""
        query_lower = query.lower()
        
        for level, keywords in self.CLASSIFICATION_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                return level
        
        return 'ΑΔΙΑΒΑΘΜΗΤΟ'


def extend_decision_maker_patterns(decision_maker):
    """
    Extend an existing decision maker with intelligence patterns.
    
    Call this to add intelligence detection to SimpleDecisionMaker.
    """
    detector = IntelligencePatternDetector()
    original_identify = decision_maker._identify_tool
    
    def enhanced_identify(query: str) -> Optional[Dict[str, Any]]:
        # First check intelligence patterns
        intel_result = detector.detect_intent(query)
        if intel_result:
            logger.info(f"Detected intelligence request: {intel_result['name']}")
            return intel_result
        
        # Fall back to original detection
        return original_identify(query)
    
    decision_maker._identify_tool = enhanced_identify
    logger.info("Extended decision maker with intelligence patterns")


# Convenience function for pattern detection
_detector: Optional[IntelligencePatternDetector] = None


def detect_intelligence_intent(query: str) -> Optional[Dict[str, Any]]:
    """Convenience function for detecting intelligence intents."""
    global _detector
    if _detector is None:
        _detector = IntelligencePatternDetector()
    return _detector.detect_intent(query)