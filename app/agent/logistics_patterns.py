# app/agent/logistics_patterns.py
"""
Pattern detection for logistics analysis requests.

Extends the decision maker to recognize commands for:
- Inventory audits
- Supply chain analysis
- Resource allocation checks
- Budget anomaly detection
"""

import re
from typing import Dict, Any, Optional, List

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class LogisticsPatternDetector:
    """
    Detects logistics analysis and audit patterns in queries.
    """
    
    # Greek and English patterns for logistics operations
    ANOMALY_DETECTION_PATTERNS = [
        # Greek patterns
        r'(?:εντ[όο]πισε|βρ[εέ]ς?)\s+(?:ανωμαλ[ίι]ες|αποκλ[ιί]σεις|ασυμφων[ιί]ες)',
        r'(?:[έε]λεγξε?|επαλ[ηή]θευσε?)\s+(?:απ[οό]θεμα|inventory|αποθ[έη]κη)',
        r'(?:σ[υύ]γκρινε|compare)\s+(?:τα\s+)?(?:αρχε[ιί]α|έγγραφα|στοιχε[ιί]α)',
        r'(?:αν[αά]λυση|analysis)\s+(?:εφοδιαστικ[ήη]ς?|logistics|προμηθει[ωώ]ν)',
        r'(?:audit|[έε]λεγχος)\s+(?:αποθ[εέ]ματος|inventory|υλικ[ωώ]ν)',
        
        # English patterns
        r'(?:detect|find|identify)\s+(?:anomal(?:y|ies)|discrepanc(?:y|ies)|inconsistenc(?:y|ies))',
        r'(?:check|verify|audit)\s+(?:inventory|stock|supplies?)',
        r'(?:compare|cross[- ]?reference)\s+(?:the\s+)?(?:records?|documents?|data)',
        r'(?:analyz?e|examine)\s+(?:logistics|supply\s+chain|procurement)',
    ]
    
    INVENTORY_CHECK_PATTERNS = [
        # Greek
        r'(?:π[οό]σα?|τι\s+ποσ[οό]τητα)\s+(?:έχουμε|υπ[αά]ρχουν|διαθ[έε]τουμε)',
        r'(?:κατ[αά]σταση|status)\s+(?:αποθ[εέ]ματος|αποθ[ηή]κης|υλικ[ωώ]ν)',
        r'(?:απογραφ[ήη]|καταμ[έε]τρηση|inventory\s+count)',
        
        # English
        r'(?:how\s+(?:much|many)|what(?:\'s| is) the)\s+(?:inventory|stock|count)',
        r'(?:inventory|stock)\s+(?:status|level|count|check)',
    ]
    
    BUDGET_ANALYSIS_PATTERNS = [
        # Greek
        r'(?:προϋπολογισμ[οό]ς?|budget)\s+(?:αν[αά]λυση|analysis|σ[υύ]γκριση)',
        r'(?:δαπ[αά]νες?|expenses?|κ[οό]στος|cost)\s+(?:[έε]λεγχος|analysis|ανωμαλ[ιί]ες?)',
        r'(?:υπ[εέ]ρβαση|over(?:run|spend|budget))',
        
        # English  
        r'(?:budget|expense|cost)\s+(?:analysis|anomal(?:y|ies)|variance)',
        r'(?:unusual|abnormal)\s+(?:spending|expense|cost)',
    ]
    
    MAINTENANCE_PATTERNS = [
        # Greek
        r'(?:συντ[ήη]ρηση|maintenance)\s+(?:αρχε[ιί]α|records?|logs?)',
        r'(?:βλ[αά]βες?|failures?|επισκευ[εέ]ς?)\s+(?:an[αά]λυση|pattern)',
        
        # English
        r'(?:maintenance|repair)\s+(?:records?|logs?|history)',
        r'(?:equipment|failure)\s+(?:pattern|trend|analysis)',
    ]
    
    REPORT_PATTERNS = [
        # Greek
        r'(?:δημι[οό]υργησε|φτι[αά]ξε|κ[αά]νε)\s+(?:αναφορ[αά]|report)\s+(?:[εέ]λ[εέ]γχου|audit)',
        r'(?:αναφορ[αά]|report)\s+(?:ευρημ[αά]των|findings|αποκλ[ιί]σεων)',
        
        # English
        r'(?:generate|create|make)\s+(?:an?\s+)?(?:audit|anomaly)\s+report',
        r'(?:summarize|compile)\s+(?:the\s+)?(?:findings|anomalies|issues)',
    ]
    
    SEVERITY_KEYWORDS = {
        'critical': ['κρ[ιί]σιμ', 'critical', 'urgent', 'επε[ιί]γον'],
        'high': ['υψηλ', 'high', 'σοβαρ', 'serious'],
        'medium': ['μεσα[ιί]', 'medium', 'moderate'],
        'low': ['χαμηλ', 'low', 'minor']
    }
    
    def __init__(self):
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        self._anomaly_regex = [
            re.compile(p, re.IGNORECASE) for p in self.ANOMALY_DETECTION_PATTERNS
        ]
        self._inventory_regex = [
            re.compile(p, re.IGNORECASE) for p in self.INVENTORY_CHECK_PATTERNS
        ]
        self._budget_regex = [
            re.compile(p, re.IGNORECASE) for p in self.BUDGET_ANALYSIS_PATTERNS
        ]
        self._maintenance_regex = [
            re.compile(p, re.IGNORECASE) for p in self.MAINTENANCE_PATTERNS
        ]
        self._report_regex = [
            re.compile(p, re.IGNORECASE) for p in self.REPORT_PATTERNS
        ]
    
    def detect_intent(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Detect if query is requesting logistics analysis.
        
        Returns tool info if detected, None otherwise.
        """
        query_lower = query.lower()
        
        # Check for anomaly detection requests
        if self._matches_patterns(query, self._anomaly_regex):
            return self._build_anomaly_params(query)
        
        # Check for inventory checks
        if self._matches_patterns(query, self._inventory_regex):
            return self._build_inventory_params(query)
        
        # Check for budget analysis
        if self._matches_patterns(query, self._budget_regex):
            return self._build_budget_params(query)
        
        # Check for maintenance pattern analysis
        if self._matches_patterns(query, self._maintenance_regex):
            return self._build_maintenance_params(query)
        
        # Check for report generation
        if self._matches_patterns(query, self._report_regex):
            return self._build_report_params(query)
        
        return None
    
    def _matches_patterns(self, query: str, patterns: List[re.Pattern]) -> bool:
        """Check if query matches any pattern."""
        return any(p.search(query) for p in patterns)
    
    def _build_anomaly_params(self, query: str) -> Dict[str, Any]:
        """Build parameters for anomaly detection."""
        return {
            'name': 'detect_logistics_anomalies',
            'params': {
                'generate_report': self._wants_report(query),
                'report_format': self._extract_format(query) or 'md',
                'include_evidence': True
            }
        }
    
    def _build_inventory_params(self, query: str) -> Dict[str, Any]:
        """Build parameters for inventory check."""
        return {
            'name': 'detect_logistics_anomalies',
            'params': {
                'generate_report': self._wants_report(query),
                'report_format': 'md',
                'focus': 'inventory'
            }
        }
    
    def _build_budget_params(self, query: str) -> Dict[str, Any]:
        """Build parameters for budget analysis."""
        return {
            'name': 'detect_logistics_anomalies',
            'params': {
                'generate_report': self._wants_report(query),
                'report_format': 'md',
                'focus': 'budget'
            }
        }
    
    def _build_maintenance_params(self, query: str) -> Dict[str, Any]:
        """Build parameters for maintenance analysis."""
        return {
            'name': 'detect_logistics_anomalies',
            'params': {
                'generate_report': self._wants_report(query),
                'report_format': 'md',
                'focus': 'maintenance'
            }
        }
    
    def _build_report_params(self, query: str) -> Dict[str, Any]:
        """Build parameters for report generation."""
        return {
            'name': 'detect_logistics_anomalies',
            'params': {
                'generate_report': True,
                'report_format': self._extract_format(query) or 'docx',
                'include_evidence': True
            }
        }
    
    def _wants_report(self, query: str) -> bool:
        """Check if user wants a report generated."""
        report_words = [
            'αναφορά', 'report', 'έγγραφο', 'document',
            'δημιούργησε', 'generate', 'create', 'φτιάξε'
        ]
        return any(word in query.lower() for word in report_words)
    
    def _extract_format(self, query: str) -> Optional[str]:
        """Extract desired output format from query."""
        query_lower = query.lower()
        
        if any(w in query_lower for w in ['pdf']):
            return 'pdf'
        elif any(w in query_lower for w in ['markdown', 'md']):
            return 'md'
        elif any(w in query_lower for w in ['word', 'docx']):
            return 'docx'
        elif any(w in query_lower for w in ['excel', 'xlsx']):
            return 'xlsx'
        
        return None


def extend_decision_maker_with_logistics(decision_maker):
    """
    Extend an existing decision maker with logistics patterns.
    
    Call this to add logistics detection to SimpleDecisionMaker.
    """
    detector = LogisticsPatternDetector()
    original_identify = decision_maker._identify_tool
    
    def enhanced_identify(query: str) -> Optional[Dict[str, Any]]:
        # Check logistics patterns first
        logistics_result = detector.detect_intent(query)
        if logistics_result:
            logger.info(f"Detected logistics request: {logistics_result['name']}")
            return logistics_result
        
        # Fall back to original detection
        return original_identify(query)
    
    decision_maker._identify_tool = enhanced_identify
    logger.info("Extended decision maker with logistics patterns")


# Convenience function
_detector: Optional[LogisticsPatternDetector] = None


def detect_logistics_intent(query: str) -> Optional[Dict[str, Any]]:
    """Convenience function for detecting logistics intents."""
    global _detector
    if _detector is None:
        _detector = LogisticsPatternDetector()
    return _detector.detect_intent(query)