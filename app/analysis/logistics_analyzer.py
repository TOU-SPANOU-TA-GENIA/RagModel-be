# app/analysis/logistics_analyzer.py
"""
Logistics anomaly detection and cross-document analysis.

Detects inconsistencies across inventory, requisition, and maintenance documents.
Supports military and general logistics contexts.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import re
import statistics

from app.utils.logger import setup_logger
from app.analysis.content_extractors import ExtractedContent

logger = setup_logger(__name__)


class AnomalySeverity(Enum):
    """Severity levels for detected anomalies."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AnomalyCategory(Enum):
    """Categories of logistics anomalies."""
    INVENTORY_DISCREPANCY = "inventory_discrepancy"
    BUDGET_ANOMALY = "budget_anomaly"
    SUPPLY_CHAIN_GAP = "supply_chain_gap"
    RESOURCE_CONFLICT = "resource_conflict"
    MAINTENANCE_PATTERN = "maintenance_pattern"
    EXPIRATION_WARNING = "expiration_warning"
    USAGE_ANOMALY = "usage_anomaly"
    DOCUMENTATION_MISMATCH = "documentation_mismatch"


@dataclass
class LogisticsEntity:
    """Represents a trackable logistics entity (equipment, supplies, vehicles)."""
    entity_id: str
    entity_type: str  # equipment, supply, vehicle, personnel
    name: str
    unit: Optional[str] = None
    quantity: Optional[float] = None
    status: Optional[str] = None
    location: Optional[str] = None
    last_updated: Optional[datetime] = None
    source_documents: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def add_source(self, source: str):
        if source not in self.source_documents:
            self.source_documents.append(source)


@dataclass
class Anomaly:
    """Detected logistics anomaly."""
    anomaly_id: str
    category: AnomalyCategory
    severity: AnomalySeverity
    title: str
    description: str
    entities_involved: List[str]
    source_documents: List[str]
    evidence: List[str]  # Specific text snippets supporting the finding
    suggested_actions: List[str]
    confidence: float  # 0.0 - 1.0
    detected_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_id": self.anomaly_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "entities_involved": self.entities_involved,
            "source_documents": self.source_documents,
            "evidence": self.evidence,
            "suggested_actions": self.suggested_actions,
            "confidence": self.confidence,
            "detected_at": self.detected_at,
            "metadata": self.metadata
        }


@dataclass
class LogisticsAnalysisResult:
    """Complete result from logistics analysis."""
    documents_analyzed: int
    analysis_timestamp: str
    entities_found: int
    anomalies: List[Anomaly]
    entity_summary: Dict[str, int]  # Counts by entity type
    severity_summary: Dict[str, int]  # Counts by severity
    category_summary: Dict[str, int]  # Counts by category
    cross_references: List[Dict[str, Any]]
    baselines: Dict[str, Any]  # Statistical baselines computed
    recommendations: List[str]
    confidence_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "documents_analyzed": self.documents_analyzed,
            "analysis_timestamp": self.analysis_timestamp,
            "entities_found": self.entities_found,
            "anomalies": [a.to_dict() for a in self.anomalies],
            "entity_summary": self.entity_summary,
            "severity_summary": self.severity_summary,
            "category_summary": self.category_summary,
            "cross_references": self.cross_references,
            "baselines": self.baselines,
            "recommendations": self.recommendations,
            "confidence_score": self.confidence_score
        }


class EntityExtractor:
    """Extracts logistics entities from document content."""
    
    # Patterns for entity identification (Greek and English)
    PATTERNS = {
        'equipment': [
            r'(?:τυφέκια?|rifles?)\s*[:\-]?\s*(\d+)',
            r'(?:όπλα|weapons?)\s*[:\-]?\s*(\d+)',
            r'(?:οχήματα?|vehicles?)\s*[:\-]?\s*(\d+)',
            r'(?:Η/Υ|computers?|υπολογιστ[έή]ς?)\s*[:\-]?\s*(\d+)',
            r'(?:ασύρματοι?|radios?)\s*[:\-]?\s*(\d+)',
        ],
        'supply': [
            r'(?:πυρομαχικ[άα]|ammunition)\s*[:\-]?\s*(\d+(?:[,\.]\d+)?)',
            r'(?:καύσιμα|fuel)\s*[:\-]?\s*(\d+(?:[,\.]\d+)?)\s*(?:lt|λ[ίι]τρα|gallons?)?',
            r'(?:τρόφιμα|rations?|food)\s*[:\-]?\s*(\d+)',
            r'(?:φάρμακα|medical supplies)\s*[:\-]?\s*(\d+)',
        ],
        'vehicle': [
            r'(?:όχημα|vehicle)\s+(?:αρ\.|#|no\.?)?\s*([A-ZΑ-Ω0-9\-]+)',
            r'(?:πινακίδα|plate)\s*[:\-]?\s*([A-ZΑ-Ω]{2,3}[\-\s]?\d{3,5})',
        ],
        'personnel': [
            r'(?:προσωπικό|personnel|στρατιώτες|soldiers)\s*[:\-]?\s*(\d+)',
            r'(?:αξιωματικοί|officers)\s*[:\-]?\s*(\d+)',
        ],
        'budget': [
            r'(?:προϋπολογισμός|budget)\s*[:\-]?\s*[€$]?\s*(\d+(?:[,\.]\d+)?)',
            r'(?:δαπάνη|expense|κόστος|cost)\s*[:\-]?\s*[€$]?\s*(\d+(?:[,\.]\d+)?)',
            r'[€$]\s*(\d{1,3}(?:[,\.]\d{3})*(?:[,\.]\d{2})?)',
        ],
        'date': [
            r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
            r'(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})',
            r'(?:λήξη|expir(?:es?|ation|y))\s*[:\-]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
        ],
        'custom_equipment': [
            r'(?:your_pattern)\s*[:\-]?\s*(\d+)',
        ],
    }
    
    def __init__(self):
        self._compiled = {
            cat: [re.compile(p, re.IGNORECASE) for p in patterns]
            for cat, patterns in self.PATTERNS.items()
        }
    
    def extract(self, text: str, source: str) -> List[LogisticsEntity]:
        """Extract entities from text content."""
        entities = []
        
        for entity_type, patterns in self._compiled.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    value = match.group(1) if match.groups() else match.group()
                    
                    # Get surrounding context for entity name
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    context = text[start:end].strip()
                    
                    entity = LogisticsEntity(
                        entity_id=f"{entity_type}_{value}_{len(entities)}",
                        entity_type=entity_type,
                        name=context[:50],
                        quantity=self._parse_quantity(value),
                        source_documents=[source],
                        attributes={"raw_value": value, "pattern_match": match.group()}
                    )
                    entities.append(entity)
        
        return entities
    
    def _parse_quantity(self, value: str) -> Optional[float]:
        """Parse numeric quantity from string."""
        try:
            # Handle European number format (comma as decimal)
            cleaned = value.replace('.', '').replace(',', '.')
            return float(cleaned)
        except (ValueError, AttributeError):
            return None


class BaselineCalculator:
    """Calculates statistical baselines for anomaly detection."""
    
    def __init__(self):
        self.baselines: Dict[str, Dict[str, float]] = {}
    
    def calculate(self, entities: List[LogisticsEntity]) -> Dict[str, Any]:
        """Calculate baselines from entity data."""
        baselines = {}
        
        # Group entities by type
        by_type = defaultdict(list)
        for entity in entities:
            if entity.quantity is not None:
                by_type[entity.entity_type].append(entity.quantity)
        
        # Calculate statistics per type
        for entity_type, quantities in by_type.items():
            if len(quantities) >= 2:
                baselines[entity_type] = {
                    "mean": statistics.mean(quantities),
                    "median": statistics.median(quantities),
                    "stdev": statistics.stdev(quantities) if len(quantities) > 1 else 0,
                    "min": min(quantities),
                    "max": max(quantities),
                    "count": len(quantities)
                }
            elif quantities:
                baselines[entity_type] = {
                    "mean": quantities[0],
                    "median": quantities[0],
                    "stdev": 0,
                    "min": quantities[0],
                    "max": quantities[0],
                    "count": 1
                }
        
        return baselines
    
    def is_anomalous(
        self, 
        entity_type: str, 
        value: float, 
        threshold_sigma: float = 2.0
    ) -> Tuple[bool, float]:
        """Check if value is anomalous based on baseline."""
        if entity_type not in self.baselines:
            return False, 0.0
        
        baseline = self.baselines[entity_type]
        if baseline["stdev"] == 0:
            return False, 0.0
        
        z_score = abs(value - baseline["mean"]) / baseline["stdev"]
        return z_score > threshold_sigma, z_score


class CrossDocumentMatcher:
    """Matches and correlates entities across documents."""
    
    def __init__(self, fuzzy_threshold: float = 0.8):
        self.fuzzy_threshold = fuzzy_threshold
    
    def match_entities(
        self, 
        entities: List[LogisticsEntity]
    ) -> List[Tuple[LogisticsEntity, LogisticsEntity, float]]:
        """Find matching entities across different documents."""
        matches = []
        
        # Group by type first
        by_type = defaultdict(list)
        for entity in entities:
            by_type[entity.entity_type].append(entity)
        
        # Match within each type
        for entity_type, type_entities in by_type.items():
            for i, e1 in enumerate(type_entities):
                for e2 in type_entities[i+1:]:
                    # Skip if same document
                    if set(e1.source_documents) & set(e2.source_documents):
                        continue
                    
                    similarity = self._calculate_similarity(e1, e2)
                    if similarity >= self.fuzzy_threshold:
                        matches.append((e1, e2, similarity))
        
        return matches
    
    def _calculate_similarity(self, e1: LogisticsEntity, e2: LogisticsEntity) -> float:
        """Calculate similarity between two entities."""
        score = 0.0
        factors = 0
        
        # Type match (required)
        if e1.entity_type != e2.entity_type:
            return 0.0
        
        # ID similarity
        if e1.entity_id and e2.entity_id:
            if e1.entity_id.lower() == e2.entity_id.lower():
                score += 1.0
            else:
                score += self._string_similarity(e1.entity_id, e2.entity_id)
            factors += 1
        
        # Name similarity
        if e1.name and e2.name:
            score += self._string_similarity(e1.name, e2.name)
            factors += 1
        
        # Location match
        if e1.location and e2.location:
            if e1.location.lower() == e2.location.lower():
                score += 1.0
            factors += 1
        
        return score / factors if factors > 0 else 0.0
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Simple string similarity (Jaccard on character n-grams)."""
        def ngrams(s: str, n: int = 3) -> Set[str]:
            s = s.lower()
            return set(s[i:i+n] for i in range(len(s) - n + 1))
        
        ng1, ng2 = ngrams(s1), ngrams(s2)
        if not ng1 or not ng2:
            return 0.0
        
        intersection = len(ng1 & ng2)
        union = len(ng1 | ng2)
        return intersection / union if union > 0 else 0.0


class AnomalyDetector:
    """Detects various types of logistics anomalies."""
    
    def __init__(self, baseline_calculator: BaselineCalculator):
        self.baseline = baseline_calculator
        self.anomaly_counter = 0
    
    def detect_inventory_discrepancies(
        self, 
        entities: List[LogisticsEntity],
        matches: List[Tuple[LogisticsEntity, LogisticsEntity, float]]
    ) -> List[Anomaly]:
        """Detect inventory count discrepancies across documents."""
        anomalies = []
        
        for e1, e2, similarity in matches:
            if e1.quantity is not None and e2.quantity is not None:
                if e1.quantity != e2.quantity:
                    diff = abs(e1.quantity - e2.quantity)
                    diff_pct = (diff / max(e1.quantity, e2.quantity)) * 100
                    
                    # Determine severity based on discrepancy magnitude
                    if diff_pct > 20:
                        severity = AnomalySeverity.CRITICAL
                    elif diff_pct > 10:
                        severity = AnomalySeverity.HIGH
                    elif diff_pct > 5:
                        severity = AnomalySeverity.MEDIUM
                    else:
                        severity = AnomalySeverity.LOW
                    
                    anomaly = Anomaly(
                        anomaly_id=self._next_id(),
                        category=AnomalyCategory.INVENTORY_DISCREPANCY,
                        severity=severity,
                        title=f"Ασυμφωνία αποθέματος: {e1.name[:30]}",
                        description=(
                            f"Εντοπίστηκε διαφορά {diff:.0f} μονάδων ({diff_pct:.1f}%) "
                            f"μεταξύ εγγράφων. Έγγραφο 1: {e1.quantity:.0f}, "
                            f"Έγγραφο 2: {e2.quantity:.0f}"
                        ),
                        entities_involved=[e1.entity_id, e2.entity_id],
                        source_documents=e1.source_documents + e2.source_documents,
                        evidence=[
                            f"'{e1.attributes.get('pattern_match', '')}' από {e1.source_documents[0]}",
                            f"'{e2.attributes.get('pattern_match', '')}' από {e2.source_documents[0]}"
                        ],
                        suggested_actions=[
                            "Διεξαγωγή φυσικής καταμέτρησης",
                            "Έλεγχος αρχείων μετακινήσεων",
                            "Επαλήθευση ημερομηνιών εγγράφων"
                        ],
                        confidence=similarity,
                        metadata={"difference": diff, "difference_pct": diff_pct}
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    def detect_budget_anomalies(
        self, 
        entities: List[LogisticsEntity]
    ) -> List[Anomaly]:
        """Detect unusual budget/expense patterns."""
        anomalies = []
        
        budget_entities = [e for e in entities if e.entity_type == 'budget']
        
        if len(budget_entities) < 2:
            return anomalies
        
        # Calculate baseline
        values = [e.quantity for e in budget_entities if e.quantity]
        if not values:
            return anomalies
        
        mean_val = statistics.mean(values)
        
        for entity in budget_entities:
            if entity.quantity and mean_val > 0:
                ratio = entity.quantity / mean_val
                
                # Flag if 200%+ above average
                if ratio > 3.0:
                    anomaly = Anomaly(
                        anomaly_id=self._next_id(),
                        category=AnomalyCategory.BUDGET_ANOMALY,
                        severity=AnomalySeverity.HIGH,
                        title=f"Ασυνήθιστη δαπάνη: {ratio:.0%} του μέσου όρου",
                        description=(
                            f"Η τιμή {entity.quantity:,.2f} είναι {ratio:.1f}x "
                            f"μεγαλύτερη από τον μέσο όρο ({mean_val:,.2f})"
                        ),
                        entities_involved=[entity.entity_id],
                        source_documents=entity.source_documents,
                        evidence=[entity.attributes.get('pattern_match', '')],
                        suggested_actions=[
                            "Επαλήθευση ορθότητας καταχώρησης",
                            "Σύγκριση με ιστορικά δεδομένα",
                            "Έλεγχος αιτιολόγησης δαπάνης"
                        ],
                        confidence=0.85,
                        metadata={"ratio_to_mean": ratio, "mean_value": mean_val}
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    def detect_resource_conflicts(
        self, 
        entities: List[LogisticsEntity]
    ) -> List[Anomaly]:
        """Detect resources assigned to multiple units/locations."""
        anomalies = []
        
        # Group by entity ID
        by_id = defaultdict(list)
        for entity in entities:
            if entity.entity_id:
                by_id[entity.entity_id].append(entity)
        
        for entity_id, occurrences in by_id.items():
            if len(occurrences) < 2:
                continue
            
            # Check for conflicting assignments
            locations = set()
            units = set()
            for occ in occurrences:
                if occ.location:
                    locations.add(occ.location)
                if occ.unit:
                    units.add(occ.unit)
            
            if len(locations) > 1 or len(units) > 1:
                anomaly = Anomaly(
                    anomaly_id=self._next_id(),
                    category=AnomalyCategory.RESOURCE_CONFLICT,
                    severity=AnomalySeverity.HIGH,
                    title=f"Διπλή ανάθεση πόρου: {entity_id}",
                    description=(
                        f"Ο πόρος {entity_id} εμφανίζεται σε πολλαπλές τοποθεσίες/μονάδες. "
                        f"Τοποθεσίες: {', '.join(locations) if locations else 'N/A'}. "
                        f"Μονάδες: {', '.join(units) if units else 'N/A'}"
                    ),
                    entities_involved=[entity_id],
                    source_documents=list(set(
                        doc for occ in occurrences for doc in occ.source_documents
                    )),
                    evidence=[occ.name for occ in occurrences[:3]],
                    suggested_actions=[
                        "Επιβεβαίωση φυσικής θέσης πόρου",
                        "Ενημέρωση αρχείων ανάθεσης",
                        "Έλεγχος για σφάλματα καταχώρησης"
                    ],
                    confidence=0.9,
                    metadata={
                        "locations": list(locations),
                        "units": list(units),
                        "occurrence_count": len(occurrences)
                    }
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def detect_expiration_warnings(
        self, 
        text_content: str,
        source: str
    ) -> List[Anomaly]:
        """Detect approaching expiration dates."""
        anomalies = []
        
        # Pattern for expiration dates
        expiry_patterns = [
            r'(?:λήξη|expir(?:es?|ation|y))[:\s]+(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
            r'(?:ισχύς\s+(?:έως|μέχρι)|valid\s+until)[:\s]+(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
        ]
        
        today = datetime.now()
        warning_threshold = today + timedelta(days=90)
        
        for pattern in expiry_patterns:
            for match in re.finditer(pattern, text_content, re.IGNORECASE):
                date_str = match.group(1)
                try:
                    # Try parsing common date formats
                    for fmt in ['%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d', '%d.%m.%Y']:
                        try:
                            exp_date = datetime.strptime(date_str, fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        continue
                    
                    if exp_date <= warning_threshold:
                        days_until = (exp_date - today).days
                        
                        if days_until < 0:
                            severity = AnomalySeverity.CRITICAL
                            status = "ΛΗΓΜΕΝΟ"
                        elif days_until < 30:
                            severity = AnomalySeverity.HIGH
                            status = f"λήγει σε {days_until} ημέρες"
                        else:
                            severity = AnomalySeverity.MEDIUM
                            status = f"λήγει σε {days_until} ημέρες"
                        
                        # Get context
                        start = max(0, match.start() - 50)
                        end = min(len(text_content), match.end() + 50)
                        context = text_content[start:end]
                        
                        anomaly = Anomaly(
                            anomaly_id=self._next_id(),
                            category=AnomalyCategory.EXPIRATION_WARNING,
                            severity=severity,
                            title=f"Προειδοποίηση λήξης: {status}",
                            description=f"Ημερομηνία λήξης {date_str} - {status}",
                            entities_involved=[],
                            source_documents=[source],
                            evidence=[context.strip()],
                            suggested_actions=[
                                "Έλεγχος εναλλακτικών προμηθειών",
                                "Δρομολόγηση παραγγελίας αντικατάστασης",
                                "Ενημέρωση αρμόδιου τμήματος"
                            ],
                            confidence=0.95,
                            metadata={
                                "expiration_date": exp_date.isoformat(),
                                "days_until_expiration": days_until
                            }
                        )
                        anomalies.append(anomaly)
                        
                except Exception as e:
                    logger.debug(f"Date parsing failed for {date_str}: {e}")
        
        return anomalies
    
    def detect_custom_anomaly(self, entities: List[LogisticsEntity]) -> List[Anomaly]:
        anomalies = []
        # Your detection logic here
        return anomalies
    
    def _next_id(self) -> str:
        """Generate unique anomaly ID."""
        self.anomaly_counter += 1
        return f"ANOM-{datetime.now().strftime('%Y%m%d')}-{self.anomaly_counter:04d}"


class LogisticsAnalyzer:
    """
    Main analyzer for logistics anomaly detection.
    
    Orchestrates entity extraction, baseline calculation, and anomaly detection.
    """
    
    def __init__(self, llm_provider=None):
        self.entity_extractor = EntityExtractor()
        self.baseline_calculator = BaselineCalculator()
        self.cross_matcher = CrossDocumentMatcher()
        self.llm = llm_provider
    
    def analyze(self, documents: List[ExtractedContent]) -> LogisticsAnalysisResult:
        """
        Perform comprehensive logistics analysis.
        
        Args:
            documents: List of extracted document contents
            
        Returns:
            LogisticsAnalysisResult with all findings
        """
        logger.info(f"Starting logistics analysis on {len(documents)} documents")
        
        # Step 1: Extract entities from all documents
        all_entities = []
        for doc in documents:
            entities = self.entity_extractor.extract(doc.text_content, doc.source_name)
            all_entities.extend(entities)
            
            # Also extract from tables if present
            for table in doc.tables:
                table_text = self._table_to_text(table)
                entities = self.entity_extractor.extract(table_text, doc.source_name)
                all_entities.extend(entities)
        
        logger.info(f"Extracted {len(all_entities)} entities")
        
        # Step 2: Calculate baselines
        baselines = self.baseline_calculator.calculate(all_entities)
        self.baseline_calculator.baselines = baselines
        
        # Step 3: Find cross-document matches
        matches = self.cross_matcher.match_entities(all_entities)
        logger.info(f"Found {len(matches)} cross-document matches")
        
        # Step 4: Detect anomalies
        detector = AnomalyDetector(self.baseline_calculator)
        anomalies = []
        
        # Inventory discrepancies
        anomalies.extend(
            detector.detect_inventory_discrepancies(all_entities, matches)
        )
        
        # Budget anomalies
        anomalies.extend(
            detector.detect_budget_anomalies(all_entities)
        )
        
        # Resource conflicts
        anomalies.extend(
            detector.detect_resource_conflicts(all_entities)
        )
        
        # Expiration warnings
        for doc in documents:
            anomalies.extend(
                detector.detect_expiration_warnings(doc.text_content, doc.source_name)
            )
        
        logger.info(f"Detected {len(anomalies)} anomalies")
        
        # Step 5: Build summaries
        entity_summary = defaultdict(int)
        for entity in all_entities:
            entity_summary[entity.entity_type] += 1
        
        severity_summary = defaultdict(int)
        category_summary = defaultdict(int)
        for anomaly in anomalies:
            severity_summary[anomaly.severity.value] += 1
            category_summary[anomaly.category.value] += 1
        
        # Step 6: Generate recommendations
        recommendations = self._generate_recommendations(anomalies)
        
        # Step 7: Build cross-reference summary
        cross_refs = [
            {
                "doc1": m[0].source_documents[0] if m[0].source_documents else "unknown",
                "doc2": m[1].source_documents[0] if m[1].source_documents else "unknown",
                "entity_type": m[0].entity_type,
                "similarity": m[2]
            }
            for m in matches
        ]
        
        # Calculate confidence
        confidence = self._calculate_confidence(all_entities, anomalies, matches)
        
        return LogisticsAnalysisResult(
            documents_analyzed=len(documents),
            analysis_timestamp=datetime.now().isoformat(),
            entities_found=len(all_entities),
            anomalies=sorted(anomalies, key=lambda a: (
                ['critical', 'high', 'medium', 'low', 'info'].index(a.severity.value)
            )),
            entity_summary=dict(entity_summary),
            severity_summary=dict(severity_summary),
            category_summary=dict(category_summary),
            cross_references=cross_refs,
            baselines=baselines,
            recommendations=recommendations,
            confidence_score=confidence
        )
    
    def _table_to_text(self, table: Dict[str, Any]) -> str:
        """Convert table data to searchable text."""
        rows = table.get("rows", [])
        return "\n".join(" | ".join(str(cell) for cell in row) for row in rows)
    
    def _generate_recommendations(self, anomalies: List[Anomaly]) -> List[str]:
        """Generate prioritized recommendations based on anomalies."""
        recommendations = []
        
        # Count by category
        by_category = defaultdict(list)
        for anomaly in anomalies:
            by_category[anomaly.category].append(anomaly)
        
        # Critical items first
        critical = [a for a in anomalies if a.severity == AnomalySeverity.CRITICAL]
        if critical:
            recommendations.append(
                f"ΕΠΕΙΓΟΝ: {len(critical)} κρίσιμα ευρήματα απαιτούν άμεση δράση"
            )
        
        # Category-specific recommendations
        if AnomalyCategory.INVENTORY_DISCREPANCY in by_category:
            count = len(by_category[AnomalyCategory.INVENTORY_DISCREPANCY])
            recommendations.append(
                f"Συνιστάται φυσική καταμέτρηση για {count} είδη με ασυμφωνίες"
            )
        
        if AnomalyCategory.RESOURCE_CONFLICT in by_category:
            recommendations.append(
                "Έλεγχος διπλών αναθέσεων πόρων σε πολλαπλές μονάδες"
            )
        
        if AnomalyCategory.EXPIRATION_WARNING in by_category:
            expiring = len(by_category[AnomalyCategory.EXPIRATION_WARNING])
            recommendations.append(
                f"Προγραμματισμός αντικατάστασης για {expiring} είδη που λήγουν"
            )
        
        if AnomalyCategory.BUDGET_ANOMALY in by_category:
            recommendations.append(
                "Επανεξέταση προϋπολογισμού - εντοπίστηκαν ασυνήθιστες δαπάνες"
            )
        
        return recommendations
    
    def _calculate_confidence(
        self,
        entities: List[LogisticsEntity],
        anomalies: List[Anomaly],
        matches: List[Tuple]
    ) -> float:
        """Calculate overall analysis confidence."""
        if not entities:
            return 0.0
        
        # Base confidence from data quality
        base = 0.5
        
        # Bonus for cross-document corroboration
        if matches:
            base += min(len(matches) * 0.02, 0.2)
        
        # Bonus for multiple entity types found
        entity_types = set(e.entity_type for e in entities)
        base += min(len(entity_types) * 0.05, 0.2)
        
        # Average anomaly confidence
        if anomalies:
            avg_conf = sum(a.confidence for a in anomalies) / len(anomalies)
            base = (base + avg_conf) / 2
        
        return min(base, 1.0)


# Convenience function
def analyze_logistics(
    documents: List[ExtractedContent], 
    llm_provider=None
) -> LogisticsAnalysisResult:
    """Convenience function for logistics analysis."""
    analyzer = LogisticsAnalyzer(llm_provider)
    return analyzer.analyze(documents)