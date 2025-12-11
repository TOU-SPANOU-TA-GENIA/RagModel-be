# app/tools/logistics_anomaly_tool.py
"""
Logistics Anomaly Detection Tool - Enhanced Version.
Detects inventory discrepancies, budget overruns, expiration issues, and more.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from app.tools.models import BaseTool, ToolResult
from app.utils.logger import setup_logger
from app.config import BASE_DIR

logger = setup_logger(__name__)


class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Category(Enum):
    INVENTORY = "inventory"
    BUDGET = "budget"
    EXPIRATION = "expiration"
    MAINTENANCE = "maintenance"
    CONSUMPTION = "consumption"
    DISCREPANCY = "discrepancy"


@dataclass
class Anomaly:
    anomaly_id: str
    severity: Severity
    category: Category
    title: str
    description: str
    evidence: List[str] = field(default_factory=list)
    source_documents: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)
    values: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    documents_analyzed: int
    anomalies: List[Anomaly]
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))
    
    @property
    def severity_summary(self) -> Dict[str, int]:
        summary = {}
        for a in self.anomalies:
            key = a.severity.value
            summary[key] = summary.get(key, 0) + 1
        return summary


class SimpleLogisticsAnalyzer:
    """
    Analyzes logistics documents for anomalies.
    Works with plain text files containing inventory, maintenance, and budget data.
    """
    
    def __init__(self):
        self.anomalies: List[Anomaly] = []
        self.anomaly_counter = 0
        self.extracted_data: Dict[str, Dict] = {}
    
    def analyze(self, file_paths: List[Path]) -> AnalysisResult:
        """Analyze files and detect anomalies."""
        self.anomalies = []
        self.anomaly_counter = 0
        self.extracted_data = {}
        
        # Read and parse all files
        documents = {}
        for path in file_paths:
            try:
                content = path.read_text(encoding='utf-8')
                documents[path.name] = content
                self.extracted_data[path.name] = self._extract_data(content, path.name)
            except Exception as e:
                logger.error(f"Failed to read {path}: {e}")
        
        if not documents:
            return AnalysisResult(documents_analyzed=0, anomalies=[])
        
        # Run detection algorithms
        self._detect_quantity_changes(documents)
        self._detect_expiration_issues(documents)
        self._detect_budget_anomalies(documents)
        self._detect_consumption_anomalies(documents)
        self._detect_cross_document_discrepancies(documents)
        
        return AnalysisResult(
            documents_analyzed=len(documents),
            anomalies=self.anomalies
        )
    
    def _extract_data(self, content: str, source: str) -> Dict:
        """Extract structured data from document content."""
        data = {
            "source": source,
            "quantities": {},
            "dates": [],
            "monetary": {},
            "status": {}
        }
        
        # Extract quantities (Greek and English patterns)
        quantity_patterns = [
            r'([Α-Ωα-ωά-ώ\w\s]+?):\s*(\d+(?:[\.,]\d+)?)\s*(?:τεμ|τεμάχια|μονάδ|λίτρα|φιάλες|κιτ|μερίδες)',
            r'([Α-Ωα-ωά-ώ\w\s]+?)\s*[-–:]\s*(\d+(?:[\.,]\d+)?)\s*(?:τεμ|τεμάχια|μονάδ|λίτρα|φιάλες|κιτ|μερίδες)',
            r'(\d+(?:[\.,]\d+)?)\s*(?:τεμ|τεμάχια|μονάδ|λίτρα|φιάλες|κιτ|μερίδες)\s+([Α-Ωα-ωά-ώ\w\s]+)',
        ]
        
        for pattern in quantity_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                groups = match.groups()
                if groups[0].replace('.', '').replace(',', '').isdigit():
                    name = groups[1].strip()
                    qty = float(groups[0].replace('.', '').replace(',', '.'))
                else:
                    name = groups[0].strip()
                    qty = float(groups[1].replace('.', '').replace(',', '.'))
                
                name = name.strip('- :').lower()
                if len(name) > 2:
                    data["quantities"][name] = qty
        
        # Extract specific items by name
        specific_items = [
            (r'τυφέκι[αά]\s*(?:G3)?\s*[-:]\s*(\d+)', 'τυφέκια'),
            (r'πιστόλι[αά]\s*[-:]\s*(\d+)', 'πιστόλια'),
            (r'φυσίγγια?\s*(?:7\.?62\s*mm)?\s*[-:]\s*(\d+(?:[\.,]\d+)?)', 'φυσίγγια 7.62mm'),
            (r'φυσίγγια?\s*(?:9\s*mm)?\s*[-:]\s*(\d+(?:[\.,]\d+)?)', 'φυσίγγια 9mm'),
            (r'νερό[ύ]?\s*(?:εμφιαλωμένο)?\s*[-:]\s*(\d+)', 'νερό'),
            (r'φαρμακευτικό?\s*(?:υλικό)?\s*[-:]\s*(\d+)', 'φάρμακα'),
            (r'τρόφιμα?\s*[-:]\s*(\d+)', 'τρόφιμα'),
            (r'καύσιμ[αά]\s*[-:]\s*(\d+(?:[\.,]\d+)?)', 'καύσιμα'),
            (r'(\d+(?:[\.,]\d+)?)\s*λίτρα', 'καύσιμα'),
        ]
        
        for pattern, name in specific_items:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                qty = float(match.group(1).replace('.', '').replace(',', '.'))
                data["quantities"][name] = qty
        
        # Extract dates (DD/MM/YYYY format)
        date_pattern = r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})'
        for match in re.finditer(date_pattern, content):
            try:
                day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
                date = datetime(year, month, day)
                
                # Find context around date
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 20)
                context = content[start:end].strip()
                
                data["dates"].append({
                    "date": date,
                    "date_str": match.group(0),
                    "context": context
                })
            except ValueError:
                pass
        
        # Extract monetary values
        money_patterns = [
            r'([Α-Ωα-ωά-ώ\w\s]+?):\s*(\d+(?:[\.,]\d+)?)\s*€',
            r'(\d+(?:[\.,]\d+)?)\s*€\s*[-–]\s*([Α-Ωα-ωά-ώ\w\s]+)',
            r'δαπάν[εέη]ς?\s*(?:μήνα)?\s*[-:]\s*(\d+(?:[\.,]\d+)?)',
            r'διαθέσιμο\s*[-:]\s*(\d+(?:[\.,]\d+)?)',
            r'κόστος\s*[-:]\s*(\d+(?:[\.,]\d+)?)',
            r'σύνολο\s*[-:]\s*(\d+(?:[\.,]\d+)?)',
        ]
        
        for pattern in money_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                groups = match.groups()
                if len(groups) == 2:
                    if groups[0].replace('.', '').replace(',', '').isdigit():
                        name = groups[1].strip().lower()
                        value = float(groups[0].replace('.', '').replace(',', '.'))
                    else:
                        name = groups[0].strip().lower()
                        value = float(groups[1].replace('.', '').replace(',', '.'))
                else:
                    name = "δαπάνες"
                    value = float(groups[0].replace('.', '').replace(',', '.'))
                
                data["monetary"][name] = value
        
        # Extract vehicle/equipment status
        status_patterns = [
            r'([Α-Ωα-ωά-ώ\w\-]+)\s*[-:]\s*(ενεργό|εκτός λειτουργίας|βλάβη|ακινητοποιημένο)',
        ]
        
        for pattern in status_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                item = match.group(1).strip()
                status = match.group(2).lower()
                data["status"][item] = status
        
        return data
    
    def _add_anomaly(self, severity: Severity, category: Category, title: str, 
                     description: str, evidence: List[str] = None, 
                     sources: List[str] = None, actions: List[str] = None,
                     values: Dict = None):
        """Add an anomaly to the list."""
        self.anomaly_counter += 1
        self.anomalies.append(Anomaly(
            anomaly_id=f"ANM-{self.anomaly_counter:03d}",
            severity=severity,
            category=category,
            title=title,
            description=description,
            evidence=evidence or [],
            source_documents=sources or [],
            suggested_actions=actions or [],
            values=values or {}
        ))
    
    def _detect_quantity_changes(self, documents: Dict[str, str]):
        """Detect significant quantity changes between documents."""
        # Group documents by type (e.g., απογραφή)
        inventory_docs = {k: v for k, v in self.extracted_data.items() 
                        if 'απογραφ' in k.lower()}
        
        if len(inventory_docs) < 2:
            return
        
        # Sort by date in filename (simple heuristic)
        sorted_docs = sorted(inventory_docs.items())
        
        if len(sorted_docs) >= 2:
            earlier_name, earlier_data = sorted_docs[0]
            later_name, later_data = sorted_docs[-1]
            
            earlier_qty = earlier_data.get("quantities", {})
            later_qty = later_data.get("quantities", {})
            
            # Compare quantities
            for item, old_val in earlier_qty.items():
                if item in later_qty:
                    new_val = later_qty[item]
                    diff = new_val - old_val
                    
                    if old_val > 0:
                        pct_change = (diff / old_val) * 100
                    else:
                        pct_change = 0
                    
                    # Detect significant changes
                    if diff != 0:
                        # Weapons shortage is critical
                        if 'τυφέκ' in item or 'πιστόλ' in item or 'όπλ' in item:
                            if diff < 0:
                                self._add_anomaly(
                                    Severity.CRITICAL,
                                    Category.INVENTORY,
                                    f"Έλλειμμα οπλισμού: {item}",
                                    f"Μείωση {abs(int(diff))} τεμαχίων ({int(old_val)} → {int(new_val)}). "
                                    f"Απαιτείται άμεση διερεύνηση.",
                                    evidence=[
                                        f"{earlier_name}: {int(old_val)} τεμάχια",
                                        f"{later_name}: {int(new_val)} τεμάχια"
                                    ],
                                    sources=[earlier_name, later_name],
                                    actions=[
                                        "Έλεγχος αρχείου μεταφορών",
                                        "Επαλήθευση φυσικής απογραφής",
                                        "Αναφορά στον διοικητή"
                                    ],
                                    values={"old": old_val, "new": new_val, "diff": diff}
                                )
                        
                        # Ammunition - high priority
                        elif 'φυσίγγ' in item or 'πυρομαχ' in item:
                            if diff < 0 and abs(pct_change) > 10:
                                self._add_anomaly(
                                    Severity.HIGH,
                                    Category.INVENTORY,
                                    f"Μείωση πυρομαχικών: {item}",
                                    f"Μείωση {abs(int(diff))} ({pct_change:+.1f}%). "
                                    f"Ελέγξτε αν υπάρχει καταγεγραμμένη χρήση.",
                                    evidence=[
                                        f"{earlier_name}: {int(old_val):,}",
                                        f"{later_name}: {int(new_val):,}",
                                        f"Διαφορά: {int(diff):,} ({pct_change:+.1f}%)"
                                    ],
                                    sources=[earlier_name, later_name],
                                    actions=[
                                        "Έλεγχος δελτίων εκγύμνασης",
                                        "Επαλήθευση αναλώσεων"
                                    ],
                                    values={"old": old_val, "new": new_val, "diff": diff, "pct": pct_change}
                                )
                        
                        # General inventory with large change
                        elif abs(pct_change) > 20:
                            severity = Severity.HIGH if abs(pct_change) > 50 else Severity.MEDIUM
                            self._add_anomaly(
                                severity,
                                Category.INVENTORY,
                                f"Σημαντική μεταβολή: {item}",
                                f"Μεταβολή {pct_change:+.1f}% ({int(old_val)} → {int(new_val)})",
                                evidence=[
                                    f"{earlier_name}: {int(old_val)}",
                                    f"{later_name}: {int(new_val)}"
                                ],
                                sources=[earlier_name, later_name],
                                values={"old": old_val, "new": new_val, "pct": pct_change}
                            )
    
    def _detect_expiration_issues(self, documents: Dict[str, str]):
        """Detect items near or past expiration."""
        today = datetime.now()
        warning_threshold = today + timedelta(days=30)
        
        for doc_name, data in self.extracted_data.items():
            for date_info in data.get("dates", []):
                exp_date = date_info["date"]
                context = date_info["context"].lower()
                
                # Check if this is an expiration date
                if any(word in context for word in ['λήξη', 'λήγ', 'ημερομηνία', 'expir']):
                    if exp_date < today:
                        # Already expired
                        days_expired = (today - exp_date).days
                        self._add_anomaly(
                            Severity.CRITICAL,
                            Category.EXPIRATION,
                            f"Ληγμένο υλικό",
                            f"Ημερομηνία λήξης {date_info['date_str']} έχει παρέλθει "
                            f"({days_expired} ημέρες). Απαιτείται άμεση απόσυρση.",
                            evidence=[f"Πηγή: {context[:80]}..."],
                            sources=[doc_name],
                            actions=[
                                "Απόσυρση ληγμένου υλικού",
                                "Αίτηση αντικατάστασης",
                                "Ενημέρωση προϊσταμένου"
                            ]
                        )
                    elif exp_date < warning_threshold:
                        # Expiring soon
                        days_left = (exp_date - today).days
                        self._add_anomaly(
                            Severity.HIGH,
                            Category.EXPIRATION,
                            f"Υλικό κοντά σε λήξη",
                            f"Λήξη σε {days_left} ημέρες ({date_info['date_str']}). "
                            f"Προγραμματίστε αντικατάσταση.",
                            evidence=[f"Πηγή: {context[:80]}..."],
                            sources=[doc_name],
                            actions=[
                                "Προτεραιοποίηση χρήσης",
                                "Αίτηση αντικατάστασης"
                            ]
                        )
    
    def _detect_budget_anomalies(self, documents: Dict[str, str]):
        """Detect budget overruns and unusual spending patterns."""
        # Collect monetary data
        budget_data = []
        for doc_name, data in self.extracted_data.items():
            monetary = data.get("monetary", {})
            if monetary:
                budget_data.append((doc_name, monetary))
        
        if len(budget_data) < 2:
            return
        
        # Compare spending between periods
        sorted_data = sorted(budget_data)
        earlier_name, earlier_money = sorted_data[0]
        later_name, later_money = sorted_data[-1]
        
        # Find spending/expenses
        earlier_spending = None
        later_spending = None
        
        for key in ['δαπάνες', 'δαπάνες μήνα', 'κόστος', 'σύνολο']:
            if key in earlier_money:
                earlier_spending = earlier_money[key]
            if key in later_money:
                later_spending = later_money[key]
        
        if earlier_spending and later_spending:
            diff = later_spending - earlier_spending
            if earlier_spending > 0:
                pct_change = (diff / earlier_spending) * 100
                
                if pct_change > 50:
                    self._add_anomaly(
                        Severity.HIGH,
                        Category.BUDGET,
                        "Υπέρβαση δαπανών",
                        f"Αύξηση δαπανών {pct_change:.0f}% ({int(earlier_spending):,}€ → {int(later_spending):,}€). "
                        f"Διερευνήστε την αιτία.",
                        evidence=[
                            f"{earlier_name}: {int(earlier_spending):,}€",
                            f"{later_name}: {int(later_spending):,}€",
                            f"Διαφορά: +{int(diff):,}€ (+{pct_change:.0f}%)"
                        ],
                        sources=[earlier_name, later_name],
                        actions=[
                            "Ανάλυση κατανομής δαπανών",
                            "Σύγκριση με προϋπολογισμό",
                            "Αναφορά στο οικονομικό"
                        ],
                        values={"old": earlier_spending, "new": later_spending, "pct": pct_change}
                    )
    
    def _detect_consumption_anomalies(self, documents: Dict[str, str]):
        """Detect unusual consumption patterns (fuel, supplies)."""
        # Look for consumption data in maintenance logs
        maintenance_docs = {k: v for k, v in self.extracted_data.items() 
                          if 'συντήρ' in k.lower() or 'maintenance' in k.lower()}
        
        for doc_name, data in maintenance_docs.items():
            quantities = data.get("quantities", {})
            status = data.get("status", {})
            
            # Count inactive vehicles
            inactive_vehicles = sum(1 for s in status.values() 
                                   if 'εκτός' in s or 'βλάβη' in s or 'ακινητ' in s)
            
            # Check fuel consumption
            if 'καύσιμα' in quantities and inactive_vehicles > 0:
                fuel = quantities['καύσιμα']
                
                # High fuel with inactive vehicles is suspicious
                if fuel > 1500:  # threshold
                    self._add_anomaly(
                        Severity.HIGH,
                        Category.CONSUMPTION,
                        "Ασυνήθιστη κατανάλωση καυσίμων",
                        f"Υψηλή κατανάλωση ({int(fuel)} λίτρα) ενώ {inactive_vehicles} "
                        f"όχημα(τα) είναι εκτός λειτουργίας.",
                        evidence=[
                            f"Κατανάλωση: {int(fuel)} λίτρα",
                            f"Οχήματα εκτός: {inactive_vehicles}"
                        ],
                        sources=[doc_name],
                        actions=[
                            "Έλεγχος δελτίων κίνησης",
                            "Επαλήθευση χιλιομετρικών",
                            "Έλεγχος για διαρροές"
                        ]
                    )
    
    def _detect_cross_document_discrepancies(self, documents: Dict[str, str]):
        """Detect inconsistencies between related documents."""
        inventory_docs = {k: v for k, v in self.extracted_data.items() 
                        if 'απογραφ' in k.lower()}
        maintenance_docs = {k: v for k, v in self.extracted_data.items() 
                          if 'συντήρ' in k.lower()}
        
        # Check maintenance vs inventory consistency
        for maint_name, maint_data in maintenance_docs.items():
            maint_content = documents.get(maint_name, "").lower()
            
            # Look for maintenance counts
            maint_patterns = [
                r'συντήρηση\s+(\d+)\s+τυφεκ',
                r'(\d+)\s+τυφεκ[ιί][αά]?\s*[-–]?\s*(?:συντήρηση|ολοκληρ)',
            ]
            
            for pattern in maint_patterns:
                match = re.search(pattern, maint_content)
                if match:
                    maintained_count = int(match.group(1))
                    
                    # Compare with inventory
                    for inv_name, inv_data in inventory_docs.items():
                        inv_qty = inv_data.get("quantities", {})
                        
                        for key in ['τυφέκια', 'τυφέκια g3']:
                            if key in inv_qty:
                                actual_count = int(inv_qty[key])
                                
                                if maintained_count > actual_count:
                                    self._add_anomaly(
                                        Severity.MEDIUM,
                                        Category.DISCREPANCY,
                                        "Ασυμφωνία συντήρησης-απογραφής",
                                        f"Αναφέρεται συντήρηση {maintained_count} τυφεκίων "
                                        f"αλλά η απογραφή δείχνει {actual_count}.",
                                        evidence=[
                                            f"{maint_name}: συντήρηση {maintained_count} τυφεκίων",
                                            f"{inv_name}: {actual_count} τυφέκια στην απογραφή"
                                        ],
                                        sources=[maint_name, inv_name],
                                        actions=[
                                            "Επαλήθευση αριθμών",
                                            "Έλεγχος για λάθη καταχώρησης"
                                        ]
                                    )


class LogisticsAnomalyTool(BaseTool):
    """
    Tool for detecting anomalies across logistics documents.
    """
    
    def __init__(self, output_dir: Path = None, llm_provider=None):
        super().__init__()
        self.output_dir = output_dir or (BASE_DIR / "outputs" / "audit_reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.llm_provider = llm_provider
    
    @property
    def name(self) -> str:
        return "detect_logistics_anomalies"
    
    @property
    def description(self) -> str:
        return """Detect anomalies and inconsistencies across logistics documents.
        
        Analyzes:
        - Inventory spreadsheets for count discrepancies
        - Requisition forms for budget anomalies
        - Maintenance logs for pattern irregularities
        - Resource assignments for conflicts
        - Expiration dates for supply chain gaps"""
    
    def _execute_impl(
        self,
        file_paths: List[str],
        generate_report: bool = True,
        report_format: str = "md",
        **kwargs
    ) -> ToolResult:
        """Execute logistics anomaly detection."""
        if not file_paths:
            return ToolResult(
                success=False,
                error="Δεν παρέχθηκαν αρχεία για ανάλυση",
                data=None
            )
        
        paths = [Path(fp) for fp in file_paths]
        existing = [p for p in paths if p.exists()]
        
        if not existing:
            return ToolResult(
                success=False,
                error=f"Δεν βρέθηκαν αρχεία",
                data=None
            )
        
        try:
            # Run analysis
            analyzer = SimpleLogisticsAnalyzer()
            result = analyzer.analyze(existing)
            
            # Build response
            response_data = {
                "documents_analyzed": result.documents_analyzed,
                "anomaly_count": len(result.anomalies),
                "severity_breakdown": result.severity_summary,
                "summary": {
                    "total_anomalies": len(result.anomalies),
                    "critical": result.severity_summary.get("critical", 0),
                    "high": result.severity_summary.get("high", 0),
                    "medium": result.severity_summary.get("medium", 0),
                    "low": result.severity_summary.get("low", 0),
                },
                "anomalies": [
                    {
                        "id": a.anomaly_id,
                        "severity": a.severity.value,
                        "category": a.category.value,
                        "title": a.title,
                        "description": a.description,
                        "evidence": a.evidence,
                        "sources": a.source_documents,
                        "actions": a.suggested_actions
                    }
                    for a in result.anomalies
                ]
            }
            
            # Generate report
            if generate_report and result.anomalies:
                report_path = self._generate_report(result, report_format)
                response_data["report_path"] = str(report_path)
            
            logger.info(f"✅ Analysis complete: {len(result.anomalies)} anomalies found")
            
            return ToolResult(success=True, data=response_data)
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return ToolResult(success=False, error=str(e))
    
    def _generate_report(self, result: AnalysisResult, fmt: str) -> Path:
        """Generate audit report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        lines = [
            "# ΑΝΑΦΟΡΑ ΕΛΕΓΧΟΥ ΕΦΟΔΙΑΣΤΙΚΗΣ",
            "",
            f"**Ημερομηνία:** {result.analysis_timestamp}",
            f"**Έγγραφα:** {result.documents_analyzed}",
            f"**Ευρήματα:** {len(result.anomalies)}",
            "",
            "## ΣΥΝΟΨΗ",
            ""
        ]
        
        for sev, count in sorted(result.severity_summary.items()):
            icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(sev, "⚪")
            lines.append(f"- {icon} {sev.upper()}: {count}")
        
        lines.extend(["", "## ΕΥΡΗΜΑΤΑ", ""])
        
        for a in result.anomalies:
            icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(a.severity.value, "⚪")
            lines.extend([
                f"### {a.anomaly_id}: {a.title}",
                f"**Σοβαρότητα:** {icon} {a.severity.value.upper()}",
                f"**Κατηγορία:** {a.category.value}",
                "",
                a.description,
                ""
            ])
            
            if a.evidence:
                lines.append("**Στοιχεία:**")
                for ev in a.evidence:
                    lines.append(f"- {ev}")
                lines.append("")
            
            if a.suggested_actions:
                lines.append("**Ενέργειες:**")
                for act in a.suggested_actions:
                    lines.append(f"- [ ] {act}")
                lines.append("")
            
            lines.append("---")
        
        content = "\n".join(lines)
        output_path = self.output_dir / f"audit_report_{timestamp}.md"
        output_path.write_text(content, encoding='utf-8')
        
        return output_path


class LogisticsComparisonTool(BaseTool):
    """Tool for comparing logistics documents between periods."""
    
    def __init__(self, llm_provider=None):
        super().__init__()
        self.llm_provider = llm_provider
    
    @property
    def name(self) -> str:
        return "compare_logistics_documents"
    
    @property
    def description(self) -> str:
        return "Compare two sets of logistics documents to identify changes."
    
    def _execute_impl(
        self,
        baseline_files: List[str] = None,
        comparison_files: List[str] = None,
        file_paths: List[str] = None,
        **kwargs
    ) -> ToolResult:
        """Compare documents."""
        # If file_paths provided, split into two groups
        if file_paths and not baseline_files:
            paths = sorted(file_paths)
            mid = len(paths) // 2
            baseline_files = paths[:mid] if mid > 0 else paths[:1]
            comparison_files = paths[mid:] if mid > 0 else paths[1:]
        
        if not baseline_files or not comparison_files:
            return ToolResult(
                success=False,
                error="Απαιτούνται αρχεία για σύγκριση"
            )
        
        # Use the same analyzer for both sets
        analyzer = SimpleLogisticsAnalyzer()
        all_files = [Path(p) for p in baseline_files + comparison_files if Path(p).exists()]
        result = analyzer.analyze(all_files)
        
        return ToolResult(
            success=True,
            data={
                "anomalies": [
                    {
                        "id": a.anomaly_id,
                        "severity": a.severity.value,
                        "title": a.title,
                        "description": a.description
                    }
                    for a in result.anomalies
                ],
                "summary": result.severity_summary
            }
        )


def register_logistics_tools(registry, llm_provider=None):
    """Register logistics analysis tools."""
    registry.register(LogisticsAnomalyTool(llm_provider=llm_provider))
    registry.register(LogisticsComparisonTool(llm_provider=llm_provider))
    logger.info("Registered logistics analysis tools")