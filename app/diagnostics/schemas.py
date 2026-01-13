from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime

class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"

class CheckResult(BaseModel):
    """Result of a single diagnostic check (e.g., 'NVIDIA Driver Check')."""
    name: str
    category: str  # 'gpu', 'system', 'network'
    status: CheckStatus
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class DiagnosticReport(BaseModel):
    """Complete system health report."""
    run_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    overall_status: CheckStatus
    checks: List[CheckResult]
    system_info: Dict[str, Any] = Field(default_factory=dict)