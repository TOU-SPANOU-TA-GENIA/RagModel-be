import uuid
import sys
from typing import List, Dict, Any

from app.diagnostics.schemas import DiagnosticReport, CheckResult, CheckStatus
from app.diagnostics.checks import SystemChecks
from app.config import get_config

class DiagnosticService:
    """
    Runs configured diagnostic checks.
    """
    
    def __init__(self):
        self.config = get_config()
        
    def run_diagnostics(self) -> DiagnosticReport:
        """Run all enabled checks and return a report."""
        checks: List[CheckResult] = []
        
        # 1. GPU Checks
        checks.append(SystemChecks.check_nvidia_driver())
        checks.append(SystemChecks.check_cuda_toolkit())
        checks.append(SystemChecks.check_pytorch_gpu())
        
        # 2. System Checks
        checks.append(SystemChecks.check_disk_space())
        
        # Determine overall status
        failed = any(c.status == CheckStatus.FAIL for c in checks)
        warning = any(c.status == CheckStatus.WARNING for c in checks)
        
        if failed:
            overall = CheckStatus.FAIL
        elif warning:
            overall = CheckStatus.WARNING
        else:
            overall = CheckStatus.PASS

        return DiagnosticReport(
            run_id=str(uuid.uuid4()),
            overall_status=overall,
            checks=checks,
            system_info={
                "platform": sys.platform,
                "python_version": sys.version.split()[0]
            }
        )

# Global instance
diagnostic_service = DiagnosticService()