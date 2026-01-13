import re
from typing import List, Dict, Any
from app.analysis.schemas import DetectedPattern, ExtractedContent
from app.config import get_config
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class PatternService:
    """
    Generic Regex and Keyword matcher.
    Configuration structure expected:
    {
        "analysis": {
            "patterns": {
                "dates": ["[0-9]{2}/[0-9]{2}/[0-9]{4}", ...],
                "locations": ["lat:", "lon:", ...],
                "weapons": ["rifle", "tank", ...]
            }
        }
    }
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or get_config().get("analysis", {}).get("patterns", {})
        self._compiled_regex = {}
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex for performance."""
        for category, patterns in self.config.items():
            self._compiled_regex[category] = []
            for p in patterns:
                try:
                    self._compiled_regex[category].append(re.compile(p, re.IGNORECASE))
                except re.error as e:
                    logger.error(f"Invalid regex pattern '{p}' for {category}: {e}")

    def scan(self, content: ExtractedContent) -> List[DetectedPattern]:
        """Scan a document against all configured patterns."""
        results = []
        text = content.text_content
        
        for category, regex_list in self._compiled_regex.items():
            for regex in regex_list:
                for match in regex.finditer(text):
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end].strip()
                    
                    results.append(DetectedPattern(
                        category=category,
                        value=match.group(0),
                        confidence=0.9,
                        source=content.source_name,
                        context=context
                    ))
        return results