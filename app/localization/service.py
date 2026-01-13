from typing import Any, Dict
from app.config import get_config
from app.localization.schemas import LocaleBundle
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# Default Greek Fallback (Hardcoded backup if config.json is empty)
DEFAULT_GREEK = {
    "language_code": "el",
    "errors": {
        "generic": "Συγγνώμη, παρουσιάστηκε σφάλμα.",
        "generation_failed": "Συγγνώμη, δεν μπόρεσα να δημιουργήσω απάντηση.",
        "tool_not_found": "Το εργαλείο '{tool}' δεν βρέθηκε.",
        "tool_failed": "Η εκτέλεση του εργαλείου απέτυχε: {error}",
        "file_not_found": "Το αρχείο δεν βρέθηκε: {path}",
        "auth_failed": "Η αυθεντικοποίηση απέτυχε.",
        "chat_not_found": "Η συνομιλία δεν βρέθηκε."
    },
    "success": {
        "file_read": "Διάβασα το αρχείο {filename}.",
        "file_created": "Δημιουργήθηκε το αρχείο {filename}.",
        "tool_executed": "Το εργαλείο εκτελέστηκε επιτυχώς.",
        "login": "Επιτυχής σύνδεση!",
        "chat_created": "Δημιουργήθηκε νέα συνομιλία."
    },
    "logs": {
        "thinking": "Σκέφτομαι...",
        "processing": "Επεξεργασία...",
        "searching": "Αναζήτηση..."
    }
}

class LocalizationService:
    """
    Serves localized strings based on configuration.
    """
    
    def __init__(self):
        self._bundle: LocaleBundle = LocaleBundle()
        self.reload()

    def reload(self):
        """Reloads language settings from the global config."""
        config = get_config()
        loc_config = config.get("localization", {})
        
        # Determine active language
        lang = loc_config.get("default_language", "greek")
        
        if lang == "greek":
            # Merge config overrides over default greek
            data = DEFAULT_GREEK.copy()
            # If config has specific overrides, apply them here
            if "overrides" in loc_config:
                self._deep_update(data, loc_config["overrides"])
            self._bundle = LocaleBundle(**data)
        else:
            # Assume English or fully config-driven
            self._bundle = LocaleBundle(**loc_config.get("messages", {}))
            
        logger.info(f"Localization loaded: {self._bundle.language_code}")

    def _deep_update(self, base_dict, update_dict):
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    # --- Accessors ---

    def format_error(self, key: str, **kwargs) -> str:
        """Get formatted error message."""
        template = getattr(self._bundle.errors, key, self._bundle.errors.generic)
        try:
            return template.format(**kwargs)
        except Exception:
            return template

    def format_success(self, key: str, **kwargs) -> str:
        """Get formatted success message."""
        template = getattr(self._bundle.success, key, "Success.")
        try:
            return template.format(**kwargs)
        except Exception:
            return template

    def get_log(self, key: str) -> str:
        """Get status message (Thinking...)"""
        return getattr(self._bundle.logs, key, "...")

# Global Instance
localization_service = LocalizationService()