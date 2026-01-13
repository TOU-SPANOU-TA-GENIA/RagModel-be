from .service import localization_service, LocalizationService
from .schemas import LocaleBundle

# Convenience functions for cleaner imports in other modules
def get_error(key: str, **kwargs) -> str:
    return localization_service.format_error(key, **kwargs)

def get_success(key: str, **kwargs) -> str:
    return localization_service.format_success(key, **kwargs)

def get_status(key: str) -> str:
    return localization_service.get_log(key)

__all__ = [
    "localization_service", 
    "LocalizationService", 
    "LocaleBundle",
    "get_error",
    "get_success",
    "get_status"
]