from typing import Callable, Dict, List, Any
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class EventBus:
    """
    Simple synchronous event bus.
    Allows components to subscribe to system events (e.g., "tool_used", "rag_retrieved").
    """
    _subscribers: Dict[str, List[Callable]] = {}

    @classmethod
    def subscribe(cls, event_name: str, callback: Callable[[Any], None]):
        if event_name not in cls._subscribers:
            cls._subscribers[event_name] = []
        cls._subscribers[event_name].append(callback)
        logger.debug(f"Subscribed to event: {event_name}")

    @classmethod
    def publish(cls, event_name: str, data: Any = None):
        if event_name in cls._subscribers:
            for callback in cls._subscribers[event_name]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_name}: {e}")

# Global instance not strictly needed as methods are classmethods, 
# but provided for dependency injection styles if needed.
event_bus = EventBus()