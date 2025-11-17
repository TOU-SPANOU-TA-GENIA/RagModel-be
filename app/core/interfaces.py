# app/core/interfaces.py
"""
Core interfaces for the agent system.
These abstract base classes define contracts that all implementations must follow.
This makes it easy to swap implementations and add new models/tools.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# Core Data Models (moved from models.py, simplified)
# ============================================================================

class Intent(Enum):
    """User intent classification."""
    QUESTION = "question"
    ACTION = "action"
    CONVERSATION = "conversation"
    UNKNOWN = "unknown"


@dataclass
class Context:
    """
    Execution context passed through the pipeline.
    This carries all information needed by components.
    """
    query: str
    chat_history: List[Dict[str, str]]
    metadata: Dict[str, Any]
    debug_info: List[str]  # For debugging
    
    def add_debug(self, message: str):
        """Add debug information for tracing."""
        self.debug_info.append(f"[{len(self.debug_info)}] {message}")


@dataclass
class Decision:
    """Agent decision about how to handle a query."""
    intent: Intent
    confidence: float
    use_tool: bool
    tool_name: Optional[str] = None
    tool_params: Optional[Dict[str, Any]] = None
    use_rag: bool = True
    reasoning: str = ""


# ============================================================================
# Component Interfaces
# ============================================================================

class IntentClassifier(ABC):
    """Interface for intent classification."""
    
    @abstractmethod
    def classify(self, context: Context) -> Intent:
        """Classify the intent of a query."""
        pass


class DecisionMaker(ABC):
    """Interface for decision making."""
    
    @abstractmethod
    def decide(self, context: Context, intent: Intent) -> Decision:
        """Make a decision based on context and intent."""
        pass


class Tool(ABC):
    """Base interface for all tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for LLM."""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool."""
        pass


class LLMProvider(ABC):
    """Interface for LLM providers (easy to swap models)."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name."""
        pass


class Retriever(ABC):
    """Interface for document retrieval."""
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant documents."""
        pass


class PromptBuilder(ABC):
    """Interface for prompt construction."""
    
    @abstractmethod
    def build(self, context: Context, **kwargs) -> str:
        """Build a prompt from context."""
        pass


# ============================================================================
# Pipeline Components (for modular processing)
# ============================================================================

class PipelineStep(ABC):
    """Base class for pipeline steps."""
    
    @abstractmethod
    def process(self, context: Context) -> Context:
        """Process the context and return updated context."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Step name for debugging."""
        pass


class Pipeline:
    """Simple pipeline for processing steps."""
    
    def __init__(self):
        self.steps: List[PipelineStep] = []
    
    def add_step(self, step: PipelineStep) -> 'Pipeline':
        """Add a processing step (fluent interface)."""
        self.steps.append(step)
        return self
    
    def process(self, context: Context) -> Context:
        """Process context through all steps."""
        for step in self.steps:
            context.add_debug(f"Processing: {step.name}")
            context = step.process(context)
        return context


# ============================================================================
# Event System (for debugging and monitoring)
# ============================================================================

class EventBus:
    """Simple event bus for debugging and monitoring."""
    
    def __init__(self):
        self.handlers: Dict[str, List[callable]] = {}
    
    def on(self, event: str, handler: callable) -> None:
        """Subscribe to an event."""
        if event not in self.handlers:
            self.handlers[event] = []
        self.handlers[event].append(handler)
    
    def emit(self, event: str, data: Any = None) -> None:
        """Emit an event."""
        if event in self.handlers:
            for handler in self.handlers[event]:
                handler(data)


# Global event bus for debugging
event_bus = EventBus()