from abc import ABC, abstractmethod
from typing import Generator
from app.core.interfaces import LLMProvider as CoreLLMProvider
from app.llm.schemas import GenerationRequest, GenerationResponse

class BaseLLMBackend(CoreLLMProvider):
    """
    Base class for specific backends (Ollama, OpenAI, etc).
    Extends the Core interface to use our Typed Schemas.
    """
    
    @abstractmethod
    def generate_response(self, request: GenerationRequest) -> GenerationResponse:
        pass
        
    @abstractmethod
    def stream_response(self, request: GenerationRequest) -> Generator[str, None, None]:
        pass

    # Bridge to Core Interface (for compatibility)
    def generate(self, prompt: str, **kwargs) -> str:
        # Quick adapter: turn string into request
        req = GenerationRequest(messages=[{"role": "user", "content": prompt}], **kwargs)
        return self.generate_response(req).content

    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        req = GenerationRequest(messages=[{"role": "user", "content": prompt}], **kwargs)
        yield from self.stream_response(req)