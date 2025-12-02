# app/llm/streaming_provider.py
"""
Streaming LLM Provider for real-time token generation.
Delivers tokens as they're generated via async generators.
"""

import asyncio
import time
from typing import AsyncGenerator, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class StreamEventType(Enum):
    """Types of streaming events."""
    TOKEN = "token"
    THINKING_START = "thinking_start"
    THINKING_END = "thinking_end"
    RESPONSE_START = "response_start"
    RESPONSE_END = "response_end"
    ERROR = "error"
    DONE = "done"


@dataclass
class StreamEvent:
    """Single streaming event."""
    event_type: StreamEventType
    data: str = ""
    metadata: Optional[Dict[str, Any]] = None
    
    def to_sse(self) -> str:
        """Convert to Server-Sent Events format."""
        import json
        payload = {
            "type": self.event_type.value,
            "data": self.data
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


class StreamingLLMProvider:
    """
    LLM provider with streaming support.
    
    Uses transformers TextIteratorStreamer for token-by-token generation.
    Wraps thinking in <think> tags and filters from output stream.
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._initialized = False
        self._thinking_mode = False
    
    def _ensure_initialized(self):
        """Lazy initialization of model."""
        if self._initialized:
            return
        
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        logger.info(f"Loading streaming model: {self.config.model_name}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map="cpu",
                trust_remote_code=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
        
        self._initialized = True
        logger.info(f"Streaming model loaded on {device}")
    
    async def generate_stream(
        self, 
        prompt: str,
        include_thinking: bool = False,
        **kwargs
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Generate response tokens as async stream.
        
        Args:
            prompt: Input prompt
            include_thinking: If True, yields thinking tokens wrapped in <think> tags.
                            If False (default), thinking is processed but not yielded.
            **kwargs: Generation parameters
        
        Yields:
            StreamEvent objects for each token/event
        """
        self._ensure_initialized()
        
        import torch
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        temperature = kwargs.get("temperature", self.config.temperature)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Create streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # Generation config
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_tokens,
            "temperature": temperature if temperature > 0 else 0.01,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        
        # Start generation in background thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # State tracking for thinking detection
        buffer = ""
        in_thinking = False
        thinking_buffer = ""
        response_started = False
        
        try:
            for token in streamer:
                buffer += token
                
                # Detect thinking block start
                if "<think>" in buffer and not in_thinking:
                    in_thinking = True
                    # Emit content before <think>
                    pre_think = buffer.split("<think>")[0]
                    if pre_think.strip():
                        yield StreamEvent(StreamEventType.TOKEN, pre_think)
                    
                    if include_thinking:
                        yield StreamEvent(StreamEventType.THINKING_START)
                    
                    thinking_buffer = buffer.split("<think>")[-1]
                    buffer = ""
                    continue
                
                # Detect thinking block end
                if "</think>" in buffer and in_thinking:
                    in_thinking = False
                    
                    # Extract thinking content
                    parts = buffer.split("</think>")
                    thinking_content = thinking_buffer + parts[0]
                    
                    if include_thinking:
                        yield StreamEvent(StreamEventType.TOKEN, thinking_content)
                        yield StreamEvent(StreamEventType.THINKING_END)
                    
                    # Continue with post-thinking content
                    buffer = parts[-1] if len(parts) > 1 else ""
                    thinking_buffer = ""
                    
                    if not response_started:
                        response_started = True
                        yield StreamEvent(StreamEventType.RESPONSE_START)
                    
                    if buffer.strip():
                        yield StreamEvent(StreamEventType.TOKEN, buffer)
                        buffer = ""
                    continue
                
                # In thinking mode - accumulate or stream based on setting
                if in_thinking:
                    thinking_buffer += token
                    if include_thinking:
                        yield StreamEvent(StreamEventType.TOKEN, token)
                    continue
                
                # Normal response mode
                if not response_started and buffer.strip():
                    response_started = True
                    yield StreamEvent(StreamEventType.RESPONSE_START)
                
                # Yield token immediately
                yield StreamEvent(StreamEventType.TOKEN, token)
                buffer = ""
            
            # Flush remaining buffer
            if buffer.strip() and not in_thinking:
                yield StreamEvent(StreamEventType.TOKEN, buffer)
            
            yield StreamEvent(StreamEventType.RESPONSE_END)
            yield StreamEvent(StreamEventType.DONE)
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield StreamEvent(StreamEventType.ERROR, str(e))
        
        thread.join()
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Non-streaming generation (for compatibility).
        Collects all tokens and returns complete response.
        """
        import asyncio
        
        async def collect():
            tokens = []
            async for event in self.generate_stream(prompt, include_thinking=False, **kwargs):
                if event.event_type == StreamEventType.TOKEN:
                    tokens.append(event.data)
            return "".join(tokens)
        
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(collect())
        finally:
            loop.close()
    
    def get_model_name(self) -> str:
        return self.config.model_name
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.config.model_name,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "streaming": True,
        }


def create_streaming_provider(config=None):
    """Factory function for streaming provider."""
    from app.config import LLMConfig, LLM_MODEL_NAME
    
    if config is None:
        config = LLMConfig(
            model_name=LLM_MODEL_NAME,
            max_tokens=256,
            temperature=0.7,
        )
    
    return StreamingLLMProvider(config)