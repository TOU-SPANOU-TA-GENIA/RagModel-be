import logging
import asyncio
import time
from typing import List, Union, AsyncGenerator
from app.llm.provider import LLMProvider

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.provider = None

    def initialize(self):
        if not self.provider:
            self.provider = LLMProvider()
            self.provider.load_model()

    async def stream_response(self, messages: list) -> AsyncGenerator[str, None]:
        if not self.provider:
            self.initialize()

        start_time = time.time()
        chunk_count = 0
        
        try:
            generator = self.provider.stream(messages)
            
            for chunk in generator:
                if chunk:
                    chunk_count += 1
                    # Removed terminal print(chunk) to clean backend
                    yield chunk
                    await asyncio.sleep(0) # Yield for async loop efficiency
            
            duration = time.time() - start_time
            print(f"🤖 [LLM] Stream complete: {chunk_count} chunks in {duration:.2f}s")

        except Exception as e:
            logger.error(f"LLM Stream Error: {e}")
            yield f" [Error: {str(e)}]"

llm_service = LLMService()