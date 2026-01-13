import sys
import os
import asyncio

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.llm import llm_service
from app.config import get_config

async def test_generation():
    print("\n🤖 Initializing LLM Service...")
    
    # 1. Force Initialize
    llm_service.initialize()
    
    prompt = "Hello, who are you?"
    print(f"\n📝 Sending Prompt: '{prompt}'")
    
    print("-" * 30)
    print("STREAMING OUTPUT:")
    print("-" * 30)
    
    # 2. Test Streaming
    token_count = 0
    try:
        if hasattr(llm_service, 'stream'):
            # Consume the generator
            iterator = llm_service.stream(prompt)
            # Handle both async and sync iterators
            if hasattr(iterator, '__aiter__'):
                async for chunk in iterator:
                    print(chunk, end="", flush=True)
                    token_count += 1
            else:
                for chunk in iterator:
                    print(chunk, end="", flush=True)
                    token_count += 1
        else:
            print("[WARNING] No 'stream' method found. Testing 'generate'...")
            response = llm_service.generate(prompt)
            print(response)
            token_count = len(response)
            
    except Exception as e:
        print(f"\n\n❌ ERROR during generation: {e}")

    print(f"\n\n" + "-" * 30)
    print(f"✅ Finished. Tokens received: {token_count}")

if __name__ == "__main__":
    asyncio.run(test_generation())