# tests/integration/test_gpu.py
"""
GPU and hardware tests.
"""

import pytest


def _torch_available():
    """Check if torch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


class TestGPUAvailability:
    """Tests for GPU detection and usage."""
    
    @pytest.mark.skipif(not _torch_available(), reason="torch not installed")
    def test_detect_gpu(self):
        """Test GPU detection."""
        import torch
        from app.utils.gpu import get_gpu_info
        
        gpu_info = get_gpu_info()
        
        if torch.cuda.is_available():
            assert gpu_info is not None
            assert "name" in gpu_info
            assert "total_memory_gb" in gpu_info
        else:
            assert gpu_info is None
    
    def test_gpu_cache_operations(self):
        """Test GPU cache clearing."""
        from app.utils.gpu import clear_gpu_cache
        
        # Should not raise even without GPU
        clear_gpu_cache()
    
    @pytest.mark.skipif(not _torch_available(), reason="torch not installed")
    def test_gpu_memory_logging(self):
        """Test GPU memory logging with actual GPU."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        from app.utils.gpu import log_gpu_memory, get_gpu_info
        
        # Should execute without error
        log_gpu_memory()
        
        info = get_gpu_info()
        assert info["allocated_memory_gb"] >= 0
        assert info["free_memory_gb"] > 0


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
class TestEmbeddingPerformance:
    """Tests for embedding model performance."""
    
    @pytest.fixture
    def embedding_provider(self):
        from app.rag.retrievers import LocalEmbeddingProvider
        return LocalEmbeddingProvider()
    
    def test_embed_single_text(self, embedding_provider):
        """Test embedding a single text."""
        import time
        
        text = "This is a test sentence for embedding."
        
        start = time.time()
        embedding = embedding_provider.embed_text(text)
        elapsed = time.time() - start
        
        assert len(embedding) > 0
        assert elapsed < 5.0  # Should complete in under 5 seconds
    
    def test_embed_batch(self, embedding_provider):
        """Test batch embedding."""
        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence.",
        ]
        
        embeddings = embedding_provider.embed_batch(texts)
        
        assert len(embeddings) == 3
        assert all(len(e) > 0 for e in embeddings)


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
class TestLLMPerformance:
    """Tests for LLM performance (requires GPU)."""
    
    def test_llm_generation_speed(self):
        """Test LLM generation with real model."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("GPU required")
        
        import time
        from app.llm.fast_providers import FastLocalModelProvider
        from app.config import LLMConfig
        
        config = LLMConfig(
            model_name="./offline_models/qwen3-4b",
            max_tokens=10000,
            quantization="4bit"
        )
        
        provider = FastLocalModelProvider(config)
        
        start = time.time()
        response = provider.generate("Hello, how are you?")
        elapsed = time.time() - start
        
        assert len(response) > 0
        print(f"Generation time: {elapsed:.2f}s")