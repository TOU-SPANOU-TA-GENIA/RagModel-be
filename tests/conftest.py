# tests/conftest.py
"""
Pytest configuration and shared fixtures.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def project_root():
    """Get project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Get test data directory."""
    test_dir = project_root / "tests" / "fixtures" / "data"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


@pytest.fixture
def sample_documents(test_data_dir):
    """Create sample documents for testing."""
    docs = {
        "test1.txt": "This is a test document about Python programming.",
        "test2.txt": "Information about machine learning and AI.",
        "test3.md": "# Markdown Test\n\nThis is markdown content."
    }
    
    created = []
    for name, content in docs.items():
        path = test_data_dir / name
        path.write_text(content)
        created.append(path)
    
    yield created
    
    # Cleanup
    for path in created:
        if path.exists():
            path.unlink()


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for testing without GPU."""
    from app.llm.providers import MockLLMProvider
    return MockLLMProvider()


@pytest.fixture
def mock_retriever():
    """Mock retriever for testing."""
    from app.rag.retrievers import create_mock_retriever
    return create_mock_retriever()


@pytest.fixture
def mock_agent(mock_llm_provider, mock_retriever):
    """Create agent with mock components."""
    from app.agent.integration import AgentConfig, AgentBuilder
    
    config = AgentConfig.for_development()
    builder = AgentBuilder(config)
    builder.llm_provider = mock_llm_provider
    builder.retriever = mock_retriever
    
    return builder.build()


# =============================================================================
# Server Fixtures
# =============================================================================

@pytest.fixture
def test_client():
    """Create FastAPI test client."""
    from fastapi.testclient import TestClient
    from app.main import app
    
    return TestClient(app)


@pytest.fixture
def base_url():
    """Base URL for server tests."""
    return "http://localhost:8000"


# =============================================================================
# Chat Fixtures
# =============================================================================

@pytest.fixture
def chat_session():
    """Create a temporary chat session."""
    from app.chat import create_chat, delete_chat
    
    chat_id = create_chat("Test Chat")
    yield chat_id
    
    try:
        delete_chat(chat_id)
    except:
        pass


# =============================================================================
# Environment Fixtures
# =============================================================================

@pytest.fixture
def gpu_available():
    """Check if GPU is available."""
    import torch
    return torch.cuda.is_available()


@pytest.fixture
def skip_without_gpu(gpu_available):
    """Skip test if GPU not available."""
    if not gpu_available:
        pytest.skip("GPU not available")