# app/agent/integration.py
"""
Integration module - shows how to wire all components together.
This replaces your original complex initialization with clean dependency injection.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
import os

from app.core.interfaces import Context, event_bus
from app.agent.orchestrator import SimpleAgentOrchestrator, AgentResponse
from app.agent.classifiers import (
    RuleBasedIntentClassifier,
    SimpleDecisionMaker,
    create_rule_based_analyzer
)
from app.llm.providers import (
    create_llm_provider,
    create_prompt_builder,
    MockLLMProvider
)
from app.rag.retrievers import (
    create_simple_retriever,
    create_mock_retriever,
    DocumentProcessor,
    LocalEmbeddingProvider
)
from app.tools.base import (
    create_default_tools,
    create_tool_registry_for_military,
    ReadFileTool,
    SimpleToolRegistry
)
from app.config import (
    LLM_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    SYSTEM_INSTRUCTION,
    KNOWLEDGE_DIR,
    INDEX_DIR
)
from app.logger import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# Agent Configurations
# ============================================================================

class AgentConfig:
    """Configuration for agent initialization."""
    
    def __init__(self):
        # Model settings
        self.llm_model_name = LLM_MODEL_NAME
        self.embedding_model_name = EMBEDDING_MODEL_NAME
        self.use_mock_llm = False
        self.use_cache = False
        
        # RAG settings
        self.use_rag = True
        self.use_faiss = True
        self.index_path = str(INDEX_DIR)
        
        # Tool settings
        self.enable_tools = True
        self.tool_config = "default"  # "default", "military", "custom"
        self.allowed_directories = [KNOWLEDGE_DIR, Path.cwd()]
        
        # System settings
        self.system_instruction = SYSTEM_INSTRUCTION
        self.debug_mode = False
        
    @classmethod
    def for_development(cls) -> 'AgentConfig':
        """Create config for development/testing."""
        config = cls()
        config.use_mock_llm = True
        config.use_cache = True
        config.use_faiss = False
        config.debug_mode = True
        return config
    
    @classmethod
    def for_production(cls) -> 'AgentConfig':
        """Create config for production."""
        config = cls()
        config.use_mock_llm = False
        config.use_cache = False
        config.use_faiss = True
        config.debug_mode = False
        return config
    
    @classmethod
    def for_military(cls) -> 'AgentConfig':
        """Create config for military deployment."""
        config = cls()
        config.tool_config = "military"
        config.allowed_directories = [
            Path("/opt/military_app/data"),
            Path("/var/log/military_app")
        ]
        return config


# ============================================================================
# Agent Builder (Factory)
# ============================================================================

class AgentBuilder:
    """
    Builder pattern for constructing agents with different configurations.
    This makes it easy to create agents for different environments.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        
        # Components (will be initialized)
        self.llm_provider = None
        self.intent_classifier = None
        self.decision_maker = None
        self.retriever = None
        self.prompt_builder = None
        self.tool_registry = None
        
    def build_llm_provider(self) -> 'AgentBuilder':
        """Build optimized LLM provider."""
        if self.config.use_mock_llm:
            logger.info("Using mock LLM provider")
            self.llm_provider = create_llm_provider(provider_type="mock")
        else:
            logger.info(f"Creating FAST LLM provider for {self.config.llm_model_name}")
            # Use pre-warmed provider for maximum speed
            self.llm_provider = create_llm_provider(
                model_name=self.config.llm_model_name,
                provider_type="prewarmed",  # Use pre-warmed provider
                use_cache=self.config.use_cache
            )
        return self
    
    def build_classifiers(self) -> 'AgentBuilder':
        """Build intent classifier and decision maker."""
        logger.info("Building classifiers")
        
        self.intent_classifier = RuleBasedIntentClassifier()
        
        # Decision maker needs tool registry
        if self.config.enable_tools and self.tool_registry:
            self.decision_maker = SimpleDecisionMaker(self.tool_registry.tools)
        else:
            self.decision_maker = SimpleDecisionMaker()
        
        return self
    
    def build_retriever(self) -> 'AgentBuilder':
        """Build retriever based on config."""
        if not self.config.use_rag:
            logger.info("RAG disabled")
            self.retriever = None
        elif self.config.use_mock_llm:
            logger.info("Using mock retriever")
            self.retriever = create_mock_retriever()
        else:
            logger.info("Creating in-memory retriever")
            # Always use in-memory for performance
            from app.rag.retrievers import SimpleRetriever, LocalEmbeddingProvider
            from app.core.memory_store import FastInMemoryVectorStore, CachedEmbeddingProvider
            
            # Create base embedding provider
            base_embedding_provider = LocalEmbeddingProvider(self.config.embedding_model_name)
            
            # Wrap with caching
            embedding_provider = CachedEmbeddingProvider(base_embedding_provider)
            
            # Use in-memory vector store
            vector_store = FastInMemoryVectorStore()
            
            self.retriever = SimpleRetriever(embedding_provider, vector_store)
            
        return self
    
    def build_tools(self) -> 'AgentBuilder':
        """Build tool registry based on config."""
        if not self.config.enable_tools:
            logger.info("Tools disabled")
            self.tool_registry = SimpleToolRegistry()
        elif self.config.tool_config == "military":
            logger.info("Creating military tool registry")
            self.tool_registry = create_tool_registry_for_military()
        elif self.config.tool_config == "custom":
            logger.info("Creating custom tool registry")
            self.tool_registry = self._build_custom_tools()
        else:
            logger.info("Creating default tool registry")
            self.tool_registry = create_default_tools()
        
        return self
    
    def _build_custom_tools(self) -> SimpleToolRegistry:
        """Build custom tool registry."""
        registry = SimpleToolRegistry()
        
        # Add custom configured read tool
        read_tool = ReadFileTool(
            allowed_dirs=self.config.allowed_directories,
            max_file_size_mb=20
        )
        registry.register(read_tool)
        
        # Add more tools as needed
        
        return registry
    
    def build_prompt_builder(self) -> 'AgentBuilder':
        """Build intelligent prompt builder."""
        logger.info("Building intelligent prompt builder")
        
        tools_dict = None
        if self.tool_registry and self.config.enable_tools:
            tools_dict = self.tool_registry.tools
        
        # Use new intelligent prompt builder
        try:
            from app.llm.intelligent_prompt_builder import IntelligentPromptBuilder
            self.prompt_builder = IntelligentPromptBuilder(
                system_instruction=self.config.system_instruction,
                tools=tools_dict
            )
            logger.info("Using IntelligentPromptBuilder with context filtering")
        except ImportError as e:
            logger.warning(f"IntelligentPromptBuilder not available: {e}, using fallback")
            # Fallback to existing builder
            from app.llm.providers import create_prompt_builder
            self.prompt_builder = create_prompt_builder(
                system_instruction=self.config.system_instruction,
                tools=tools_dict
            )
        
        return self
    
    def build(self) -> SimpleAgentOrchestrator:
        """Build the complete agent."""
        logger.info("Building agent orchestrator")
        
        # Build all components
        self.build_llm_provider()
        self.build_tools()  # Build tools before classifiers
        self.build_classifiers()
        self.build_retriever()
        self.build_prompt_builder()
        
        # Create orchestrator
        orchestrator = SimpleAgentOrchestrator(
            intent_classifier=self.intent_classifier,
            decision_maker=self.decision_maker,
            llm_provider=self.llm_provider,
            retriever=self.retriever,
            prompt_builder=self.prompt_builder
        )
        
        # Add tools to orchestrator
        if self.tool_registry:
            for tool in self.tool_registry.tools.values():
                orchestrator.add_tool(tool)
        
        # Setup debug events if enabled
        if self.config.debug_mode:
            self._setup_debug_events()
        
        logger.info("Agent orchestrator built successfully")
        return orchestrator
    
    def _setup_debug_events(self):
        """Setup debug event handlers."""
        def debug_handler(data):
            print(f"[DEBUG EVENT] {data}")
        
        event_bus.on("intent_classified", debug_handler)
        event_bus.on("decision_made", debug_handler)
        event_bus.on("tool_executed", debug_handler)
        event_bus.on("rag_retrieved", debug_handler)


# ============================================================================
# High-Level Factory Functions
# ============================================================================

def create_agent(
    mode: str = "production",
    config: Optional[AgentConfig] = None
) -> SimpleAgentOrchestrator:
    """
    Main factory function to create an agent.
    This replaces your original create_agent function.
    
    Args:
        mode: "development", "production", or "military"
        config: Optional custom configuration
    
    Returns:
        Configured agent orchestrator
    """
    # Get appropriate config
    if config is None:
        if mode == "development":
            config = AgentConfig.for_development()
        elif mode == "military":
            config = AgentConfig.for_military()
        else:
            config = AgentConfig.for_production()
    
    # Build agent
    builder = AgentBuilder(config)
    agent = builder.build()
    
    logger.info(f"Created {mode} agent")
    return agent


def create_minimal_agent() -> SimpleAgentOrchestrator:
    """
    Create a minimal agent for testing.
    No RAG, no tools, mock LLM.
    """
    config = AgentConfig()
    config.use_mock_llm = True
    config.use_rag = False
    config.enable_tools = False
    
    builder = AgentBuilder(config)
    return builder.build()


def create_rag_only_agent() -> SimpleAgentOrchestrator:
    """
    Create an agent with only RAG capabilities.
    Useful for Q&A systems.
    """
    config = AgentConfig()
    config.enable_tools = False
    config.use_rag = True
    
    builder = AgentBuilder(config)
    return builder.build()


def create_tool_only_agent() -> SimpleAgentOrchestrator:
    """
    Create an agent with only tool capabilities.
    Useful for automation tasks.
    """
    config = AgentConfig()
    config.use_rag = False
    config.enable_tools = True
    
    builder = AgentBuilder(config)
    return builder.build()


# ============================================================================
# Agent Manager (Singleton Pattern)
# ============================================================================

class AgentManager:
    """
    Singleton manager for the agent.
    Ensures only one agent instance exists.
    """
    
    _instance = None
    _agent = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_agent(self, mode: str = "production") -> SimpleAgentOrchestrator:
        """Get or create the agent."""
        if self._agent is None:
            logger.info(f"Initializing agent in {mode} mode")
            self._agent = create_agent(mode)
        return self._agent
    
    def reset(self):
        """Reset the agent (useful for testing)."""
        self._agent = None
        logger.info("Agent manager reset")
    
    def update_config(self, config: AgentConfig):
        """Update agent with new configuration."""
        logger.info("Updating agent configuration")
        builder = AgentBuilder(config)
        self._agent = builder.build()


# Global agent manager
agent_manager = AgentManager()


# ============================================================================
# Convenience Functions (for backward compatibility)
# ============================================================================

def get_agent() -> SimpleAgentOrchestrator:
    """
    Get the global agent instance.
    This maintains compatibility with your existing code.
    """
    return agent_manager.get_agent()


def process_query(
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> AgentResponse:
    """
    Process a query using the global agent.
    Convenience function for simple usage.
    """
    agent = get_agent()
    return agent.process_query(query, chat_history)


# ============================================================================
# Testing Utilities
# ============================================================================

def test_agent_pipeline():
    """Test the agent pipeline with different configurations."""
    
    print("\n" + "="*60)
    print("Testing Agent Pipeline")
    print("="*60)
    
    # Test 1: Minimal agent
    print("\n1. Testing minimal agent...")
    agent = create_minimal_agent()
    response = agent.process_query("Hello, how are you?")
    print(f"   Response: {response.answer[:50]}...")
    print(f"   Intent: {response.intent}")
    
    # Test 2: RAG agent
    print("\n2. Testing RAG agent...")
    agent = create_rag_only_agent()
    response = agent.process_query("What is Panos's favorite food?")
    print(f"   Response: {response.answer[:50]}...")
    print(f"   Sources: {len(response.sources)}")
    
    # Test 3: Tool agent
    print("\n3. Testing tool agent...")
    agent = create_tool_only_agent()
    response = agent.process_query("Read the file at /data/test.txt")
    print(f"   Response: {response.answer[:50]}...")
    print(f"   Tool used: {response.tool_used}")
    
    print("\n" + "="*60)
    print("Pipeline tests complete!")
    print("="*60)


if __name__ == "__main__":
    # Run tests if executed directly
    test_agent_pipeline()