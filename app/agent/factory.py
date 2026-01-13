from typing import Dict, Any, Optional
from app.agent.orchestrator import AgentOrchestrator
from app.agent.router import IntentRouter
from app.agent.steps import (
    IntentRecognitionStep, 
    ContextRetrievalStep, 
    ToolExecutionStep, 
    ResponseGenerationStep
)
from app.core.interfaces import LLMProvider, Retriever, Tool
# Assume we have a generic way to get tools and config
from app.config import get_config 

class AgentFactory:
    """
    Builds the Agent Orchestrator by assembling steps defined in configuration.
    """
    
    @staticmethod
    def create(
        llm_provider: LLMProvider,
        retriever: Optional[Retriever] = None,
        tools: Dict[str, Tool] = None,
        config: Dict[str, Any] = None
    ) -> AgentOrchestrator:
        
        cfg = config or get_config()
        agent_cfg = cfg.get("agent", {})
        
        # 1. Router
        router = IntentRouter(agent_cfg, llm_provider)
        
        # 2. Steps Assembly
        steps = []
        
        # Step: Identify Intent
        steps.append(IntentRecognitionStep(router))
        
        # Step: Tool Execution (if intent matches)
        if tools:
            steps.append(ToolExecutionStep(tools))
            
        # Step: RAG (if enabled)
        if retriever and agent_cfg.get("enable_rag", True):
            steps.append(ContextRetrievalStep(retriever, agent_cfg))
            
        # Step: Final Generation
        steps.append(ResponseGenerationStep(llm_provider, agent_cfg))
        
        return AgentOrchestrator(steps)