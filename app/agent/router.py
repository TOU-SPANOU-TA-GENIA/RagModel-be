import re
from typing import Dict, Any, List, Optional
from app.agent.schemas import AgentContext, AgentIntent
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class IntentRouter:
    """
    Determines intent (Chat vs Tool) based on configuration.
    Supports both Regex (fast) and LLM-based (smart) routing.
    """
    
    def __init__(self, config: Dict[str, Any], llm_provider=None):
        self.config = config
        self.llm = llm_provider
        # Load regex patterns from config, not hardcoded files
        self.patterns = config.get("intent_patterns", {})
        
    def route(self, context: AgentContext) -> AgentContext:
        """Analyze query and update context with intent."""
        query_lower = context.query.lower()
        
        # 1. Fast Path: Regex Matching defined in config.json
        for intent_name, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    logger.info(f"Router: Matched pattern for '{intent_name}'")
                    # Assuming config maps intent names to tool names or generic intents
                    if intent_name == "chat":
                        context.intent = AgentIntent.CHAT
                    else:
                        context.intent = AgentIntent.TOOL_USE
                        context.suggested_tool = intent_name
                    return context

        # 2. Smart Path: LLM Routing (if configured and no regex matched)
        if self.config.get("use_llm_routing", False) and self.llm:
            return self._route_with_llm(context)
            
        # 3. Default Fallback
        context.intent = AgentIntent.CHAT
        return context

    def _route_with_llm(self, context: AgentContext) -> AgentContext:
        """
        Ask LLM to classify intent. 
        Implementation should use a lightweight routing prompt defined in config.
        """
        # Placeholder for LLM routing logic
        # In a real scenario, this would format a prompt with available tools
        # and ask the LLM to pick one.
        context.intent = AgentIntent.CHAT
        return context