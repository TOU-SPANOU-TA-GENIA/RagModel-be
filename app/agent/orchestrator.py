# app/agent/orchestrator.py
"""
Simplified Agent Orchestrator - the main conductor of the agent system.
This replaces the monolithic Agent class with a modular, debuggable design.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from app.core.interfaces import (
    Context, Intent, Decision, 
    IntentClassifier, DecisionMaker, Tool,
    LLMProvider, Retriever, PromptBuilder,
    Pipeline, PipelineStep, event_bus
)
from app.logger import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# Simple Response Model
# ============================================================================

@dataclass
class AgentResponse:
    """Simplified response model."""
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    tool_used: Optional[str] = None
    tool_result: Optional[Dict[str, Any]] = None
    intent: str = "unknown"
    debug_info: List[str] = field(default_factory=list)
    execution_time: float = 0.0


# ============================================================================
# Pipeline Steps (each does ONE thing)
# ============================================================================

class IntentClassificationStep(PipelineStep):
    """Step 1: Classify user intent."""
    
    def __init__(self, classifier: IntentClassifier):
        self.classifier = classifier
    
    @property
    def name(self) -> str:
        return "Intent Classification"
    
    def process(self, context: Context) -> Context:
        intent = self.classifier.classify(context)
        context.metadata["intent"] = intent
        event_bus.emit("intent_classified", {"intent": intent.value})
        logger.info(f"Intent classified as: {intent.value}")
        return context


class DecisionMakingStep(PipelineStep):
    """Step 2: Decide what to do based on intent."""
    
    def __init__(self, decision_maker: DecisionMaker):
        self.decision_maker = decision_maker
    
    @property
    def name(self) -> str:
        return "Decision Making"
    
    def process(self, context: Context) -> Context:
        intent = context.metadata.get("intent", Intent.UNKNOWN)
        decision = self.decision_maker.decide(context, intent)
        context.metadata["decision"] = decision
        event_bus.emit("decision_made", {"decision": decision})
        logger.info(f"Decision: use_tool={decision.use_tool}, use_rag={decision.use_rag}")
        return context


class RAGRetrievalStep(PipelineStep):
    """Step 3: Retrieve relevant documents if needed."""
    
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
    
    @property
    def name(self) -> str:
        return "RAG Retrieval"
    
    def process(self, context: Context) -> Context:
        decision = context.metadata.get("decision")
        
        if decision and decision.use_rag:
            try:
                sources = self.retriever.retrieve(context.query)
                context.metadata["rag_sources"] = sources
                context.metadata["rag_context"] = self._format_sources(sources)
                logger.info(f"Retrieved {len(sources)} sources")
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
                context.metadata["rag_sources"] = []
                context.metadata["rag_context"] = ""
        else:
            context.metadata["rag_sources"] = []
            context.metadata["rag_context"] = ""
        
        return context
    
    def _format_sources(self, sources: List[Dict[str, Any]]) -> str:
        """Format sources into context text."""
        context_parts = []
        for source in sources:
            content = source.get("content", "")
            metadata = source.get("metadata", {})
            source_name = metadata.get("source", "Unknown")
            context_parts.append(f"[{source_name}]\n{content}")
        return "\n\n".join(context_parts)


class ToolExecutionStep(PipelineStep):
    """Step 4: Execute tools if needed."""
    
    def __init__(self, tool_registry: Dict[str, Tool]):
        self.tools = tool_registry
    
    @property
    def name(self) -> str:
        return "Tool Execution"
    
    def process(self, context: Context) -> Context:
        decision = context.metadata.get("decision")
        
        if decision and decision.use_tool and decision.tool_name:
            tool = self.tools.get(decision.tool_name)
            
            if tool:
                try:
                    result = tool.execute(**decision.tool_params)
                    context.metadata["tool_result"] = result
                    context.metadata["tool_used"] = decision.tool_name
                    logger.info(f"Tool {decision.tool_name} executed successfully")
                except Exception as e:
                    logger.error(f"Tool execution failed: {e}")
                    context.metadata["tool_result"] = {
                        "success": False,
                        "error": str(e)
                    }
            else:
                logger.warning(f"Tool not found: {decision.tool_name}")
        
        return context


class PromptBuildingStep(PipelineStep):
    """Step 5: Build the prompt for LLM."""
    
    def __init__(self, prompt_builder: PromptBuilder):
        self.prompt_builder = prompt_builder
    
    @property
    def name(self) -> str:
        return "Prompt Building"
    
    def process(self, context: Context) -> Context:
        # Gather all components for prompt
        rag_context = context.metadata.get("rag_context", "")
        tool_result = context.metadata.get("tool_result")
        
        # Build prompt with all relevant info
        prompt = self.prompt_builder.build(
            context,
            rag_context=rag_context,
            tool_result=tool_result
        )
        
        context.metadata["prompt"] = prompt
        logger.debug(f"Built prompt: {len(prompt)} characters")
        return context


class LLMGenerationStep(PipelineStep):
    """Step 6: Generate response using LLM."""
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
    
    @property
    def name(self) -> str:
        return "LLM Generation"
    
    def process(self, context: Context) -> Context:
        prompt = context.metadata.get("prompt", context.query)
        
        try:
            response = self.llm.generate(prompt)
            context.metadata["llm_response"] = response
            logger.info(f"Generated response: {len(response)} characters")
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            context.metadata["llm_response"] = "I apologize, but I encountered an error generating a response."
        
        return context


# ============================================================================
# Main Orchestrator (simplified)
# ============================================================================

class SimpleAgentOrchestrator:
    """
    Simplified agent orchestrator that's easy to understand and debug.
    Each component has a single responsibility.
    """
    
    def __init__(self,
                 intent_classifier: IntentClassifier,
                 decision_maker: DecisionMaker,
                 llm_provider: LLMProvider,
                 retriever: Optional[Retriever] = None,
                 prompt_builder: Optional[PromptBuilder] = None):
        """
        Initialize with injected dependencies.
        This makes it easy to swap implementations.
        """
        self.intent_classifier = intent_classifier
        self.decision_maker = decision_maker
        self.llm_provider = llm_provider
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.tools: Dict[str, Tool] = {}
        
        # Build the processing pipeline
        self.pipeline = self._build_pipeline()
        
        # Setup debug event handlers
        self._setup_debug_handlers()
        
        logger.info("SimpleAgentOrchestrator initialized")
    
    def _build_pipeline(self) -> Pipeline:
        """Build the processing pipeline."""
        pipeline = Pipeline()
        
        # Add steps in order
        pipeline.add_step(IntentClassificationStep(self.intent_classifier))
        pipeline.add_step(DecisionMakingStep(self.decision_maker))
        
        if self.retriever:
            pipeline.add_step(RAGRetrievalStep(self.retriever))
        
        pipeline.add_step(ToolExecutionStep(self.tools))
        
        if self.prompt_builder:
            pipeline.add_step(PromptBuildingStep(self.prompt_builder))
        
        pipeline.add_step(LLMGenerationStep(self.llm_provider))
        
        return pipeline
    
    def _setup_debug_handlers(self):
        """Setup event handlers for debugging."""
        def log_event(data):
            logger.debug(f"Event: {data}")
        
        event_bus.on("intent_classified", log_event)
        event_bus.on("decision_made", log_event)
    
    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent."""
        self.tools[tool.name] = tool
        logger.info(f"Added tool: {tool.name}")
    
    def process_query(self,
                     query: str,
                     chat_history: Optional[List[Dict[str, str]]] = None) -> AgentResponse:
        """
        Process a query through the pipeline.
        Much simpler than the original 150+ line method!
        """
        start_time = time.time()
        
        # Create context
        context = Context(
            query=query,
            chat_history=chat_history or [],
            metadata={},
            debug_info=[]
        )
        
        logger.info(f"Processing query: {query[:100]}...")
        
        # Process through pipeline
        try:
            context = self.pipeline.process(context)
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            return AgentResponse(
                answer="I encountered an error processing your request.",
                debug_info=[str(e)],
                execution_time=time.time() - start_time
            )
        
        # Build response
        response = self._build_response(context)
        response.execution_time = time.time() - start_time
        
        logger.info(f"Query processed in {response.execution_time:.2f}s")
        
        return response
    
    def _build_response(self, context: Context) -> AgentResponse:
        """Build the final response from context."""
        return AgentResponse(
            answer=context.metadata.get("llm_response", "No response generated"),
            sources=context.metadata.get("rag_sources", []),
            tool_used=context.metadata.get("tool_used"),
            tool_result=context.metadata.get("tool_result"),
            intent=context.metadata.get("intent", Intent.UNKNOWN).value,
            debug_info=context.debug_info
        )


# ============================================================================
# Factory Function (replaces create_agent)
# ============================================================================

def create_simple_agent(
    llm_provider: LLMProvider,
    intent_classifier: IntentClassifier,
    decision_maker: DecisionMaker,
    retriever: Optional[Retriever] = None,
    prompt_builder: Optional[PromptBuilder] = None
) -> SimpleAgentOrchestrator:
    """
    Factory function to create an agent with dependencies.
    This replaces the old create_agent function.
    """
    agent = SimpleAgentOrchestrator(
        intent_classifier=intent_classifier,
        decision_maker=decision_maker,
        llm_provider=llm_provider,
        retriever=retriever,
        prompt_builder=prompt_builder
    )
    
    return agent