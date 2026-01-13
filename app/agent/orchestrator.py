from typing import List, Dict, Any
from app.agent.schemas import AgentContext, AgentResponse
from app.agent.steps import PipelineStep
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class AgentOrchestrator:
    """
    Generic pipeline runner.
    Executes a sequence of steps to transform a query into a response.
    """
    
    def __init__(self, steps: List[PipelineStep]):
        self.steps = steps

    def run(self, query: str, chat_history: List[Dict[str, str]] = None) -> AgentResponse:
        context = AgentContext(
            query=query,
            chat_history=chat_history or []
        )
        
        logger.info(f"Starting pipeline execution for query: {query[:50]}...")
        
        for step in self.steps:
            try:
                context = step.process(context)
            except Exception as e:
                logger.error(f"Pipeline failed at step {step.__class__.__name__}: {e}")
                context.response = "An internal error occurred."
                break
                
        return AgentResponse(
            answer=context.response or "No response generated.",
            thinking=context.thinking or "",
            sources=context.sources,
            meta=context.metadata
        )