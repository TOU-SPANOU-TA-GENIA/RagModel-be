from abc import ABC, abstractmethod
from typing import Dict, Any
from app.agent.schemas import AgentContext, AgentIntent
from app.agent.router import IntentRouter
from app.core.interfaces import LLMProvider, Retriever, Tool
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class PipelineStep(ABC):
    @abstractmethod
    def process(self, context: AgentContext) -> AgentContext:
        pass

class IntentRecognitionStep(PipelineStep):
    def __init__(self, router: IntentRouter):
        self.router = router

    def process(self, context: AgentContext) -> AgentContext:
        return self.router.route(context)

class ContextRetrievalStep(PipelineStep):
    """Handles RAG retrieval."""
    def __init__(self, retriever: Retriever, config: Dict[str, Any]):
        self.retriever = retriever
        self.config = config

    def process(self, context: AgentContext) -> AgentContext:
        if context.intent == AgentIntent.TOOL_USE and not self.config.get("rag_on_tools", False):
            return context
            
        try:
            k = self.config.get("rag_k", 3)
            results = self.retriever.retrieve(context.query, k=k)
            context.rag_docs = results
            for res in results:
                src = res.get("metadata", {}).get("source", "unknown")
                context.add_source(src)
        except Exception as e:
            logger.error(f"RAG failed: {e}")
        
        return context

class ToolExecutionStep(PipelineStep):
    """Executes a tool if the intent dictates it."""
    def __init__(self, tools: Dict[str, Tool]):
        self.tools = tools

    def process(self, context: AgentContext) -> AgentContext:
        if context.intent != AgentIntent.TOOL_USE or not context.suggested_tool:
            return context

        tool_name = context.suggested_tool
        tool = self.tools.get(tool_name)
        
        if not tool:
            logger.warning(f"Tool '{tool_name}' not found in registry.")
            return context

        try:
            # In a unified system, we might need an extraction step before execution
            # to get parameters. For now, we assume simple execution or extraction logic here.
            # Simplified for brevity: passing raw query or params if extracted earlier
            params = context.tool_params if context.tool_params else {"query": context.query}
            
            result = tool.execute(**params)
            
            context.metadata["tool_result"] = result
            context.metadata["tool_executed"] = tool_name
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            context.metadata["tool_error"] = str(e)

        return context

class ResponseGenerationStep(PipelineStep):
    """Generates the final answer using the LLM."""
    def __init__(self, llm: LLMProvider, config: Dict[str, Any]):
        self.llm = llm
        self.system_prompt = config.get("system_prompt", "You are a helpful AI.")

    def process(self, context: AgentContext) -> AgentContext:
        # Build Prompt dynamically
        prompt = self._build_prompt(context)
        
        # Generate
        raw_response = self.llm.generate(prompt)
        
        # Simple parsing (can be enhanced with a Cleaner class)
        # Assuming the LLM provider handles raw generation
        context.response = self._clean_response(raw_response)
        context.thinking = self._extract_thinking(raw_response)
        
        return context

    def _build_prompt(self, context: AgentContext) -> str:
        # Construct prompt based on context (RAG, Tool Results, History)
        # This string construction should ideally be in a prompt_template from config
        docs_text = "\n".join([d.get('content', '')[:500] for d in context.rag_docs])
        tool_out = str(context.metadata.get("tool_result", ""))
        
        return (
            f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\nQuery: {context.query}\n"
            f"Context: {docs_text}\n"
            f"Tool Output: {tool_out}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _clean_response(self, text: str) -> str:
        # Basic cleanup logic
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def _extract_thinking(self, text: str) -> str:
        match = re.search(r'<think>(.*?)</think>', text, flags=re.DOTALL)
        return match.group(1).strip() if match else ""