# app/llm/thinking_aware_generation.py
"""
Thinking-Aware LLM Generation Step.
Combines internal reasoning with response generation in a single pipeline step.
"""

from typing import Optional
from app.core.interfaces import PipelineStep, Context, LLMProvider
from app.llm.enhanced_response_cleaner import clean_response
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class ThinkingAwareLLMGenerationStep(PipelineStep):
    """
    Enhanced LLM generation that includes an internal thinking phase.
    
    Flow:
    1. Generate internal thinking/reasoning (not shown to user)
    2. Use thinking to guide final response generation
    3. Clean response to remove any leaked thinking artifacts
    
    The thinking helps the model:
    - Understand what the user really needs
    - Choose appropriate tone and structure
    - Identify key points to address
    """
    
    def __init__(
        self, 
        llm_provider: LLMProvider, 
        enable_thinking: bool = True,
        thinking_tokens: int = 120,
        thinking_temperature: float = 0.3
    ):
        self.llm = llm_provider
        self.enable_thinking = enable_thinking
        self.thinking_tokens = thinking_tokens
        self.thinking_temperature = thinking_temperature
    
    @property
    def name(self) -> str:
        return "LLM Generation"
    
    def process(self, context: Context) -> Context:
        """Generate response with optional thinking phase."""
        prompt = context.metadata.get("prompt", context.query)
        
        try:
            if self.enable_thinking:
                # Phase 1: Internal reasoning
                thinking = self._generate_thinking(context)
                context.metadata["_internal_thinking"] = thinking
                logger.debug(f"Generated thinking: {thinking[:100]}...")
                
                # Phase 2: Response guided by thinking
                enhanced_prompt = self._build_enhanced_prompt(prompt, thinking)
                raw_response = self.llm.generate(enhanced_prompt)
            else:
                raw_response = self.llm.generate(prompt)
            
            # Clean any artifacts
            cleaned_response = clean_response(raw_response)
            context.metadata["llm_response"] = cleaned_response
            
            logger.info(f"Generated response: {len(cleaned_response)} chars")
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            context.metadata["llm_response"] = "I apologize, but I encountered an error."
        
        return context
    
    def _generate_thinking(self, context: Context) -> str:
        """Generate internal reasoning about how to respond."""
        # Gather context signals
        intent = context.metadata.get("intent", "unknown")
        if hasattr(intent, "value"):
            intent = intent.value
        
        has_rag = bool(context.metadata.get("rag_context"))
        has_tool = bool(context.metadata.get("tool_result"))
        history_len = len(context.chat_history) if context.chat_history else 0
        
        thinking_prompt = f"""<internal_analysis>
User message: "{context.query}"

Context signals:
- Intent: {intent}
- Has knowledge context: {has_rag}
- Has tool result: {has_tool}
- Conversation history: {history_len} messages

Briefly analyze:
1. What does the user need?
2. What response style fits?
3. Key points to address?

Keep analysis to 2-3 sentences.
</internal_analysis>

Analysis:"""
        
        thinking = self.llm.generate(
            thinking_prompt,
            max_tokens=self.thinking_tokens,
            temperature=self.thinking_temperature
        )
        
        return thinking.strip()
    
    def _build_enhanced_prompt(self, base_prompt: str, thinking: str) -> str:
        """Inject thinking as internal guidance into the prompt."""
        guidance = f"""<response_guidance>
Internal analysis (DO NOT include in response):
{thinking}

Guidelines:
- Respond directly to the user
- Match the appropriate tone identified above
- Address the key points naturally
- Keep response clean and professional
</response_guidance>

"""
        # Insert guidance at the start of the prompt
        return guidance + base_prompt