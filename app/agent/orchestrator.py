# app/agent/orchestrator.py
"""
Simplified Agent Orchestrator with Greek language and Qwen3 thinking mode support.
FIXED: max_new_tokens properly passed, Qwen3 /think mode enabled.
"""

import time
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional

from app.core.interfaces import (
    Context, Intent, Decision, 
    IntentClassifier, DecisionMaker, Tool,
    LLMProvider, Retriever, PromptBuilder,
    Pipeline, PipelineStep, event_bus
)
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# Greek System Prompt with Qwen3 Thinking Mode
# ============================================================================

# Qwen3 native thinking: Add /think at end of prompt to enable thinking mode
# The model will output <think>...</think> before the response

GREEK_SYSTEM_PROMPT = """Είσαι ένας εξυπηρετικός βοηθός AI που μιλάει Ελληνικά.

ΚΡΙΣΙΜΟΣ ΚΑΝΟΝΑΣ: ΑΠΑΝΤΑ ΠΑΝΤΑ ΣΤΑ ΕΛΛΗΝΙΚΑ. Η τελική σου απάντηση ΠΡΕΠΕΙ να είναι στα Ελληνικά.

# ΒΑΣΙΚΕΣ ΑΡΧΕΣ

1. **Αμεσότητα:** Απάντα άμεσα και πλήρως.
2. **Πληρότητα:** Δώσε ολοκληρωμένες απαντήσεις. Μην κόβεις τη σκέψη σου.
3. **Ακρίβεια:** Χρησιμοποίησε τις πληροφορίες από το context.

# ΒΑΣΗ ΓΝΩΣΕΩΝ

Όταν υπάρχουν πληροφορίες σε <knowledge_base> tags:
- ΧΡΗΣΙΜΟΠΟΙΗΣΕ αυτές τις πληροφορίες
- ΑΝΑΛΥΣΕ τα δεδομένα που σου δίνονται
- ΣΥΝΘΕΣΕ τις πληροφορίες για να απαντήσεις

# ΓΛΩΣΣΑ

- Απάντα ΜΟΝΟ στα Ελληνικά
- Τεχνικοί όροι (ROI, API) μπορούν να παραμείνουν στα Αγγλικά
- Ονόματα αρχείων διατηρούνται ως έχουν"""


# ============================================================================
# Response Cleaner
# ============================================================================

class ResponseCleaner:
    """Cleans LLM responses by removing thinking blocks and artifacts."""
    
    # Patterns for thinking blocks - Qwen3 uses <think>...</think>
    THINKING_PATTERNS = [
        (r'<think>.*?</think>', re.DOTALL | re.IGNORECASE),
        (r'<thinking>.*?</thinking>', re.DOTALL | re.IGNORECASE),
        (r'<σκέψη>.*?</σκέψη>', re.DOTALL | re.IGNORECASE),
    ]
    
    # Tags to remove from output
    TAG_PATTERNS = [
        r'</?think>',
        r'</?thinking>',
        r'</?response>',
        r'<\|(?:system|user|assistant|end|im_start|im_end)\|>',
        r'</?s>',
        r'</?knowledge_base>',
        r'</?context>',
        r'</?current_query>',
        r'</?conversation_history>',
        r'</?response_instruction>',
        r'/think',  # Remove the thinking trigger if it appears in output
        r'/no_think',
    ]
    
    @classmethod
    def clean(cls, response: str) -> str:
        """Clean response by removing all thinking artifacts."""
        if not response:
            return response
        
        cleaned = response
        
        # Remove thinking blocks
        for pattern, flags in cls.THINKING_PATTERNS:
            cleaned = re.sub(pattern, '', cleaned, flags=flags)
        
        # Remove stray tags
        for pattern in cls.TAG_PATTERNS:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Normalize whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r' +', ' ', cleaned)
        
        return cleaned.strip()
    
    @classmethod
    def extract_thinking(cls, response: str) -> tuple:
        """Extract thinking and clean response separately."""
        thinking = ""
        
        # Look for Qwen3's native <think> block
        match = re.search(r'<think>(.*?)</think>', response, re.DOTALL | re.IGNORECASE)
        if match:
            thinking = match.group(1).strip()
        
        clean_response = cls.clean(response)
        return thinking, clean_response


def clean_response(response: str) -> str:
    """Convenience function for response cleaning."""
    return ResponseCleaner.clean(response)


# ============================================================================
# Response Model
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
    internal_thinking: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Pipeline Steps
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
            logger.info("Retrieving relevant documents...")
            try:
                sources = self.retriever.retrieve(context.query, k=5)
                context.metadata["rag_sources"] = sources
                context.metadata["rag_context"] = self._format_context(sources)
                context.add_debug(f"Retrieved {len(sources)} documents")
                logger.info(f"Retrieved {len(sources)} documents")
            except Exception as e:
                logger.error(f"RAG retrieval failed: {e}")
                context.metadata["rag_sources"] = []
                context.metadata["rag_context"] = ""
        
        return context
    
    def _format_context(self, sources: List[Dict]) -> str:
        if not sources:
            return ""
        
        context_parts = []
        for i, source in enumerate(sources, 1):
            content = source.get("content", "")
            metadata = source.get("metadata", {})
            source_name = metadata.get("source", "Unknown")
            context_parts.append(f"[Έγγραφο {i}: {source_name}]\n{content}")
        return "\n\n---\n\n".join(context_parts)


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
                    logger.info(f"Executing tool: {decision.tool_name}")
                    result = tool.execute(**decision.tool_params)
                    context.metadata["tool_result"] = result
                    context.metadata["tool_used"] = decision.tool_name
                except Exception as e:
                    logger.error(f"Tool execution failed: {e}")
                    context.metadata["tool_result"] = {"success": False, "error": str(e)}
        
        return context


class GreekPromptBuildingStep(PipelineStep):
    """Step 5: Build Greek prompt with Qwen3 thinking mode."""
    
    def __init__(self, prompt_builder: Optional[PromptBuilder] = None, enable_thinking: bool = True):
        self.prompt_builder = prompt_builder
        self.enable_thinking = enable_thinking
    
    @property
    def name(self) -> str:
        return "Prompt Building"
    
    def process(self, context: Context) -> Context:
        prompt = self._build_greek_prompt(context)
        context.metadata["prompt"] = prompt
        logger.debug(f"Built Greek prompt: {len(prompt)} characters")
        return context
    
    def _build_greek_prompt(self, context: Context) -> str:
        """Build Greek prompt with Qwen3 thinking mode."""
        sections = []
        
        # System instruction
        sections.append(GREEK_SYSTEM_PROMPT)
        
        # Chat history
        if context.chat_history:
            history_parts = []
            for msg in context.chat_history[-10:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    history_parts.append(f"Χρήστης: {content}")
                else:
                    history_parts.append(f"Βοηθός: {content}")
            if history_parts:
                sections.append("<conversation_history>\n" + "\n".join(history_parts) + "\n</conversation_history>")
        
        # RAG context - CRITICAL
        rag_context = context.metadata.get("rag_context", "")
        if rag_context:
            sections.append(f"<knowledge_base>\nΠληροφορίες από τη βάση γνώσεων:\n\n{rag_context}\n</knowledge_base>")
        
        # Tool result
        tool_result = context.metadata.get("tool_result")
        if tool_result and tool_result.get("success"):
            data = tool_result.get("data", {})
            if "content" in data:
                sections.append(f"<tool_result>\nΑποτέλεσμα ({data.get('file_name', 'αρχείο')}):\n\n{data['content']}\n</tool_result>")
        
        # Current query
        sections.append(f"Χρήστης: {context.query}")
        
        # Response instruction - FORCE GREEK + COMPLETE RESPONSE
        sections.append("\nΒοηθός (ΑΠΑΝΤΗΣΕ ΣΤΑ ΕΛΛΗΝΙΚΑ, ΠΛΗΡΗΣ ΑΠΑΝΤΗΣΗ):")
        
        # Enable Qwen3 thinking mode with /think suffix
        if self.enable_thinking:
            sections.append("/think")
        
        return "\n\n".join(sections)


class LLMGenerationStep(PipelineStep):
    """Step 6: Generate response using LLM with proper max_new_tokens."""
    
    def __init__(self, llm_provider: LLMProvider, enable_thinking: bool = True):
        self.llm = llm_provider
        self.enable_thinking = enable_thinking
    
    @property
    def name(self) -> str:
        return "LLM Generation"
    
    def process(self, context: Context) -> Context:
        prompt = context.metadata.get("prompt", context.query)
        
        try:
            # Get max_new_tokens from config
            from app.config import LLM
            max_tokens = LLM.max_new_tokens
            
            logger.info(f"Generating with max_new_tokens={max_tokens}")
            
            # Generate response - pass max_tokens correctly
            raw_response = self.llm.generate(
                prompt,
                max_tokens=max_tokens,  # Key parameter name for fast_providers.py
                max_new_tokens=max_tokens  # Alternative name some providers use
            )
            
            # Extract thinking (from Qwen3's <think> block) and clean response
            thinking, clean_answer = ResponseCleaner.extract_thinking(raw_response)
            
            # Store results
            context.metadata["raw_response"] = raw_response
            context.metadata["llm_response"] = clean_answer
            
            if thinking:
                context.metadata["_internal_thinking"] = thinking
                logger.info(f"Extracted thinking: {len(thinking)} chars")
            
            logger.info(f"Generated response: {len(clean_answer)} chars")
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            import traceback
            traceback.print_exc()
            context.metadata["llm_response"] = "Συγγνώμη, παρουσιάστηκε σφάλμα."
        
        return context


# Backwards compatibility
ThinkingAwareLLMGenerationStep = LLMGenerationStep


# ============================================================================
# Main Orchestrator
# ============================================================================

class SimpleAgentOrchestrator:
    """Agent orchestrator with Greek language and Qwen3 thinking support."""
    
    def __init__(
        self,
        intent_classifier: IntentClassifier,
        decision_maker: DecisionMaker,
        llm_provider: LLMProvider,
        retriever: Optional[Retriever] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        enable_thinking: bool = True
    ):
        self.intent_classifier = intent_classifier
        self.decision_maker = decision_maker
        self.llm_provider = llm_provider
        self.retriever = retriever
        self.prompt_builder = prompt_builder  # Ignored - we use Greek prompt
        self.enable_thinking = enable_thinking
        self.tools: Dict[str, Tool] = {}
        
        self.pipeline = self._build_pipeline()
        self._setup_debug_handlers()
        
        logger.info(f"SimpleAgentOrchestrator initialized (thinking={enable_thinking}, language=Greek)")
    
    def _build_pipeline(self) -> Pipeline:
        """Build the processing pipeline."""
        pipeline = Pipeline()
        
        pipeline.add_step(IntentClassificationStep(self.intent_classifier))
        pipeline.add_step(DecisionMakingStep(self.decision_maker))
        
        if self.retriever:
            pipeline.add_step(RAGRetrievalStep(self.retriever))
        
        pipeline.add_step(ToolExecutionStep(self.tools))
        pipeline.add_step(GreekPromptBuildingStep(None, self.enable_thinking))
        pipeline.add_step(LLMGenerationStep(self.llm_provider, self.enable_thinking))
        
        return pipeline
    
    def _setup_debug_handlers(self):
        def log_event(data):
            logger.debug(f"Event: {data}")
        
        event_bus.on("intent_classified", log_event)
        event_bus.on("decision_made", log_event)
    
    def add_tool(self, tool: Tool) -> None:
        self.tools[tool.name] = tool
        logger.info(f"Added tool: {tool.name}")
    
    def process_query(
        self, 
        query: str, 
        chat_history: List[Dict[str, str]] = None,
        metadata: Dict[str, Any] = None
    ) -> AgentResponse:
        """Process query with conversation memory support."""
        start_time = time.time()
        
        context = Context(
            query=query,
            chat_history=chat_history or [],
            metadata=metadata or {},
            debug_info=[]
        )
        
        logger.info(f"Processing query: {query[:100]}...")
        
        try:
            context = self.pipeline.process(context)
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            import traceback
            traceback.print_exc()
            return AgentResponse(
                answer="Παρουσιάστηκε σφάλμα κατά την επεξεργασία.",
                debug_info=[str(e)],
                execution_time=time.time() - start_time
            )
        
        response = self._build_response(context)
        response.execution_time = time.time() - start_time
        
        self._store_assistant_response(context, response.answer)
        
        logger.info(f"Query processed in {response.execution_time:.2f}s")
        
        return response
    
    def run_preprocessing(self, context: Context) -> Context:
        """Run preprocessing steps only (for streaming)."""
        for step in self.pipeline.steps:
            if step.name == "LLM Generation":
                continue
            context = step.process(context)
        return context
    
    def _store_assistant_response(self, context: Context, answer: str):
        try:
            from app.core.conversation_memory import conversation_memory
            session_id = context.metadata.get("session_id")
            if session_id:
                session = conversation_memory.get_session(session_id)
                if session:
                    session.add_message("assistant", answer)
        except Exception as e:
            logger.debug(f"Could not store assistant response: {e}")
    
    def _build_response(self, context: Context) -> AgentResponse:
        intent = context.metadata.get("intent", Intent.UNKNOWN)
        if hasattr(intent, "value"):
            intent = intent.value
        
        return AgentResponse(
            answer=context.metadata.get("llm_response", ""),
            sources=context.metadata.get("rag_sources", []),
            tool_used=context.metadata.get("tool_used"),
            tool_result=context.metadata.get("tool_result"),
            intent=intent,
            debug_info=context.debug_info,
            internal_thinking=context.metadata.get("_internal_thinking")
        )