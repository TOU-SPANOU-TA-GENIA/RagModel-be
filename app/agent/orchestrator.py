# app/agent/orchestrator.py
"""
Simplified Agent Orchestrator with Greek language and Qwen3 thinking mode support.
FIXED: 
- max_new_tokens properly passed
- Qwen3 /think mode enabled
- Thinking in English, response in Greek
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
# Greek System Prompt with English Thinking Mode
# ============================================================================

# Qwen3 native thinking: Add /think at end of prompt to enable thinking mode
# The model will output <think>...</think> before the response

GREEK_SYSTEM_PROMPT = """Είσαι ένας εξυπηρετικός βοηθός AI που μιλάει Ελληνικά.

# ΚΡΙΣΙΜΟΙ ΚΑΝΟΝΕΣ

1. **ΓΛΩΣΣΑ:**
   - ΤΕΛΙΚΗ ΑΠΑΝΤΗΣΗ: ΠΑΝΤΑ στα Ελληνικά
   - ΣΚΕΨΗ (<think>): Σκέψου στα Αγγλικά ή Ελληνικά - ΠΟΤΕ Κινέζικα
   - Τεχνικοί όροι (API, ROI, CPU) επιτρέπονται στα Αγγλικά

2. **ΣΚΕΨΗ vs ΑΠΑΝΤΗΣΗ:**
   - Μέσα στο <think>: Think in English. Analyze the query, plan the response structure, consider key points.
   - ΤΕΛΙΚΗ ΑΠΑΝΤΗΣΗ: Σύντομη, ουσιαστική, φυσική - σαν να μιλάς σε φίλο
   
3. **ΜΟΡΦΗ ΑΠΑΝΤΗΣΗΣ:**
   - Μίλα φυσικά, όχι σαν αναφορά ή έκθεση
   - Αποφυγή bullet points εκτός αν ζητηθούν
   - Αποφυγή επαναλήψεων και περιττών εξηγήσεων
   - Μέγιστο 2-3 προτάσεις για απλές ερωτήσεις
   - Μέγιστο 1 παράγραφος για πιο σύνθετα θέματα

4. **EARLY STOPPING:**
   - Όταν η απάντηση είναι ολοκληρωμένη, ΣΤΑΜΑΤΑ
   - ΜΗΝ προσθέτεις περισσότερα αν δεν χρειάζονται
   - Απλές ερωτήσεις = σύντομες απαντήσεις

5. **ΠΟΤΕ ΜΗΝ:**
   - Μην αρχίζεις με "Βεβαίως!", "Φυσικά!", "Καλή ερώτηση!"
   - Μην επαναλαμβάνεις την ερώτηση
   - Μην εξηγείς τι θα κάνεις - απλά κάντο
   - Μην δίνεις περισσότερες πληροφορίες από ότι ζητήθηκαν
   - Μην γράφεις Κινέζικα ούτε στη σκέψη

# ΒΑΣΗ ΓΝΩΣΕΩΝ

Όταν υπάρχουν <knowledge_base> tags:
- Χρησιμοποίησε τις πληροφορίες για να απαντήσεις
- Μην αναφέρεις ότι "βρήκες" ή "είδες" κάτι - απλά απάντα

# ΠΑΡΑΔΕΙΓΜΑΤΑ

Ερώτηση: "1+1;"
<think>Simple math: 1+1=2. Direct answer needed.</think>
2

Ερώτηση: "Τι ώρα είναι;"
<think>User asks for time. I don't have real-time access. Brief response.</think>
Δεν έχω πρόσβαση στην τρέχουσα ώρα.

Ερώτηση: "Πώς δουλεύει το RAG;"
<think>RAG explanation needed. Keep it concise but informative in Greek.</think>
Το RAG ανακτά σχετικά έγγραφα από μια βάση δεδομένων και τα χρησιμοποιεί ως context για να δώσει πιο ακριβείς απαντήσεις.

Ερώτηση: "Γεια σου"
<think>Simple greeting. Respond briefly in Greek.</think>
Γεια! Πώς μπορώ να βοηθήσω;
"""


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
        r'</s>',  # Explicit EOS token
        r'</?knowledge_base>',
        r'</?context>',
        r'</?current_query>',
        r'</?conversation_history>',
        r'</?response_instruction>',
        r'/think',  # Remove the thinking trigger if it appears in output
        r'/no_think',
        r'/(zh|en|el)',  # Language markers
    ]
    
    # Patterns that indicate junk/padding at end
    JUNK_PATTERNS = [
        r'(\s*</s>)+\s*$',  # Repeated </s> at end
        r'(\s*<pad>)+\s*$',  # Repeated <pad> at end
        r'(\s*\[PAD\])+\s*$',
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
        
        # Remove junk patterns at end
        for pattern in cls.JUNK_PATTERNS:
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
    thinking: str = ""
    intent: str = ""
    rag_used: bool = False
    sources: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Pipeline Steps
# ============================================================================

class RAGRetrievalStep(PipelineStep):
    """Retrieves relevant documents from knowledge base."""
    
    def __init__(self, retriever: Optional[Retriever]):
        self.retriever = retriever
    
    @property
    def name(self) -> str:
        return "RAG Retrieval"
    
    def process(self, context: Context) -> Context:
        if not self.retriever:
            return context
        
        try:
            results = self.retriever.retrieve(context.query, k=3)
            
            if results:
                context.metadata["rag_context"] = results
                context.metadata["sources"] = [
                    r.get("metadata", {}).get("source", "unknown")
                    for r in results
                ]
                logger.info(f"RAG retrieved {len(results)} results")
            
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
        
        return context


class PromptBuildStep(PipelineStep):
    """Builds the final prompt for LLM."""
    
    def __init__(self, system_prompt: str = None):
        self.system_prompt = system_prompt or GREEK_SYSTEM_PROMPT
    
    @property
    def name(self) -> str:
        return "Prompt Building"
    
    def process(self, context: Context) -> Context:
        # Build knowledge base section if RAG results exist
        kb_section = ""
        rag_context = context.metadata.get("rag_context", [])
        
        if rag_context:
            kb_parts = []
            for i, result in enumerate(rag_context, 1):
                content = result.get("content", result.get("page_content", ""))
                source = result.get("metadata", {}).get("source", "unknown")
                kb_parts.append(f"[{i}] {source}:\n{content[:500]}")
            
            kb_section = f"\n<knowledge_base>\n{''.join(kb_parts)}\n</knowledge_base>\n"
        
        # Build conversation history
        history_section = ""
        if context.chat_history:
            history_parts = []
            for msg in context.chat_history[-5:]:  # Last 5 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")[:200]
                history_parts.append(f"{role}: {content}")
            
            if history_parts:
                history_section = f"\n<conversation_history>\n{'chr(10)'.join(history_parts)}\n</conversation_history>\n"
        
        # Construct final prompt with English thinking instruction
        prompt = f"""<|im_start|>system
{self.system_prompt}
{kb_section}
{history_section}
<|im_end|>
<|im_start|>user
{context.query}
<|im_end|>
<|im_start|>assistant
"""
        
        context.metadata["prompt"] = prompt
        return context


class LLMGenerationStep(PipelineStep):
    """Generates response using LLM with thinking extraction."""
    
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
                max_tokens=max_tokens,
                max_new_tokens=max_tokens
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
        self.llm = llm_provider
        self.llm_provider = llm_provider  # Alias for compatibility
        self.retriever = retriever
        self.prompt_builder = prompt_builder  # Kept for interface, but we use Greek prompt
        self.enable_thinking = enable_thinking
        self.tools: Dict[str, Tool] = {}
        
        # Build pipeline steps
        self.preprocessing_steps = [
            RAGRetrievalStep(retriever),
            PromptBuildStep(),
        ]
        
        self.generation_step = LLMGenerationStep(llm_provider, enable_thinking)
        
        logger.info(f"✅ SimpleAgentOrchestrator initialized (thinking={enable_thinking})")
    
    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the orchestrator."""
        self.tools[tool.name] = tool
        logger.info(f"Added tool: {tool.name}")
    
    def run_preprocessing(self, context: Context) -> Context:
        """Run preprocessing steps (RAG, prompt building)."""
        for step in self.preprocessing_steps:
            try:
                context = step.process(context)
                logger.debug(f"Completed: {step.name}")
            except Exception as e:
                logger.error(f"Step {step.name} failed: {e}")
        
        return context
    
    def run_generation(self, context: Context) -> Context:
        """Run LLM generation step."""
        return self.generation_step.process(context)
    
    def process(self, query: str, chat_history: List[Dict] = None) -> AgentResponse:
        """Full pipeline: preprocess + generate."""
        context = Context(
            query=query,
            chat_history=chat_history or [],
            metadata={},
            debug_info=[]
        )
        
        # Preprocessing
        context = self.run_preprocessing(context)
        
        # Generation
        context = self.run_generation(context)
        
        # Build response
        return AgentResponse(
            answer=context.metadata.get("llm_response", ""),
            thinking=context.metadata.get("_internal_thinking", ""),
            intent=str(context.metadata.get("intent", "")),
            rag_used=bool(context.metadata.get("rag_context")),
            sources=context.metadata.get("sources", [])
        )
    
    def process_query(
        self, 
        query: str, 
        chat_history: List[Dict[str, str]] = None,
        metadata: Dict[str, Any] = None
    ) -> AgentResponse:
        """Process query with conversation memory support (alias for process)."""
        return self.process(query, chat_history)