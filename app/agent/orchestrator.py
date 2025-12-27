# app/agent/orchestrator.py
"""
Simplified Agent Orchestrator with Greek language and Qwen3 thinking mode support.
FIXED: 
- max_new_tokens properly passed
- Qwen3 /think mode enabled
- Thinking in English, response in Greek
- FILE SERVER DETECTION before RAG
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
    
    THINKING_PATTERNS = [
        (r'<think>.*?</think>', re.DOTALL | re.IGNORECASE),
        (r'<think>.*$', re.DOTALL | re.IGNORECASE), # Handle unclosed tags
        (r'<thinking>.*?</thinking>', re.DOTALL | re.IGNORECASE),
        (r'<σκέψη>.*?</σκέψη>', re.DOTALL | re.IGNORECASE),
    ]
    
    TAG_PATTERNS = [
        r'</?think>',
        r'</?thinking>',
        r'</?response>',
        r'<\|(?:system|user|assistant|end|im_start|im_end)\|>',
        r'</?s>',
        r'</s>',
        r'</?knowledge_base>',
        r'</?context>',
        r'</?kb>',
        r'</?current_query>',
        r'</?conversation_history>',
        r'</?response_instruction>',
        r'/think',
    ]
    
    @classmethod
    def clean(cls, response: str) -> str:
        if not response:
            return response
        
        cleaned = response
        # 1. Remove blocks with content
        for pattern, flags in cls.THINKING_PATTERNS:
            cleaned = re.sub(pattern, '', cleaned, flags=flags)
        
        # 2. Strip remaining lone tags
        for pattern in cls.TAG_PATTERNS:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()
    
    @classmethod
    def extract_thinking(cls, response: str) -> tuple:
        thinking = ""
        # Match from opening tag to closing tag, OR to end of string if unclosed
        match = re.search(r'<think>(.*?)(?:</think>|$)', response, re.DOTALL | re.IGNORECASE)
        if match:
            thinking = match.group(1).strip()
        
        clean_response = cls.clean(response)
        return thinking, clean_response

def clean_response(response: str) -> str:
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
    tool_used: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# File Server Detection (NEW)
# ============================================================================

def is_file_server_query(query: str) -> bool:
    """Check if query references file server folders."""
    query_lower = query.lower()
    patterns = [
        r'(?:από|μέσα σ?τ[οα]ν?|στ[οα]ν?)\s+φάκελο',
        r'φάκελο[ςσ]?\s+\S+',
        r'(?:εντόπισε|βρες|έλεγξε|ανάλυσε).*(?:φάκελο|αρχεία)',
        r'(?:ανωμαλ[ιί][εέ]ς?|αποκλ[ιί]σ[εέ]ις?).*(?:φάκελο|αρχεί)',
    ]
    return any(re.search(p, query_lower) for p in patterns)


def extract_folder_from_query(query: str) -> Optional[str]:
    """Extract folder name from query."""
    patterns = [
        r'(?:από|μέσα σ?τ[οα]ν?|στ[οα]ν?)\s+φάκελο\s+([^,\.!;]+)',
        r'φάκελο[ςσ]?\s+([^,\.!;]+?)(?:\s*,|\s+(?:εντόπισε|βρες|έλεγξε))',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            return match.group(1).strip()
    return None


def extract_action_from_query(query: str) -> str:
    """Extract action type from query."""
    query_lower = query.lower()
    if re.search(r'(?:δείξ|εμφάνισ|list|show)', query_lower):
        return 'browse'
    if re.search(r'(?:ψάξε|αναζήτησε|search)', query_lower):
        return 'search'
    return 'analyze'


# ============================================================================
# Pipeline Steps
# ============================================================================

class FileServerStep(PipelineStep):
    """Check for file server queries and execute tool if needed."""
    
    def __init__(self, tools: Dict[str, Tool]):
        self.tools = tools
    
    @property
    def name(self) -> str:
        return "File Server Check"
    
    def process(self, context: Context) -> Context:
        query = context.query
        
        # Check if this is a file server query
        if not is_file_server_query(query):
            return context
        
        # Check if file_server tool is available
        file_server_tool = self.tools.get('file_server')
        if not file_server_tool:
            logger.warning("File server query detected but tool not available")
            return context
        
        # Extract folder and action
        folder = extract_folder_from_query(query)
        action = extract_action_from_query(query)
        
        logger.info(f"📁 File server query detected: folder='{folder}', action='{action}'")
        
        if not folder:
            logger.warning("Could not extract folder name from query")
            return context
        
        # Execute file server tool
        try:
            result = file_server_tool.execute(
                folder=folder,
                action=action,
                query=query
            )
            
            if result.get('success'):
                context.metadata['file_server_result'] = result
                context.metadata['tool_used'] = 'file_server'
                
                # Get file paths for analysis
                file_paths = result.get('data', {}).get('file_paths', [])
                
                if file_paths and action == 'analyze':
                    # Chain to logistics analyzer
                    context.metadata['files_for_analysis'] = file_paths
                    logger.info(f"📁 Got {len(file_paths)} files for analysis")
                    
                    # Try to run logistics analysis
                    logistics_tool = self.tools.get('detect_logistics_anomalies')
                    if logistics_tool:
                        logger.info("🔍 Running logistics anomaly detection...")
                        analysis_result = logistics_tool.execute(file_paths=file_paths)
                        if analysis_result.get('success'):
                            context.metadata['analysis_result'] = analysis_result
                            context.metadata['tool_used'] = 'file_server + logistics'
                            logger.info("✅ Logistics analysis complete")
            else:
                logger.warning(f"File server tool failed: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"File server execution failed: {e}")
        
        return context


class RAGRetrievalStep(PipelineStep):
    """Retrieves relevant documents from knowledge base."""
    
    def __init__(self, retriever: Optional[Retriever]):
        self.retriever = retriever
    
    @property
    def name(self) -> str:
        return "RAG Retrieval"
    
    def process(self, context: Context) -> Context:
        # Skip RAG if file server already handled this
        if context.metadata.get('tool_used'):
            logger.info("Skipping RAG - tool already handled query")
            return context
        
        if not self.retriever:
            return context
        
        try:
            results = self.retriever.retrieve(context.query, k=3)
            
            if results:
                context.metadata["rag_context"] = results
                for i, r in enumerate(results):
                    source = r.get("metadata", {}).get("source", "unknown")
                    content_preview = r.get("content", "")[:100]
                    logger.info(f"RAG doc {i+1}: {source} - {content_preview}...")
                context.metadata["sources"] = [
                    r.get("metadata", {}).get("source", "unknown")
                    for r in results
                ]
                logger.info(f"RAG retrieved {len(results)} results")
            
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
        
        return context

class PromptBuildStep(PipelineStep):
    """
    Generalized Logic Engine.
    Enforces a strict Variable-to-Constraint validation hierarchy.
    """
    
    MAX_RAG_CONTENT_PER_DOC = 1200 
    MAX_RAG_DOCS = 5              
    MAX_HISTORY_MESSAGES = 3
    MAX_MESSAGE_LENGTH = 150
    MAX_SYSTEM_PROMPT = 1000
    
    def __init__(self, system_prompt: str = None):
        base_prompt = system_prompt or GREEK_SYSTEM_PROMPT
        self.system_prompt = base_prompt[:self.MAX_SYSTEM_PROMPT]
    
    @property
    def name(self) -> str:
        return "Prompt Building"
    
    def process(self, context: Context) -> Context:
        analysis_result = context.metadata.get('analysis_result')
        if analysis_result and analysis_result.get('success'):
            data = analysis_result.get('data', {})
            analysis_text = self._format_analysis(data.get('anomalies', []), data.get('summary', {}))
            prompt = f"<|im_start|>system\n{self.system_prompt}\n<|im_end|>\n<|im_start|>user\n{context.query}\n\nDATA_ANALYSIS:\n{analysis_text}\n<|im_end|>\n<|im_start|>assistant\n"
            context.metadata["prompt"] = prompt
            return context
        
        kb_section = ""
        rag_context = context.metadata.get("rag_context", [])
        if rag_context:
            kb_parts = []
            for i, result in enumerate(rag_context[:self.MAX_RAG_DOCS], 1):
                content = result.get("content", result.get("page_content", ""))
                source = result.get("metadata", {}).get("fileName", f"src_{i}")
                kb_parts.append(f"REFERENCE_SOURCE_{i} (File: {source}):\n{content[:self.MAX_RAG_CONTENT_PER_DOC]}\n---")
            kb_section = f"\n<knowledge_base>\n{''.join(kb_parts)}\n</knowledge_base>\n"
        
        history_section = ""
        if context.chat_history:
            history_parts = [f"{m.get('role')}: {m.get('content')[:self.MAX_MESSAGE_LENGTH]}" for m in context.chat_history[-self.MAX_HISTORY_MESSAGES:]]
            history_section = f"\n<history>\n{chr(10).join(history_parts)}\n</history>\n"
        
        # GENERALIZED LOGIC PROTOCOL - Scenario Agnostic
        prompt = f"""<|im_start|>system
{self.system_prompt}

# ΠΡΩΤΟΚΟΛΛΟ ΛΟΓΙΚΗΣ ΕΠΕΞΕΡΓΑΣΙΑΣ:
1. **Fact Harvesting:** Εντόπισε όλες τις τιμές (ημερομηνίες, υπόλοιπα, status) που αφορούν την οντότητα του χρήστη στην <knowledge_base>.
2. **Global Constraints:** Εντόπισε κανόνες που λειτουργούν ως "πύλες" (π.χ. ελάχιστος χρόνος, βασικό status) και πρέπει να πληρούνται ΠΡΙΝ εξεταστεί οποιαδήποτε επιμέρους κατηγορία.
3. **Variable Validation:** Σύγκρινε τα δεδομένα του Βήματος 1 με τους κανόνες του Βήματος 2. Αν υπάρχει απόκλιση, η απάντηση είναι αρνητική και εξηγεί το κώλυμα.
4. **Anti-Hallucination Policy:** Αν μια ποσοτική τιμή (π.χ. "5 ημέρες", "100 ευρώ") ταιριάζει με μια κατηγορία αλλά ο χρήστης δεν την έχει ορίσει ρητά, ΑΠΑΓΟΡΕΥΕΤΑΙ να την επιλέξεις. Ζήτησε διευκρίνιση για την αιτιολογία.

# FORMAT:
- Ξεκίνα με <think> για την ανάλυση και κλείσε με </think>.
- Η τελική απάντηση στα Ελληνικά.

CONTEXT:
{kb_section}{history_section}<|im_end|>
<|im_start|>user
{context.query}
<|im_end|>
<|im_start|>assistant
"""
        context.metadata["prompt"] = prompt
        return context
    
    def _format_analysis(self, anomalies: List[Dict], summary: Dict) -> str:
        lines = []
        if summary: lines.append(f"Summary: {summary.get('total_anomalies', 0)} detected")
        for a in anomalies[:5]:
            lines.append(f"- [{a.get('severity', 'LOW')}] {a.get('description', '')[:100]}")
        return "\n".join(lines)
 
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
            from app.config import LLM
            max_tokens = LLM.max_new_tokens
            
            logger.info(f"Generating with max_new_tokens={max_tokens}")
            
            raw_response = self.llm.generate(
                prompt,
                max_tokens=max_tokens,
                max_new_tokens=max_tokens
            )
            
            thinking, clean_answer = ResponseCleaner.extract_thinking(raw_response)
            
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
        self.llm_provider = llm_provider
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.enable_thinking = enable_thinking
        self.tools: Dict[str, Tool] = {}
        
        # Pipeline steps will be built after tools are added
        self._pipeline_built = False
        
        logger.info(f"✅ SimpleAgentOrchestrator initialized (thinking={enable_thinking})")
    
    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the orchestrator."""
        self.tools[tool.name] = tool
        self._pipeline_built = False  # Need to rebuild pipeline
        logger.info(f"Added tool: {tool.name}")
    
    def _build_pipeline(self):
        """Build pipeline steps with current tools."""
        self.preprocessing_steps = [
            FileServerStep(self.tools),  # NEW: Check file server FIRST
            RAGRetrievalStep(self.retriever),
            PromptBuildStep(),
        ]
        self.generation_step = LLMGenerationStep(self.llm, self.enable_thinking)
        self._pipeline_built = True
    
    def run_preprocessing(self, context: Context) -> Context:
        """Run preprocessing steps (File Server, RAG, prompt building)."""
        if not self._pipeline_built:
            self._build_pipeline()
        
        for step in self.preprocessing_steps:
            try:
                context = step.process(context)
                logger.debug(f"Completed: {step.name}")
            except Exception as e:
                logger.error(f"Step {step.name} failed: {e}")
        
        return context
    
    def run_generation(self, context: Context) -> Context:
        """Run LLM generation step."""
        if not self._pipeline_built:
            self._build_pipeline()
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
            sources=context.metadata.get("sources", []),
            tool_used=context.metadata.get("tool_used", "")
        )
    
    def process_query(
        self, 
        query: str, 
        chat_history: List[Dict[str, str]] = None,
        metadata: Dict[str, Any] = None
    ) -> AgentResponse:
        """Process query with conversation memory support (alias for process)."""
        return self.process(query, chat_history)