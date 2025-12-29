# app/agent/orchestrator.py
"""
Simplified Agent Orchestrator with Greek language and Qwen3 thinking mode support.
FIXED: 
- REVERSE LOOKUP BAN: Prevents guessing Category based on matching Quantities.
- KEYWORD-FIRST LOGIC: Forces identification via text names, ignoring numbers initially.
- GENERIC IMPLEMENTATION: No scenario-specific data involved.
"""

import time
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime
from app.core.interfaces import (
    Context, Intent, Decision, 
    IntentClassifier, DecisionMaker, Tool,
    LLMProvider, Retriever, PromptBuilder,
    Pipeline, PipelineStep, event_bus
)
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# Greek System Prompt with English Thinking Mode (GENERIC STRICT MODE)
# ============================================================================

GREEK_SYSTEM_PROMPT = """Είσαι ένας αυστηρός αλλά εξυπηρετικός Βοηθός AI.

# ΑΠΟΛΥΤΟΙ ΚΑΝΟΝΕΣ ΛΟΓΙΚΗΣ (ANTI-REVERSE LOOKUP)

1. **ΑΠΑΓΟΡΕΥΣΗ ΑΝΤΙΣΤΡΟΦΗΣ ΑΝΑΖΗΤΗΣΗΣ (NO REVERSE LOOKUP):**
   - Μην χρησιμοποιείς τα νούμερα/ποσότητες (Values) του χρήστη για να βρεις ποια Κατηγορία ταιριάζει.
   - **Παράδειγμα Λάθους:** "Ο χρήστης ζητάει 5. Η Κατηγορία Χ δίνει 5. Άρα θέλει την Κατηγορία Χ." (ΛΑΘΟΣ)
   - **Σωστή Λογική:** "Ο χρήστης δεν είπε το όνομα της Κατηγορίας. Άρα η αίτηση είναι ΑΣΑΦΗΣ, ανεξαρτήτως αν το 5 ταιριάζει κάπου."

2. **ΑΡΧΗ ΤΗΣ ΟΝΟΜΑΣΤΙΚΗΣ ΤΑΥΤΟΠΟΙΗΣΗΣ:**
   - Η Κατηγορία (Category) αναγνωρίζεται ΜΟΝΟ από λέξεις-κλειδιά (Keywords) και ΠΟΤΕ από νούμερα.
   - Αν λείπει η λέξη-κλειδί -> ΡΩΤΑ.

3. **ΓΛΩΣΣΑ:**
   - ΤΕΛΙΚΗ ΑΠΑΝΤΗΣΗ: ΠΑΝΤΑ στα Ελληνικά.
   - ΣΚΕΨΗ (<think>): Σκέψου στα Αγγλικά.

4. **ΔΙΑΔΙΚΑΣΙΑ ΣΚΕΨΗΣ (<think>):**
   - Step 1: Scan for TEXT KEYWORDS defining the Category. Ignore numbers.
   - Step 2: If Keyword missing -> STOP. Ignore that numbers might match a rule. Ask Clarification.
   - Step 3: If Keyword found -> Check Prerequisites & Limits.

# ΧΕΙΡΙΣΜΟΣ ΑΣΑΦΕΙΑΣ

ΛΑΘΟΣ: "Ζητάς 5 ημέρες, και βλέπω στον κανονισμό ότι αυτό αντιστοιχεί στον Γάμο. Εγκρίνεται."
ΣΩΣΤΟ: "Ζητάς 5 ημέρες, αλλά δεν διευκρινίζεις τον λόγο/κατηγορία της άδειας. Παρακαλώ διευκρίνισε."
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
# File Server Detection
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
        """
        Build the prompt with fast cross-reference logic.
        """
        # Handle analysis results (from workflow/tool execution)
        analysis_result = context.metadata.get('analysis_result')
        if analysis_result and analysis_result.get('success'):
            data = analysis_result.get('data', {})
            analysis_text = self._format_analysis(
                data.get('anomalies', []), 
                data.get('summary', {})
            )
            prompt = (
                f"<|im_start|>system\n{self.system_prompt}\n<|im_end|>\n"
                f"<|im_start|>user\n{context.query}\n\n"
                f"DATA_ANALYSIS:\n{analysis_text}\n<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            context.metadata["prompt"] = prompt
            return context
        
        # Build knowledge base section from RAG results
        kb_section = self._build_knowledge_base_section(context)
        
        # Build chat history section
        history_section = self._build_history_section(context)
        
        # Get current date for context
        current_date = datetime.now().strftime("%d/%m/%Y")
        
        # COMPACT LOGIC PROTOCOL - STRICT ANTI-REVERSE LOOKUP
        prompt = f"""<|im_start|>system
{self.system_prompt}

# LOGIC PROTOCOL (ANTI-REVERSE LOOKUP):
Date: {current_date}

Perform this EXACT internal check in <think>:

1. **KEYWORD SCAN (CRITICAL):**
   - Does the user text contain an EXPLICIT NAME for the Request Category?
   - **Ignore numbers/dates during this step.**
   - FOUND_NAME: [Yes/No]

2. **AMBIGUITY CHECK:**
   - **IF FOUND_NAME = NO:** - STOP. 
     - DO NOT check if the numbers match any rule in the context.
     - DECISION: ASK CLARIFICATION.
   - **IF FOUND_NAME = YES:** - Proceed to Validation.

3. **VALIDATION (Only if Name is explicit):**
   - Check Prerequisites vs User Data.
   - Check Limits vs Requested Quantity.

4. **DECISION:**
   - Ambiguous -> ASK.
   - Valid -> APPROVE.
   - Invalid -> REJECT.

*Keep thinking concise.*

CONTEXT:
{kb_section}{history_section}<|im_end|>
<|im_start|>user
{context.query}
<|im_end|>
<|im_start|>assistant
"""
        context.metadata["prompt"] = prompt
        return context
    
    def _build_knowledge_base_section(self, context: Context) -> str:
        """
        Build the knowledge base section from RAG results.
        
        Args:
            context: Execution context with rag_context in metadata
            
        Returns:
            Formatted knowledge base section string
        """
        rag_context = context.metadata.get("rag_context", [])
        if not rag_context:
            return ""
        
        kb_parts = []
        for i, result in enumerate(rag_context[:self.MAX_RAG_DOCS], 1):
            content = result.get("content", result.get("page_content", ""))
            source = result.get("metadata", {}).get("fileName", f"src_{i}")
            truncated_content = content[:self.MAX_RAG_CONTENT_PER_DOC]
            kb_parts.append(
                f"REFERENCE_SOURCE_{i} (File: {source}):\n{truncated_content}\n---"
            )
        
        return f"\n<knowledge_base>\n{''.join(kb_parts)}\n</knowledge_base>\n"
    
    def _build_history_section(self, context: Context) -> str:
        """
        Build the chat history section.
        
        Args:
            context: Execution context with chat_history
            
        Returns:
            Formatted history section string
        """
        if not context.chat_history:
            return ""
        
        recent_messages = context.chat_history[-self.MAX_HISTORY_MESSAGES:]
        history_parts = []
        for msg in recent_messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')[:self.MAX_MESSAGE_LENGTH]
            history_parts.append(f"{role}: {content}")
        
        return f"\n<history>\n{chr(10).join(history_parts)}\n</history>\n"
    
    def _format_analysis(self, anomalies: List[Dict], summary: Dict) -> str:
        """
        Format analysis results for inclusion in prompt.
        
        Args:
            anomalies: List of detected anomalies
            summary: Summary dictionary
            
        Returns:
            Formatted analysis text
        """
        lines = []
        if summary:
            lines.append(f"Summary: {summary.get('total_anomalies', 0)} detected")
        
        for anomaly in anomalies[:5]:
            severity = anomaly.get('severity', 'LOW')
            description = anomaly.get('description', '')[:100]
            lines.append(f"- [{severity}] {description}")
        
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
            
            # Note: We let the model run constraints, but system prompt enforces brevity
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