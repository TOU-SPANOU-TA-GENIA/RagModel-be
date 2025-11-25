# app/agent/classifiers.py
"""
Concrete implementations of Intent Classification and Decision Making.
These are simplified, focused components that replace the complex logic in your original Agent class.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re

from app.core.interfaces import (
    Intent, Context, Decision,
    IntentClassifier, DecisionMaker, Tool
)
from app.logger import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# Intent Classifier Implementations
# ============================================================================

class RuleBasedIntentClassifier(IntentClassifier):
    """
    Simple rule-based intent classifier.
    Easy to understand and debug - no black box ML here!
    """
    
    def __init__(self):
        # Define patterns for each intent
        self.patterns = {
            Intent.ACTION: {
                'keywords': [
                    'read', 'open', 'show', 'display', 'view', 'check', 'see',
                    'get', 'fetch', 'look at', 'look in', "what's in",
                    'execute', 'run', 'create', 'write', 'update', 'delete',
                    'cat', 'show me', 'display the', 'read the',
                    'build', 'generate', 'make', 'produce',  # ADD THESE
                    'document', 'powerpoint', 'word', 'pdf', 'presentation'  # ADD THESE
                ],
                'patterns': [
                    r'(read|show|open|view)\s+(the\s+)?file',
                    r'file\s+at\s+[/\w]+',
                    r'(content|contents)\s+of',
                    r'(build|create|generate|make)\s+(a|an)?\s+(word|powerpoint|pdf|document|presentation)',  # ADD THIS
                    r'(build|create|generate)\s+.*\s+(document|powerpoint|presentation|word|pdf)',  # ADD THIS
                ],
                'weight': 1.5
            },
            Intent.QUESTION: {
                'keywords': [
                    'what', 'how', 'why', 'when', 'where', 'who', 'which',
                    'explain', 'describe', 'tell me', 'can you',
                    'do you know', 'what does', 'information about'
                ],
                'patterns': [
                    r'^(what|how|why|when|where|who)',
                    r'\?$',
                    r'tell\s+me\s+about',
                    r'explain\s+\w+'
                ],
                'weight': 1.0
            },
            Intent.CONVERSATION: {
                'keywords': [
                    'hello', 'hi', 'hey', 'thanks', 'thank you',
                    'goodbye', 'bye', 'ok', 'okay', 'good'
                ],
                'patterns': [
                    r'^(hello|hi|hey|thanks|bye)$',
                    r'^(good|ok|okay|alright)$'
                ],
                'weight': 0.5  # Lower priority
            }
        }
    
    def classify(self, context: Context) -> Intent:
        """Classify intent based on rules."""
        query = context.query.lower().strip()
        scores = {intent: 0.0 for intent in Intent}
        
        # Check each intent's patterns
        for intent, config in self.patterns.items():
            score = 0.0
            
            # Check keywords
            for keyword in config['keywords']:
                if keyword in query:
                    score += 1.0
                    context.add_debug(f"Found keyword '{keyword}' for {intent.value}")
            
            # Check regex patterns
            for pattern in config['patterns']:
                if re.search(pattern, query):
                    score += 2.0  # Patterns are more specific
                    context.add_debug(f"Matched pattern for {intent.value}")
            
            # Apply weight
            scores[intent] = score * config['weight']
        
        # Find best match
        best_intent = Intent.UNKNOWN
        best_score = 0.0
        
        for intent, score in scores.items():
            if score > best_score:
                best_score = score
                best_intent = intent
        
        # Default to QUESTION if uncertain
        if best_score < 0.5:
            best_intent = Intent.QUESTION
        
        context.add_debug(f"Intent scores: {scores}")
        logger.info(f"Classified '{query[:50]}...' as {best_intent.value} (score: {best_score})")
        
        return best_intent


class HybridIntentClassifier(IntentClassifier):
    """
    More sophisticated classifier that can use both rules and LLM.
    Demonstrates how to extend the simple classifier.
    """
    
    def __init__(self, rule_classifier: RuleBasedIntentClassifier = None, 
                 use_llm: bool = False):
        self.rule_classifier = rule_classifier or RuleBasedIntentClassifier()
        self.use_llm = use_llm
    
    def classify(self, context: Context) -> Intent:
        """First try rules, then optionally use LLM for uncertain cases."""
        # Start with rule-based classification
        intent = self.rule_classifier.classify(context)
        
        # If uncertain and LLM enabled, ask LLM
        if intent == Intent.UNKNOWN and self.use_llm:
            intent = self._classify_with_llm(context)
        
        return intent
    
    def _classify_with_llm(self, context: Context) -> Intent:
        """Use LLM for classification (placeholder for now)."""
        # This would use the LLM to classify
        # For now, default to QUESTION
        return Intent.QUESTION


# ============================================================================
# Decision Maker Implementations
# ============================================================================

# app/agent/classifiers.py - REPLACE SimpleDecisionMaker class

class SimpleDecisionMaker(DecisionMaker):
    """
    Simple decision maker that decides based on intent and available tools.
    FIXED: Properly extracts file references and uses correct parameter names.
    """
    
    def __init__(self, tools: Optional[Dict[str, Tool]] = None):
        self.tools = tools or {}
    
    def decide(self, context: Context, intent: Intent) -> Decision:
        """Make a decision based on intent."""
        
        decision = Decision(
            intent=intent,
            confidence=1.0,
            use_tool=False,
            tool_name=None,
            tool_params=None,
            use_rag=True,
            reasoning=""
        )
        
        if intent == Intent.CONVERSATION:
            decision.use_rag = False
            decision.reasoning = "Casual conversation, no tools or RAG needed"
            
        elif intent == Intent.QUESTION:
            decision.use_rag = True
            decision.use_tool = False
            decision.reasoning = "Information query, will use RAG knowledge base"
            
        elif intent == Intent.ACTION:
            # Check if we can identify which tool to use
            tool_info = self._identify_tool(context.query)
            if tool_info:
                decision.use_tool = True
                decision.tool_name = tool_info['name']
                decision.tool_params = tool_info['params']
                decision.reasoning = f"Action requested, will use {tool_info['name']} tool"
                
                # Log for debugging
                logger.info(f"Tool decision: {tool_info['name']} with params {tool_info['params']}")
            else:
                decision.use_tool = False
                decision.reasoning = "Action requested but no suitable tool found"
        
        context.add_debug(f"Decision: {decision.reasoning}")
        return decision
    
    def _identify_tool(self, query: str) -> Optional[Dict[str, Any]]:
        """Identify which tool to use based on query."""
        query_lower = query.lower()
        
        # File reading operations
        if any(word in query_lower for word in ['read', 'show', 'open', 'view', 'display', 'cat']):
            if 'file' in query_lower or self._has_file_reference(query):
                file_ref = self._extract_file_reference(query)
                if file_ref:
                    logger.debug(f"Identified file read operation for: {file_ref}")
                    return {
                        'name': 'read_file',
                        'params': {'file_identifier': file_ref}  # FIXED: correct param name
                    }
        
        # File search operations
        if any(phrase in query_lower for phrase in ['find file', 'search file', 'locate file', 'look for']):
            search_term = self._extract_search_term(query)
            if search_term:
                return {
                    'name': 'search_files',
                    'params': {'query': search_term}
                }
        
        # File writing operations
        if any(word in query_lower for word in ['write', 'create', 'save']):
            if 'file' in query_lower:
                file_path, content = self._extract_write_params(query)
                if file_path:
                    return {
                        'name': 'write_file',
                        'params': {'file_path': file_path, 'content': content or ''}
                    }
        
        # List directory operations
        if any(phrase in query_lower for phrase in ['list files', 'show files', 'files in']):
            directory = self._extract_directory(query)
            return {
                'name': 'list_files',
                'params': {'directory': directory or '.'}
            }
            
        # Document generation operations
        if any(word in query_lower for word in ['create', 'build', 'generate', 'make']):
            doc_type, title, description = self._extract_document_request(query)
            if doc_type:
                return {
                    'name': 'generate_document',
                    'params': {
                        'doc_type': doc_type,
                        'title': title or 'Generated Document',
                        'content': description or ''
                    }
                }
        
        return None
    
    def _extract_document_request(self, query: str) -> tuple:
        """Extract document type, title, and description from query."""
        import re
        
        query_lower = query.lower()
        
        # Detect document type
        doc_types = {
            'word': 'docx',
            'doc': 'docx',
            'docx': 'docx',
            'powerpoint': 'pptx',
            'ppt': 'pptx',
            'pptx': 'pptx',
            'presentation': 'pptx',
            'pdf': 'pdf',
            'text': 'txt',
            'markdown': 'md'
        }
        
        doc_type = None
        for keyword, file_type in doc_types.items():
            if keyword in query_lower:
                doc_type = file_type
                break
        
        if not doc_type:
            return None, None, None
        
        # Extract title/topic
        patterns = [
            r'(?:about|regarding|on)\s+(.+?)(?:\s+with|\s+that|\s*$)',
            r'(?:build|create|generate|make)\s+(?:a|an)?\s+\w+\s+(.+?)(?:\s+with|\s*$)',
        ]
        
        title = None
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                title = match.group(1).strip()
                break
        
        return doc_type, title, query
    
    def _has_file_reference(self, query: str) -> bool:
        """Check if query likely references a file."""
        import re
        # Has file extension
        if re.search(r'\.\w{2,4}\b', query):
            return True
        # Common file-related words
        file_words = ['txt', 'file', 'document', 'config', 'log', 'data']
        return any(word in query.lower() for word in file_words)
    
    # CRITICAL FIX: File Reference Extraction
    # app/agent/classifiers.py - Replace _extract_file_reference method

    def _extract_file_reference(self, query: str) -> Optional[str]:
        """
        Extract file path or name from query.
        Handles spaces in paths, complex phrases, and various patterns.
        """
        import re
        
        # Remove common prefixes
        query_clean = re.sub(
            r'^(please\s+|could\s+you\s+|can\s+you\s+|would\s+you\s+)', 
            '', 
            query, 
            flags=re.IGNORECASE
        )
        
        # Pattern 1: "show me the contents of test.txt"
        # Must handle "show me" separately to avoid extracting "me"
        match = re.search(
            r'(?:show|display)\s+'  # Action word
            r'(?:me\s+)?'           # Optional "me"
            r'(?:all\s+)?'          # Optional "all" - NEW
            r'(?:the\s+)?'          # Optional "the"
            r'(?:contents?\s+)?'    # Optional "content/contents"
            r'(?:of\s+)?'           # Optional "of"
            r'(?:the\s+)?'          # Optional "the" again
            r'(?:file\s+)?'         # Optional "file"
            r'(?:called\s+)?'       # Optional "called" - NEW
            r'(?:named\s+)?'        # Optional "named" - NEW
            r'([^\s,]+(?:\.\w+)?)', # THE FILENAME
            query_clean,
            re.IGNORECASE
        )
        if match:
            candidate = match.group(1).strip('.,!?')
            # Additional validation - reject if it's still a stop word
            if candidate.lower() not in ['all', 'the', 'of', 'me', 'contents', 'content', 'file']:
                return candidate
        
        # Pattern 2: "read test.txt" or "open file.txt"
        match = re.search(
            r'(?:read|open|view|cat)\s+(?:the\s+)?(?:file\s+)?([^\s,]+(?:\.\w+)?)',
            query_clean,
            re.IGNORECASE
        )
        if match:
            return match.group(1).strip('.,!?')
        
        # Pattern 3: "read file test.txt"
        match = re.search(
            r'(?:read|open|show|view)\s+(?:the\s+)?file\s+([^\s,]+)',
            query_clean,
            re.IGNORECASE
        )
        if match:
            return match.group(1).strip('.,!?')
        
        # Pattern 4: "contents of test.txt"
        match = re.search(
            r'contents?\s+(?:of\s+)?([^\s,]+(?:\.\w+)?)',
            query_clean,
            re.IGNORECASE
        )
        if match:
            return match.group(1).strip('.,!?')
        
        # Pattern 5: Full path (with or without quotes)
        # Handles: C:\path\with spaces\file.txt
        # Must read from action word to end of string or comma
        match = re.search(
            r'(?:read|open)\s+["\']?([A-Za-z]:[^\'"]+?)["\']?(?:\s*$|,)',
            query_clean,
            re.IGNORECASE
        )
        if match:
            # Path with spaces - take everything
            return match.group(1).strip()
        
        # Pattern 6: Unix-style path
        match = re.search(
            r'(?:read|open)\s+(/[^\s,]+)',
            query_clean,
            re.IGNORECASE
        )
        if match:
            return match.group(1).strip()
        
        # Pattern 7: Just filename with extension at end (last resort)
        match = re.search(
            r'([^\s/\\]+\.\w{2,4})\s*$',
            query_clean,
            re.IGNORECASE
        )
        if match:
            return match.group(1).strip('.,!?')
        
        return None


    # ADDITIONAL: Add helper to validate extracted reference
    def _validate_file_reference(self, file_ref: str) -> bool:
        """Check if extracted file reference is valid."""
        if not file_ref:
            return False
        
        # Filter out common false positives
        stop_words = {'me', 'the', 'of', 'file', 'contents', 'content'}
        if file_ref.lower() in stop_words:
            return False
        
        # Must have reasonable length
        if len(file_ref) < 2 or len(file_ref) > 500:
            return False
        
        return True


    # UPDATE: Modify _identify_tool to use validation
    def _identify_tool(self, query: str) -> Optional[Dict[str, Any]]:
        """Identify which tool to use based on query."""
        query_lower = query.lower()
        
        # Document generation operations - CHECK THIS FIRST (before file operations)
        if any(word in query_lower for word in ['create', 'build', 'generate', 'make']):
            if any(doc_word in query_lower for doc_word in ['document', 'word', 'powerpoint', 'presentation', 'pdf', 'ppt', 'docx']):
                doc_type, title, content_hint = self._extract_document_request(query)
                if doc_type:
                    logger.info(f"📄 Detected document generation: {doc_type} - {title}")
                    return {
                        'name': 'generate_document',
                        'params': {
                            'doc_type': doc_type,
                            'title': title or 'Generated Document',
                            'content': content_hint or query  # Pass full query as content hint
                        }
                    }
        
        # File reading operations
        if any(word in query_lower for word in ['read', 'show', 'open', 'view', 'display', 'cat']):
            if 'file' in query_lower or self._has_file_reference(query):
                file_ref = self._extract_file_reference(query)
                
                # Validate extracted reference
                if file_ref and self._validate_file_reference(file_ref):
                    logger.debug(f"Identified file read operation for: {file_ref}")
                    return {
                        'name': 'read_file',
                        'params': {'file_identifier': file_ref}
                    }
                else:
                    logger.warning(f"Invalid file reference extracted: {file_ref}")
        
        return None
    
    def _extract_search_term(self, query: str) -> Optional[str]:
        """Extract search term from query."""
        import re
        
        # find file test
        match = re.search(r'(?:find|search|locate|look for)\s+(?:file\s+)?(.+)', query, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        return None
    
    def _extract_write_params(self, query: str) -> tuple:
        """Extract file path and content for write operation."""
        import re
        
        # write "content" to file.txt
        match = re.search(r'write\s+["\'](.+?)["\']\s+to\s+([^\s]+)', query, re.IGNORECASE)
        if match:
            return match.group(2), match.group(1)
        
        # create file.txt
        match = re.search(r'(?:create|write)\s+(?:file\s+)?([^\s]+)', query, re.IGNORECASE)
        if match:
            return match.group(1), None
        
        return None, None
    
    def _extract_directory(self, query: str) -> Optional[str]:
        """Extract directory path from query."""
        import re
        
        # list files in /path/to/dir
        match = re.search(r'(?:in|from)\s+([^\s]+)', query, re.IGNORECASE)
        if match:
            return match.group(1)
        
        return None

class LLMDecisionMaker(DecisionMaker):
    """
    Advanced decision maker that uses LLM for complex decisions.
    This shows how to extend for more sophisticated behavior.
    """
    
    def __init__(self, simple_decision_maker: SimpleDecisionMaker, 
                 llm_provider=None):
        self.simple_decision_maker = simple_decision_maker
        self.llm_provider = llm_provider
    
    def decide(self, context: Context, intent: Intent) -> Decision:
        """
        First try simple rules, then use LLM for complex cases.
        """
        # Start with simple decision
        decision = self.simple_decision_maker.decide(context, intent)
        
        # If action but no tool identified, try LLM
        if intent == Intent.ACTION and not decision.use_tool and self.llm_provider:
            decision = self._decide_with_llm(context, intent, decision)
        
        return decision
    
    def _decide_with_llm(self, context: Context, intent: Intent, 
                        initial_decision: Decision) -> Decision:
        """Use LLM to make decision (placeholder)."""
        # This would use LLM to parse the query and decide on tools
        # For now, return initial decision
        return initial_decision


# ============================================================================
# Analyzer (combines classification and decision)
# ============================================================================

class QueryAnalyzer:
    """
    High-level analyzer that combines intent classification and decision making.
    This is a facade that makes it easy to use both components together.
    """
    
    def __init__(self, 
                 classifier: Optional[IntentClassifier] = None,
                 decision_maker: Optional[DecisionMaker] = None):
        self.classifier = classifier or RuleBasedIntentClassifier()
        self.decision_maker = decision_maker or SimpleDecisionMaker()
    
    def analyze(self, context: Context) -> Decision:
        """Analyze query and return decision."""
        # Classify intent
        intent = self.classifier.classify(context)
        context.metadata['intent'] = intent
        
        # Make decision
        decision = self.decision_maker.decide(context, intent)
        context.metadata['decision'] = decision
        
        return decision


# ============================================================================
# Factory Functions
# ============================================================================

def create_rule_based_analyzer(tools: Optional[Dict[str, Tool]] = None) -> QueryAnalyzer:
    """Create a simple rule-based analyzer."""
    classifier = RuleBasedIntentClassifier()
    decision_maker = SimpleDecisionMaker(tools)
    return QueryAnalyzer(classifier, decision_maker)


def create_hybrid_analyzer(tools: Optional[Dict[str, Tool]] = None, 
                          use_llm: bool = False) -> QueryAnalyzer:
    """Create a hybrid analyzer with optional LLM support."""
    classifier = HybridIntentClassifier(use_llm=use_llm)
    decision_maker = SimpleDecisionMaker(tools)
    
    if use_llm:
        # Upgrade to LLM decision maker if LLM is available
        decision_maker = LLMDecisionMaker(decision_maker)
    
    return QueryAnalyzer(classifier, decision_maker)