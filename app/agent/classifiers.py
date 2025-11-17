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
                    'cat', 'show me', 'display the', 'read the'
                ],
                'patterns': [
                    r'(read|show|open|view)\s+(the\s+)?file',
                    r'file\s+at\s+[/\w]+',
                    r'(content|contents)\s+of',
                ],
                'weight': 1.5  # Actions get priority
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

class SimpleDecisionMaker(DecisionMaker):
    """
    Simple decision maker that decides based on intent and available tools.
    Much cleaner than the original _make_decision method!
    """
    
    def __init__(self, tools: Optional[Dict[str, Tool]] = None):
        self.tools = tools or {}
    
    def decide(self, context: Context, intent: Intent) -> Decision:
        """Make a decision based on intent."""
        
        # Start with defaults
        decision = Decision(
            intent=intent,
            confidence=1.0,
            use_tool=False,
            tool_name=None,
            tool_params=None,
            use_rag=True,
            reasoning=""
        )
        
        # Handle based on intent
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
            else:
                decision.use_tool = False
                decision.reasoning = "Action requested but no suitable tool found"
        
        context.add_debug(f"Decision: {decision.reasoning}")
        logger.info(f"Decision made: {decision.reasoning}")
        
        return decision
    
    def _identify_tool(self, query: str) -> Optional[Dict[str, Any]]:
        """Identify which tool to use based on query."""
        query_lower = query.lower()
        
        # Check for file reading patterns
        if any(word in query_lower for word in ['read', 'show', 'open', 'view', 'file']):
            # Extract file path
            file_path = self._extract_file_path(query)
            if file_path:
                return {
                    'name': 'read_file',
                    'params': {'file_path': file_path}
                }
        
        # Add more tool patterns here as you add tools
        
        return None
    
    def _extract_file_path(self, query: str) -> Optional[str]:
        """Extract file path from query."""
        # Look for paths like /path/to/file or C:\path\to\file
        patterns = [
            r'(?:at|from|in)\s+([/\w\-\.]+)',
            r'([/\w\-\.]+\.\w+)',  # Files with extensions
            r'((?:/|[A-Z]:)[^\s]+)',  # Absolute paths
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query)
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