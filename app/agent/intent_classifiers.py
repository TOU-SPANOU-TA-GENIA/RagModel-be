# app/agent/intent_classifiers.py
"""
Intent classification implementations.
Determines user intent from queries.
"""

import re
from typing import Optional

from app.core.interfaces import Intent, Context, IntentClassifier
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class RuleBasedIntentClassifier(IntentClassifier):
    """
    Rule-based intent classifier using keywords and patterns.
    """
    
    def __init__(self):
        self.patterns = {
            Intent.ACTION: {
                'keywords': [
                    'read', 'open', 'show', 'display', 'view', 'check', 'see',
                    'get', 'fetch', 'look at', 'look in', "what's in",
                    'execute', 'run', 'create', 'write', 'update', 'delete',
                    'cat', 'show me', 'display the', 'read the',
                    'build', 'generate', 'make', 'produce',
                    'document', 'powerpoint', 'word', 'pdf', 'presentation'
                ],
                'patterns': [
                    r'(read|show|open|view)\s+(the\s+)?file',
                    r'file\s+at\s+[/\w]+',
                    r'(content|contents)\s+of',
                    r'(build|create|generate|make)\s+(a|an)?\s+(word|powerpoint|pdf|document|presentation)',
                    r'(build|create|generate)\s+.*\s+(document|powerpoint|presentation|word|pdf)',
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
                'weight': 0.5
            }
        }
    
    def classify(self, context: Context) -> Intent:
        """Classify intent based on rules."""
        query = context.query.lower().strip()
        scores = {intent: 0.0 for intent in Intent}
        
        for intent, config in self.patterns.items():
            score = self._calculate_score(query, config, context)
            scores[intent] = score * config['weight']
        
        best_intent, best_score = self._get_best_match(scores)
        
        if best_score < 0.5:
            best_intent = Intent.QUESTION
        
        context.add_debug(f"Intent scores: {scores}")
        logger.info(f"Classified '{query[:50]}...' as {best_intent.value} (score: {best_score})")
        
        return best_intent
    
    def _calculate_score(self, query: str, config: dict, context: Context) -> float:
        """Calculate score for an intent."""
        score = 0.0
        
        for keyword in config['keywords']:
            if keyword in query:
                score += 1.0
                context.add_debug(f"Found keyword '{keyword}'")
        
        for pattern in config['patterns']:
            if re.search(pattern, query):
                score += 2.0
                context.add_debug(f"Matched pattern")
        
        return score
    
    def _get_best_match(self, scores: dict) -> tuple:
        """Find best matching intent."""
        best_intent = Intent.UNKNOWN
        best_score = 0.0
        
        for intent, score in scores.items():
            if score > best_score:
                best_score = score
                best_intent = intent
        
        return best_intent, best_score


class HybridIntentClassifier(IntentClassifier):
    """
    Classifier combining rules with optional LLM fallback.
    """
    
    def __init__(self, rule_classifier: RuleBasedIntentClassifier = None, 
                 use_llm: bool = False):
        self.rule_classifier = rule_classifier or RuleBasedIntentClassifier()
        self.use_llm = use_llm
    
    def classify(self, context: Context) -> Intent:
        """First try rules, then optionally use LLM."""
        intent = self.rule_classifier.classify(context)
        
        if intent == Intent.UNKNOWN and self.use_llm:
            intent = self._classify_with_llm(context)
        
        return intent
    
    def _classify_with_llm(self, context: Context) -> Intent:
        """LLM classification placeholder."""
        return Intent.QUESTION