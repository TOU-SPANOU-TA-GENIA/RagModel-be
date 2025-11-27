# app/agent/classifiers.py
"""
Query analysis components - re-exports from split modules.
"""

from app.agent.intent_classifiers import (
    RuleBasedIntentClassifier,
    HybridIntentClassifier
)
from app.agent.decision_makers import (
    SimpleDecisionMaker,
    LLMDecisionMaker,
    ParameterExtractors
)
from app.core.interfaces import Intent, Context, Decision, Tool
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

__all__ = [
    "RuleBasedIntentClassifier",
    "HybridIntentClassifier",
    "SimpleDecisionMaker",
    "LLMDecisionMaker",
    "QueryAnalyzer",
    "create_rule_based_analyzer",
    "create_hybrid_analyzer"
]


class QueryAnalyzer:
    """
    High-level analyzer combining classification and decision making.
    """
    
    def __init__(self, classifier=None, decision_maker=None):
        self.classifier = classifier or RuleBasedIntentClassifier()
        self.decision_maker = decision_maker or SimpleDecisionMaker()
    
    def analyze(self, context: Context) -> Decision:
        """Analyze query and return decision."""
        intent = self.classifier.classify(context)
        context.metadata['intent'] = intent
        
        decision = self.decision_maker.decide(context, intent)
        context.metadata['decision'] = decision
        
        return decision


def create_rule_based_analyzer(tools=None) -> QueryAnalyzer:
    """Create rule-based analyzer."""
    return QueryAnalyzer(
        RuleBasedIntentClassifier(),
        SimpleDecisionMaker(tools)
    )


def create_hybrid_analyzer(tools=None, use_llm=False) -> QueryAnalyzer:
    """Create hybrid analyzer with optional LLM."""
    classifier = HybridIntentClassifier(use_llm=use_llm)
    decision_maker = SimpleDecisionMaker(tools)
    
    if use_llm:
        decision_maker = LLMDecisionMaker(decision_maker)
    
    return QueryAnalyzer(classifier, decision_maker)