# tests/unit/test_agent.py
"""
Unit tests for agent components.
"""

import pytest
from app.core.interfaces import Context, Intent


class TestIntentClassifier:
    """Tests for intent classification."""
    
    @pytest.fixture
    def classifier(self):
        from app.agent.classifiers import RuleBasedIntentClassifier
        return RuleBasedIntentClassifier()
    
    def _create_context(self, query: str) -> Context:
        return Context(
            query=query,
            chat_history=[],
            metadata={},
            debug_info=[]
        )
    
    def test_question_intent(self, classifier):
        """Test question intent detection."""
        queries = [
            "What is Python?",
            "How do I install packages?",
            "Why is the sky blue?",
            "Explain machine learning",
        ]
        
        for query in queries:
            context = self._create_context(query)
            intent = classifier.classify(context)
            assert intent == Intent.QUESTION, f"'{query}' should be QUESTION"
    
    def test_action_intent(self, classifier):
        """Test action intent detection."""
        queries = [
            "Read the file test.txt",
            "Show me the contents of config.json",
            "Create a document about Python",
            "Execute the backup script",
        ]
        
        for query in queries:
            context = self._create_context(query)
            intent = classifier.classify(context)
            assert intent == Intent.ACTION, f"'{query}' should be ACTION"
    
    def test_conversation_intent(self, classifier):
        """Test conversation intent detection."""
        queries = [
            "Hello",
            "Hi there",
            "Thanks",
            "Goodbye",
        ]
        
        for query in queries:
            context = self._create_context(query)
            intent = classifier.classify(context)
            assert intent == Intent.CONVERSATION, f"'{query}' should be CONVERSATION"


class TestDecisionMaker:
    """Tests for decision making."""
    
    @pytest.fixture
    def decision_maker(self):
        from app.agent.classifiers import SimpleDecisionMaker
        return SimpleDecisionMaker()
    
    def _create_context(self, query: str) -> Context:
        return Context(
            query=query,
            chat_history=[],
            metadata={},
            debug_info=[]
        )
    
    def test_question_uses_rag(self, decision_maker):
        """Questions should use RAG."""
        context = self._create_context("What is Python?")
        decision = decision_maker.decide(context, Intent.QUESTION)
        
        assert decision.use_rag is True
        assert decision.use_tool is False
    
    def test_action_identifies_tool(self, decision_maker):
        """Actions should identify appropriate tools."""
        context = self._create_context("Read the file test.txt")
        decision = decision_maker.decide(context, Intent.ACTION)
        
        assert decision.use_tool is True
        assert decision.tool_name == "read_file"
    
    def test_conversation_no_tools(self, decision_maker):
        """Conversations should not use tools or RAG."""
        context = self._create_context("Hello")
        decision = decision_maker.decide(context, Intent.CONVERSATION)
        
        assert decision.use_tool is False
        assert decision.use_rag is False


class TestAgentOrchestrator:
    """Tests for agent orchestrator."""
    
    def test_process_simple_query(self, mock_agent):
        """Test processing a simple query."""
        response = mock_agent.process_query("Hello")
        
        assert response.answer is not None
        assert len(response.answer) > 0
        assert response.intent in ["question", "action", "conversation", "unknown"]
    
    def test_process_with_history(self, mock_agent):
        """Test processing with chat history."""
        history = [
            {"role": "user", "content": "My name is Test"},
            {"role": "assistant", "content": "Nice to meet you, Test!"}
        ]
        
        response = mock_agent.process_query("What is my name?", chat_history=history)
        
        assert response.answer is not None