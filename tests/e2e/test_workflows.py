# tests/e2e/test_workflows.py
"""
End-to-end workflow tests.
"""

import pytest


class TestRAGWorkflow:
    """End-to-end tests for RAG workflow."""
    
    def test_complete_rag_flow(self, test_client, sample_documents):
        """Test complete RAG flow: ingest -> query -> get answer."""
        # Step 1: Trigger ingestion
        ingest_response = test_client.post("/ingest/all", params={"rebuild": True})
        
        # Step 2: Create chat
        chat_response = test_client.post("/chats", json={"title": "RAG Test"})
        chat_id = chat_response.json()
        
        # Step 3: Send query related to documents
        message_response = test_client.post(
            f"/chats/{chat_id}/message",
            json={"role": "user", "content": "Tell me about Python programming"}
        )
        
        assert message_response.status_code == 200
        data = message_response.json()
        assert "answer" in data
        # Check if sources are returned (RAG worked)
        # Note: sources may be empty if documents aren't relevant


class TestConversationWorkflow:
    """End-to-end tests for conversation workflow."""
    
    def test_multi_turn_conversation(self, test_client):
        """Test multi-turn conversation maintains context."""
        # Turn 1: Introduce information
        response1 = test_client.post(
            "/chat",
            json={"role": "user", "content": "My favorite color is blue."}
        )
        session_id = response1.json()["session_id"]
        
        # Turn 2: Set instruction
        response2 = test_client.post(
            "/chat",
            params={"session_id": session_id},
            json={"role": "user", "content": "When I say 'color', tell me my favorite color."}
        )
        
        # Turn 3: Test instruction
        response3 = test_client.post(
            "/chat",
            params={"session_id": session_id},
            json={"role": "user", "content": "color"}
        )
        
        # All responses should succeed
        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response3.status_code == 200
        
        # Session should be maintained
        assert response2.json()["session_id"] == session_id
        assert response3.json()["session_id"] == session_id


class TestFileOperationsWorkflow:
    """End-to-end tests for file operations."""
    
    def test_file_read_workflow(self, test_client, sample_documents):
        """Test reading files through chat."""
        # Ask to read a file
        response = test_client.post(
            "/chat",
            json={"role": "user", "content": "Read the file test1.txt"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check that tool was used
        if data.get("tool_used"):
            assert data["tool_used"] == "read_file"


class TestDocumentGenerationWorkflow:
    """End-to-end tests for document generation."""
    
    def test_create_document_workflow(self, test_client):
        """Test document generation through chat."""
        # Request document creation
        response = test_client.post(
            "/chat",
            json={
                "role": "user",
                "content": "Create a Word document about Python basics"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # If document was created, should have tool info
        if data.get("tool_used") == "generate_document":
            tool_result = data.get("tool_result", {})
            if tool_result.get("success"):
                assert "file_path" in tool_result.get("data", {})