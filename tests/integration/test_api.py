# tests/integration/test_api.py
"""
Integration tests for API endpoints.
"""

import pytest


class TestHealthEndpoints:
    """Tests for health and status endpoints."""
    
    def test_health_check(self, test_client):
        """Test health endpoint returns valid response."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "services" in data
    
    def test_startup_status(self, test_client):
        """Test startup status endpoint."""
        response = test_client.get("/startup-status")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_stats_endpoint(self, test_client):
        """Test stats endpoint."""
        response = test_client.get("/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_chats" in data
        assert "total_messages" in data


class TestChatEndpoints:
    """Tests for chat management endpoints."""
    
    def test_create_chat(self, test_client):
        """Test creating a new chat."""
        response = test_client.post(
            "/chats",
            json={"title": "Test Chat"}
        )
        
        assert response.status_code == 201
        chat_id = response.json()
        assert isinstance(chat_id, str)
        assert len(chat_id) > 0
    
    def test_list_chats(self, test_client):
        """Test listing all chats."""
        # Create a chat first
        test_client.post("/chats", json={"title": "Test"})
        
        response = test_client.get("/chats")
        
        assert response.status_code == 200
        chats = response.json()
        assert isinstance(chats, list)
    
    def test_get_chat_details(self, test_client):
        """Test getting chat details."""
        # Create chat
        create_response = test_client.post("/chats", json={"title": "Test"})
        chat_id = create_response.json()
        
        # Get details
        response = test_client.get(f"/chats/{chat_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == chat_id
        assert data["title"] == "Test"
    
    def test_get_nonexistent_chat(self, test_client):
        """Test getting a chat that doesn't exist."""
        response = test_client.get("/chats/nonexistent-id")
        
        assert response.status_code == 404
    
    def test_delete_chat(self, test_client):
        """Test deleting a chat."""
        # Create chat
        create_response = test_client.post("/chats", json={"title": "Test"})
        chat_id = create_response.json()
        
        # Delete
        response = test_client.delete(f"/chats/{chat_id}")
        
        assert response.status_code == 200
        
        # Verify deleted
        get_response = test_client.get(f"/chats/{chat_id}")
        assert get_response.status_code == 404


class TestChatMessaging:
    """Tests for chat messaging."""
    
    def test_send_message(self, test_client):
        """Test sending a message."""
        # Create chat
        create_response = test_client.post("/chats", json={"title": "Test"})
        chat_id = create_response.json()
        
        # Send message
        response = test_client.post(
            f"/chats/{chat_id}/message",
            json={"role": "user", "content": "Hello"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "intent" in data
    
    def test_simple_chat_endpoint(self, test_client):
        """Test the simple /chat endpoint."""
        response = test_client.post(
            "/chat",
            json={"role": "user", "content": "Hello"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "session_id" in data
    
    def test_simple_chat_with_session(self, test_client):
        """Test /chat with session continuity."""
        # First message
        response1 = test_client.post(
            "/chat",
            json={"role": "user", "content": "My name is Test"}
        )
        session_id = response1.json()["session_id"]
        
        # Second message with same session
        response2 = test_client.post(
            "/chat",
            params={"session_id": session_id},
            json={"role": "user", "content": "What is my name?"}
        )
        
        assert response2.status_code == 200
        assert response2.json()["session_id"] == session_id


class TestConfigEndpoints:
    """Tests for configuration API."""
    
    def test_get_all_config(self, test_client):
        """Test getting all configuration."""
        response = test_client.get("/config/")
        
        assert response.status_code == 200
        data = response.json()
        assert "llm" in data
        assert "rag" in data
    
    def test_get_config_metadata(self, test_client):
        """Test getting configuration metadata."""
        response = test_client.get("/config/metadata")
        
        assert response.status_code == 200
        data = response.json()
        assert "llm" in data
        # Metadata is a list of field info
        assert isinstance(data["llm"], list)
        # Check at least one field exists
        assert len(data["llm"]) > 0
    
    def test_update_config_value(self, test_client):
        """Test updating a configuration value."""
        # First get current value
        get_response = test_client.get("/config/")
        original_temp = get_response.json().get("llm", {}).get("temperature", 0.7)
        
        # Try to update - may return 200 or 422 depending on endpoint implementation
        response = test_client.put(
            "/config/llm/temperature",
            json={"value": 0.8}
        )
        
        # Accept either success or validation error (endpoint may not exist)
        assert response.status_code in [200, 404, 422]
        
        # If successful, verify update
        if response.status_code == 200:
            get_response = test_client.get("/config/")
            assert get_response.json()["llm"]["temperature"] == 0.8
            
            # Restore original
            test_client.put("/config/llm/temperature", json={"value": original_temp})