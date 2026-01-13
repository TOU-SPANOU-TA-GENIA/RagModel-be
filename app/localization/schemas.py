from pydantic import BaseModel, Field
from typing import Dict, Optional

class LogMessages(BaseModel):
    """System status messages (Thinking, Loading, etc)."""
    thinking: str = "Thinking..."
    processing: str = "Processing..."
    searching: str = "Searching..."
    loading: str = "Loading..."
    
class ErrorMessages(BaseModel):
    """Error notifications."""
    generic: str = "An error occurred."
    generation_failed: str = "Could not generate response."
    tool_not_found: str = "Tool '{tool}' not found."
    tool_failed: str = "Tool execution failed: {error}"
    file_not_found: str = "File not found: {path}"
    auth_failed: str = "Authentication failed."
    chat_not_found: str = "Chat session not found."

class SuccessMessages(BaseModel):
    """Success notifications."""
    file_read: str = "Read file {filename}."
    file_created: str = "Created file {filename}."
    tool_executed: str = "Tool executed successfully."
    login: str = "Login successful!"
    logout: str = "Logged out."
    chat_created: str = "New chat created."

class LocaleBundle(BaseModel):
    """The complete language pack."""
    language_code: str = "en"
    errors: ErrorMessages = Field(default_factory=ErrorMessages)
    success: SuccessMessages = Field(default_factory=SuccessMessages)
    logs: LogMessages = Field(default_factory=LogMessages)
    
    # Generic bucket for extra UI labels (buttons, titles)
    ui: Dict[str, str] = Field(default_factory=dict)