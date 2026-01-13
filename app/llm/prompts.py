from typing import List
from app.llm.schemas import Message
from app.config import get_config

class PromptManager:
    """
    Constructs prompts based on configured templates.
    """
    def __init__(self):
        self.config = get_config()

    def format_chat(self, messages: List[Message]) -> str:
        """
        Formats a list of messages into a single string for the local model.
        Uses Qwen-style formatting by default (<|im_start|>...).
        """
        # Retrieve format from config or default to Qwen
        template_style = self.config.get("llm", {}).get("prompt_style", "qwen")
        
        if template_style == "qwen":
            return self._format_qwen(messages)
        else:
            return self._format_generic(messages)

    def _format_qwen(self, messages: List[Message]) -> str:
        """Qwen/ChatML format."""
        formatted = ""
        for msg in messages:
            formatted += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
        formatted += "<|im_start|>assistant\n"
        return formatted

    def _format_generic(self, messages: List[Message]) -> str:
        """Fallback format."""
        formatted = ""
        for msg in messages:
            role = msg.role.capitalize()
            formatted += f"{role}: {msg.content}\n"
        formatted += "Assistant:"
        return formatted

prompt_manager = PromptManager()