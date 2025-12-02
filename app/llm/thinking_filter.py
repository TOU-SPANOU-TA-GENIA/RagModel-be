# app/llm/thinking_filter.py
"""
Thinking Filter - Handles Qwen's internal thinking/reasoning.

Qwen3 models can produce internal reasoning that should not be shown to users.
This module:
1. Wraps internal thinking in <think> tags during generation
2. Filters thinking from user-facing responses
3. Preserves thinking for debugging/logging

Key insight: We add rules to make Qwen output thinking in tags,
then strip those tags from the response. This is more reliable than
trying to detect implicit thinking patterns.
"""

import re
from typing import Tuple, Optional
from dataclasses import dataclass

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ThinkingResult:
    """Result of thinking extraction."""
    response: str           # Clean response for user
    thinking: str          # Extracted thinking (empty if none)
    had_thinking: bool     # Whether thinking was present


class ThinkingFilter:
    """
    Filter for handling model thinking/reasoning.
    
    Strategy:
    1. Add instruction to prompt that makes model wrap thinking in <think> tags
    2. Extract and remove <think> blocks from response
    3. Clean any residual artifacts
    """
    
    # Pattern to match thinking blocks
    THINK_PATTERN = re.compile(
        r'<think>(.*?)</think>',
        re.DOTALL | re.IGNORECASE
    )
    
    # Alternative patterns for thinking that might leak
    ALT_THINKING_PATTERNS = [
        # Qwen-style thinking markers
        re.compile(r'^Thinking:.*?(?=\n\n|Response:|$)', re.DOTALL | re.IGNORECASE | re.MULTILINE),
        re.compile(r'^Analysis:.*?(?=\n\n|Response:|$)', re.DOTALL | re.IGNORECASE | re.MULTILINE),
        re.compile(r'^Reasoning:.*?(?=\n\n|$)', re.DOTALL | re.IGNORECASE | re.MULTILINE),
        
        # Internal process narration
        re.compile(r"^(?:Let me|I'll|I will)\s+(?:think|analyze|consider).*?(?:\.|$)", re.IGNORECASE | re.MULTILINE),
        
        # Numbered reasoning steps at start
        re.compile(r'^1\.\s*(?:First|Analyze|Consider).*?(?:\n\n|$)', re.DOTALL | re.IGNORECASE),
    ]
    
    # Thinking instruction to prepend to prompts
    THINKING_INSTRUCTION = """
<thinking_rule>
Εσωτερική σκέψη: Αν χρειάζεται να σκεφτείς πριν απαντήσεις, βάλε τη σκέψη σου μέσα σε <think> tags.
Η σκέψη δεν θα εμφανιστεί στον χρήστη.

Μορφή:
<think>
[Η εσωτερική σου σκέψη εδώ]
</think>

[Η απάντησή σου εδώ]

ΣΗΜΑΝΤΙΚΟ: Μην συμπεριλάβεις τη σκέψη σου στην απάντηση. Απάντα φυσικά.
</thinking_rule>

"""
    
    def add_thinking_instruction(self, prompt: str) -> str:
        """
        Add thinking instruction to prompt.
        This makes the model output thinking in <think> tags.
        """
        return self.THINKING_INSTRUCTION + prompt
    
    def extract_thinking(self, response: str) -> ThinkingResult:
        """
        Extract thinking from response.
        
        Returns clean response and extracted thinking separately.
        """
        if not response:
            return ThinkingResult(response="", thinking="", had_thinking=False)
        
        # Find all thinking blocks
        thinking_parts = []
        
        # Extract <think> blocks
        for match in self.THINK_PATTERN.finditer(response):
            thinking_parts.append(match.group(1).strip())
        
        # Remove thinking blocks from response
        clean_response = self.THINK_PATTERN.sub('', response)
        
        # Check for alternative thinking patterns
        for pattern in self.ALT_THINKING_PATTERNS:
            match = pattern.search(clean_response)
            if match:
                thinking_parts.append(match.group(0).strip())
                clean_response = pattern.sub('', clean_response)
        
        # Clean up response
        clean_response = self._clean_response(clean_response)
        
        # Combine thinking
        combined_thinking = "\n\n".join(thinking_parts) if thinking_parts else ""
        
        had_thinking = bool(thinking_parts)
        
        if had_thinking:
            logger.debug(f"Extracted {len(thinking_parts)} thinking blocks")
        
        return ThinkingResult(
            response=clean_response,
            thinking=combined_thinking,
            had_thinking=had_thinking
        )
    
    def _clean_response(self, response: str) -> str:
        """Clean residual artifacts from response."""
        cleaned = response
        
        # Remove residual tags
        cleaned = re.sub(r'</?think>', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'</?response>', '', cleaned, flags=re.IGNORECASE)
        
        # Remove "Response:" prefix if present
        cleaned = re.sub(r'^Response:\s*', '', cleaned, flags=re.IGNORECASE)
        
        # Normalize whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    
    def filter_streaming_token(
        self, 
        token: str, 
        state: dict
    ) -> Tuple[str, dict]:
        """
        Filter a streaming token for thinking content.
        
        Args:
            token: Current token
            state: Mutable state dict tracking thinking status
                   Keys: 'in_thinking', 'buffer'
        
        Returns:
            (output_token, updated_state)
            output_token is empty string if token is part of thinking
        """
        buffer = state.get('buffer', '') + token
        in_thinking = state.get('in_thinking', False)
        
        # Check for thinking start
        if '<think>' in buffer and not in_thinking:
            parts = buffer.split('<think>', 1)
            state['in_thinking'] = True
            state['buffer'] = parts[1] if len(parts) > 1 else ''
            return parts[0], state
        
        # Check for thinking end
        if '</think>' in buffer and in_thinking:
            parts = buffer.split('</think>', 1)
            state['in_thinking'] = False
            state['buffer'] = ''
            # Return the part after thinking
            return parts[1] if len(parts) > 1 else '', state
        
        # In thinking mode - don't output
        if in_thinking:
            state['buffer'] = buffer
            return '', state
        
        # Normal mode - output token
        state['buffer'] = ''
        return token, state


# Global instance
thinking_filter = ThinkingFilter()


def add_thinking_instruction(prompt: str) -> str:
    """Add thinking instruction to prompt."""
    return thinking_filter.add_thinking_instruction(prompt)


def extract_thinking(response: str) -> ThinkingResult:
    """Extract thinking from response."""
    return thinking_filter.extract_thinking(response)


def filter_thinking(response: str) -> str:
    """
    Simple function to remove thinking from response.
    Returns clean response only.
    """
    result = thinking_filter.extract_thinking(response)
    return result.response