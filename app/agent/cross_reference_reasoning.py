# app/agent/cross_reference_reasoning.py
"""
Cross-Reference Reasoning for Multi-Document Queries

Simple approach:
- When RAG returns multiple documents, add reasoning instructions
- Instructions are GENERIC - no patterns, no keywords
- Works for ANY type of query with multiple knowledge sources
"""

from datetime import datetime
from typing import Optional

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


# Generic reasoning instructions - NO specific patterns or keywords
CROSS_REFERENCE_INSTRUCTIONS = """
<REASONING>
Σημερινή ημερομηνία: {current_date}

Σου δόθηκαν πολλαπλές πηγές πληροφοριών. Πριν απαντήσεις:

1. ΔΙΑΒΑΣΕ όλες τις πηγές προσεκτικά
2. ΕΝΤΟΠΙΣΕ ποιες πληροφορίες από κάθε πηγή σχετίζονται με την ερώτηση
3. ΣΥΝΔΥΑΣΕ τις πληροφορίες - αν μια πηγή έχει κανόνες/προϋποθέσεις και άλλη έχει δεδομένα, έλεγξε αν τα δεδομένα πληρούν τους κανόνες
4. ΥΠΟΛΟΓΙΣΕ αν χρειάζεται (ημερομηνίες, ποσά, διαφορές χρόνου)
5. ΣΥΜΠΕΡΑΝΕ βάσει των παραπάνω

Στην απάντησή σου, δείξε τον συλλογισμό σου με συγκεκριμένα στοιχεία από τις πηγές.
</REASONING>
"""


def get_reasoning_instructions() -> str:
    """
    Get generic cross-reference reasoning instructions.
    
    These are injected when multiple documents are in context.
    """
    current_date = datetime.now().strftime("%d/%m/%Y")
    return CROSS_REFERENCE_INSTRUCTIONS.format(current_date=current_date)


def should_add_reasoning(rag_context) -> bool:
    """
    Determine if reasoning instructions should be added.
    
    Simple rule: if there are multiple documents, add reasoning.
    """
    if not rag_context:
        return False
    
    if isinstance(rag_context, list):
        return len(rag_context) > 1
    
    return False


def enhance_prompt(prompt: str, rag_context) -> str:
    """
    Enhance prompt with reasoning instructions if multiple docs present.
    
    Args:
        prompt: Original prompt
        rag_context: RAG results (list of documents)
        
    Returns:
        Enhanced prompt (or original if no enhancement needed)
    """
    if not should_add_reasoning(rag_context):
        return prompt
    
    instructions = get_reasoning_instructions()
    logger.info(f"Adding cross-reference reasoning ({len(rag_context)} docs)")
    
    # Find best injection point
    if '<|im_start|>user' in prompt:
        parts = prompt.split('<|im_start|>user', 1)
        return parts[0] + instructions + '\n<|im_start|>user' + parts[1]
    
    if '<knowledge_base>' in prompt:
        parts = prompt.split('<knowledge_base>', 1)
        return parts[0] + instructions + '\n<knowledge_base>' + parts[1]
    
    # Fallback: prepend
    return instructions + '\n\n' + prompt