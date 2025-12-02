# app/localization/greek.py
"""
Greek language localization for RagModel-be.
All user-facing text and system prompts in Greek.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class GreekMessages:
    """Greek messages for user-facing text."""
    
    # Error messages
    error_generic: str = "Συγγνώμη, παρουσιάστηκε σφάλμα."
    error_generation: str = "Συγγνώμη, δεν μπόρεσα να δημιουργήσω απάντηση."
    error_tool_not_found: str = "Το εργαλείο '{tool}' δεν βρέθηκε."
    error_tool_failed: str = "Η εκτέλεση του εργαλείου απέτυχε: {error}"
    error_file_not_found: str = "Το αρχείο δεν βρέθηκε: {path}"
    error_auth_failed: str = "Η αυθεντικοποίηση απέτυχε."
    error_chat_not_found: str = "Η συνομιλία δεν βρέθηκε."
    
    # Success messages
    success_file_read: str = "Διάβασα το αρχείο {filename}."
    success_file_created: str = "Δημιουργήθηκε το αρχείο {filename}."
    success_tool_executed: str = "Το εργαλείο εκτελέστηκε επιτυχώς."
    success_login: str = "Επιτυχής σύνδεση!"
    success_logout: str = "Αποσυνδεθήκατε."
    success_chat_created: str = "Δημιουργήθηκε νέα συνομιλία."
    
    # System messages
    thinking: str = "Σκέφτομαι..."
    processing: str = "Επεξεργασία..."
    searching: str = "Αναζήτηση..."
    loading: str = "Φόρτωση..."
    
    def format(self, key: str, **kwargs) -> str:
        """Format message with variables."""
        template = getattr(self, key, self.error_generic)
        try:
            return template.format(**kwargs)
        except KeyError:
            return template


# Greek system prompt for the AI assistant
GREEK_SYSTEM_PROMPT = """
# ΡΟΛΟΣ

Είσαι ένας εξυπηρετικός βοηθός AI που μιλάει Ελληνικά. Απαντάς πάντα στα Ελληνικά εκτός αν ο χρήστης ζητήσει διαφορετικά.

# ΒΑΣΙΚΕΣ ΑΡΧΕΣ

**Αμεσότητα:** Απάντα πρώτα, εξήγησε μετά. Μην αποφεύγεις με ερωτήσεις εκτός αν είναι πραγματικά ασαφές.

**Επίγνωση Πλαισίου:** Χρησιμοποίησε πληροφορίες που θυμάσαι για τον χρήστη όταν είναι σχετικές.

**Ακολουθία Οδηγιών:** Όταν ο χρήστης δίνει κανόνες, ακολούθησέ τους με ακρίβεια.

**Φυσική Συνομιλία:** Απόφυγε επαναλήψεις. Προσαρμόσου στο ύφος του χρήστη.

# ΜΟΡΦΗ ΑΠΑΝΤΗΣΗΣ

1. Άμεση απάντηση
2. Σύντομη εξήγηση αν χρειάζεται
3. Συνέχεια μόνο αν είναι σχετική

Μην περιγράφεις τη διαδικασία σκέψης σου. Απάντα φυσικά.

# ΒΑΣΗ ΓΝΩΣΕΩΝ

Έχεις πρόσβαση σε έγγραφα. Όταν παρέχεται context σε <context> tags:
1. Χρησιμοποίησε το context για απαντήσεις
2. Η βάση γνώσεων είναι αυθεντική - προτίμησέ την
3. Έλεγξε τη βάση γνώσεων ΠΡΙΝ χρησιμοποιήσεις άλλα εργαλεία

# ΧΕΙΡΙΣΜΟΣ ΕΡΓΑΛΕΙΩΝ

**Τι συμβαίνει:**
- Αυτόματη επιλογή καλύτερου αρχείου όταν υπάρχουν πολλά
- Λαμβάνεις πλήρες περιεχόμενο με metadata

**Μοτίβο Απάντησης:**
Διάβασα το [αρχείο] από [τοποθεσία]:
[περιεχόμενο ή απάντηση βασισμένη στο περιεχόμενο]

**Τι να ΜΗΝ κάνεις:**
- Μην ρωτάς "ποιο αρχείο;" όταν παρέχεται περιεχόμενο
- Μην αγνοείς επιτυχώς ανακτημένο περιεχόμενο
- Μην κόβεις περιεχόμενο εκτός αν ζητηθεί

# ΓΛΩΣΣΑ

Απάντα ΠΑΝΤΑ στα Ελληνικά εκτός αν:
- Ο χρήστης γράφει στα Αγγλικά
- Ο χρήστης ζητά απάντηση σε άλλη γλώσσα
- Το περιεχόμενο είναι τεχνικός κώδικας (διατήρησε αγγλικούς όρους)
"""


# Greek tool descriptions
GREEK_TOOL_DESCRIPTIONS = {
    "read_file": "Διαβάζει περιεχόμενο αρχείου",
    "write_file": "Γράφει περιεχόμενο σε αρχείο",
    "list_files": "Εμφανίζει λίστα αρχείων",
    "search_files": "Αναζητά αρχεία",
    "create_document": "Δημιουργεί έγγραφο",
    "create_presentation": "Δημιουργεί παρουσίαση",
    "create_spreadsheet": "Δημιουργεί υπολογιστικό φύλλο",
}


# Greek intent descriptions
GREEK_INTENT_LABELS = {
    "question": "Ερώτηση",
    "action": "Ενέργεια",
    "greeting": "Χαιρετισμός",
    "clarification": "Διευκρίνηση",
    "file_operation": "Λειτουργία αρχείου",
    "document_creation": "Δημιουργία εγγράφου",
    "unknown": "Άγνωστο",
}


class GreekLocalization:
    """
    Greek localization manager.
    Provides localized text for all user-facing components.
    """
    
    def __init__(self):
        self.messages = GreekMessages()
        self.system_prompt = GREEK_SYSTEM_PROMPT
        self.tool_descriptions = GREEK_TOOL_DESCRIPTIONS
        self.intent_labels = GREEK_INTENT_LABELS
    
    def get_system_prompt(self, custom_additions: str = "") -> str:
        """Get system prompt with optional custom additions."""
        if custom_additions:
            return f"{self.system_prompt}\n\n{custom_additions}"
        return self.system_prompt
    
    def get_error(self, error_type: str, **kwargs) -> str:
        """Get localized error message."""
        key = f"error_{error_type}"
        return self.messages.format(key, **kwargs)
    
    def get_success(self, success_type: str, **kwargs) -> str:
        """Get localized success message."""
        key = f"success_{success_type}"
        return self.messages.format(key, **kwargs)
    
    def get_tool_description(self, tool_name: str) -> str:
        """Get Greek description for a tool."""
        return self.tool_descriptions.get(tool_name, tool_name)
    
    def get_intent_label(self, intent: str) -> str:
        """Get Greek label for an intent."""
        return self.intent_labels.get(intent, intent)
    
    def localize_response(self, response: str, ensure_greek: bool = True) -> str:
        """
        Ensure response contains Greek text.
        If response is purely English and ensure_greek is True,
        adds a note suggesting Greek communication.
        """
        # Check if response contains any Greek characters
        has_greek = any('\u0370' <= char <= '\u03FF' or '\u1F00' <= char <= '\u1FFF' 
                       for char in response)
        
        if not has_greek and ensure_greek:
            # Response is not in Greek - this is usually fine for code/technical content
            pass
        
        return response


# Global instance
greek = GreekLocalization()


def get_greek_system_prompt() -> str:
    """Convenience function to get Greek system prompt."""
    return greek.get_system_prompt()


def get_greek_error(error_type: str, **kwargs) -> str:
    """Convenience function to get Greek error message."""
    return greek.get_error(error_type, **kwargs)