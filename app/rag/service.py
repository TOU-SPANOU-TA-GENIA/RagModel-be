import os
import logging
from typing import List

logger = logging.getLogger(__name__)

class RagService:
    def __init__(self):
        self.initialized = False
        # Update this to the specific folder on your network drive
        self.z_drive_path = "Z:\\" 

    def scan_filesystem(self) -> int:
        """
        Recursively counts physical files on the Z drive to check raw document availability.
        """
        try:
            if not os.path.exists(self.z_drive_path):
                print(f"⚠️ [FILESYSTEM] Z drive not detected or inaccessible at {self.z_drive_path}")
                return 0
            
            # Recursively count all files
            file_count = sum(len(files) for _, _, files in os.walk(self.z_drive_path))
            print(f"📂 [FILESYSTEM] Z Drive Scan: Found {file_count} total files on the network drive.")
            return file_count
        except Exception as e:
            print(f"❌ [FILESYSTEM] Scan failed: {e}")
            return 0

    def initialize(self):
        """
        Initializes the service by scanning the physical filesystem and 
        checking the AI's current vector knowledge base.
        """
        print("\n--- RAG AWARENESS CHECK ---")
        # 1. Physical Scan
        physical_count = self.scan_filesystem()

        # 2. Vector Index Check
        try:
            from app.rag.store import vector_store
            if vector_store and hasattr(vector_store, 'index') and vector_store.index:
                vector_count = vector_store.index.ntotal #
                print(f"🧠 [RAG INDEX] Knowledge Base: The AI is currently aware of {vector_count} document snippets.")
                
                # Diagnostic check for "Blindness"
                if physical_count > 0 and vector_count <= 1:
                    print("⚠️ [DIAGNOSTIC] ALERT: Files exist on Z: drive, but the AI index is nearly empty.")
                    print("   -> Action needed: Run the ingestion script to 'teach' the AI about these files.")
            else:
                print("🧠 [RAG INDEX] FAISS Index is empty or could not be loaded.")
            
            print("---------------------------\n")
            self.initialized = True
        except Exception as e:
            print(f"❌ [RAG SYSTEM] Error checking vector store awareness: {e}")

    def search(self, query: str, limit: int = 3) -> List[str]:
        """
        Searches the knowledge base for snippets relevant to the user query.
        """
        if not self.initialized:
            self.initialize()

        print(f"🔍 [RAG SEARCH] Searching for: '{query}'")
        
        try:
            from app.rag.store import search_vector_store
            results = search_vector_store(query, k=limit)
            
            if results:
                print(f"✅ [RAG AWARENESS] Found {len(results)} relevant snippets to use for this response.")
            else:
                print(f"⚠️ [RAG AWARENESS] No matching context found in filesystem.")
            
            return results
        except Exception as e:
            print(f"❌ [RAG ERROR] Search failed: {e}")
            return []

# Singleton Instance
rag_service = RagService()