import os
import uuid
import numpy as np
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import app components (Run this script with 'python -m scripts.ingest_z_drive')
from app.rag.embeddings import get_embeddings
from app.rag.store import vector_store
from app.rag.schemas import RagChunk

def ingest_z_drive():
    # Configuration
    z_path = "Z:\\"
    print(f"\n🚀 Starting Knowledge Base Ingestion from: {z_path}")
    print("-" * 50)

    # 1. Load Documents
    # Uses utf-8 for Greek support and silent_errors to prevent crashes on locked files
    loader = DirectoryLoader(
        z_path, 
        glob="**/*.txt", 
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'},
        silent_errors=True 
    )
    
    try:
        documents = loader.load()
        if not documents:
            print("⚠️ No .txt documents found on Z: drive. Check path or file extensions.")
            return
            
        # --- FILENAME AUDIT ---
        print(f"📄 Successfully loaded {len(documents)} documents:")
        for i, doc in enumerate(documents, 1):
            full_path = doc.metadata.get('source', 'Unknown')
            filename = os.path.basename(full_path)
            print(f"   {i}. {filename}")
        print("-" * 50)

    except Exception as e:
        print(f"❌ Critical error during loading phase: {e}")
        return

    # 2. Split into Chunks
    # Small chunks (800 chars) help the AI find specific answers more accurately
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ Created {len(chunks)} text snippets for the Knowledge Base.")

    # 3. Generate Embeddings
    print("🧠 Generating mathematical vectors via GPU (cuda)...")
    embeddings_model = get_embeddings()
    texts = [doc.page_content for doc in chunks]
    
    # Generate vectors and ensure they are float32 for FAISS compatibility
    vectors = embeddings_model.encode(texts)
    vectors = np.array(vectors).astype('float32')

    # 4. Map to Dataclass
    # Each snippet is converted to your specific RagChunk schema
    rag_chunks = []
    for t, c in zip(texts, chunks):
        file_source = c.metadata.get("source", "Z:\\unknown.txt")
        
        chunk = RagChunk(
            chunk_id=str(uuid.uuid4()),  # Required unique ID
            content=t,                   # The actual text
            source=file_source,          # Metadata: File path
            metadata=c.metadata          # Metadata: Full dictionary
        )
        rag_chunks.append(chunk)

    # 5. Save to Vector Store
    try:
        # Clear old data if you want a fresh start (optional)
        # vector_store.clear() 
        
        vector_store.add(rag_chunks, vectors)
        print(f"\n✅ SUCCESS! Knowledge Base updated.")
        print(f"🤖 The AI is now 'Aware' of {len(rag_chunks)} total snippets from your files.")
    except Exception as e:
        print(f"❌ Failed to save to FAISS index: {e}")

if __name__ == "__main__":
    ingest_z_drive()