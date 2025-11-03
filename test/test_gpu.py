#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import time
from app.gpu_utils import get_gpu_info, log_gpu_memory
from app.logger import setup_logger

logger = setup_logger(__name__)


def test_gpu_availability():
    logger.info("=" * 50)
    logger.info("Testing GPU Availability")
    logger.info("=" * 50)
    
    if torch.cuda.is_available():
        logger.info(" GPU is available!")
        gpu_info = get_gpu_info()
        
        if gpu_info:
            logger.info(f"GPU Name: {gpu_info['name']}")
            logger.info(f"Total Memory: {gpu_info['total_memory_gb']:.2f} GB")
            logger.info(f"CUDA Version: {gpu_info['cuda_version']}")
            logger.info(f"PyTorch Version: {gpu_info['pytorch_version']}")
    else:
        logger.warning("  GPU is NOT available. Using CPU.")
        logger.info("To use GPU, ensure:")
        logger.info("1. You have an NVIDIA GPU")
        logger.info("2. CUDA is installed")
        logger.info("3. PyTorch with CUDA support is installed")


def test_embedding_speed():
    logger.info("\n" + "=" * 50)
    logger.info("Testing Embedding Speed")
    logger.info("=" * 50)
    
    from app.vectorstore import get_embeddings
    
    embeddings = get_embeddings()
    
    test_texts = [
        "Πώς κάνω restart το PostgreSQL;",
        "How do I create a new Linux user?",
        "What is the backup procedure?",
    ] * 10 
    
    logger.info(f"Generating embeddings for {len(test_texts)} texts...")
    
    start_time = time.time()
    _ = embeddings.embed_documents(test_texts)
    elapsed = time.time() - start_time
    
    logger.info(f" Time taken: {elapsed:.2f} seconds")
    logger.info(f"Average per text: {elapsed/len(test_texts):.3f} seconds")
    
    log_gpu_memory()


def test_llm_inference():
    logger.info("\n" + "=" * 50)
    logger.info("Testing LLM Inference")
    logger.info("=" * 50)
    
    from app.llm import init_llm
    
    logger.info("Loading LLM model...")
    pipe = init_llm()
    
    log_gpu_memory()
    
    test_prompt = "Explain what PostgreSQL is in one sentence."
    
    logger.info(f"Generating response for: '{test_prompt}'")
    start_time = time.time()
    
    outputs = pipe(test_prompt, max_new_tokens=50)
    
    elapsed = time.time() - start_time
    
    logger.info(f" Time taken: {elapsed:.2f} seconds")
    logger.info(f"Response: {outputs[0]['generated_text']}")
    
    log_gpu_memory()

if __name__ == "__main__":
    test_gpu_availability()
    
    if torch.cuda.is_available():
        print("\n" + "="*50)
        response = input("Run speed tests? (y/n): ")
        if response.lower() == 'y':
            test_embedding_speed()
            test_llm_inference()
    else:
        logger.warning("Skipping speed tests (no GPU available)")