# app/rag/enhanced_retriever.py
"""
Enhanced Retriever with Keyword Boosting

Problem: Semantic search misses documents containing specific names/terms
         because embeddings don't capture exact keyword matches well.

Solution: Combine semantic search with simple keyword matching.
          If a word from the query appears in a document, boost that document.

No patterns, no rules - just text matching.
"""

import re
from typing import List, Dict, Any, Set
import numpy as np

from app.rag.retrievers import SimpleRetriever
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class KeywordBoostedRetriever:
    """
    Retriever that combines semantic search with keyword matching.
    
    When the query contains specific words (names, terms), documents
    containing those words get boosted even if semantic similarity is low.
    """
    
    # Common words to ignore (stopwords)
    STOPWORDS = {
        # Greek
        'ο', 'η', 'το', 'τον', 'την', 'του', 'της', 'οι', 'τα', 'των', 'τους', 'τις',
        'και', 'να', 'με', 'για', 'από', 'σε', 'στο', 'στη', 'στον', 'στην',
        'είναι', 'έχω', 'έχει', 'θα', 'θέλω', 'μου', 'σου', 'είμαι',
        'ένα', 'μια', 'ένας', 'αυτό', 'αυτή', 'αυτός', 'που', 'πως', 'τι',
        # English
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your',
        'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'from', 'by', 'about', 'into', 'this', 'that', 'what', 'how',
    }
    
    def __init__(
        self, 
        semantic_retriever: SimpleRetriever,
        keyword_boost: float = 0.3,
        min_keyword_length: int = 3
    ):
        """
        Args:
            semantic_retriever: Base semantic retriever
            keyword_boost: Score boost for keyword matches (0.0-1.0)
            min_keyword_length: Minimum word length to consider as keyword
        """
        self.semantic_retriever = semantic_retriever
        self.keyword_boost = keyword_boost
        self.min_keyword_length = min_keyword_length
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve documents using semantic search + keyword boosting.
        """
        # Step 1: Get more results from semantic search (we'll filter later)
        semantic_k = k * 2  # Get extra results to allow for reranking
        semantic_results = self.semantic_retriever.retrieve(query, k=semantic_k)
        
        # Step 2: Extract meaningful keywords from query
        query_keywords = self._extract_keywords(query)
        logger.debug(f"Query keywords: {query_keywords}")
        
        if not query_keywords:
            # No meaningful keywords, return semantic results as-is
            return semantic_results[:k]
        
        # Step 3: Also search ALL documents for keyword matches
        keyword_matches = self._find_keyword_matches(query_keywords)
        
        # Step 4: Merge and rerank results
        merged = self._merge_results(semantic_results, keyword_matches, query_keywords)
        
        # Step 5: Return top k
        return merged[:k]
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text."""
        # Tokenize
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter
        keywords = set()
        for word in words:
            # Skip stopwords
            if word in self.STOPWORDS:
                continue
            # Skip short words
            if len(word) < self.min_keyword_length:
                continue
            # Skip pure numbers
            if word.isdigit():
                continue
            keywords.add(word)
        
        return keywords
    
    def _find_keyword_matches(self, keywords: Set[str]) -> List[Dict[str, Any]]:
        """Find documents containing any of the keywords."""
        matches = []
        
        # Access all documents from the vector store
        if hasattr(self.semantic_retriever, 'vector_store'):
            store = self.semantic_retriever.vector_store
            
            # For FastInMemoryVectorStore
            if hasattr(store, 'db') and hasattr(store.db, 'documents'):
                for doc_id, doc_data in store.db.documents.items():
                    content = doc_data.get('content', '').lower()
                    
                    # Check for keyword matches
                    matching_keywords = []
                    for kw in keywords:
                        if kw in content:
                            matching_keywords.append(kw)
                    
                    if matching_keywords:
                        matches.append({
                            'content': doc_data.get('content', ''),
                            'metadata': doc_data.get('metadata', {}),
                            'score': 0.0,  # Will be boosted
                            'keyword_matches': matching_keywords
                        })
                        logger.debug(f"Keyword match in {doc_data.get('metadata', {}).get('source', 'unknown')}: {matching_keywords}")
        
        return matches
    
    def _merge_results(
        self, 
        semantic: List[Dict], 
        keyword: List[Dict],
        query_keywords: Set[str]
    ) -> List[Dict]:
        """Merge semantic and keyword results with boosting."""
        # Create lookup by content hash
        seen_content = {}
        results = []
        
        # Process semantic results first
        for result in semantic:
            content = result.get('content', '')
            content_key = hash(content[:100])  # Use first 100 chars as key
            
            # Check for keyword matches in this result
            content_lower = content.lower()
            matching_keywords = [kw for kw in query_keywords if kw in content_lower]
            
            # Calculate boosted score
            base_score = result.get('score', 0.5)
            boost = len(matching_keywords) * self.keyword_boost
            final_score = min(base_score + boost, 1.0)
            
            result['score'] = final_score
            result['keyword_matches'] = matching_keywords
            
            seen_content[content_key] = True
            results.append(result)
        
        # Add keyword-only matches that weren't in semantic results
        for result in keyword:
            content = result.get('content', '')
            content_key = hash(content[:100])
            
            if content_key not in seen_content:
                # Give keyword-only matches a base score
                num_matches = len(result.get('keyword_matches', []))
                result['score'] = 0.2 + (num_matches * self.keyword_boost)
                
                seen_content[content_key] = True
                results.append(result)
                logger.info(f"Added keyword-only match: {result.get('metadata', {}).get('source', 'unknown')}")
        
        # Sort by final score
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return results


def create_keyword_boosted_retriever(
    semantic_retriever: SimpleRetriever,
    keyword_boost: float = 0.3
) -> KeywordBoostedRetriever:
    """Create a keyword-boosted retriever wrapping an existing semantic retriever."""
    return KeywordBoostedRetriever(
        semantic_retriever=semantic_retriever,
        keyword_boost=keyword_boost
    )