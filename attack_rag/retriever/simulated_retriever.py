from typing import List, Tuple
import numpy as np
from attack_rag.retriever.base_retriever import BaseRetriever
from attack_rag.schemas import Document


class SimulatedRetriever(BaseRetriever):
    """
    Simulated retriever for testing without external dependencies
    Uses random embeddings for demonstration
    """
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.cache = {}
        
    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate deterministic pseudo-embeddings based on text content"""
        embeddings = []
        for text in texts:
            if text in self.cache:
                embeddings.append(self.cache[text])
            else:
                # Create deterministic embedding based on text
                np.random.seed(hash(text) % (2**32))
                emb = np.random.randn(self.embedding_dim)
                emb = emb / np.linalg.norm(emb)  # Normalize
                self.cache[text] = emb
                embeddings.append(emb)
        return np.array(embeddings)
    
    def retrieve(self, query: str, corpus: List[Document], top_k: int = 5) -> List[Tuple[Document, float]]:
        """Retrieve top-k documents based on embedding similarity"""
        query_emb = self.encode([query])[0]
        
        scores = []
        for doc in corpus:
            if doc.embedding is None:
                doc.embedding = self.encode([doc.content])[0]
            score = self.compute_similarity(query_emb, doc.embedding)
            scores.append((doc, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]