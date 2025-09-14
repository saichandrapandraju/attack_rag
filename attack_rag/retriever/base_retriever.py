from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from schemas import Document


class BaseRetriever(ABC):
    """Abstract base class for retriever implementations"""
    
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, corpus: List[Document], top_k: int = 5) -> List[Tuple[Document, float]]:
        """Retrieve top-k documents for a query"""
        pass
    
    def compute_similarity(self, query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        # Cosine similarity
        return np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))