from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np


@dataclass
class Document:
    """Represents a document in the corpus"""
    id: str
    content: str
    metadata: Dict[str, Any] = None
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self):
        return {
            'id': self.id,
            'content': self.content,
            'metadata': self.metadata or {}
        }


@dataclass
class AttackResult:
    """Results from an attack execution"""
    poisoned_docs: List[Document]
    target_queries: List[str]
    target_answers: List[str]
    optimization_history: List[Dict]
    success_rate: float = 0.0
    metadata: Dict[str, Any] = None


@dataclass
class EvaluationResult:
    """Evaluation metrics for an attack"""
    attack_success_rate: float
    retrieval_success_rate: float
    top_k_scores: Dict[int, float]
    poisoned_doc_rankings: List[int]
    detailed_results: List[Dict]