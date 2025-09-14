from typing import List, Optional, Dict, Tuple
from attack_rag.schemas import Document, AttackResult, EvaluationResult
from attack_rag.retriever.base_retriever import BaseRetriever
from attack_rag.retriever.simulated_retriever import SimulatedRetriever
from attack_rag.evaluator import Evaluator
import logging
import json

logger = logging.getLogger(__name__)

class BlackBoxRAGAttack:
    """
    Main framework for black-box RAG attacks
    Phase 1: Focus on core PoisonedRAG implementation
    """
    
    def __init__(
        self,
        retriever: Optional[BaseRetriever] = None,
        attack_config: Optional[Dict] = None
    ):
        self.retriever = retriever or SimulatedRetriever()
        self.attack_config = attack_config or {}
        self.evaluator = Evaluator(self.retriever)
        
    def run_poisoned_rag_attack(
        self,
        target_queries: List[str],
        target_answers: List[str],
        corpus: Optional[List[Document]] = None,
        **kwargs
    ) -> Tuple[AttackResult, EvaluationResult]:
        """
        Execute PoisonedRAG attack and evaluate results
        """
        from attack_rag.attacks.poisoned_rag import PoisonedRAGAttack
        
        # Merge configurations
        config = {**self.attack_config, **kwargs}
        
        # Initialize attack
        attack = PoisonedRAGAttack(
            retriever=self.retriever,
            num_poison_docs=config.get('num_poison_docs', 5),
            optimization_iterations=config.get('optimization_iterations', 10),
            similarity_threshold=config.get('similarity_threshold', 0.7)
        )
        
        # Execute attack
        logger.info("Starting PoisonedRAG attack...")
        attack_result = attack.attack(target_queries, target_answers, corpus)
        
        # Evaluate results
        logger.info("Evaluating attack effectiveness...")
        eval_result = self.evaluator.evaluate(
            attack_result,
            corpus or [] + attack_result.poisoned_docs
        )
        
        # Log summary
        logger.info(f"Attack completed:")
        logger.info(f"  - Poisoned documents generated: {len(attack_result.poisoned_docs)}")
        logger.info(f"  - Attack Success Rate: {eval_result.attack_success_rate:.2%}")
        logger.info(f"  - Retrieval Success Rate: {eval_result.retrieval_success_rate:.2%}")
        
        return attack_result, eval_result
    
    def save_results(
        self,
        attack_result: AttackResult,
        eval_result: EvaluationResult,
        filepath: str
    ):
        """Save attack and evaluation results to JSON"""
        output = {
            'attack': {
                'type': 'PoisonedRAG',
                'target_queries': attack_result.target_queries,
                'target_answers': attack_result.target_answers,
                'poisoned_docs': [doc.to_dict() for doc in attack_result.poisoned_docs],
                'metadata': attack_result.metadata
            },
            'evaluation': {
                'attack_success_rate': eval_result.attack_success_rate,
                'retrieval_success_rate': eval_result.retrieval_success_rate,
                'top_k_scores': eval_result.top_k_scores,
                'poisoned_doc_rankings': eval_result.poisoned_doc_rankings,
                'detailed_results': eval_result.detailed_results
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")