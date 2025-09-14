from typing import List, Optional
from attack_rag.schemas import Document, AttackResult, EvaluationResult
from attack_rag.retriever.base_retriever import BaseRetriever


class Evaluator:
    """Evaluate attack effectiveness with various metrics"""
    
    def __init__(self, retriever: BaseRetriever):
        self.retriever = retriever
    
    def evaluate(
        self,
        attack_result: AttackResult,
        corpus: List[Document],
        test_queries: Optional[List[str]] = None,
        k_values: List[int] = [1, 5, 10, 20]
    ) -> EvaluationResult:
        """
        Comprehensive evaluation of attack effectiveness
        """
        
        if test_queries is None:
            test_queries = attack_result.target_queries
        
        # Calculate ASR (Attack Success Rate)
        asr = self._calculate_asr(
            attack_result.poisoned_docs,
            test_queries,
            corpus
        )
        
        # Calculate RSR (Retrieval Success Rate)
        rsr = self._calculate_rsr(
            attack_result.poisoned_docs,
            test_queries,
            corpus,
            k=5
        )
        
        # Calculate top-k scores
        top_k_scores = {}
        for k in k_values:
            top_k_scores[k] = self._calculate_rsr(
                attack_result.poisoned_docs,
                test_queries,
                corpus,
                k=k
            )
        
        # Get detailed rankings
        rankings = self._get_poison_rankings(
            attack_result.poisoned_docs,
            test_queries,
            corpus
        )
        
        # Compile detailed results
        detailed_results = []
        for query in test_queries:
            results = self.retriever.retrieve(query, corpus, top_k=20)
            query_result = {
                'query': query,
                'top_5_retrieved': [doc.id for doc, _ in results[:5]],
                'poison_found': any(doc in attack_result.poisoned_docs for doc, _ in results[:5]),
                'poison_ranks': [i for i, (doc, _) in enumerate(results) if doc in attack_result.poisoned_docs]
            }
            detailed_results.append(query_result)
        
        return EvaluationResult(
            attack_success_rate=asr,
            retrieval_success_rate=rsr,
            top_k_scores=top_k_scores,
            poisoned_doc_rankings=rankings,
            detailed_results=detailed_results
        )
    
    def _calculate_asr(
        self,
        poisoned_docs: List[Document],
        queries: List[str],
        corpus: List[Document]
    ) -> float:
        """Calculate Attack Success Rate"""
        successes = 0
        
        for query in queries:
            results = self.retriever.retrieve(query, corpus, top_k=5)
            if any(doc in poisoned_docs for doc, _ in results):
                successes += 1
        
        return successes / len(queries) if queries else 0.0
    
    def _calculate_rsr(
        self,
        poisoned_docs: List[Document],
        queries: List[str],
        corpus: List[Document],
        k: int = 5
    ) -> float:
        """Calculate Retrieval Success Rate at k"""
        successes = 0
        
        for query in queries:
            results = self.retriever.retrieve(query, corpus, top_k=k)
            if any(doc in poisoned_docs for doc, _ in results):
                successes += 1
        
        return successes / len(queries) if queries else 0.0
    
    def _get_poison_rankings(
        self,
        poisoned_docs: List[Document],
        queries: List[str],
        corpus: List[Document]
    ) -> List[int]:
        """Get ranking positions of poisoned documents"""
        all_rankings = []
        
        for query in queries:
            results = self.retriever.retrieve(query, corpus, top_k=100)
            for rank, (doc, _) in enumerate(results):
                if doc in poisoned_docs:
                    all_rankings.append(rank + 1)  # 1-indexed
        
        return all_rankings