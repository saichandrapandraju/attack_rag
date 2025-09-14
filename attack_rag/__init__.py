"""
PoisonedRAG Core Implementation
===============================

This package implements the core adversarial document generation from the PoisonedRAG paper.

Usage:
    from attack_rag import generate_adversarial_document
    
    result = generate_adversarial_document(
        query="What is the capital of France?",
        correct_answer="Paris",
        openai_api_key="your-api-key"
    )
"""

from .poisoned_rag_core import (
    PoisonedRAGGenerator,
    generate_adversarial_document,
    generate_batch_adversarial_documents
)

__all__ = [
    "PoisonedRAGGenerator",
    "generate_adversarial_document", 
    "generate_batch_adversarial_documents"
]
