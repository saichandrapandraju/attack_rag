#!/usr/bin/env python3
"""
PoisonedRAG Core Demo
====================

This demo shows the CORE research contribution of PoisonedRAG:
How to generate adversarial documents using LLM prompting.

This is what researchers actually care about - the adversarial generation technique.
Everything else (corpus management, retrieval evaluation) can be implemented by the user.
"""

import os
import sys
import json
from attack_rag import generate_adversarial_document, generate_batch_adversarial_documents

def main():
    print("=" * 60)
    print("POISONEDRAG CORE ADVERSARIAL DOCUMENT GENERATION")
    print("=" * 60)
    print()
    print("This demo shows the core research contribution of PoisonedRAG:")
    print("How to generate adversarial documents using LLM prompting.")
    print()
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OpenAI API key not found!")
        print("Please set your API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print()
        print("Or edit this script and set api_key variable directly.")
        return
    
    print("✅ OpenAI API key found!")
    print()
    
    # Example 1: Single query
    print("=" * 40)
    print("EXAMPLE 1: Single Query")
    print("=" * 40)
    
    query = "What is the capital of France?"
    correct_answer = "Paris"
    
    print(f"Query: {query}")
    print(f"Correct Answer: {correct_answer}")
    print()
    print("Generating adversarial documents...")
    
    try:
        result = generate_adversarial_document(
            query=query,
            correct_answer=correct_answer,
            openai_api_key=api_key,
            model_name="gpt-4-1106-preview",
            adv_per_query=3  # Generate 3 adversarial texts
        )
        
        print()
        print("🎯 GENERATED ADVERSARIAL DOCUMENTS:")
        print(f"   Incorrect Answer: '{result['incorrect_answer']}'")
        print()
        print("   Adversarial Corpus Texts:")
        for i, text in enumerate(result['adversarial_texts'], 1):
            print(f"   {i}. {text}")
            print()
        
        print("✅ Successfully generated adversarial documents!")
        print()
        print("🔍 How this works:")
        print("   1. The LLM is prompted to create an INCORRECT answer")
        print("   2. The LLM generates supporting texts that make the incorrect answer seem correct")
        print("   3. These texts can be injected into a RAG corpus to poison retrieval")
        
    except Exception as e:
        print(f"❌ Failed to generate adversarial documents: {e}")
        return
    
    print()
    
    # Example 2: Batch processing
    print("=" * 40)
    print("EXAMPLE 2: Batch Processing")
    print("=" * 40)
    
    queries_and_answers = [
        ("Who wrote Romeo and Juliet?", "William Shakespeare"),
        ("What is the largest planet in our solar system?", "Jupiter"),
        ("What year did World War II end?", "1945")
    ]
    
    print("Processing multiple queries:")
    for i, (q, a) in enumerate(queries_and_answers, 1):
        print(f"  {i}. Q: {q}")
        print(f"     A: {a}")
    print()
    
    print("Generating adversarial documents for all queries...")
    
    try:
        results = generate_batch_adversarial_documents(
            queries_and_answers=queries_and_answers,
            openai_api_key=api_key,
            model_name="gpt-4-1106-preview",
            adv_per_query=2  # 2 adversarial texts per query
        )
        
        print()
        print("🎯 BATCH RESULTS:")
        for i, result in enumerate(results, 1):
            print(f"   Query {i}: {result['question']}")
            print(f"   Incorrect Answer: {result['incorrect_answer']}")
            print(f"   Generated {len(result['adversarial_texts'])} adversarial texts")
            print(f"   Adversarial Texts: {result['adversarial_texts']}")
            print()
        
        print(f"✅ Successfully processed {len(results)}/{len(queries_and_answers)} queries!")
        
        # Save results
        output_file = "adversarial_documents.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to {output_file}")
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        return
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    print("✨ This is the CORE of PoisonedRAG research:")
    print("   • Input: Query + Correct Answer")
    print("   • Output: Incorrect Answer + Supporting Adversarial Texts")
    print("   • Method: Carefully crafted LLM prompts")
    print()
    print("🚀 Next steps:")
    print("   • Inject these adversarial texts into your RAG corpus")
    print("   • Test retrieval effectiveness")
    print("   • Measure attack success rates")
    print("   • Develop defenses")
    print()
    print("The rest is just standard RAG evaluation - this is the novel contribution!")


if __name__ == "__main__":
    main()
