import sys
import os
import logging
import json

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.db.rag import RAGPipeline

def main():
    # Initialize Pipeline
    pipeline = RAGPipeline()
    
    # Load tests
    test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test.json')
    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found.")
        return

    with open(test_file, 'r') as f:
        data = json.load(f)
    
    tests = data.get('tests', [])
    print(f"Found {len(tests)} tests in {test_file}...")

    for i, test in enumerate(tests):
        query = test['question']
        
        # Execute
        result = pipeline.run(query)
        
        # Output
        print("-" * 50)
        print(f"QUESTION: {query}")
        print(f"ANSWER:\n{result['answer']}")
        print(f"EXPECTED: {test['expected_answer']}")
        print("-" * 50)

if __name__ == "__main__":
    main()