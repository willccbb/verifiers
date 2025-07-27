import sys
sys.path.insert(0, 'environments/truthful_qa')

import verifiers as vf
from openai import OpenAI
import os

# Import the environment directly
from truthful_qa import load_environment

# Create a test to evaluate the environment
def test_environment():
    print("Loading TruthfulQA environment...")
    
    # Load with a small number of examples for testing
    env = load_environment(num_eval_examples=10, difficulty="easy")
    
    print(f"Environment loaded with {len(env.dataset)} examples")
    
    # Print a few example prompts
    print("\nExample prompts:")
    for i in range(min(3, len(env.dataset))):
        print(f"\n{i+1}. {env.dataset[i]['prompt']}")
        print(f"   Expected answer: {env.dataset[i]['answer']}")
    
    # Check if API key is provided
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\nPlease set OPENAI_API_KEY environment variable to test with GPT-4.1-mini")
        print("You can run: export OPENAI_API_KEY='your-api-key'")
        return
    
    # Evaluate with the model
    print("\nEvaluating with gpt-4o-mini...")
    client = OpenAI(api_key=api_key)
    
    try:
        results = env.evaluate(client, "gpt-4o-mini", num_examples=5)
        
        # Print results
        print("\nEvaluation Results:")
        print(f"Average reward: {results['mean_reward']:.3f}")
        
        if 'examples' in results:
            print("\nSample responses:")
            for i, example in enumerate(results['examples'][:2]):
                print(f"\n{i+1}. Question: {example['prompt']}")
                print(f"   Model response: {example.get('completion', 'N/A')}")
                print(f"   Expected: {example.get('answer', 'N/A')}")
                print(f"   Reward: {example.get('reward', 'N/A')}")
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        print("Make sure you have a valid OpenAI API key set.")

if __name__ == "__main__":
    test_environment()