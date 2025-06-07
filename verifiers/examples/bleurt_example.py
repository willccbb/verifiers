"""
Example usage of BleurtRubric for semantic similarity evaluation.

This example demonstrates how to use BLEURT for text summarization tasks,
where semantic adequacy is more important than exact string matching.
"""

from datasets import load_dataset
import verifiers as vf

# Check if BLEURT is available
try:
    from verifiers.rubrics import BleurtRubric
    BLEURT_AVAILABLE = True
except ImportError:
    BLEURT_AVAILABLE = False
    print("BLEURT not available. Install with: pip install bleurt tensorflow")


def main():
    if not BLEURT_AVAILABLE:
        return
    
    print("Setting up BLEURT summarization example...")
    
    # Load a small dataset for demonstration
    # For real use, consider using cnn_dailymail or similar
    dataset = load_dataset('wikipedia', '20220301.simple', split='train[:50]')
    
    # Transform dataset for summarization task
    def transform_for_summarization(examples):
        return {
            'question': [f"Summarize this article in 2-3 sentences:\n\n{text[:500]}..." 
                        for text in examples['text']],
            'answer': [text[:200] + "..." for text in examples['text']]  # First 200 chars as reference
        }
    
    dataset = dataset.map(transform_for_summarization, batched=True)
    
    # Setup structured output format
    parser = vf.XMLParser(['summary'])
    
    system_prompt = f"""You are a helpful assistant that creates concise summaries.
Given an article, write a clear 2-3 sentence summary that captures the main points.

Respond in the following format:
{parser.get_format_str()}"""
    
    # Create BLEURT rubric
    print("Creating BLEURT rubric...")
    rubric = BleurtRubric(
        use_fast_model=True,        # Use distilled model for speed
        bleurt_weight=0.8,          # Main reward
        add_format_reward=True,     # Add format compliance
        format_weight=0.2,          # Format checking weight
        parser=parser
    )
    
    # Alternative: Manual rubric composition
    # rubric = vf.Rubric(funcs=[
    #     BleurtRubric()._bleurt_reward_func,
    #     parser.get_format_reward_func()
    # ], weights=[0.8, 0.2], parser=parser)
    
    # Create environment
    print("Setting up environment...")
    eval_dataset = dataset.select(range(10))    # Small eval set
    train_dataset = dataset.select(range(10, 50))  # Small training set
    
    env = vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric
    )
    
    # Optional: Test with API client first
    print("Testing with API client...")
    try:
        from openai import OpenAI
        import os
        
        if os.getenv('OPENAI_API_KEY'):
            client = OpenAI()
            results = env.evaluate(client, model="gpt-4o-mini", num_samples=3)
            print(f"API evaluation results: {results['rewards_avg']:.3f}")
        else:
            print("No OpenAI API key found, skipping API evaluation")
    except ImportError:
        print("OpenAI client not available")
    
    # Training configuration
    print("Configuring training...")
    model_name = 'microsoft/DialoGPT-small'  # Small model for demo
    
    args = vf.grpo_defaults(run_name='bleurt_summarization_demo')
    args.num_iterations = 1
    args.per_device_train_batch_size = 2
    args.num_generations = 4
    args.gradient_accumulation_steps = 2
    args.eval_strategy = "steps"
    args.eval_steps = 5
    args.max_steps = 10  # Very short for demo
    args.save_strategy = "no"  # Don't save checkpoints for demo
    
    # Initialize model and trainer
    print("Loading model...")
    model, tokenizer = vf.get_model_and_tokenizer(model_name)
    
    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=env,
        args=args
    )
    
    # Start training
    print("Starting GRPO training with BLEURT rewards...")
    trainer.train()
    
    print("Training completed! BLEURT rubric successfully integrated.")


def example_hybrid_usage():
    """Example of combining BLEURT with other rewards."""
    
    if not BLEURT_AVAILABLE:
        return
    
    print("\nDemonstrating hybrid reward usage...")
    
    parser = vf.XMLParser(['reasoning', 'answer'])
    
    # Create custom hybrid rubric
    def exact_match_reward(completion, answer, **kwargs):
        """Simple exact match for final answer."""
        parsed = parser.parse_field(completion, 'answer')
        return 1.0 if parsed and parsed.strip() == str(answer).strip() else 0.0
    
    # Combine exact match + BLEURT + format
    rubric = vf.Rubric(funcs=[
        exact_match_reward,                              # Exact answer matching
        BleurtRubric()._bleurt_reward_func,             # Semantic similarity
        parser.get_format_reward_func()                  # Format compliance
    ], weights=[0.5, 0.3, 0.2], parser=parser)
    
    print("Hybrid rubric created: 50% exact + 30% BLEURT + 20% format")
    
    # Test the rubric
    test_completion = """<reasoning>
    This is a math problem. Let me solve step by step.
    First, I need to add the numbers: 5 + 3 = 8
    </reasoning>
    
    <answer>8</answer>"""
    
    test_answer = "8"
    
    # This would normally be called by the environment
    # rewards = rubric.score_rollouts([test_completion], [test_answer], ...)
    print("Hybrid rubric ready for use in environment")


if __name__ == "__main__":
    main()
    example_hybrid_usage()