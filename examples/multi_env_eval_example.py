import asyncio
from openai import AsyncOpenAI
import os

from verifiers.scripts.eval import eval_environments_parallel, push_eval_to_prime_hub


async def example_multi_env_eval():
    """Example: Evaluate multiple environments in parallel."""
    
    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
        base_url="https://api.openai.com/v1"
    )
    
    # Define environments to evaluate
    envs = ["gsm8k", "math500"]
    
    # Run parallel evaluation
    results = await eval_environments_parallel(
        envs=envs,
        env_args_dict={
            "gsm8k": {},
            "math500": {},
        },
        client=client,
        model="gpt-4o-mini",
        num_examples=[10, 10],
        rollouts_per_example=[3, 3],
        max_concurrent=[32, 32],
        sampling_args={
            "temperature": 0.7,
            "max_tokens": 2048,
        },
    )
    
    # Process results
    for env_name, output in results.items():
        print(f"\n=== {env_name} ===")
        print(f"Number of samples: {len(output.reward)}")
        print(f"Average reward: {sum(output.reward) / len(output.reward):.3f}")
        print(f"Rewards: {output.reward[:5]}...")  # Show first 5
        
        # Show metrics if available
        if output.metrics:
            for metric_name, metric_values in output.metrics.items():
                avg = sum(metric_values) / len(metric_values)
                print(f"Average {metric_name}: {avg:.3f}")


async def example_with_prime_hub():
    """Example: Evaluate and save to Prime Hub."""
    
    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
        base_url="https://api.openai.com/v1"
    )
    
    envs = ["gsm8k", "math500"]
    model = "gpt-4o-mini"
    
    # Run evaluation
    results = await eval_environments_parallel(
        envs=envs,
        env_args_dict={"gsm8k": {}, "math500": {}},
        client=client,
        model=model,
        num_examples=[10, 10],
        rollouts_per_example=[3, 3],
        max_concurrent=[32, 32],
        sampling_args={"temperature": 0.7, "max_tokens": 2048},
    )
    
    # Push each environment's results to Prime Hub
    for env_name, output in results.items():
        # Calculate metrics
        avg_reward = sum(output.reward) / len(output.reward)
        
        metrics = {
            "avg_reward": float(avg_reward),
            "num_samples": len(output.reward),
        }
        
        # Add any additional metrics from the output
        for metric_name, metric_values in output.metrics.items():
            metrics[f"avg_{metric_name}"] = float(sum(metric_values) / len(metric_values))
        
        # Prepare metadata
        metadata = {
            "environment": env_name,
            "model": model,
            "num_examples": 10,
            "rollouts_per_example": 3,
            "sampling_args": {"temperature": 0.7, "max_tokens": 2048},
        }
        
        # Save to hub
        push_eval_to_prime_hub(
            eval_name=f"{model.replace('/', '-')}-{env_name}",
            model_name=model,
            dataset=env_name,
            metrics=metrics,
            metadata=metadata,
        )


async def example_from_prime_rl_style():
    """
    Example: How prime-rl would use this for checkpoint evaluation.
    
    This shows how prime-rl can replace its current eval logic with the
    verifiers multi-env evaluation.
    """
    
    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
        base_url="http://localhost:8000/v1"  # Local vLLM server
    )
    
    # prime-rl style config
    eval_config = {
        "environment_ids": ["gsm8k", "math500", "aime2025"],
        "environment_args": {
            "gsm8k": {},
            "math500": {"subset": "full"},
            "aime2025": {},
        },
        "num_examples": [100, 50, 20],
        "rollouts_per_example": [3, 5, 10],
        "max_concurrent": [32, 16, 8],
    }
    
    model_name = "meta-llama/llama-3.1-70b-instruct"
    checkpoint_step = 1000
    
    # Run evaluation
    results = await eval_environments_parallel(
        envs=eval_config["environment_ids"],
        env_args_dict=eval_config["environment_args"],
        client=client,
        model=model_name,
        num_examples=eval_config["num_examples"],
        rollouts_per_example=eval_config["rollouts_per_example"],
        max_concurrent=eval_config["max_concurrent"],
        sampling_args={
            "temperature": 0.7,
            "max_tokens": 2048,
            "top_p": 0.95,
        },
    )
    
    # Log to wandb (prime-rl style)
    for env_name, output in results.items():
        avg_reward = sum(output.reward) / len(output.reward)
        print(f"eval/{env_name}/avg_reward: {avg_reward:.4f} (step={checkpoint_step})")
        
        # Save to Prime Hub with checkpoint info
        push_eval_to_prime_hub(
            eval_name=f"{model_name.replace('/', '-')}-{env_name}-step{checkpoint_step}",
            model_name=model_name,
            dataset=env_name,
            metrics={
                "avg_reward": float(avg_reward),
                "num_samples": len(output.reward),
            },
            metadata={
                "environment": env_name,
                "checkpoint_step": checkpoint_step,
                "num_examples": eval_config["num_examples"][eval_config["environment_ids"].index(env_name)],
            },
        )


if __name__ == "__main__":
    print("Example 1: Basic multi-environment evaluation")
    asyncio.run(example_multi_env_eval())
    
    print("\n" + "="*80 + "\n")
    print("Example 2: With Prime Hub integration")
    asyncio.run(example_with_prime_hub())
    
    print("\n" + "="*80 + "\n")
    print("Example 3: prime-rl style checkpoint evaluation")
    asyncio.run(example_from_prime_rl_style())

