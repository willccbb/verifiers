"""
Example script demonstrating async generation with GRPOEnvTrainer.

This shows how to enable async generation to improve GPU utilization by
generating future batches while training on current batches.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from verifiers import Environment
from verifiers.trainers import GRPOEnvTrainer, GRPOEnvConfig

def main():
    # Initialize model and tokenizer
    model_name = "meta-llama/Llama-3.2-1B"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Initialize environment (example with a simple math environment)
    env = Environment(name="math")  # Replace with your environment
    
    # Configure training with async generation
    training_args = GRPOEnvConfig(
        output_dir="./grpo_async_output",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        learning_rate=1e-6,
        
        # GRPO specific args
        num_generations=8,
        max_prompt_length=512,
        max_completion_length=256,
        temperature=0.9,
        
        # Async generation settings
        use_async_generation=True,  # Enable async generation
        num_steps_async=2,          # Generate 2 batches ahead
        async_generation_timeout=300.0,  # 5 minute timeout
        async_max_queue_size=4,     # Max 4 batches in queue
        
        # vLLM server settings
        vllm_server_host="localhost",
        vllm_server_port=8000,
        
        # Logging
        logging_steps=10,
        log_completions=True,
        report_to=["wandb"],
    )
    
    # Initialize trainer
    trainer = GRPOEnvTrainer(
        model=model,
        env=env,
        args=training_args,
        processing_class=tokenizer,
    )
    
    # Train with async generation
    trainer.train()
    
    # The async generator will automatically:
    # 1. Start generating batches in the background when training begins
    # 2. Keep 2 batches ready ahead of the current training step
    # 3. Stop cleanly when training ends
    
    print("Training completed with async generation!")
    
    # Performance comparison:
    # - Without async: GPU idle during generation, generation blocks training
    # - With async: GPU continuously utilized, generation happens in parallel
    
if __name__ == "__main__":
    main() 