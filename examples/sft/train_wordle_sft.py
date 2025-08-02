import argparse
import verifiers as vf
from transformers import TrainingArguments
from trl import SFTTrainer
import os

def main(args):
    size = args.size
    # Use Qwen models with proper HF authentication
    if size == "1.7B":
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    elif size == "4B":
        model_name = "Qwen/Qwen2.5-3B-Instruct"
    else:
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print(f"Loading model: {model_name}")
    model, tokenizer = vf.get_model_and_tokenizer(model_name, model_kwargs={"attn_implementation": "sdpa"})
    
    # Load wordle environment to get dataset
    vf_env = vf.load_environment(env_id="vf_wordle")
    
    # Get train dataset
    dataset = vf_env.dataset
    print(f"Dataset size: {len(dataset)}")
    
    # Convert to text format for SFT
    def format_example(example):
        messages = example["prompt"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text += f"Let me think about this Wordle puzzle...\n\nAnswer: {example['answer']}"
        return {"text": text}
    
    dataset = dataset.map(format_example)
    
    # Training arguments
    run_name = f"wordle-sft-{size}"
    training_args = TrainingArguments(
        output_dir=f"./outputs/{run_name}",
        run_name=run_name,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        logging_steps=1,
        save_steps=50,
        eval_strategy="no",
        save_strategy="steps",
        warmup_steps=10,
        bf16=True,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        remove_unused_columns=False,
        max_steps=int(os.environ.get("VERIFIERS_MAX_STEPS", "100")),
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=512,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model
    trainer.save_model()
    print(f"Model saved to {training_args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", "-s", type=str, default="0.5B")
    args = parser.parse_args()
    main(args)