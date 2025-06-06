# Training

The verifiers framework integrates with state-of-the-art training methods, particularly Group Relative Policy Optimization (GRPO), to train language models using the reward signals from your environments.

## Training Overview

The training pipeline consists of:

1. **Data Generation**: Use environments to create rollouts
2. **Reward Calculation**: Score rollouts with rubrics
3. **Policy Optimization**: Train models using GRPO
4. **Evaluation**: Validate improvements using same environments

## GRPO: Group Relative Policy Optimization

GRPO is a reinforcement learning algorithm designed specifically for LLMs that:
- Learns from relative comparisons within groups
- Reduces reward hacking through comparative evaluation
- Provides stable training dynamics
- Works with any differentiable model

### Basic Training Setup

```python
from verifiers.trainers import GRPOTrainer, GRPOConfig
from verifiers.envs import SingleTurnEnv
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Create environment
env = SingleTurnEnv(
    dataset=dataset,
    parser=XMLParser(["reasoning", "answer"]),
    rubric=custom_rubric
)

# Configure training
config = GRPOConfig(
    learning_rate=1e-5,
    batch_size=32,
    group_size=4,  # Compare 4 samples per prompt
    num_epochs=3,
    gradient_accumulation_steps=4,
    temperature=0.7,
    kl_coef=0.1,  # KL penalty coefficient
)

# Create trainer
trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    env=env,
    config=config
)

# Train
trainer.train()
```

## Data Generation Pipeline

### Online Generation

Generate training data on-the-fly during training:

```python
class OnlineGRPOTrainer(GRPOTrainer):
    def get_batch(self):
        """Generate fresh batch of data."""
        # Sample prompts from dataset
        prompts = self.env.sample_prompts(self.config.batch_size)
        
        # Generate multiple completions per prompt
        all_completions = []
        all_rewards = []
        
        for prompt in prompts:
            # Generate group_size completions
            completions = []
            for _ in range(self.config.group_size):
                completion, _ = self.env.rollout(
                    client=self.client,
                    model=self.model,
                    prompt=prompt["prompt"],
                    answer=prompt["answer"],
                    temperature=self.config.temperature
                )
                completions.append(completion)
            
            # Score completions
            rewards = []
            for completion in completions:
                score = self.env.rubric.score_rollout_sync(
                    prompt=prompt["prompt"],
                    completion=completion,
                    answer=prompt["answer"],
                    state={}
                )
                rewards.append(score["reward"])
            
            all_completions.extend(completions)
            all_rewards.extend(rewards)
        
        return prompts, all_completions, all_rewards
```

### Offline Generation

Pre-generate large datasets for faster training:

```python
# Generate dataset
print("Generating training data...")
prompts, completions, rewards = env.generate(
    model="gpt-3.5-turbo",
    n_samples=10000,
    temperature=0.8,
    batch_size=100,
    max_concurrent=10
)

# Save for training
dataset = {
    "prompts": prompts,
    "completions": completions,
    "rewards": rewards
}
torch.save(dataset, "training_data.pt")

# Load in trainer
trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,  # Pre-generated data
    config=config
)
```

## Advanced Training Patterns

### Curriculum Learning

Gradually increase task difficulty:

```python
class CurriculumGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, difficulty_schedule, **kwargs):
        super().__init__(*args, **kwargs)
        self.difficulty_schedule = difficulty_schedule
    
    def get_current_difficulty(self, epoch, step):
        """Determine current difficulty level."""
        progress = epoch + step / self.num_steps_per_epoch
        
        for threshold, difficulty in self.difficulty_schedule:
            if progress < threshold:
                return difficulty
        
        return self.difficulty_schedule[-1][1]
    
    def get_batch(self, epoch, step):
        """Sample batch based on current difficulty."""
        difficulty = self.get_current_difficulty(epoch, step)
        
        # Filter dataset by difficulty
        filtered_data = [
            d for d in self.env.dataset 
            if d.get("difficulty", "easy") == difficulty
        ]
        
        # Sample from filtered data
        return self.sample_batch(filtered_data)

# Usage
trainer = CurriculumGRPOTrainer(
    model=model,
    env=env,
    difficulty_schedule=[
        (1.0, "easy"),    # First epoch: easy
        (2.0, "medium"),  # Second epoch: medium
        (3.0, "hard"),    # Third epoch: hard
    ],
    config=config
)
```

### Multi-Reward Training

Train with multiple objectives:

```python
from verifiers.rubrics import RubricGroup

# Create multiple rubrics for different objectives
correctness_rubric = Rubric(
    funcs=[exact_match_reward],
    weights=[1.0]
)

efficiency_rubric = Rubric(
    funcs=[length_penalty, time_penalty],
    weights=[0.5, 0.5]
)

style_rubric = Rubric(
    funcs=[clarity_reward, formality_reward],
    weights=[0.7, 0.3]
)

# Combine with weights
combined_rubric = RubricGroup([
    correctness_rubric,  # Most important
    efficiency_rubric,   # Secondary
    style_rubric        # Least important
])

# Set different weights for training phases
class AdaptiveGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, rubric_weights_schedule, **kwargs):
        super().__init__(*args, **kwargs)
        self.rubric_weights_schedule = rubric_weights_schedule
    
    def compute_rewards(self, completions, prompts, epoch):
        """Compute rewards with adaptive weights."""
        # Get current weights
        weights = self.rubric_weights_schedule.get(
            epoch, 
            self.rubric_weights_schedule['default']
        )
        
        # Temporarily update rubric weights
        original_weights = self.env.rubric.get_group_weights()
        self.env.rubric.set_group_weights(weights)
        
        # Compute rewards
        rewards = super().compute_rewards(completions, prompts)
        
        # Restore weights
        self.env.rubric.set_group_weights(original_weights)
        
        return rewards
```

### Reinforcement Learning from AI Feedback (RLAIF)

Use AI models for evaluation:

```python
from verifiers.rubrics import JudgeRubric

# Create judge-based environment
judge_rubric = JudgeRubric(
    judge_models=["gpt-4", "claude-2"],  # Multiple judges
    aggregation="mean",  # Average scores
    client=openai_client,
    template="""Evaluate this response for helpfulness and accuracy.

Question: {prompt}
Response: {completion}

Score from 0-10 and explain your reasoning.

<reasoning>
[Your evaluation here]
</reasoning>
<score>
[0-10]
</score>"""
)

env = SingleTurnEnv(
    dataset=dataset,
    rubric=judge_rubric
)

# Train with AI feedback
trainer = GRPOTrainer(
    model=model,
    env=env,
    config=config
)
```

## Training Configuration

### Key Hyperparameters

```python
config = GRPOConfig(
    # Learning
    learning_rate=1e-5,          # Model learning rate
    batch_size=32,               # Total batch size
    group_size=4,                # Samples per prompt for comparison
    gradient_accumulation_steps=4, # Effective batch = 32 * 4 = 128
    
    # Generation
    temperature=0.7,             # Sampling temperature
    max_new_tokens=512,          # Maximum generation length
    top_p=0.9,                   # Nucleus sampling
    
    # GRPO specific
    kl_coef=0.1,                # KL divergence penalty
    gamma=1.0,                  # Discount factor
    gae_lambda=0.95,            # GAE parameter
    
    # Optimization
    num_epochs=3,               # Training epochs
    warmup_ratio=0.1,           # LR warmup
    weight_decay=0.01,          # L2 regularization
    
    # Efficiency
    fp16=True,                  # Mixed precision training
    gradient_checkpointing=True, # Memory optimization
    
    # Logging
    logging_steps=10,           # Log frequency
    eval_steps=100,             # Evaluation frequency
    save_steps=500,             # Checkpoint frequency
)
```

### Memory Optimization

For large models:

```python
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model

# Use LoRA for parameter-efficient training
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)

# Use gradient checkpointing
model.gradient_checkpointing_enable()

# Initialize accelerator for distributed training
accelerator = Accelerator(
    mixed_precision="fp16",
    gradient_accumulation_steps=config.gradient_accumulation_steps,
)

# Prepare for training
model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)
```

## Evaluation During Training

### Periodic Evaluation

```python
class EvaluatingGRPOTrainer(GRPOTrainer):
    def evaluate(self):
        """Run evaluation on validation set."""
        self.model.eval()
        total_reward = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                prompts = batch["prompts"]
                answers = batch["answers"]
                
                # Generate completions
                completions = self.generate(prompts)
                
                # Score with rubric
                rewards = self.env.rubric.score_rollouts(
                    prompts=prompts,
                    completions=completions,
                    answers=answers,
                    states=[{} for _ in prompts]
                )
                
                total_reward += sum(rewards["reward"])
                num_samples += len(prompts)
        
        avg_reward = total_reward / num_samples
        print(f"Validation reward: {avg_reward:.4f}")
        
        self.model.train()
        return avg_reward
```

### Multi-Metric Tracking

```python
def comprehensive_evaluation(model, env, eval_dataset):
    """Evaluate model on multiple metrics."""
    metrics = {
        "reward": [],
        "format_compliance": [],
        "answer_accuracy": [],
        "response_length": [],
        "inference_time": []
    }
    
    for sample in eval_dataset:
        start_time = time.time()
        
        # Generate response
        completion, _ = env.rollout(
            model=model,
            prompt=sample["prompt"],
            answer=sample["answer"]
        )
        
        # Compute metrics
        scores = env.rubric.score_rollout_sync(
            prompt=sample["prompt"],
            completion=completion,
            answer=sample["answer"],
            state={}
        )
        
        metrics["reward"].append(scores["reward"])
        metrics["format_compliance"].append(scores.get("format", 0))
        metrics["answer_accuracy"].append(scores.get("correct_answer", 0))
        metrics["response_length"].append(len(completion))
        metrics["inference_time"].append(time.time() - start_time)
    
    # Aggregate metrics
    return {
        metric: np.mean(values) 
        for metric, values in metrics.items()
    }
```

## Production Training Pipeline

### Complete Training Script

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from verifiers.envs import SingleTurnEnv
from verifiers.trainers import GRPOTrainer, GRPOConfig
from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric
import wandb

def main():
    # Initialize wandb
    wandb.init(project="verifiers-training")
    
    # Load model and tokenizer
    model_name = "meta-llama/Llama-2-7b-hf"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create environment
    parser = XMLParser(["reasoning", "answer"])
    rubric = create_task_rubric(parser)  # Your custom rubric
    
    env = SingleTurnEnv(
        dataset=load_dataset("your_dataset"),
        parser=parser,
        rubric=rubric,
        system_prompt=SYSTEM_PROMPT
    )
    
    # Training configuration
    config = GRPOConfig(
        output_dir="./checkpoints",
        learning_rate=1e-5,
        batch_size=32,
        group_size=4,
        num_epochs=3,
        eval_steps=100,
        save_steps=500,
        logging_steps=10,
        warmup_ratio=0.1,
        fp16=True,
        gradient_checkpointing=True,
        report_to="wandb"
    )
    
    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        env=env,
        config=config,
        eval_dataset=load_eval_dataset()
    )
    
    # Train
    trainer.train()
    
    # Save final model
    trainer.save_model("./final_model")
    
    # Run final evaluation
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")

if __name__ == "__main__":
    main()
```

### Distributed Training

For multi-GPU training:

```python
# Launch with accelerate
# accelerate launch --multi_gpu --num_processes 4 train.py

from accelerate import Accelerator

accelerator = Accelerator()

# Modify trainer to use accelerator
class DistributedGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accelerator = accelerator
        
        # Prepare model and optimizer
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )
```

## Best Practices

### 1. Start Simple
- Begin with a simple rubric focusing on correctness
- Add complexity gradually as model improves
- Monitor for reward hacking

### 2. Validation Strategy
- Hold out test data with same distribution
- Use same environment for evaluation
- Track multiple metrics, not just reward

### 3. Hyperparameter Tuning
- Start with small group_size (2-4)
- Use lower learning rates than supervised fine-tuning
- Adjust KL coefficient based on divergence

### 4. Data Quality
- Ensure diverse prompts in dataset
- Balance difficulty levels
- Include edge cases and failure modes

### 5. Monitoring
- Track reward distribution, not just mean
- Monitor format compliance separately
- Watch for mode collapse

The verifiers framework makes it easy to go from environment design to trained models, providing a complete pipeline for reinforcement learning with human or AI feedback.