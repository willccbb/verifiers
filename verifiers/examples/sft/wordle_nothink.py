import verifiers as vf
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer, SFTConfig

"""
accelerate launch --config-file configs/zero3.yaml --num-processes 8 verifiers/examples/sft/wordle_nothink.py
"""

# convenience function for FA2 initialization
model, tokenizer = vf.get_model_and_tokenizer("willcb/Qwen3-1.7B", use_liger=False)

def strip_nothink_from_prompt(x):
    x['prompt'] = [
        {'role': 'system', 'content': x['prompt'][0]['content'].replace('/no_think', '')}
    ] + x['prompt'][1:]
    return x

dataset_v1 = load_dataset('willcb/V3-wordle-nothink', split='train')
dataset_v2 = load_dataset('willcb/V3-wordle-nothink-100', split='train')
dataset_v3 = load_dataset('willcb/mini-wordle-nothink-100', split='train')

dataset = concatenate_datasets([dataset_v1, dataset_v2, dataset_v3]) # type: ignore
dataset = dataset.map(strip_nothink_from_prompt)

tok_counts = []
for row in dataset:
    # count tokens in (prompt, completion)
    messages = row['prompt'] + row['completion'] # type: ignore
    toks = tokenizer.apply_chat_template( 
        messages,
        tokenize=True
    )
    tok_counts.append(len(toks))

# tok count stats
print(f"Dataset size: {len(tok_counts)}")
print(f"Min tokens: {min(tok_counts)}")
print(f"Max tokens: {max(tok_counts)}")
print(f"Mean tokens: {sum(tok_counts) / len(tok_counts)}")
print(f"Median tokens: {sorted(tok_counts)[len(tok_counts) // 2]}")

args = SFTConfig(
    max_length=1024,
    output_dir="sft-wordle-nothink",
    per_device_train_batch_size=12,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    bf16=True,
    learning_rate=2e-5,
    num_train_epochs=10,
    weight_decay=0.01,
    max_grad_norm=0.1,
    report_to="wandb",
    save_strategy="steps",
    save_steps=300,
    logging_steps=1,
    save_only_model=True,
    log_on_each_node=True,
    push_to_hub=True,
    hub_model_id="Qwen3-1.7B-Wordle",
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset # type: ignore
)
trainer.train()