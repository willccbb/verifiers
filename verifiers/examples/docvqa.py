from datasets import load_dataset
import verifiers as vf

"""
CUDA_VISIBLE_DEVICES=0 uv run vf-vllm --model 'Qwen/Qwen2.5-VL-3B-Instruct' 
CUDA_VISIBLE_DEVICES=1 uv run accelerate launch verifiers/examples/docvqa.py
TODO:
    - check liger kernel support
    - check that generic_model_loader didn't break anything
"""

def preprocess_docvqa(x):
    return {
        "question": x["question"],
        "images": [x["image"]],
        "answer": x["answers"][0],
    }


dataset = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
dataset = dataset.map(
    preprocess_docvqa, num_proc=10, remove_columns=dataset.column_names
)

parser = vf.XMLParser(["think", "answer"], answer_field="answer")
system_prompt = f"""Answer the questions.

Respond in the following format:
{parser.get_format_str()}"""

rubric = vf.Rubric(
    funcs=[
        parser.get_format_reward_func(),
    ]
)

vf_env = vf.SingleTurnEnv(
    dataset=dataset, system_prompt=system_prompt, parser=parser, rubric=rubric
)

model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
model, processor = vf.get_model_and_processor(model_name, use_liger=False) # TODO: modify model loading to add liger support
run_name = "docvqa_" + model_name.split("/")[-1].lower()

training_args = vf.grpo_defaults(run_name=run_name)
training_args.per_device_train_batch_size = 4
training_args.num_generations = 4

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=processor,
    env=vf_env,
    args=training_args,
)
trainer.train()
