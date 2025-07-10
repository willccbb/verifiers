import verifiers as vf 
from openai import OpenAI
from datasets import load_dataset

model_name = "Qwen/Qwen2.5-7B-Instruct"
judge_prompt = "Q: {question}\nA: {answer}\nGiven: {response}\nRespond with a score between 0.0 and 1.0."
rubric = vf.JudgeRubric(client=OpenAI(base_url="http://0.0.0.0:8000/v1"), model=model_name, judge_prompt=judge_prompt)
vf_env = vf.SingleTurnEnv( 
    dataset=load_dataset("willcb/my-dataset", data_files="train"), # HF dataset with "question" and "answer" columns
    system_prompt="You are a helpful assistant.",
    rubric=rubric
)
model, tokenizer = vf.get_model_and_tokenizer(model_name)
trainer = vf.GRPOTrainer(env=vf_env, model=model, processing_class=tokenizer, args=vf.grpo_defaults(run_name="self_reward"))
trainer.train()