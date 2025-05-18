from datasets import load_dataset
import verifiers as vf

#model = 'Qwen/Qwen2.5-1.5B-Instruct'
model_name = "willcb/Qwen2.5-7B-Reverse-SFT"
dataset = load_dataset('agentlans/wikipedia-paragraphs', split='train').map(lambda x: {'question': x['text'], 'answer': x['text'][::-1]})
parser = vf.XMLParser(['think', 'answer'], answer_field='answer')
system_prompt = f"""Reverse the given text.

Respond in the following format:
{parser.get_format_str()}"""

def lcs_reward_func(completions, answer, **kwargs) -> list[float]:
    """
    LCS ratio of the reversed prompt and the parsed completion.
    """
    def lcs_ratio(x: str, y: str) -> float:
        """
        Return the longest common subsequence ratio of x and y.
        """
        from difflib import SequenceMatcher
        return SequenceMatcher(None, x, y).ratio()
    responses = [parser.parse_answer(c) or '' for c in completions]
    return [lcs_ratio(r, a) for r, a in zip(responses, answer)]

rubric = vf.Rubric(funcs=[
	lcs_reward_func,
	parser.get_format_reward_func(),
], weights=[1.0, 0.2])

vf_env = vf.SingleTurnEnv(
    dataset=dataset, # type: ignore
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric
)

run_name = "rev_warmup_grpo_fft_gr=1e-5_delta=4_beta=1e-2"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

args = vf.GRPOEnvConfig(
    output_dir=f"outputs/{run_name}",
    run_name=run_name,
    learning_rate=1e-6,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=10,
    num_train_epochs=1,
    max_steps=1000,
    bf16=True,
    beta=0.001,
    sync_ref_model=True,
    ref_model_sync_steps=100,
    ref_model_mixup_alpha=0.5,
    max_grad_norm=0.001,
    num_iterations=2,
    max_prompt_length=1024,
    max_completion_length=2048,
    per_device_train_batch_size=16,
    num_generations=8,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    save_strategy="steps",
    save_steps=500,
    save_only_model=True,
    use_vllm=True,
    logging_steps=1,
    log_on_each_node=False,
    log_completions=True,
    report_to="wandb",
)

trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=args
)
trainer.train()
