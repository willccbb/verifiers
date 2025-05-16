from datasets import load_dataset
import verifiers as vf

#model = 'Qwen/Qwen2.5-1.5B-Instruct'
model_name = 'willcb/Qwen2.5-1.5B-Reverse-SFT'
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

model, tokenizer = vf.get_model_and_tokenizer(model_name)
trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    peft_config=vf.lora_defaults(),
    args=vf.grpo_defaults(run_name='reverse_text_warmup') # TRL GRPOConfig, modify as needed
)
trainer.train()


