from datasets import load_dataset
import verifiers as vf

#model = 'Qwen/Qwen2.5-1.5B-Instruct'
"""
inference:
CUDA_VISIBLE_DEVICES=0 vf-vllm --model willcb/Qwen2.5-0.5B-Reverse-SFT

training:
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 --config-file configs/zero3.yaml verifiers/examples/reverse_text.py
"""


model_name = 'willcb/Qwen2.5-0.5B-Reverse-SFT'
dataset = load_dataset('agentlans/wikipedia-paragraphs', split='train').map(lambda x: {'question': x['text'], 'answer': x['text'][::-1]})
# evaluate on the first 32 examples, train on the rest
eval_dataset = dataset.select(range(32)) # type: ignore
train_dataset = dataset.select(range(32, len(dataset))) # type: ignore

parser = vf.XMLParser(['think', 'answer'], answer_field='answer')
system_prompt = f"""Reverse the given text.

Respond in the following format:
{parser.get_format_str()}"""

def lcs_reward_func(completion, answer, **kwargs) -> float:
    """
    LCS ratio of the reversed prompt and the parsed completion.
    """
    def lcs_ratio(x: str, y: str) -> float:
        """
        Return the longest common subsequence ratio of x and y.
        """
        from difflib import SequenceMatcher
        return SequenceMatcher(None, x, y).ratio()
    response = parser.parse_answer(completion) or ''
    return lcs_ratio(response, answer)

rubric = vf.Rubric(funcs=[
	lcs_reward_func,
	parser.get_format_reward_func(),
], weights=[1.0, 0.2])

vf_env = vf.SingleTurnEnv(
    dataset=train_dataset, # type: ignore
    eval_dataset=eval_dataset, # type: ignore
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric
)
args = vf.grpo_defaults(run_name='reverse_text_warmup')
args.num_iterations = 2
args.per_device_train_batch_size = 10
args.num_generations = 10
args.gradient_accumulation_steps = 4
args.eval_strategy = "steps"
args.eval_steps = 10
args.max_steps = 100

model, tokenizer = vf.get_model_and_tokenizer(model_name)
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    #peft_config=vf.lora_defaults(),
    args=args
)
trainer.train()