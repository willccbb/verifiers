import verifiers as vf
from verifiers.tools.calculator import calculator
from verifiers.prompts import CALCULATOR_FEW_SHOT
import torch 

model_name = "microsoft/Phi-4-mini-instruct"

model_kwargs = dict(
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    use_cache=False, # needed for phi4 and vllm to work
    device_map="cuda", # must use in order to avoid `model.to('cuda')` error from Flash Atention
)
model, tokenizer = vf.get_model_and_tokenizer(model_name, model_kwargs=model_kwargs)

# Initialize tool environment for GSM8K
vf_env = vf.ToolEnv(
    dataset="gsm8k",
    few_shot=CALCULATOR_FEW_SHOT[0],
    tools=[calculator],
    max_steps=5,
    sampling_args={"stop": ["</tool>", "</answer>"], "include_stop_str_in_output": True}
)
dataset = vf_env.get_dataset()
eval_dataset = vf_env.get_eval_dataset(n=100)
rubric = vf_env.get_rubric()

# notable defaults: lr = 1e-6, max_grad_norm = 0.01, constant lr 10 warmup steps, 1024 tokens in+out
training_args = vf.get_default_grpo_config(
    run_name="math_calculator_phi-4-mini-instruct",
    num_gpus=2
)
# rollouts per prompt
training_args.num_generations = 2
# minibatch size per GPU ( bs 8 * 7 gpus / 7 rollouts -> 8 prompts per batch)
training_args.per_device_train_batch_size = 2
# batches to accumulate (6 prompts * 4 -> 32 prompts per global batch)
training_args.gradient_accumulation_steps = 4
# steps per global batch (1 on-policy, 1 off-policy)
training_args.num_iterations = 2
# no ref model
training_args.beta = 0.0
# evals
training_args.eval_strategy = "steps"
training_args.eval_on_start = True
training_args.eval_steps = 100
training_args.per_device_eval_batch_size = 2
training_args.eval_accumulation_steps = 1

trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric,
    env=vf_env,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
)

trainer.train() 