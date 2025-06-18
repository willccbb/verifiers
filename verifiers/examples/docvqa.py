import re

from datasets import load_dataset
from qwen_vl_utils import process_vision_info

import verifiers as vf

"""
# install qwen stuff
uv pip install qwen-vl-utils
# inference
CUDA_VISIBLE_DEVICES=0,1,2,3 vf-vllm --model 'Qwen/Qwen2.5-VL-7B-Instruct' --max-model-len 32000 --tensor_parallel_size 4 
# train
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config-file configs/zero3.yaml --num-processes 4 verifiers/examples/docvqa.py
"""


def data_collator(batch: list[dict]) -> list[dict]:
    processed_samples = []
    for sample in batch:
        messages = []
        messages.append({"role": "system", "content": system_prompt})
        content_block = []
        content_block.append({"type": "text", "text": sample["question"]})
        content_block.append(
            {
                "type": "image",
                "image": sample["image"],  # only one image in this ds
                "resized_height": 768,  # XGA resolution
                "resized_width": 1024,
            }
        )
        messages.append({"role": "user", "content": content_block})
        processed_images, *_ = process_vision_info(  # process with qwen utils
            messages.copy()
        )
        sample["prompt"] = messages
        sample["images"] = processed_images
        sample["answer"] = sample["answers"]
        processed_samples.append(sample)
    return processed_samples


dataset = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation[10%:]")
eval_dataset = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation[:10%]")

parser = vf.XMLParser(["think", "answer"], answer_field="answer")
system_prompt = f"""Answer the questions.

Respond in the following format:
{parser.get_format_str()}"""


def correctness_reward_func(completion: list[dict[str, str]], **kwargs) -> float:
    def get_assistant_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        return [msg for msg in messages if msg.get("role") == "assistant"]

    def parse_xml_content(text: str, tag: str, strip: bool = True) -> str | None:
        pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            content = match.group(1)
            return content.strip() if strip else content
        return None

    assistant_messages = get_assistant_messages(completion)
    if assistant_messages is None:
        return 0.0
    msgs_scores = []
    for msg in assistant_messages:
        content = msg.get("content", "")
        answer = parse_xml_content(content, "answer")
        if answer is None:
            continue
        gt_answers = kwargs["answer"]
        mean_gt_len = sum([len(gt_answer) for gt_answer in gt_answers]) / len(
            gt_answers
        )
        if len(answer) > 0:
            diff_from_mean = min(mean_gt_len / len(answer), 1.0)  # penalize long answers
        else:
            diff_from_mean = 0.0
        if answer in gt_answers:
            msgs_scores.append(2.0)
        elif answer.lower() in [ans.lower() for ans in gt_answers]:
            msgs_scores.append(1.0)
        elif any(ans.lower() in answer.lower() for ans in gt_answers):
            msgs_scores.append(diff_from_mean)
    if msgs_scores == []:
        return 0.0
    else:
        return sum(msgs_scores) / len(msgs_scores) / 2.0


rubric = vf.Rubric(
    funcs=[
        parser.get_format_reward_func(),
        correctness_reward_func,
    ]
)

vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    eval_dataset=eval_dataset,
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric,
    data_collator=data_collator,
)

model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
model, processor = vf.get_model_and_tokenizer(model_name)
run_name = "docvqa_" + model_name.split("/")[-1].lower()

training_args = vf.grpo_defaults(run_name=run_name)
training_args.learning_rate = 3e-6
training_args.max_steps = -1
training_args.eval_strategy = "steps"
training_args.eval_steps = 100
training_args.gradient_checkpointing_kwargs = {
    "use_reentrant": False,
}

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=processor,
    env=vf_env,
    args=training_args,
)
trainer.train()
