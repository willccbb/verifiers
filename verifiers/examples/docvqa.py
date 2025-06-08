import re

from datasets import load_dataset

import verifiers as vf
from qwen_vl_utils import smart_resize

"""
# install qwen stuff
uv pip install qwen-vl-utils
# inference
CUDA_VISIBLE_DEVICES=0 uv run vf-vllm --model 'Qwen/Qwen2.5-VL-3B-Instruct' --max-model-len 64000
# train
CUDA_VISIBLE_DEVICES=1 uv run accelerate launch verifiers/examples/docvqa.py
TODO:
    - check liger kernel support
    - check that generic_model_loader didn't break anything
    - check completions format, not just chat
    - what happens if dataset is already formatted?
    - fix wandb log
NOTES:
    - transformers seems to be having an issue, so it's pinned for now: https://github.com/volcengine/verl/issues/1710
"""


def preprocess_docvqa(x):
    return {
        "question": x["question"],
        "images": [x["image"].resize(smart_resize(768, 1024))], # XGA
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
        diff_from_mean = min(mean_gt_len / len(answer), 1.0)  # penalize long answers
        if answer in gt_answers:
            msgs_scores.append(2.0)
        elif answer.lower() in [ans.lower() for ans in gt_answers]:
            msgs_scores.append(1.0)
        elif any(ans.lower() in answer.lower() for ans in gt_answers):
            msgs_scores.append(diff_from_mean)
    if msgs_scores == []:
        return 0.0
    else:
        return sum(msgs_scores) / len(msgs_scores)


rubric = vf.Rubric(
    funcs=[
        parser.get_format_reward_func(),
        correctness_reward_func,
    ]
)

vf_env = vf.SingleTurnEnv(
    dataset=dataset, system_prompt=system_prompt, parser=parser, rubric=rubric
)

model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
model, processor = vf.get_model_and_processor(model_name, use_liger=False)
run_name = "docvqa_" + model_name.split("/")[-1].lower()

training_args = vf.grpo_defaults(run_name=run_name)
training_args.log_completions = True
training_args.num_train_epochs = 2

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=processor,
    env=vf_env,
    args=training_args,
)
trainer.train()
