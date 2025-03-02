from datasets import Dataset, load_dataset
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024)

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", tensor_parallel_size=2)

SYSTEM_PROMPT = """
Respond in the following format, using careful step-by-step thinking:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_reasoning(text: str) -> str:
    reason = text.split("<reasoning>")[-1]
    reason = reason.split("</reasoning>")[0]
    return reason.strip()

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")

def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': 'What is the largest single-digit prime number?'},
            {'role': 'assistant', 'content': XML_COT_FORMAT.format(
                reasoning="9 is divisble by 3 and 8 is divisible by 2, but 7 is prime.",
                answer="7"
            )},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

dataset = get_gsm8k_questions()

messages = dataset[0]['prompt']
response = llm.chat(messages, sampling_params)[0]
prompt = response.prompt
prompt_ids = response.prompt_token_ids
output = response.outputs[0].text
output_ids = response.outputs[0].token_ids
print("prompt:", prompt)
print("prompt ids:", prompt_ids)
print("output:", output)
print("output ids:", output_ids)