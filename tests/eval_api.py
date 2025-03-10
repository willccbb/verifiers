import verifiers as vf
from verifiers.prompts import CODE_PROMPT, CODE_FEW_SHOT

import os
from openai import OpenAI
model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
api_key = os.getenv("DEEPINFRA_API_KEY")
base_url = "https://api.deepinfra.com/v1/openai"

client = OpenAI(api_key=api_key, base_url=base_url)
vf_env = vf.CodeEnv(
    dataset="gsm8k",
    few_shot=CODE_FEW_SHOT[0],
    system_prompt=CODE_PROMPT + "\n\nYour final answer should be an integer."
)
vf_env.eval_api(client, model_name)
