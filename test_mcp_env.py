from verifiers.envs.mcp_env import MCPEnv

from openai import OpenAI

client = OpenAI()
model = "gpt-4.1-mini"
endpoints = {
    "calculator": "http://localhost:8004",
    "dictionary": "http://localhost:8005"
}
env = MCPEnv(mcp_endpoints=endpoints)

prompt = [{'role': 'user', 'content': 'Test out the tools'}]
answer = "answer"

env.rollout(client, model, prompt, answer)