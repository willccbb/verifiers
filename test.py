from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import time

#MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # Or your specific model
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"

print("--- Testing /v1/models ---")
client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
print(client.models.list())

print("--- Testing /v1/chat/completions ---")
messages = [    
    {"role": "user", "content": "What is the capital of France?"}
]
print("input:", messages)
chat_completion = client.chat.completions.create(
    model=MODEL_NAME, messages=messages
)
print(chat_completion)
print(f"Chat: {chat_completion.choices[0].message.content}")

print("--- Testing throughput ---")

# test throughput for 20 parallel requests via ThreadPoolExecutor + randomized prompts + chat completions

# prompts: Give 3 fun facts about the country
prompts = [
    "Give 3 fun facts about France",
    "Give 3 fun facts about Germany",
    "Give 3 fun facts about Italy",
    "Give 3 fun facts about Spain",
    "Give 3 fun facts about Portugal",
    "Give 3 fun facts about Japan",
    "Give 3 fun facts about China",
    "Give 3 fun facts about India",
    "Give 3 fun facts about Brazil",
    "Give 3 fun facts about Mexico",
    "Give 3 fun facts about Canada",
    "Give 3 fun facts about Australia",
    "Give 3 fun facts about Russia",
    "Give 3 fun facts about South Korea",
    "Give 3 fun facts about United Kingdom",
    "Give 3 fun facts about Netherlands",
    "Give 3 fun facts about Sweden",
    "Give 3 fun facts about Norway",
    "Give 3 fun facts about Denmark",
    "Give 3 fun facts about Finland"
]
messages = [
    {"role": "user", "content": prompt}
    for prompt in prompts
]

def test_throughput(max_workers=20):
    start_time = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(client.chat.completions.create, model=MODEL_NAME, messages=messages[i:i+1]) for i in range(20)]
        for i, future in enumerate(futures):
            completion = future.result()
            results.append(completion.choices[0].message.content)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds for {max_workers} workers with {len(prompts)} prompts")
    return results

#test_throughput(max_workers=1)
test_throughput(max_workers=2)
test_throughput(max_workers=4)
test_throughput(max_workers=8)
test_throughput(max_workers=16)
results = test_throughput(max_workers=20)

