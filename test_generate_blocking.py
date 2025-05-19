# script to test throughput of vLLM LLM() object with ThreadPoolExecutor

import time
from concurrent.futures import ThreadPoolExecutor
from vllm import AsyncLLM

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
llm = AsyncLLM(model=model_name)

def test_generate_blocking(max_workers=20):
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(llm.generate, "Hello, world!") for i in range(20)]
        for i, future in enumerate(futures):
            completion = future.result()

test_generate_blocking(max_workers=1)
test_generate_blocking(max_workers=4)
test_generate_blocking(max_workers=8)
test_generate_blocking(max_workers=16)
