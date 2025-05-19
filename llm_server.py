"""
async_batch_openai.py
————————————————————————————————————————
FastAPI + Uvicorn server that

* loads a single vLLM `LLM()` worker (blocking API);
* accepts OpenAI-compatible `/v1/completions` and `/v1/chat/completions`;
* pools incoming requests into an asyncio queue;
* every `BATCH_WINDOW_MS` (or sooner if `MAX_BATCH_SIZE` reached) it calls
  `llm.generate(prompts)` once, then routes the N results back to the
  individual HTTP callers.

Test quickly:

    MODEL="Qwen/Qwen2.5-1.5B-Instruct" \
    uvicorn async_batch_openai:app --host 0.0.0.0 --port 8000

"""

import asyncio, os, time
from dataclasses import dataclass
from typing import List, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams

# ────────────── CONFIG  ─────────────────────────────────────────────

MODEL             = os.getenv("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
TP                = int(os.getenv("TP", 1))
GPU_UTIL          = float(os.getenv("GPU_UTIL", 0.9))

MAX_BATCH_SIZE    = int(os.getenv("MAX_BATCH_SIZE", 32))
BATCH_WINDOW_MS   = int(os.getenv("BATCH_WINDOW_MS", 100))      # 0 → no wait

# Default sampling – can be overridden per-request
DEFAULT_PARAMS = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=64)

# ────────────── MODELS  ─────────────────────────────────────────────

class CompletionRequest(BaseModel):
    prompt: str
    sampling_params: dict[str, Any] | None = None

class ChatCompletionMessage(BaseModel):
    role: str = Field(..., pattern="system|user|assistant")
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatCompletionMessage]
    sampling_params: dict[str, Any] | None = None

# ────────────── INTERNAL  ───────────────────────────────────────────

@dataclass
class _Task:
    prompt: str
    sampling: SamplingParams
    fut: asyncio.Future   # where we put the string result

_request_q: "asyncio.Queue[_Task]" = asyncio.Queue()

# ────────────── LLM LOAD  ───────────────────────────────────────────

llm = LLM(
    model=MODEL,
    tensor_parallel_size=TP,
    gpu_memory_utilization=GPU_UTIL,
    # graphs give +5-10 % throughput; safe with blocking LLM
)

# ────────────── BATCHER  ────────────────────────────────────────────

async def _batch_loop():
    """Gather tasks every BATCH_WINDOW_MS, run llm.generate once."""
    while True:
        task = await _request_q.get()
        batch   = [task]
        started = time.time()

        # small grace window to accumulate extra requests
        while len(batch) < MAX_BATCH_SIZE:
            try:
                timeout = max(0, BATCH_WINDOW_MS/1000 - (time.time() - started))
                # 0 timeout => poll instantly; >0 wait until window expires
                nxt = await asyncio.wait_for(_request_q.get(), timeout=timeout)
                batch.append(nxt)
            except asyncio.TimeoutError:
                break

        prompts  = [t.prompt for t in batch]
        samps    = [t.sampling for t in batch]

        # vLLM expects *one* SamplingParams – we split by identical fields.
        # Minimal implementation: **force identical sampling per batch**.
        if not all(s == samps[0] for s in samps):
            for t in batch:
                t.fut.set_exception(
                    HTTPException(400, "Sampling params must match current batch"))
            continue

        outs = llm.generate(prompts, samps[0])

        # map results back
        for t, o in zip(batch, outs):
            t.fut.set_result(o.outputs[0].text)

asyncio.create_task(_batch_loop())

# ────────────── FASTAPI  ────────────────────────────────────────────

app = FastAPI()

@app.get("/v1/models")
def models():
    return {"data": [{"id": MODEL, "object": "model"}], "object": "list"}

@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    future: asyncio.Future = asyncio.get_event_loop().create_future()
    _request_q.put_nowait(_Task(
        prompt=req.prompt,
        sampling=_merge_sampling(req.sampling_params),
        fut=future,
    ))
    res = await future
    return {"id": "cmpl", "object": "text_completion", "choices":[{"text": res}]}

@app.post("/v1/chat/completions")
async def chat(req: ChatCompletionRequest):
    prompt = _chat_to_prompt(req.messages)
    future: asyncio.Future = asyncio.get_event_loop().create_future()
    _request_q.put_nowait(_Task(
        prompt=prompt,
        sampling=_merge_sampling(req.sampling_params),
        fut=future,
    ))
    res = await future
    return {"id": "chatcmpl", "object": "chat.completion",
            "choices":[{"message":{"role":"assistant","content":res}}]}

# ────────────── HELPERS  ────────────────────────────────────────────
def _merge_sampling(override: dict[str, Any] | None) -> SamplingParams:
    if not override:
        return DEFAULT_PARAMS
    merged = DEFAULT_PARAMS.model_copy()   # pydantic v2
    for k,v in override.items():
        setattr(merged, k, v)
    return merged

def _chat_to_prompt(messages: List[ChatCompletionMessage]) -> str:
    # minimal: OpenAI style "<role>: <content>\n"
    return "\n".join(f"{m.role}: {m.content}" for m in messages) + "\nassistant:"

# ────────────────────────────────────────────────────────────────────
# No if __name__ guard needed; run with:  uvicorn async_batch_openai:app
