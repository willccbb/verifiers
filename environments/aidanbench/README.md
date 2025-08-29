## AidanBench (Verifiers Environment)

Multi-turn environment that reproduces AidanBench’s creativity loop in the Verifiers framework.

Behavior:
- Prompts the model with an open-ended question and requests one concise answer in `<answer></answer>` tags.
- After each model reply, evaluates coherence (LLM judge `o1-mini`) and novelty (OpenAI embeddings cosine distance) against previous accepted answers.
- Optionally uses an LLM similarity judge for novelty (`o1-mini`).
- Stops when any threshold fails. Reward equals the number of valid answers produced.

Requirements:
- `OPENAI_API_KEY` for embeddings (configurable)
- `OPEN_ROUTER_KEY` for judge if using OpenRouter (configurable)
- A local clone of `AidanBench` is NOT required. The environment can:
  - Use your provided dataset (with a `question` column), or
  - Accept a questions list via `-a '{"questions": [...]}'`, or
  - Load from a local file via `-a '{"questions_path": "path/to/questions.json|.txt"}'`, or
  - Use the bundled canonical AidanBench question list by default, or
  - Fall back to a small built-in set if nothing is provided.

Install locally:
```
uv run vf-install aidanbench -p ./environments
```

Quick eval:
```
uv run vf-eval aidanbench -m gpt-4.1-mini -n 3 -r 3 -T 0.7 \
  -a '{"use_llm_similarity": false, "chain_of_thought": false, "num_questions": 10}'
```

Generation model and base URL come from `vf-eval` flags:
- `-m/--model` (e.g., `gpt-4.1-mini`)
- `-b/--api-base-url` (e.g., `https://api.openai.com/v1` or your vLLM server)

Configurable Args (JSON via `-a`):
- `questions` (list[str]): provide your own list of questions.
- `questions_path` (str): path to JSON (list of strings) or text file (one question per line).
- `judge_model` (str): default `"o1-mini"`. Set to your judge model.
- `judge_api_base_url` (str): default `"https://openrouter.ai/api/v1"`.
- `judge_api_key_var` (str): env var for judge API key, default `"OPEN_ROUTER_KEY"`.
- `embedding_model` (str): default `"text-embedding-3-large"`.
- `embedding_api_base_url` (str): default `"https://api.openai.com/v1"`.
- `embedding_api_key_var` (str): env var for embeddings API key, default `"OPENAI_API_KEY"`.
- `thresholds` (dict): `{coherence_score, embedding_dissimilarity_score, llm_dissimilarity_score}`.

Example (custom judge + base URLs):
```
uv run vf-eval aidanbench -m gpt-4.1-mini -n 2 -r 2 \
  -a '{
        "judge_model": "openai/o1-mini",
        "judge_api_base_url": "https://openrouter.ai/api/v1",
        "judge_api_key_var": "OPEN_ROUTER_KEY",
        "embedding_model": "text-embedding-3-large",
        "embedding_api_base_url": "https://api.openai.com/v1",
        "embedding_api_key_var": "OPENAI_API_KEY"
      }'
```

Notes:
- Defaults follow AidanBench CLI thresholds: `coherence_score=15`, `embedding_dissimilarity_score=0.15`, `llm_dissimilarity_score=0.15`.
- Provide a custom HF dataset with a `question` field if you don’t have the `AidanBench` repo available; the environment will use the dataset as-is.
