# Search-R1 Port Planning (precise)

## Objective
Port Search-R1 as a verifiers Environment with identical data format, rollout protocol, and reward semantics as the reference implementations. Only resource hosting may differ (retriever server, API keys), not data or scoring logic.

## Authoritative references
- Official repo: /tmp/Search-R1
  - Dataset processing: scripts/data_process/nq_search.py
  - Rollout core: search_r1/llm_agent/generation.py
  - Retriever API: search_r1/search/retrieval_server.py (local retriever), search_r1/search/serp_search_server.py (online SERP), search_r1/search/google_search_server.py (Serper via slime copy also matches)
  - Reward: verl/utils/reward_score/qa_em_format.py (and qa_em.py)
  - Inference demo: infer.py
- Slimed “lite” example: /tmp/slime/examples/search-r1
  - generate_with_search.py (loop logic/parsers match official)
  - qa_em_format.py (reward copy matches official)
  - google_search_server.py (Serper wrapper)

## Data: exact source and schema
- Source dataset: HuggingFace dataset RUC-NLPIR/FlashRAG_datasets with subset 'nq'.
- The official preprocessing (scripts/data_process/nq_search.py) constructs per-sample dicts:
  - data_source: 'nq'
  - prompt: a single ChatMessage: {role: 'user', content: make_prefix(question)}
    - make_prefix produces the full instruction with tag protocol and the question appended; identical string:
      """
      Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n
      """
    - question is stripped and ensured to end with '?'
  - reward_model: { style: 'rule', ground_truth: { target: golden_answers } }
  - extra_info: { split: 'train'|'test', index: idx }
- Verifiers mapping:
  - Keep prompt content byte-identical to make_prefix output.
  - Store ground truth under info.ground_truth = {'target': [...]} for the Rubric to consume without transformation.
  - Preserve data_source='nq' in info to mirror upstream selection hooks.

## Rollout protocol (must match Search-R1)
- Single-agent, multi-turn loop with interleaved “tool-less” tags produced by the model:
  - The assistant emits segments enclosed by tags: <think>...</think>, <search>query</search>, <information>...</information>, <answer>final</answer>.
  - Environment executes only when <search>…</search> appears; it responds by appending a user-side observation string: "\n\n<information>{passages}</information>\n\n".
  - Termination when <answer>…</answer> is emitted or max_turns is reached.
- Parsing (identical regex as upstream): r"<(search|answer)>(.*?)</\\1>" with DOTALL; extract last action/content per turn.
- Observation formatting: passages are concatenated exactly as in generation.py/_passages2string:
  - For each retrieved doc item in order: content = doc['document']['contents'] where contents = '"{title}"\n{text}'.
  - title = first line of contents without quotes; text = remaining lines joined by \n.
  - Emit: f"Doc {i+1}(Title: {title}) {text}\n"; then wrap the whole concatenation inside <information>…</information> as above.
- Max turns: configurable; official scripts use 2–3 in examples; training config exposes max_turns; we will expose max_turns with default 2 (per train_ppo.sh) and allow override.

## Search engine API (must match)
- Primary mode: local retriever server POST to /retrieve, identical payload and response:
  - Request JSON: { "queries": [str], "topk": int, "return_scores": true }
  - Response JSON: { "result": [ [ { "document": {"contents": '"Title"\nText'}, "score": float }, ... ] ] }
- Top-k: default 3, configurable; pass-through to request.
- Alternative mode (optional): online SERP proxy at /retrieve matching search_r1/search/serp_search_server.py semantics (constructs "Title" + snippet entries), or Serper.dev as in slime google_search_server.py. The Environment will accept a base_url; caller can deploy either service as long as it returns the same schema as above.

## Reward function (exact semantics)
- Implement qa_em_format.compute_score_em from official repo verbatim:
  - normalize_answer: lowercase, remove punctuation, articles, collapse whitespace.
  - is_valid_sequence: checks balanced <think|search|information|answer> tags and their order on the assistant side; official function expects "<|im_start|>assistant" anchor in decoded sequence. Since verifiers uses OpenAI-style messages, we will prepend this anchor when composing the scoring string to preserve identical semantics.
  - extract_solution: take the last <answer>…</answer> occurrence; return None if fewer than 2 matches (matches official behavior accounting for the initial prompt).
  - is_retrieval_correct: returns True if any normalized golden answer substring appears inside any <information> block.
  - Scoring table (weights configurable, defaults 0 unless set):
    - If answer None:
      - valid format + retrieval_correct: structure_format_score + retrieval_score
      - valid format only: structure_format_score
      - else: 0
    - If answer present and EM-correct:
      - valid format: score (typically 1)
      - invalid format: score - structure_format_score
    - If answer present but not EM-correct:
      - valid format + retrieval_correct: structure_format_score + retrieval_score
      - valid format only: structure_format_score
      - invalid format: final_format_score
- Also expose plain EM variant qa_em.compute_score_em for experiments; default to format-aware function but allow weights all-zero to replicate pure EM.
- Weight defaults:
  - Official ppo_trainer.yaml sets structure_format_score=0, final_format_score=0, retrieval_score=0; slime-lite uses format_score=0.2 only. We will parameterize all weights with defaults at 0 to match official training unless overridden.

## Tokenization detail for scoring string
- Compose scoring input string similarly to official trainer decode path:
  - Concatenate the serialized prompt and responses. To satisfy is_valid_sequence’s assistant-anchor expectation, build string as: f"<|im_start|>assistant{assistant_side_content}" where assistant_side_content is the concatenation of assistant outputs and environment <information> observations in chronological order. This preserves the same format checks without relying on HF chat templates.

## Verifiers design
- Environment: custom vf.MultiTurnEnv implementation (not ToolEnv) to mirror tag-parsing and string observations exactly.
  - State: turn counter, retriever config {url, topk}, dataset’s ground_truth, and accumulated messages.
  - is_completed: True if <answer> was emitted or turn >= max_turns.
  - env_response: parse last assistant segment; on <search>, call retriever, inject observation string; else on invalid action, inject the same corrective message as upstream:
    "\nMy previous action is invalid. If I want to search, I should put the query between <search> and </search>. If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n"
- Dataset loader in load_environment:
  - Load split='train' from RUC-NLPIR/FlashRAG_datasets, subset 'nq'.
  - Map each sample to the exact make_prefix prompt content and ground_truth mapping; do not alter golden_answers.
  - Optionally expose split selection and sampling count without changing content.
- Rubric:
  - Wrap the exact qa_em_format scoring in a vf.Rubric function that receives parser, completion (messages), and info.ground_truth; construct sequences_str with assistant anchor as above; apply weights from environment args.

## Configurables exposed by load_environment
- max_turns: int, default 2
- retriever_url: str, default "http://127.0.0.1:8000/retrieve"
- retriever_topk: int, default 3
- reward weights: structure_format_score=0.0, final_format_score=0.0, retrieval_score=0.0, score=1.0
- data_split: 'train'|'test', default 'train'
- data_limit: Optional[int] for quick runs (sampling only; content unchanged)

## Non-goals
- Do not modify question text, golden answers, or passage formatting.
- Do not introduce alternate prompt templates.
- No additional Python files unless asked; all logic lives in vf_search_r1.py.

## Implementation steps
1) Implement dataset loader exactly per scripts/data_process/nq_search.py (ensure question ends with '?', build make_prefix string, map golden_answers to ground_truth.target).
2) Implement MultiTurnEnv subclass with regex/action parsing and observation injection identical to generation.py and slime example; wire retriever POST request and passage formatting.
3) Implement Rubric with qa_em_format semantics; build sequences_str with assistant anchor to preserve format checks; expose weight args matching official.
4) Expose load_environment args per Configurables.
5) Style: uv run ruff format; quick smoke via vf-eval with a small n (no code prints added).