# Search-R1 Port Planning (revised for ToolEnv)

## Objective
Replicate Search-R1 data and reward semantics while leveraging verifiers patterns. We will use ToolEnv with Python-wrapped tools (function-calling), keep the official dataset and the EM(+format/retrieval) reward, and minimally adapt prompts/loop to fit ToolEnv.

## Ground truth references (unchanged)
- Dataset: RUC-NLPIR/FlashRAG_datasets (subset 'nq'); preprocessing spec in /tmp/Search-R1/scripts/data_process/nq_search.py
- Rollout logic: /tmp/Search-R1/search_r1/llm_agent/generation.py and slime’s generate_with_search.py
- Retriever API shape: /tmp/Search-R1/search_r1/search/retrieval_server.py and serp_search_server.py; slime’s google_search_server.py
- Reward: /tmp/Search-R1/verl/utils/reward_score/qa_em_format.py (and qa_em.py)

## Strategic deviations (aligned with verifiers)
- Use ToolEnv instead of a bespoke MultiTurnEnv. Tools are plain Python functions with type hints/docstrings; verifiers will expose them via JSON schema.
- Keep dataset content identical to Search-R1’s make_prefix(question) prompt string and ground truth structure. No changes to data fields or text content.
- Keep reward identical to qa_em_format semantics. Tool returns will be formatted to supply <information>…</information> blocks so the reward logic remains valid without relying on <search> tags.

## Dataset (unchanged content)
- Load RUC-NLPIR/FlashRAG_datasets 'nq' split.
- For each example:
  - Ensure question ends with '?'
  - Build prompt content using the exact make_prefix template from nq_search.py
  - Store info.ground_truth = { 'target': golden_answers }
  - Store info.data_source = 'nq' (for potential downstream compatibility)

## ToolEnv design
- Tools
  - search(query: str, topk: int = 3) -> str
    - Calls a retriever endpoint at retriever_url (/retrieve) with payload {queries: [query], topk, return_scores: True}
    - Response: {result: [[{document: {contents: '"Title"\nText'}, score: float}, …]]}
    - Formats passages exactly as in generation.py/_passages2string and returns a single string wrapped in <information>…</information>:
      - For each doc: Doc {i+1}(Title: {title}) {text}\n
  - Optionally expose search_snippets(query: str, topk: int = 3) -> str for SERP-only mode (SerpAPI/Serper); returns <information> built from title+snippet like serp_search_server.py.

- System prompt
  - Instruct the model to:
    - Think within <think>…</think>
    - Use the search tool when lacking knowledge
    - Provide the final answer within <answer>…</answer> without extra explanation
  - We DO NOT require the model to emit <search>…</search> tags; tool calls replace that. This remains compatible with qa_em_format (it tolerates missing <search> when other tags are valid).

- Rollout behavior under ToolEnv
  - The model invokes the search tool as needed; ToolEnv injects the returned string as a tool result message that includes <information>…</information> content.
  - The model continues reasoning and eventually emits <answer>…</answer>.
  - max_turns enforced by ToolEnv config.

## Reward semantics (unchanged)
- Implement qa_em_format.compute_score_em exactly:
  - normalize_answer; extract_solution (last <answer>); is_valid_sequence; is_retrieval_correct (checks golden answer substrings inside <information> blocks)
  - Scoring weights: structure_format_score, final_format_score, retrieval_score, score (default all zeros except score=1.0 to match official defaults; can be configured)
- Sequence string for reward:
  - Build by concatenating prompt + assistant/tool contents into one string and prefixing with <|im_start|>assistant before assistant-side content so is_valid_sequence behaves as in the reference.

## Retriever modes and indexing
- Local retriever (fully local, deterministic)
  - Matches official: FastAPI server with /retrieve that serves passages from a local corpus and FAISS/BM25 index.
  - Corpus: Wikipedia subset JSONL (e.g., wiki-18.jsonl) where each line has {id, contents: '"Title"\nText'}.
  - Index: prebuilt FAISS (e5_Flat.index) or BM25 via Pyserini. Official provides build_index.sh and retrieval_server.py covering both sparse and dense.
  - Recommended for reproducible evals and offline use.

- Online SERP proxy (web search)
  - Matches official alternatives: serp_search_server.py; slime uses Serper.dev (google_search_server.py) with snippet_only or snippet+page-fetch.
  - Returns the same schema so our tool formatter remains identical.
  - Useful if you want fresh web results beyond the static wiki corpus.

- What makes most sense?
  - For faithful, repeatable evaluation: Prefer the local retriever with prebuilt Wikipedia corpus + index. This mirrors the official training/eval setup and avoids API variability.
  - For quick demos or broader coverage: Use SERP proxy (SerpAPI/Serper) and return snippets only. This is simplest to set up but less deterministic.
  - We will parameterize retriever_url and a mode flag (local | serp) and keep the return schema identical.

## Why ToolEnv here
- Pros
  - Zero custom parsing: tool JSON schema is auto-generated and supported by many models.
  - Clear separation of concerns: tools handle I/O; reward reads messages; dataset remains untouched.
  - Extensible: can add rerankers or alternate retrievers by swapping tool function implementations without changing env logic.
- Considerations
  - Some base models may not natively support tool calls; for them, a MultiTurnEnv with tag-based parsing might be closer to the reference. We can keep the ToolEnv-first design and document a fallback.
  - The qa_em_format validator tolerates absence of <search> tags; we depend on <think>, <information>, and <answer>. Our tool returns include <information> to preserve retrieval checks.

## Configurables (load_environment)
- max_turns: int (default 2)
- retriever_url: str (default "http://127.0.0.1:8000/retrieve")
- retriever_mode: Literal['local', 'serp'] (default 'local'); affects error messaging only (schema is identical)
- retriever_topk: int (default 3)
- reward weights: structure_format_score=0.0, final_format_score=0.0, retrieval_score=0.0, score=1.0
- data_split: 'train'|'test' (default 'train')
- data_limit: Optional[int] for subsampling without changing content

## Implementation steps
1) Dataset loader: replicate nq_search.py behavior (prompt string + ground_truth mapping) exactly.
2) Define tools: `search(query, topk=3)` that calls retriever_url and returns <information>…</information> with the exact Doc i(Title: …) formatting.
3) ToolEnv: set system_prompt enforcing <think> and <answer>; register the tool(s); set max_turns.
4) Reward: implement qa_em_format faithfully; build the sequence string with assistant anchor and message contents (including tool returns) for scoring.
5) Args and docs: expose retriever_url/topk/weights and document two retrieval modes.

## Local vs. web: summary
- Fully local makes sense and is supported by the official implementation via FAISS/BM25 over a Wikipedia JSONL corpus with a FastAPI /retrieve endpoint.
- True web search is also supported (SerpAPI/Serper) and can be integrated by pointing retriever_url to an online proxy returning the same schema. Determinism and reproducibility are weaker vs. local.