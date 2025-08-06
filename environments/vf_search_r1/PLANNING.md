# Search-R1 Port Planning

## Goal
Port the Search-R1 environment into a verifiers Environment module (`vf-search-r1`) that supports vf-eval and training, using the minimal, idiomatic MultiTurnEnv/ToolEnv patterns.

## References
- THUDM slime example: /tmp/slime/examples/search-r1
- PeterGriffinJin/Search-R1: /tmp/Search-R1

## Scope
- One module under environments/vf_search_r1 with dependencies declared via uv-managed pyproject.
- No edits outside the module. No extra Python files unless requested.

## Task Structure to Port
- Dataset: QA-style prompts with gold answers; Search-R1 expects open-domain questions.
- Tools: Web search and page/paragraph retrieval; upstream uses Google/Serp and scraping.
- Rollout: Agent loops tool calls and emits final answer within max turns.
- Reward: EM/F1-style exact/normalized match vs gold; optional judge rubric for long-form.

## Proposed Verifiers Mapping
- Use vf.ToolEnv with custom tools wired to upstream behavior:
  - search(query: str) -> list[{title, url}]
  - open(url: str) -> str
  - read(url: str, span: Optional[str]) -> str (optional; may be folded into open)
- System prompt: align with Search-R1’s think + tool JSON blocks, enforce max_turns.
- Dataset: Start with a small HF subset with fields question/answer; normalize to `prompt`+`answer`.
- Rubric: primary EM via normalized comparison; optional JudgeRubric for long-form answers if needed.

## Minimal Viable Plan
1) Dataset
   - For initial viability, reuse an available HF QA dataset (e.g., `nq_open`/`trivia_qa`) and map to fields expected by verifiers.
   - If upstream repo has example/dev set, import small JSONL via datasets.load_dataset with local files.

2) Tools
   - Start with a simple HTTP search provider to avoid complex API keys: use Serper.dev or Tavily (if available), otherwise duckduckgo via requests. Keep provider pluggable.
   - Implement `search` returning top-k results with title/url; implement `open` fetching page HTML and extracting text via readability/boilerplate removal.
   - Keep requests bounded and deterministic (timeouts, headers). No parallelism initially.

3) Env
   - Instantiate vf.ToolEnv with system_prompt mirroring upstream format (think/tool/answer blocks), max_turns ~8-10.
   - No custom MultiTurnEnv unless JSON parsing demands it; rely on native tool-calling with function schemas.

4) Rubric
   - Implement normalized EM: lowercase, strip, remove punctuation/articles.
   - Consider partial credit metric (F1 token overlap) as auxiliary metric with weight 0.

5) Configurables
   - max_turns, top_k, http timeouts, user agent, search provider base_url+api_key env var names.

6) Out-of-Scope (for first PR)
   - Full retriever index building.
   - Browser automation, JS rendering, or complex anti-bot handling.

## Open Questions
- Preferred search provider within constraints; default to duckduckgo-lite unless instructed otherwise.
- Dataset choice for evaluation; confirm with maintainer.

## Next Steps
- Implement tool stubs and system prompt in vf_search_r1.py.
- Add minimal rubric and dataset mapping.
- `uv run ruff format && uv run pytest -q` if tests exist; `uv run vf-eval vf-search-r1 -n 5 -r 2` for smoke.