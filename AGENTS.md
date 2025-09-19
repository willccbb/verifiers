## AGENTS.md

This guide explains how to contribute to the core repository, and how to use it for developing new environments and tools. It reflects the existing code, documentation patterns, and best practices for the `verifiers` repo.

## Core repository development

Relevant for:
- Immediate PRs to `verifiers` (small features, bug fixes).
- Independent forks of `verifiers` which you may want to upstream later (larger projects).

### Setup

We use `uv`, not `pip`, for developing `verifiers`. 

When developing in a project which *uses* `verifiers`:
```bash
# install uv (first time only)
curl -LsSf https://astral.sh/uv/install.sh | sh

# for a fresh project -- 3.11 + 3.12 supported
uv init && uv venv --python 3.12 
source .venv/bin/activate

# install verifiers (choose extras as needed)
uv add verifiers            # core
uv add 'verifiers[dev]'     # + notebooks + test helpers
uv add 'verifiers[all]'     # + trainer extras
```

When developing or contributing directly to the `verifiers` core repository:

```bash
git clone https://github.com/willccbb/verifiers.git
cd verifiers

# for CPU-only dev:
uv sync --extra dev

# for GPU-based trainer dev:
uv sync --all-extras && uv pip install flash-attn --no-build-isolation

# install pre-commit hooks
uv run pre-commit install

# run style + lint checks locally
uv run ruff check --fix .
uv run pre-commit run --all-files
```

### Code style
- **Style**: We strictly enforce ruff style (imports, formatting, linting). Keep diffs small and focused. All PRs must pass `ruff check --fix .`.
- **Typing**: Prefer explicit typing throughout the core codebase.
  - **OK**: `cast(...)`, `assert ...` for type narrowing.
  - **NOT OK**: `# type: ignore` (avoid unless there is a very strong justification).

### Error handling philosophy
- **Fail fast, fail loud.** Do not add defensive programming or silent fallbacks.
- Avoid excessive conditionals; prefer one clear, direct code path. Any conditional logic (if/else, try/except) **must** be justified by necessity.

### Dependencies

Avoid adding new dependencies to `verifiers` via PRs; if they are needed for core-but-optional features (e.g. TextArena, trainer dependencies), these should live in an optional dependency group.

The main exceptions are for cases where introducing a very lightweight dependency allows majorly streamlining or importing logic which serves a clear utility purpose.

### PR hygiene
- Keep PRs self‑contained and concise. Smaller diffs are better.
- Prefer single, well‑scoped changes per PR. Remove dead code instead of adding conditionals around it.

### Tests
- Tests live in `tests/` and follow `pytest` discovery (`test_*.py`). See `[tool.pytest.ini_options]` in `pyproject.toml` for markers and options.
- Add unit tests for new components and behaviors. Favor simple, deterministic tests over broad integration coverage unless necessary.

### Docs
- Keep docs short and actionable. Update relevant pages when behavior changes. Prefer adding or refining a single section over duplicating content elsewhere.
- For the authoritative testing/setup walkthrough, see `docs/source/development.md`.

## Developing environments

The environment APIs are defined under `verifiers/envs/`.

### Lifecycle and what to implement
- Most multi-turn environments should extend `MultiTurnEnv` and implement exactly:
  - `is_completed(messages, state, **kwargs) -> bool`
  - `env_response(messages, state, **kwargs) -> tuple[Messages, State]`
- Use `setup_state(state, **kwargs) -> State` for complex or dynamic per‑rollout initialization, e.g. provisioning and tracking resources such as sandboxes.

- You should almost never override `rollout`. If you must, you **must** call `super().rollout`. The loop provided by `MultiTurnEnv` wires together prompting, tool calls, and environment responses.

### Environment patterns and when to use them
Tools are one pattern; many environments do not use tools.

- **Non‑tool environments** (user simulations, interactive games, multi‑agent tasks): extend `MultiTurnEnv` directly and implement `is_completed` and `env_response` (and optionally `setup_state`).
- **Stateless tools** (pure functions that can call external APIs directly): use `ToolEnv`.
  - Provide `tools: list[Callable]`. They will be exposed as OpenAI‑style tools automatically.
  - Completion ends when the model produces an assistant turn with no tool calls (handled by `ToolEnv.is_completed`).
- **Stateful tools** (tools that depend on per-rollout state, e.g., ephemeral sandboxes): use `StatefulToolEnv`.
  - Implement `update_tool_args(tool_args, messages, state, **kwargs) -> dict` to inject or transform arguments using state.
- **MCP-backed tools**: see `environments/mcp_env/` for integrating MCP tool servers.

### Environment quickstart
- Bootstrap from the template: `vf-init vf-new-environment` (defaults to creating a package under `./environments/`).
- Install your local module while iterating: `vf-install vf-new-environment` (editable install via `uv`).
- Smoke-test rollouts: `vf-eval vf-new-environment -n 5 -m gpt-4.1-mini`.
- Every packaged environment must expose `load_environment(...)` and maintain its own `pyproject.toml`.

### Resource management
- **Heavy and reusable resources** (datasets, clients, long‑lived caches) should be provisioned in `__init__`.
- **Per‑rollout resources** (e.g., ephemeral sandboxes, temp dirs, per‑attempt credentials) should be created in `setup_state`. Tear‑down should be centralized and predictable.

### Dependencies and “source files as dependencies”
- Each environment lives in `environments/<name>/` with its own `pyproject.toml`. Put environment‑specific dependencies there.
- Avoid vendoring external source files. Prefer clean library dependencies or well‑scoped local modules.
- If an environment relies on external services (API keys, endpoints), document the required environment variables and fail fast if they are not present. Do not add fallbacks.

### Error handling and conditionals (environments)
- No defensive programming and no silent fallbacks.
- Prefer a single code path. Every `if` should exist for a clear, documented reason.
- Example of what NOT to handle: if a required API key is missing and a tool call fails, it should fail immediately and loudly.

### Typing guidance (environments)
- Prefer explicit types in public methods and shared utilities.
- **OK**: `cast`, `assert` for refinement.
- **SOMETIMES OK**: untyped arguments for narrow, performance‑sensitive cases (e.g., simple reward functions), when types add more friction than value.
- **NOT OK**: `# type: ignore` except with strong, documented justification.

### Tools
- Tools provided to `ToolEnv`/`StatefulToolEnv` should be small, composable callables.
- Stateless tools should be pure (idempotent with the same inputs). Keep side effects to a minimum.
- Use exceptions to surface errors; let the environment surface them without retries or fallbacks.

### State structure
- The default rollout state includes keys like `prompt`, `completion`, `responses`, `turn`, `timing`, `task`, and `info`. Only rely on keys your environment sets or updates explicitly.
- Examples covering the most common patterns live in `environments/README.md`—skim it to find the closest reference implementation before building from scratch.

## Checklists

### New environment checklist
- Choose `ToolEnv` (stateless) or `StatefulToolEnv` (stateful).
- Put heavy resources in `__init__`; per‑rollout setup in `setup_state`.
- Implement `is_completed` and `env_response` only (and `update_tool_args` if stateful).
- Declare dependencies in `environments/<name>/pyproject.toml`.
- Add concise tests under `tests/` and minimal docs under `environments/<name>/README.md`.
- Keep conditionals minimal and avoid fallbacks.

### PR checklist
- Small, self‑contained diff
- Explicit typing, ruff‑clean
- Tests updated/added
- Docs updated when behavior changes
