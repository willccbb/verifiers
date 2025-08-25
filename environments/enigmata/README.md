# Enigmata

## Overview

- **Environment ID**: `enigmata`
- **Short description**: Synthetic, verifiable puzzle tasks with rule-based scoring across 36 tasks in 7 categories
- **Tags**: enigmata, single-turn, reasoning, puzzles, verifiable, generator-verifier

This environment exposes the Enigmata suite as a self-contained evaluation and data-generation environment. Problems are programmatically generated and scored with task-specific, rule-based verifiers. It is designed for training and evaluating reasoning models without external LLM judges.

Notes:

- Python ≥ 3.11; dependencies are declared in `pyproject.toml`.
- If the embedded `Enigmata` submodule is missing, the environment will automatically clone `BytedTsinghua-SIA/Enigmata` on first use.

## Datasets

- **Primary dataset(s)**: Enigmata-Data (synthetic) and Enigmata-Eval (benchmark)
- **Source links**:
  - Enigmata: [Github Repo](https://github.com/BytedTsinghua-SIA/Enigmata)
  - Enigmata-Eval: [HuggingFace Dataset](https://huggingface.co/datasets/BytedTsinghua-SIA/Enigmata-Eval)
- **Split sizes (Eval)**: 4,758 puzzle instances across Easy/Medium/Hard

## Task

- **Type**: single-turn
- **Parser**: Identity parser (returns the raw completion)
- **Rubric overview**: Single numeric score from a task-specific verifier (`verify`) with unit weight

## Quickstart

Run an evaluation with defaults (no API keys required):

```bash
uv run vf-eval enigmata
```

Evaluate with a fixed number of examples and specific tasks:

```bash
uv run vf-eval enigmata \
  -a '{"num_train_examples": 200, "num_eval_examples": 200, "tasks": ["sudoku", "maze"]}'
```

Use the predefined benchmark split (downloads Enigmata-Eval from HuggingFace) and evaluate only `sudoku`:

```bash
uv run vf-eval enigmata \
  -a '{"use_predefined_eval_dataset": true, "tasks": "sudoku"}'
```

Notes:

- Use `-a` / `--env-args` to pass environment configuration as JSON.
- When `use_predefined_eval_dataset` is `false`, both train and eval sets are generated on the fly.
- You can also generate and evaluate offline using the scripts in `Enigmata/` (see that README for details).

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_train_examples` | int | `-1` | Number of generated training examples per run (-1 means generator-chosen) |
| `num_eval_examples` | int | `-1` | Number of generated evaluation examples per run (-1 means generator-chosen) |
| `use_predefined_eval_dataset` | bool | `false` | If `true`, loads `BytedTsinghua-SIA/Enigmata-Eval` from HF for eval |
| `tasks` | str or list | `"all"` | Filter to a task or list of tasks (e.g., `"sudoku"`, `["sudoku","maze"]`) |
| `system_prompt` | str | `""` | Optional system prompt propagated to the environment |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Verifier score per example (typically 0 or 1; aggregated as mean) |

### Example Structure

Normalized examples produced by this environment follow this schema:

```txt
question: str
answer: str
task_name: str
difficulty: str
split: str
language: str
meta_json: Optional[str]  # JSON-encoded metadata including the fields above
```

### Verifier Integration

Per-example scoring dynamically imports `verifiable_tasks.tasks.<task_name>.verifier` and calls `verify(solution: str, answer: str, meta: dict) -> float|int`. If a verifier cannot be resolved, the reward defaults to `0.0` to fail closed.

## Evaluation Reports
<!-- Do not edit below this line. Content is auto-generated. -->
<!-- vf:begin:reports -->
No reports found. Run `uv run vf-eval enigmata` to generate one.
<!-- vf:end:reports -->