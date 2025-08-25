# misguided-attention

## Overview

- **Environment ID**: `misguided-attention`
- **Short description**: Complex reasoning tasks with multi-criteria weighted evaluation using LLM judges
- **Tags**: misguided-attention, single-turn, reasoning, logic-puzzles, multi-criteria, llm-judge

This is a collection of prompts to challenge the reasoning abilities of large language models in presence of misguiding information. They are slight variations of commonly known thought experiments, riddles or paradoxes ("trick questions").

The expected behavior would be that the LLMs solve the problems, as they are stated, by logical deduction. However, many LLMs will mistakenly recognize the unmodified problem due to frequent occurrence in their training data. In consequence, they will respond with a solution to the unmodified problem instead of going through the details step-by-step to find a solution for the modified problem. In some cases it's also possible to observe intertwined strings of reasoning where conflicting thoughts are alternating in the same text.

Parallels to this can be drawn to human behavior, where recognition of familiar patterns leads to the execution of previously learned routines, even if they are not applicable to the current situation. This is known as the Einstellungseffekt. However, we would expect that a computerized reasoning system would not be subject to such a fallacy.

## Datasets

- **Primary dataset(s)**: MisguidedAttention v4 and MisguidedAttention v4 Long
- **Source links**:
  - MisguidedAttention v4: [GitHub Repository](https://github.com/cpldcpu/MisguidedAttention)
  - Dataset files: [v4.scr](https://github.com/cpldcpu/MisguidedAttention/blob/main/eval/harness/misguided_attention_v4.scr), [v4_long.scr](https://github.com/cpldcpu/MisguidedAttention/blob/main/eval/harness/misguided_attention_v4_long.scr)
- **Split sizes**:
  - Evaluation:
    - MisguidedAttention v4: 13 examples
    - MisguidedAttention v4 Long: 52 examples

## Task

- **Type**: single-turn
- **Parser**: Identity parser (returns stripped response)
- **Rubric overview**: Multi-criteria weighted evaluation using LLM judge with configurable providers (DeepSeek, OpenAI, Gemini)

## Quickstart

Run an evaluation with default settings:

```bash
export OPENAI_API_KEY="your-api-key"
uv run vf-eval misguided-attention
```

Configure model and evaluation parameters:

Use with OpenAI judge:

```bash
export OPENAI_API_KEY="your-api-key"
uv run vf-eval misguided-attention \
  -a '{"judge_base_url": "https://api.openai.com/v1", "judge_api_key_var": "OPENAI_API_KEY", "judge_model": "gpt-4o-mini"}'
```

Notes:

- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- The environment automatically downloads and descrambles datasets if not present locally.
- Default configuration uses OpenAI as the judge provider.

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_name` | str | `"long"` | Dataset version ("normal" for 13 examples, "long" for 52 examples) |
| `judge_model` | str | `"gpt-4o-mini"` | Judge model name |
| `judge_base_url` | str | `"https://api.openai.com/v1"` | Judge API base URL |
| `judge_api_key_var` | str | `"OPENAI_API_KEY"` | Environment variable name for judge API key |
| `judge_temperature` | float | `0.0` | Judge sampling temperature |
| `judge_max_tokens` | int | `1000` | Judge maximum number of tokens |
| `max_examples` | int | `-1` | Maximum number of examples to load (-1 for all) |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Final weighted score across all criteria (0.0 to 1.0) |

### Example Structure

Each example contains:

- **prompt**: Complex reasoning task or ethical dilemma
- **criteria**: List of evaluation criteria (1-3 per example)
- **weights**: Numerical weights for each criterion
- **metadata**: prompt_id, category, title

Example:

```txt
prompt_id: trolley_problem_easy
criteria: [
  "States that pulling the lever is the morally correct choice",
  "Mentions the utilitarian principle"
]
weights: [0.5, 0.5]
```

### Judge Integration

The environment supports multiple LLM judge providers:

**OpenAI:**

```json
{
  "judge_base_url": "https://api.openai.com/v1", 
  "judge_api_key_var": "OPENAI_API_KEY",
  "judge_model": "gpt-4o-mini"
}
```

## Evaluation Reports
<!-- Do not edit below this line. Content is auto-generated. -->
<!-- vf:begin:reports -->
<p>No reports found. Run <code>uv run vf-eval misguided-attention</code> to generate one.</p>
<!-- vf:end:reports -->