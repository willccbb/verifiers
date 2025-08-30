# futurex-past

### Overview
- **Environment ID**: `futurex-past`
- **Short description**: Historical question–answer dataset from the FutureX benchmark (past events). This README documents the dataset to support building a future environment for QA/retrieval and time-aware reasoning. The runnable environment code is not added yet.
- **Tags**: qa, retrieval, time-aware, single-turn, dataset

### Datasets
- **Primary dataset(s)**: FutureX-Past (local dump of past FutureX questions)
- **Source links**: FutureX live benchmark (https://futurex-ai.github.io/); local path `../futurex-ai/Futurex-Past/`
- **Split sizes**: `train` ≈ 851 examples (one Parquet shard)

### Schema
Each row corresponds to a single question whose outcome is already known. The current Parquet schema (from the local dump) includes:

- `question_id` (string): Unique identifier for the question.
- `question` (string): The prediction question posed to the agent.
- `answer` (string): Ground-truth answer recorded after the event occurred.
- `options` (list[string]): Options for multiple-choice questions (Levels 1–2); may be empty/null for open-ended.
- `end-time` (string): Timestamp string marking the relevant time context for the question.
- `prompt` (string): Full prompt provided to the LLM agent for the task.
- `level` (int): Difficulty level 1–4 (1=Basic, 2=Wide Search, 3=Deep Search, 4=Super Agent).

Example (illustrative):

```json
{
  "question_id": "620165c0-1c39-442a-9ac9-93e179e8c33e",
  "question": "北京时间2024年8月1日晚上8点，美联储的联邦基金利率目标范围是多少？",
  "answer": "5.25%",
  "options": null,
  "end-time": "2024-08-01T20:00:00+08:00",
  "prompt": "... full prompt provided to the agent ...",
  "level": 3
}
```

### Usage Notes
- This dataset contains historical questions whose outcomes are known; it should not be used to evaluate live future prediction. For live evaluation, see the weekly challenge: https://futurex-ai.github.io/.
- Suitable for:
  - Static QA benchmarking with known answers.
  - RL training for retrieval and time-aware reasoning (e.g., constraining search/browse tools to the question's time window).
- Local data (as provided): `../futurex-ai/Futurex-Past/data/train-00000-of-00001.parquet`.

### Next Steps (Environment)
- Add an environment loader (`environments/futurex_past/futurex_past.py`) and `pyproject.toml`.
- Implement a parser (e.g., `ThinkParser` or XML-style) and a rubric for answer correctness and format.
- Expose arguments for subsetting by `level`, filtering MCQ vs. open-ended, and limiting dataset size.

### Citation
If you use this dataset, please cite FutureX:

```
@misc{zeng2025futurexadvancedlivebenchmark,
      title={FutureX: An Advanced Live Benchmark for LLM Agents in Future Prediction}, 
      author={Zhiyuan Zeng and Jiashuo Liu and Siyuan Chen and Tianci He and Yali Liao and Jinpeng Wang and Zaiyuan Wang and Yang Yang and Lingyue Yin and Mingren Yin and Zhenwei Zhu and Tianle Cai and Zehui Chen and Jiecao Chen and Yantao Du and Xiang Gao and Jiacheng Guo and Liang Hu and Jianpeng Jiao and Xiangsheng Li and Jingkai Liu and Shuang Ni and Zhoufutu Wen and Ge Zhang and Kaiyuan Zhang and Xin Zhou and Jose Blanchet and Xipeng Qiu and Mengdi Wang and Wenhao Huang},
      year={2025},
      eprint={2508.11987},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2508.11987}, 
}
```

## Quickstart
Run an evaluation (assumes local data path):

```bash
uv run vf-eval futurex-past \
  -a '{"data_dir": "../futurex-ai/Futurex-Past/data", "use_think": true, "num_eval_examples": 50}'
```

Configure sampling and model:

```bash
uv run vf-eval futurex-past \
  -m gpt-4.1-mini \
  -n 100 -r 1 -t 1024 -T 0.7 \
  -a '{"data_dir": "../futurex-ai/Futurex-Past/data", "filter_levels": [1,2], "mcq_only": true}'
```

### Run Eval (OpenAI)
- Set your API key and install the env if needed:

```bash
export OPENAI_API_KEY=sk-...
uv run vf-install futurex-past -p ./environments
```

- Run a small eval (n=5, r=1) against OpenAI `gpt-4.1-mini`:

```bash
uv run vf-eval futurex-past \
  -m gpt-4.1-mini -n 5 -r 1 -t 512 -T 0.7 \
  -a '{"data_dir":"../futurex-ai/Futurex-Past/data","num_eval_examples":5}'
```

- Optional (explicit provider flags):

```bash
uv run vf-eval futurex-past \
  -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY \
  -n 5 -r 1 -t 512 -T 0.7 \
  -a '{"data_dir":"../futurex-ai/Futurex-Past/data","num_eval_examples":5}'
```

Notes:
- The HTML report is written under `./environments/futurex_past/reports/`.
- To surface a new report below, add an `<a href="reports/<filename>.html">` link inside the reports block.

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `data_dir` | str | required | Path to Parquet shard(s), e.g., `../futurex-ai/Futurex-Past/data` |
| `use_think` | bool | `true` | Use `<think>` + `<answer>` XML format; if false, `<answer>` only |
| `num_train_examples` | int | `-1` | Limit train set size (`-1` for all) |
| `num_eval_examples` | int | `-1` | Limit eval set size; if `-1`, mirrors train |
| `filter_levels` | list[int] | `null` | Keep only examples whose `level` is in this list |
| `mcq_only` | bool/null | `null` | `true` → only with non-empty `options`; `false` → only open-ended; `null` → all |
| `lowercase_compare` | bool | `false` | Normalize answers to lowercase for equality check |
| `strip_whitespace_compare` | bool | `true` | Strip surrounding whitespace before comparison |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | 1.0 if parsed `<answer>` exactly equals ground truth (with optional normalization) |
| `format_reward` | Adherence to the expected XML format |

### MCQ Handling
- If `info.options` is present for an example, the reward accepts either:
  - The option text (exact match after optional normalization), or
  - A label identifying the option by index: `A`/`B`/`C`/… or `1`/`2`/`3`/… (1-indexed)
- Label ↔ text matching works both ways:
  - Predicted `A` matches the text of the first option if the gold `answer` is that text.
  - Predicted text of an option matches a gold `answer` that is a label like `B`.

## Evaluation Reports

Notes:
- Reports are written under `./environments/futurex_past/reports/` and auto-embedded below.

<!-- Do not edit below this line. Content is auto-generated. -->
<!-- vf:begin:reports -->
<p><a href="reports/vf-futurex-past--v0.1.0--model=dry-run--n=0--r=0--args=noargs.html" target="_blank">Open dry-run report (placeholder)</a></p>
<!-- To link a real report, replace the placeholder above with the generated filename. Example:
<p><a href="reports/vf-futurex-past--v0.1.0--model=gpt-4.1-mini--n=5--r=1--args=noargs.html" target="_blank">Open full report</a></p>
-->
<!-- vf:end:reports -->
