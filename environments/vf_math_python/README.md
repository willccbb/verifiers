# vf-math-python

> Replace the placeholders below, then remove this callout. Keep the Evaluation Reports section at the bottom intact so reports can auto-render.

### Overview
- **Environment ID**: `vf-math-python`
- **Short description**: <one-sentence description>
- **Tags**: <comma-separated tags>

### Datasets
- **Primary dataset(s)**: <name(s) and brief description>
- **Source links**: <links>
- **Split sizes**: <train/eval counts>

### Task
- **Type**: <single-turn | multi-turn | tool use>
- **Parser**: <e.g., ThinkParser, XMLParser, custom>
- **Rubric overview**: <briefly list reward functions and key metrics>

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval vf-math-python
```

Configure model and sampling:

```bash
uv run vf-eval vf-math-python   -m gpt-4.1-mini   -n 20 -r 3 -t 1024 -T 0.7   -a '{"key": "value"}'  # env-specific args as JSON
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- Reports are written under `./environments/vf_math_python/reports/` and auto-embedded below.

### Environment Arguments
Document any supported environment arguments and their meaning. Example:

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `foo` | str | `"bar"` | What this controls |
| `max_examples` | int | `-1` | Limit on dataset size (use -1 for all) |

### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of criteria) |
| `accuracy` | Exact match on target answer |

## Evaluation Reports

<!-- Do not edit below this line. Content is auto-generated. -->
<!-- vf:begin:reports -->
<details><summary>Reports</summary>
<details><summary>vf-math-python--v0.1.0--model=gpt-4.1-mini--n=5--r=3--args=noargs</summary>
<p><a href="reports/vf-math-python--v0.1.0--model=gpt-4.1-mini--n=5--r=3--args=noargs.html" target="_blank">Open full report</a></p>
<h1>Verifiers Eval Report</h1>
<div class="meta">
<div><b>Environment</b>: vf-math-python (v0.1.0)</div>
<div><b>Model</b>: <span class="code">gpt-4.1-mini</span></div>
<div><b>Provider</b>: https://api.openai.com/v1</div>
<div><b>Samples</b>: n=5, r=3</div>
<div><b>Sampling</b>: max_tokens=1024, temperature=0.7</div>
</div>

<h2>Reward</h2>
<table>
<tr><th>mean</th><th>std</th><th>n</th><th>p5</th><th>p25</th><th>p50</th><th>p75</th><th>p95</th></tr>
<tr>
<td>0.8533</td>
<td>0.3757</td>
<td>15</td>
<td>-0.06</td>
<td>1.0</td>
<td>1.0</td>
<td>1.0</td>
<td>1.0</td>
</tr>
</table>


<h2>Metrics</h2>
<table>
<tr>
<th>metric</th><th>mean</th><th>std</th><th>n</th><th>p5</th><th>p25</th><th>p50</th><th>p75</th><th>p95</th>
</tr>

<tr>
<td>correct_answer_reward_func</td>
<td>0.8667</td>
<td>0.3399</td>
<td>15</td>
<td>0.0</td>
<td>1.0</td>
<td>1.0</td>
<td>1.0</td>
<td>1.0</td>
</tr>

<tr>
<td>num_turns</td>
<td>1.7333</td>
<td>0.5735</td>
<td>15</td>
<td>1.0</td>
<td>1.0</td>
<td>2.0</td>
<td>2.0</td>
<td>2.3</td>
</tr>

<tr>
<td>num_tool_calls</td>
<td>0.8667</td>
<td>0.8055</td>
<td>15</td>
<td>0.0</td>
<td>0.0</td>
<td>1.0</td>
<td>1.0</td>
<td>2.3</td>
</tr>

<tr>
<td>num_errors</td>
<td>0.1333</td>
<td>0.4989</td>
<td>15</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.6</td>
</tr>

</table>


<h2>Examples <span class="muted">(showing up to 15 of 15)</span></h2>
<div class="examples">
<table>
<tr><th>#</th><th>reward</th><th>correct_answer_reward_func</th><th>completion</th></tr>

<tr>
<td>0</td>
<td>1.0</td>
<td>1.0</td>
<td><pre>The values of \(a\) and \(b\) that make the piecewise function continuous are \(a = -3\) and \(b = 3\).

Thus, the sum \(a + b = -3 + 3 = 0\).

The answer is \(\boxed{0}\).</pre></td>
</tr>

<tr>
<td>1</td>
<td>1.0</td>
<td>1.0</td>
<td><pre>Let the original number of rows be \( r \) and the number of members in each row be \( m \). Then the total number of band members \( N \) is given by:
\[
N = mr + 2
\]
since there are 2 members left over when arranged in \( r \) rows and \( m \) members per row.

When the number of members in each…</pre></td>
</tr>

<tr>
<td>2</td>
<td>1.0</td>
<td>1.0</td>
<td><pre>The given polynomial is:

\[4 + 5x^3 + 100 + 2\pi x^4 + \sqrt{10} x^4 + 9\]

Let&#39;s identify the degrees of each term:

- \(4\) is a constant term, degree 0.
- \(5x^3\) is degree 3.
- \(100\) is a constant term, degree 0.
- \(2\pi x^4\) is degree 4.
- \(\sqrt{10} x^4\) is degree 4.
- \(9\) is a cons…</pre></td>
</tr>

<tr>
<td>3</td>
<td>1.0</td>
<td>1.0</td>
<td><pre>The value of \(\left\lceil3\left(6-\frac12\right)\right\rceil\) is \(17\).</pre></td>
</tr>

<tr>
<td>4</td>
<td>1.0</td>
<td>1.0</td>
<td><pre>Sam did not work for 6 days.</pre></td>
</tr>

<tr>
<td>5</td>
<td>1.0</td>
<td>1.0</td>
<td><pre>The piecewise function is continuous if the values at the boundaries match. Solving for continuity at \(x=2\) and \(x=-2\), we find \(a = -3\) and \(b = 3\).

Therefore, \(a + b = -3 + 3 = 0\).</pre></td>
</tr>

<tr>
<td>6</td>
<td>1.0</td>
<td>1.0</td>
<td><pre>Let&#39;s denote the number of band members as \(N\), the original number of rows as \(r\), and the original number of members per row as \(m\).

From the problem:
1. The total number of band members is \(N = mr + 2\) because 2 members are left over when arranged in the original formation.
2. When the …</pre></td>
</tr>

<tr>
<td>7</td>
<td>1.0</td>
<td>1.0</td>
<td><pre>The polynomial given is:

\[
4 + 5x^3 + 100 + 2\pi x^4 + \sqrt{10} x^4 + 9
\]

Let&#39;s identify the degree of each term:

- \(4\) is a constant term (degree 0).
- \(5x^3\) has degree 3.
- \(100\) is a constant term (degree 0).
- \(2\pi x^4\) has degree 4.
- \(\sqrt{10} x^4\) has degree 4.
- \(9\) is …</pre></td>
</tr>

<tr>
<td>8</td>
<td>1.0</td>
<td>1.0</td>
<td><pre>The value of \(\left\lceil3\left(6-\frac12\right)\right\rceil\) is \(\boxed{17}\).</pre></td>
</tr>

<tr>
<td>9</td>
<td>1.0</td>
<td>1.0</td>
<td><pre>Sam did not work for 6 days.</pre></td>
</tr>

<tr>
<td>10</td>
<td>-0.2</td>
<td>0.0</td>
<td><pre></pre></td>
</tr>

<tr>
<td>11</td>
<td>0.0</td>
<td>0.0</td>
<td><pre>The largest number of members the band could have is 98. This corresponds to an original formation with 6 members in each of 16 rows.</pre></td>
</tr>

<tr>
<td>12</td>
<td>1.0</td>
<td>1.0</td>
<td><pre>The polynomial is:

\[4 + 5x^3 + 100 + 2\pi x^4 + \sqrt{10} x^4 + 9\]

First, let&#39;s combine the constant terms:

\[4 + 100 + 9 = 113\]

So the polynomial becomes:

\[113 + 5x^3 + (2\pi + \sqrt{10}) x^4\]

The term with the highest degree is \( (2\pi + \sqrt{10}) x^4 \), which has degree 4.

Therefo…</pre></td>
</tr>

<tr>
<td>13</td>
<td>1.0</td>
<td>1.0</td>
<td><pre>The value of \(\left\lceil 3 \left(6 - \frac{1}{2}\right) \right\rceil\) is \(17\).</pre></td>
</tr>

<tr>
<td>14</td>
<td>1.0</td>
<td>1.0</td>
<td><pre>Sam did not work for 6 days.</pre></td>
</tr>

</table>
</div>
</details>
</details>
<!-- vf:end:reports -->
