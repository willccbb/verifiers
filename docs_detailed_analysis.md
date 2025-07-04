# Detailed Documentation Analysis

## Section-by-Section Issues

### 1. `docs/source/index.md`

**Issues:**
- Installation shows `pip install verifiers` but project uses uv
- Import pattern `from verifiers.envs import SingleTurnEnv` doesn't match actual usage
- Basic example is overly complex and doesn't match actual API

**Current (Wrong):**
```python
from verifiers.envs import SingleTurnEnv
from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric
import openai

parser = XMLParser(fields=["think", "answer"])
rubric = Rubric(funcs=[correct_answer, parser.get_format_reward_func()], weights=[0.8, 0.2])
env = SingleTurnEnv(dataset=your_dataset, parser=parser, rubric=rubric, ...)
```

**Should be (Correct):**
```python
import verifiers as vf

dataset = vf.load_example_dataset("gsm8k", split="train")
parser = vf.ThinkParser(extract_fn=vf.extract_boxed_answer)
rubric = vf.Rubric(funcs=[...], weights=[...])
vf_env = vf.SingleTurnEnv(dataset=dataset, system_prompt=system_prompt, parser=parser, rubric=rubric)
```

### 2. `docs/source/overview.md`

**Issues:**
- Shows XMLParser as primary/recommended parser
- API examples don't match actual usage patterns
- Missing key concepts like `load_example_dataset`, `get_model_and_tokenizer`

**Current (Wrong):**
```python
parser = XMLParser(fields=["reasoning", "answer"])
```

**Should be (Correct):**
```python
# Most examples use ThinkParser or let environment choose automatically
parser = vf.ThinkParser(extract_fn=vf.extract_boxed_answer)
# or often just let the environment handle it automatically
```

### 3. `docs/source/examples.md`

**Major Issues:**
- All examples are fictional and overly complex
- Shows `MathEnv(SingleTurnEnv)` custom class pattern not used in actual code
- Missing actual environment types used in examples

**Current (Wrong):**
```python
class MathEnv(SingleTurnEnv):
    def __init__(self, dataset, **kwargs):
        parser = XMLParser(fields=["reasoning", "answer"])
        def correct_answer_reward_func(completion, answer, **kwargs):
            parsed_completion = parser.parse(completion)
            return float(check_math_equivalence(parsed_completion.answer, answer))
        rubric = Rubric(funcs=[correct_answer_reward_func], weights=[1.0], parser=parser)
        super().__init__(dataset=dataset, system_prompt=MATH_SYSTEM_PROMPT, parser=parser, rubric=rubric, **kwargs)
```

**Should be (Correct):**
```python
# From actual gsm8k.py example
dataset = vf.load_example_dataset("gsm8k", split="train")
parser = vf.ThinkParser(extract_fn=vf.extract_boxed_answer)
def correct_answer_reward_func(completion, answer, **kwargs):
    response = parser.parse_answer(completion) or ''
    return 1.0 if response == answer else 0.0
rubric = vf.Rubric(funcs=[correct_answer_reward_func, parser.get_format_reward_func()], weights=[1.0, 0.2])
vf_env = vf.SingleTurnEnv(dataset=dataset, system_prompt=system_prompt, parser=parser, rubric=rubric)
```

### 4. `docs/source/environments.md`

**Issues:**
- Missing actually-used environments: `ToolEnv`, `TextArenaEnv`, `ReasoningGymEnv`, `SmolaToolEnv`, `DoubleCheckEnv`
- Over-emphasizes `SingleTurnEnv` and `MultiTurnEnv` which are less commonly used
- No mention of automatic parser/rubric selection

**Missing Environments:**
- `ToolEnv` - Used in math_python.py
- `TextArenaEnv` - Used in wordle.py  
- `ReasoningGymEnv` - Used in arc_1d.py
- `SmolaToolEnv` - Used in smola_math_tools.py
- `DoubleCheckEnv` - Used in doublecheck.py

**Example of missing documentation:**
```python
# ToolEnv - completely missing from docs but used in examples
vf_env = vf.ToolEnv(
    dataset=dataset,
    system_prompt=TOOL_PROMPT,
    tools=[python],
    max_steps=3
)
```

### 5. `docs/source/training.md`

**Issues:**
- Shows complex manual training setup
- Missing `vf.grpo_defaults()` which is used in all examples
- No mention of vLLM server integration
- Missing multi-GPU setup patterns

**Current (Wrong):**
```python
# Complex setup not used in practice
training_args = TrainingArguments(...)
trainer = GRPOTrainer(model=model, args=training_args, ...)
```

**Should be (Correct):**
```python
# From actual examples
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
args = vf.grpo_defaults(run_name="my-experiment")
trainer = vf.GRPOTrainer(model=model, processing_class=tokenizer, env=vf_env, args=args)
```

### 6. `docs/source/parsers.md`

**Issues:**
- Over-emphasizes XMLParser when most examples use ThinkParser
- Missing SmolaParser which is used in smola examples
- No mention of automatic parser selection by environments

**Current emphasis (Wrong):**
```python
# XMLParser is presented as the main parser
parser = XMLParser(fields=["think", "answer"])
```

**Should emphasize (Correct):**
```python
# ThinkParser is more commonly used
parser = vf.ThinkParser(extract_fn=vf.extract_boxed_answer)
# or let environment choose automatically
```

### 7. `docs/source/rubrics.md`

**Issues:**
- Shows complex multi-criteria rubric setup
- Missing that many environments create rubrics automatically
- No mention of built-in rubric types like `MathRubric`, `ToolRubric`

**Current (Wrong):**
```python
# Complex manual rubric setup
rubric = Rubric(
    funcs=[correct_answer, reasoning_reward, format_compliance],
    weights=[0.7, 0.2, 0.1],
    parser=parser
)
```

**Should show (Correct):**
```python
# Many environments create rubrics automatically
# Or show built-in rubric types
rubric = vf.MathRubric(parser=parser)
# or
rubric = vf.ToolRubric(tools=tools, parser=parser)
```

### 8. `docs/source/tools.md`

**Issues:**
- Shows complex tool integration
- Missing SmolaAgents integration used in actual examples
- No mention of built-in tools like `python`, `calculator`

**Current (Wrong):**
```python
# Complex tool setup not used in practice
def custom_tool(arg1, arg2):
    # complex implementation
    pass
```

**Should show (Correct):**
```python
# From actual examples
from verifiers.tools import python
# or
from smolagents.default_tools import PythonInterpreterTool
from verifiers.tools.smolagents import CalculatorTool
```

## Priority Issues to Fix

### Critical (Blocks Users)
1. **Import errors** - All import statements in docs are wrong
2. **Installation instructions** - pip vs uv mismatch
3. **Basic API patterns** - Constructor signatures don't match

### High Priority (Misleading)
1. **Missing environment types** - Users can't find docs for what they see in examples
2. **Parser emphasis** - XMLParser vs ThinkParser mismatch
3. **Training setup** - Complex vs simple API mismatch

### Medium Priority (Confusing)
1. **Overly complex examples** - Users implement harder solutions
2. **Missing key utilities** - `load_example_dataset`, `get_model_and_tokenizer`
3. **Infrastructure gaps** - vLLM, multi-GPU setup

## Specific Actions Needed

### 1. Replace All Import Statements
- Find: `from verifiers.envs import`
- Replace: `import verifiers as vf`

### 2. Fix Installation Section
- Remove: `pip install verifiers`
- Add: `uv add verifiers`

### 3. Replace Examples with Working Code
- Copy actual examples from `verifiers/examples/`
- Test that they work as-is
- Add minimal explanations

### 4. Update API Signatures
- Use actual constructor signatures from code
- Show default parameters
- Remove verbose parameter passing

### 5. Add Missing Environment Documentation
- Document each environment type in `verifiers/examples/`
- Show actual usage patterns
- Include infrastructure setup (vLLM, multi-GPU)

This analysis shows the documentation-reality gap is extensive and requires systematic fixing rather than incremental updates.