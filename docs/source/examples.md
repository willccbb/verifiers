# Examples

This section provides practical examples of how to use the verifiers framework, based on real-world usage patterns from the examples folder.

## Quick Start: GSM8K Math Problem Solving

A simple example using `SingleTurnEnv` for math problem solving:

```python
import verifiers as vf
from verifiers.utils.data_utils import load_example_dataset, extract_boxed_answer

# Load dataset
dataset = vf.load_example_dataset("gsm8k", split="train") 
eval_dataset = vf.load_example_dataset("gsm8k", split="test").select(range(100))

# Define system prompt
system_prompt = """
Think step-by-step inside <think>...</think> tags.

Then, give your final numerical answer inside \\boxed{{...}}.
"""

# Setup parser and rubric
parser = vf.ThinkParser(extract_fn=extract_boxed_answer)

def correct_answer_reward_func(completion, answer, **kwargs):
    response = parser.parse_answer(completion) or ''
    return 1.0 if response == answer else 0.0

rubric = vf.Rubric(funcs=[
    correct_answer_reward_func,
    parser.get_format_reward_func()
], weights=[1.0, 0.2])

# Create environment
vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    eval_dataset=eval_dataset,
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric,
)

# Evaluate the environment
results = vf_env.evaluate(
    client=openai_client,
    model="gpt-4",
    num_examples=10
)
print(results)
```

## Tool-Augmented Math with Python

Using `ToolEnv` for complex mathematical reasoning with Python execution:

```python
import verifiers as vf
from verifiers.tools import python
from verifiers.utils import load_example_dataset

TOOL_PROMPT = """
Think step-by-step inside <think>...</think> tags in each message, then either call a tool inside <tool>...</tool> tags, or give your final answer inside <answer>...</answer> tags.

You have access to the following tools to help solve problems:

{tool_descriptions}

Tools can be called by writing a JSON command inside <tool> tags with:
- "name": the name of the tool to use
- "args": the arguments for the tool

Example usage:
<tool>
{{"name": "python", "args": {{"code": "import sympy\nx = sympy.symbols('x')\nprint(sympy.solve(x**2 - 4, x))"}}}}
</tool>

After concluding your message with a tool call,
you will then see the tool's output inside <result> tags as a new message. \
You may call tools multiple times if needed. \
Tool state does not persist between calls. \
Always use tools to solve problems whenever possible, rather than using your own knowledge.

The <answer>...</answer> tags should contain only your final answer as a numeric expression.

Example:
<think>
Let's submit the answer.
</think>
<answer>
\\frac{{1}}{{2}}
</answer>
"""

dataset = load_example_dataset("math", split="train")

vf_env = vf.ToolEnv(
    dataset=dataset,
    system_prompt=TOOL_PROMPT,
    few_shot=[],
    tools=[python],
    max_steps=3
)

# Generate training data
results = vf_env.generate(
    client=openai_client,
    model="gpt-4",
    n_samples=100
)
```

## Interactive Wordle Game

Using `TextArenaEnv` for game-based training:

```python
import verifiers as vf
from verifiers.envs.textarena_env import TextArenaEnv

# Create game environment
vf_env = TextArenaEnv(
    game="Wordle-v0",
    num_train_examples=2000, 
    num_eval_examples=20,
)

# Evaluate model performance
results = vf_env.evaluate(
    client=openai_client,
    model="gpt-4",
    num_examples=10
)
```

## Multi-Turn Wiki Search

A complex example using custom `MultiTurnEnv` for interactive search:

```python
import verifiers as vf
from verifiers.rubrics.judge_rubric import JudgeRubric

# Custom environment for wiki search
class WikiSearchEnv(vf.MultiTurnEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_turns = 10
        
    def is_completed(self, messages, state, **kwargs):
        """End after max turns or when answer is found."""
        return len(state['responses']) >= self.max_turns or state.get('found_answer', False)
    
    def env_response(self, messages, state, **kwargs):
        """Process tool calls and provide results."""
        # Extract tool calls from last message
        last_message = messages[-1]['content']
        
        # Parse tool calls and execute them
        # This is a simplified version - see wiki_search.py for full implementation
        if '<tool>' in last_message:
            # Execute tool and return result
            tool_result = self.execute_tool_call(last_message)
            return {"role": "user", "content": f"<result>{tool_result}</result>"}, state
        
        return {"role": "user", "content": "Please use tools to find the answer."}, state

# Create environment with judge rubric
judge_rubric = JudgeRubric(
    judge_models=["gpt-4"],
    client=openai_client,
    template="""Evaluate this response for accuracy and completeness.

Question: {prompt}
Answer: {completion}
Expected: {answer}

Score from 0-10:""",
    parser=vf.XMLParser(["score", "feedback"])
)

vf_env = WikiSearchEnv(
    dataset=dataset,
    system_prompt=system_prompt,
    parser=parser,
    rubric=judge_rubric
)
```

## Custom Parser Example

Creating a custom parser for specific output formats:

```python
import verifiers as vf
import re

class CodeParser(vf.Parser):
    """Parse code blocks from markdown responses."""
    
    def parse(self, text: str) -> str:
        """Extract code from markdown code blocks."""
        code_match = re.search(r'```(?:\w+)?\n(.*?)\n```', text, re.DOTALL)
        return code_match.group(1) if code_match else text
    
    def get_format_reward_func(self):
        """Reward function for proper code block formatting."""
        def format_reward_func(completion, **kwargs):
            if isinstance(completion, str):
                return 1.0 if '```' in completion else 0.0
            return 0.0
        return format_reward_func

# Use custom parser
parser = CodeParser()

def code_correctness_reward(completion, answer, **kwargs):
    """Check if generated code produces correct output."""
    code = parser.parse(completion)
    if not code:
        return 0.0
    
    try:
        # Execute code and compare with expected output
        exec_result = exec_code_safely(code)
        return 1.0 if exec_result == answer else 0.0
    except:
        return 0.0

rubric = vf.Rubric(funcs=[
    code_correctness_reward,
    parser.get_format_reward_func()
], weights=[0.8, 0.2])

vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    parser=parser,
    rubric=rubric
)
```

## Custom Rubric Example

Creating a custom rubric for specific evaluation criteria:

```python
import verifiers as vf

class MathRubric(vf.Rubric):
    """Custom rubric for mathematical reasoning tasks."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def correctness_reward(self, completion, answer, **kwargs):
        """Check if the final answer is correct."""
        parsed = self.parser.parse(completion)
        return 1.0 if parsed.answer.strip() == answer.strip() else 0.0
    
    def reasoning_steps_reward(self, completion, **kwargs):
        """Reward for showing step-by-step reasoning."""
        parsed = self.parser.parse(completion)
        steps = parsed.reasoning.count('\n') + 1
        return min(steps / 5.0, 1.0)  # Up to 5 steps = full score
    
    def mathematical_accuracy_reward(self, completion, **kwargs):
        """Check for mathematical errors in reasoning."""
        parsed = self.parser.parse(completion)
        reasoning = parsed.reasoning.lower()
        
        # Check for common math errors
        errors = 0
        if '2+2=5' in reasoning:
            errors += 1
        if 'divide by zero' in reasoning:
            errors += 1
            
        return max(0.0, 1.0 - errors * 0.5)  # -0.5 per error
    
    def format_reward(self, completion, **kwargs):
        """Reward for proper formatting."""
        return self.parser.get_format_reward_func()(completion)

# Use custom rubric
math_rubric = MathRubric(
    funcs=[
        MathRubric.correctness_reward,
        MathRubric.reasoning_steps_reward,
        MathRubric.mathematical_accuracy_reward,
        MathRubric.format_reward
    ],
    weights=[0.5, 0.2, 0.2, 0.1],
    parser=vf.XMLParser(["reasoning", "answer"])
)

vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    parser=vf.XMLParser(["reasoning", "answer"]),
    rubric=math_rubric
)
```

## Environment Evaluation

Environments are powerful evaluation tools:

```python
# Evaluate model performance
results = vf_env.evaluate(
    client=openai_client,
    model="gpt-4",
    num_examples=100,
    rollouts_per_example=3
)

print(f"Average reward: {sum(results['rewards']) / len(results['rewards'])}")
print(f"Correct answers: {sum(1 for r in results['rewards'] if r > 0.8)}")

# Generate training data
results = vf_env.generate(
    client=openai_client,
    model="gpt-4",
    n_samples=1000
)

# Process for training
processed = vf_env.process_env_results(
    prompts=results['prompts'],
    completions=results['completions'],
    states=results['states'],
    rewards=results['rewards'],
    processing_class=tokenizer
)
```

## Training Integration

Using environments with training frameworks:

```python
# Setup model and training
model, tokenizer = vf.get_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")
training_args = vf.grpo_defaults(run_name="my-experiment")

# Configure training
training_args.per_device_train_batch_size = 8
training_args.num_generations = 16
training_args.max_steps = 500

# Create trainer
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)

# Train
trainer.train()
```

## Key Patterns

### 1. Start Simple
Begin with `SingleTurnEnv` and basic parsers:

```python
vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    parser=vf.ThinkParser(),
    rubric=vf.Rubric(funcs=[correct_answer_func])
)
```

### 2. Add Complexity Gradually
Add custom parsers and rubrics as needed:

```python
# Custom parser for specific format
parser = CustomParser()

# Custom rubric for specific evaluation
rubric = CustomRubric()

vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    parser=parser,
    rubric=rubric
)
```

### 3. Use MultiTurnEnv for Interactive Tasks
For tasks requiring multiple exchanges:

```python
class MyMultiTurnEnv(vf.MultiTurnEnv):
    def is_completed(self, messages, state, **kwargs):
        return some_completion_condition
    
    def env_response(self, messages, state, **kwargs):
        return response, updated_state
```

### 4. Always Include Format Rewards
Ensure your rubric includes format compliance:

```python
rubric = vf.Rubric(funcs=[
    task_specific_reward,
    parser.get_format_reward_func()  # Always include this
], weights=[0.8, 0.2])
```

## Best Practices

1. **Test Environments First**: Verify your environment works before large-scale training
2. **Use Built-in Datasets**: Start with `vf.load_example_dataset()` for common tasks
3. **Monitor Performance**: Use appropriate eval datasets and logging
4. **Scale Gradually**: Start with small models and datasets
5. **Document Format**: Clearly specify expected output format in system prompts
6. **Handle Errors**: Ensure parsers and rubrics handle malformed input gracefully