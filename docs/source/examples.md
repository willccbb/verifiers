# Examples

This guide walks through real-world examples from the verifiers codebase, explaining implementation patterns and design decisions.

## Math Problem Solving

The math example demonstrates core patterns for educational tasks.

### Implementation Overview

```python
# From examples/math_python.py
from verifiers.envs import SingleTurnEnv
from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric

class MathEnv(SingleTurnEnv):
    def __init__(self, dataset, **kwargs):
        # XMLParser for structured output
        parser = XMLParser(fields=["reasoning", "answer"])
        
        # Custom evaluation for mathematical equivalence
        def correct_answer_reward_func(completion, answer, **kwargs):
            parsed_completion = parser.parse(completion)
            return float(check_math_equivalence(
                parsed_completion.answer, 
                answer
            ))
        
        rubric = Rubric(
            funcs=[correct_answer_reward_func],
            weights=[1.0],
            parser=parser
        )
        
        super().__init__(
            dataset=dataset,
            system_prompt=MATH_SYSTEM_PROMPT,
            parser=parser,
            rubric=rubric,
            **kwargs
        )
```

### Key Design Decisions

1. **XMLParser for Clear Structure**: Math solutions benefit from separated reasoning and final answer.

2. **Mathematical Equivalence**: The reward function handles equivalent representations:
   ```python
   def check_math_equivalence(answer1, answer2):
       # "1/2" == "0.5" == "½"
       try:
           val1 = parse_math_answer(answer1)
           val2 = parse_math_answer(answer2)
           return abs(val1 - val2) < 1e-9
       except:
           return answer1.strip() == answer2.strip()
   ```

3. **System Prompt Engineering**:
   ```python
   MATH_SYSTEM_PROMPT = """Solve the given math problem step by step.
   
   Format your response as:
   <reasoning>
   Show your work here. Explain each step clearly.
   </reasoning>
   <answer>
   Your final numerical answer
   </answer>"""
   ```

### Usage Example

```python
# Load dataset
dataset = load_dataset("gsm8k", split="train")

# Create environment
env = MathEnv(dataset)

# Generate training data
prompts, completions, rewards = env.generate(
    model="gpt-4",
    n_samples=1000,
    temperature=0.7
)

# Example output
print(completions[0])
# <reasoning>
# Let me solve this step by step.
# First, I need to find the total number of apples...
# 5 baskets × 20 apples per basket = 100 apples
# </reasoning>
# <answer>100</answer>
```

## Tool-Augmented Math

The Smola math tools example shows advanced tool integration.

### Complex Tool Environment

```python
# From examples/smola_math_tools.py
from verifiers.envs import SmolToolEnv
from verifiers.tools import calculator, python

class ToolMathEnv(SmolToolEnv):
    def __init__(self, dataset, **kwargs):
        # Multiple tools for different operations
        tools = [
            calculator,    # Basic arithmetic
            python,       # Complex computation
            plot_tool,    # Visualization
            verify_tool   # Answer checking
        ]
        
        super().__init__(
            dataset=dataset,
            tools=tools,
            system_prompt=TOOL_MATH_PROMPT,
            **kwargs
        )
```

### Tool Usage Pattern

```python
# Model output with tool calls
completion = """
<reasoning>
I need to solve this differential equation. Let me use Python.
</reasoning>

<tool>
{
    "name": "python",
    "arguments": {
        "code": "import sympy as sp\nx = sp.Symbol('x')\neq = sp.diff(x**2 + 3*x + 2, x)\nprint(eq)"
    }
}
</tool>

<result>2*x + 3</result>

<reasoning>
The derivative is 2x + 3. Let me verify this is correct.
</reasoning>

<answer>2x + 3</answer>
"""
```

### Advanced Rubric Design

```python
from verifiers.rubrics import ToolRubric

# Rewards both correctness and appropriate tool use
rubric = ToolRubric(
    tools=tools,
    weights=[
        0.6,  # correct_answer
        0.2,  # tool_execution 
        0.1,  # format
        0.1   # tool-specific rewards
    ]
)

# Encourages:
# - Correct final answers (60%)
# - Successful tool usage (20%)
# - Proper formatting (10%)
# - Using right tool for task (10%)
```

## Interactive Games: Wordle

The Wordle example demonstrates multi-turn interaction patterns.

### Multi-Turn Environment Design

```python
# From examples/wordle.py
from verifiers.envs import MultiTurnEnv

class WordleEnv(MultiTurnEnv):
    def __init__(self, dataset, max_attempts=6, **kwargs):
        self.max_attempts = max_attempts
        super().__init__(
            dataset=dataset,
            system_prompt=WORDLE_SYSTEM_PROMPT,
            **kwargs
        )
    
    def env_response(self, messages, state):
        """Generate Wordle feedback for guess."""
        last_message = messages[-1]["content"]
        guess = self.extract_guess(last_message)
        target = state["target_word"]
        
        if not self.is_valid_word(guess):
            return "Invalid word. Please guess a valid 5-letter word.", state
        
        # Generate color feedback
        feedback = self.get_color_feedback(guess, target)
        state["attempts"] += 1
        state["history"].append((guess, feedback))
        
        if guess == target:
            state["won"] = True
            return f"🎉 Correct! The word was {target}.", state
        
        return self.format_feedback(feedback), state
    
    def is_completed(self, messages, state):
        """Check if game should end."""
        return state.get("won", False) or state.get("attempts", 0) >= self.max_attempts
```

### State Management Pattern

```python
def initial_state(self, task_data):
    """Initialize game state."""
    return {
        "target_word": task_data["answer"],
        "attempts": 0,
        "history": [],
        "won": False
    }

def get_color_feedback(self, guess, target):
    """Generate Wordle-style feedback."""
    feedback = []
    for i, (g, t) in enumerate(zip(guess, target)):
        if g == t:
            feedback.append("🟩")  # Green - correct position
        elif g in target:
            feedback.append("🟨")  # Yellow - wrong position
        else:
            feedback.append("⬜")  # Gray - not in word
    return "".join(feedback)
```

### Custom Scoring

```python
def score_rollout_sync(self, prompt, messages, answer, state):
    """Score based on success and efficiency."""
    scores = {}
    
    # Base success score
    scores["success"] = 1.0 if state.get("won", False) else 0.0
    
    # Efficiency bonus
    if state.get("won", False):
        attempts = state.get("attempts", 6)
        scores["efficiency"] = (7 - attempts) / 6  # More points for fewer guesses
    else:
        scores["efficiency"] = 0.0
    
    # Strategy score (using feedback effectively)
    scores["strategy"] = self.evaluate_strategy(state.get("history", []))
    
    # Weighted combination
    scores["reward"] = (
        scores["success"] * 0.7 +
        scores["efficiency"] * 0.2 +
        scores["strategy"] * 0.1
    )
    
    return scores
```

## Code Generation with Testing

The ARC (Abstract Reasoning Corpus) example shows code generation evaluation.

### Code Evaluation Pattern

```python
# From examples/arc_1d.py
from verifiers.rubrics import ToolRubric

class CodeGenRubric(ToolRubric):
    def evaluate_code(self, code, test_cases):
        """Run code against test cases."""
        results = []
        
        for test in test_cases:
            try:
                # Create safe execution environment
                namespace = {"input_seq": test["input"]}
                
                # Execute generated code
                exec(code, namespace)
                
                # Check output
                output = namespace.get("output_seq", [])
                expected = test["output"]
                
                results.append(output == expected)
                
            except Exception as e:
                results.append(False)
        
        # Return proportion of passed tests
        return sum(results) / len(results) if results else 0.0
```

### Structured Code Generation

```python
CODE_SYSTEM_PROMPT = """Generate Python code to transform the input sequence.

Format your response as:
<reasoning>
Analyze the pattern and explain your approach
</reasoning>

<code>
def transform(input_seq):
    # Your transformation logic here
    return output_seq

output_seq = transform(input_seq)
</code>"""

parser = XMLParser(fields=["reasoning", "code"])
```

### Progressive Test Evaluation

```python
def progressive_scoring(code, test_cases):
    """Award partial credit for partially correct solutions."""
    scores = []
    
    for i, test in enumerate(test_cases):
        result = run_test(code, test)
        
        if result["status"] == "correct":
            scores.append(1.0)
        elif result["status"] == "partial":
            # Check how close the output is
            similarity = calculate_similarity(
                result["output"], 
                test["expected"]
            )
            scores.append(similarity * 0.5)
        else:
            scores.append(0.0)
    
    # Weight earlier tests more (they're usually simpler)
    weights = [1.0 / (i + 1) for i in range(len(test_cases))]
    weighted_score = sum(s * w for s, w in zip(scores, weights))
    
    return weighted_score / sum(weights)
```

## Self-Improvement: DoubleCheck

The DoubleCheck example shows self-verification patterns.

### Two-Stage Verification

```python
# From examples/doublecheck.py
from verifiers.envs import DoubleCheckEnv

class MathDoubleCheckEnv(DoubleCheckEnv):
    def __init__(self, dataset, **kwargs):
        super().__init__(
            dataset=dataset,
            solve_prompt=SOLVE_PROMPT,
            check_prompt=CHECK_PROMPT,
            system_prompt=SYSTEM_PROMPT,
            **kwargs
        )
    
    def check_solution(self, solution, check_response):
        """Determine if solution is verified as correct."""
        check_parsed = self.parser.parse(check_response)
        
        # Look for explicit verification
        is_correct = "correct" in check_parsed.answer.lower()
        has_error = "error" in check_parsed.reasoning.lower()
        
        return is_correct and not has_error
```

### Verification Prompts

```python
SOLVE_PROMPT = "Solve this problem: {question}"

CHECK_PROMPT = """Review this solution:

Problem: {question}
Solution: {solution}

Is the solution correct? If not, identify any errors.

<reasoning>
Verify each step of the solution
</reasoning>
<answer>
State whether the solution is "correct" or "incorrect"
</answer>"""
```

### Self-Consistency Rewards

```python
def self_consistency_reward(messages, state):
    """Reward consistent reasoning across solve/check."""
    solve_reasoning = state["solve_parsed"].reasoning
    check_reasoning = state["check_parsed"].reasoning
    
    # Extract key concepts/numbers from both
    solve_concepts = extract_concepts(solve_reasoning)
    check_concepts = extract_concepts(check_reasoning)
    
    # Reward mentioning same key ideas
    overlap = len(solve_concepts & check_concepts)
    total = len(solve_concepts | check_concepts)
    
    return overlap / total if total > 0 else 0.0
```

## Best Practices from Examples

### 1. Start Simple, Add Complexity

```python
# Phase 1: Basic environment
env = SingleTurnEnv(dataset=dataset)

# Phase 2: Add parsing
env = SingleTurnEnv(
    dataset=dataset,
    parser=XMLParser(["answer"])
)

# Phase 3: Add evaluation
env = SingleTurnEnv(
    dataset=dataset,
    parser=XMLParser(["reasoning", "answer"]),
    rubric=Rubric(funcs=[correct_answer], weights=[1.0])
)

# Phase 4: Multi-criteria evaluation
env = SingleTurnEnv(
    dataset=dataset,
    parser=parser,
    rubric=Rubric(
        funcs=[correct_answer, reasoning_quality, efficiency],
        weights=[0.6, 0.3, 0.1]
    )
)
```

### 2. Design for Debugging

```python
class DebugEnv(SingleTurnEnv):
    def rollout(self, *args, **kwargs):
        """Add logging for debugging."""
        print(f"Prompt: {args[2][:50]}...")
        
        completion, state = super().rollout(*args, **kwargs)
        
        parsed = self.parser.parse(completion)
        print(f"Answer extracted: {parsed.answer}")
        print(f"State: {state}")
        
        return completion, state
```

### 3. Flexible Task Configuration

```python
def create_math_env(difficulty="easy", tools=False, verification=False):
    """Factory function for different math environments."""
    dataset = load_math_dataset(difficulty)
    
    if tools and verification:
        return ToolDoubleCheckMathEnv(dataset)
    elif tools:
        return ToolMathEnv(dataset)
    elif verification:
        return DoubleCheckMathEnv(dataset)
    else:
        return MathEnv(dataset)
```

### 4. Reusable Components

```python
# Shared parsers
MATH_PARSER = XMLParser(["reasoning", "answer"])
CODE_PARSER = XMLParser(["reasoning", "code"])

# Shared reward functions
def get_format_reward(parser):
    return parser.get_format_reward_func()

def get_length_penalty(max_length=1000):
    def penalty(completion, **kwargs):
        return max(0, 1 - len(completion) / max_length)
    return penalty

# Compose into environments
env = SingleTurnEnv(
    parser=MATH_PARSER,
    rubric=Rubric(
        funcs=[
            correct_answer,
            get_format_reward(MATH_PARSER),
            get_length_penalty(500)
        ],
        weights=[0.7, 0.2, 0.1]
    )
)
```

These examples demonstrate the flexibility and power of the verifiers framework. By understanding these patterns, you can create sophisticated evaluation environments for your own tasks.