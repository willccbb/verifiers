# Examples

The examples folder contains working implementations that demonstrate how to use the verifiers framework. These examples serve as the **ground truth for usage patterns** and show both evaluation and training workflows.

## Quick Start: Basic Evaluation

**Most users should start here** - a simple evaluation of a model on a math reasoning task:

```python
import verifiers as vf
from typing import Dict, Any, Union, List
from datasets import Dataset
from openai import OpenAI

# 1. Load dataset
dataset: Dataset = vf.load_example_dataset("gsm8k", split="train")

# 2. Setup parser and rubric
parser = vf.ThinkParser()

def correct_answer(completion: Union[str, List[Dict[str, str]]], answer: str, **kwargs) -> float:
    response = parser.parse_answer(completion) or ''
    return 1.0 if response.strip() == answer.strip() else 0.0

rubric = vf.Rubric(
    funcs=[correct_answer, parser.get_format_reward_func()],
    weights=[0.8, 0.2]
)

# 3. Create environment
vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    system_prompt="Think step-by-step inside <think>...</think> tags, then give your answer.",
    parser=parser,
    rubric=rubric,
    message_type="chat"  # Recommended: use "chat" for most cases
)

# 4. Evaluate the environment
client = OpenAI()
results: Dict[str, Any] = vf_env.evaluate(
    client=client,
    model="gpt-4",
    num_examples=10
)

print(f"Results: {results}")
```

## Dataset Format Examples

The framework supports two dataset formats:

### Simple Format (answer column)

```python
from datasets import Dataset
from typing import List, Dict

# Option 1: Simple format with answer column
dataset: Dataset = Dataset.from_list([
    {"question": "What is 2+2?", "answer": "4"},
    {"question": "What is 3*5?", "answer": "15"},
    {"question": "What is 10-7?", "answer": "3"},
])
```

### Complex Format (info dict)

```python
from datasets import Dataset
from typing import List, Dict, Any

# Option 2: Complex format with info dict
dataset: Dataset = Dataset.from_list([
    {
        "question": "Solve this math problem: 2+2", 
        "info": {
            "answer": "4",
            "difficulty": "easy",
            "category": "arithmetic",
            "explanation": "Basic addition"
        }
    },
    {
        "question": "What is the capital of France?",
        "info": {
            "answer": "Paris",
            "difficulty": "easy", 
            "category": "geography",
            "explanation": "Capital city of France"
        }
    }
])
```

## Message Type Examples

### Chat Format (Recommended)

```python
from typing import List, Dict

# Chat format supports system prompts
system_prompt: str = "You are a helpful math tutor. Think step-by-step."

vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    system_prompt=system_prompt,
    message_type="chat"  # Recommended format
)
```

### Completion Format (Legacy)

```python
# Completion format is simpler but more limited
vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    message_type="completion"  # Legacy format
)
```

## Advanced Evaluation with Custom Components

### Custom Parser

```python
from verifiers.parsers import Parser
from typing import Any, Dict
import re

class MathParser(Parser):
    """Custom parser for mathematical reasoning."""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Extract reasoning and answer from math response."""
        # Extract reasoning between <think> tags
        reasoning_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        
        # Extract final answer
        answer_match = re.search(r'The answer is (\d+)', text)
        answer = answer_match.group(1) if answer_match else ""
        
        return {
            "reasoning": reasoning,
            "answer": answer,
            "has_reasoning": len(reasoning) > 0,
            "has_answer": len(answer) > 0
        }
    
    def get_format_reward_func(self):
        """Reward function for proper formatting."""
        def format_reward(completion: Union[str, List[Dict[str, str]]], **kwargs) -> float:
            if isinstance(completion, str):
                text = completion
            else:
                text = completion[-1]["content"]
            
            # Check for required elements
            has_think = "<think>" in text and "</think>" in text
            has_answer = "The answer is" in text
            
            return 1.0 if (has_think and has_answer) else 0.0
        
        return format_reward

# Use custom parser
parser = MathParser()
```

### Custom Rubric

```python
from verifiers.rubrics import Rubric
from typing import Union, List, Dict

def correctness_reward(completion: Union[str, List[Dict[str, str]]], answer: str, **kwargs) -> float:
    """Check if the answer is correct."""
    parsed = parser.parse(completion)
    return 1.0 if parsed["answer"] == answer else 0.0

def reasoning_quality_reward(completion: Union[str, List[Dict[str, str]]], **kwargs) -> float:
    """Evaluate reasoning quality."""
    parsed = parser.parse(completion)
    reasoning = parsed["reasoning"]
    
    # Reward for step-by-step reasoning
    steps = reasoning.count('\n') + 1
    return min(steps / 5.0, 1.0)  # Up to 5 steps = full score

def mathematical_accuracy_reward(completion: Union[str, List[Dict[str, str]]], **kwargs) -> float:
    """Check for mathematical errors in reasoning."""
    parsed = parser.parse(completion)
    reasoning = parsed["reasoning"].lower()
    
    # Check for common math errors
    errors = 0
    if '2+2=5' in reasoning:
        errors += 1
    if 'divide by zero' in reasoning:
        errors += 1
    
    return max(0.0, 1.0 - errors * 0.5)  # -0.5 per error

# Create comprehensive rubric
rubric = Rubric(
    funcs=[
        correctness_reward,
        reasoning_quality_reward,
        mathematical_accuracy_reward,
        parser.get_format_reward_func()
    ],
    weights=[0.5, 0.2, 0.2, 0.1]
)
```

## Tool-Augmented Reasoning

### Basic Tool Environment

```python
from verifiers.tools import calculator
from typing import List, Callable

# Create tool environment
vf_env = vf.ToolEnv(
    dataset=dataset,
    system_prompt="""
    Think step-by-step inside <think>...</think> tags.
    If you need to calculate, use the calculator tool inside <tool>...</tool> tags.
    Then give your final answer inside <answer>...</answer> tags.
    """,
    tools=[calculator],  # List[Callable]
    max_steps=3,
    message_type="chat"  # Recommended format
)

# Evaluate with tools
results: Dict[str, Any] = vf_env.evaluate(
    client=client,
    model="gpt-4",
    num_examples=10
)
```

### Advanced Tool Environment with Smolagents

```python
from verifiers.envs.smola_tool_env import SmolaToolEnv
from typing import List, Callable

try:    
    from smolagents.default_tools import PythonInterpreterTool
    from verifiers.tools.smolagents import CalculatorTool
except ImportError:
    raise ImportError("Please install smolagents")

# Setup tools
python_tool: PythonInterpreterTool = PythonInterpreterTool(
    authorized_imports=["math", "sympy", "numpy"]
)
calculator_tool: CalculatorTool = CalculatorTool()

# Create advanced tool environment
vf_env = SmolaToolEnv(
    dataset=dataset,
    system_prompt=MATH_SMOLA_PROMPT_TEMPLATE,
    tools=[python_tool, calculator_tool],  # List[Callable]
    max_steps=5,
    message_type="chat"  # Recommended format
)
```

## Multi-Turn Conversations

### Custom Multi-Turn Environment

```python
from verifiers import MultiTurnEnv
from typing import List, Dict, Any

class MathTutorEnv(MultiTurnEnv):
    """Interactive math tutoring environment."""
    
    def is_completed(
        self, 
        messages: List[Dict[str, str]], 
        state: Dict[str, Any], 
        **kwargs
    ) -> bool:
        """End when student gets correct answer or gives up."""
        # Check if student got the answer right
        if state.get("correct_answer_given"):
            return True
        
        # End after 5 exchanges
        if len(state.get("responses", [])) >= 5:
            return True
        
        return False
    
    def env_response(
        self, 
        messages: List[Dict[str, str]], 
        state: Dict[str, Any], 
        **kwargs
    ) -> tuple[Dict[str, str], Dict[str, Any]]:
        """Provide tutoring feedback."""
        last_message = messages[-1]["content"]
        correct_answer = state["answer"]
        
        # Check if student got it right
        if correct_answer in last_message:
            state["correct_answer_given"] = True
            return {"role": "user", "content": "Great job! You got it right!"}, state
        
        # Provide hints
        hints = [
            "Try breaking it down step by step.",
            "What operation do you need to perform?",
            "Think about what the question is asking for.",
            "You're close! Double-check your calculation.",
            "Let me help you with a hint..."
        ]
        
        hint_index = len(state.get("responses", [])) - 1
        hint = hints[min(hint_index, len(hints) - 1)]
        
        return {"role": "user", "content": hint}, state

# Use the custom environment
vf_env = MathTutorEnv(
    dataset=dataset,
    system_prompt="You are a helpful math tutor. Guide the student step by step.",
    message_type="chat"  # Recommended format
)
```

## State and Token Information

### Accessing Token-Level Data

```python
from typing import Dict, Any, List, Union

def token_aware_reward(
    completion: Union[str, List[Dict[str, str]]], 
    state: Dict[str, Any], 
    **kwargs
) -> float:
    """Reward function that uses token-level information."""
    
    # Access token information from state
    if state.get("responses"):
        last_response = state["responses"][-1]
        
        # Check if we have token-level data
        if hasattr(last_response, 'choices') and last_response.choices:
            choice = last_response.choices[0]
            
            # Access logprobs if available
            if hasattr(choice, 'logprobs') and choice.logprobs:
                logprobs = choice.logprobs.content
                token_ids = choice.logprobs.token_ids
                
                # Calculate average log probability
                if logprobs:
                    avg_logprob = sum(logprob.logprob for logprob in logprobs) / len(logprobs)
                    return max(0.0, min(1.0, (avg_logprob + 5.0) / 5.0))  # Normalize to [0,1]
    
    # Fallback to basic reward
    return 1.0 if "correct" in str(completion).lower() else 0.0

# Use in rubric
rubric = Rubric(
    funcs=[correctness_reward, token_aware_reward],
    weights=[0.8, 0.2]
)
```

### Sampling Arguments for Fine-Grained Control

```python
from typing import Dict, Any

# vLLM-specific sampling arguments
sampling_args: Dict[str, Any] = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 2048,
    "extra_body": {
        "skip_special_tokens": False,  # vLLM flag
        "spaces_between_special_tokens": False,  # vLLM flag
        "logprobs": True,  # Get token-level logprobs
        "top_logprobs": 5,  # Top-k logprobs per token
    }
}

# Use in environment
vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    sampling_args=sampling_args
)

# Or pass during evaluation
results = vf_env.evaluate(
    client=client,
    model="gpt-4",
    sampling_args=sampling_args
)
```

## Training Workflows

### Generate Training Data

```python
from typing import Dict, Any

# Generate training data with rewards
results: Dict[str, Any] = vf_env.generate(
    client=client,
    model="gpt-4",
    n_samples=1000,
    sampling_args=sampling_args
)

# Process for training
processed_data = vf_env.process_env_results(
    prompts=results['prompts'],
    completions=results['completions'],
    states=results['states'],
    rewards=results['rewards'],
    processing_class=tokenizer
)
```

### Custom Training Loop

```python
from typing import List, Dict, Any
import torch

# Custom training with environment rewards
def train_with_environment(
    model: torch.nn.Module,
    vf_env: vf.SingleTurnEnv,
    client: Any,
    num_epochs: int = 10
):
    """Train model using environment rewards."""
    
    for epoch in range(num_epochs):
        # Generate training data
        results: Dict[str, Any] = vf_env.generate(
            client=client,
            model="gpt-4",
            n_samples=100
        )
        
        # Process data
        processed_data = vf_env.process_env_results(
            prompts=results['prompts'],
            completions=results['completions'],
            states=results['states'],
            rewards=results['rewards']
        )
        
        # Train model (your training logic here)
        # ...
        
        # Evaluate progress
        eval_results = vf_env.evaluate(
            client=client,
            model="gpt-4",
            num_examples=50
        )
        
        print(f"Epoch {epoch}: Results = {eval_results}")
```

## Complete Working Example

Here's a complete example that demonstrates all the key concepts:

```python
import verifiers as vf
from typing import Dict, Any, Union, List
from datasets import Dataset
from openai import OpenAI

# 1. Load and prepare dataset
dataset: Dataset = vf.load_example_dataset("gsm8k", split="train")

# 2. Create custom parser
class MathParser(vf.Parser):
    def parse(self, text: str) -> Dict[str, Any]:
        import re
        reasoning_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        answer_match = re.search(r'The answer is (\d+)', text)
        
        return {
            "reasoning": reasoning_match.group(1).strip() if reasoning_match else "",
            "answer": answer_match.group(1) if answer_match else "",
        }

# 3. Create custom rubric
parser = MathParser()

def correctness_reward(completion: Union[str, List[Dict[str, str]]], answer: str, **kwargs) -> float:
    parsed = parser.parse(completion)
    return 1.0 if parsed["answer"] == answer else 0.0

def reasoning_reward(completion: Union[str, List[Dict[str, str]]], **kwargs) -> float:
    parsed = parser.parse(completion)
    return 1.0 if len(parsed["reasoning"]) > 20 else 0.0

rubric = vf.Rubric(
    funcs=[correctness_reward, reasoning_reward],
    weights=[0.8, 0.2]
)

# 4. Create environment
vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    system_prompt="Think step-by-step inside <think>...</think> tags, then give your answer.",
    parser=parser,
    rubric=rubric,
    message_type="chat"  # Recommended format
)

# 5. Evaluate
client = OpenAI()
results: Dict[str, Any] = vf_env.evaluate(
    client=client,
    model="gpt-4",
    num_examples=10
)

print(f"Results: {results}")
```

## Key Takeaways

1. **Start with SingleTurnEnv**: Most users should begin with `SingleTurnEnv` for Q&A tasks
2. **Use Chat Format**: Use `message_type="chat"` for better flexibility and features
3. **Write Custom Components**: For nontrivial tasks, write custom parsers and rubrics
4. **Include Format Rewards**: Always include `parser.get_format_reward_func()` in rubrics
5. **Access Token Information**: Use `state["responses"]` to access token-level data when available
6. **Use Sampling Args**: Pass vLLM-specific arguments through `sampling_args` for fine-grained control
7. **Type Hints**: Use proper type hints for better code clarity and IDE support

The examples folder contains many more working implementations that demonstrate these patterns in practice.