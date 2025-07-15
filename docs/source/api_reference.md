# API Reference

This document provides detailed API documentation for the core verifiers components.

## Environments

### SingleTurnEnv

**Most users should start with `SingleTurnEnv`** for one-shot question-answer tasks.

```python
class SingleTurnEnv(MultiTurnEnv):
    """Single-turn environment for Q&A tasks."""
    
    def __init__(
        self,
        dataset: Dataset,
        system_prompt: str | None = None,
        few_shot: List[Dict[str, str]] | None = None,
        parser: Parser | None = None,
        rubric: Rubric | None = None,
        message_type: str = "chat",  # "chat" or "completion"
        sampling_args: Dict[str, Any] | None = None,
        **kwargs
    ):
        """
        Args:
            dataset: Dataset with 'question' and 'answer' columns or 'info' dict
            system_prompt: System prompt for the model
            few_shot: Few-shot examples as message lists
            parser: Parser for extracting structured output
            rubric: Rubric for evaluation
            message_type: "chat" (recommended) or "completion" format
            sampling_args: vLLM-specific arguments for fine-grained control
        """
```

**Dataset Format**: Datasets should have either `answer` (str) or `info` (dict) columns:

```python
from datasets import Dataset
from typing import List, Dict, Any

# Option 1: Simple format with answer column
dataset: Dataset = Dataset.from_list([
    {"question": "What is 2+2?", "answer": "4"},
    {"question": "What is 3*5?", "answer": "15"},
])

# Option 2: Complex format with info dict
dataset: Dataset = Dataset.from_list([
    {
        "question": "Solve this math problem: 2+2", 
        "info": {
            "answer": "4",
            "difficulty": "easy",
            "category": "arithmetic"
        }
    }
])
```

**Message Types**: The framework supports two formats:

```python
# Chat format (recommended for most cases)
message_type = "chat"
# Input: List[Dict[str, str]] with "role" and "content" keys
# Example: [{"role": "user", "content": "What is 2+2?"}]
# Supports: system prompts, multi-turn conversations

# Completion format (for legacy models)
message_type = "completion"  
# Input: str (raw text)
# Example: "What is 2+2?"
# Limited: no system prompts
```

**Sampling Arguments**: Pass vLLM-specific arguments for fine-grained control:

```python
from typing import Dict, Any

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
```

### MultiTurnEnv

Base class for interactive environments:

```python
class MultiTurnEnv(Environment):
    """Base class for multi-turn interactive environments."""
    
    def __init__(
        self,
        dataset: Dataset,
        system_prompt: str | None = None,
        few_shot: List[Dict[str, str]] | None = None,
        parser: Parser | None = None,
        rubric: Rubric | None = None,
        message_type: str = "chat",  # "chat" or "completion"
        sampling_args: Dict[str, Any] | None = None,
        **kwargs
    ):
        """
        Args:
            dataset: Dataset with 'question' and 'answer' columns or 'info' dict
            system_prompt: System prompt for the model
            few_shot: Optional few-shot examples (not recommended)
            parser: Parser for extracting structured output
            rubric: Rubric for evaluation
            message_type: "chat" (recommended) or "completion" format
            sampling_args: vLLM-specific arguments for fine-grained control
        """
    
    def is_completed(
        self, 
        messages: List[Dict[str, str]], 
        state: Dict[str, Any], 
        **kwargs
    ) -> bool:
        """
        Define when the conversation should end.
        
        Args:
            messages: List of conversation messages
            state: Environment state dictionary
            **kwargs: Additional arguments
            
        Returns:
            True if conversation should end, False otherwise
        """
        raise NotImplementedError
    
    def env_response(
        self, 
        messages: List[Dict[str, str]], 
        state: Dict[str, Any], 
        **kwargs
    ) -> tuple[Dict[str, str], Dict[str, Any]]:
        """
        Define how the environment responds to the model.
        
        Args:
            messages: List of conversation messages
            state: Environment state dictionary
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (response_message, updated_state)
        """
        raise NotImplementedError
```

**State Object**: The `state` object contains rollout information and accumulates LLM responses:

```python
from typing import Dict, Any, List

# State structure
state: Dict[str, Any] = {
    "prompt": List[Dict[str, str]],  # Original prompt
    "completion": List[Dict[str, str]],  # Model's response
    "answer": str,  # Ground truth answer
    "task": str,  # Task identifier
    "info": Dict[str, Any],  # Additional metadata
    "responses": List[Any],  # Full LLM response objects with token_ids, logprobs, etc.
}
```

The `state["responses"]` list accumulates the complete LLM response objects, which can include:
- `token_ids`: List of token IDs
- `logprobs`: Token-level log probabilities  
- `finish_reason`: Why generation stopped
- `usage`: Token usage statistics

### ToolEnv

Environment for tool-augmented reasoning:

```python
class ToolEnv(MultiTurnEnv):
    """Environment for tool-augmented reasoning tasks."""
    
    def __init__(
        self,
        dataset: Dataset,
        system_prompt: str | None = None,
        few_shot: List[Dict[str, str]] | None = None,
        tools: List[Callable] | None = None,
        max_steps: int = 3,
        message_type: str = "chat",  # "chat" or "completion"
        sampling_args: Dict[str, Any] | None = None,
        **kwargs
    ):
        """
        Args:
            dataset: Dataset with 'question' and 'answer' columns or 'info' dict
            system_prompt: System prompt for the model
            few_shot: Optional few-shot examples (not recommended)
            tools: List of available tools (callable functions)
            max_steps: Maximum number of tool use steps
            message_type: "chat" (recommended) or "completion" format
            sampling_args: vLLM-specific arguments for fine-grained control
        """
```

## Parsers

### Parser

Base parser class:

```python
class Parser:
    """Base parser class for extracting structured information from model outputs."""
    
    def parse(self, text: str) -> Any:
        """
        Parse text and return structured data.
        
        Args:
            text: Raw text to parse
            
        Returns:
            Parsed structured data (default: returns text as-is)
        """
        return text
    
    def parse_answer(
        self, 
        completion: Union[str, List[Dict[str, str]]]
    ) -> str | None:
        """
        Extract the final answer from a completion.
        
        Args:
            completion: Either a string (completion format) or list of messages (chat format)
            
        Returns:
            Extracted answer string or None if not found
        """
        if isinstance(completion, str):
            return self.parse(completion)
        else:
            # For chat format, parse the last message's content
            return self.parse(completion[-1]["content"])
    
    def get_format_reward_func(self) -> Callable:
        """
        Return a reward function that checks format compliance.
        
        Returns:
            Function that returns float reward for format compliance
        """
        def format_reward_func(completion: List[Dict[str, str]], **kwargs) -> float:
            return 1.0  # Default: always return 1.0
        return format_reward_func
    
    def get_assistant_messages(self, completion: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Helper function to extract assistant messages from a completion.
        
        Args:
            completion: List of conversation messages
            
        Returns:
            List of assistant messages only
        """
        return [msg for msg in completion if msg['role'] == 'assistant']
```

### XMLParser

Convenience parser for XML-tagged field extraction:

```python
class XMLParser(Parser):
    """Parser for extracting structured fields from XML-tagged output."""
    
    def __init__(
        self, 
        fields: List[str | Tuple[str, ...]] | None = None
    ):
        """
        Args:
            fields: List of field names or tuples of alternative names
                   Example: ["reasoning", "answer"] or [("reasoning", "thinking"), "answer"]
        """
    
    def parse(self, text: str) -> Any:
        """
        Parse XML-tagged text and extract fields.
        
        Args:
            text: XML-tagged text
            
        Returns:
            Object with extracted fields as attributes
        """
```

### ThinkParser

Convenience parser for step-by-step reasoning:

```python
class ThinkParser(Parser):
    """Parser for extracting content after </think> tags."""
    
    def __init__(self, extract_fn: Callable[[str], str] | None = None):
        """
        Args:
            extract_fn: Optional function to extract final answer from text
                       Example: extract boxed answers with r'\\boxed\{([^}]+)\}'
        """
    
    def parse(self, text: str) -> str:
        """
        Extract content after </think> tags.
        
        Args:
            text: Text containing <think>...</think> tags
            
        Returns:
            Content after </think> tags
        """
```

## Rubrics

### Rubric

Base rubric class for combining multiple reward functions:

```python
class Rubric:
    """Base rubric class for combining multiple reward functions."""
    
    def __init__(
        self,
        funcs: List[Callable] | None = None,
        weights: List[float] | None = None,
        parser: Parser | None = None
    ):
        """
        Args:
            funcs: List of reward functions
            weights: List of weights for each function (should sum to reasonable value)
            parser: Parser for extracting structured output
        """
    
    def score_rollout_sync(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        completion: Union[str, List[Dict[str, str]]],
        answer: str,
        state: Dict[str, Any] | None = None,
        task: str | None = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Score a single rollout synchronously.
        
        Args:
            prompt: Input prompt (string or messages)
            completion: Model response (string or messages)
            answer: Ground truth answer
            state: Environment state dictionary
            task: Task type identifier
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with individual scores and weighted reward
        """
    
    def score_rollouts(
        self,
        prompts: List[Union[str, List[Dict[str, str]]]],
        completions: List[Union[str, List[Dict[str, str]]]],
        answers: List[str],
        states: List[Dict[str, Any]] | None = None,
        tasks: List[str] | None = None,
        **kwargs
    ) -> Dict[str, List[float]]:
        """
        Score multiple rollouts efficiently.
        
        Args:
            prompts: List of input prompts
            completions: List of model responses
            answers: List of ground truth answers
            states: List of environment state dictionaries
            tasks: List of task type identifiers
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with lists of individual scores and weighted rewards
        """
```

### ToolRubric

Rubric for evaluating tool usage:

```python
class ToolRubric(Rubric):
    """Rubric for evaluating tool-augmented reasoning."""
    
    def __init__(
        self,
        tools: List[Callable],
        weights: List[float] | None = None,
        parser: Parser | None = None
    ):
        """
        Args:
            tools: List of available tools
            weights: Weights for [correct_answer, tool_execution, format]
            parser: Parser for extracting structured output
        """
```

**Tool-specific rewards**:
- `{tool}_reward`: 1.0 if tool used successfully, 0.0 otherwise
- `{tool}_count`: Number of successful tool uses
- `{tool}_attempt`: Number of tool use attempts

### JudgeRubric

LLM-based evaluation rubric:

```python
class JudgeRubric(Rubric):
    """Rubric using another LLM to evaluate responses."""
    
    def __init__(
        self,
        judge_models: List[str],
        client: Any,
        template: str,
        parser: Parser | None = None,
        **kwargs
    ):
        """
        Args:
            judge_models: List of model names to use as judges
            client: OpenAI-compatible client
            template: Prompt template for evaluation
            parser: Parser for extracting judge scores
        """
```

### RubricGroup

Combines multiple rubrics:

```python
class RubricGroup:
    """Combines multiple rubrics for comprehensive evaluation."""
    
    def __init__(self, rubrics: List[Rubric]):
        """
        Args:
            rubrics: List of rubrics to combine
        """
    
    def score_rollouts(
        self,
        prompts: List[Union[str, List[Dict[str, str]]]],
        completions: List[Union[str, List[Dict[str, str]]]],
        answers: List[str],
        states: List[Dict[str, Any]] | None = None,
        tasks: List[str] | None = None,
        **kwargs
    ) -> Dict[str, List[float]]:
        """
        Score using all rubrics and aggregate results.
        
        Returns:
            Dictionary with aggregated scores from all rubrics
        """
```

## Environment Methods

### evaluate()

Evaluate a model on the environment:

```python
def evaluate(
    self,
    client: Any,
    model: str,
    num_examples: int | None = None,
    rollouts_per_example: int = 1,
    sampling_args: Dict[str, Any] | None = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Evaluate a model on the environment.
    
    Args:
        client: OpenAI-compatible client
        model: Model name to evaluate
        num_examples: Number of examples to evaluate (None = all)
        rollouts_per_example: Number of rollouts per example
        sampling_args: vLLM-specific arguments for fine-grained control
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with evaluation results including:
        - prompts: List of prompts used
        - completions: List of model responses
        - states: List of environment states
        - rewards: List of reward scores
        - metrics: Aggregated performance metrics
    """
```

### generate()

Generate training data with rewards:

```python
def generate(
    self,
    client: Any,
    model: str,
    n_samples: int,
    sampling_args: Dict[str, Any] | None = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate training data with rewards.
    
    Args:
        client: OpenAI-compatible client
        model: Model name to generate with
        n_samples: Number of samples to generate
        sampling_args: vLLM-specific arguments for fine-grained control
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with generated data including:
        - prompts: List of prompts
        - completions: List of model responses
        - states: List of environment states
        - rewards: List of reward scores
    """
```

### process_env_results()

Process results for training:

```python
def process_env_results(
    self,
    prompts: List[Union[str, List[Dict[str, str]]]],
    completions: List[Union[str, List[Dict[str, str]]]],
    states: List[Dict[str, Any]],
    rewards: List[float],
    processing_class: Any | None = None,
    **kwargs
) -> Any:
    """
    Process environment results for training.
    
    Args:
        prompts: List of prompts
        completions: List of model responses
        states: List of environment states
        rewards: List of reward scores
        processing_class: Tokenizer or processing class
        **kwargs: Additional arguments
        
    Returns:
        Processed data ready for training
    """
```

## Key Type Definitions

```python
from typing import Union, List, Dict, Any, Callable

# Message formats
Messages = List[Dict[str, str]]  # Chat format
Completion = Union[str, Messages]  # Either format

# Dataset formats
DatasetRow = Dict[str, Union[str, Dict[str, Any]]]  # question + answer/info

# State object
State = Dict[str, Any]  # Environment state dictionary

# Reward function signature
RewardFunc = Callable[..., float]  # Returns float reward

# Parser signature
ParserFunc = Callable[[str], Any]  # Parses text to structured data

# Sampling arguments
SamplingArgs = Dict[str, Any]  # vLLM-specific arguments
```

## Message Format Examples

### Chat Format (Recommended)

```python
from typing import List, Dict

# Single message
messages: List[Dict[str, str]] = [
    {"role": "user", "content": "What is 2+2?"}
]

# Multi-turn conversation
messages: List[Dict[str, str]] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "Let me think about this..."},
    {"role": "user", "content": "Can you be more specific?"}
]


```

### Completion Format (Legacy)

```python
# Simple text completion
completion: str = "The answer is 4."

# With structured output
completion: str = """
<reasoning>
To solve 2+2, I need to add two and two together.
Two plus two equals four.
</reasoning>
<answer>
4
</answer>
"""
```

## State Object Details

The state object accumulates information throughout the rollout:

```python
from typing import Dict, Any, List

# Initial state
state: Dict[str, Any] = {
    "prompt": [],  # Will be populated with original prompt
    "completion": [],  # Will be populated with model response
    "answer": "",  # Ground truth answer
    "task": "",  # Task identifier
    "info": {},  # Additional metadata from dataset
    "responses": [],  # Will accumulate LLM response objects
}

# After first turn
state: Dict[str, Any] = {
    "prompt": [{"role": "user", "content": "What is 2+2?"}],
    "completion": [{"role": "assistant", "content": "Let me think..."}],
    "answer": "4",
    "task": "math",
    "info": {"difficulty": "easy"},
    "responses": [response_object],  # Full LLM response with token_ids, logprobs, etc.
}
```

## Sampling Arguments Reference

Common vLLM-specific arguments:

```python
from typing import Dict, Any

# Basic sampling
sampling_args: Dict[str, Any] = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 2048,
    "stop": ["</answer>", "\n\n"],
}

# With token-level information
sampling_args: Dict[str, Any] = {
    "temperature": 0.7,
    "max_tokens": 2048,
    "extra_body": {
        "logprobs": True,  # Get token-level logprobs
        "top_logprobs": 5,  # Top-k logprobs per token
        "skip_special_tokens": False,  # vLLM flag
        "spaces_between_special_tokens": False,  # vLLM flag
    }
}

# Advanced vLLM features
sampling_args: Dict[str, Any] = {
    "extra_body": {
        "use_beam_search": True,  # Use beam search
        "best_of": 5,  # Number of candidates
        "ignore_eos": True,  # Don't stop at EOS token
    }
}
```