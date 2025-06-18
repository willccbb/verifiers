# API Reference

Complete API documentation for the verifiers framework.

## Environments

### Base Environment

```python
class Environment:
    """Base class for all environments."""
    
    def __init__(
        self,
        dataset: List[Dict[str, Any]],
        system_prompt: str = "",
        parser: Parser = None,
        rubric: Rubric = None,
        client: Any = None,
        message_type: str = "chat",
        **kwargs
    ):
        """
        Args:
            dataset: List of task dictionaries with 'prompt', 'answer', etc.
            system_prompt: System message for the model
            parser: Parser for extracting structured output
            rubric: Rubric for evaluation
            client: OpenAI-compatible client
            message_type: "chat" or "completion"
            **kwargs: Additional attributes stored on the environment
        """
    
    def rollout(
        self,
        client: Any,
        model: str,
        prompt: str,
        answer: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs
    ) -> Tuple[Union[str, List[Dict]], Dict]:
        """Execute a single rollout.
        
        Returns:
            completion: Model's response
            state: Rollout state dictionary
        """
    
    def generate(
        self,
        model: str,
        n_samples: int = 100,
        temperature: float = 0.7,
        system_prompt: str = None,
        **kwargs
    ) -> Tuple[List[str], List[Union[str, List]], List[Dict]]:
        """Generate multiple rollouts with rewards.
        
        Returns:
            prompts: List of prompts
            completions: List of completions
            rewards: List of reward dictionaries
        """
    
    async def generate_async(
        self,
        model: str,
        n_samples: int = 100,
        batch_size: int = 10,
        **kwargs
    ) -> Tuple[List[str], List[Union[str, List]], List[Dict]]:
        """Async batch generation for efficiency."""
```

### SingleTurnEnv

```python
class SingleTurnEnv(Environment):
    """Environment for single-turn interactions."""
    
    def format_prompt(
        self,
        prompt: str,
        task: str = None,
        **kwargs
    ) -> str:
        """Format prompt based on task type.
        
        Override this method for custom formatting.
        """
    
    def rollout(
        self,
        client: Any,
        model: str, 
        prompt: str,
        answer: str,
        max_retries: int = 3,
        **kwargs
    ) -> Tuple[Union[str, List[Dict]], Dict]:
        """Execute rollout with retry logic for parsing failures."""
```

### MultiTurnEnv

```python
class MultiTurnEnv(Environment):
    """Environment for multi-turn interactions."""
    
    def __init__(
        self,
        dataset: List[Dict[str, Any]],
        max_turns: int = 10,
        **kwargs
    ):
        """
        Additional Args:
            max_turns: Maximum conversation turns
        """
    
    def env_response(
        self,
        messages: List[Dict[str, str]],
        state: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate environment's response.
        
        Args:
            messages: Conversation history
            state: Current environment state
            
        Returns:
            response: Environment's response
            state: Updated state
            
        Must be overridden by subclasses.
        """
    
    def is_completed(
        self,
        messages: List[Dict[str, str]],
        state: Dict[str, Any]
    ) -> bool:
        """Check if interaction should end.
        
        Must be overridden by subclasses.
        """
    
    def score_rollout(
        self,
        messages: List[Dict[str, str]],
        answer: str,
        state: Dict[str, Any]
    ) -> Dict[str, float]:
        """Score a completed conversation.
        
        Override for custom scoring logic.
        """
```

### ToolEnv

```python
class ToolEnv(SingleTurnEnv):
    """Environment with tool support."""
    
    def __init__(
        self,
        dataset: List[Dict[str, Any]],
        tools: List[Callable],
        tool_format: str = "xml",  # or "json"
        **kwargs
    ):
        """
        Additional Args:
            tools: List of tool functions
            tool_format: Format for tool calls
        """
    
    def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> str:
        """Execute a tool with given arguments.
        
        Returns:
            Tool execution result as string
        """
```

## Parsers

### Base Parser

```python
class Parser:
    """Base parser class."""
    
    def parse(
        self,
        output: Union[str, List[Dict[str, str]]]
    ) -> Any:
        """Parse model output.
        
        Args:
            output: String or message list
            
        Returns:
            Parsed result (implementation-specific)
        """
    
    def parse_answer(
        self,
        output: Union[str, List[Dict[str, str]]]
    ) -> str:
        """Extract answer from output.
        
        Default: returns full output as string
        """
```

### XMLParser

```python
class XMLParser(Parser):
    """Parse XML-tagged fields from output."""
    
    def __init__(
        self,
        fields: List[Union[str, Tuple[str, ...]]],
        strip: bool = True
    ):
        """
        Args:
            fields: List of field names or tuples of alternatives
                   e.g., ["reasoning", ("code", "answer")]
            strip: Whether to strip whitespace from extracted content
        """
    
    def parse(
        self,
        output: Union[str, List[Dict[str, str]]]
    ) -> ParsedOutput:
        """Parse XML fields into object with attributes.
        
        Returns:
            Object with field names as attributes
        """
    
    def get_format_reward_func(
        self,
        per_field_weight: float = None
    ) -> Callable:
        """Get reward function for format compliance.
        
        Args:
            per_field_weight: Weight per field (default: 1/n_fields)
            
        Returns:
            Reward function returning [0, 1]
        """
```

## Rubrics

### Base Rubric

```python
class Rubric:
    """Base rubric for evaluation."""
    
    def __init__(
        self,
        funcs: List[Callable] = None,
        weights: List[float] = None,
        parser: Parser = None,
        **kwargs
    ):
        """
        Args:
            funcs: List of reward functions
            weights: Weights for each function (default: all 1.0)
            parser: Parser for structured output
            **kwargs: Additional attributes
        """
    
    def add_reward_func(
        self,
        func: Callable,
        weight: float = 1.0
    ):
        """Add a reward function to the rubric."""
    
    def score_rollout_sync(
        self,
        prompt: str,
        completion: Union[str, List[Dict]],
        answer: str,
        state: Dict[str, Any],
        task: str = None
    ) -> Dict[str, float]:
        """Score a single rollout synchronously.
        
        Returns:
            Dictionary with individual scores and 'reward' key
        """
    
    async def score_rollout(
        self,
        prompt: str,
        completion: Union[str, List[Dict]],
        answer: str,
        state: Dict[str, Any],
        task: str = None
    ) -> Dict[str, float]:
        """Score a single rollout asynchronously."""
    
    def score_rollouts(
        self,
        prompts: List[str],
        completions: List[Union[str, List[Dict]]],
        answers: List[str],
        states: List[Dict[str, Any]],
        tasks: List[str] = None
    ) -> Dict[str, List[float]]:
        """Score multiple rollouts in batch."""
```

### ToolRubric

```python
class ToolRubric(Rubric):
    """Rubric for evaluating tool usage."""
    
    def __init__(
        self,
        tools: List[Callable] = None,
        weights: List[float] = None,
        **kwargs
    ):
        """
        Args:
            tools: List of tool functions
            weights: Custom weights (default: auto-generated)
        
        Automatically includes:
        - correct_answer_reward_func
        - tool_execution_reward_func
        - format_reward_func
        - Per-tool reward functions
        """
    
    def evaluate_code(
        self,
        code: str,
        test_cases: str
    ) -> float:
        """Evaluate code against test cases.
        
        Args:
            code: Python code to evaluate
            test_cases: JSON string with test cases
            
        Returns:
            Proportion of tests passed [0, 1]
        """
```

### RubricGroup

```python
class RubricGroup(Rubric):
    """Combine multiple rubrics."""
    
    def __init__(
        self,
        rubrics: List[Rubric]
    ):
        """
        Args:
            rubrics: List of rubrics to combine
            
        Note: Functions with same name are summed
        """
```

## Reward Functions

### Common Signatures

```python
# Minimal signature
def reward_func(completion: Union[str, List[Dict]]) -> float:
    """Reward based only on completion."""
    
# Common signature
def reward_func(
    completion: Union[str, List[Dict]],
    answer: str,
    **kwargs
) -> float:
    """Reward based on completion and answer."""
    
# Full signature
def reward_func(
    prompt: str,
    completion: Union[str, List[Dict]],
    answer: str,
    state: Dict[str, Any],
    task: str,
    **kwargs
) -> float:
    """Reward with full context."""
```

### Built-in Reward Functions

```python
# From XMLParser
format_reward = parser.get_format_reward_func()

# From ToolRubric
correct_answer_reward = rubric.correct_answer_reward_func
tool_execution_reward = rubric.tool_execution_reward_func

# Per-tool rewards (automatically generated)
calculator_reward = rubric.calculator_reward_func
calculator_count = rubric.calculator_count_reward_func
calculator_attempt = rubric.calculator_attempt_reward_func
```

## Training

### GRPOConfig

```python
@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    
    # Training parameters
    learning_rate: float = 1e-5
    batch_size: int = 32
    group_size: int = 4
    num_epochs: int = 3
    gradient_accumulation_steps: int = 1
    
    # Generation parameters
    temperature: float = 0.7
    max_new_tokens: int = 512
    top_p: float = 0.9
    
    # GRPO specific
    kl_coef: float = 0.1
    gamma: float = 1.0
    gae_lambda: float = 0.95
    
    # Optimization
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    
    # Efficiency
    fp16: bool = False
    gradient_checkpointing: bool = False
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    output_dir: str = "./output"
```

### GRPOTrainer

```python
class GRPOTrainer:
    """Trainer for Group Relative Policy Optimization."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        env: Environment,
        config: GRPOConfig,
        dataset: Dict[str, List] = None,
        eval_dataset: Dict[str, List] = None
    ):
        """
        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
            env: Environment for generation/evaluation
            config: Training configuration
            dataset: Pre-generated training data (optional)
            eval_dataset: Evaluation data (optional)
        """
    
    def train(self):
        """Run training loop."""
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
    
    def save_model(self, output_dir: str):
        """Save trained model."""
```

## Tools

### Tool Function Format

```python
def tool_name(arg1: type1, arg2: type2 = default) -> str:
    """One-line description of the tool.
    
    Detailed description of what the tool does.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2 (optional)
        
    Returns:
        String result of tool execution
        
    Examples:
        tool_name("input1") -> "output1"
        tool_name("input2", arg2=value) -> "output2"
        
    Notes:
        - Additional usage notes
        - Error handling behavior
    """
    try:
        # Tool implementation
        result = process(arg1, arg2)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"
```

### Built-in Tools

```python
# Calculator
from verifiers.tools import calculator
result = calculator("2 + 2 * 3")  # "8"

# Python executor
from verifiers.tools import python
result = python("print('Hello'); x = 5; print(x * 2)")  # "Hello\n10"

# Search (example interface)
from verifiers.tools import search
results = search("quantum computing", max_results=3)
```

## Utilities

### Async Helpers

```python
from verifiers.utils import run_async_with_retry

result = await run_async_with_retry(
    async_func,
    max_retries=3,
    retry_delay=1.0,
    *args,
    **kwargs
)
```

### Dataset Loading

```python
from verifiers.utils import load_dataset_from_hub

dataset = load_dataset_from_hub(
    "username/dataset-name",
    split="train",
    streaming=False
)
```

### Tokenization

```python
from verifiers.utils import get_token_length

# Get token count for pricing estimates
n_tokens = get_token_length(
    text="Your text here",
    model="gpt-4"
)
```

## Type Definitions

```python
from typing import TypedDict, Union, List, Dict, Any, Callable, Optional

# Dataset entry
class DatasetEntry(TypedDict):
    prompt: str
    answer: str
    task: Optional[str]
    
# Message format
class Message(TypedDict):
    role: str  # "system", "user", "assistant"
    content: str
    
# Rollout state
State = Dict[str, Any]

# Completion format
Completion = Union[str, List[Message]]

# Reward function
RewardFunc = Callable[..., float]

# Tool function
ToolFunc = Callable[..., str]
```

This API reference provides comprehensive documentation for all major components of the verifiers framework. For detailed examples and usage patterns, refer to the other documentation sections.