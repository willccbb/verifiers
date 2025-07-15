# API Reference

Complete API documentation for the verifiers framework.

## Environments

### Base Environment

```python
class Environment:
    """Base class for all environments."""
    
    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
        dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        system_prompt: str | None = None,
        few_shot: List[ChatMessage] = [],
        parser: Parser = Parser(),
        rubric: Rubric = Rubric(),
        sampling_args: SamplingArgs = {},
        message_type: MessageType = "chat",
        max_workers: int = 512,
        **kwargs
    ):
        """
        Args:
            client: OpenAI-compatible client
            model: Model name for generation
            dataset: Training dataset with 'question' and 'answer' columns
            eval_dataset: Evaluation dataset (optional)
            system_prompt: System message for the model
            few_shot: Few-shot examples as chat messages
            parser: Parser for extracting structured output
            rubric: Rubric for evaluation
            sampling_args: Generation parameters
            message_type: "chat" or "completion"
            max_workers: Maximum concurrent workers
            **kwargs: Additional attributes stored on the environment
        """
    
    async def rollout(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt: Messages,
        answer: str = "",
        task: str = "default",
        info: Info = {},
        sampling_args: SamplingArgs = {},
        **kwargs
    ) -> Tuple[Messages, State]:
        """Execute a single rollout.
        
        Returns:
            completion: Model's response
            state: Rollout state dictionary
        """
    
    def generate(
        self,
        inputs: GenerateInputs | Dataset,
        client: AsyncOpenAI | OpenAI,
        model: str | None = None,
        sampling_args: SamplingArgs = {},
        score_rollouts: bool = True,
        max_concurrent: int = -1,
        **kwargs
    ) -> GenerateOutputs:
        """Generate multiple rollouts with rewards.
        
        Returns:
            Dictionary with prompts, completions, states, and rewards
        """
    
    def evaluate(
        self,
        client: AsyncOpenAI | OpenAI,
        model: str,
        sampling_args: SamplingArgs = {},
        num_examples: int = -1,
        rollouts_per_example: int = 1,
        score_rollouts: bool = True,
        max_concurrent: int = -1,
        **kwargs
    ) -> GenerateOutputs:
        """Evaluate model performance on the environment."""
```

### MultiTurnEnv

```python
class MultiTurnEnv(Environment):
    """Base class for multi-turn interactive environments."""
    
    def __init__(
        self,
        message_type: MessageType = 'chat',
        max_turns: int = 10,
        **kwargs
    ):
        """
        Args:
            message_type: "chat" or "completion"
            max_turns: Maximum conversation turns
            **kwargs: Additional arguments passed to Environment
        """
    
    @abstractmethod
    def is_completed(
        self,
        messages: Messages,
        state: State,
        **kwargs
    ) -> bool:
        """Check if interaction should end.
        
        Must be overridden by subclasses.
        """
    
    @abstractmethod
    def env_response(
        self,
        messages: Messages,
        state: State,
        **kwargs
    ) -> Tuple[Message, State]:
        """Generate environment's response.
        
        Args:
            messages: Conversation history
            state: Current environment state
            
        Returns:
            response: Environment's response
            state: Updated state
            
        Must be overridden by subclasses.
        """
    
    async def rollout(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt: Messages,
        answer: str = "",
        task: str = "default",
        info: Info = {},
        sampling_args: SamplingArgs = {},
        **kwargs
    ) -> Tuple[Messages, State]:
        """Generate a multi-turn rollout with the environment."""
```

### SingleTurnEnv

```python
class SingleTurnEnv(MultiTurnEnv):
    """Environment for single-turn interactions (most common entry point)."""
    
    def __init__(
        self,
        message_type: MessageType = 'chat',
        **kwargs
    ):
        """
        Args:
            message_type: "chat" or "completion"
            **kwargs: Additional arguments passed to MultiTurnEnv
        """
    
    def is_completed(
        self,
        messages: Messages,
        state: State,
        **kwargs
    ) -> bool:
        """Single-turn environments complete after one response."""
        if len(state['responses']) > 0:
            return True
        return False
    
    def env_response(
        self,
        messages: Messages,
        state: State,
        **kwargs
    ) -> Tuple[Message, State]:
        """Never called in SingleTurnEnv - single turn only."""
        return {'role': 'user', 'content': ""}, state
```

### ToolEnv

```python
class ToolEnv(SingleTurnEnv):
    """Environment with tool support."""
    
    def __init__(
        self,
        tools: List[Callable],
        max_steps: int = 3,
        **kwargs
    ):
        """
        Args:
            tools: List of tool functions
            max_steps: Maximum tool execution steps
            **kwargs: Additional arguments passed to SingleTurnEnv
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
    """Base parser class for extracting structured information from model outputs."""
    
    def __init__(self, **kwargs):
        """Initialize parser with additional attributes."""
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def parse(self, text: str) -> Any:
        """Parse text and return structured data. Default: return text as-is."""
        return text
    
    def parse_answer(self, completion: Messages) -> str | None:
        """Extract the final answer from a completion."""
        if isinstance(completion, str):
            return self.parse(completion)
        else:
            return self.parse(completion[-1]["content"])
    
    def get_format_reward_func(self) -> Callable:
        """Return a reward function that checks format compliance."""
        def format_reward_func(completion: List[Dict[str, str]], **kwargs) -> float:
            return 1.0  # Default: always return 1.0
        return format_reward_func
    
    def get_assistant_messages(self, completion: List[ChatMessage]) -> List[ChatMessage]:
        """Helper function to extract assistant messages from a completion."""
        return [msg for msg in completion if msg['role'] == 'assistant']
```

### XMLParser

```python
class XMLParser(Parser):
    """Convenience parser for extracting structured fields from XML-tagged output."""
    
    def __init__(
        self,
        fields: List[Union[str, Tuple[str, ...]]],
        answer_field: str = "answer"
    ):
        """
        Args:
            fields: List of field names or tuples of alternative names
            answer_field: Field name to use for answer extraction
        """
    
    def parse(self, text: str, strip: bool = True) -> Any:
        """Parse XML and return object with field attributes."""
    
    def parse_answer(self, completion: Messages) -> str | None:
        """Extract the last answer from a completion."""
    
    def get_format_str(self) -> str:
        """Return a string describing the expected format."""
    
    def get_format_reward_func(self) -> Callable:
        """Return a reward function that checks XML format compliance."""
    
    def format(self, **kwargs) -> str:
        """Format keyword arguments into XML string."""
```

### ThinkParser

```python
class ThinkParser(Parser):
    """Convenience parser for extracting content after </think> tags."""
    
    def __init__(
        self,
        extract_fn: Callable[[str], str] = lambda x: x,
        **kwargs
    ):
        """
        Args:
            extract_fn: Function to extract final answer from text after </think>
            **kwargs: Additional arguments passed to Parser
        """
    
    def parse(self, text: str) -> str:
        """Extract content after </think> tags and apply extract_fn."""
    
    def get_format_reward_func(self) -> Callable:
        """Return a reward function that checks <think> tag format."""
```

## Rubrics

### Base Rubric

```python
class Rubric:
    """Base rubric class for combining multiple reward functions."""
    
    def __init__(
        self,
        funcs: List[RewardFunc] = [],
        weights: List[float] = [],
        parser: Parser = Parser(),
        **kwargs
    ):
        """
        Args:
            funcs: List of reward functions
            weights: Weights for each reward function
            parser: Parser for extracting structured output
            **kwargs: Additional attributes stored on the rubric
        """
    
    def get_reward_func_names(self) -> List[str]:
        """Return names of all reward functions."""
    
    def get_reward_funcs(self) -> List[RewardFunc]:
        """Return list of reward functions."""
    
    def get_reward_weights(self) -> List[float]:
        """Return list of reward weights."""
    
    def add_reward_func(self, func: RewardFunc, weight: float = 1.0):
        """Add a new reward function with weight."""
    
    async def call_reward_func(
        self,
        func: RewardFunc,
        prompt: Union[str, List[ChatMessage]],
        completion: Union[str, List[ChatMessage]],
        answer: str,
        state: State,
        task: str = "default",
        info: Info = {},
        **kwargs
    ) -> float:
        """Invoke a single reward function with appropriate arguments."""
    
    async def score_rollout(
        self,
        prompt: Union[str, List[ChatMessage]],
        completion: Union[str, List[ChatMessage]],
        answer: str,
        state: State,
        task: str = "default",
        info: Info = {},
        **kwargs
    ) -> Dict[str, float]:
        """Evaluate all reward functions for a single rollout."""
    
    async def score_rollouts(
        self,
        prompts: List[Union[str, List[ChatMessage]]],
        completions: List[Union[str, List[ChatMessage]]],
        answers: List[str],
        states: List[State],
        tasks: List[str],
        infos: List[Info] = [],
        **kwargs
    ) -> Dict[str, List[float]]:
        """Compute reward scores for a group of rollouts."""
```

### RubricGroup

```python
class RubricGroup(Rubric):
    """Class for aggregating multiple rubrics."""
    
    def __init__(self, rubrics: List[Rubric], **kwargs):
        """
        Args:
            rubrics: List of rubrics to combine
            **kwargs: Additional arguments passed to Rubric
        """
    
    def get_reward_func_names(self) -> List[str]:
        """Return names from all rubrics."""
    
    def get_reward_funcs(self) -> List[RewardFunc]:
        """Return functions from all rubrics."""
    
    def get_reward_weights(self) -> List[float]:
        """Return weights from all rubrics."""
    
    async def score_rollouts(
        self,
        prompts: List[Messages],
        completions: List[Messages],
        answers: List[str],
        states: List[State],
        tasks: List[str],
        infos: List[Info] = [],
        **kwargs
    ) -> Dict[str, List[float]]:
        """Run all rubrics sequentially and return aggregated scores."""
```

### ToolRubric

```python
class ToolRubric(Rubric):
    """Rubric for evaluating tool usage and execution."""
    
    def __init__(
        self,
        tools: List[Callable],
        weights: List[float] = [0.5, 0.3, 0.2],
        **kwargs
    ):
        """
        Args:
            tools: List of available tools
            weights: Weights for [correct_answer, tool_execution, format]
            **kwargs: Additional arguments passed to Rubric
        """
```

### JudgeRubric

```python
class JudgeRubric(Rubric):
    """Rubric using LLM-based evaluation."""
    
    def __init__(
        self,
        judge_models: List[str],
        client: AsyncOpenAI,
        template: str,
        parser: Parser = None,
        **kwargs
    ):
        """
        Args:
            judge_models: List of model names for judging
            client: OpenAI client for judge models
            template: Prompt template for evaluation
            parser: Parser for judge responses
            **kwargs: Additional arguments passed to Rubric
        """
```

## Type Definitions

```python
# Core types
Messages = Union[str, List[ChatMessage]]
MessageType = Literal["chat", "completion"]
State = Dict[str, Any]
Info = Dict[str, Any]
SamplingArgs = Dict[str, Any]

# Function types
RewardFunc = Callable[..., Union[float, List[float], Dict[str, float]]]

# Input/Output types
GenerateInputs = List[Dict[str, Any]]
GenerateOutputs = Dict[str, Any]
ProcessedOutputs = Dict[str, Any]

# Message types
ChatMessage = Dict[str, str]
Message = Union[str, ChatMessage]
```

## Common Patterns

### Creating a Custom Environment

```python
class MyCustomEnv(vf.MultiTurnEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add custom initialization
    
    def is_completed(self, messages, state, **kwargs):
        """Define completion criteria."""
        return some_condition
    
    def env_response(self, messages, state, **kwargs):
        """Define environment responses."""
        return response_message, updated_state
```

### Creating a Custom Parser

```python
class MyCustomParser(vf.Parser):
    def parse(self, text: str) -> Any:
        """Parse text according to custom logic."""
        # Your parsing logic here
        return parsed_data
    
    def get_format_reward_func(self):
        """Return format compliance reward function."""
        def format_reward_func(completion, **kwargs):
            # Your format checking logic here
            return format_score
        return format_reward_func
```

### Creating a Custom Rubric

```python
class MyCustomRubric(vf.Rubric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def custom_reward(self, completion, answer, **kwargs):
        """Custom reward function."""
        # Your evaluation logic here
        return reward_score
```