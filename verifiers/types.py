from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam as ChatMessage,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,  # noqa: F401
)
from openai.types.chat.chat_completion_role import ChatCompletionRole  # noqa: F401
from openai.types.chat.chat_completion_tool_param import (
    ChatCompletionToolParam,  # noqa: F401
)
from openai.types.completion import Completion
from openai.types.shared_params import (  # noqa: F401
    FunctionDefinition,
    FunctionParameters,
)
from pydantic import BaseModel

# typing aliases
MessageType = Literal["chat", "completion"]
ModelResponse = Union[Completion, ChatCompletion, None]


Message = Union[str, ChatMessage]
Messages = Union[str, List[ChatMessage]]
Info = Dict[str, Any]
State = Dict[str, Any]
SamplingArgs = Dict[str, Any]
RewardFunc = Callable[..., float]

# oai tools
JsonPrimitive = Literal["string", "number", "integer", "boolean", "array", "object"]


class GenerateInputs(BaseModel):
    """Pydantic model for generation inputs."""

    prompt: List[Messages]
    answer: Optional[List[str]] = None
    info: Optional[List[Dict]] = None
    task: Optional[List[str]] = None
    completion: Optional[List[Messages]] = None


class GenerateOutputs(BaseModel):
    """Pydantic model for generation outputs."""

    prompt: List[Messages]
    completion: List[Messages]
    answer: List[str]
    state: List[State]
    info: List[Info]
    task: List[str]
    reward: List[float]
    metrics: Dict[str, List[float]] = {}


class RolloutScore(BaseModel):
    """Pydantic model for rollout scores."""

    reward: float
    metrics: Dict[str, float] = {}


class RolloutScores(BaseModel):
    """Pydantic model for rubric outputs."""

    reward: List[float]
    metrics: Dict[str, List[float]] = {}


class ProcessedOutputs(BaseModel):
    """Pydantic model for processed outputs."""

    prompt_ids: List[List[int]]
    prompt_mask: List[List[int]]
    completion_ids: List[List[int]]
    completion_mask: List[List[int]]
    completion_logprobs: List[List[float]]
    rewards: List[float]
