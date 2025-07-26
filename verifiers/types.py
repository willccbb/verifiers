from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    NotRequired,
    Optional,
    TypedDict,
    Union,
)

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
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


class ChatMessage(TypedDict):
    role: str
    content: str
    tool_calls: NotRequired[List[ChatCompletionMessageToolCall]]
    tool_call_id: NotRequired[str]


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


GenerateOutputs = Dict[str, Any]


class ProcessedOutputs(BaseModel):
    """Pydantic model for processed outputs."""
    prompt_ids: List[int]
    prompt_mask: List[int]
    completion_ids: List[int]
    completion_mask: List[int]
    completion_logprobs: List[float]
    rewards: List[float]
