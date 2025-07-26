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


class ChatMessage(BaseModel):
    """Pydantic model for chat messages.
    
    Provides dict-like access for backward compatibility with TypedDict usage.
    """
    role: str
    content: str
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None
    tool_call_id: Optional[str] = None
    
    # Configuration for Pydantic model
    model_config = {
        "extra": "allow",  # Allow extra fields for flexibility
    }
    
    # Allow dict-like access for backward compatibility
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        """Create ChatMessage from dictionary for easy migration."""
        return cls(**data)


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
    
    model_config = {
        "extra": "allow",
    }


GenerateOutputs = Dict[str, Any]


class ProcessedOutputs(BaseModel):
    """Pydantic model for processed outputs."""
    prompt_ids: List[int]
    prompt_mask: List[int]
    completion_ids: List[int]
    completion_mask: List[int]
    completion_logprobs: List[float]
    rewards: List[float]
    
    model_config = {
        "extra": "allow",
    }
