from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict, Union

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion

# typing aliases
MessageType = Literal["chat", "completion"]
ModelResponse = Union[Completion, ChatCompletion, None]
ChatMessageField = Literal["role", "content"]
ChatMessage = Dict[ChatMessageField, str]
Message = Union[str, ChatMessage]
Messages = Union[str, List[ChatMessage]]
Info = Dict[str, Any]
State = Dict[str, Any]
SamplingArgs = Dict[str, Any]
RewardFunc = Callable[..., float]


class GenerateInputs(TypedDict):
    prompt: List[Messages]
    answer: Optional[List[str]]
    info: Optional[List[Dict]]
    task: Optional[List[str]]
    completion: Optional[List[Messages]]


GenerateOutputs = Dict[str, Any]


class ProcessedOutputs(TypedDict):
    prompt_ids: List[int]
    prompt_mask: List[int]
    completion_ids: List[int]
    completion_mask: List[int]
    completion_logprobs: List[float]
    rewards: List[float]
