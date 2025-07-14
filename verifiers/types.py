from typing import Callable, Optional, Literal, Union, Dict, Any, List, TypedDict
import logging
import sys

from openai.types.completion import Completion
from openai.types.chat.chat_completion import ChatCompletion

# typing aliases
MessageType = Literal['chat', 'completion']
ModelResponse = Union[Completion, ChatCompletion, None]
ChatMessageField = Literal['role', 'content']
ChatMessage = Dict[ChatMessageField, str]
Message = Union[str, ChatMessage]
Messages = Union[str, List[ChatMessage]]
Info = Dict[str, Any]
State = Dict[str, Any]
SamplingArgs = Dict[str, Any]
RewardFunc = Callable[..., float]

from dataclasses import dataclass, field
from typing import List, Dict, Optional

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
    rewards: List[float]