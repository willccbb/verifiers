from typing import Protocol, List

from verifiers.types import ChatMessage


class Sampler(Protocol):
    """
    Universal interface for any LLM provider.

    All samplers must implement the sample method which takes messages
    and returns a single message response.
    """

    async def sample(self, messages: List[ChatMessage], **config) -> ChatMessage:
        """
        Generate a response message from the given conversation.

        Args:
            messages: List of conversation messages
            **config: Provider-specific configuration (model, temperature, etc.)

        Returns:
            ChatMessage with role='assistant', content, and optional tool_calls
        """
        ...
