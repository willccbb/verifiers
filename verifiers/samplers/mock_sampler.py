"""
Mock sampler implementation for testing purposes.
"""

from typing import List, Optional

from verifiers.types import ChatMessage
from .sampler import Sampler


class MockSampler(Sampler):
    """
    Mock sampler that returns predetermined responses.
    Useful for testing and development.
    """

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        default_response: str = "Mock response",
    ):
        """
        Initialize the mock sampler.

        Args:
            responses: List of responses to cycle through
            default_response: Default response if responses list is empty
        """
        self.responses = responses or []
        self.default_response = default_response
        self.call_count = 0
        self.call_history: List[List[ChatMessage]] = []

    async def sample(self, messages: List[ChatMessage], **config) -> ChatMessage:
        """
        Return a mock response.

        Args:
            messages: Input messages (stored for inspection)
            **config: Ignored in mock implementation

        Returns:
            Mock assistant message
        """
        self.call_history.append(messages)

        if self.responses:
            response = self.responses[self.call_count % len(self.responses)]
        else:
            response = self.default_response

        self.call_count += 1

        return {"role": "assistant", "content": response, "tool_calls": None}

    def reset(self):
        """Reset the call count and history."""
        self.call_count = 0
        self.call_history = []
