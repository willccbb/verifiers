from typing import TYPE_CHECKING, List, Optional

from verifiers.types import ChatMessage

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI


class OpenAISampler:
    """
    Sampler implementation for OpenAI API and compatible endpoints.

    This provides backwards compatibility with the existing OpenAI client usage.
    """

    def __init__(
        self,
        client: Optional["AsyncOpenAI | OpenAI"] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the OpenAI sampler.

        Args:
            client: Existing OpenAI client (for backwards compatibility)
            api_key: API key for OpenAI
            base_url: Base URL for API (supports other OpenAI-compatible endpoints)
            model: Default model to use
        """
        if client is not None:
            self.client = client
            # Convert sync client to async if needed
            if hasattr(client, "chat") and not hasattr(
                client.chat.completions, "acreate"
            ):
                from openai import AsyncOpenAI

                self.client = AsyncOpenAI(
                    api_key=client.api_key, base_url=client.base_url
                )
        else:
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        self.default_model = model

    async def sample(self, messages: List[ChatMessage], **config) -> ChatMessage:
        """
        Generate a response using the OpenAI API.

        Args:
            messages: List of conversation messages
            **config: OpenAI-specific parameters (model, temperature, max_tokens, etc.)

        Returns:
            ChatMessage with the assistant's response
        """
        # Use provided model or default
        model = config.pop("model", self.default_model or "gpt-4")
        if "max_tokens" in config:
            max_tokens = config.pop("max_tokens")
            if max_tokens is not None:
                config["max_completion_tokens"] = max_tokens

        # Remove None values
        config = {k: v for k, v in config.items() if v is not None}

        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
            **config,
        )

        msg = response.choices[0].message
        return {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": msg.tool_calls if hasattr(msg, "tool_calls") else None,
        }
