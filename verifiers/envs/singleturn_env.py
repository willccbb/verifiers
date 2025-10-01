from openai import AsyncOpenAI
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.types import Messages, MessageType, ModelResponse, SamplingArgs, State


class SingleTurnEnv(MultiTurnEnv):
    """
    Environment for single-turn tasks (chat or completion).
    """

    async def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        return len(state["responses"]) > 0

    async def get_model_response(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt: Messages,
        oai_tools: list[ChatCompletionToolParam] | None = None,
        sampling_args: SamplingArgs | None = None,
        message_type: MessageType | None = None,
        **kwargs,
    ) -> ModelResponse:
        try:
            return await self._get_model_response(
                client, model, prompt, oai_tools, sampling_args, message_type, **kwargs
            )
        except Exception as e:
            self.logger.error(f"Error getting model response: {e} \n\nExiting...")
            raise e

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> tuple[Messages, State]:
        # never called in MultiTurnEnv.rollout
        return [{"role": "user", "content": ""}], state
