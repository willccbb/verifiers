from abc import abstractmethod

from openai import AsyncOpenAI

from verifiers.envs.environment import Environment
from verifiers.samplers import Sampler
from verifiers.types import (
    Info,
    Messages,
    SamplingArgs,
    State,
)
from verifiers.utils.async_utils import maybe_await


class MultiTurnEnv(Environment):
    def __init__(self, max_turns: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.max_turns = max_turns

    async def setup_state(self, state: State, **kwargs) -> State:
        return state

    @abstractmethod
    async def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        pass

    @abstractmethod
    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> tuple[Messages, State]:
        """
        Generate a response from the environment (messages, state).
        """
        pass

    async def rollout(
        self,
        client: AsyncOpenAI | None = None,  # Optional: for backwards compatibility
        model: str | None = None,  # Optional: for backwards compatibility
        prompt: Messages | None = None,
        answer: str = "",
        task: str = "default",
        info: Info | None = None,
        sampling_args: SamplingArgs | None = None,
        sampler: Sampler | None = None,  # Use provided sampler or self.sampler
        **kwargs,
    ) -> tuple[Messages, State]:
        """
        Generate a multi-turn rollout with the environment (messages, state).
        """
        if sampler is None and (client is not None or model is not None):
            from verifiers.samplers import OpenAISampler

            sampler = OpenAISampler(client=client, model=model)

        info = info or {}
        is_completed = False
        state = {
            "prompt": prompt,
            "completion": [],
            "answer": answer,
            "task": task,
            "info": info,
            "responses": [],
            "turn": 0,
        }
        state = await maybe_await(self.setup_state, state, **kwargs)
        if self.message_type == "chat":
            assert isinstance(prompt, list)
            completion = []
        else:
            assert isinstance(prompt, str)
            completion = ""
            state["responses_start_idx"] = []
        rollout = list(prompt) if not isinstance(prompt, str) else prompt
        while not is_completed:
            if await maybe_await(self.is_completed, rollout, state, **kwargs):
                is_completed = True
                break

            active_sampler = sampler or self.sampler
            sample_config = dict(sampling_args or {})
            if info.get("oai_tools"):
                sample_config["tools"] = info.get("oai_tools")

            if self.message_type == "chat":
                assert isinstance(rollout, list)
                assert isinstance(completion, list)
                response_message = await active_sampler.sample(rollout, **sample_config)
                state["responses"].append(response_message)
                rollout.append(response_message)
                completion.append(response_message)
            else:
                assert isinstance(rollout, str)
                assert isinstance(completion, str)
                chat_messages = [{"role": "user", "content": rollout}]
                response_message = await active_sampler.sample(
                    chat_messages, **sample_config
                )
                response_text = response_message["content"]
                state["responses"].append(response_message)
                state["responses_start_idx"].append(len(completion))
                rollout += response_text
                completion += response_text
            state["turn"] += 1
            if (
                await maybe_await(self.is_completed, rollout, state, **kwargs)
                or state["turn"] >= self.max_turns
            ):
                is_completed = True
            else:
                env_msgs, state = await maybe_await(
                    self.env_response, rollout, state, **kwargs
                )
                if self.message_type == "chat":
                    assert isinstance(env_msgs, list)
                    assert isinstance(rollout, list)
                    assert isinstance(completion, list)
                    rollout += env_msgs
                    completion += env_msgs
                else:
                    assert isinstance(env_msgs, str)
                    assert isinstance(rollout, str)
                    assert isinstance(completion, str)
                    rollout += env_msgs
                    completion += env_msgs
        return completion, state
