from abc import abstractmethod
from copy import deepcopy
from typing import List, Dict, Any, Tuple

from openai import AsyncOpenAI

from verifiers.envs.environment import (
    Environment,
    ChatMessage,
    Completion,
)


class MultiTurnCompletionEnv(Environment):
    def __init__(self, max_turns: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.max_turns = max_turns

    @abstractmethod
    def is_completed(self,
                     prompt: str,
                     state: Dict[str, Any],
                     **kwargs: Any) -> bool:
        pass

    @abstractmethod
    def env_response(self,
                     prompt: str,
                     state: Dict[str, Any],
                     **kwargs: Any) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a response from the environment (message, state).
        """
        pass

    async def rollout(self,
                      client: AsyncOpenAI,
                      model: str,
                      prompt: str | List[ChatMessage],
                      answer: str,
                      task: str = "default",
                      info: Dict[str, Any] = {},
                      sampling_args: Dict[str, Any] = {},
                      **kwargs: Any) -> Tuple[str, Dict[str, Any]]:
        assert isinstance(prompt, str)
        is_completed = False
        state = {
            'responses': [],
            'prompt': prompt,
            'answer': answer,
            'info': info,
            'task': task,
        }
        assert isinstance(prompt, str)
        input = deepcopy(prompt) 
        completion = ""
        turn = 0
        while not is_completed:
            response = await self.get_model_response(
                prompt=input,
                client=client,
                model=model,
                sampling_args=sampling_args,
                message_type=self.message_type
            )
            assert isinstance(response, Completion)
            response_text = response.choices[0].text or ""
            state['responses'].append(response)
            input = input + response_text
            completion += response_text
            turn += 1
            if self.is_completed(input, state, **kwargs) or turn >= self.max_turns:
                is_completed = True
            else:
                env_msg, state = self.env_response(input, state, **kwargs)
                input = input + env_msg
                completion += env_msg
        return completion, state