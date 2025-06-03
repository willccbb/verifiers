from abc import abstractmethod
from copy import deepcopy
from typing import List, Dict, Any, Tuple, Union

from openai import OpenAI

from verifiers.envs.environment import Environment


class MultiTurnEnv(Environment):
    def __init__(self, max_turns: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.max_turns = max_turns

    @abstractmethod
    def is_completed(self,
                     messages: List[Dict[str, Any]],
                     state: Dict[str, Any],
                     **kwargs: Any) -> bool:
        pass

    @abstractmethod
    def env_response(self,
                     messages: List[Dict[str, Any]],
                     state: Dict[str, Any],
                     **kwargs: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generate a response from the environment (message, state).
        """
        pass

    def rollout(self,
                client: OpenAI,
                model: str,
                prompt: Union[str, List[Dict[str, Any]]],
                answer: str,
                sampling_args: Dict[str, Any] = {},
                **kwargs: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate a multi-turn rollout with the environment (messages, state).
        """
        is_completed = False
        state = {'answer': answer}
        assert isinstance(prompt, list)
        messages = deepcopy(prompt) 
        completion = []
        turn = 0
        while not is_completed:
            if self.is_completed(messages, state, **kwargs):
                is_completed = True
                break
            response = self.get_model_response(
                prompt=messages,
                client=client,
                model=model,
                sampling_args=sampling_args,
                message_type=self.message_type
            )
            has_error = response.startswith("[ERROR]")
            messages.append({"role": "assistant", "content": response})
            completion.append({"role": "assistant", "content": response})
            turn += 1
            if self.is_completed(messages, state, **kwargs) or turn >= self.max_turns or has_error:
                is_completed = True
            else:
                env_msg, state = self.env_response(messages, state, **kwargs)
                messages.append(env_msg)
                completion.append(env_msg)
        return completion, state