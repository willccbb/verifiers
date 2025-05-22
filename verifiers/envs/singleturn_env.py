from typing import List, Dict, Any, Literal, Tuple

from datasets import Dataset
from openai import OpenAI

from verifiers.envs.environment import Environment

class SingleTurnEnv(Environment):
    """
    Environment for single-turn tasks (chat or completion).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def rollout(self,
                client: OpenAI,
                model: str,
                prompt: str | List[Dict[str, str]],
                sampling_args: Dict[str, Any] = {},
                **kwargs: Any) -> Tuple[str, Dict[str, Any]] | Tuple[List[Dict[str, str]], Dict[str, Any]]:
        completion = self.get_model_response(
            client=client,
            model=model,
            prompt=prompt,
            sampling_args=sampling_args,
            message_type=self.message_type
        )
        if self.message_type == 'chat': 
            return [{'role': 'assistant', 'content': completion}], {}
        return completion, {}