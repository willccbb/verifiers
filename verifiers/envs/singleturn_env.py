from typing import List, Dict, Any, Literal, Tuple, Union

from openai import OpenAI

from verifiers.envs.environment import Environment


class SingleTurnEnv(Environment):
    """
    Environment for single-turn tasks (chat or completion).
    """
    def __init__(self,
                 message_type: Literal['chat', 'completion'] = 'chat',
                 **kwargs):
        super().__init__(message_type=message_type, **kwargs)
        self.message_type = message_type

    def rollout(self,
                client: OpenAI,
                model: str,
                prompt: Union[str, List[Dict[str, Any]]],
                answer: str,
                sampling_args: Dict[str, Any] = {},
                **kwargs: Any) -> Tuple[Union[str, List[Dict[str, str]]], Dict[str, Any]]:
        """
        Returns completion (str or message list) and null state.
        """
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
    
