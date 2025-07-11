from abc import abstractmethod
from copy import deepcopy
from typing import List, Dict, Any, Tuple, Union

from openai import AsyncOpenAI

from verifiers.envs.environment import (
    Environment,
    ChatMessage,
    ChatCompletion,
    SamplingArgs,
    Info,
    State,
    MessageType,
)

class MultiTurnEnv(Environment):
    def __init__(self,
                 message_type: MessageType = 'chat',
                 max_turns: int = 10,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_turns = max_turns
        self.message_type = message_type

    @abstractmethod
    def is_completed(self,
                     messages: List[ChatMessage],
                     state: State,
                     **kwargs: Any) -> bool:
        pass

    @abstractmethod
    def env_response(self,
                     messages: List[ChatMessage],
                     state: State,
                     **kwargs: Any) -> Tuple[ChatMessage, State]:
        """
        Generate a response from the environment (message, state).
        """
        pass

    async def rollout(self,
                      client: AsyncOpenAI,
                      model: str,
                      prompt: Union[str, List[ChatMessage]],
                      answer: str,
                      task: str = "default",
                      info: Info = {},
                      sampling_args: SamplingArgs = {},
                      **kwargs: Any) -> Tuple[List[ChatMessage], State]:
        """
        Generate a multi-turn rollout with the environment (messages, state).
        """
        assert isinstance(prompt, list)
        messages = deepcopy(prompt) 
        is_completed = False
        state = {
            'prompt': prompt,
            'completion': [],
            'answer': answer,
            'task': task,
            'info': info,
            'responses': []
        }
        completion = []
        turn = 0
        while not is_completed:
            if self.is_completed(messages, state, **kwargs):
                is_completed = True
                break
            response = await self.get_model_response(
                prompt=messages,
                client=client,
                model=model,
                sampling_args=sampling_args,
                message_type=self.message_type
            )
            assert isinstance(response, ChatCompletion)
            response_text = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": response_text})
            completion.append({"role": "assistant", "content": response_text})
            state['responses'].append(response)
            turn += 1
            if self.is_completed(messages, state, **kwargs) or turn >= self.max_turns:
                is_completed = True
            else:
                env_msg, state = self.env_response(messages, state, **kwargs)
                messages.append(env_msg)
                completion.append(env_msg)
        return completion, state