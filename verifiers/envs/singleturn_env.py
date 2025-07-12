from typing import List, Any, Tuple, Union

from verifiers import (
    ChatMessage,
    MessageType,
    State,
    MultiTurnEnv,
)

class SingleTurnEnv(MultiTurnEnv):
    """
    Environment for single-turn tasks (chat or completion).
    """
    def __init__(self,
                 message_type: MessageType = 'chat',
                 **kwargs):
        super().__init__(message_type=message_type, **kwargs)
        self.message_type = message_type

    def is_completed(self,
                     messages: Union[str, List[ChatMessage]],
                     state: State,
                     **kwargs: Any) -> bool:
        if len(state['responses']) > 0:
            return True
        return False

    def env_response(self,
                     messages: Union[str, List[ChatMessage]],
                     state: State,
                     **kwargs: Any) -> Tuple[Union[str, ChatMessage], State]:
        # never called in MultiTurnEnv.rollout
        return {'role': 'user', 'content': ""}, state