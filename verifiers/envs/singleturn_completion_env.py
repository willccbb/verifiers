from typing import Any, Tuple

from verifiers.envs.environment import (
    MessageType,
    State,
)
from verifiers.envs.multiturn_completion_env import MultiTurnCompletionEnv

class SingleTurnCompletionEnv(MultiTurnCompletionEnv):
    """
    Environment for single-turn tasks (chat or completion).
    """
    def __init__(self,
                 message_type: MessageType = 'completion',
                 **kwargs):
        super().__init__(message_type=message_type, **kwargs)
        self.message_type = message_type

    def is_completed(self,
                     prompt: str,
                     state: State,
                     **kwargs: Any) -> bool:
        if len(state['responses']) > 0:
            return True
        return False
    
    def env_response(self, prompt: str, state: State, **kwargs: Any) -> Tuple[str, State]:
        # never called in MultiTurnCompletionEnv.rollout
        return "", state 