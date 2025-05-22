from typing import Any

from verifiers.envs.singleturn_env import SingleTurnEnv

class ReasoningGymEnv(SingleTurnEnv):
    def __init__(self,
                 **kwargs: Any):
        super().__init__(**kwargs)
