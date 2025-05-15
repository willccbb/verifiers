from typing import List, Dict, Any

from datasets import Dataset

from verifiers import RewardFunc
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.prompts import SIMPLE_PROMPT, DOUBLECHECK_FEW_SHOT
from verifiers.rubrics import MathRubric

class SingleTurnEnv(MultiTurnEnv):
    def __init__(self, 
                 dataset: Dataset | None = None,
                 system_prompt: str = SIMPLE_PROMPT,
                 few_shot: List[Dict[str, str]] = DOUBLECHECK_FEW_SHOT[0],
                 **kwargs):
        super().__init__(dataset=dataset, system_prompt=system_prompt, few_shot=few_shot, **kwargs)
        self.rubric = MathRubric()

    def get_reward_funcs(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()
    
    def get_reward_weights(self, **kwargs: Any) -> List[float]:
        return self.rubric.get_reward_weights()

    def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> bool:
        return True
    
    def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
        return {'role': 'user', 'content': 'ERROR'}