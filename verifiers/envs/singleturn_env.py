from typing import List, Dict, Any

from datasets import Dataset

from verifiers import RewardFunc
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.parsers import Parser
from verifiers.rubrics import Rubric

class SingleTurnEnv(MultiTurnEnv):
    def __init__(self, 
                 dataset: Dataset | None = None,
                 eval_dataset: Dataset | None = None,
                 system_prompt: str | None = None,
                 few_shot: List[Dict[str, str]] = [],
                 parser: Parser = Parser(),
                 rubric: Rubric = Rubric(),
                 **kwargs):
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=system_prompt,
            few_shot=few_shot,
            parser=parser,
            rubric=rubric,
            **kwargs
        )

    def get_reward_funcs(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()
    
    def get_reward_weights(self, **kwargs: Any) -> List[float]:
        return self.rubric.get_reward_weights()

    def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> bool:
        return True
    
    def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
        return {'role': 'user', 'content': 'ERROR'}