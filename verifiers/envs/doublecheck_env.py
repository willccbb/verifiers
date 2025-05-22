from typing import List, Dict, Any, Tuple

from datasets import Dataset
from openai import OpenAI
from verifiers import RewardFunc
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.prompts import SIMPLE_PROMPT
from verifiers.rubrics import MathRubric

class DoubleCheckEnv(MultiTurnEnv):
    def __init__(self,
                 client: OpenAI | None = None,
                 model: str | None = None,
                 dataset: Dataset | None = None,
                 eval_dataset: Dataset | None = None,
                 system_prompt: str = SIMPLE_PROMPT,
                 few_shot: List[Dict[str, str]] = [],
                 **kwargs):
        super().__init__(
            client=client,
            model=model,
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=system_prompt,
            few_shot=few_shot,
            **kwargs
        )
        self.rubric = MathRubric()

    def get_reward_funcs(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()
    
    def get_reward_weights(self, **kwargs: Any) -> List[float]:
        return self.rubric.get_reward_weights()

    def is_completed(self,
                     messages: List[Dict[str, str]],
                     state: Dict[str, Any],
                     **kwargs: Any) -> bool:
        return len(messages) > 1 and messages[-2]['content'] == 'Are you sure?'
    
    def env_response(self,
                     messages: List[Dict[str, str]],
                     state: Dict[str, Any],
                     **kwargs: Any) -> Tuple[Dict[str, str], Dict[str, Any]]:
        return {'role': 'user', 'content': 'Are you sure?'}, state