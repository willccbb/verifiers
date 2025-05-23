from typing import Any, Tuple

from datasets import Dataset
from openai import OpenAI
import reasoning_gym as rg

from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric
from verifiers.envs.singleturn_env import SingleTurnEnv

class ReasoningGymEnv(SingleTurnEnv):
    def __init__(self,
                 client: OpenAI,
                 model: str,
                 gym: str,
                 num_samples: int = 1000,
                 num_eval_samples: int = 100,   
                 seed: int = 0,
                 **kwargs: Any):
        self.gym = gym
        self.num_samples = num_samples
        self.num_eval_samples = num_eval_samples
        self.seed = seed
        total_samples = num_samples + num_eval_samples
        self.rg_dataset = rg.create_dataset(gym, size=total_samples, seed=seed)
        dataset, eval_dataset = self.rg_to_hf(self.rg_dataset)
        # <think>...</think><answer>...</answer>
        system_prompt = rg.utils.SYSTEM_PROMPTS["DeepSeekZero"] # type: ignore
        super().__init__(
            client=client,
            model=model,
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=system_prompt,
            message_type='chat',
            **kwargs
        )
        self.parser = XMLParser(fields=['think', 'answer'])
        self.rubric = Rubric(parser=self.parser)
        def check_answer_reward_func(prompt, completion, answer, **kwargs) -> float:
            entry = self.rg_dataset[answer]
            response = str(self.parser.parse_answer(completion)).strip()
            reward = self.rg_dataset.score_answer(answer=response, entry=entry)
            return reward
        self.rubric.add_reward_func(check_answer_reward_func)

    def rg_to_hf(self, rg_dataset) -> Tuple[Dataset, Dataset]:
        dataset_rows = []
        eval_rows = []
        for i, x in enumerate(rg_dataset):
            row = {
                'question': x['question'],
                'answer': i,
                'task': x['metadata']['source_dataset'],
            }
            if i < self.num_samples:
                dataset_rows.append(row)
            else:
                eval_rows.append(row)
        dataset = Dataset.from_list(dataset_rows)
        eval_dataset = Dataset.from_list(eval_rows)
        return dataset, eval_dataset