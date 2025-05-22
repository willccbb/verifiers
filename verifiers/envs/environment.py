import asyncio
from asyncio import Semaphore
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Literal
import logging

from datasets import Dataset
from openai import OpenAI

from verifiers import RewardFunc
from verifiers.parsers import Parser
from verifiers.rubrics import Rubric

class Environment(ABC):
    """
    Base class for all environments.
    """
    def __init__(self,
                 client: OpenAI | None = None,
                 model: str | None = None,
                 dataset: Dataset | None = None,
                 eval_dataset: Dataset | None = None,
                 system_prompt: str | None = None,
                 few_shot: List[Dict[str, str]] = [],
                 parser: Parser = Parser(),
                 rubric: Rubric = Rubric(),
                 sampling_args: Dict[str, Any] = {},
                 max_concurrent: int = 32,
                 message_type: Literal['chat', 'completion'] = 'chat',
                 **kwargs: Any):
        self.client = client
        self.model = model
        self.message_type: Literal['chat', 'completion'] = message_type
        self.system_prompt = system_prompt
        self.few_shot = few_shot
        self.max_concurrent = max_concurrent
        if self.message_type == 'chat':
            if dataset is not None:
                self.dataset = self.format_dataset(dataset, self.system_prompt, self.few_shot)
            else:
                self.dataset = None 
            if eval_dataset is not None:
                self.eval_dataset = self.format_dataset(eval_dataset, self.system_prompt, self.few_shot)
            else:
                self.eval_dataset = None
        else:
            if self.system_prompt or self.few_shot:
                raise ValueError(
                    'The fields "system_prompt" and "few_shot" are not supported for completion tasks.' \
                    'Please use message_type="chat" instead, or pre-format your dataset ' \
                    'to contain "prompt" and "answer" columns.'
                )
            self.dataset = dataset
            self.eval_dataset = eval_dataset
        self.parser = parser
        self.rubric = rubric
        self.sampling_args = {
            'n': 1, # n > 1 not supported; duplicate prompts for multiple completions
            'extra_body': {
                'skip_special_tokens': False,
                'spaces_between_special_tokens': False,
            },
        }
        if sampling_args is not None and 'extra_body' in sampling_args:
            self.sampling_args['extra_body'].update(sampling_args['extra_body'])
        for k, v in sampling_args.items():
            if k != 'extra_body':
                self.sampling_args[k] = v
        self.logger = logging.getLogger(f'verifiers.envs.{self.__class__.__name__}')
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        if self.dataset is None and self.eval_dataset is None:
            raise ValueError('Either dataset or eval_dataset must be provided')
    
    def format_prompt(self,
                      prompt: str,
                      system_prompt: str | None = None,
                      few_shot: List[Dict[str, str]] | None = None
                      ) -> List[Dict[str, str]]:
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        if few_shot:
            messages.extend(few_shot)
        messages.append({'role': 'user', 'content': prompt})
        return messages

    def format_dataset(self,
                       dataset: Dataset,
                       system_prompt: str | None = None,
                       few_shot: List[Dict[str, str]] | None = None,
                       question_key: str = "question",
                       answer_key: str = "answer") -> Dataset:
        if answer_key == "answer":
            return dataset.map(lambda x: {
                "prompt": self.format_prompt(x[question_key], system_prompt, few_shot),
            }, num_proc=self.max_concurrent)
        else:
            return dataset.map(lambda x: {
                "prompt": self.format_prompt(x[question_key], system_prompt, few_shot),
                "answer": x[answer_key]
            }, num_proc=self.max_concurrent)

    def get_dataset(self, n: int = -1, seed: int = 0, **kwargs: Any) -> Dataset | None:
        if n > 0 and self.dataset is not None:
            return self.dataset.shuffle(seed=seed).select(range(n)) # type: ignore
        return self.dataset

    def get_eval_dataset(self, n: int = -1, seed: int = 0, **kwargs: Any) -> Dataset | None:
        if n > 0 and self.eval_dataset is not None:
            return self.eval_dataset.shuffle(seed=seed).select(range(n)) # type: ignore
        return self.eval_dataset

    def get_reward_funcs(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()
    
    def get_reward_weights(self, **kwargs: Any) -> List[float]:
        return self.rubric.get_reward_weights()
    
    def get_model_response(self,
                           prompt: str | List[Dict[str, str]],
                           client: OpenAI,
                           model: str,
                           sampling_args: Dict[str, Any] = {},
                           message_type: Literal['chat', 'completion'] | None = None,
                           **kwargs: Any) -> str:
        """
        Get model response for a given prompt (chat or completion).
        
        Convenience function for wrapping (chat, completion) API calls.
        """
        if message_type is None:
            message_type = self.message_type #

        if message_type == 'chat':
            assert isinstance(prompt, list)
            response = client.chat.completions.create(
                model=model,
                messages=prompt, # type: ignore
                **sampling_args
            )
            return response.choices[0].message.content # type: ignore
        elif message_type == 'completion':
            assert isinstance(prompt, str)
            response: str = client.completions.create(
                model=model,
                prompt=prompt,
                **sampling_args
            )
            return response.choices[0].text # type: ignore
  
    @abstractmethod
    def rollout(self,
                client: OpenAI,
                model: str,
                prompt: str | List[Dict[str, str]],
                sampling_args: Dict[str, Any] = {},
                **kwargs: Any) -> str | List[Dict[str, str]]:
        """
        Run a rollout for a given prompt and return the completion.
        """
        pass

    async def _run_single(self,
                          semaphore: Semaphore,
                          client: OpenAI,
                          model: str,
                          prompt: str | List[Dict[str, str]],
                          sampling_args: Dict[str, Any] = {},
                          **kwargs: Any) -> str | List[Dict[str, str]]:
        """
        Run a rollout for a given prompt and return the completion.
        """
        async with semaphore:
            return self.rollout(client, model, prompt, sampling_args, **kwargs)

    async def _run_all(self,
                       client: OpenAI,
                       model: str,
                       prompts: List[str | List[Dict[str, str]]],
                       sampling_args: Dict[str, Any] = {},
                       max_concurrent: int = 32,
                       **kwargs: Any) -> List[str | List[Dict[str, str]]]:
        """
        Run rollouts for a given list of prompts and return the completions.
        """
        from tqdm.asyncio import tqdm_asyncio
        semaphore = Semaphore(max_concurrent)
        rollout_tasks = [
            self._run_single(semaphore, client, model, prompt, sampling_args, **kwargs)
            for prompt in prompts
        ]
        return await tqdm_asyncio.gather(
            *rollout_tasks,
            total=len(prompts),
            desc=f'Running {len(prompts)} rollouts'
        )

    def run_rollouts(self,
                     prompts: List[str | List[Dict[str, str]]],
                     client: OpenAI,
                     model: str,
                     sampling_args: Dict[str, Any] = {},
                     max_concurrent: int = 32,
                     **kwargs: Any) -> List[str | List[Dict[str, str]]]:
        """
        Run rollouts for a given list of prompts and return the completions.
        """

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self._run_all(
                    prompts=prompts, client=client, model=model,
                    sampling_args=sampling_args, max_concurrent=max_concurrent, 
                    **kwargs
                )
            )
        else:
            # Jupyter notebook
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(
                self._run_all(
                    prompts=prompts, client=client, model=model, 
                    sampling_args=sampling_args, max_concurrent=max_concurrent, 
                    **kwargs
                )
            )

    def generate(self,
                 inputs: Dict[str, List[Any]] | Dataset,
                 client: OpenAI | None = None,
                 model: str | None = None,
                 sampling_args: Dict[str, Any] = {},
                 max_concurrent: int | None = None,
                 score_rollouts: bool = True,
                 **kwargs: Any) -> Dict[str, Any]:
        """
        Generate completions and rewards for a given set of inputs.
        """
        # use class-level client and model if not provided
        if client is None:
            assert self.client is not None
            client = self.client
        if model is None:
            assert self.model is not None
            model = self.model
        gen_sampling_args = deepcopy(self.sampling_args)
        gen_sampling_args.update(sampling_args)
        if max_concurrent is None:
            max_concurrent = self.max_concurrent

        # run rollouts    
        if isinstance(inputs, Dataset):
            # get prompt column
            results = {col: deepcopy(inputs[col]) for col in inputs.column_names}
        else:
            results = deepcopy(inputs)
        results['completion'] = self.run_rollouts(
            prompts=results['prompt'],
            client=client,
            model=model,
            sampling_args=gen_sampling_args,
            max_concurrent=max_concurrent,
            **kwargs
        )
        if score_rollouts:
            results_rewards = self.rubric.score_rollouts( 
                prompts=results['prompt'],
                completions=results['completion'],
                answers=results['answer'],
                tasks=results['task'],
                max_concurrent=max_concurrent,
                apply_weights=True
            )       
            results.update(results_rewards)
        return results

    def evaluate(self,
                 client: OpenAI | None = None,
                 model: str | None = None,
                 sampling_args: Dict[str, Any] = {},
                 num_samples: int = -1,
                 max_concurrent: int = 32,
                 **kwargs: Any
                ) -> Dict[str, Any]:
        """
        Evaluate model on the Environment evaluation dataset.
        """
        # use class-level client and model if not provided
        if client is None:
            assert self.client is not None
            client = self.client
        if model is None:
            assert self.model is not None
            model = self.model

        if self.eval_dataset is None:
            self.logger.info('eval_dataset is not set, falling back to train dataset')
            assert self.dataset is not None
            num_rows = len(self.dataset)
            inputs = self.dataset.select(range(num_rows - num_samples, num_rows))
        else:
            num_rows = len(self.eval_dataset)
            inputs = self.eval_dataset

        results = self.generate(
            inputs, client, model, sampling_args, **kwargs
        )
        return results

    def make_dataset(self,
                     results: Dict[str, Any] | None = None,
                     push_to_hub: bool = False,
                     hub_name: str | None = None,
                     client: OpenAI | None = None,
                     model: str | None = None,
                     max_concurrent: int | None = None,
                     num_samples: int = -1,
                     sampling_args: Dict[str, Any] = {'temperature': 0.6},
                     **kwargs: Any) -> Dataset:
        """
        Make a dataset from the evaluation results.
        """
        if results is None and client is None:
            raise ValueError('Either results or client must be provided')
        if push_to_hub and hub_name is None:
            raise ValueError('hub_name must be provided if push_to_hub is True')
        
        # use class-level client and model if not provided
        if client is None:
            assert self.client is not None
            client = self.client
        if model is None:
            assert self.model is not None
            model = self.model
        if max_concurrent is None:
            max_concurrent = self.max_concurrent
        if results is None:
            results = self.evaluate(
                client,
                model, 
                sampling_args,
                max_concurrent, 
                num_samples, 
                **kwargs
            )
        if results['task'][0] is not None:
            dataset = Dataset.from_dict({
                'prompt': results['prompt'],
                'completion': results['completion'],
                'answer': results['answer'],
                'reward': results['reward'],
                'task': results['task']
            })
        else:
            dataset = Dataset.from_dict({
                'prompt': results['prompt'],
                'completion': results['completion'],
                'answer': results['answer'],
                'reward': results['reward'],
            })
        if push_to_hub:
            assert hub_name is not None
            dataset.push_to_hub(hub_name)
        return dataset
