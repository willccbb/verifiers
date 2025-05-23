import asyncio
from asyncio import Semaphore
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Literal, Tuple
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
        # Extract format_prompt as a standalone function to avoid capturing self
        def format_prompt_fn(prompt: str) -> List[Dict[str, str]]:
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            if few_shot:
                messages.extend(few_shot)
            messages.append({'role': 'user', 'content': prompt})
            return messages
        
        if answer_key == "answer":
            return dataset.map(lambda x: {
                "prompt": format_prompt_fn(x[question_key]),
            }, num_proc=self.max_concurrent)
        else:
            return dataset.map(lambda x: {
                "prompt": format_prompt_fn(x[question_key]),
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
    
    def sanitize_sampling_args(self, client: OpenAI, sampling_args: Dict[str, Any]) -> Dict[str, Any]:
        from urllib.parse import urlparse
        url = urlparse(str(client.base_url))
        # check if url is not localhost/127.0.0.1/0.0.0.0
        if url.netloc not in ["localhost", "127.0.0.1", "0.0.0.0"]:
            sanitized_args = deepcopy(sampling_args)
            # remove extra_body
            sanitized_args.pop('extra_body', None)
            return sanitized_args
        return sampling_args

    def get_model_response(self,
                           prompt: str | List[Dict[str, str]],
                           client: OpenAI,
                           model: str,
                           sampling_args: Dict[str, Any] = {},
                           message_type: Literal['chat', 'completion'] | None = None,
                           sanitize_sampling_args: bool = True,
                           **kwargs: Any) -> str:
        """
        Get model response for a given prompt (chat or completion).
        
        Convenience function for wrapping (chat, completion) API calls.
        """
        if sanitize_sampling_args:
            sanitized_args = self.sanitize_sampling_args(client, sampling_args)
        else:
            sanitized_args = sampling_args
        if message_type is None:
            message_type = self.message_type #

        if message_type == 'chat':
            assert isinstance(prompt, list)
            response = client.chat.completions.create(
                model=model,
                messages=prompt, # type: ignore
                **sanitized_args
            )
            return response.choices[0].message.content # type: ignore
        elif message_type == 'completion':
            assert isinstance(prompt, str)
            response: str = client.completions.create(
                model=model,
                prompt=prompt,
                **sanitized_args
            )
            return response.choices[0].text # type: ignore
  
    @abstractmethod
    def rollout(self,
                client: OpenAI,
                model: str,
                prompt: str | List[Dict[str, str]],
                sampling_args: Dict[str, Any] = {},
                **kwargs: Any) -> Tuple[str, Dict[str, Any]] | Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        Run a rollout for a given prompt.
        Returns a tuple of (completion, state).
        """
        pass

    async def _run_single(self,
                          semaphore: Semaphore,
                          client: OpenAI,
                          model: str,
                          prompt: str | List[Dict[str, str]],
                          sampling_args: Dict[str, Any] = {},
                          **kwargs: Any) -> Tuple[str, Dict[str, Any]] | Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        Run a rollout for a given prompt.
        Returns a tuple of (completion, state).
        """
        async with semaphore:
            return await asyncio.to_thread(self.rollout, client, model, prompt, sampling_args, **kwargs)

    async def _run_all(self,
                       client: OpenAI,
                       model: str,
                       prompts: List[str | List[Dict[str, str]]],
                       sampling_args: Dict[str, Any] = {},
                       max_concurrent: int = 32,
                       **kwargs: Any) -> List[Tuple[str, Dict[str, Any]]] | List[Tuple[List[Dict[str, str]], Dict[str, Any]]]:
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
                     **kwargs: Any) -> List[Tuple[str, Dict[str, Any]]] | List[Tuple[List[Dict[str, str]], Dict[str, Any]]]:
        """
        Run rollouts for a given list of prompts and return the completions.
        """
        coro = self._run_all(
            prompts=prompts, client=client, model=model,
            sampling_args=sampling_args, max_concurrent=max_concurrent, 
            **kwargs
        )
        try:
            return asyncio.run(coro)
        except RuntimeError:
            # Jupyter notebook
            import nest_asyncio
            nest_asyncio.apply()
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(coro)

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
        print("Running rollouts")
        rollouts = self.run_rollouts(
            prompts=results['prompt'],
            client=client,
            model=model,
            sampling_args=gen_sampling_args,
            max_concurrent=max_concurrent,
            **kwargs
        )
        print(f"Rollouts done: {len(rollouts)}")
        results['completion'] = [rollout[0] for rollout in rollouts]
        results['state'] = [rollout[1] for rollout in rollouts]
        if 'task' not in results:
            results['task'] = ['default'] * len(results['prompt'])
        if score_rollouts:
            results_rewards = self.rubric.score_rollouts( 
                prompts=results['prompt'],
                completions=results['completion'],
                answers=results['answer'],
                states=results['state'],
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
            inputs = self.dataset
        else:
            inputs = self.eval_dataset
        if num_samples > 0:
            inputs = inputs.select(range(num_samples))

        results = self.generate(
            inputs, client, model, sampling_args, max_concurrent, **kwargs
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
                     state_columns: List[str] = [],
                     extra_columns: List[str] = [],
                     **kwargs: Any) -> Dataset:
        """
        Make a dataset from the evaluation results.
        """
        if results is None and client is None:
            raise ValueError('Either results or client must be provided')
        if push_to_hub and hub_name is None:
            raise ValueError('hub_name must be provided if push_to_hub is True')
        
        if results is None:
            # use class-level client and model if not provided
            if client is None:
                assert self.client is not None
                client = self.client
            if model is None:
                assert self.model is not None
                model = self.model
            if max_concurrent is None:
                max_concurrent = self.max_concurrent
            results = self.evaluate(
                client,
                model, 
                sampling_args,
                num_samples, 
                max_concurrent, 
                **kwargs
            )
        cols = ['prompt', 'completion', 'answer', 'reward']
        if results['task'][0] is not None:
            cols.append('task')
        if 'state' in results:
            for col in state_columns:
                if col in results['state'][0]:
                    results[col] = [state[col] for state in results['state']]
                    cols.append(col)
                else:
                    self.logger.warning(f'Column {col} not found in state, skipping from dataset.')
        for col in extra_columns:
            if col in results:
                cols.append(col)
            else:
                self.logger.warning(f'Column {col} not found in results, skipping from dataset.')
        dataset = Dataset.from_dict({
            col: results[col] for col in cols
        })
        if push_to_hub:
            assert hub_name is not None
            dataset.push_to_hub(hub_name)
        return dataset