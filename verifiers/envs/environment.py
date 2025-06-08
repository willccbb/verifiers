import asyncio
import logging
from asyncio import Semaphore
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Literal, Tuple, Optional, Union
import concurrent.futures

from transformers.tokenization_utils_base import PreTrainedTokenizerBase 

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
                 few_shot: List[Dict[str, Any]] = [],
                 parser: Parser = Parser(),
                 rubric: Rubric = Rubric(),
                 sampling_args: Dict[str, Any] = {},
                 max_concurrent: int = 128,
                 message_type: Literal['chat', 'completion'] = 'chat',
                 **kwargs: Any):
        self.client = client
        self.model = model
        self.message_type: Literal['chat', 'completion'] = message_type
        self.system_prompt = system_prompt
        self.few_shot = few_shot
        self.max_concurrent = max_concurrent
        
        # Ensure asyncio.to_thread doesn't hit default 32 thread limit
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent)
            loop.set_default_executor(executor)
        
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
                      few_shot: List[Dict[str, Any]] | None = None
                      ) -> List[Dict[str, Any]]:
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
                       few_shot: List[Dict[str, Any]] | None = None,
                       question_key: str = "question",
                       answer_key: str = "answer") -> Dataset:
        # Extract format_prompt as a standalone function to avoid capturing self
        def format_prompt_fn(prompt: str) -> List[Dict[str, Any]]:
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
            }, num_proc=min(self.max_concurrent, 32))
        else:
            return dataset.map(lambda x: {
                "prompt": format_prompt_fn(x[question_key]),
                "answer": x[answer_key]
            }, num_proc=min(self.max_concurrent, 32))

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
        Returns special error messages for context length issues.
        """
        if sanitize_sampling_args:
            sanitized_args = self.sanitize_sampling_args(client, sampling_args)
        else:
            sanitized_args = sampling_args
        if message_type is None:
            message_type = self.message_type

        try:
            if message_type == 'chat':
                assert isinstance(prompt, list)
                response = client.chat.completions.create(
                    model=model,
                    messages=prompt, # type: ignore
                    **sanitized_args
                )
                # Check if generation was truncated due to max_tokens
                if response.choices[0].finish_reason == 'length':
                    return "[ERROR] max_tokens_reached"
                return response.choices[0].message.content # type: ignore
            elif message_type == 'completion':
                assert isinstance(prompt, str)
                response = client.completions.create(
                    model=model,
                    prompt=prompt,
                    **sanitized_args
                )
                # Check if generation was truncated due to max_tokens
                if response.choices[0].finish_reason == 'length':
                    return "[ERROR] max_tokens_reached"
                return response.choices[0].text # type: ignore
        except Exception as e:
            # Check for prompt too long errors
            error_msg = str(e)
            if "longer than the maximum" in error_msg or "exceeds the model" in error_msg:
                return "[ERROR] prompt_too_long"
            # Re-raise other errors
            raise

    @abstractmethod
    def rollout(self,
                client: OpenAI,
                model: str,
                prompt: str | List[Dict[str, Any]],
                answer: str,
                sampling_args: Dict[str, Any] = {},
                **kwargs: Any) -> Tuple[Union[str, List[Dict[str, Any]]], Dict[str, Any]]:
        """
        Run a rollout for a given prompt.
        Returns a tuple of (completion, state).
        """
        pass

    async def _run_single(self,
                          semaphore: Semaphore,
                          client: OpenAI,
                          model: str,
                          prompt: str | List[Dict[str, Any]],
                          answer: str,
                          sampling_args: Dict[str, Any] = {},
                          **kwargs: Any) -> Tuple[Union[str, List[Dict[str, Any]]], Dict[str, Any]]:
        """
        Run a rollout for a given prompt.
        Returns a tuple of (completion, state).
        """
        async with semaphore:
            return await asyncio.to_thread(self.rollout, client, model, prompt, answer, sampling_args, **kwargs)

    async def _run_all(self,
                       client: OpenAI,
                       model: str,
                       prompts: List[str | List[Dict[str, str]]],
                       answers: List[str],
                       sampling_args: Dict[str, Any] = {},
                       max_concurrent: int = 128,
                       **kwargs: Any) -> List[Tuple[Union[str, List[Dict[str, Any]]], Dict[str, Any]]]:
        """
        Run rollouts for a given list of prompts and return the completions.
        """
        from tqdm.asyncio import tqdm_asyncio
        semaphore = Semaphore(max_concurrent)
        rollout_tasks = [
            self._run_single(semaphore, client, model, prompt, answer, sampling_args, **kwargs)
            for prompt, answer in zip(prompts, answers)
        ]
        return await tqdm_asyncio.gather(
            *rollout_tasks,
            total=len(prompts),
            desc=f'Running {len(prompts)} rollouts'
        )

    def run_rollouts(self,
                     client: OpenAI,
                     model: str,
                     prompts: List[Union[str, List[Dict[str, Any]]]],
                     answers: List[str],
                     sampling_args: Dict[str, Any] = {},
                     max_concurrent: int = 128,
                     **kwargs: Any) -> List[Tuple[Union[str, List[Dict[str, Any]]], Dict[str, Any]]]:
        """
        Run rollouts for a given list of prompts and return the completions.
        """
        def setup_executor(loop):
            if loop._default_executor is None:
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent)
                loop.set_default_executor(executor)
        
        coro = self._run_all(
            client=client,
            model=model,
            prompts=prompts,
            answers=answers, 
            sampling_args=sampling_args,
            max_concurrent=max_concurrent, 
            **kwargs
        )
        try:
            # Create new event loop with custom executor
            loop = asyncio.new_event_loop()
            setup_executor(loop)
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        except RuntimeError:
            # Jupyter notebook or existing event loop
            import nest_asyncio
            nest_asyncio.apply()
            loop = asyncio.get_running_loop()
            setup_executor(loop)
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
        rollouts = self.run_rollouts(
            prompts=results['prompt'],
            answers=results['answer'],
            client=client,
            model=model,
            sampling_args=gen_sampling_args,
            max_concurrent=max_concurrent,
            **kwargs
        )
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
    
    def process_chat_format(
        self,
        prompt: List[Dict[str, str]],
        completion: List[Dict[str, str]],
        processing_class: PreTrainedTokenizerBase,
        mask_env_responses: bool = False
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Process chat format conversations using incremental prefixes.
        
        Logic:
        1. For each step, tokenize conversation prefix (prompt + completion[:i])
        2. Calculate token differences between steps to get individual message tokens
        3. Apply masking for intermediate responses if needed
        
        Returns:
            prompt_ids, prompt_mask, completion_ids, completion_mask
        """
        # tokenize just the prompt
        prompt_text = processing_class.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        assert isinstance(prompt_text, str)
        prompt_ids = processing_class.encode(prompt_text)
        prompt_mask = [1] * len(prompt_ids)
        
        # track completion tokens and masks by processing incrementally
        completion_ids = []
        completion_mask = []
        
        # previous tokenization (starts with just prompt)
        prev_ids = prompt_ids
        
        # process each completion message incrementally
        for i, msg in enumerate(completion):
            # create conversation prefix: prompt + completion[:i+1]
            conversation_prefix = prompt + completion[:i+1]
            
            # tokenize the full prefix
            prefix_text = processing_class.apply_chat_template(
                conversation_prefix, 
                tokenize=False, 
                add_generation_prompt=False,
            )
            assert isinstance(prefix_text, str), f"Expected string from apply_chat_template, got {type(prefix_text)}"
            current_ids = processing_class.encode(prefix_text)
            assert current_ids[:len(prev_ids)] == prev_ids, f"Tokenization difference in chat format. Current ids: {current_ids}, previous ids: {prev_ids}"
            
            # add new tokens to completion tokens
            new_tokens = current_ids[len(prev_ids):] 
            assert len(new_tokens) > 0, f"No new tokens in chat format. Current ids: {current_ids}, previous ids: {prev_ids}"
            completion_ids.extend(new_tokens)

            # create mask
            if msg["role"] == "assistant":
                msg_mask = [1] * len(new_tokens)
            elif msg["role"] != "assistant" and mask_env_responses:
                # mask intermediate 'user' and/or 'tool' messages 
                msg_mask = [0] * len(new_tokens)
            else:
                # default to not masking
                msg_mask = [1] * len(new_tokens)
            
            completion_mask.extend(msg_mask)
            # Update previous tokenization for next iteration
            prev_ids = current_ids
            assert len(completion_ids) == len(completion_mask), f"Length mismatch in chat format. Completion ids: {completion_ids}, completion mask: {completion_mask}"

        return prompt_ids, prompt_mask, completion_ids, completion_mask

    def process_completion_format(
        self,
        prompt: str,
        completion: str,
        processing_class: PreTrainedTokenizerBase
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Process completion format text.
        
        Logic:
        1. Tokenize prompt separately to get boundary
        2. Tokenize completion 
        3. Create masks (prompt mask all 1s, completion mask handles EOS)
        
        Returns:
            prompt_ids, prompt_mask, completion_ids, completion_mask
        """
        # Tokenize prompt
        prompt_ids = processing_class.encode(prompt)
        prompt_mask = [1] * len(prompt_ids)
        
        # Tokenize completion
        completion_ids = processing_class.encode(completion)
        completion_mask = [1] * len(completion_ids)
        
        return prompt_ids, prompt_mask, completion_ids, completion_mask

    def process_env_results(
        self,
        prompts: List[Union[str, List[Dict[str, Any]]]],
        completions: List[Union[str, List[Dict[str, Any]]]],
        states: List[Dict[str, Any]],
        rewards: List[float],
        processing_class: PreTrainedTokenizerBase,
        max_completion_length: int = -1,
        mask_truncated_completions: bool = False,
        mask_env_responses: bool = False,
    ) -> Dict[str, List[Any]]:
        """
        Main tokenization pipeline that handles both chat and completion formats.
        
        Returns:
            Dict with prompt_ids, prompt_mask, completion_ids, completion_mask, rewards
        """
        # TODO: states + rewards for intermediate-reward objectives

        # Determine format from first prompt
        is_chat_format = isinstance(prompts[0], list)
 
        all_prompt_ids = []
        all_prompt_masks = []
        all_completion_ids = []
        all_completion_masks = []

        for i, (prompt, completion, state, reward) in enumerate(zip(prompts, completions, states, rewards)):
            # Format-specific processing
            if is_chat_format:
                assert isinstance(prompt, list) and isinstance(completion, list)
                prompt_ids, prompt_mask, completion_ids, completion_mask = self.process_chat_format(
                    prompt, completion, processing_class, mask_env_responses
                )
            else:
                assert isinstance(prompt, str) and isinstance(completion, str)
                prompt_ids, prompt_mask, completion_ids, completion_mask = self.process_completion_format(
                    prompt, completion, processing_class
                )
            if mask_truncated_completions and max_completion_length > 0 and len(completion_ids) > max_completion_length:
                completion_ids = completion_ids[:max_completion_length]
                completion_mask = [0] * len(completion_ids)
            all_prompt_ids.append(prompt_ids)
            all_prompt_masks.append(prompt_mask)
            all_completion_ids.append(completion_ids)
            all_completion_masks.append(completion_mask)
 
        return {
            "prompt_ids": all_prompt_ids,
            "prompt_mask": all_prompt_masks,
            "completion_ids": all_completion_ids,
            "completion_mask": all_completion_masks,
            "rewards": rewards,
        }

    # Evaluation and dataset generation
    def evaluate(self,
                 client: OpenAI | None = None,
                 model: str | None = None,
                 sampling_args: Dict[str, Any] = {},
                 num_samples: int = -1,
                 max_concurrent: int = 128,
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