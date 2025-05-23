import asyncio
import logging
from asyncio import Semaphore
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Literal, Tuple, Optional, Union


import torch
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
    
    # Processing functions for GRPOEnvTrainer
    def process_state_tokens(
        self,
        state: Dict[str, Any],
        completion_text: str,
        processing_class: PreTrainedTokenizerBase
    ) -> Tuple[List[int], List[float]]:
        """
        Extract tokens and rewards from a single state dict.
        
        Returns:
            Tuple of (token_ids, token_rewards)
        """
        if "token_ids" in state:
            token_ids = state["token_ids"]
            
            if "token_rewards" in state:
                token_rewards = state["token_rewards"]
                # Ensure lengths match
                if len(token_rewards) < len(token_ids):
                    # Repeat last reward
                    if len(token_rewards) > 0:
                        token_rewards = token_rewards + [token_rewards[-1]] * (len(token_ids) - len(token_rewards))
                    else:
                        # Fallback to zero rewards if empty
                        token_rewards = [0.0] * len(token_ids)
                elif len(token_rewards) > len(token_ids):
                    # Truncate rewards
                    token_rewards = token_rewards[:len(token_ids)]
            else:
                # Use sequence-level reward for all tokens
                sequence_reward = state.get("reward", 0.0)
                token_rewards = [sequence_reward] * len(token_ids)
                
            return token_ids, token_rewards
        else:
            # Fallback to retokenization
            token_ids = processing_class.encode(completion_text, add_special_tokens=False)
            sequence_reward = state.get("reward", 0.0)
            token_rewards = [sequence_reward] * len(token_ids)
            return token_ids, token_rewards

    def process_chat_format(
        self,
        prompt: List[Dict[str, str]],
        completion: List[Dict[str, str]],
        processing_class: PreTrainedTokenizerBase,
        mask_intermediate_responses: bool = False
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Process chat format conversations.
        
        Logic:
        1. Apply chat template to prompt messages
        2. Extract assistant responses from completion
        3. If mask_intermediate_responses:
        - Identify tool calls and intermediate user messages
        - Create mask to exclude them from loss computation
        4. Tokenize prompt and completion separately
        
        Returns:
            prompt_ids, prompt_mask, completion_ids, completion_mask
        """
        # Apply chat template to prompt
        prompt_text = processing_class.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        prompt_ids = processing_class.encode(prompt_text, add_special_tokens=False)
        prompt_mask = [1] * len(prompt_ids)
        
        # Extract completion text from messages
        completion_texts = []
        completion_mask_parts = []
        
        for msg in completion:
            if msg["role"] == "assistant":
                msg_text = processing_class.apply_chat_template([msg], tokenize=False, add_generation_prompt=False)
                # Remove any chat template prefix/suffix, keep just content
                content = msg["content"]
                completion_texts.append(content)
                
                # Create mask - mask out tool calls if needed
                if mask_intermediate_responses and ("tool_calls" in msg or content.strip().startswith("<")):
                    # This is a tool call or system message, mask it
                    msg_ids = processing_class.encode(content, add_special_tokens=False)
                    completion_mask_parts.extend([0] * len(msg_ids))
                else:
                    msg_ids = processing_class.encode(content, add_special_tokens=False)
                    completion_mask_parts.extend([1] * len(msg_ids))
            elif msg["role"] == "user" and mask_intermediate_responses:
                # Mask intermediate user messages (like tool results)
                content = msg["content"]
                completion_texts.append(content)
                msg_ids = processing_class.encode(content, add_special_tokens=False)
                completion_mask_parts.extend([0] * len(msg_ids))
        
        # Combine completion texts and tokenize
        full_completion_text = "".join(completion_texts)
        completion_ids = processing_class.encode(full_completion_text, add_special_tokens=False)
        
        # Adjust mask length to match tokenized length
        if len(completion_mask_parts) != len(completion_ids):
            # Fallback: create mask based on actual tokenization
            completion_mask = [1] * len(completion_ids)
        else:
            completion_mask = completion_mask_parts
        
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
        prompt_ids = processing_class.encode(prompt, add_special_tokens=False)
        prompt_mask = [1] * len(prompt_ids)
        
        # Tokenize completion
        completion_ids = processing_class.encode(completion, add_special_tokens=False)
        completion_mask = [1] * len(completion_ids)
        
        return prompt_ids, prompt_mask, completion_ids, completion_mask

    def extract_rewards(
        self,
        env_results: Dict[str, Any],
        states: List[Dict[str, Any]],
        reward_field: str = "reward",
        state_reward_strategy: str = "default"
    ) -> List[float]:
        """
        Extract rewards using priority order:
        
        1. Sequence-level rewards from reward_field
        2. Sequence-level rewards from individual reward functions
        3. Zero rewards as fallback
        
        Returns:
            List of sequence-level rewards
        """
        if reward_field in env_results and env_results[reward_field] is not None:
            return env_results[reward_field]
        
        # Fallback to extracting from states
        rewards = []
        for state in states:
            if "reward" in state and state["reward"] is not None:
                rewards.append(float(state["reward"]))
            else:
                rewards.append(0.0)
        
        return rewards

    def aggregate_token_rewards(
        self,
        token_rewards: List[List[float]], 
        strategy: str = "mean"
    ) -> List[float]:
        """
        Aggregate token-level rewards to sequence level.
        
        Strategies:
        - "mean": Average of all token rewards
        - "sum": Sum of all token rewards  
        - "last": Last token reward only
        - "first": First token reward only
        """
        sequence_rewards = []
        
        for token_reward_seq in token_rewards:
            if len(token_reward_seq) == 0:
                sequence_rewards.append(0.0)
                continue
                
            if strategy == "mean":
                reward = sum(token_reward_seq) / len(token_reward_seq)
            elif strategy == "sum":
                reward = sum(token_reward_seq)
            elif strategy == "last":
                reward = token_reward_seq[-1]
            elif strategy == "first":
                reward = token_reward_seq[0]
            else:
                # Default to mean
                reward = sum(token_reward_seq) / len(token_reward_seq)
                
            sequence_rewards.append(reward)
        
        return sequence_rewards

    def tokenize_and_mask(
        self,
        prompts: List[Union[str, List[Dict[str, str]]]],
        completions: List[Union[str, List[Dict[str, str]]]],
        states: List[Dict[str, Any]],
        processing_class: PreTrainedTokenizerBase,
        mask_intermediate_responses: bool = False,
        max_prompt_length: Optional[int] = None,
        max_completion_length: Optional[int] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Main tokenization pipeline that handles both formats.
        
        Data Flow:
        1. Determine format (chat vs completion) from first prompt
        2. For each prompt/completion pair:
        a. Check state for token_ids (priority)
        b. If not found, tokenize from text
        c. Extract token rewards from state
        d. Apply format-specific processing
        e. Create appropriate masks
        3. Pad sequences to max lengths
        4. Handle EOS token masking
        5. Apply truncation if needed
        
        Returns:
            Dict with prompt_ids, prompt_mask, completion_ids, completion_mask tensors
        """
        if device is None:
            device = torch.device("cpu")
        
        # Determine format from first prompt
        is_chat_format = isinstance(prompts[0], list)
        
        all_prompt_ids = []
        all_prompt_masks = []
        all_completion_ids = []
        all_completion_masks = []
        
        for i, (prompt, completion, state) in enumerate(zip(prompts, completions, states)):
            # Check if state has token_ids (priority)
            if "token_ids" in state:
                completion_text = completion if isinstance(completion, str) else str(completion)
                token_ids, token_rewards = self.process_state_tokens(state, completion_text, processing_class)
                
                # For state tokens, we need to determine prompt boundary
                if is_chat_format:
                    # Apply chat template to get prompt length
                    prompt_text = processing_class.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                    prompt_ids = processing_class.encode(prompt_text, add_special_tokens=False)
                    prompt_mask = [1] * len(prompt_ids)
                else:
                    prompt_ids = processing_class.encode(prompt, add_special_tokens=False)
                    prompt_mask = [1] * len(prompt_ids)
                
                # token_ids from state are completion tokens
                completion_ids = token_ids
                completion_mask = [1] * len(completion_ids)
                
            else:
                # Fallback to format-specific processing
                if is_chat_format:
                    prompt_ids, prompt_mask, completion_ids, completion_mask = self.process_chat_format(
                        prompt, completion, processing_class, mask_intermediate_responses
                    )
                else:
                    prompt_ids, prompt_mask, completion_ids, completion_mask = self.process_completion_format(
                        prompt, completion, processing_class
                    )
            
            # Apply truncation
            if max_prompt_length is not None and len(prompt_ids) > max_prompt_length:
                prompt_ids = prompt_ids[-max_prompt_length:]
                prompt_mask = prompt_mask[-max_prompt_length:]
                
            if max_completion_length is not None and len(completion_ids) > max_completion_length:
                completion_ids = completion_ids[:max_completion_length]
                completion_mask = completion_mask[:max_completion_length]
            
            # Handle EOS token masking
            eos_token_id = getattr(processing_class, 'eos_token_id', None)
            if eos_token_id is not None and eos_token_id in completion_ids:
                eos_idx = completion_ids.index(eos_token_id)
                # Mask everything after first EOS token (but include the EOS token itself)
                completion_mask = completion_mask[:eos_idx + 1] + [0] * (len(completion_mask) - eos_idx - 1)
            
            all_prompt_ids.append(prompt_ids)
            all_prompt_masks.append(prompt_mask)
            all_completion_ids.append(completion_ids)
            all_completion_masks.append(completion_mask)
        
        # Pad sequences
        pad_token_id = getattr(processing_class, 'pad_token_id', 0)
        
        # Convert to tensors and pad
        max_prompt_len = max(len(ids) for ids in all_prompt_ids) if all_prompt_ids else 0
        max_completion_len = max(len(ids) for ids in all_completion_ids) if all_completion_ids else 0
        
        padded_prompt_ids = []
        padded_prompt_masks = []
        padded_completion_ids = []
        padded_completion_masks = []
        
        for prompt_ids, prompt_mask, completion_ids, completion_mask in zip(
            all_prompt_ids, all_prompt_masks, all_completion_ids, all_completion_masks
        ):
            # Pad prompts (left padding)
            prompt_padding = max_prompt_len - len(prompt_ids)
            padded_prompt_ids.append([pad_token_id] * prompt_padding + prompt_ids)
            padded_prompt_masks.append([0] * prompt_padding + prompt_mask)
            
            # Pad completions (right padding)
            completion_padding = max_completion_len - len(completion_ids)
            padded_completion_ids.append(completion_ids + [pad_token_id] * completion_padding)
            padded_completion_masks.append(completion_mask + [0] * completion_padding)
        
        return {
            "prompt_ids": torch.tensor(padded_prompt_ids, dtype=torch.long, device=device),
            "prompt_mask": torch.tensor(padded_prompt_masks, dtype=torch.long, device=device),
            "completion_ids": torch.tensor(padded_completion_ids, dtype=torch.long, device=device),
            "completion_mask": torch.tensor(padded_completion_masks, dtype=torch.long, device=device),
        }

    def process_environment_results(
        self,
        env_results: Dict[str, Any],
        processing_class: PreTrainedTokenizerBase,
        num_generations: int,
        max_prompt_length: Optional[int] = None,
        max_completion_length: Optional[int] = None,
        mask_intermediate_responses: bool = False,
        reward_field: str = "reward",
        state_reward_strategy: str = "default",
        device: Optional[torch.device] = None,
        **kwargs
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """
        Process Environment generation results into trainer format.
        
        This function converts Python types from Environment to torch tensors.
        Tensors are created directly on the target device to avoid CPU→GPU transfers.
        
        Args:
            env_results: Output from Environment.generate()
            processing_class: Tokenizer for encoding/decoding
            num_generations: Number of generations per prompt
            max_prompt_length: Maximum prompt length for truncation
            max_completion_length: Maximum completion length for truncation
            mask_intermediate_responses: Whether to mask intermediate tool responses
            reward_field: Field name to use for rewards (default: "reward")
            state_reward_strategy: How to extract rewards from state dicts
            device: Target device for tensors (typically main process GPU)
            
        Returns:
            Dict with keys: prompt_ids, prompt_mask, completion_ids, completion_mask, 
                        rewards, raw_completions (for logging)
        """
        if device is None:
            device = torch.device("cpu")
        
        # Extract data from env_results
        prompts = env_results["prompt"]
        completions = env_results["completion"]
        states = env_results.get("state", [{}] * len(prompts))
        
        # Tokenize and create masks
        tokenized_results = self.tokenize_and_mask(
            prompts=prompts,
            completions=completions,
            states=states,
            processing_class=processing_class,
            mask_intermediate_responses=mask_intermediate_responses,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            device=device
        )
        
        # Extract rewards
        rewards = self.extract_rewards(
            env_results=env_results,
            states=states,
            reward_field=reward_field,
            state_reward_strategy=state_reward_strategy
        )
        
        # Convert rewards to tensor
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        
        # Store raw completions for logging
        raw_completions = []
        for completion in completions:
            if isinstance(completion, list):
                # Chat format - extract content
                raw_completions.append(" ".join([msg.get("content", "") for msg in completion if msg.get("content")]))
            else:
                raw_completions.append(str(completion))
        
        return {
            "prompt_ids": tokenized_results["prompt_ids"],
            "prompt_mask": tokenized_results["prompt_mask"],
            "completion_ids": tokenized_results["completion_ids"],
            "completion_mask": tokenized_results["completion_mask"],
            "rewards": rewards_tensor,
            "raw_completions": raw_completions,
        }

    # Evaluation and dataset generation
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