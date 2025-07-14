import asyncio
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import TYPE_CHECKING, List, Literal, Tuple

from datasets import Dataset
from openai import AsyncOpenAI, OpenAI

from verifiers import (
    ChatCompletion,
    ChatMessage,
    GenerateInputs,
    GenerateOutputs,
    Info,
    Messages,
    MessageType,
    ModelResponse,
    Parser,
    ProcessedOutputs,
    RewardFunc,
    Rubric,
    SamplingArgs,
    State,
)

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class Environment(ABC):
    """
    Base class for all environments.
    """

    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
        dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        system_prompt: str | None = None,
        few_shot: List[ChatMessage] = [],
        parser: Parser = Parser(),
        rubric: Rubric = Rubric(),
        sampling_args: SamplingArgs = {},
        message_type: MessageType = "chat",
        max_workers: int = 512,
        **kwargs,
    ):
        self.client = client
        self.model = model
        self.message_type: Literal["chat", "completion"] = message_type
        self.system_prompt = system_prompt
        self.few_shot = few_shot

        if self.message_type == "chat":
            if dataset is not None:
                self.dataset = self.format_dataset(
                    dataset, self.system_prompt, self.few_shot
                )
            else:
                self.dataset = None
            if eval_dataset is not None:
                self.eval_dataset = self.format_dataset(
                    eval_dataset, self.system_prompt, self.few_shot
                )
            else:
                self.eval_dataset = None
        else:
            if self.system_prompt or self.few_shot:
                raise ValueError(
                    'The fields "system_prompt" and "few_shot" are not supported for completion tasks.'
                    'Please use message_type="chat" instead, or pre-format your dataset '
                    'to contain "prompt" and "answer" columns.'
                )
            self.dataset = dataset
            self.eval_dataset = eval_dataset
        self.parser = parser
        self.rubric = rubric
        self.sampling_args = {
            "n": 1,  # n > 1 not supported; use duplicate prompts for multiple completions
            "extra_body": {
                #    'skip_special_tokens': False,
                #    'spaces_between_special_tokens': False,
            },
        }
        if sampling_args is not None and "extra_body" in sampling_args:
            self.sampling_args["extra_body"].update(sampling_args["extra_body"])
        for k, v in sampling_args.items():
            if k != "extra_body":
                self.sampling_args[k] = v
        self.max_workers = max_workers
        self.logger = logging.getLogger(f"verifiers.envs.{self.__class__.__name__}")
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.dataset is None and self.eval_dataset is None:
            raise ValueError("Either dataset or eval_dataset must be provided")

    def format_prompt(
        self,
        prompt_str: str,
        system_prompt: str | None = None,
        few_shot: List[ChatMessage] | None = None,
    ) -> List[ChatMessage]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if few_shot:
            messages.extend(few_shot)
        messages.append({"role": "user", "content": prompt_str})
        return messages

    def format_dataset(
        self,
        dataset: Dataset,
        system_prompt: str | None = None,
        few_shot: List[ChatMessage] | None = None,
        question_key: str = "question",
        answer_key: str = "answer",
    ) -> Dataset:
        # skip if "prompt" already exists
        if "prompt" in dataset.column_names:
            return dataset

        # extract format_prompt as a standalone function to avoid capturing self
        def format_prompt_fn(prompt_str: str) -> List[ChatMessage]:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if few_shot:
                messages.extend(few_shot)
            messages.append({"role": "user", "content": prompt_str})
            return messages

        if answer_key == "answer":
            return dataset.map(
                lambda x: {
                    "prompt": format_prompt_fn(x[question_key]),
                }
            )
        else:
            return dataset.map(
                lambda x: {
                    "prompt": format_prompt_fn(x[question_key]),
                    "answer": x[answer_key],
                }
            )

    def get_dataset(self, n: int = -1, seed: int | None = None, **kwargs) -> Dataset:
        if self.dataset is None:
            raise ValueError("dataset is not set")
        if seed is not None:
            self.dataset = self.dataset.shuffle(seed=seed)
        if n > 0:
            return self.dataset.select(range(n))
        return self.dataset

    def get_eval_dataset(
        self, n: int = -1, seed: int | None = None, **kwargs
    ) -> Dataset | None:
        if self.eval_dataset is None:
            self.logger.warning(
                "eval_dataset is not set, falling back to train dataset"
            )
            return self.get_dataset(n, seed, **kwargs)
        if seed is not None:
            self.eval_dataset = self.eval_dataset.shuffle(seed=seed)
        if n > 0:
            return self.eval_dataset.select(range(n))
        return self.eval_dataset

    def get_reward_funcs(self, **kwargs) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()

    def get_reward_weights(self, **kwargs) -> List[float]:
        return self.rubric.get_reward_weights()

    async def get_model_response(
        self,
        prompt: Messages,
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs = {},
        message_type: MessageType | None = None,
        **kwargs,
    ) -> ModelResponse:
        """
        Get model response for a given prompt (chat or completion).

        Convenience function for wrapping (chat, completion) API calls.
        Returns special error messages for context length issues.
        """
        if message_type is None:
            message_type = self.message_type

        if message_type == "chat":
            assert isinstance(prompt, list)
            response = await client.chat.completions.create(
                model=model,
                messages=prompt,  # type: ignore
                **sampling_args,
            )
            return response
        elif message_type == "completion":
            assert isinstance(prompt, str)
            response = await client.completions.create(
                model=model, prompt=prompt, **sampling_args
            )
            return response

    @abstractmethod
    async def rollout(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt: Messages,
        answer: str = "",
        task: str = "default",
        info: Info = {},
        sampling_args: SamplingArgs = {},
        **kwargs,
    ) -> Tuple[Messages, State]:
        """
        Run a rollout for a given prompt.
        Returns a tuple of (completion, state).
        """
        pass

    async def run_rollout_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        client: AsyncOpenAI,
        model: str,
        prompt: Messages,
        answer: str = "",
        task: str = "default",
        info: Info = {},
        sampling_args: SamplingArgs = {},
        **kwargs,
    ) -> Tuple[Messages, State]:
        """
        Run a rollout with a semaphore.
        """
        async with semaphore:
            return await self.rollout(
                client, model, prompt, answer, task, info, sampling_args, **kwargs
            )

    async def run_rollouts(
        self,
        client: AsyncOpenAI,
        model: str,
        prompts: List[Messages],
        answers: List[str],
        tasks: List[str] = [],
        infos: List[Info] = [],
        sampling_args: SamplingArgs = {},
        max_concurrent: int = -1,
        **kwargs,
    ) -> List[Tuple[Messages, State]]:
        """
        Run rollouts for a given list of prompts and return the completions.
        """
        from tqdm.asyncio import tqdm_asyncio

        if max_concurrent > 0:
            semaphore = asyncio.Semaphore(max_concurrent)
            rollout_tasks = [
                self.run_rollout_with_semaphore(
                    semaphore,
                    client,
                    model,
                    prompt,
                    answer,
                    task,
                    info,
                    sampling_args,
                    **kwargs,
                )
                for prompt, answer, task, info in zip(prompts, answers, tasks, infos)
            ]
        else:
            rollout_tasks = [
                self.rollout(
                    client, model, prompt, answer, task, info, sampling_args, **kwargs
                )
                for prompt, answer, task, info in zip(prompts, answers, tasks, infos)
            ]
        return await tqdm_asyncio.gather(
            *rollout_tasks, total=len(prompts), desc=f"Running {len(prompts)} rollouts"
        )

    async def a_generate(
        self,
        inputs: GenerateInputs | Dataset,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
        sampling_args: SamplingArgs = {},
        score_rollouts: bool = True,
        max_concurrent: int = -1,
        **kwargs,
    ) -> GenerateOutputs:
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

        # run rollouts
        if isinstance(inputs, Dataset):
            # get prompt column
            results = {col: deepcopy(inputs[col]) for col in inputs.column_names}
        else:
            results = {col: deepcopy(inputs[col]) for col in inputs}
        if "prompt" not in results:
            raise ValueError("prompt column not found in inputs")
        if "answer" not in results and "info" not in results:
            raise ValueError("answer or info column must be found in inputs")
        if "answer" not in results:
            results["answer"] = [""] * len(results["prompt"])
        if "task" not in results:
            results["task"] = ["default"] * len(results["prompt"])
        if "info" not in results:
            results["info"] = [{}] * len(results["prompt"])

        rollouts = await self.run_rollouts(
            prompts=results["prompt"],
            answers=results["answer"],
            tasks=results["task"],
            infos=results["info"],
            client=client,
            model=model,
            sampling_args=gen_sampling_args,
            max_concurrent=max_concurrent,
            **kwargs,
        )
        results["completion"] = [rollout[0] for rollout in rollouts]
        results["state"] = [rollout[1] for rollout in rollouts]
        if score_rollouts:
            results_rewards = await self.rubric.score_rollouts(
                prompts=results["prompt"],
                completions=results["completion"],
                answers=results["answer"],
                states=results["state"],
                tasks=results["task"],
                infos=results["info"],
                apply_weights=True,
            )
            # add rewards to results
            results.update(results_rewards)
        return results

    def generate(
        self,
        inputs: GenerateInputs | Dataset,
        client: AsyncOpenAI | OpenAI,
        model: str | None = None,
        sampling_args: SamplingArgs = {},
        score_rollouts: bool = True,
        max_concurrent: int = -1,
        **kwargs,
    ) -> GenerateOutputs:
        if isinstance(client, OpenAI):
            client = AsyncOpenAI(api_key=client.api_key, base_url=client.base_url)
        coro = self.a_generate(
            inputs,
            client,
            model,
            sampling_args,
            score_rollouts,
            max_concurrent,
            **kwargs,
        )

        executor = ThreadPoolExecutor(max_workers=self.max_workers)

        try:
            loop = asyncio.new_event_loop()
            loop.set_default_executor(executor)
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        except RuntimeError:
            import nest_asyncio  # type: ignore

            nest_asyncio.apply()
            loop = asyncio.get_running_loop()
            loop.set_default_executor(executor)
            return loop.run_until_complete(coro)
        finally:
            # Critical: shutdown the executor to prevent thread leaks
            executor.shutdown(wait=False)

    def process_chat_format(
        self,
        prompt: List[ChatMessage],
        completion: List[ChatMessage],
        processing_class: "PreTrainedTokenizerBase",
        mask_env_responses: bool = False,
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
        prompt_text = processing_class.apply_chat_template(
            prompt,  # type: ignore
            tokenize=False,
            add_generation_prompt=True,
        )
        assert isinstance(prompt_text, str)
        prompt_ids = processing_class.encode(prompt_text)
        prompt_mask = [0] * len(prompt_ids)

        # track completion tokens and masks by processing incrementally
        completion_ids = []
        completion_mask = []

        # previous tokenization (starts with just prompt)
        prev_ids = prompt_ids

        # process each completion message incrementally
        for i, msg in enumerate(completion):
            # create conversation prefix: prompt + completion[:i+1]
            conversation_prefix = prompt + completion[: i + 1]

            # tokenize the full prefix
            prefix_text = processing_class.apply_chat_template(
                conversation_prefix,  # type: ignore
                tokenize=False,
                add_generation_prompt=False,
            )
            assert isinstance(prefix_text, str)
            current_ids = processing_class.encode(prefix_text)
            assert current_ids[: len(prev_ids) - 1] == prev_ids[:-1], (
                f"Tokenization difference in chat format. Current ids: {current_ids[: len(prev_ids) - 1]}, previous ids: {prev_ids[:-1]}"
            )

            # add new tokens to completion tokens
            new_tokens = current_ids[len(prev_ids) :]
            assert len(new_tokens) > 0, (
                f"No new tokens in chat format. Current ids: {current_ids}, previous ids: {prev_ids}"
            )
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
            # update previous tokenization for next iteration
            prev_ids = current_ids
            assert len(completion_ids) == len(completion_mask), (
                f"Length mismatch in chat format. \
Completion ids: {completion_ids}, completion mask: {completion_mask}. \
This often occurs with models whose tokenizer chat templates discard <think> tokens \
from previous turns, such as Qwen3 or DeepSeek-R1-Distill models. \
For Qwen3 models, you may want to replace the chat template with the Qwen2.5 chat template. \
Model copies with swapped templates are available here: https://huggingface.co/collections/willcb/qwen3-68434f4883925bfdb4570ee5"
            )

        return prompt_ids, prompt_mask, completion_ids, completion_mask

    def process_completion_format(
        self, prompt: str, completion: str, processing_class: "PreTrainedTokenizerBase"
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
        prompt_mask = [0] * len(prompt_ids)

        # Tokenize completion
        completion_ids = processing_class.encode(completion)
        completion_mask = [1] * len(completion_ids)

        return prompt_ids, prompt_mask, completion_ids, completion_mask

    def process_env_results(
        self,
        prompts: List[Messages],
        completions: List[Messages],
        states: List[State],
        rewards: List[float],
        processing_class: "PreTrainedTokenizerBase",
        max_seq_len: int = -1,
        mask_env_responses: bool = False,
        mask_truncated_completions: bool = False,
        zero_truncated_completions: bool = False,
    ) -> ProcessedOutputs:
        """
        Main tokenization pipeline that handles both chat and completion formats.

        Returns:
            Dict with prompt_ids, prompt_mask, completion_ids, completion_mask, rewards
        """
        # Determine format from first prompt
        is_chat_format = isinstance(prompts[0], list)

        all_prompt_ids = []
        all_prompt_masks = []
        all_completion_ids = []
        all_completion_masks = []
        all_completion_logprobs = []
        all_rewards = []
        for i, (prompt, completion, state, reward) in enumerate(
            zip(prompts, completions, states, rewards)
        ):
            # Format-specific processing
            if is_chat_format:
                assert isinstance(prompt, list) and isinstance(completion, list)
                prompt_ids, prompt_mask, completion_ids, completion_mask = (
                    self.process_chat_format(
                        prompt, completion, processing_class, mask_env_responses
                    )
                )
            else:
                assert isinstance(prompt, str) and isinstance(completion, str)
                prompt_ids, prompt_mask, completion_ids, completion_mask = (
                    self.process_completion_format(prompt, completion, processing_class)
                )
            is_truncated = False
            if max_seq_len > 0 and len(prompt_ids) + len(completion_ids) > max_seq_len:
                if len(prompt_ids) > max_seq_len:
                    prompt_ids = prompt_ids[:max_seq_len]
                completion_ids = completion_ids[: max_seq_len - len(prompt_ids)]
                completion_mask = completion_mask[: max_seq_len - len(prompt_ids)]
                is_truncated = True
                assert len(prompt_ids) + len(completion_ids) <= max_seq_len, (
                    f"Prompt length: {len(prompt_ids)}, completion length: {len(completion_ids)}, max_seq_len: {max_seq_len}"
                )
            if is_truncated and mask_truncated_completions:
                completion_mask = [0] * len(completion_ids)
            assert len(prompt_ids) == len(prompt_mask), (
                f"Prompt ids: {len(prompt_ids)}, prompt mask: {len(prompt_mask)}"
            )
            assert len(completion_ids) == len(completion_mask), (
                f"Completion ids: {len(completion_ids)}, completion mask: {len(completion_mask)}"
            )
            all_prompt_ids.append(prompt_ids)
            all_prompt_masks.append(prompt_mask)
            all_completion_ids.append(completion_ids)
            all_completion_masks.append(completion_mask)
            all_completion_logprobs.append([0.0] * len(completion_ids))
            if zero_truncated_completions and is_truncated:
                all_rewards.append(0.0)
            else:
                all_rewards.append(reward)
        return {
            "prompt_ids": all_prompt_ids,
            "prompt_mask": all_prompt_masks,
            "completion_ids": all_completion_ids,
            "completion_mask": all_completion_masks,
            "completion_logprobs": all_completion_logprobs,
            "rewards": all_rewards,
        }

    def parse_chat_completion_logprobs(
        self, chat_completion: ChatCompletion
    ) -> List[float]:
        """Parses the completion logprobs from a vLLM chat completion"""
        assert len(chat_completion.choices) == 1, (
            "Response should always have one choice"
        )
        assert chat_completion.choices[0].logprobs is not None, (
            "Logprobs should not be None. Make sure to set logprobs=True in the extra body when making the request to /v1/chat/completions"
        )
        assert chat_completion.choices[0].logprobs.content is not None, (
            "Logprob content should not be None. Make sure to set logprobs=True in the extra body when making the request to /v1/chat/completions"
        )
        logprobs = [
            logprob.logprob for logprob in chat_completion.choices[0].logprobs.content
        ]
        return logprobs

    def parse_chat_completion_tokens(
        self, chat_completion: ChatCompletion
    ) -> List[int]:
        """Parses the output token ids from a list of chat completions returned by vLLM OAI server."""
        assert len(chat_completion.choices) == 1, (
            "Response should always have one choice"
        )
        assert chat_completion.choices[0].logprobs is not None, (
            "Logprobs should not be None. Make sure to set logprobs=True in the extra body when making the request to /v1/chat/completions"
        )
        assert chat_completion.choices[0].logprobs.content is not None, (
            "Logprob content should not be None. Make sure to set logprobs=True in the extra body when making the request to /v1/chat/completions"
        )
        tokens = [
            int(token.token.split(":")[-1])
            for token in chat_completion.choices[0].logprobs.content
        ]
        return tokens

    def process_chat_format_vllm(
        self,
        prompt: List[ChatMessage],
        completion: List[ChatMessage],
        state: State,
        processing_class: "PreTrainedTokenizerBase",
        mask_env_responses: bool = False,
    ) -> Tuple[List[int], List[int], List[int], List[int], List[float]]:
        """
        Process chat format conversations using incremental prefixes.
        """
        responses = state["responses"]
        zipped = []
        for turn in completion:
            if turn["role"] == "assistant":
                # tuple = turn + popped first response
                zipped.append((turn, responses.pop(0)))
            else:
                zipped.append((turn, None))
        assert len(responses) == 0, "Responses not fully consumed"
        assert len(zipped) == len(completion), "Length mismatch"
        prompt_ids: list[int] = processing_class.apply_chat_template(
            conversation=prompt,  # type: ignore
            add_generation_prompt=True,
        )
        messages_consumed = deepcopy(prompt)
        prompt_mask: list[int] = [0] * len(prompt_ids)
        completion_ids: list[int] = []
        completion_mask: list[int] = []
        completion_logprobs: list[float] = []
        for message, response in zipped:
            # assistant case -- use response
            if message["role"] == "assistant":
                assert response is not None, "Response should not be None"
                completion_turn_ids = self.parse_chat_completion_tokens(response)
                completion_turn_mask = [1] * len(completion_turn_ids)
                completion_turn_logprobs = self.parse_chat_completion_logprobs(response)
                completion_ids.extend(completion_turn_ids)
                completion_mask.extend(completion_turn_mask)
                completion_logprobs.extend(completion_turn_logprobs)
                messages_consumed.append(message)
            # user case -- use message
            else:
                assert message["role"] == "user" or message["role"] == "tool"
                token_prefix: list[int] = processing_class.apply_chat_template(
                    conversation=messages_consumed  # type: ignore
                )
                token_prefix_with_turn: list[int] = (
                    processing_class.apply_chat_template(
                        conversation=messages_consumed + [message],  # type: ignore
                    )
                )
                assert token_prefix_with_turn[: len(token_prefix)] == token_prefix, (
                    f"Token prefix mismatch. Token prefix: {token_prefix}, token prefix with turn: {token_prefix_with_turn}"
                )
                completion_turn_ids = token_prefix_with_turn[len(token_prefix) :]
                if mask_env_responses:
                    completion_turn_mask = [0] * len(completion_turn_ids)
                else:
                    completion_turn_mask = [1] * len(completion_turn_ids)
                completion_turn_logprobs = [0.0] * len(completion_turn_ids)
                completion_ids.extend(completion_turn_ids)
                completion_mask.extend(completion_turn_mask)
                completion_logprobs.extend(completion_turn_logprobs)
                messages_consumed.append(message)
        return (
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
            completion_logprobs,
        )

    def process_env_results_vllm(
        self,
        prompts: List[Messages],
        completions: List[Messages],
        states: List[State],
        rewards: List[float],
        processing_class: "PreTrainedTokenizerBase",
        max_seq_len: int = -1,
        mask_env_responses: bool = False,
        mask_truncated_completions: bool = False,
        zero_truncated_completions: bool = False,
    ) -> ProcessedOutputs:
        """
        Process results with vLLM tokens/logprobs.
        """
        # Determine format from first prompt
        is_chat_format = isinstance(prompts[0], list)
        assert is_chat_format, (
            "vLLM output parsing is not yet supported for completion format"
        )

        all_prompt_ids = []
        all_prompt_masks = []
        all_completion_ids = []
        all_completion_masks = []
        all_completion_logprobs = []
        all_rewards = []
        for i, (prompt, completion, state, reward) in enumerate(
            zip(prompts, completions, states, rewards)
        ):
            # Format-specific processing
            if is_chat_format:
                assert isinstance(prompt, list) and isinstance(completion, list)
                (
                    prompt_ids,
                    prompt_mask,
                    completion_ids,
                    completion_mask,
                    completion_logprobs,
                ) = self.process_chat_format_vllm(
                    prompt, completion, state, processing_class, mask_env_responses
                )
            else:
                assert isinstance(prompt, str) and isinstance(completion, str)
                prompt_ids, prompt_mask, completion_ids, completion_mask = (
                    self.process_completion_format(prompt, completion, processing_class)
                )
            is_truncated = False
            if max_seq_len > 0 and len(prompt_ids) + len(completion_ids) > max_seq_len:
                if len(prompt_ids) > max_seq_len:
                    prompt_ids = prompt_ids[:max_seq_len]
                completion_ids = completion_ids[: max_seq_len - len(prompt_ids)]
                completion_mask = completion_mask[: max_seq_len - len(prompt_ids)]
                is_truncated = True
                assert len(prompt_ids) + len(completion_ids) <= max_seq_len, (
                    f"Prompt length: {len(prompt_ids)}, completion length: {len(completion_ids)}, max_seq_len: {max_seq_len}"
                )
            if is_truncated and mask_truncated_completions:
                completion_mask = [0] * len(completion_ids)
            assert len(prompt_ids) == len(prompt_mask), (
                f"Prompt ids: {len(prompt_ids)}, prompt mask: {len(prompt_mask)}"
            )
            assert len(completion_ids) == len(completion_mask), (
                f"Completion ids: {len(completion_ids)}, completion mask: {len(completion_mask)}"
            )
            completion_logprobs = [0] * len(completion_ids)
            all_prompt_ids.append(prompt_ids)
            all_prompt_masks.append(prompt_mask)
            all_completion_ids.append(completion_ids)
            all_completion_masks.append(completion_mask)
            all_completion_logprobs.append(completion_logprobs)
            if zero_truncated_completions:
                all_rewards.append(0)
            else:
                all_rewards.append(reward)
        return {
            "prompt_ids": all_prompt_ids,
            "prompt_mask": all_prompt_masks,
            "completion_ids": all_completion_ids,
            "completion_mask": all_completion_masks,
            "completion_logprobs": all_completion_logprobs,
            "rewards": all_rewards,
        }

    # Evaluation and dataset generation
    def evaluate(
        self,
        client: AsyncOpenAI | OpenAI,
        model: str,
        sampling_args: SamplingArgs = {},
        num_examples: int = -1,
        rollouts_per_example: int = 1,
        score_rollouts: bool = True,
        max_concurrent: int = -1,
        **kwargs,
    ) -> GenerateOutputs:
        """
        Evaluate model on the Environment evaluation dataset.
        """
        if self.eval_dataset is None:
            self.logger.info("eval_dataset is not set, falling back to train dataset")
            assert self.dataset is not None
            inputs = self.get_dataset(n=num_examples)
        else:
            inputs = self.get_eval_dataset(n=num_examples)
        assert inputs is not None, "No dataset found"
        if rollouts_per_example > 1:
            inputs = inputs.repeat(rollouts_per_example)
        results = self.generate(
            inputs,
            client,
            model,
            sampling_args,
            score_rollouts,
            max_concurrent,
            **kwargs,
        )
        return results

    def make_dataset(
        self,
        results: GenerateOutputs,
        push_to_hub: bool = False,
        hub_name: str | None = None,
        state_columns: List[str] = [],
        extra_columns: List[str] = [],
        **kwargs,
    ) -> Dataset:
        """
        Make a dataset from the evaluation results.
        """
        if push_to_hub and hub_name is None:
            raise ValueError("hub_name must be provided if push_to_hub is True")

        cols = ["prompt", "completion", "answer", "reward"]
        if results["task"][0] is not None:
            cols.append("task")
        if "state" in results:
            for col in state_columns:
                if col in results["state"][0]:
                    results[col] = [state[col] for state in results["state"]]
                    cols.append(col)
                else:
                    self.logger.warning(
                        f"Column {col} not found in state, skipping from dataset."
                    )
        for col in extra_columns:
            if col in results:
                cols.append(col)
            else:
                self.logger.warning(
                    f"Column {col} not found in results, skipping from dataset."
                )
        dataset = Dataset.from_dict({col: results[col] for col in cols})
        if push_to_hub:
            assert hub_name is not None
            dataset.push_to_hub(hub_name)
        return dataset
