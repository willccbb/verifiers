import asyncio
import json
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import TYPE_CHECKING, Literal

from datasets import Dataset
from openai import AsyncOpenAI, BadRequestError, OpenAI

from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import (
    ChatCompletion,
    ChatCompletionToolParam,
    ChatMessage,
    Completion,
    GenerateInputs,
    GenerateOutputs,
    Info,
    Messages,
    MessageType,
    ModelResponse,
    ProcessedOutputs,
    RewardFunc,
    SamplingArgs,
    State,
)
from verifiers.utils.message_utils import (
    cleanup_messages,
    get_overlong_prompt_dummy_response,
    sanitize_tool_calls,
)

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import (  # type: ignore
        PreTrainedTokenizerBase,
    )


class Environment(ABC):
    """
    Base class for all environments.
    """

    def __init__(
        self,
        dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        system_prompt: str | None = None,
        few_shot: list[ChatMessage] | None = None,
        parser: Parser | None = None,
        rubric: Rubric | None = None,
        sampling_args: SamplingArgs | None = None,
        message_type: MessageType = "chat",
        oai_tools: list[ChatCompletionToolParam] | None = None,
        max_workers: int = 512,
        **kwargs,
    ):
        self.logger = logging.getLogger(f"verifiers.envs.{self.__class__.__name__}")
        self.message_type: Literal["chat", "completion"] = message_type
        self.oai_tools: list[ChatCompletionToolParam] | None = oai_tools
        self.system_prompt = system_prompt
        self.few_shot = few_shot
        self.parser = parser or Parser()
        self.rubric = rubric or Rubric()
        if self.parser.__class__ != self.rubric.parser.__class__:
            self.logger.warning(
                "The parser and rubric parser are different. This may cause unexpected behavior."
            )

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
                    'to contain a "prompt" column.'
                )
            self.dataset = dataset
            self.eval_dataset = eval_dataset

        self.sampling_args = {"n": 1, "extra_body": {}}
        if sampling_args is not None:
            # merge extra_body if provided
            self.sampling_args["extra_body"].update(sampling_args.get("extra_body", {}))
            # copy other keys
            for key, value in sampling_args.items():
                if key != "extra_body":
                    self.sampling_args[key] = value

        self.max_workers = max_workers
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.dataset is None and self.eval_dataset is None:
            raise ValueError("Either dataset or eval_dataset must be provided")

    def format_prompt(
        self,
        prompt_str: str,
        system_prompt: str | None = None,
        few_shot: list[ChatMessage] | None = None,
    ) -> list[ChatMessage]:
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
        few_shot: list[ChatMessage] | None = None,
        question_key: str = "question",
        answer_key: str = "answer",
    ) -> Dataset:
        # skip if "prompt" already exists
        if "id" not in dataset.column_names:
            dataset = dataset.add_column("id", range(len(dataset)))  # type: ignore

        # extract format_prompt as a standalone function to avoid capturing self
        def format_prompt_fn(prompt_str: str) -> list[ChatMessage]:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if few_shot:
                messages.extend(few_shot)
            messages.append({"role": "user", "content": prompt_str})
            return messages

        if "prompt" not in dataset.column_names:
            if answer_key == "answer":
                dataset = dataset.map(
                    lambda x: {
                        "prompt": format_prompt_fn(x[question_key]),
                    }
                )
            else:
                dataset = dataset.map(
                    lambda x: {
                        "prompt": format_prompt_fn(x[question_key]),
                        "answer": x[answer_key],
                    }
                )
        assert "id" in dataset.column_names
        assert "prompt" in dataset.column_names
        return dataset

    def get_dataset(self, n: int = -1, seed: int | None = None) -> Dataset:
        if self.dataset is None:
            raise ValueError("dataset is not set")
        if seed is not None:
            self.dataset = self.dataset.shuffle(seed=seed)
        if n > 0:
            # Cap n to the length of the dataset to prevent IndexError
            n = min(n, len(self.dataset))
            return self.dataset.select(range(n))
        return self.dataset

    def get_eval_dataset(self, n: int = -1, seed: int | None = None) -> Dataset | None:
        if self.eval_dataset is None:
            self.logger.warning(
                "eval_dataset is not set, falling back to train dataset"
            )
            return self.get_dataset(n, seed)
        if seed is not None:
            self.eval_dataset = self.eval_dataset.shuffle(seed=seed)
        if n > 0:
            # Cap n to the length of the dataset to prevent IndexError
            n = min(n, len(self.eval_dataset))
            return self.eval_dataset.select(range(n))
        return self.eval_dataset

    def get_reward_funcs(self) -> list[RewardFunc]:
        return self.rubric.get_reward_funcs()

    def get_reward_weights(self) -> list[float]:
        return self.rubric.get_reward_weights()

    async def get_model_response(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt: Messages,
        oai_tools: list[ChatCompletionToolParam] | None = None,
        sampling_args: SamplingArgs | None = None,
        message_type: MessageType | None = None,
        **kwargs,
    ) -> ModelResponse:
        """
        Get model response for a given prompt (chat or completion).

        Convenience function for wrapping (chat, completion) API calls.
        Returns special error messages for context length issues.
        """
        sampling_args = sampling_args or {}
        # Resolve message type first
        if message_type is None:
            message_type = self.message_type
        # Normalize sampling args:
        # - If max_tokens is provided for chat, rename to max_completion_tokens
        # - Drop any None-valued entries to avoid sending them to the client
        if "max_tokens" in sampling_args:
            if sampling_args["max_tokens"] is None:
                sampling_args.pop("max_tokens")
            elif message_type == "chat":
                sampling_args["max_completion_tokens"] = sampling_args.pop("max_tokens")
        if (
            "max_completion_tokens" in sampling_args
            and sampling_args["max_completion_tokens"] is None
        ):
            sampling_args.pop("max_completion_tokens")
        clean_sampling_args = {k: v for k, v in sampling_args.items() if v is not None}
        try:
            if message_type == "chat":
                assert isinstance(prompt, list)
                # --- detect audio parts and force text-only modality if caller didn't set one ---
                has_audio = False
                try:
                    for m in prompt:
                        c = m.get("content")  # type: ignore[assignment]
                        if isinstance(c, list):
                            for p in c:
                                if isinstance(p, dict) and str(
                                    p.get("type", "")
                                ).startswith("input_audio"):
                                    has_audio = True
                                    break
                        if has_audio:
                            break
                except Exception:
                    has_audio = False
                if has_audio and "modalities" not in clean_sampling_args:
                    clean_sampling_args = {
                        **clean_sampling_args,
                        "modalities": ["text"],
                    }

                if oai_tools:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=prompt,  # type: ignore
                        tools=oai_tools,
                        **clean_sampling_args,
                    )
                else:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=prompt,  # type: ignore
                        **clean_sampling_args,
                    )
                return response
            elif message_type == "completion":
                if oai_tools:
                    raise ValueError(
                        "oai_tools are not supported for completion tasks."
                    )
                assert isinstance(prompt, str)
                response = await client.completions.create(
                    model=model, prompt=prompt, **clean_sampling_args
                )
                return response
        except Exception as e:
            # In case of making a request with an overlong prompt, e.g from a too-long
            # environment response, we return a dummy response to with finish_reason "length"
            if isinstance(e, BadRequestError) and e.response.text.startswith(
                '{"error":{"message":"This model\'s maximum context length is'
            ):
                self.logger.debug("Caught overlong prompt.")
                return get_overlong_prompt_dummy_response(
                    message_type or self.message_type
                )
            self.logger.error(f"Error getting model response: {e} \n\nExiting...")
            raise e

    @abstractmethod
    async def rollout(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt: Messages,
        answer: str = "",
        task: str = "default",
        info: Info | None = None,
        sampling_args: SamplingArgs | None = None,
        **kwargs,
    ) -> tuple[Messages, State]:
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
        info: Info | None = None,
        sampling_args: SamplingArgs | None = None,
        **kwargs,
    ) -> tuple[Messages, State]:
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
        prompts: list[Messages],
        answers: list[str],
        tasks: list[str],
        infos: list[Info],
        sampling_args: SamplingArgs | None = None,
        max_concurrent: int = -1,
        **kwargs,
    ) -> list[tuple[Messages, State]]:
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
        inputs: GenerateInputs | Dataset | dict,
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
        score_rollouts: bool = True,
        max_concurrent: int = -1,
        max_concurrent_generation: int | None = None,
        max_concurrent_scoring: int | None = None,
        interleave_scoring: bool = True,
        **kwargs,
    ) -> GenerateOutputs:
        """
        Generate completions and rewards for a given set of inputs.
        """
        if isinstance(inputs, GenerateInputs):
            inputs = inputs.model_dump()
        gen_sampling_args = deepcopy(self.sampling_args)
        if sampling_args is not None:
            gen_sampling_args.update(sampling_args)

        # preprocess dataset or GenerateInputs to GenerateOutputs
        results_dict = {}
        if isinstance(inputs, Dataset):
            # get prompt column
            results_dict = {}
            for col in inputs.column_names:
                if col == "info":
                    # handle info column to ensure mutable dicts
                    if isinstance(inputs[col][0], str):
                        results_dict[col] = [json.loads(item) for item in inputs[col]]
                    else:
                        results_dict[col] = [dict(item) for item in inputs[col]]
                else:
                    results_dict[col] = deepcopy(inputs[col])
        else:
            results_dict = {col: deepcopy(inputs[col]) for col in inputs}
        if "prompt" not in results_dict:
            raise ValueError("prompt column not found in inputs")
        if "answer" not in results_dict and "info" not in results_dict:
            self.logger.warning(
                "Neither 'answer' nor 'info' column found in inputs. "
                "Some environments can evaluate using only prompt/completion/state, "
                "but reward functions requiring ground truth data may return 0.0. "
                "Proceeding with empty values."
            )
        if not results_dict.get("answer"):
            results_dict["answer"] = [""] * len(results_dict["prompt"])
        if not results_dict.get("task"):
            results_dict["task"] = ["default"] * len(results_dict["prompt"])
        if not results_dict.get("info"):
            results_dict["info"] = [{}] * len(results_dict["prompt"])
        for i, info in enumerate(results_dict["info"]):
            if isinstance(info, str):
                info = json.loads(info)
            if self.oai_tools and "oai_tools" not in info:
                info["oai_tools"] = self.oai_tools

        results_dict["prompt"] = [cleanup_messages(p) for p in results_dict["prompt"]]

        # prepare GenerateOutputs and run rollouts
        results = GenerateOutputs(
            prompt=results_dict["prompt"],
            answer=results_dict["answer"],
            task=results_dict["task"],
            info=results_dict["info"],
            completion=[],
            state=[],
            reward=[],
            metrics={},
        )
        n = len(results.prompt)

        # Resolve concurrency knobs
        gen_limit = max_concurrent_generation
        score_limit = max_concurrent_scoring
        if gen_limit is None:
            gen_limit = max_concurrent
        if score_limit is None:
            score_limit = max_concurrent

        if interleave_scoring and score_rollouts:
            # Interleaved pipeline: generate and score per rollout with separate semaphores
            results_completion: list[Messages] = [None] * n  # type: ignore[assignment]
            results_state: list[State] = [None] * n  # type: ignore[assignment]
            rewards: list[float] = [0.0] * n
            # Pre-allocate metrics using known reward function names
            reward_func_names = self.rubric.get_reward_func_names()
            metrics: dict[str, list[float]] = {
                name: [0.0] * n for name in reward_func_names
            }

            gen_semaphore = (
                asyncio.Semaphore(gen_limit) if gen_limit and gen_limit > 0 else None
            )
            score_semaphore = (
                asyncio.Semaphore(score_limit)
                if score_limit and score_limit > 0
                else None
            )

            async def run_one(i: int) -> None:
                prompt_i = results.prompt[i]
                answer_i = results.answer[i]
                task_i = results.task[i]
                info_i = results.info[i]
                # Generation stage
                if gen_semaphore is not None:
                    async with gen_semaphore:
                        comp_i, state_i = await self.rollout(
                            client,
                            model,
                            prompt_i,
                            answer_i,
                            task_i,
                            info_i,
                            gen_sampling_args,
                            **kwargs,
                        )
                else:
                    comp_i, state_i = await self.rollout(
                        client,
                        model,
                        prompt_i,
                        answer_i,
                        task_i,
                        info_i,
                        gen_sampling_args,
                        **kwargs,
                    )
                results_completion[i] = comp_i
                results_state[i] = state_i
                # Scoring stage
                if score_semaphore is not None:
                    async with score_semaphore:
                        rs = await self.rubric.score_rollout(
                            prompt=prompt_i,
                            completion=comp_i,
                            answer=answer_i,
                            state=state_i,
                            task=task_i,
                            info=info_i,
                            **kwargs,
                        )
                else:
                    rs = await self.rubric.score_rollout(
                        prompt=prompt_i,
                        completion=comp_i,
                        answer=answer_i,
                        state=state_i,
                        task=task_i,
                        info=info_i,
                        **kwargs,
                    )
                rewards[i] = rs.reward
                for k, v in rs.metrics.items():
                    # Ensure key exists in case of EnvGroup/RubricGroup dynamics
                    if k not in metrics:
                        metrics[k] = [0.0] * n
                    metrics[k][i] = v

            tasks = [run_one(i) for i in range(n)]
            from tqdm.asyncio import tqdm_asyncio

            await tqdm_asyncio.gather(
                *tasks, total=n, desc=f"Running {n} rollouts (interleaved)"
            )

            results.completion = results_completion  # type: ignore[assignment]
            results.state = results_state  # type: ignore[assignment]
            results.reward = rewards
            results.metrics = metrics
            return results
        else:
            # Non-interleaved: generate all then score all
            rollouts = await self.run_rollouts(
                prompts=results.prompt,
                answers=results.answer,
                tasks=results.task,
                infos=results.info,
                client=client,
                model=model,
                sampling_args=gen_sampling_args,
                max_concurrent=gen_limit if gen_limit is not None else max_concurrent,
                **kwargs,
            )
            results.completion = [rollout[0] for rollout in rollouts]
            results.state = [rollout[1] for rollout in rollouts]
            if score_rollouts:
                rollout_scores = await self.rubric.score_rollouts(
                    prompts=results.prompt,
                    completions=results.completion,
                    answers=results.answer,
                    states=results.state,
                    tasks=results.task,
                    infos=results.info,
                    max_concurrent=score_limit
                    if score_limit is not None
                    else max_concurrent,
                )
                results.reward = rollout_scores.reward
                results.metrics = rollout_scores.metrics
            return results

    def generate(
        self,
        inputs: GenerateInputs | Dataset,
        client: AsyncOpenAI | OpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
        score_rollouts: bool = True,
        max_concurrent: int = -1,
        max_concurrent_generation: int | None = None,
        max_concurrent_scoring: int | None = None,
        interleave_scoring: bool = True,
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
            max_concurrent_generation,
            max_concurrent_scoring,
            interleave_scoring,
            **kwargs,
        )

        # check if we're in existing event loop (e.g. Jupyter)
        try:
            loop = asyncio.get_running_loop()
            import nest_asyncio  # type: ignore

            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        except RuntimeError:
            pass

        # script case: create new loop and executor
        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        loop = asyncio.new_event_loop()
        try:
            loop.set_default_executor(executor)
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)
            # shutdown the executor to prevent thread leaks
            executor.shutdown(wait=False)

    #########################################################
    # Helper functions for evaluation and dataset generation
    #########################################################

    def evaluate(
        self,
        client: AsyncOpenAI | OpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
        num_examples: int = -1,
        rollouts_per_example: int = 1,
        score_rollouts: bool = True,
        max_concurrent: int = -1,
        max_concurrent_generation: int | None = None,
        max_concurrent_scoring: int | None = None,
        interleave_scoring: bool = True,
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
            max_concurrent_generation,
            max_concurrent_scoring,
            interleave_scoring,
            **kwargs,
        )
        return results

    def make_dataset(
        self,
        results: GenerateOutputs,
        rollouts_per_example: int = 1,
        push_to_hf_hub: bool = False,
        hub_name: str | None = None,
        state_columns: list[str] | None = None,
        **kwargs,
    ) -> Dataset:
        """
        Make a dataset from the evaluation results.

        Args:
            results: The evaluation results to convert to a dataset
            rollouts_per_example: The number of rollouts per example
            push_to_hf_hub: Whether to push the dataset to the Hugging Face Hub
            hub_name: The name of the dataset on the Hugging Face Hub
            state_columns: List of state columns to include in the dataset
            concatenate_safe: Whether to ensure the dataset can be concatenated with others
                          by standardizing column types (useful for combining results from
                          different environments). Defaults to True for consistent schemas.
            **kwargs: Additional arguments passed to Dataset creation
        """
        # TODO: enable saving of multimodal datasets
        state_columns = state_columns or []

        if push_to_hf_hub and hub_name is None:
            raise ValueError("hub_name must be provided if push_to_hf_hub is True")

        cols = ["prompt", "completion", "answer", "task", "reward", "average_reward"]

        if rollouts_per_example > 1:
            average_reward = []
            for i in range(0, len(results.reward), rollouts_per_example):
                chunk = results.reward[i : i + rollouts_per_example]
                avg = sum(chunk) / len(chunk)
                average_reward.extend([avg] * rollouts_per_example)
        else:
            average_reward = results.reward

        results_dict = {
            "prompt": results.prompt,
            "completion": [],
            "answer": results.answer,
            "task": results.task,
            "reward": results.reward,
            "average_reward": average_reward,
        }
        if results.info[0] != {}:
            results_dict["info"] = results.info
            cols.append("info")
        for i in range(len(results.completion)):
            results_dict["completion"].append(
                sanitize_tool_calls(results.completion[i])
            )
        results_dict.update(results.metrics)
        cols.extend(results.metrics.keys())
        if results.state[0] is not None:
            for col in state_columns:
                if col in results.state[0]:
                    results_dict[col] = [state[col] for state in results.state]
                    cols.append(col)
                else:
                    self.logger.warning(
                        f"Column {col} not found in state, skipping from dataset."
                    )
        dataset = Dataset.from_dict({col: results_dict[col] for col in cols})
        if push_to_hf_hub:
            assert hub_name is not None
            dataset.push_to_hub(hub_name)
        return dataset

    #########################################################
    # Optional helper functions for parsing vLLM completions
    #########################################################

    def parse_chat_completion_logprobs(
        self, chat_completion: ChatCompletion
    ) -> list[float]:
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

    def parse_completion_logprobs(self, completion: Completion) -> list[float]:
        """Parses the completion logprobs from a vLLM chat completion"""
        assert len(completion.choices) == 1, "Response should always have one choice"
        assert completion.choices[0].logprobs is not None, (
            "Logprobs should not be None. Make sure to set logprobs=True in the extra body when making the request to /v1/completions"
        )
        assert completion.choices[0].logprobs.token_logprobs is not None, (
            "Logprob token_logprobs should not be None. Make sure to set logprobs=True in the extra body when making the request to /v1/completions"
        )
        return completion.choices[0].logprobs.token_logprobs

    def parse_chat_completion_tokens(
        self, chat_completion: ChatCompletion
    ) -> list[int]:
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
            # tokens are token_id:<int> because we request `return_tokens_as_token_ids` from vllm in GRPOTrainer
            int(token.token.split(":")[-1])
            for token in chat_completion.choices[0].logprobs.content
        ]
        return tokens

    def parse_completion_tokens(self, completion: Completion) -> list[int]:
        """Parses the output token ids from a list of chat completions returned by vLLM OAI server."""
        assert len(completion.choices) == 1, "Response should always have one choice"
        assert completion.choices[0].logprobs is not None, (
            "Logprobs should not be None. Make sure to set logprobs=True in the extra body when making the request to /v1/completions"
        )
        assert completion.choices[0].logprobs.tokens is not None, (
            "Logprob tokens should not be None. Make sure to set logprobs=True in the extra body when making the request to /v1/completions"
        )
        tokens = [
            # tokens are token_id:<int> because we request `return_tokens_as_token_ids` from vllm in GRPOTrainer
            int(token.split(":")[-1])
            for token in completion.choices[0].logprobs.tokens
        ]
        return tokens

    def process_chat_format_vllm(
        self,
        prompt: list[ChatMessage],
        completion: list[ChatMessage],
        state: State,
        processing_class: "PreTrainedTokenizerBase",
        mask_env_responses: bool = False,
    ) -> tuple[list[int], list[int], list[int], list[int], list[float]]:
        """
        Process chat format conversations using incremental prefixes.
        """
        responses = state["responses"]
        responses_idx = 0
        zipped = []
        for turn in completion:
            if turn["role"] == "assistant":
                zipped.append((turn, responses[responses_idx]))
                responses_idx += 1
            else:
                zipped.append((turn, None))
        assert len(responses) == responses_idx, "Responses not fully consumed"
        assert len(zipped) == len(completion), "Length mismatch"
        prompt_ids: list[int] = processing_class.apply_chat_template(
            conversation=prompt,  # type: ignore
            add_generation_prompt=True,
        )
        messages_consumed = [m for m in prompt]
        prompt_mask: list[int] = [0] * len(prompt_ids)
        completion_ids: list[int] = []
        completion_mask: list[int] = []
        completion_logprobs: list[float] = []
        i = 0
        while i < len(zipped):
            message, response = zipped[i]

            def deserialize_tool_calls(message: dict) -> dict:
                """
                Deserialize tool calls in messages, if any are present. Iterates
                over all messages in a message list and tries to find
                "tool_calls" key. If found, assumes it is a OAI format and has
                key "function" with "arguments" key which is stringified. It
                will then deserialize the argument so that chat tmeplates like
                Qwen3's can be used.
                """

                def deserialize_tool_call(tool_call) -> dict:
                    tool_call = dict(tool_call)
                    function = dict(tool_call["function"])
                    return {
                        **tool_call,
                        "function": {
                            **function,
                            "arguments": json.loads(function["arguments"]),
                        },
                    }

                return {
                    **message,
                    "tool_calls": [
                        deserialize_tool_call(tool_call)
                        for tool_call in message.get("tool_calls", []) or []
                    ],
                }

            message = deserialize_tool_calls(message)

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
                i += 1
            # user/tool case -- use message
            else:
                assert message["role"] == "user" or message["role"] == "tool"
                # Collect all consecutive non-assistant messages
                consecutive_messages = [message]
                j = i + 1
                while j < len(zipped) and zipped[j][0]["role"] != "assistant":
                    consecutive_messages.append(zipped[j][0])
                    j += 1
                token_prefix: list[int] = processing_class.apply_chat_template(
                    conversation=messages_consumed  # type: ignore
                )
                token_prefix_with_turn: list[int] = (
                    processing_class.apply_chat_template(
                        conversation=messages_consumed + consecutive_messages,  # type: ignore
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
                messages_consumed.extend(consecutive_messages)
                i = j
        return (
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
            completion_logprobs,
        )

    def process_completion_format_vllm(
        self,
        prompt: str,
        completion: str,
        state: State,
        processing_class: "PreTrainedTokenizerBase",
        mask_env_responses: bool = False,
    ) -> tuple[list[int], list[int], list[int], list[int], list[float]]:
        """
        Process completion format conversations using incremental prefixes.
        """
        responses: list[Completion] = state["responses"]
        responses_start_idx: list[int] = state["responses_start_idx"]
        assert len(responses) == len(responses_start_idx), (
            "Should have an index for each completion response"
        )

        idx = 0
        zipped: list[tuple[str, Completion | None]] = []
        for response, response_start_idx in zip(responses, responses_start_idx):
            if response_start_idx > idx:
                # non-model-generated section
                zipped.append((completion[idx:response_start_idx], None))
            response_text = response.choices[0].text or ""
            zipped.append((response_text, response))
            idx = response_start_idx + len(response_text)
        assert idx == len(completion), "Completion not fully consumed"

        prompt_ids: list[int] = processing_class.encode(prompt)
        rollout_consumed = prompt
        prompt_mask: list[int] = [0] * len(prompt_ids)
        completion_ids: list[int] = []
        completion_mask: list[int] = []
        completion_logprobs: list[float] = []
        i = 0
        while i < len(zipped):
            text, response = zipped[i]
            # model-generated case -- use response
            if response is not None:
                completion_turn_ids = self.parse_completion_tokens(response)
                completion_turn_mask = [1] * len(completion_turn_ids)
                completion_turn_logprobs = self.parse_completion_logprobs(response)
                completion_ids.extend(completion_turn_ids)
                completion_mask.extend(completion_turn_mask)
                completion_logprobs.extend(completion_turn_logprobs)
                rollout_consumed += text
                i += 1
            # non-model-generated (user/tool case) -- use text
            else:
                token_prefix: list[int] = processing_class.encode(rollout_consumed)
                token_prefix_with_turn: list[int] = processing_class.encode(
                    rollout_consumed + text
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
                rollout_consumed += text
                i += 1
        return (
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
            completion_logprobs,
        )

    def process_env_results_vllm(
        self,
        prompts: list[Messages],
        completions: list[Messages],
        states: list[State],
        rewards: list[float],
        processing_class: "PreTrainedTokenizerBase",
        max_seq_len: int = -1,
        mask_env_responses: bool = False,
        mask_truncated_completions: bool = False,
        zero_truncated_completions: bool = False,
    ) -> ProcessedOutputs:
        """
        Process results with vLLM tokens/logprobs.
        """
        is_chat_format = self.message_type == "chat"

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
                (
                    prompt_ids,
                    prompt_mask,
                    completion_ids,
                    completion_mask,
                    completion_logprobs,
                ) = self.process_completion_format_vllm(
                    prompt, completion, state, processing_class, mask_env_responses
                )
            is_truncated = False
            if max_seq_len > 0 and len(prompt_ids) + len(completion_ids) > max_seq_len:
                if len(prompt_ids) > max_seq_len:
                    prompt_ids = prompt_ids[:max_seq_len]
                    prompt_mask = prompt_mask[:max_seq_len]
                completion_ids = completion_ids[: max_seq_len - len(prompt_ids)]
                completion_mask = completion_mask[: max_seq_len - len(prompt_ids)]
                completion_logprobs = completion_logprobs[
                    : max_seq_len - len(prompt_ids)
                ]
                is_truncated = True
            if is_truncated and mask_truncated_completions:
                completion_mask = [0] * len(completion_ids)
            assert len(prompt_ids) == len(prompt_mask), (
                f"Prompt ids: {len(prompt_ids)}, prompt mask: {len(prompt_mask)}"
            )
            assert len(completion_ids) == len(completion_mask), (
                f"Completion ids: {len(completion_ids)}, completion mask: {len(completion_mask)}"
            )
            assert (
                len(completion_mask) == len(completion_ids) == len(completion_logprobs)
            ), (
                f"completion mask: {len(completion_mask)}, completion ids: {len(completion_ids)}, completion logprobs: {len(completion_logprobs)}"
            )
            all_prompt_ids.append(prompt_ids)
            all_prompt_masks.append(prompt_mask)
            all_completion_ids.append(completion_ids)
            all_completion_masks.append(completion_mask)
            all_completion_logprobs.append(completion_logprobs)
            if zero_truncated_completions and is_truncated:
                all_rewards.append(0)
            else:
                all_rewards.append(reward)
        return ProcessedOutputs(
            prompt_ids=all_prompt_ids,
            prompt_mask=all_prompt_masks,
            completion_ids=all_completion_ids,
            completion_mask=all_completion_masks,
            completion_logprobs=all_completion_logprobs,
            rewards=all_rewards,
        )

    # alias for process_env_results_vllm
    process_env_results = process_env_results_vllm
