"""Tests for the base Environment class."""

import json
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from datasets import Dataset
from openai import BadRequestError

from verifiers import Environment, Parser, Rubric, ThinkParser
from verifiers.types import ChatCompletion, Completion, GenerateOutputs, RolloutScores


# Create a concrete implementation for testing the abstract base class
class SimpleEnvironment(Environment):
    """Simple implementation of Environment for testing."""

    async def rollout(
        self,
        client,
        model,
        prompt,
        answer: str = "",
        task: str = "default",
        info: dict | None = None,
        sampling_args: dict | None = None,
        **kwargs,
    ):
        """Simple test rollout implementation."""
        response = await self.get_model_response(
            prompt=prompt, client=client, model=model, sampling_args=sampling_args or {}
        )
        if self.message_type == "chat":
            assert isinstance(response, ChatCompletion)
            completion = [
                {"role": "assistant", "content": response.choices[0].message.content}
            ]
            state = {"responses": [response]}
        else:
            assert isinstance(response, Completion)
            completion = response.choices[0].text
            state = {}
        return completion, state


class TestEnvironmentBase:
    """Test cases for the base Environment class."""

    def test_environment_initialization(self, mock_openai_client, sample_dataset):
        """Test that Environment initializes correctly."""
        env = SimpleEnvironment(
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )
        assert env.message_type == "chat"
        assert isinstance(env.parser, Parser)
        assert isinstance(env.rubric, Rubric)

    def test_environment_with_eval_dataset_only(
        self, mock_openai_client, sample_dataset
    ):
        """Test Environment with only eval_dataset."""
        env = SimpleEnvironment(
            eval_dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )
        assert env.dataset is None
        assert env.eval_dataset is not None

    def test_environment_no_datasets_raises_error(self, mock_openai_client):
        """Test that Environment raises error when no datasets provided."""
        with pytest.raises(
            ValueError, match="Either dataset or eval_dataset must be provided"
        ):
            SimpleEnvironment(
                client=mock_openai_client,
                model="test-model",
                parser=Parser(),
                rubric=Rubric(),
            )

    def test_completion_mode_with_system_prompt_raises_error(
        self, mock_openai_client, sample_dataset
    ):
        """Test that completion mode with system prompt raises error."""
        with pytest.raises(ValueError, match="not supported for completion tasks"):
            SimpleEnvironment(
                dataset=sample_dataset,
                message_type="completion",
                system_prompt="test prompt",
                parser=Parser(),
                rubric=Rubric(),
            )

    def test_different_parser_rubric_parser_warns(
        self, mock_openai_client, sample_dataset
    ):
        """Test that warning is logged when parser and rubric parser are different."""
        from unittest.mock import Mock, patch

        think_parser = ThinkParser()
        rubric = Rubric()  # Different parser class

        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            _ = SimpleEnvironment(
                client=mock_openai_client,
                model="test-model",
                dataset=sample_dataset,
                parser=think_parser,
                rubric=rubric,
            )

            mock_logger.warning.assert_called_once_with(
                "The parser and rubric parser are different. This may cause unexpected behavior."
            )

    def test_format_prompt(self, mock_openai_client, sample_dataset):
        """Test prompt formatting."""
        env = SimpleEnvironment(
            client=mock_openai_client,
            model="test-model",
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )

        prompt = "What is 2+2?"
        system_prompt = "You are a helpful assistant."
        few_shot = [
            {"role": "user", "content": "What is 1+1?"},
            {"role": "assistant", "content": "2"},
        ]

        formatted = env.format_prompt(prompt, system_prompt, few_shot)

        assert len(formatted) == 4
        assert formatted[0]["role"] == "system"
        assert formatted[0]["content"] == system_prompt
        assert formatted[1]["role"] == "user"
        assert formatted[1]["content"] == "What is 1+1?"
        assert formatted[2]["role"] == "assistant"
        assert formatted[2]["content"] == "2"
        assert formatted[3]["role"] == "user"
        assert formatted[3]["content"] == prompt

    def test_get_dataset(self, mock_openai_client, sample_dataset):
        """Test dataset retrieval."""
        env = SimpleEnvironment(
            client=mock_openai_client,
            model="test-model",
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )

        # Get full dataset
        full_dataset = env.get_dataset()
        assert len(full_dataset) == 2

        # Get subset
        subset = env.get_dataset(n=1)
        assert len(subset) == 1

    @pytest.mark.asyncio
    async def test_get_model_response_chat(self, mock_openai_client):
        """Test get_model_response with chat format."""
        env = SimpleEnvironment(
            client=mock_openai_client,
            model="test-model",
            eval_dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            parser=Parser(),
            rubric=Rubric(),
        )

        prompt = [{"role": "user", "content": "Hello"}]
        response = await env.get_model_response(
            prompt=prompt,
            client=mock_openai_client,
            model="test-model",
            message_type="chat",
        )

        # Check response structure
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert hasattr(response.choices[0], "message")
        assert hasattr(response.choices[0].message, "content")
        mock_openai_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_model_response_completion(self, mock_openai_client):
        """Test get_model_response with completion format."""
        env = SimpleEnvironment(
            client=mock_openai_client,
            model="test-model",
            eval_dataset=Dataset.from_dict({"prompt": ["test"], "answer": ["test"]}),
            message_type="completion",
            parser=Parser(),
            rubric=Rubric(),
        )

        prompt = "Complete this:"
        response = await env.get_model_response(
            prompt=prompt,
            client=mock_openai_client,
            model="test-model",
            message_type="completion",
        )

        # Check response structure
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert hasattr(response.choices[0], "text")
        mock_openai_client.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_model_response_chat_context_length_guard(
        self, mock_openai_client
    ):
        """Ensure chat responses gracefully handle context length errors."""

        mock_openai_client.base_url = "https://api.openai.com/v1"
        env = SimpleEnvironment(
            client=mock_openai_client,
            model="gpt-test",
            eval_dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            parser=Parser(),
            rubric=Rubric(),
        )

        error_message = (
            "This model's maximum context length is 4097 tokens. "
            "However, you requested 4120 tokens (4020 in the messages, 100 in the completion). "
            "Please reduce the length of the messages or completion."
        )
        error_body = {
            "error": {
                "message": error_message,
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
            }
        }
        mock_openai_client.chat.completions.create.side_effect = BadRequestError(
            message=f"Error code: 400 - {json.dumps(error_body)}",
            response=httpx.Response(
                status_code=400,
                request=httpx.Request(
                    "POST", "https://api.openai.com/v1/chat/completions"
                ),
            ),
            body=error_body,
        )

        prompt = [{"role": "user", "content": "Hello"}]
        with patch.object(env.logger, "warning") as mock_warning:
            response = await env.get_model_response(
                prompt=prompt,
                client=mock_openai_client,
                model="gpt-test",
                message_type="chat",
            )

        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content == ""
        assert response.choices[0].finish_reason == "length"
        assert response.usage.prompt_tokens == 4020
        assert response.usage.completion_tokens == 100
        assert response.usage.total_tokens == 4120
        assert mock_openai_client.chat.completions.create.await_count == 1
        mock_warning.assert_called_once()
        warning_text = mock_warning.call_args.args[0]
        assert "Context length exceeded" in warning_text
        assert "requested 4120 tokens" in warning_text

    @pytest.mark.asyncio
    async def test_get_model_response_completion_context_length_guard_vllm(
        self, mock_openai_client
    ):
        """Ensure completion responses handle vLLM-style context length errors."""

        mock_openai_client.base_url = "http://localhost:8000/v1"
        env = SimpleEnvironment(
            client=mock_openai_client,
            model="vllm-model",
            eval_dataset=Dataset.from_dict({"prompt": ["test"], "answer": ["test"]}),
            message_type="completion",
            parser=Parser(),
            rubric=Rubric(),
        )

        error_message = (
            "'max_tokens' or 'max_completion_tokens' is too large: 400. "
            "This model's maximum context length is 8192 tokens and your request has "
            "8000 input tokens (400 > 8192 - 8000)."
        )
        error_body = {
            "error": {
                "message": error_message,
                "type": "context_length_exceeded",
                "code": "context_length_exceeded",
            }
        }
        mock_openai_client.completions.create.side_effect = BadRequestError(
            message=f"Error code: 400 - {json.dumps(error_body)}",
            response=httpx.Response(
                status_code=400,
                request=httpx.Request("POST", "http://localhost:8000/v1/completions"),
            ),
            body=error_body,
        )

        with patch.object(env.logger, "warning") as mock_warning:
            response = await env.get_model_response(
                prompt="Hello",
                client=mock_openai_client,
                model="vllm-model",
                message_type="completion",
            )

        assert isinstance(response, Completion)
        assert response.choices[0].text == ""
        assert response.choices[0].finish_reason == "length"
        assert response.usage.prompt_tokens == 8000
        assert response.usage.completion_tokens == 400
        assert response.usage.total_tokens == 8400
        assert mock_openai_client.completions.create.await_count == 1
        mock_warning.assert_called_once()
        warning_text = mock_warning.call_args.args[0]
        assert "Context length exceeded" in warning_text
        assert "limit 8192" in warning_text
        assert "over by 208 tokens" in warning_text

    @pytest.mark.asyncio
    async def test_get_model_response_completion_context_length_guard_vllm_prompt(
        self, mock_openai_client
    ):
        """Prompt-only vLLM context limits should gracefully truncate."""

        mock_openai_client.base_url = "http://localhost:8000/v1"
        env = SimpleEnvironment(
            client=mock_openai_client,
            model="vllm-model",
            eval_dataset=Dataset.from_dict({"prompt": ["test"], "answer": ["test"]}),
            message_type="completion",
            parser=Parser(),
            rubric=Rubric(),
        )

        error_message = (
            "This model's maximum context length is 8192 tokens. "
            "However, your request has 9000 input tokens. "
            "Please reduce the length of the input messages."
        )
        error_body = {
            "error": {
                "message": error_message,
                "type": "context_length_exceeded",
                "code": "context_length_exceeded",
            }
        }
        mock_openai_client.completions.create.side_effect = BadRequestError(
            message=f"Error code: 400 - {json.dumps(error_body)}",
            response=httpx.Response(
                status_code=400,
                request=httpx.Request("POST", "http://localhost:8000/v1/completions"),
            ),
            body=error_body,
        )

        with patch.object(env.logger, "warning") as mock_warning:
            response = await env.get_model_response(
                prompt="Hello",
                client=mock_openai_client,
                model="vllm-model",
                message_type="completion",
            )

        assert isinstance(response, Completion)
        assert response.choices[0].text == ""
        assert response.choices[0].finish_reason == "length"
        assert response.usage.prompt_tokens == 9000
        assert response.usage.completion_tokens == 0
        assert response.usage.total_tokens == 9000
        assert mock_openai_client.completions.create.await_count == 1
        mock_warning.assert_called_once()
        warning_text = mock_warning.call_args.args[0]
        assert "Context length exceeded" in warning_text
        assert "limit 8192" in warning_text
        assert "prompt 9000 tokens" in warning_text

    def test_process_chat_format(self, mock_openai_client, sample_dataset):
        """Test processing chat format conversations."""
        env = SimpleEnvironment(
            client=mock_openai_client,
            model="test-model",
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )

        # Create a mock tokenizer
        mock_tokenizer = Mock()

        def apply_template(conversation, tokenize=False, add_generation_prompt=True):
            # Return deterministic token ids ensuring prefix property
            return list(range(10 if add_generation_prompt else 14))

        mock_tokenizer.apply_chat_template = Mock(side_effect=apply_template)
        mock_tokenizer.encode = Mock(side_effect=lambda text: list(range(len(text))))

        prompt = [{"role": "user", "content": "What is 2+2?"}]
        completion = [{"role": "assistant", "content": "4"}]
        # Minimal vLLM-style chat completion with tokens/logprobs
        token_entries = [
            Mock(logprob=-0.1, token="token_id:11"),
            Mock(logprob=-0.2, token="token_id:12"),
            Mock(logprob=-0.3, token="token_id:13"),
            Mock(logprob=-0.4, token="token_id:14"),
        ]
        mock_choice = Mock()
        mock_choice.logprobs = Mock()
        mock_choice.logprobs.content = token_entries
        mock_chat_completion = Mock()
        mock_chat_completion.choices = [mock_choice]
        state = {"responses": [mock_chat_completion]}

        (
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
            completion_logprobs,
        ) = env.process_chat_format_vllm(
            prompt, completion, state, mock_tokenizer, mask_env_responses=False
        )

        assert isinstance(prompt_ids, list)
        assert isinstance(prompt_mask, list)
        assert isinstance(completion_ids, list)
        assert isinstance(completion_mask, list)
        assert len(prompt_ids) == len(prompt_mask)
        assert len(completion_ids) == len(completion_mask)
        assert all(m == 0 for m in prompt_mask)  # Prompt mask should be all 0s
        assert all(m == 1 for m in completion_mask)  # Completion mask should be all 1s

    def test_process_completion_format(self, mock_openai_client, sample_dataset):
        """Test processing completion format text."""
        env = SimpleEnvironment(
            client=mock_openai_client,
            model="test-model",
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )

        # Create a mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode = Mock(side_effect=lambda text: list(range(len(text))))

        prompt = "Complete this: 2+2="
        completion = "4"
        # Minimal vLLM-style completion covering entire completion string
        mock_choice = Mock()
        mock_choice.text = completion
        mock_choice.logprobs = Mock()
        mock_choice.logprobs.tokens = ["token_id:1"] * len(completion)
        mock_choice.logprobs.token_logprobs = [-0.1] * len(completion)
        mock_completion = Mock()
        mock_completion.choices = [mock_choice]
        state = {"responses": [mock_completion], "responses_start_idx": [0]}

        (
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
            completion_logprobs,
        ) = env.process_completion_format_vllm(
            prompt, completion, state, mock_tokenizer
        )

        assert isinstance(prompt_ids, list)
        assert isinstance(prompt_mask, list)
        assert isinstance(completion_ids, list)
        assert isinstance(completion_mask, list)
        assert len(prompt_ids) == len(prompt)
        assert len(completion_ids) == len(completion)
        assert all(m == 0 for m in prompt_mask)
        assert all(m == 1 for m in completion_mask)

    def test_process_env_results_chat(self, mock_openai_client, sample_dataset):
        """Test processing environment results for chat format."""
        env = SimpleEnvironment(
            client=mock_openai_client,
            model="test-model",
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )

        # Create a mock tokenizer
        mock_tokenizer = Mock()

        # Track the conversation state
        def mock_apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        ):
            # Return token id list; size scales with number of messages
            return list(range(len(conversation) * 5))

        def mock_encode(text, **kwargs):
            return list(range(3))

        mock_tokenizer.apply_chat_template = Mock(side_effect=mock_apply_chat_template)
        mock_tokenizer.encode = Mock(side_effect=mock_encode)

        prompts = [[{"role": "user", "content": "Hello"}]]
        completions = [[{"role": "assistant", "content": "Hi there!"}]]
        # Minimal vLLM-style chat completion mock for assistant turn
        token_entries = [
            Mock(logprob=-0.1, token="token_id:101"),
            Mock(logprob=-0.2, token="token_id:102"),
            Mock(logprob=-0.3, token="token_id:103"),
        ]
        mock_choice = Mock()
        mock_choice.logprobs = Mock()
        mock_choice.logprobs.content = token_entries
        mock_chat_completion = Mock()
        mock_chat_completion.choices = [mock_choice]
        states = [{"responses": [mock_chat_completion]}]
        rewards = [1.0]

        results = env.process_env_results_vllm(
            prompts, completions, states, rewards, mock_tokenizer
        )

        assert hasattr(results, "prompt_ids")
        assert hasattr(results, "prompt_mask")
        assert hasattr(results, "completion_ids")
        assert hasattr(results, "completion_mask")
        assert hasattr(results, "completion_logprobs")
        assert hasattr(results, "rewards")
        assert len(results.rewards) == 1
        assert results.rewards[0] == 1.0

    def test_process_env_results_with_truncation(
        self, mock_openai_client, sample_dataset
    ):
        """Test processing environment results with sequence length truncation."""
        env = SimpleEnvironment(
            client=mock_openai_client,
            model="test-model",
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )

        # Create a mock tokenizer
        mock_tokenizer = Mock()

        # Track the conversation state
        def mock_apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        ):
            # Return deterministic token ids based on number of messages
            # Ensures prefix property when conversation grows
            return list(range(len(conversation) * 5))

        def mock_encode(text, **kwargs):
            # Not used by chat path; keep for compatibility
            return list(range(3))

        mock_tokenizer.apply_chat_template = Mock(side_effect=mock_apply_chat_template)
        mock_tokenizer.encode = Mock(side_effect=mock_encode)

        prompts = [[{"role": "user", "content": "Hello"}]]
        completions = [[{"role": "assistant", "content": "Hi there!"}]]
        # Produce enough assistant tokens to force truncation
        token_entries = [
            Mock(logprob=-0.1, token="token_id:201"),
            Mock(logprob=-0.2, token="token_id:202"),
            Mock(logprob=-0.3, token="token_id:203"),
            Mock(logprob=-0.4, token="token_id:204"),
            Mock(logprob=-0.5, token="token_id:205"),
            Mock(logprob=-0.6, token="token_id:206"),
        ]
        mock_choice = Mock()
        mock_choice.logprobs = Mock()
        mock_choice.logprobs.content = token_entries
        mock_chat_completion = Mock()
        mock_chat_completion.choices = [mock_choice]
        states = [{"responses": [mock_chat_completion]}]
        rewards = [1.0]

        results = env.process_env_results_vllm(
            prompts,
            completions,
            states,
            rewards,
            mock_tokenizer,
            max_seq_len=8,  # Force truncation
            mask_truncated_completions=True,
        )

        # Check that total length respects max_seq_len
        total_len = len(results.prompt_ids[0]) + len(results.completion_ids[0])
        assert total_len <= 8
        # Check that truncated completion is masked
        assert all(m == 0 for m in results.completion_mask[0])

    def test_parse_chat_completion_logprobs(self, mock_openai_client, sample_dataset):
        """Test parsing logprobs from a vLLM chat completion."""
        env = SimpleEnvironment(
            client=mock_openai_client,
            model="test-model",
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )

        # Create mock chat completion with logprobs
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].logprobs = Mock()
        mock_completion.choices[0].logprobs.content = [
            Mock(logprob=-0.5),
            Mock(logprob=-1.2),
            Mock(logprob=-0.3),
        ]

        logprobs = env.parse_chat_completion_logprobs(mock_completion)
        assert logprobs == [-0.5, -1.2, -0.3]

    def test_parse_chat_completion_tokens(self, mock_openai_client, sample_dataset):
        """Test parsing tokens from a vLLM chat completion."""
        env = SimpleEnvironment(
            client=mock_openai_client,
            model="test-model",
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )

        # Create mock chat completion with tokens
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].logprobs = Mock()
        mock_completion.choices[0].logprobs.content = [
            Mock(token="id:1234"),
            Mock(token="id:5678"),
            Mock(token="id:9012"),
        ]

        tokens = env.parse_chat_completion_tokens(mock_completion)
        assert tokens == [1234, 5678, 9012]

    @pytest.mark.asyncio
    async def test_run_rollouts(self, mock_openai_client):
        """Test running multiple rollouts."""
        env = SimpleEnvironment(
            client=mock_openai_client,
            model="test-model",
            eval_dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            parser=Parser(),
            rubric=Rubric(),
        )

        prompts = [
            [{"role": "user", "content": "Hello"}],
            [{"role": "user", "content": "Hi"}],
        ]
        answers = ["response1", "response2"]
        tasks = ["default", "default"]
        infos = [{}, {}]

        # Mock the rollout method calls
        results = await env.run_rollouts(
            client=mock_openai_client,
            model="test-model",
            prompts=prompts,
            answers=answers,
            tasks=tasks,
            infos=infos,
        )

        assert len(results) == 2
        assert all(
            len(result) == 2 for result in results
        )  # Each result is (completion, state)

    @pytest.mark.asyncio
    async def test_a_generate_with_score_rollouts(
        self, mock_openai_client, sample_dataset
    ):
        """Test async generate with scoring enabled."""
        env = SimpleEnvironment(
            client=mock_openai_client,
            model="test-model",
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )

        # Mock the rubric scoring
        env.rubric.score_rollouts = AsyncMock(  # type: ignore[attr-defined]
            return_value=RolloutScores(reward=[1.0], metrics={})
        )

        inputs = {"prompt": [[{"role": "user", "content": "Hello"}]], "answer": ["Hi"]}

        results = await env.a_generate(
            inputs,
            client=mock_openai_client,
            model="test-model",
            score_rollouts=True,
            interleave_scoring=False,
        )

        assert hasattr(results, "completion")
        assert hasattr(results, "state")
        assert hasattr(results, "reward")
        assert results.reward == [1.0]

    def test_generate_sync_wrapper(self, mock_openai_client, sample_dataset):
        """Test synchronous generate wrapper."""
        env = SimpleEnvironment(
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )

        # Mock the rubric scoring
        env.rubric.score_rollouts = AsyncMock(  # type: ignore[attr-defined]
            return_value=RolloutScores(reward=[1.0], metrics={})
        )

        from verifiers.types import GenerateInputs

        gi = GenerateInputs(
            prompt=[[{"role": "user", "content": "Hello"}]], answer=["Hi"]
        )  # type: ignore[arg-type]
        results = env.generate(
            gi,
            client=mock_openai_client,
            model="test-model",
            interleave_scoring=False,
        )

        assert hasattr(results, "completion")
        assert hasattr(results, "state")
        assert hasattr(results, "reward")

    def test_make_dataset(self, mock_openai_client, sample_dataset):
        """Test creating a dataset from evaluation results."""
        env = SimpleEnvironment(
            client=mock_openai_client,
            model="test-model",
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )

        results = GenerateOutputs(
            prompt=[[{"role": "user", "content": "Hello"}]],
            completion=[[{"role": "assistant", "content": "Hi"}]],
            answer=["Hi"],
            reward=[1.0],
            task=["default"],
            state=[{"custom_field": "value"}],
            info=[{}],
            metrics={},
        )

        dataset = env.make_dataset(results, state_columns=["custom_field"])

        assert len(dataset) == 1
        assert "prompt" in dataset.column_names
        assert "completion" in dataset.column_names
        assert "answer" in dataset.column_names
        assert "reward" in dataset.column_names
        assert "task" in dataset.column_names
        assert "custom_field" in dataset.column_names
