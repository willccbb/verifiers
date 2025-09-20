"""Tests for the JudgeRubric class."""

import pytest
from types import SimpleNamespace
from pydantic import BaseModel, Field
from verifiers import JudgeRubric, Parser
from tests.mock_openai_client import MockOpenAIClient, MockCompletionResponse


@pytest.mark.asyncio
async def test_judge_with_string_prompt_returns_yes():
    """Judge should return mock 'yes' when question pattern matches."""
    client = MockOpenAIClient(chat_responses={"ID-STRING": "yes"})
    rubric = JudgeRubric(parser=Parser(), judge_client=client, judge_model="test-model")

    prompt = "What is 1+1? ID-STRING"
    completion = "2"
    answer = "2"
    state = {}

    result = await rubric.judge(
        prompt=prompt, completion=completion, answer=answer, state=state
    )

    assert result == "yes"


@pytest.mark.asyncio
async def test_judge_with_list_prompt_returns_no():
    """Judge should return mock 'no' when list prompt pattern matches last message."""
    client = MockOpenAIClient(chat_responses={"ID-LIST": "no"})
    rubric = JudgeRubric(parser=Parser(), judge_client=client, judge_model="test-model")

    prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 3+3? ID-LIST"},
    ]
    completion = "6"
    answer = "6"
    state = {}

    result = await rubric.judge(
        prompt=prompt, completion=completion, answer=answer, state=state
    )

    assert result == "no"


@pytest.mark.asyncio
async def test_judge_response_is_cached_by_prompt():
    """Second call with identical prompt should hit cache and avoid client call."""
    client = MockOpenAIClient(chat_responses={"CACHE-ID": "yes"})
    rubric = JudgeRubric(parser=Parser(), judge_client=client, judge_model="test-model")

    prompt = "Simple Q CACHE-ID"
    completion = "A"
    answer = "A"
    state = {}

    first = await rubric.judge(
        prompt=prompt, completion=completion, answer=answer, state=state
    )
    second = await rubric.judge(
        prompt=prompt, completion=completion, answer=answer, state=state
    )

    assert first == "yes"
    assert second == "yes"
    # Only the first call should hit the client
    assert client.chat.completions.call_count == 1


@pytest.mark.asyncio
async def test_judge_sampling_args_normalization_passes_max_completion_tokens(
    monkeypatch,
):
    """max_tokens should be renamed to max_completion_tokens and None-valued args removed."""

    class RecordingClient:
        def __init__(self):
            self.last_kwargs = None
            self.chat = SimpleNamespace()
            self.chat.completions = SimpleNamespace()

            def create(model, messages, **kwargs):
                self.last_kwargs = kwargs
                return MockCompletionResponse("yes")

            self.chat.completions.create = create

    client = RecordingClient()
    rubric = JudgeRubric(
        parser=Parser(),
        judge_client=client,  # type: ignore[arg-type]
        judge_model="test-model",
        judge_sampling_args={"max_tokens": 5, "temperature": 0.1, "top_p": None},
    )

    prompt = "P"
    completion = "C"
    answer = "A"
    state = {}

    result = await rubric.judge(
        prompt=prompt, completion=completion, answer=answer, state=state
    )

    assert result == "yes"
    assert client.last_kwargs is not None
    # None-valued args should be removed
    assert "top_p" not in client.last_kwargs
    # max_tokens should be converted to max_completion_tokens
    assert "max_tokens" not in client.last_kwargs
    assert client.last_kwargs.get("max_completion_tokens") == 5


@pytest.mark.asyncio
async def test_response_format_validation_accepts_pydantic_model():
    class JudgeResponse(BaseModel):
        description: str = Field(description="Description of the response")
        ok: bool = Field(description="Whether the response is OK")

    mock_response = JudgeResponse(description="This is a sample response", ok=True)

    client = MockOpenAIClient(
        chat_responses={
            "LLM-RESPONSE": mock_response,
        }
    )
    rubric = JudgeRubric(
        parser=Parser(),
        judge_client=client,
        judge_model="test-model",
        judge_sampling_args={"response_format": JudgeResponse},
    )

    prompt = "P"
    completion = "LLM-RESPONSE"
    answer = "A"
    state = {}

    result = await rubric.judge(
        prompt=prompt, completion=completion, answer=answer, state=state
    )

    assert result == mock_response


@pytest.mark.asyncio
async def test_judge_error_handling_rate_limit(monkeypatch):
    """Rate limit errors should be converted to a RuntimeError with helpful message."""

    class FakeRateLimitError(Exception):
        pass

    client = MockOpenAIClient()

    def raise_rate_limit(model, messages, **kwargs):  # noqa: ARG001
        raise FakeRateLimitError("over limit")

    client.chat.completions.create = raise_rate_limit
    # Patch the exception type checked inside the module
    monkeypatch.setattr(
        "verifiers.rubrics.judge_rubric.RateLimitError",
        FakeRateLimitError,
        raising=True,
    )

    rubric = JudgeRubric(parser=Parser(), judge_client=client, judge_model="test-model")

    with pytest.raises(RuntimeError, match="rate limit exceeded"):
        await rubric.judge(prompt="Q", completion="A", answer="A", state={})


@pytest.mark.asyncio
async def test_judge_error_handling_timeout(monkeypatch):
    """Timeout errors should be converted to a RuntimeError with helpful message."""

    class FakeTimeoutError(Exception):
        pass

    client = MockOpenAIClient()

    def raise_timeout(model, messages, **kwargs):  # noqa: ARG001
        raise FakeTimeoutError("timeout")

    client.chat.completions.create = raise_timeout
    monkeypatch.setattr(
        "verifiers.rubrics.judge_rubric.APITimeoutError", FakeTimeoutError, raising=True
    )

    rubric = JudgeRubric(parser=Parser(), judge_client=client, judge_model="test-model")

    with pytest.raises(RuntimeError, match="timeout"):
        await rubric.judge(prompt="Q", completion="A", answer="A", state={})


@pytest.mark.asyncio
async def test_judge_error_handling_api_error(monkeypatch):
    """API errors should be converted to a RuntimeError with helpful message."""

    class FakeAPIError(Exception):
        pass

    client = MockOpenAIClient()

    def raise_api_error(model, messages, **kwargs):  # noqa: ARG001
        raise FakeAPIError("api error")

    client.chat.completions.create = raise_api_error
    monkeypatch.setattr(
        "verifiers.rubrics.judge_rubric.APIError", FakeAPIError, raising=True
    )

    rubric = JudgeRubric(parser=Parser(), judge_client=client, judge_model="test-model")

    with pytest.raises(RuntimeError, match="API error"):
        await rubric.judge(prompt="Q", completion="A", answer="A", state={})


@pytest.mark.asyncio
async def test_judge_error_handling_generic_exception():
    """Generic exceptions should be converted to a RuntimeError with generic message."""

    client = MockOpenAIClient()

    def raise_generic(model, messages, **kwargs):  # noqa: ARG001
        raise Exception("boom")

    client.chat.completions.create = raise_generic

    rubric = JudgeRubric(parser=Parser(), judge_client=client, judge_model="test-model")

    with pytest.raises(RuntimeError, match="Unexpected error"):
        await rubric.judge(prompt="Q", completion="A", answer="A", state={})
