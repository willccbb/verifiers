import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from verifiers.inference.vllm_client import VLLMClient

# Note: These tests may produce a `RuntimeWarning: coroutine 'AsyncClient.aclose' was never awaited`.
# This is a known issue related to the test environment's cleanup of the underlying httpx client.
# The tests pass and correctly verify the logic, so this warning can be safely ignored.

class MockTokenizer:
    def encode(self, text):
        # Return token count equal to the length of text for simplicity
        return list(range(len(text)))
    def apply_chat_template(self, messages, tokenize=False):
        # Join all message contents for tokenization
        return " ".join([m.get("content", "") for m in messages])

@pytest.mark.asyncio
async def test_max_tokens_none_uses_full_context():
    with patch("requests.Session.get") as mock_get, \
         patch.object(VLLMClient, "check_server", return_value=None), \
         patch("openai.resources.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock_create:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"max_model_len": 2000}
        mock_create.return_value = "ok"

        tokenizer = MockTokenizer()
        client = VLLMClient(tokenizer=tokenizer)

        # Prompt of length 100, so available tokens = 2000 - 100 = 1900
        prompt = "a" * 100
        result = await client.completions_create(model="test-model", prompt=prompt, max_tokens=None)

        mock_create.assert_awaited_once()
        args, kwargs = mock_create.call_args
        assert kwargs["max_tokens"] == 1900
        assert result == "ok"

@pytest.mark.asyncio
async def test_error_when_prompt_exceeds_context():
    with patch("requests.Session.get") as mock_get, \
         patch.object(VLLMClient, "check_server", return_value=None):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"max_model_len": 50}

        tokenizer = MockTokenizer()
        client = VLLMClient(tokenizer=tokenizer)

        # Prompt of length 60, exceeds max_model_len
        with pytest.raises(ValueError, match="Prompt exceeds model's max_model_len"):
            await client.completions_create(model="test-model", prompt="a" * 60, max_tokens=None)

@pytest.mark.asyncio
async def test_stability_various_prompt_lengths():
    with patch("requests.Session.get") as mock_get, \
         patch.object(VLLMClient, "check_server", return_value=None), \
         patch("openai.resources.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock_create:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"max_model_len": 100}
        mock_create.return_value = "ok"

        tokenizer = MockTokenizer()
        client = VLLMClient(tokenizer=tokenizer)

        # Prompt of length 0
        await client.completions_create(model="test-model", prompt="", max_tokens=None)
        args, kwargs = mock_create.call_args
        assert kwargs["max_tokens"] == 100
        mock_create.reset_mock()

        # Prompt of length 99
        await client.completions_create(model="test-model", prompt="a" * 99, max_tokens=None)
        args, kwargs = mock_create.call_args
        assert kwargs["max_tokens"] == 1
        mock_create.reset_mock()

        # Prompt of length 100 (should raise error)
        with pytest.raises(ValueError):
            await client.completions_create(model="test-model", prompt="a" * 100, max_tokens=None)
        mock_create.assert_not_called()

@pytest.mark.asyncio
async def test_explicit_max_tokens_respected():
    with patch("requests.Session.get") as mock_get, \
         patch.object(VLLMClient, "check_server", return_value=None), \
         patch("openai.resources.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock_create:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"max_model_len": 100}
        mock_create.return_value = "ok"

        tokenizer = MockTokenizer()
        client = VLLMClient(tokenizer=tokenizer)

        # Explicit max_tokens should be used as-is
        await client.completions_create(model="test-model", prompt="a" * 10, max_tokens=5)
        args, kwargs = mock_create.call_args
        assert kwargs["max_tokens"] == 5

@pytest.mark.asyncio
async def test_vllm_client_with_mock_openai_client(mock_openai_client):
    # This test simulates using the VLLMClient with a mock OpenAI environment.
    # We patch the underlying `create` call to prevent real network requests.
    with patch("requests.Session.get") as mock_get, \
         patch.object(VLLMClient, "check_server", return_value=None), \
         patch("openai.resources.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock_create:

        # Configure server and tokenizer mocks
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"max_model_len": 128}

        # Use the mock_openai_client's behavior for the patched method
        mock_create.side_effect = mock_openai_client.completions.create

        tokenizer = MockTokenizer()
        client = VLLMClient(tokenizer=tokenizer)

        prompt = "hello world"
        result = await client.completions_create(model="test-model", prompt=prompt, max_tokens=None)

        # Verify the mock was called with the correct, calculated max_tokens
        mock_create.assert_awaited_once()
        args, kwargs = mock_create.call_args
        assert kwargs["max_tokens"] == 117  # 128 - len("hello world")

        # Verify the result is the mocked response from mock_openai_client
        assert result is not None
