import pytest
from unittest.mock import AsyncMock, patch
from openai import NOT_GIVEN

from verifiers.inference.vllm_client import VLLMClient

class MockTokenizer:
    def encode(self, text):
        # Return token count equal to the length of text for simplicity
        return list(range(len(text)))
    def apply_chat_template(self, messages, tokenize=False):
        # Join all message contents for tokenization
        return " ".join([m.get("content", "") for m in messages])

@pytest.mark.asyncio
async def test_max_tokens_none_omitted():
    with patch("requests.Session.get") as mock_get, \
         patch.object(VLLMClient, "check_server", return_value=None), \
         patch("openai.resources.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock_create:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"max_model_len": 2000}
        mock_create.return_value = "ok"

        async with VLLMClient(tokenizer=MockTokenizer()) as client:
            result = await client.completions_create(model="test-model", prompt="a"*100, max_tokens=None)

            mock_create.assert_awaited_once()
            _, kwargs = mock_create.call_args
            assert kwargs["max_tokens"] is NOT_GIVEN
            assert result == "ok"

@pytest.mark.asyncio
async def test_no_error_for_long_prompt_and_omitted_max_tokens():
    with patch("requests.Session.get") as mock_get, \
         patch.object(VLLMClient, "check_server", return_value=None), \
         patch("openai.resources.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock_create:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"max_model_len": 50}
        mock_create.return_value = "ok"

        async with VLLMClient(tokenizer=MockTokenizer()) as client:
            result = await client.completions_create(model="test-model", prompt="a"*60, max_tokens=None)

            mock_create.assert_awaited_once()
            _, kwargs = mock_create.call_args
            assert kwargs["max_tokens"] is NOT_GIVEN
            assert result == "ok"

@pytest.mark.asyncio
async def test_stability_various_prompt_lengths():
    with patch("requests.Session.get") as mock_get, \
         patch.object(VLLMClient, "check_server", return_value=None), \
         patch("openai.resources.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock_create:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"max_model_len": 100}
        mock_create.return_value = "ok"

        async with VLLMClient(tokenizer=MockTokenizer()) as client:
            for prompt in ["", "a"*99, "a"*100]:
                await client.completions_create(model="test-model", prompt=prompt, max_tokens=None)
                _, kwargs = mock_create.call_args
                assert kwargs["max_tokens"] is NOT_GIVEN
                mock_create.reset_mock()

@pytest.mark.asyncio
async def test_explicit_max_tokens_respected():
    with patch("requests.Session.get") as mock_get, \
         patch.object(VLLMClient, "check_server", return_value=None), \
         patch("openai.resources.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock_create:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"max_model_len": 100}
        mock_create.return_value = "ok"

        async with VLLMClient(tokenizer=MockTokenizer()) as client:
            await client.completions_create(model="test-model", prompt="a"*10, max_tokens=5)
            _, kwargs = mock_create.call_args
            assert kwargs["max_tokens"] == 5

@pytest.mark.asyncio
async def test_vllm_client_with_mock_openai_client(mock_openai_client):
    with patch("requests.Session.get") as mock_get, \
         patch.object(VLLMClient, "check_server", return_value=None), \
         patch("openai.resources.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock_create:

        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"max_model_len": 128}
        mock_create.side_effect = mock_openai_client.completions.create

        async with VLLMClient(tokenizer=MockTokenizer()) as client:
            result = await client.completions_create(model="test-model", prompt="hello world", max_tokens=None)

            mock_create.assert_awaited_once()
            _, kwargs = mock_create.call_args
            assert kwargs["max_tokens"] is NOT_GIVEN
            assert result is not None
