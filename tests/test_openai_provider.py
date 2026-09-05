"""Tests for the UnTowAI OpenAI provider."""

import json

import pytest

from untowai.providers.base import Provider
from untowai.providers.openai import OpenAIProvider


class FakeResponse:
    """Fake HTTP response used without network access."""

    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class FakeTransport:
    """Capture HTTP requests without contacting the network."""

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.request = None
        self.timeout = None

    def __call__(self, request, timeout):
        self.request = request
        self.timeout = timeout
        return FakeResponse(self.payload)


def test_openai_provider_matches_provider_protocol():
    """OpenAIProvider structurally implements the Provider protocol."""
    provider: Provider = OpenAIProvider(
        api_key="test-key",
        transport=FakeTransport({"output_text": "test response"}),
    )

    assert provider.name == "openai"


def test_openai_provider_generates_normalized_response():
    """The provider converts an API response into ProviderResponse."""
    transport = FakeTransport(
        {"output_text": "A Linux process is a running program."}
    )
    provider = OpenAIProvider(
        api_key="test-key",
        transport=transport,
    )

    response = provider.generate(
        model="test-model",
        prompt="What is a Linux process?",
    )

    assert response.text == "A Linux process is a running program."
    assert response.provider == "openai"
    assert response.model == "test-model"


def test_openai_provider_builds_expected_request():
    """The provider constructs the expected authenticated POST request."""
    transport = FakeTransport({"output_text": "test response"})
    provider = OpenAIProvider(
        api_key="test-key",
        transport=transport,
    )

    provider.generate(
        model="test-model",
        prompt="Hello",
    )

    assert transport.request is not None
    assert transport.request.full_url == (
        "https://api.openai.com/v1/responses"
    )
    assert transport.request.method == "POST"
    assert transport.request.get_header("Authorization") == "Bearer test-key"
    assert transport.request.get_header("Content-type") == "application/json"
    assert json.loads(transport.request.data) == {
        "model": "test-model",
        "input": "Hello",
    }
    assert transport.timeout == 30


def test_openai_provider_rejects_empty_api_key():
    """An empty API key is rejected before any request can occur."""
    with pytest.raises(ValueError, match="API key must not be empty"):
        OpenAIProvider(
            api_key="",
            transport=FakeTransport({"output_text": "unused"}),
        )


def test_openai_provider_rejects_empty_model():
    """An empty model is rejected before network access."""
    transport = FakeTransport({"output_text": "unused"})
    provider = OpenAIProvider(
        api_key="test-key",
        transport=transport,
    )

    with pytest.raises(ValueError, match="model must not be empty"):
        provider.generate(
            model="",
            prompt="Hello",
        )

    assert transport.request is None


def test_openai_provider_rejects_empty_prompt():
    """An empty prompt is rejected before network access."""
    transport = FakeTransport({"output_text": "unused"})
    provider = OpenAIProvider(
        api_key="test-key",
        transport=transport,
    )

    with pytest.raises(ValueError, match="Prompt must not be empty"):
        provider.generate(
            model="test-model",
            prompt="",
        )

    assert transport.request is None


def test_openai_provider_rejects_missing_output_text():
    """A malformed API response is rejected."""
    transport = FakeTransport({})
    provider = OpenAIProvider(
        api_key="test-key",
        transport=transport,
    )

    with pytest.raises(
        RuntimeError,
        match="did not contain output text",
    ):
        provider.generate(
            model="test-model",
            prompt="Hello",
        )
