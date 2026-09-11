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


def make_response_payload(text: str) -> dict[str, object]:
    """Create a raw Responses API payload containing generated text."""
    return {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                    }
                ],
            }
        ]
    }


def test_openai_provider_matches_provider_protocol():
    """OpenAIProvider structurally implements the Provider protocol."""
    provider: Provider = OpenAIProvider(
        api_key="test-key",
        transport=FakeTransport(make_response_payload("test response")),
    )

    assert provider.name == "openai"


def test_openai_provider_generates_normalized_response():
    """The provider converts a raw API response into ProviderResponse."""
    transport = FakeTransport(
        make_response_payload(
            "A Linux process is a running program."
        )
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


def test_openai_provider_extracts_multiple_output_text_items():
    """The provider combines multiple output text content items."""
    transport = FakeTransport(
        {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "First part. ",
                        },
                        {
                            "type": "output_text",
                            "text": "Second part.",
                        },
                    ],
                }
            ]
        }
    )
    provider = OpenAIProvider(
        api_key="test-key",
        transport=transport,
    )

    response = provider.generate(
        model="test-model",
        prompt="Hello",
    )

    assert response.text == "First part. Second part."


def test_openai_provider_ignores_non_message_output_items():
    """The provider ignores output items that do not contain message text."""
    transport = FakeTransport(
        {
            "output": [
                {
                    "type": "reasoning",
                    "summary": [],
                },
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Actual response.",
                        }
                    ],
                },
            ]
        }
    )
    provider = OpenAIProvider(
        api_key="test-key",
        transport=transport,
    )

    response = provider.generate(
        model="test-model",
        prompt="Hello",
    )

    assert response.text == "Actual response."


def test_openai_provider_builds_expected_request():
    """The provider constructs the expected authenticated POST request."""
    transport = FakeTransport(make_response_payload("test response"))
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
            transport=FakeTransport(make_response_payload("unused")),
        )


def test_openai_provider_rejects_empty_model():
    """An empty model is rejected before network access."""
    transport = FakeTransport(make_response_payload("unused"))
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
    transport = FakeTransport(make_response_payload("unused"))
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
        match="did not contain output",
    ):
        provider.generate(
            model="test-model",
            prompt="Hello",
        )


def test_openai_provider_rejects_output_without_message_text():
    """An output response without usable message text is rejected."""
    transport = FakeTransport(
        {
            "output": [
                {
                    "type": "reasoning",
                    "summary": [],
                }
            ]
        }
    )
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
