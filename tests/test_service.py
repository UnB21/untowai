"""Tests for the UnTowAI application service."""

import pytest

from untowai.providers.base import ProviderResponse
from untowai.providers.registry import ProviderRegistry
from untowai.service import AIService


class FakeProvider:
    """Test provider that never performs network access."""

    name = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def generate(
        self,
        model: str,
        prompt: str,
    ) -> ProviderResponse:
        self.calls.append((model, prompt))

        return ProviderResponse(
            text=f"Fake response to: {prompt}",
            provider=self.name,
            model=model,
        )


def test_service_routes_request_to_named_provider():
    """The service sends a request to the selected provider."""
    registry = ProviderRegistry()
    provider = FakeProvider()
    registry.register(provider)

    service = AIService(registry)

    response = service.ask(
        provider_name="fake",
        model="test-model",
        prompt="What is a Linux process?",
    )

    assert response.text == "Fake response to: What is a Linux process?"
    assert response.provider == "fake"
    assert response.model == "test-model"
    assert provider.calls == [
        ("test-model", "What is a Linux process?"),
    ]


def test_service_supports_multiple_registered_providers():
    """The service routes each request to the requested provider."""
    registry = ProviderRegistry()
    first = FakeProvider()
    second = FakeProvider()

    first.name = "first"
    second.name = "second"

    registry.register(first)
    registry.register(second)

    service = AIService(registry)

    response = service.ask(
        provider_name="second",
        model="test-model",
        prompt="Hello",
    )

    assert response.provider == "second"
    assert second.calls == [("test-model", "Hello")]
    assert first.calls == []


def test_service_rejects_unknown_provider():
    """The service reports an unknown provider through the registry."""
    registry = ProviderRegistry()
    service = AIService(registry)

    with pytest.raises(
        KeyError,
        match="Provider 'missing' is not registered",
    ):
        service.ask(
            provider_name="missing",
            model="test-model",
            prompt="Hello",
        )


def test_service_rejects_empty_provider_name():
    """An empty provider name is rejected before registry access."""
    registry = ProviderRegistry()
    service = AIService(registry)

    with pytest.raises(
        ValueError,
        match="Provider name must not be empty",
    ):
        service.ask(
            provider_name="",
            model="test-model",
            prompt="Hello",
        )


def test_service_rejects_empty_model():
    """An empty model is rejected before provider execution."""
    registry = ProviderRegistry()
    provider = FakeProvider()
    registry.register(provider)

    service = AIService(registry)

    with pytest.raises(
        ValueError,
        match="Model must not be empty",
    ):
        service.ask(
            provider_name="fake",
            model="",
            prompt="Hello",
        )

    assert provider.calls == []


def test_service_rejects_empty_prompt():
    """An empty prompt is rejected before provider execution."""
    registry = ProviderRegistry()
    provider = FakeProvider()
    registry.register(provider)

    service = AIService(registry)

    with pytest.raises(
        ValueError,
        match="Prompt must not be empty",
    ):
        service.ask(
            provider_name="fake",
            model="test-model",
            prompt="",
        )

    assert provider.calls == []
