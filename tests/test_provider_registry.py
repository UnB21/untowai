"""Tests for the UnTowAI provider registry."""

import pytest

from untowai.providers.base import Provider, ProviderResponse
from untowai.providers.registry import ProviderRegistry


class FakeProvider:
    """Test provider that never performs network access."""

    def __init__(self, name: str) -> None:
        self.name = name

    def generate(
        self,
        model: str,
        prompt: str,
    ) -> ProviderResponse:
        return ProviderResponse(
            text=f"{self.name} response to: {prompt}",
            provider=self.name,
            model=model,
        )


def test_registry_starts_empty():
    """A new registry contains no providers."""
    registry = ProviderRegistry()

    assert registry.names() == ()


def test_registry_registers_and_retrieves_provider():
    """A registered provider can be retrieved by name."""
    registry = ProviderRegistry()
    provider: Provider = FakeProvider("fake")

    registry.register(provider)

    assert registry.has("fake")
    assert registry.get("fake") is provider
    assert registry.names() == ("fake",)


def test_registry_supports_multiple_providers():
    """The registry can store multiple providers independently."""
    registry = ProviderRegistry()
    first = FakeProvider("first")
    second = FakeProvider("second")

    registry.register(first)
    registry.register(second)

    assert registry.names() == ("first", "second")
    assert registry.get("first") is first
    assert registry.get("second") is second


def test_registry_rejects_duplicate_provider_names():
    """A provider name cannot silently replace an existing provider."""
    registry = ProviderRegistry()

    registry.register(FakeProvider("fake"))

    with pytest.raises(
        ValueError,
        match="Provider 'fake' is already registered",
    ):
        registry.register(FakeProvider("fake"))


def test_registry_rejects_unknown_provider():
    """Retrieving an unknown provider produces a clear error."""
    registry = ProviderRegistry()

    with pytest.raises(
        KeyError,
        match="Provider 'missing' is not registered",
    ):
        registry.get("missing")


def test_registry_has_returns_false_for_unknown_provider():
    """The has method returns false for an unregistered provider."""
    registry = ProviderRegistry()

    assert registry.has("missing") is False
