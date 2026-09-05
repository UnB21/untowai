"""Tests for the UnTowAI provider abstraction."""

from untowai.providers.base import Provider, ProviderResponse


class FakeProvider:
    """Test provider that never performs network access."""

    name = "fake"

    def generate(
        self,
        model: str,
        prompt: str,
    ) -> ProviderResponse:
        return ProviderResponse(
            text=f"Fake response to: {prompt}",
            provider=self.name,
            model=model,
        )


def test_provider_response_contains_normalized_data():
    """A provider response stores the normalized response fields."""
    response = ProviderResponse(
        text="A Linux process is a running instance of a program.",
        provider="test",
        model="test-model",
    )

    assert response.text == "A Linux process is a running instance of a program."
    assert response.provider == "test"
    assert response.model == "test-model"


def test_provider_response_is_immutable():
    """A provider response cannot be modified after creation."""
    response = ProviderResponse(
        text="original",
        provider="test",
        model="test-model",
    )

    try:
        response.text = "modified"
    except AttributeError:
        pass
    else:
        raise AssertionError("ProviderResponse should be immutable")


def test_fake_provider_matches_provider_protocol():
    """A structurally compatible provider can be used as a Provider."""
    provider: Provider = FakeProvider()

    response = provider.generate(
        model="test-model",
        prompt="What is a Linux process?",
    )

    assert response.text == "Fake response to: What is a Linux process?"
    assert response.provider == "fake"
    assert response.model == "test-model"
