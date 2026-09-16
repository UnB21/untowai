"""Tests for UnTowAI application construction."""

import pytest

from untowai.application import OPENAI_API_KEY_NAME, create_service
from untowai.credentials import CredentialNotFoundError, CredentialSource
from untowai.providers.base import ProviderResponse


class FakeCredentialSource:
    """Test credential source that never accesses real storage."""

    def __init__(self, credentials: dict[str, str]) -> None:
        self.credentials = credentials
        self.requests: list[str] = []

    def get(self, name: str) -> str:
        self.requests.append(name)

        try:
            return self.credentials[name]
        except KeyError as exc:
            raise CredentialNotFoundError(
                f"Credential '{name}' was not found."
            ) from exc


class FalseyCredentialSource(FakeCredentialSource):
    """Test credential source whose object is false-y."""

    def __bool__(self) -> bool:
        return False


class FakeOpenAIProvider:
    """Test provider that never performs network access."""

    name = "openai"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.calls: list[tuple[str, str]] = []

    def generate(
        self,
        model: str,
        prompt: str,
    ) -> ProviderResponse:
        self.calls.append((model, prompt))

        return ProviderResponse(
            text="fake response",
            provider=self.name,
            model=model,
        )


def test_create_service_uses_credential_source(monkeypatch):
    """Application construction retrieves the OpenAI API key by name."""
    credential_source: CredentialSource = FakeCredentialSource(
        {OPENAI_API_KEY_NAME: "test-key"}
    )

    captured: dict[str, FakeOpenAIProvider] = {}

    def fake_provider(api_key: str) -> FakeOpenAIProvider:
        provider = FakeOpenAIProvider(api_key)
        captured["provider"] = provider
        return provider

    monkeypatch.setattr(
        "untowai.application.OpenAIProvider",
        fake_provider,
    )

    service = create_service(credential_source)

    response = service.ask(
        provider_name="openai",
        model="test-model",
        prompt="Hello",
    )

    assert response.text == "fake response"
    assert captured["provider"].api_key == "test-key"
    assert credential_source.requests == [OPENAI_API_KEY_NAME]


def test_create_service_uses_falsey_credential_source(monkeypatch):
    """A false-y credential source is still used when explicitly provided."""
    credential_source: CredentialSource = FalseyCredentialSource(
        {OPENAI_API_KEY_NAME: "test-key"}
    )

    monkeypatch.setattr(
        "untowai.application.OpenAIProvider",
        FakeOpenAIProvider,
    )

    service = create_service(credential_source)

    response = service.ask(
        provider_name="openai",
        model="test-model",
        prompt="Hello",
    )

    assert response.text == "fake response"
    assert credential_source.requests == [OPENAI_API_KEY_NAME]


def test_create_service_registers_openai_provider(monkeypatch):
    """Application construction registers the OpenAI provider."""
    credential_source = FakeCredentialSource(
        {OPENAI_API_KEY_NAME: "test-key"}
    )

    monkeypatch.setattr(
        "untowai.application.OpenAIProvider",
        FakeOpenAIProvider,
    )

    service = create_service(credential_source)

    response = service.ask(
        provider_name="openai",
        model="test-model",
        prompt="Hello",
    )

    assert response.text == "fake response"
    assert response.provider == "openai"
    assert response.model == "test-model"


def test_create_service_uses_process_environment_by_default(monkeypatch):
    """The default application construction uses environment credentials."""
    monkeypatch.setenv(OPENAI_API_KEY_NAME, "environment-key")
    monkeypatch.setattr(
        "untowai.application.OpenAIProvider",
        FakeOpenAIProvider,
    )

    service = create_service()

    response = service.ask(
        provider_name="openai",
        model="test-model",
        prompt="Hello",
    )

    assert response.text == "fake response"


def test_create_service_rejects_missing_openai_credential(monkeypatch):
    """Missing credentials prevent application construction."""
    monkeypatch.delenv(OPENAI_API_KEY_NAME, raising=False)

    with pytest.raises(
        CredentialNotFoundError,
        match="Credential 'OPENAI_API_KEY' was not found",
    ):
        create_service()


def test_create_service_does_not_make_network_requests(monkeypatch):
    """Application construction only assembles components."""
    credential_source = FakeCredentialSource(
        {OPENAI_API_KEY_NAME: "test-key"}
    )

    captured: dict[str, FakeOpenAIProvider] = {}

    def fake_provider(api_key: str) -> FakeOpenAIProvider:
        provider = FakeOpenAIProvider(api_key)
        captured["provider"] = provider
        return provider

    monkeypatch.setattr(
        "untowai.application.OpenAIProvider",
        fake_provider,
    )

    create_service(credential_source)

    assert captured["provider"].calls == []
