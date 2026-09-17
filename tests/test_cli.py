"""Tests for the UnTowAI command-line interface."""

import sys

from untowai.cli import main, run_prompt
from untowai.credentials import CredentialNotFoundError
from untowai.providers.base import ProviderError, ProviderResponse
from untowai.providers.registry import ProviderRegistry
from untowai.service import AIService


class FakeProvider:
    """Test provider that never performs network access."""

    def __init__(self, name: str = "fake") -> None:
        self.name = name
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


def make_fake_service(
    provider_name: str = "fake",
) -> tuple[AIService, FakeProvider]:
    """Create an application service backed by a fake provider."""
    provider = FakeProvider(name=provider_name)
    registry = ProviderRegistry()
    registry.register(provider)

    return AIService(registry), provider


def test_cli_without_prompt(capsys, monkeypatch):
    """The CLI shows version and usage information without a prompt."""
    monkeypatch.setattr(sys, "argv", ["untowai"])

    exit_code = main()

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "UnTowAI 0.1.0" in captured.out
    assert "Usage: untowai <prompt>" in captured.out


def test_cli_constructs_default_service(capsys, monkeypatch):
    """The CLI constructs the default service when none is provided."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["untowai", "Hello"],
    )

    service, provider = make_fake_service(provider_name="openai")

    def fake_create_service() -> AIService:
        return service

    monkeypatch.setattr(
        "untowai.cli.create_service",
        fake_create_service,
    )

    exit_code = main()

    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out == "Fake response to: Hello\n"
    assert captured.err == ""
    assert provider.calls == [
        ("gpt-5", "Hello"),
    ]


def test_cli_reports_missing_credential(capsys, monkeypatch):
    """The CLI reports a missing credential without exposing a secret."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["untowai", "Hello"],
    )

    def fake_create_service() -> AIService:
        raise CredentialNotFoundError(
            "Credential 'OPENAI_API_KEY' was not found."
        )

    monkeypatch.setattr(
        "untowai.cli.create_service",
        fake_create_service,
    )

    exit_code = main()

    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == (
        "Error: Credential 'OPENAI_API_KEY' was not found.\n"
    )


def test_cli_reports_provider_error(capsys, monkeypatch):
    """The CLI converts provider failures into a safe process result."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["untowai", "Hello"],
    )

    service, _ = make_fake_service(provider_name="openai")

    def fake_run_prompt(*, service, provider_name, model, prompt):
        raise ProviderError(
            "OpenAI API request failed with HTTP 401."
        )

    monkeypatch.setattr(
        "untowai.cli.run_prompt",
        fake_run_prompt,
    )

    exit_code = main(service=service)

    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == (
        "Error: OpenAI API request failed with HTTP 401.\n"
    )


def test_run_prompt_routes_prompt_through_service(capsys):
    """The CLI sends prompts through the application service."""
    service, provider = make_fake_service()

    exit_code = run_prompt(
        service=service,
        provider_name="fake",
        model="test-model",
        prompt="What is a Linux process?",
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Fake response to: What is a Linux process?" in captured.out
    assert provider.calls == [
        ("test-model", "What is a Linux process?"),
    ]


def test_run_prompt_displays_normalized_provider_response(capsys):
    """The CLI displays the response text returned by the service."""
    service, _ = make_fake_service()

    exit_code = run_prompt(
        service=service,
        provider_name="fake",
        model="test-model",
        prompt="Hello",
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out == "Fake response to: Hello\n"


def test_cli_does_not_execute_prompt(capsys):
    """Prompt text remains data passed to the service."""
    service, provider = make_fake_service()

    dangerous_text = "rm -rf ~/important-data"

    exit_code = run_prompt(
        service=service,
        provider_name="fake",
        model="test-model",
        prompt=dangerous_text,
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert dangerous_text in captured.out
    assert provider.calls == [
        ("test-model", dangerous_text),
    ]
