"""Tests for the UnTowAI credential abstraction."""

import pytest

from untowai.credentials import (
    CredentialNotFoundError,
    CredentialSource,
    EnvironmentCredentialSource,
    validate_credential_name,
)


class FakeCredentialSource:
    """Test credential source that never accesses real storage."""

    def __init__(self, credentials: dict[str, str]) -> None:
        self.credentials = credentials

    def get(self, name: str) -> str:
        try:
            return self.credentials[name]
        except KeyError as exc:
            raise CredentialNotFoundError(
                f"Credential '{name}' was not found."
            ) from exc


def test_credential_source_returns_credential():
    """A credential source can return a requested credential."""
    source: CredentialSource = FakeCredentialSource(
        {"OPENAI_API_KEY": "test-key"}
    )

    assert source.get("OPENAI_API_KEY") == "test-key"


def test_credential_source_matches_protocol():
    """A structurally compatible source satisfies the protocol."""
    source: CredentialSource = FakeCredentialSource(
        {"TEST_CREDENTIAL": "test-value"}
    )

    assert source.get("TEST_CREDENTIAL") == "test-value"


def test_missing_credential_raises_specific_error():
    """A missing credential produces a specific error."""
    source: CredentialSource = FakeCredentialSource({})

    with pytest.raises(
        CredentialNotFoundError,
        match="Credential 'OPENAI_API_KEY' was not found",
    ):
        source.get("OPENAI_API_KEY")


@pytest.mark.parametrize(
    "name",
    [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "TEST_CREDENTIAL",
        "API_KEY_2",
        "A",
        "A_B",
        "A__B",
        "KEY123",
    ],
)
def test_valid_credential_names_are_accepted(name):
    """Valid credential identifiers pass validation."""
    assert validate_credential_name(name) == name


@pytest.mark.parametrize(
    "name",
    [
        "",
        "openai_api_key",
        "OpenAI_API_KEY",
        "OPENAI-API-KEY",
        "OPENAI/API_KEY",
        r"OPENAI\API_KEY",
        "../OPENAI_API_KEY",
        "/path/to/key",
        "OPENAI API KEY",
        ".OPENAI_API_KEY",
        "_OPENAI_API_KEY",
        "OPENAI_API_KEY_",
        "OPENAI.API.KEY",
    ],
)
def test_invalid_credential_names_are_rejected(name):
    """Invalid credential identifiers raise ValueError."""
    with pytest.raises(ValueError):
        validate_credential_name(name)


@pytest.mark.parametrize(
    "name",
    [
        None,
        123,
        b"OPENAI_API_KEY",
        ["OPENAI_API_KEY"],
    ],
)
def test_non_string_credential_names_are_rejected(name):
    """Credential identifiers must be strings."""
    with pytest.raises(TypeError):
        validate_credential_name(name)


def test_environment_source_returns_credential():
    """The environment source returns a requested credential."""
    environment = {
        "OPENAI_API_KEY": "test-key",
    }

    source: CredentialSource = EnvironmentCredentialSource(environment)

    assert source.get("OPENAI_API_KEY") == "test-key"


def test_environment_source_matches_protocol():
    """The environment source satisfies the credential protocol."""
    source: CredentialSource = EnvironmentCredentialSource(
        {"TEST_CREDENTIAL": "test-value"}
    )

    assert source.get("TEST_CREDENTIAL") == "test-value"


def test_environment_source_uses_process_environment_by_default(monkeypatch):
    """The default source reads from the process environment."""
    monkeypatch.setenv("TEST_CREDENTIAL", "environment-value")

    source = EnvironmentCredentialSource()

    assert source.get("TEST_CREDENTIAL") == "environment-value"


def test_environment_source_missing_credential_raises_specific_error():
    """A missing environment credential raises the specific error."""
    source = EnvironmentCredentialSource({})

    with pytest.raises(
        CredentialNotFoundError,
        match="Credential 'OPENAI_API_KEY' was not found",
    ):
        source.get("OPENAI_API_KEY")


@pytest.mark.parametrize(
    "name",
    [
        "",
        "openai_api_key",
        "OpenAI_API_KEY",
        "OPENAI-API-KEY",
        "OPENAI/API_KEY",
        r"OPENAI\API_KEY",
        "../OPENAI_API_KEY",
        "/path/to/key",
        "OPENAI API KEY",
        ".OPENAI_API_KEY",
        "_OPENAI_API_KEY",
        "OPENAI_API_KEY_",
        "OPENAI.API.KEY",
    ],
)
def test_environment_source_rejects_invalid_credential_names(name):
    """The environment source rejects invalid credential identifiers."""
    source = EnvironmentCredentialSource({name: "test-value"})

    with pytest.raises(ValueError):
        source.get(name)


def test_environment_source_does_not_expose_credential_value_in_errors():
    """Missing-credential errors contain the name but not a credential value."""
    source = EnvironmentCredentialSource({})

    with pytest.raises(CredentialNotFoundError) as exc_info:
        source.get("OPENAI_API_KEY")

    assert "OPENAI_API_KEY" in str(exc_info.value)
    assert "secret-value" not in str(exc_info.value)
