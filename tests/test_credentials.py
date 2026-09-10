"""Tests for the UnTowAI credential abstraction."""

import pytest

from untowai.credentials import CredentialNotFoundError, CredentialSource


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
