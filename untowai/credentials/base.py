"""Credential source interfaces for UnTowAI."""

from typing import Protocol


class CredentialNotFoundError(RuntimeError):
    """Raised when a requested credential cannot be found."""


class CredentialSource(Protocol):
    """Interface for retrieving credentials by name."""

    def get(self, name: str) -> str:
        """Return the requested credential."""
        ...
