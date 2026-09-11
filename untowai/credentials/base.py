"""Credential source interfaces for UnTowAI."""

import re
from typing import Protocol


_CREDENTIAL_NAME_PATTERN = re.compile(
    r"^[A-Z](?:[A-Z0-9_]*[A-Z0-9])?$"
)


class CredentialNotFoundError(RuntimeError):
    """Raised when a requested credential cannot be found."""


def validate_credential_name(name: str) -> str:
    """Validate and return a credential identifier."""
    if not isinstance(name, str):
        raise TypeError("Credential name must be a string.")

    if not _CREDENTIAL_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "Credential name must contain only uppercase letters, "
            "digits, and underscores; it must start with an uppercase "
            "letter and end with an uppercase letter or digit."
        )

    return name


class CredentialSource(Protocol):
    """Interface for retrieving credentials by name."""

    def get(self, name: str) -> str:
        """Return the requested credential."""
        ...
