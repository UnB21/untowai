"""Environment-backed credential source for UnTowAI."""

import os
from collections.abc import Mapping

from .base import CredentialNotFoundError, validate_credential_name


class EnvironmentCredentialSource:
    """Retrieve credentials from environment variables."""

    def __init__(
        self,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize the source with an environment mapping."""
        self._environment = environment if environment is not None else os.environ

    def get(self, name: str) -> str:
        """Return a credential from the environment."""
        credential_name = validate_credential_name(name)

        try:
            return self._environment[credential_name]
        except KeyError as exc:
            raise CredentialNotFoundError(
                f"Credential '{credential_name}' was not found."
            ) from exc
