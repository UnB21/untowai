"""Provider interfaces for UnTowAI."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ProviderResponse:
    """Normalized response returned by an AI provider."""

    text: str
    provider: str
    model: str


class Provider(Protocol):
    """Interface that AI provider implementations must satisfy."""

    name: str

    def generate(
        self,
        model: str,
        prompt: str,
    ) -> ProviderResponse:
        """Generate a response from a text prompt."""
        ...
