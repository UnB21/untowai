"""Application service for UnTowAI."""

from .providers import ProviderRegistry
from .providers.base import ProviderResponse


class AIService:
    """Coordinate AI requests through registered providers."""

    def __init__(self, registry: ProviderRegistry) -> None:
        """Initialize the service with a provider registry."""
        self._registry = registry

    def ask(
        self,
        provider_name: str,
        model: str,
        prompt: str,
    ) -> ProviderResponse:
        """Send a prompt to a named provider and return its response."""
        if not provider_name:
            raise ValueError("Provider name must not be empty.")

        if not model:
            raise ValueError("Model must not be empty.")

        if not prompt:
            raise ValueError("Prompt must not be empty.")

        provider = self._registry.get(provider_name)

        return provider.generate(
            model=model,
            prompt=prompt,
        )
