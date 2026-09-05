"""Provider registry for UnTowAI."""

from .base import Provider


class ProviderRegistry:
    """Store and retrieve AI providers by their names."""

    def __init__(self) -> None:
        """Initialize an empty provider registry."""
        self._providers: dict[str, Provider] = {}

    def register(self, provider: Provider) -> None:
        """Register a provider by its name."""
        if provider.name in self._providers:
            raise ValueError(
                f"Provider '{provider.name}' is already registered."
            )

        self._providers[provider.name] = provider

    def get(self, name: str) -> Provider:
        """Return a registered provider by name."""
        try:
            return self._providers[name]
        except KeyError as exc:
            raise KeyError(
                f"Provider '{name}' is not registered."
            ) from exc

    def has(self, name: str) -> bool:
        """Return whether a provider is registered."""
        return name in self._providers

    def names(self) -> tuple[str, ...]:
        """Return the names of all registered providers."""
        return tuple(self._providers)
