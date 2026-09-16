"""Application construction for UnTowAI."""

from .credentials import CredentialSource, EnvironmentCredentialSource
from .providers import ProviderRegistry
from .providers.openai import OpenAIProvider
from .service import AIService


OPENAI_API_KEY_NAME = "OPENAI_API_KEY"


def create_service(
    credential_source: CredentialSource | None = None,
) -> AIService:
    """Construct the default UnTowAI application service."""
    source = (
        credential_source
        if credential_source is not None
        else EnvironmentCredentialSource()
    )
    api_key = source.get(OPENAI_API_KEY_NAME)

    provider = OpenAIProvider(api_key=api_key)

    registry = ProviderRegistry()
    registry.register(provider)

    return AIService(registry)
