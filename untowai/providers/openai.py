"""OpenAI provider implementation for UnTowAI."""

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from .base import ProviderResponse


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


class OpenAIProvider:
    """Generate text responses using the OpenAI Responses API."""

    name = "openai"

    def __init__(
        self,
        api_key: str,
        transport: Callable[..., Any] | None = None,
    ) -> None:
        """Initialize the provider with an API key and optional HTTP transport."""
        if not api_key:
            raise ValueError("OpenAI API key must not be empty.")

        self._api_key = api_key
        self._transport = transport or urllib.request.urlopen

    def generate(
        self,
        model: str,
        prompt: str,
    ) -> ProviderResponse:
        """Generate a response from a text prompt."""
        if not model:
            raise ValueError("OpenAI model must not be empty.")

        if not prompt:
            raise ValueError("Prompt must not be empty.")

        payload = {
            "model": model,
            "input": prompt,
        }

        request = urllib.request.Request(
            OPENAI_RESPONSES_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with self._transport(request, timeout=30) as response:
                response_data = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"OpenAI API request failed with HTTP {exc.code}."
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("Unable to reach the OpenAI API.") from exc

        text = response_data.get("output_text")

        if not isinstance(text, str):
            raise RuntimeError("OpenAI API response did not contain output text.")

        return ProviderResponse(
            text=text,
            provider=self.name,
            model=model,
        )
