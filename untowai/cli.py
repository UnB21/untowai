"""Command-line interface for UnTowAI."""

import sys

from . import __version__
from .application import create_service
from .credentials import CredentialNotFoundError
from .service import AIService


def run_prompt(
    service: AIService,
    provider_name: str,
    model: str,
    prompt: str,
) -> int:
    """Send a prompt through the application service and display the response."""
    response = service.ask(
        provider_name=provider_name,
        model=model,
        prompt=prompt,
    )

    print(response.text)

    return 0


def main(service: AIService | None = None) -> int:
    """Run the UnTowAI command-line interface."""
    if len(sys.argv) == 1:
        print(f"UnTowAI {__version__}")
        print("Usage: untowai <prompt>")
        return 0

    prompt = " ".join(sys.argv[1:])

    if service is None:
        try:
            service = create_service()
        except CredentialNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    return run_prompt(
        service=service,
        provider_name="openai",
        model="gpt-5",
        prompt=prompt,
    )


if __name__ == "__main__":
    raise SystemExit(main())
