"""Command-line interface for UnTowAI."""

import argparse
import sys

from . import __version__
from .application import create_service
from .credentials import CredentialNotFoundError
from .providers.base import ProviderError
from .service import AIService


DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-5"


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


def _build_parser() -> argparse.ArgumentParser:
    """Build the UnTowAI command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="untowai",
        description="Send a text prompt through UnTowAI.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL}).",
    )

    parser.add_argument(
        "prompt",
        nargs="*",
        help="Text prompt to send to the selected model.",
    )

    return parser


def main(service: AIService | None = None) -> int:
    """Run the UnTowAI command-line interface."""
    parser = _build_parser()

    if len(sys.argv) == 1:
        print(f"UnTowAI {__version__}")
        print("Usage: untowai [--model MODEL] <prompt>")
        return 0

    args = parser.parse_args()

    if not args.prompt:
        parser.error("a prompt is required")

    prompt = " ".join(args.prompt)

    if service is None:
        try:
            service = create_service()
        except CredentialNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    try:
        return run_prompt(
            service=service,
            provider_name=DEFAULT_PROVIDER,
            model=args.model,
            prompt=prompt,
        )
    except ProviderError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
