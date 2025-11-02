"""Simple client for requesting YandexGPT chat completions."""

from __future__ import annotations

import json
import os
from typing import Any

import requests

API_URL = "https://api.eliza.yandex.net/openai/v1/chat/completions"
DEFAULT_MODEL = "gpt-4.1-nano"


def _get_token() -> str:
    token = os.getenv("SOY_TOKEN")
    if not token:
        raise RuntimeError("SOY_TOKEN environment variable is not set")
    return token


def create_chat_completion(message: str, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """Request a chat completion for the provided user message."""

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": message,
            }
        ],
    }

    headers = {
        "authorization": f"OAuth {_get_token()}",
        "content-type": "application/json",
    }

    response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def pretty_print_response(response_body: dict[str, Any]) -> None:
    """Helper to pretty-print the model response for debugging."""

    print(json.dumps(response_body, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    result = create_chat_completion("Hello!")
    pretty_print_response(result)

