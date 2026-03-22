"""MiniMax Genie — LLM inference via MiniMax's OpenAI-compatible API."""

import importlib.util as importutil
import json
import os
import re
from typing import TYPE_CHECKING, Any, Optional, cast

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

try:
    from openai import APIError, APITimeoutError, AsyncOpenAI, OpenAI, RateLimitError
except ImportError:

    class APIError(Exception):
        """API error."""

    class APITimeoutError(Exception):
        """API timeout error."""

    class RateLimitError(Exception):
        """Rate limit error."""

    OpenAI = None  # type: ignore
    AsyncOpenAI = None  # type: ignore

from .base import BaseGenie

if TYPE_CHECKING:
    from pydantic import BaseModel

# MiniMax OpenAI-compatible API endpoint
_MINIMAX_BASE_URL = "https://api.minimax.io/v1"

# Regex to strip <think>…</think> blocks from M2.5/M2.7 reasoning models
_THINK_TAG_RE = re.compile(r"<think>[\s\S]*?</think>\s*", re.DOTALL)


class MiniMaxGenie(BaseGenie):
    """MiniMax's Genie — powered by MiniMax M2.7 / M2.5 models.

    Uses the OpenAI-compatible chat-completions endpoint provided by MiniMax.

    Args:
        model: MiniMax model name (default: "MiniMax-M2.7").
        api_key: MiniMax API key. Falls back to the ``MINIMAX_API_KEY``
                 environment variable.
        base_url: Override the API base URL (default:
                  ``https://api.minimax.io/v1``).
        temperature: Sampling temperature in (0, 1]. Defaults to 0.7.

    """

    AVAILABLE_MODELS = [
        "MiniMax-M2.7",
        "MiniMax-M2.7-highspeed",
        "MiniMax-M2.5",
        "MiniMax-M2.5-highspeed",
    ]

    def __init__(
        self,
        model: str = "MiniMax-M2.7",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
    ):
        """Initialize the MiniMaxGenie.

        Args:
            model: MiniMax model identifier.
            api_key: API key for MiniMax (or set ``MINIMAX_API_KEY`` env var).
            base_url: Custom base URL for the API endpoint.
            temperature: Sampling temperature (0, 1].

        Raises:
            ImportError: If the ``openai`` or ``pydantic`` packages are missing.
            ValueError: If no API key is provided.

        """
        super().__init__()

        if not self._is_available():
            raise ImportError(
                "One or more of the required modules are not available: [pydantic, openai]. "
                "Please install the dependencies via `pip install chonkie[openai]`"
            )

        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY")
        if not self.api_key:
            raise ValueError(
                "MiniMaxGenie requires an API key. Either pass the `api_key` parameter "
                "or set the `MINIMAX_API_KEY` environment variable.",
            )

        # Clamp temperature to MiniMax's accepted range
        self.temperature = max(0.01, min(temperature, 1.0))
        self.model = model

        resolved_base_url = base_url or _MINIMAX_BASE_URL
        assert OpenAI is not None, "openai package is required but not installed"
        self.client = OpenAI(api_key=self.api_key, base_url=resolved_base_url)
        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=resolved_base_url)

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """Remove ``<think>…</think>`` reasoning blocks from model output."""
        return _THINK_TAG_RE.sub("", text).strip()

    # -- sync methods ---------------------------------------------------------

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, max=60),
        retry=retry_if_exception_type(
            cast(tuple[type[BaseException], ...], (RateLimitError, APIError, APITimeoutError))
        ),
    )
    def generate(self, prompt: str) -> str:
        """Generate a response from the given prompt.

        Args:
            prompt: User prompt string.

        Returns:
            Generated text with think-tags stripped.

        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("MiniMax response content is None")
        return self._strip_think_tags(content)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, max=60),
        retry=retry_if_exception_type(
            cast(tuple[type[BaseException], ...], (RateLimitError, APIError, APITimeoutError))
        ),
    )
    def generate_json(self, prompt: str, schema: "BaseModel") -> dict[str, Any]:
        """Generate a structured JSON response.

        Uses ``response_format={"type": "json_object"}`` because MiniMax does
        not support the ``beta.chat.completions.parse`` endpoint.  The schema
        is injected into the system prompt to guide the model.

        Args:
            prompt: User prompt string.
            schema: Pydantic model whose JSON schema describes the output.

        Returns:
            Parsed dictionary matching the schema.

        """
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        system_msg = (
            "You must respond with valid JSON that matches the following JSON schema. "
            "Do not include any additional text, markdown, or explanation.\n\n"
            f"Schema:\n{schema_json}"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("MiniMax response content is None")
        return json.loads(self._strip_think_tags(content))

    # -- async methods --------------------------------------------------------

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, max=60),
        retry=retry_if_exception_type(
            cast(tuple[type[BaseException], ...], (RateLimitError, APIError, APITimeoutError))
        ),
    )
    async def agenerate(self, prompt: str) -> str:
        """Generate a response asynchronously.

        Args:
            prompt: User prompt string.

        Returns:
            Generated text with think-tags stripped.

        """
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("MiniMax response content is None")
        return self._strip_think_tags(content)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, max=60),
        retry=retry_if_exception_type(
            cast(tuple[type[BaseException], ...], (RateLimitError, APIError, APITimeoutError))
        ),
    )
    async def agenerate_json(self, prompt: str, schema: "BaseModel") -> dict[str, Any]:
        """Generate a structured JSON response asynchronously.

        Args:
            prompt: User prompt string.
            schema: Pydantic model whose JSON schema describes the output.

        Returns:
            Parsed dictionary matching the schema.

        """
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        system_msg = (
            "You must respond with valid JSON that matches the following JSON schema. "
            "Do not include any additional text, markdown, or explanation.\n\n"
            f"Schema:\n{schema_json}"
        )
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("MiniMax response content is None")
        return json.loads(self._strip_think_tags(content))

    # -- utilities ------------------------------------------------------------

    @classmethod
    def _is_available(cls) -> bool:
        """Check if the required packages are available."""
        return (
            importutil.find_spec("pydantic") is not None
            and importutil.find_spec("openai") is not None
        )

    def __repr__(self) -> str:
        """Return a string representation of the MiniMaxGenie instance."""
        return f"MiniMaxGenie(model={self.model})"
