"""Groq Genie."""

import importlib.util as importutil
import json
import os
from typing import TYPE_CHECKING, Any, Optional

from .base import BaseGenie

if TYPE_CHECKING:
    from pydantic import BaseModel


class GroqGenie(BaseGenie):
    """Groq's Genie - fast inference on Groq hardware."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
    ):
        """Initialize the GroqGenie class.

        Args:
            model (str): The model to use.
            api_key (Optional[str]): The API key to use. Defaults to env var GROQ_API_KEY.

        """
        super().__init__()

        try:
            from groq import Groq
        except ImportError as ie:
            raise ImportError(
                "One or more of the required modules are not available: [pydantic, groq]. "
                "Please install the dependencies via `pip install chonkie[groq]`"
            ) from ie

        # Initialize the API key
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GroqGenie requires an API key. Either pass the `api_key` parameter or set the `GROQ_API_KEY` in your environment.",
            )

        # Initialize the client and model
        self.client = Groq(api_key=self.api_key)
        self.model = model

    def _extract_content(self, response: Any) -> str:
        """Extract and validate content from API response."""
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Groq response content is None")
        return content

    def generate(self, prompt: str) -> str:
        """Generate a response based on the given prompt."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return self._extract_content(response)

    def generate_json(self, prompt: str, schema: "BaseModel") -> dict[str, Any]:
        """Generate a JSON response based on the given prompt and schema.

        Uses Groq's JSON schema support to enforce structured output.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema.model_json_schema(),
                },
            },
        )
        return json.loads(self._extract_content(response))

    @classmethod
    def _is_available(cls) -> bool:
        """Check if all the dependencies are available in the environment."""
        return (
            importutil.find_spec("pydantic") is not None
            and importutil.find_spec("groq") is not None
        )

    def __repr__(self) -> str:
        """Return a string representation of the GroqGenie instance."""
        return f"GroqGenie(model={self.model})"
