"""Implementation of the GeminiGenie class."""

import importlib.util as importutil
import json
import os
from typing import TYPE_CHECKING, Any, Optional

from .base import BaseGenie

if TYPE_CHECKING:
    from pydantic import BaseModel


class GeminiGenie(BaseGenie):
    """Gemini's Genie."""

    def __init__(self, model: str = "gemini-3-pro-preview", api_key: Optional[str] = None):
        """Initialize the GeminiGenie class.

        Args:
            model (str): The model to use.
            api_key (Optional[str]): The API key to use.

        """
        super().__init__()

        # Initialize the API key
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GeminiGenie requires an API key. Either pass the `api_key` parameter or set the `GEMINI_API_KEY` in your environment.",
            )

        try:
            from google.genai import Client as GenAIClient
        except ImportError as ie:
            raise ImportError(
                "One or more of the required modules are not available: [google-genai]. "
                "Please install the dependencies via `pip install chonkie[gemini]`"
            ) from ie

        # Initialize the client and model
        self.client = GenAIClient(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        """Generate a response based on the given prompt."""
        response = self.client.models.generate_content(model=self.model, contents=prompt)
        return str(response.text)

    def generate_json(self, prompt: str, schema: "BaseModel") -> dict[str, Any]:
        """Generate a JSON response based on the given prompt and schema."""
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": schema,
            },
        )
        try:
            return dict(json.loads(response.text or ""))
        except Exception as e:
            raise ValueError(f"Failed to parse JSON response: {e}")

    @classmethod
    def _is_available(cls) -> bool:
        """Check if all the dependencies are available in the environment."""
        if (
            importutil.find_spec("pydantic") is not None
            and importutil.find_spec("google") is not None
        ):
            return True
        return False

    def __repr__(self) -> str:
        """Return a string representation of the GeminiGenie instance."""
        return f"GeminiGenie(model={self.model})"
