"""Cerebras Genie."""

import importlib.util as importutil
import json
import os
from typing import TYPE_CHECKING, Any, Optional

from .base import BaseGenie

if TYPE_CHECKING:
    from pydantic import BaseModel


class CerebrasGenie(BaseGenie):
    """Cerebras Genie - fastest inference on Cerebras hardware."""

    def __init__(
        self,
        model: str = "llama-3.3-70b",
        api_key: Optional[str] = None,
    ):
        """Initialize the CerebrasGenie class.

        Args:
            model (str): The model to use.
            api_key (Optional[str]): The API key to use. Defaults to env var CEREBRAS_API_KEY.

        """
        super().__init__()

        try:
            from cerebras.cloud.sdk import Cerebras
        except ImportError as ie:
            raise ImportError(
                "One or more of the required modules are not available: [pydantic, cerebras-cloud-sdk]. "
                "Please install the dependencies via `pip install chonkie[cerebras]`"
            ) from ie

        # Initialize the API key
        self.api_key = api_key or os.environ.get("CEREBRAS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "CerebrasGenie requires an API key. Either pass the `api_key` parameter or set the `CEREBRAS_API_KEY` in your environment.",
            )

        # Initialize the client and model
        self.client = Cerebras(api_key=self.api_key)
        self.model = model

    def _extract_content(self, response: Any) -> str:
        """Extract and validate content from API response."""
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Cerebras response content is None")
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

        Note: Cerebras currently supports basic JSON mode. The schema is included
        in the prompt to guide the model's output structure.
        """
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        enhanced_prompt = (
            f"{prompt}\n\nRespond with valid JSON matching this schema:\n{schema_json}"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": enhanced_prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(self._extract_content(response))

    @classmethod
    def _is_available(cls) -> bool:
        """Check if all the dependencies are available in the environment."""
        return (
            importutil.find_spec("pydantic") is not None
            and importutil.find_spec("cerebras.cloud.sdk") is not None
        )

    def __repr__(self) -> str:
        """Return a string representation of the CerebrasGenie instance."""
        return f"CerebrasGenie(model={self.model})"
