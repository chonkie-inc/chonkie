"""Cerebras Genie."""

import importlib.util as importutil
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

    def generate(self, prompt: str) -> str:
        """Generate a response based on the given prompt."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Cerebras response content is None")
        return content

    def generate_json(self, prompt: str, schema: "BaseModel") -> dict[str, Any]:
        """Generate a JSON response based on the given prompt and schema."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Cerebras response content is None")
        import json

        return json.loads(content)

    @classmethod
    def _is_available(cls) -> bool:
        """Check if all the dependencies are available in the environment."""
        if (
            importutil.find_spec("pydantic") is not None
            and importutil.find_spec("cerebras") is not None
        ):
            return True
        return False

    def __repr__(self) -> str:
        """Return a string representation of the CerebrasGenie instance."""
        return f"CerebrasGenie(model={self.model})"
