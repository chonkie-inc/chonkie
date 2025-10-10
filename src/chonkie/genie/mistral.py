"""Implementation of the MistralGenie class."""
import importlib.util as importutil
import json
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

from .base import BaseGenie

if TYPE_CHECKING:
    from pydantic import BaseModel


class MistralGenie(BaseGenie):
    """Mistral's Genie."""

    def __init__(
        self,
        model: str = "mistral-large-latest",
        api_key: Optional[str] = None
    ):
        """Initialize the MistralGenie class.

        Args:
            model (str): The model to use.
            api_key (Optional[str]): The API key to use.

        """
        super().__init__()

        # Lazily import the dependencies
        self._import_dependencies()

        # Initialize the API key
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "MistralGenie requires an API key. Either pass the `api_key` "
                "parameter or set the `MISTRAL_API_KEY` in your environment."
            )

        # Initialize the client and model
        self.client = MistralClient(api_key=self.api_key)  # type: ignore
        self.model = model

    def generate(self, prompt: str) -> str:
        """Generate a response based on the given prompt."""
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.complete(model=self.model, messages=messages)
        return str(response.choices[0].message.content)

    def generate_json(self, prompt: str, schema: "BaseModel") -> Dict[str, Any]:
        """Generate a JSON response based on the given prompt and schema."""
        json_schema = json.dumps(schema.model_json_schema(), indent=2)
        full_prompt = (
            f"{prompt}\n\n"
            "Please provide the output in a JSON format that strictly adheres to the "
            f"following schema:\n```json\n{json_schema}\n```"
        )
        messages = [{"role": "user", "content": full_prompt}]

        response = self.client.chat.complete(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
        )

        try:
            content = response.choices[0].message.content
            return dict(json.loads(content))
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}")

    def _is_available(self) -> bool:
        """Check if all the dependencies are available in the environment."""
        if (
            importutil.find_spec("pydantic") is not None
            and importutil.find_spec("mistralai") is not None
        ):
            return True
        return False

    def _import_dependencies(self) -> None:
        """Import all the required dependencies."""
        if self._is_available():
            global BaseModel, MistralClient
            from mistralai import Mistral as MistralClient
            from pydantic import BaseModel
        else:
            raise ImportError(
                "One or more of the required modules are not available: "
                "[pydantic, mistralai]"
            )

    def __repr__(self) -> str:
        """Return a string representation of the MistralGenie instance."""
        return f"MistralGenie(model={self.model})"
