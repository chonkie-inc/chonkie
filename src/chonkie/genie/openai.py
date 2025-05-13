"""OpenAI Genie."""
import importlib.util as importutil
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

from .base import BaseGenie

if TYPE_CHECKING:
    try:
        from openai import OpenAI
    except ImportError:
        OpenAI = Any # type: ignore

    try:
        from pydantic import BaseModel
    except ImportError:
        BaseModel = Any # type: ignore


class OpenAIGenie(BaseGenie):
    """OpenAI's Genie."""

    def __init__(self,
                 model: str = "gpt-4.1",
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None):
        """Initialize the OpenAIGenie class.

        Args:
            model (str): The model to use.
            base_url (Optional[str]): The base URL to use. If not provided, the default OpenAI base URL will be used.
            api_key (Optional[str]): The API key to use. Defaults to env var OPENAI_API_KEY.

        """
        super().__init__()

        # Lazily import the dependencies
        self._import_dependencies()

        # Initialize the API key
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAIGenie requires an API key. Either pass the `api_key` parameter or set the `OPENAI_API_KEY` in your environment.")

        # Initialize the client and model
        if base_url is None:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        self.model = model

    def generate(self, prompt: str) -> str:
        """Generate a response based on the given prompt."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI response content is None")
        return content

    def generate_json(self, prompt: str, schema: "BaseModel") -> Dict[str, Any]:
        """Generate a JSON response based on the given prompt and schema."""
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=schema
        )
        content = response.choices[0].message.parsed
        if content is None:
            raise ValueError("OpenAI response content is None")
        return content.model_dump()

    def _is_available(self) -> bool:
        """Check if all the dependencies are available in the environement."""
        if (importutil.find_spec("pydantic") is not None
                and importutil.find_spec("openai") is not None):
            return True
        return False

    def _import_dependencies(self) -> None:
        """Import all the required dependencies."""
        if self._is_available():
            global OpenAI, BaseModel
            from openai import OpenAI
            from pydantic import BaseModel
        else:
            raise ImportError("One or more of the required modules are not available: [pydantic, openai]", "Please install the dependencies via `pip install chonkie[openai]`")

    def __repr__(self) -> str:
        """Return a string representation of the OpenAIGenie instance."""
        return f"OpenAIGenie(model={self.model})"