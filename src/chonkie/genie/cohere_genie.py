"""Cohere Genie."""
import importlib.util as importutil
import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .base import BaseGenie

if TYPE_CHECKING:
    try:
        from cohere import Client
    except ImportError:
        Client = Any  # type: ignore

    try:
        from pydantic import BaseModel
    except ImportError:
        BaseModel = Any  # type: ignore


class CohereGenie(BaseGenie):
    """Cohere's Genie.
    
    Supports Cohere's Command series of models such as:
    - Command
    - Command Light
    - Command R
    - Command R+
    """

    def __init__(self,
                 model: str = "command",
                 api_key: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 1024):
        """Initialize the CohereGenie class.

        Args:
            model (str): The model to use. Defaults to "command".
            api_key (Optional[str]): The API key to use. Defaults to env var COHERE_API_KEY.
            temperature (float): The temperature to use for generation. Defaults to 0.7.
            max_tokens (int): The maximum number of tokens to generate. Defaults to 1024.
        """
        super().__init__()

        # Lazily import the dependencies
        self._import_dependencies()

        # Initialize the API key
        self.api_key = api_key or os.environ.get("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("CohereGenie requires an API key. Either pass the `api_key` parameter or set the `COHERE_API_KEY` in your environment.")

        # Initialize the client and model
        self.client = Client(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        """Generate a response based on the given prompt."""
        response = self.client.chat(
            model=self.model,
            message=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        text = response.text
        if text is None or text == "":
            raise ValueError("Cohere response content is empty")
        return text

    def generate_json(self, prompt: str, schema: "BaseModel") -> Dict[str, Any]:
        """Generate a JSON response based on the given prompt and schema."""
        # Construct a prompt that asks for a JSON response according to the schema
        schema_json = schema.schema_json()
        json_prompt = f"{prompt}\n\nPlease respond with valid JSON that conforms to this schema:\n{schema_json}"
        
        response = self.client.chat(
            model=self.model,
            message=json_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format="json"
        )
        
        text = response.text
        if text is None or text == "":
            raise ValueError("Cohere response content is empty")
        
        try:
            return dict(json.loads(text))
        except Exception as e:
            raise ValueError(f"Failed to parse JSON response: {e}")

    def _is_available(self) -> bool:
        """Check if all the dependencies are available in the environment."""
        if (importutil.find_spec("pydantic") is not None
                and importutil.find_spec("cohere") is not None):
            return True
        return False

    def _import_dependencies(self) -> None:
        """Import all the required dependencies."""
        if self._is_available():
            global Client, BaseModel
            from cohere import Client
            from pydantic import BaseModel
        else:
            raise ImportError("One or more of the required modules are not available: [pydantic, cohere]", 
                             "Please install the dependencies via `pip install chonkie[cohere]`")

    def __repr__(self) -> str:
        """Return a string representation of the CohereGenie instance."""
        return f"CohereGenie(model={self.model})" 