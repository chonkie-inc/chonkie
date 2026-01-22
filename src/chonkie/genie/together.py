"""TogetherGenie for TogetherAI's models."""
import importlib.util as importutil
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

from .base import BaseGenie

class TogetherGenie(BaseGenie):
    """Genie using TogetherAI's hosted models (OpenAI-compatible style)."""

    SUPPORTED_JSON_MODELS = {
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "deepseek-ai/DeepSeek-V3",
        "Qwen/Qwen3-235B-A22B-fp8-tput",
        "Qwen/Qwen2.5-VL-72B-Instruct",
    }

    def __init__(
        self,
        model: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        api_key: Optional[str] = None,
    ):
        """Initialize the TogetherGenie class.

        Args:
            model (str): The TogetherAI model to use.
            api_key (Optional[str]): The API key to use. Defaults to the environment variable TOGETHER_API_KEY.
        """
        super().__init__()
        try:
            from pydantic import BaseModel
            from together import Together
        except ImportError as e:
            raise ImportError(
                "TogetherGenie requires `together` and `pydantic`. Install with `pip install chonkie[togetherai]`"
            ) from e

        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "TogetherGenie requires an API key. Pass `api_key` or set TOGETHER_API_KEY env var."
            )

        self.model = model
        self.client = Together(api_key=self.api_key)

    def generate(self, prompt: str) -> str:
        """Generate a basic chat completion."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("TogetherAI returned empty response content.")
        return content

    def generate_json(self, prompt: str, schema: "BaseModel") -> Dict[str, Any]:
        """Generate a JSON-formatted response using Together's structured output feature.

        Note:
            This only works with Together models that support JSON mode.
            See: https://docs.together.ai/docs/json-mode

        """     
        if self.model not in self.SUPPORTED_JSON_MODELS:
            raise ValueError(f"Model `{self.model}` does not support JSON mode.")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format={
                "type": "json_object",
                "schema": schema.model_json_schema(),
            },
        )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("TogetherAI returned no content in the JSON response.")

        return schema.model_validate_json(content).model_dump()

    def _is_available(self) -> bool:
        return (
            importutil.find_spec("together") is not None
            and importutil.find_spec("pydantic") is not None
        )

    def __repr__(self) -> str:
        return f"TogetherGenie(model={self.model})"
