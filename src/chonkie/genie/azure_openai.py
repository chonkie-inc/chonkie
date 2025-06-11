"""Azure OpenAI Genie."""

import importlib.util as importutil
from typing import TYPE_CHECKING, Any, Dict, Optional

from .base import BaseGenie

if TYPE_CHECKING:
    from openai import AzureOpenAI
    from pydantic import BaseModel


class AzureOpenAIGenie(BaseGenie):
    """Azure-hosted OpenAI Genie wrapper."""

    def __init__(
        self,
        azure_endpoint: str,
        model: str = "gpt-4o",
        deployment: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        api_version: str = "2024-10-21",
    ):
        """Initialize AzureOpenAIGenie.

        Args:
            model (str): Logical model name, used for tokenization etc.
            deployment (Optional[str]): Azure deployment name (used for API call). Defaults to `model`.
            azure_endpoint (Optional[str]): Your Azure OpenAI endpoint.
            azure_api_key (Optional[str]): Your Azure OpenAI API key. If omitted, uses Entra ID.
            api_version (str): Required API version (default: "2024-10-21").

        """
        super().__init__()

        if not azure_endpoint:
            raise ValueError("`azure_endpoint` is required for Azure OpenAI.")

        self._import_dependencies()

        self.model = model
        self._deployment = deployment or model
        self.api_version = api_version
        self.base_url = azure_endpoint

        if azure_api_key:
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version=api_version,
            )
        else:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider

            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            )
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                azure_ad_token_provider=token_provider,
                api_version=api_version,
            )

    def generate(self, prompt: str) -> str:
        """Generate a plain-text response."""
        response = self.client.chat.completions.create(
            model=self._deployment, messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Azure OpenAI response content is None")
        return content

    def generate_json(self, prompt: str, schema: "BaseModel") -> Dict[str, Any]:
        """Generate a structured JSON response based on a Pydantic schema."""
        response = self.client.beta.chat.completions.parse(
            model=self._deployment,
            messages=[{"role": "user", "content": prompt}],
            response_format=schema,  # type: ignore[arg-type]
        )
        content = response.choices[0].message.parsed
        if content is None:
            raise ValueError("Azure OpenAI response content is None")
        return content.model_dump()

    def _is_available(self) -> bool:
        return (
            importutil.find_spec("openai") is not None
            and importutil.find_spec("pydantic") is not None
            and importutil.find_spec("azure.identity") is not None
        )

    def _import_dependencies(self) -> None:
        if self._is_available():
            global AzureOpenAI, BaseModel
            from openai import AzureOpenAI
            from pydantic import BaseModel
        else:
            raise ImportError(
                "Missing required modules: [openai, azure-identity, pydantic]. "
                "Install via `pip install chonkie[azure-openai]`."
            )

    def __repr__(self) -> str:
        """Return a string representation of the AzureOpenAIGenie instance."""
        return f"AzureOpenAIGenie(model={self.model}, deployment={self._deployment}, endpoint={self.base_url})"
