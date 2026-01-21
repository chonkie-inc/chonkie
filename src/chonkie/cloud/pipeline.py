"""Cloud Pipeline for Chonkie API."""

import builtins
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import httpx

from chonkie.types import Chunk

from .file import FileManager

# Slug validation pattern: lowercase letters, numbers, dashes, underscores
SLUG_PATTERN = re.compile(r"^[a-z0-9_-]+$")


@dataclass
class PipelineStep:
    """A single step in a pipeline configuration."""

    type: str
    component: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API payload."""
        return {
            "type": self.type,
            "component": self.component,
            **self.params,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineStep":
        """Create from dictionary."""
        step_type = data.get("type", "")
        component = data.get("component", "")
        params = {k: v for k, v in data.items() if k not in ["type", "component"]}
        return cls(type=step_type, component=component, params=params)


class Pipeline:
    """Cloud Pipeline - build and execute pipelines via Chonkie API.

    Examples:
        ```python
        from chonkie.cloud import Pipeline

        # Create new pipeline
        pipeline = Pipeline(
            slug="my-rag-pipeline",
            description="Pipeline for RAG processing"
        ).chunk_with("recursive", chunk_size=512).refine_with("overlap", context_size=64)

        # Execute with text
        chunks = pipeline.run(text="Your document text here")

        # Execute with file (auto-uploaded)
        chunks = pipeline.run(file="document.pdf")

        # Fetch existing pipeline
        pipeline = Pipeline.get("my-rag-pipeline")
        chunks = pipeline.run(text="...")

        # List all pipelines
        for p in Pipeline.list():
            print(f"{p.slug}: {p.describe()}")
        ```

    """

    BASE_URL = "https://api.chonkie.ai"
    VERSION = "v1"

    def __init__(
        self,
        slug: str,
        description: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize a new cloud Pipeline.

        Args:
            slug: Unique identifier for the pipeline. Must contain only lowercase
                  letters, numbers, dashes, and underscores.
            description: Optional description of the pipeline.
            api_key: Chonkie API key. If None, reads from CHONKIE_API_KEY env var.

        Raises:
            ValueError: If slug format is invalid or API key is not provided.

        """
        # Validate API key
        self._api_key = api_key or os.getenv("CHONKIE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "No API key provided. Please set the CHONKIE_API_KEY environment "
                "variable or pass an api_key to the Pipeline constructor.",
            )

        # Validate slug format
        if not SLUG_PATTERN.match(slug):
            raise ValueError(
                f"Invalid slug '{slug}'. Slug must contain only lowercase letters, "
                "numbers, dashes, and underscores.",
            )

        self._slug = slug
        self._description = description
        self._steps: list[PipelineStep] = []
        self._is_saved = False
        self._id: Optional[str] = None
        self._created_at: Optional[str] = None
        self._updated_at: Optional[str] = None

        # Initialize file manager for file uploads
        self._file_manager = FileManager(api_key=self._api_key)

    @property
    def slug(self) -> str:
        """Return the pipeline slug."""
        return self._slug

    @property
    def description(self) -> Optional[str]:
        """Return the pipeline description."""
        return self._description

    @property
    def is_saved(self) -> bool:
        """Return True if pipeline exists in cloud."""
        return self._is_saved

    @property
    def steps(self) -> list[PipelineStep]:
        """Return the pipeline steps."""
        return self._steps.copy()

    def _get_headers(self) -> dict[str, str]:
        """Get headers for API requests."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    @classmethod
    def get(cls, slug: str, api_key: Optional[str] = None) -> "Pipeline":
        """Fetch an existing pipeline from the cloud.

        Args:
            slug: The pipeline slug to fetch.
            api_key: Chonkie API key. If None, reads from CHONKIE_API_KEY env var.

        Returns:
            Pipeline instance with the fetched configuration.

        Raises:
            ValueError: If pipeline not found or API error occurs.

        """
        api_key = api_key or os.getenv("CHONKIE_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key provided. Please set the CHONKIE_API_KEY environment "
                "variable or pass an api_key.",
            )

        response = httpx.get(
            f"{cls.BASE_URL}/{cls.VERSION}/pipeline/{slug}",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        if response.status_code == 404:
            raise ValueError(f"Pipeline '{slug}' not found.")
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch pipeline: {response.text}")

        data = response.json()

        # Create pipeline instance
        pipeline = cls(
            slug=data["slug"],
            description=data.get("description"),
            api_key=api_key,
        )

        # Populate from response
        pipeline._is_saved = True
        pipeline._id = data.get("id")
        pipeline._created_at = data.get("created_at")
        pipeline._updated_at = data.get("updated_at")
        pipeline._steps = [PipelineStep.from_dict(step) for step in data.get("steps", [])]

        return pipeline

    @classmethod
    def list(cls, api_key: Optional[str] = None) -> list["Pipeline"]:
        """List all pipelines from the cloud.

        Args:
            api_key: Chonkie API key. If None, reads from CHONKIE_API_KEY env var.

        Returns:
            List of Pipeline instances.

        Raises:
            ValueError: If API error occurs.

        """
        api_key = api_key or os.getenv("CHONKIE_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key provided. Please set the CHONKIE_API_KEY environment "
                "variable or pass an api_key.",
            )

        response = httpx.get(
            f"{cls.BASE_URL}/{cls.VERSION}/pipeline",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        if response.status_code != 200:
            raise ValueError(f"Failed to list pipelines: {response.text}")

        data = response.json()
        pipelines = []

        for p in data.get("pipelines", []):
            pipeline = cls(
                slug=p["slug"],
                description=p.get("description"),
                api_key=api_key,
            )
            pipeline._is_saved = True
            pipeline._id = p.get("id")
            pipeline._created_at = p.get("created_at")
            pipeline._updated_at = p.get("updated_at")
            pipeline._steps = [PipelineStep.from_dict(step) for step in p.get("steps", [])]
            pipelines.append(pipeline)

        return pipelines

    @classmethod
    def validate(
        cls,
        steps: builtins.list[dict[str, Any]],
        api_key: Optional[str] = None,
    ) -> tuple[bool, Optional[builtins.list[str]]]:
        """Validate a pipeline configuration via the cloud API.

        Args:
            steps: List of step configurations to validate.
            api_key: Chonkie API key. If None, reads from CHONKIE_API_KEY env var.

        Returns:
            Tuple of (is_valid, errors). errors is None if valid.

        Raises:
            ValueError: If API error occurs.

        """
        api_key = api_key or os.getenv("CHONKIE_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key provided. Please set the CHONKIE_API_KEY environment "
                "variable or pass an api_key.",
            )

        # Convert to API format
        formatted_steps = []
        for step in steps:
            if isinstance(step, PipelineStep):
                formatted_steps.append(step.to_dict())
            elif isinstance(step, dict):
                formatted_steps.append(step)
            else:
                raise ValueError(f"Invalid step format: {type(step)}")

        response = httpx.post(
            f"{cls.BASE_URL}/{cls.VERSION}/pipeline/validate",
            json={"steps": formatted_steps},
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        if response.status_code != 200:
            raise ValueError(f"Validation request failed: {response.text}")

        data = response.json()
        return data.get("valid", False), data.get("errors")

    def chunk_with(self, chunker_type: str, **kwargs: Any) -> "Pipeline":
        """Add a chunking step to the pipeline.

        Args:
            chunker_type: Type of chunker (e.g., "recursive", "semantic", "sentence").
            **kwargs: Chunker parameters (e.g., chunk_size=512).

        Returns:
            Pipeline instance for method chaining.

        Example:
            ```python
            pipeline.chunk_with("recursive", chunk_size=512, tokenizer="gpt2")
            ```

        """
        self._steps.append(PipelineStep(type="chunk", component=chunker_type, params=kwargs))
        return self

    def refine_with(self, refinery_type: str, **kwargs: Any) -> "Pipeline":
        """Add a refinement step to the pipeline.

        Args:
            refinery_type: Type of refinery (e.g., "overlap", "embeddings").
            **kwargs: Refinery parameters.

        Returns:
            Pipeline instance for method chaining.

        Example:
            ```python
            pipeline.refine_with("overlap", context_size=64)
            pipeline.refine_with("embeddings", embedding_model="text-embedding-3-small")
            ```

        """
        self._steps.append(PipelineStep(type="refine", component=refinery_type, params=kwargs))
        return self

    def process_with(self, chef_type: str, **kwargs: Any) -> "Pipeline":
        """Add a processing step to the pipeline.

        Args:
            chef_type: Type of chef/processor (e.g., "markdown", "text").
            **kwargs: Processor parameters.

        Returns:
            Pipeline instance for method chaining.

        Example:
            ```python
            pipeline.process_with("markdown")
            ```

        """
        self._steps.append(PipelineStep(type="process", component=chef_type, params=kwargs))
        return self

    def _save(self) -> "Pipeline":
        """Save the pipeline to the cloud (create new).

        Returns:
            Pipeline instance for method chaining.

        Raises:
            ValueError: If pipeline already exists or API error occurs.

        """
        if not self._steps:
            raise ValueError("Cannot save pipeline with no steps.")

        payload = {
            "slug": self._slug,
            "description": self._description,
            "steps": [step.to_dict() for step in self._steps],
        }

        response = httpx.post(
            f"{self.BASE_URL}/{self.VERSION}/pipeline",
            json=payload,
            headers=self._get_headers(),
        )

        if response.status_code != 200:
            # Pipeline might already exist, try to update instead
            return self.update()

        data = response.json()
        self._is_saved = True
        self._id = data.get("id")
        self._created_at = data.get("created_at")
        self._updated_at = data.get("updated_at")

        return self

    def update(self, description: Optional[str] = None) -> "Pipeline":
        """Update the pipeline in the cloud.

        Args:
            description: New description (optional).

        Returns:
            Pipeline instance for method chaining.

        Raises:
            ValueError: If pipeline doesn't exist or API error occurs.

        """
        payload: dict[str, Any] = {}

        if description is not None:
            payload["description"] = description
            self._description = description

        if self._steps:
            payload["steps"] = [step.to_dict() for step in self._steps]

        if not payload:
            return self  # Nothing to update

        response = httpx.put(
            f"{self.BASE_URL}/{self.VERSION}/pipeline/{self._slug}",
            json=payload,
            headers=self._get_headers(),
        )

        if response.status_code == 404:
            raise ValueError(f"Pipeline '{self._slug}' not found.")
        if response.status_code != 200:
            raise ValueError(f"Failed to update pipeline: {response.text}")

        data = response.json()
        self._is_saved = True
        self._updated_at = data.get("updated_at")

        return self

    def delete(self) -> None:
        """Delete the pipeline from the cloud.

        Raises:
            ValueError: If pipeline doesn't exist or API error occurs.

        """
        response = httpx.delete(
            f"{self.BASE_URL}/{self.VERSION}/pipeline/{self._slug}",
            headers=self._get_headers(),
        )

        if response.status_code == 404:
            raise ValueError(f"Pipeline '{self._slug}' not found.")
        if response.status_code != 200:
            raise ValueError(f"Failed to delete pipeline: {response.text}")

        self._is_saved = False
        self._id = None

    def run(
        self,
        text: Optional[Union[str, builtins.list[str]]] = None,
        file: Optional[str] = None,
    ) -> builtins.list[Chunk]:
        """Execute the pipeline via the cloud API.

        Args:
            text: Text to process. Can be a single string or list of strings.
            file: Path to file to process. Will be uploaded automatically.

        Returns:
            List of Chunk objects.

        Raises:
            ValueError: If neither text nor file provided, or API error occurs.

        """
        if text is None and file is None:
            raise ValueError("Either 'text' or 'file' must be provided.")
        if text is not None and file is not None:
            raise ValueError("Cannot provide both 'text' and 'file'.")

        # Save pipeline if not already saved
        if not self._is_saved:
            self._save()

        # Build payload
        payload: dict[str, Any] = {}

        if file is not None:
            # Upload file first
            uploaded = self._file_manager.upload(file)
            payload["file"] = {"type": "document", "content": uploaded.name}
        else:
            payload["text"] = text

        # Execute pipeline
        response = httpx.post(
            f"{self.BASE_URL}/{self.VERSION}/pipeline/{self._slug}",
            json=payload,
            headers=self._get_headers(),
        )

        if response.status_code == 404:
            raise ValueError(f"Pipeline '{self._slug}' not found.")
        if response.status_code != 200:
            raise ValueError(f"Pipeline execution failed: {response.text}")

        data = response.json()
        chunks = [Chunk.from_dict(chunk) for chunk in data.get("chunks", [])]

        return chunks

    def to_config(self) -> builtins.list[dict[str, Any]]:
        """Export pipeline configuration as a list of step dictionaries.

        Returns:
            List of step configurations.

        """
        return [step.to_dict() for step in self._steps]

    def describe(self) -> str:
        """Get a human-readable description of the pipeline.

        Returns:
            String description of the pipeline steps.

        """
        if not self._steps:
            return "Empty pipeline"

        descriptions = []
        for step in self._steps:
            descriptions.append(f"{step.type}({step.component})")

        return " -> ".join(descriptions)

    def reset(self) -> "Pipeline":
        """Reset the pipeline steps.

        Returns:
            Pipeline instance for method chaining.

        """
        self._steps.clear()
        return self

    def __repr__(self) -> str:
        """Return string representation of the pipeline."""
        saved_status = "saved" if self._is_saved else "not saved"
        return f"Pipeline(slug='{self._slug}', {saved_status}, {self.describe()})"
