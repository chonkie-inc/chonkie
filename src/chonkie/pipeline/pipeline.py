"""Core Pipeline class for chonkie."""

import inspect
import json
from pathlib import Path
from typing import Any, Optional, Union

from chonkie.types import Document
from chonkie.utils import Hubbie

from .registry import ComponentRegistry


class Pipeline:
    """A fluent API for building and executing chonkie pipelines.

    The Pipeline class provides a clean, chainable interface for processing
    documents through the CHOMP pipeline: CHef -> CHunker -> Refinery -> Porter/Handshake.

    Examples:
        ```python
        from chonkie.pipeline import Pipeline

        # Simple pipeline with single file - returns Document with chunks
        doc = (Pipeline()
            .fetch_from("file", path="document.txt")
            .process_with("text")
            .chunk_with("recursive", chunk_size=512)
            .run())

        # Access chunks via doc.chunks
        for chunk in doc.chunks:
            print(chunk.text)

        # Process multiple files from directory - returns list[Document]
        docs = (Pipeline()
            .fetch_from("file", dir="./docs", ext=[".txt", ".md"])
            .process_with("text")
            .chunk_with("recursive", chunk_size=512)
            .run())

        # Complex pipeline with refinement and export
        doc = (Pipeline()
            .fetch_from("file", path="document.txt")
            .process_with("text")
            .chunk_with("recursive", chunk_size=512)
            .refine_with("overlap", context_size=50)
            .export_with("json", file="chunks.json")
            .run())
        ```

    """

    def __init__(self) -> None:
        """Initialize a new Pipeline."""
        self._steps: list[dict[str, Any]] = []
        self._component_instances: dict[
            tuple[str, str],
            Any,
        ] = {}  # Cache: (name, json_kwargs) -> instance

    @classmethod
    def from_recipe(cls, name: str, path: Optional[str] = None) -> "Pipeline":
        """Create pipeline from a pre-defined recipe.

        Recipes are loaded from the Chonkie Hub (chonkie-ai/recipes repo)
        under the 'pipelines' subfolder.

        Args:
            name: Name of the pipeline recipe (e.g., 'markdown')
            path: Optional local path to recipe file (overrides hub download)

        Returns:
            Configured Pipeline instance

        Raises:
            ValueError: If recipe is not found or invalid
            ImportError: If huggingface_hub is not installed

        Examples:
            ```python
            # Load from hub
            pipeline = Pipeline.from_recipe('markdown')

            # Load from local file
            pipeline = Pipeline.from_recipe('custom', path='my_recipe.json')

            # Run the pipeline
            doc = pipeline.run(texts='Your markdown here')
            ```

        """
        # Create Hubbie instance to load recipe
        hubbie = Hubbie()
        recipe = hubbie.get_pipeline_recipe(name, path=path)

        # Extract steps from recipe
        steps = recipe.get("steps", [])
        if not steps:
            raise ValueError(f"Pipeline recipe '{name}' has no steps defined.")

        # Create pipeline from steps
        return cls.from_config(steps)

    @classmethod
    def from_config(
        cls,
        config: Union[str, list[Union[tuple[Any, ...], dict[str, Any]]]],
    ) -> "Pipeline":
        """Create pipeline from config list or JSON file path.

        Args:
            config: Either a list of step configs or path to JSON file

        Returns:
            Configured Pipeline instance

        Raises:
            ValueError: If config format is invalid
            FileNotFoundError: If config file path doesn't exist

        Examples:
            ```python
            # From list
            Pipeline.from_config([
                ('chunk', 'token', {'chunk_size': 512}),
                ('refine', 'overlap', {'context_size': 50})
            ])

            # From file
            Pipeline.from_config('pipeline.json')
            ```

        """
        # If string, load from file
        if isinstance(config, str):
            import json

            with open(config, "r") as f:
                config_data = json.load(f)
        else:
            config_data = config

        # Build pipeline from steps
        pipeline = cls()

        for i, step in enumerate(config_data):
            try:
                # Handle both tuple and dict formats
                if isinstance(step, (tuple, list)):
                    if len(step) == 3:
                        step_type, component_name, kwargs = step
                    elif len(step) == 2:
                        step_type, component_name = step
                        kwargs = {}
                    else:
                        raise ValueError(f"Tuple must have 2 or 3 elements, got {len(step)}")
                elif isinstance(step, dict):
                    step_type = step.get("type")
                    component_name = step.get("component")
                    if not step_type or not component_name:
                        raise ValueError("Dict must have 'type' and 'component' keys")
                    kwargs = {k: v for k, v in step.items() if k not in ["type", "component"]}
                else:
                    raise ValueError(f"Step must be tuple or dict, got {type(step)}")

                # Map to appropriate method
                if step_type == "fetch":
                    pipeline.fetch_from(component_name, **kwargs)
                elif step_type == "process":
                    pipeline.process_with(component_name, **kwargs)
                elif step_type == "chunk":
                    pipeline.chunk_with(component_name, **kwargs)
                elif step_type == "refine":
                    pipeline.refine_with(component_name, **kwargs)
                elif step_type == "export":
                    pipeline.export_with(component_name, **kwargs)
                elif step_type == "write":
                    pipeline.store_in(component_name, **kwargs)
                else:
                    raise ValueError(f"Unknown step type: '{step_type}'")

            except Exception as e:
                raise ValueError(f"Error processing step {i + 1}: {e}") from e

        return pipeline

    def fetch_from(self, source_type: str, **kwargs: Any) -> "Pipeline":
        """Fetch data from a source.

        Args:
            source_type: Type of source fetcher to use (e.g., "file")
            **kwargs: Arguments passed to the fetcher component

        Returns:
            Pipeline instance for method chaining

        Raises:
            ValueError: If source_type is not a registered fetcher

        Examples:
            ```python
            # Single file
            pipeline.fetch_from("file", path="document.txt")

            # Directory with extension filter
            pipeline.fetch_from("file", dir="./docs", ext=[".txt", ".md"])
            ```

        """
        component = ComponentRegistry.get_fetcher(source_type)
        self._steps.append({"type": "fetch", "component": component, "kwargs": kwargs})
        return self

    def process_with(self, chef_type: str, **kwargs: Any) -> "Pipeline":
        """Process data with a chef component.

        Args:
            chef_type: Type of chef to use (e.g., "text")
            **kwargs: Arguments passed to the chef component

        Returns:
            Pipeline instance for method chaining

        Raises:
            ValueError: If chef_type is not a registered chef

        Example:
            ```python
            pipeline.process_with("text", clean_whitespace=True)
            ```

        """
        component = ComponentRegistry.get_chef(chef_type)
        self._steps.append({
            "type": "process",
            "component": component,
            "kwargs": kwargs,
        })
        return self

    def chunk_with(self, chunker_type: str, **kwargs: Any) -> "Pipeline":
        """Chunk data with a chunker component.

        Args:
            chunker_type: Type of chunker to use (e.g., "recursive", "semantic")
            **kwargs: Arguments passed to the chunker component

        Returns:
            Pipeline instance for method chaining

        Raises:
            ValueError: If chunker_type is not a registered chunker

        Example:
            ```python
            pipeline.chunk_with("recursive", chunk_size=512, chunk_overlap=50)
            ```

        """
        component = ComponentRegistry.get_chunker(chunker_type)
        self._steps.append({"type": "chunk", "component": component, "kwargs": kwargs})
        return self

    def refine_with(self, refinery_type: str, **kwargs: Any) -> "Pipeline":
        """Refine chunks with a refinery component.

        Args:
            refinery_type: Type of refinery to use (e.g., "overlap", "embedding")
            **kwargs: Arguments passed to the refinery component

        Returns:
            Pipeline instance for method chaining

        Raises:
            ValueError: If refinery_type is not a registered refinery

        Example:
            ```python
            pipeline.refine_with("overlap", merge_threshold=0.8)
            ```

        """
        component = ComponentRegistry.get_refinery(refinery_type)
        self._steps.append({"type": "refine", "component": component, "kwargs": kwargs})
        return self

    def export_with(self, porter_type: str, **kwargs: Any) -> "Pipeline":
        """Export chunks with a porter component.

        Args:
            porter_type: Type of porter to use (e.g., "json", "datasets")
            **kwargs: Arguments passed to the porter component

        Returns:
            Pipeline instance for method chaining

        Raises:
            ValueError: If porter_type is not a registered porter

        Example:
            ```python
            pipeline.export_with("json", output_path="chunks.json")
            ```

        """
        component = ComponentRegistry.get_porter(porter_type)
        self._steps.append({"type": "export", "component": component, "kwargs": kwargs})
        return self

    def store_in(self, handshake_type: str, **kwargs: Any) -> "Pipeline":
        """Store chunks in a vector database with a handshake component.

        Args:
            handshake_type: Type of handshake to use (e.g., "chroma", "qdrant")
            **kwargs: Arguments passed to the handshake component

        Returns:
            Pipeline instance for method chaining

        Raises:
            ValueError: If handshake_type is not a registered handshake

        Example:
            ```python
            pipeline.store_in("chroma", collection_name="documents")
            ```

        """
        component = ComponentRegistry.get_handshake(handshake_type)
        self._steps.append({"type": "write", "component": component, "kwargs": kwargs})
        return self

    def run(
        self,
        texts: Optional[Union[str, list[str]]] = None,
    ) -> Union[Document, list[Document]]:
        """Run the pipeline and return the final result.

        The pipeline automatically reorders steps according to the CHOMP flow:
        Fetcher -> Chef -> Chunker -> Refinery(ies) -> Porter/Handshake

        This allows components to be defined in any order during pipeline building,
        but ensures correct execution order.

        Args:
            texts: Optional text input. Can be a single string or list of strings.
                   When provided, the fetcher step becomes optional.

        Returns:
            Document or list[Document] with processed chunks

        Raises:
            ValueError: If pipeline has no steps or invalid step configuration
            RuntimeError: If pipeline execution fails

        Examples:
            ```python
            # Single file pipeline - returns Document
            pipeline = (Pipeline()
                .fetch_from("file", path="doc.txt")
                .process_with("text")
                .chunk_with("recursive", chunk_size=512))
            doc = pipeline.run()
            print(f"Chunked into {len(doc.chunks)} chunks")

            # Directory pipeline - returns list[Document]
            pipeline = (Pipeline()
                .fetch_from("file", dir="./docs", ext=[".txt", ".md"])
                .process_with("text")
                .chunk_with("recursive", chunk_size=512))
            docs = pipeline.run()
            for doc in docs:
                print(f"File chunked into {len(doc.chunks)} chunks")

            # Direct text input (fetcher optional)
            pipeline = (Pipeline()
                .process_with("text")
                .chunk_with("recursive", chunk_size=512))
            doc = pipeline.run(texts="Hello world")

            # Access chunks via doc.chunks
            for chunk in doc.chunks:
                print(chunk.text)

            # Multiple texts - returns list[Document]
            docs = pipeline.run(texts=["Text 1", "Text 2", "Text 3"])
            all_chunks = [chunk for doc in docs for chunk in doc.chunks]
            ```

        """
        if not self._steps:
            raise ValueError("Pipeline has no steps to execute")

        # Reorder steps according to CHOMP flow (adds default TextChef if needed)
        ordered_steps = self._reorder_steps()

        # Validate after reordering (when we know the final structure)
        self._validate_pipeline(ordered_steps, has_text_input=(texts is not None))

        # Execute pipeline steps
        data = texts  # Start with input texts (or None for fetcher-based pipelines)
        for i, step in enumerate(ordered_steps):
            # Skip fetcher if we have direct text input
            if texts is not None and step["type"] == "fetch":
                continue

            try:
                data = self._execute_step(step, data)
            except Exception as e:
                raise RuntimeError(f"Pipeline failed at step {i + 1} ({step['type']}): {e}") from e

        return data  # type: ignore[return-value]

    def _reorder_steps(self) -> list[dict[str, Any]]:
        """Reorder pipeline steps according to CHOMP flow.

        Automatically adds a default TextChef if no chef is present.

        Returns:
            List of steps in correct execution order

        """
        # Group steps by type
        steps_by_type: dict[str, list[dict[str, Any]]] = {}
        for step in self._steps:
            step_type = step["type"]
            if step_type not in steps_by_type:
                steps_by_type[step_type] = []
            steps_by_type[step_type].append(step)

        # Add default TextChef if no chef is present
        if "process" not in steps_by_type:
            text_chef = ComponentRegistry.get_chef("text")
            steps_by_type["process"] = [{"type": "process", "component": text_chef, "kwargs": {}}]

        # Build ordered list following CHOMP: Fetch -> Process -> Chunk -> Refine -> Export/Write
        ordered = []
        for step_type in ["fetch", "process", "chunk", "refine", "export", "write"]:
            if step_type in steps_by_type:
                if step_type == "process":
                    # Only one chef allowed - use the last one if multiple
                    ordered.append(steps_by_type[step_type][-1])
                else:
                    # Multiple allowed - preserve order
                    ordered.extend(steps_by_type[step_type])

        return ordered

    def _validate_pipeline(
        self,
        ordered_steps: list[dict[str, Any]],
        has_text_input: bool = False,
    ) -> None:
        """Validate that the pipeline configuration is valid.

        Args:
            ordered_steps: Steps in execution order (after reordering)
            has_text_input: Whether direct text input is provided

        Raises:
            ValueError: If pipeline configuration is invalid

        """
        step_types = [step["type"] for step in ordered_steps]

        # Must have a chunker
        if "chunk" not in step_types:
            raise ValueError("Pipeline must include a chunker component (use chunk_with())")

        # Must have fetcher OR text input
        if not has_text_input and "fetch" not in step_types:
            raise ValueError(
                "Pipeline must include a fetcher component (use fetch_from()) "
                "or provide text input to run(texts=...)",
            )

        # Only one chef allowed (enforced during reordering, but double-check user's input)
        user_process_count = sum(1 for step in self._steps if step["type"] == "process")
        if user_process_count > 1:
            raise ValueError(
                f"Multiple process steps found ({user_process_count}). Only one chef is allowed per pipeline.",
            )

    def _execute_step(self, step: dict[str, Any], input_data: Any) -> Any:
        """Execute a single pipeline step.

        Args:
            step: Step configuration dictionary
            input_data: Input data from previous step

        Returns:
            Output data from this step

        """
        component_info = step["component"]
        kwargs = step["kwargs"]
        step_type = step["type"]

        # Extract recipe parameters before splitting (they're meta-parameters, not component params)
        recipe_name = kwargs.pop("recipe", None)
        recipe_lang = kwargs.pop("lang", "en")

        # Split parameters into init vs call kwargs
        init_kwargs, call_kwargs = Pipeline._split_parameters(
            component_info.component_class,
            kwargs,
            step_type,
        )

        # Create cache key for component instance
        try:
            kwargs_json = json.dumps(init_kwargs, sort_keys=True)
        except (TypeError, ValueError):
            kwargs_json = repr(sorted(init_kwargs.items()))

        component_key = (component_info.name, kwargs_json)

        # Get or create component instance
        if component_key not in self._component_instances:
            if recipe_name and hasattr(component_info.component_class, "from_recipe"):
                self._component_instances[component_key] = (
                    component_info.component_class.from_recipe(
                        name=recipe_name,
                        lang=recipe_lang,
                        **init_kwargs,
                    )
                )
            else:
                self._component_instances[component_key] = component_info.component_class(
                    **init_kwargs,
                )

        # Execute the component
        return self._call_component(
            self._component_instances[component_key],
            step_type,
            input_data,
            call_kwargs,
        )

    @staticmethod
    def _get_positional_params(step_type: str) -> set[str]:
        """Get parameter names that are passed as positional input_data for each step type.

        These parameters are excluded from method kwargs because they come from
        the previous pipeline step's output, not from user-provided kwargs.

        Args:
            step_type: Type of pipeline step

        Returns:
            Set of parameter names that are positional for this step type

        """
        positional_params = {
            "fetch": set(),  # Fetch has no input_data, all params are kwargs
            "process": {"path", "text"},  # process(path) or parse(text)
            "chunk": {"text", "document"},  # chunk methods take document
            "refine": {"chunks", "document"},  # refine methods take document
            "export": {"chunks"},  # export(chunks)
            "write": {"chunks"},  # write(chunks)
        }
        return positional_params.get(step_type, set())

    @staticmethod
    def _split_parameters(
        component_class: type[Any],
        kwargs: dict[str, Any],
        step_type: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Split kwargs into init and call parameters based on method signatures.

        Args:
            component_class: The component class to inspect
            kwargs: All parameters provided by user
            step_type: Type of step (to determine which method to check)

        Returns:
            Tuple of (init_kwargs, call_kwargs)

        Raises:
            ValueError: If unknown parameters are provided

        """
        # Get __init__ signature
        init_sig = inspect.signature(component_class.__init__)
        init_param_names = set(init_sig.parameters.keys()) - {"self"}

        # Get method signature
        method_param_names: set[str] = set()
        if hasattr(component_class, step_type):
            method = getattr(component_class, step_type)
            method_sig = inspect.signature(method)
            exclude_params = {"self"} | Pipeline._get_positional_params(step_type)
            method_param_names = set(method_sig.parameters.keys()) - exclude_params

        # Split parameters
        init_kwargs = {k: v for k, v in kwargs.items() if k in init_param_names}
        call_kwargs = {k: v for k, v in kwargs.items() if k in method_param_names}
        unknown = {
            k: v
            for k, v in kwargs.items()
            if k not in init_param_names and k not in method_param_names
        }

        # Validate - no unknown parameters
        if unknown:
            raise ValueError(
                f"Unknown parameters for {component_class.__name__}: {list(unknown.keys())}.\n"
                f"  Available __init__ parameters: {sorted(init_param_names)}\n"
                f"  Available {step_type}() parameters: {sorted(method_param_names) or 'none'}",
            )

        return init_kwargs, call_kwargs

    def _call_component(
        self,
        component: Any,
        step_type: str,
        input_data: Any,
        kwargs: dict[str, Any],
    ) -> Any:
        """Call the appropriate method on a component based on step type.

        Args:
            component: The component instance to call
            step_type: Type of step (fetch, process, chunk, refine, export, write)
            input_data: Input data from previous step
            kwargs: Additional keyword arguments

        Returns:
            Output from the component method

        """
        if step_type == "fetch":
            return component.fetch(**kwargs)

        if step_type == "process":
            # Path objects → process(path), strings → parse(text)
            if isinstance(input_data, list):
                return [
                    component.process(item) if isinstance(item, Path) else component.parse(item)
                    for item in input_data
                ]
            return (
                component.process(input_data)
                if isinstance(input_data, Path)
                else component.parse(input_data)
            )

        if step_type == "chunk":
            return (
                [component.chunk_document(doc) for doc in input_data]
                if isinstance(input_data, list)
                else component.chunk_document(input_data)
            )

        if step_type == "refine":
            return (
                [component.refine_document(doc) for doc in input_data]
                if isinstance(input_data, list)
                else component.refine_document(input_data)
            )

        if step_type == "export":
            # Extract chunks and export
            chunks = (
                [c for doc in input_data for c in doc.chunks]
                if isinstance(input_data, list)
                else input_data.chunks
            )
            component.export(chunks, **kwargs)
            return input_data  # Return Documents for chaining

        if step_type == "write":
            # Extract chunks and write to vector DB
            chunks = (
                [c for doc in input_data for c in doc.chunks]
                if isinstance(input_data, list)
                else input_data.chunks
            )
            return component.write(chunks, **kwargs)

        raise ValueError(f"Unknown step type: {step_type}")

    def reset(self) -> "Pipeline":
        """Reset the pipeline to its initial state.

        Returns:
            Pipeline instance for method chaining

        """
        self._steps.clear()
        self._component_instances.clear()
        return self

    def to_config(self, path: Optional[str] = None) -> list[dict[str, Any]]:
        """Export pipeline to config format and optionally save to file.

        Args:
            path: Optional file path to save config as JSON

        Returns:
            List of step configurations

        Examples:
            ```python
            # Get config as list
            config = pipeline.to_config()

            # Save to file
            pipeline.to_config('my_pipeline.json')
            ```

        """
        config = []
        for step in self._steps:
            step_config = {
                "type": step["type"],
                "component": step["component"].alias,
                **step["kwargs"],
            }
            config.append(step_config)

        # Save to file if path provided
        if path:
            import json

            with open(path, "w") as f:
                json.dump(config, f, indent=2)

        return config

    def describe(self) -> str:
        """Get a human-readable description of the pipeline.

        Shows steps in CHOMP execution order (not definition order).

        Returns:
            String description of the pipeline steps

        """
        if not self._steps:
            return "Empty pipeline"

        # Get steps in correct CHOMP order
        ordered_steps = self._reorder_steps()

        descriptions = []
        for step in ordered_steps:
            component = step["component"]
            step_type = step["type"]
            descriptions.append(f"{step_type}({component.alias})")

        return " -> ".join(descriptions)

    def __repr__(self) -> str:
        """Return string representation of the pipeline."""
        return f"Pipeline({self.describe()})"
