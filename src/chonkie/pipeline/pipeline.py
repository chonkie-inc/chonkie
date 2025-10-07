"""Core Pipeline class for chonkie."""

import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from chonkie.types import Document
from chonkie.utils import Hubbie

from .registry import ComponentRegistry


class Pipeline:
    """A fluent API for building and executing chonkie pipelines.

    The Pipeline class provides a clean, chainable interface for processing
    documents through the CHOMP pipeline: CHef -> CHunker -> Refinery -> Porter/Handshake.

    Example:
        ```python
        from chonkie.pipeline import Pipeline

        # Simple pipeline - returns Document with chunks
        doc = (Pipeline()
            .fetch_from("file", path="document.txt")
            .process_with("text")
            .chunk_with("recursive", chunk_size=512)
            .run())

        # Access chunks via doc.chunks
        for chunk in doc.chunks:
            print(chunk.text)

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
        self._steps: List[Dict[str, Any]] = []
        self._data: Any = None
        self._component_instances: Dict[tuple[str, tuple[tuple[str, Any], ...]], Any] = {}  # Cache for component instances

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
    def from_config(cls, config: Union[str, List[Union[tuple[Any, ...], Dict[str, Any]]]]) -> "Pipeline":
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
            with open(config, 'r') as f:
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
                    step_type = step.get('type')
                    component_name = step.get('component')
                    if not step_type or not component_name:
                        raise ValueError("Dict must have 'type' and 'component' keys")
                    kwargs = {k: v for k, v in step.items() if k not in ['type', 'component']}
                else:
                    raise ValueError(f"Step must be tuple or dict, got {type(step)}")

                # Map to appropriate method
                if step_type == 'fetch':
                    pipeline.fetch_from(component_name, **kwargs)
                elif step_type == 'process':
                    pipeline.process_with(component_name, **kwargs)
                elif step_type == 'chunk':
                    pipeline.chunk_with(component_name, **kwargs)
                elif step_type == 'refine':
                    pipeline.refine_with(component_name, **kwargs)
                elif step_type == 'export':
                    pipeline.export_with(component_name, **kwargs)
                elif step_type == 'write':
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

        Example:
            ```python
            pipeline.fetch_from("file", path="document.txt")
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

    def run(self, texts: Optional[Union[str, List[str]]] = None) -> Union[Document, List[Document]]:
        """Run the pipeline and return the final result.

        The pipeline automatically reorders steps according to the CHOMP flow:
        Fetcher -> Chef -> Chunker -> Refinery(ies) -> Porter/Handshake

        This allows components to be defined in any order during pipeline building,
        but ensures correct execution order.

        Args:
            texts: Optional text input. Can be a single string or list of strings.
                   When provided, the fetcher step becomes optional.

        Returns:
            Document or List[Document] with processed chunks

        Raises:
            ValueError: If pipeline has no steps or invalid step configuration
            RuntimeError: If pipeline execution fails

        Examples:
            ```python
            # Traditional fetcher-based pipeline - returns Document
            pipeline = (Pipeline()
                .fetch_from("file", path="doc.txt")
                .process_with("text")
                .chunk_with("recursive", chunk_size=512))
            doc = pipeline.run()
            print(f"Chunked into {len(doc.chunks)} chunks")

            # Direct text input (fetcher optional)
            pipeline = (Pipeline()
                .process_with("text")
                .chunk_with("recursive", chunk_size=512))
            doc = pipeline.run(texts="Hello world")

            # Access chunks via doc.chunks
            for chunk in doc.chunks:
                print(chunk.text)

            # Multiple texts - returns List[Document]
            docs = pipeline.run(texts=["Text 1", "Text 2", "Text 3"])
            all_chunks = [chunk for doc in docs for chunk in doc.chunks]
            ```

        """
        if not self._steps:
            raise ValueError("Pipeline has no steps to execute")

        # Validate the pipeline configuration BEFORE reordering
        # (reordering may drop duplicates, so validate on original list)
        self._validate_pipeline(self._steps, has_text_input=(texts is not None))

        # Reorder steps according to CHOMP pipeline flow
        ordered_steps = self._reorder_steps()

        # Initialize data based on input
        data: Any
        if texts is not None:
            # Check if pipeline has a chef step
            has_chef = any(step["type"] == "process" for step in ordered_steps)

            if has_chef:
                # Let chef convert text to Document
                data = texts
            else:
                # No chef - wrap text in Document ourselves
                if isinstance(texts, str):
                    data = Document(content=texts)
                elif isinstance(texts, list):
                    data = [Document(content=text) for text in texts]
                else:
                    data = texts
        else:
            data = None

        # Execute pipeline steps
        for i, step in enumerate(ordered_steps):
            try:
                # Skip fetcher step if we have direct text input
                if texts is not None and step["type"] == "fetch":
                    continue

                data = self._execute_step(step, data)
            except Exception as e:
                step_info = f"step {i + 1} ({step['type']})"
                raise RuntimeError(f"Pipeline failed at {step_info}: {e}") from e

        return data  # type: ignore[no-any-return]

    def _reorder_steps(self) -> List[Dict[str, Any]]:
        """Reorder pipeline steps according to CHOMP flow.
        
        Returns:
            List of steps in correct execution order

        """
        # Define the correct order of component types
        type_order = {
            "fetch": 0,
            "process": 1,
            "chunk": 2,
            "refine": 3,
            "export": 4,
            "write": 5
        }
        
        # Group steps by type
        steps_by_type: Dict[str, List[Dict[str, Any]]] = {}
        for step in self._steps:
            step_type = step["type"]
            if step_type not in steps_by_type:
                steps_by_type[step_type] = []
            steps_by_type[step_type].append(step)
        
        # Build ordered list
        ordered_steps = []
        
        # Add steps in the correct order
        for step_type in sorted(type_order.keys(), key=lambda x: type_order[x]):
            if step_type in steps_by_type:
                if step_type in ["refine", "fetch", "chunk", "export", "write"]:
                    # Multiple allowed: maintain the order they were added
                    ordered_steps.extend(steps_by_type[step_type])
                else:
                    # process (chef) - should only have one (validated earlier)
                    # If somehow multiple exist, use the last one defined
                    ordered_steps.append(steps_by_type[step_type][-1])
        
        return ordered_steps

    def _validate_pipeline(self, ordered_steps: List[Dict[str, Any]], has_text_input: bool = False) -> None:
        """Validate that the pipeline configuration is valid.
        
        Args:
            ordered_steps: Steps in execution order
            has_text_input: Whether direct text input is provided to execute()
            
        Raises:
            ValueError: If pipeline configuration is invalid

        """
        if not ordered_steps:
            raise ValueError("Pipeline has no steps to execute")

        step_types = [step["type"] for step in ordered_steps]

        # Check for duplicate process steps (only one chef allowed)
        process_count = step_types.count("process")
        if process_count > 1:
            raise ValueError(
                f"Multiple process steps found ({process_count}). "
                f"Only one chef is allowed per pipeline."
            )

        # Multiple allowed for: fetch, chunk, refine, export, write

        # Check that we have at least a chunker (minimum viable pipeline)
        if "chunk" not in step_types:
            raise ValueError("Pipeline must include a chunker component (use chunk_with())")

        # Check fetcher requirements based on input method
        if not has_text_input and "fetch" not in step_types:
            raise ValueError(
                "Pipeline must include a fetcher component (use fetch_from()) "
                "or provide text input to execute(texts=...)"
            )

    def _execute_step(self, step: Dict[str, Any], input_data: Any) -> Any:
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

        # Auto-detect parameter separation
        try:
            init_kwargs, call_kwargs = Pipeline._split_parameters(
                component_info.component_class, kwargs, step_type
            )
        except Exception as e:
            raise ValueError(
                f"Parameter analysis failed for {component_info.component_class.__name__}: {e}"
            ) from e

        # Create component instance with init parameters only
        component_key = (component_info.name, tuple(sorted(init_kwargs.items())))
        if component_key not in self._component_instances:
            try:
                self._component_instances[component_key] = component_info.component_class(**init_kwargs)
            except Exception as e:
                raise ValueError(
                    f"Failed to create {component_info.component_class.__name__} with parameters {init_kwargs}: {e}"
                ) from e

        component_instance = self._component_instances[component_key]

        # Execute the component using its appropriate method
        try:
            return self._call_component(component_instance, step_type, input_data, call_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to execute {step_type} step with {component_info.component_class.__name__}: {e}"
            ) from e

    @staticmethod
    def _split_parameters(component_class: Type[Any], kwargs: Dict[str, Any], step_type: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
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
        try:
            # Get __init__ signature to determine init params
            init_sig = inspect.signature(component_class.__init__)
            init_param_names = set(init_sig.parameters.keys()) - {"self"}

            # Get method signature (step_type matches method name directly)
            method_param_names: set[str] = set()
            if hasattr(component_class, step_type):
                method = getattr(component_class, step_type)
                method_sig = inspect.signature(method)
                method_param_names = set(method_sig.parameters.keys()) - {"self", "chunks", "chunk", "text", "document", "path"}

            # Split parameters
            init_kwargs = {k: v for k, v in kwargs.items() if k in init_param_names}
            call_kwargs = {k: v for k, v in kwargs.items() if k in method_param_names}
            unknown = {k: v for k, v in kwargs.items() if k not in init_param_names and k not in method_param_names}

            # Raise error for unknown parameters
            if unknown:
                error_msg = (
                    f"Unknown parameters for {component_class.__name__}: {list(unknown.keys())}.\n"
                    f"  Available __init__ parameters: {sorted(init_param_names)}\n"
                    f"  Available {step_type}() parameters: {sorted(method_param_names) if method_param_names else 'none (only positional args)'}"
                )
                raise ValueError(error_msg)

            return init_kwargs, call_kwargs
        except ValueError:
            # Re-raise ValueError with parameter info
            raise
        except Exception:
            # Fallback: assume all params go to __init__
            return kwargs, {}

    def _call_component(self, component: Any, step_type: str, input_data: Any, kwargs: Dict[str, Any]) -> Any:
        """Call the appropriate method on a component based on step type.

        Pipeline always works with Documents for consistency.

        Args:
            component: The component instance to call
            step_type: Type of step (fetch, process, chunk, refine, export, write)
            input_data: Input data from previous step
            kwargs: Additional keyword arguments

        Returns:
            Output from the component method

        """
        if step_type == "fetch":
            # Fetcher.fetch(**kwargs) → paths/files
            return component.fetch(**kwargs)

        elif step_type == "process":
            # Chef.process(path) → Document (from file)
            # OR Chef.parse(text) → Document (from raw text)
            # Note: process/parse don't accept **kwargs (all config in __init__)
            if isinstance(input_data, list):
                # Check if list contains paths or raw text/Documents
                if input_data and isinstance(input_data[0], str):
                    # Could be paths or text - try to detect
                    if Path(input_data[0]).exists():
                        # File paths
                        return [component.process(path) for path in input_data]
                    else:
                        # Raw text strings
                        return [component.parse(text) for text in input_data]
                else:
                    # Assume paths
                    return [component.process(path) for path in input_data]
            else:
                # Single input - check if it's a path or text
                if isinstance(input_data, str) and not Path(input_data).exists():
                    # Raw text (not a file)
                    return component.parse(input_data)
                else:
                    # File path
                    return component.process(input_data)

        elif step_type == "chunk":
            # Chunker.chunk_document(document) → Document (with chunks)
            # Note: chunk_document doesn't accept **kwargs (all config in __init__)
            if isinstance(input_data, list):
                # List of Documents
                return [component.chunk_document(doc) for doc in input_data]
            else:
                # Single Document
                return component.chunk_document(input_data)

        elif step_type == "refine":
            # Refinery.refine_document(document) → Document (with refined chunks)
            # Note: refine_document doesn't accept **kwargs (all config in __init__)
            if isinstance(input_data, list):
                # List of Documents
                return [component.refine_document(doc) for doc in input_data]
            else:
                # Single Document
                return component.refine_document(input_data)

        elif step_type == "export":
            # Porter.export(chunks, **kwargs) → None (return input for chaining)
            # Extract chunks from Document(s)
            if isinstance(input_data, list):
                all_chunks = []
                for doc in input_data:
                    all_chunks.extend(doc.chunks)
                component.export(all_chunks, **kwargs)
            else:
                component.export(input_data.chunks, **kwargs)
            return input_data  # Return Documents for potential further processing

        elif step_type == "write":
            # Handshake.write(chunks) → result
            # Extract chunks from Document(s)
            if isinstance(input_data, list):
                all_chunks = []
                for doc in input_data:
                    all_chunks.extend(doc.chunks)
                return component.write(all_chunks, **kwargs)
            else:
                return component.write(input_data.chunks, **kwargs)

        else:
            raise ValueError(f"Unknown step type: {step_type}")

    def reset(self) -> "Pipeline":
        """Reset the pipeline to its initial state.

        Returns:
            Pipeline instance for method chaining

        """
        self._steps.clear()
        self._data = None
        self._component_instances.clear()
        return self

    def copy(self) -> "Pipeline":
        """Create a copy of the current pipeline.

        Returns:
            New Pipeline instance with the same steps

        """
        new_pipeline = Pipeline()
        new_pipeline._steps = self._steps.copy()
        return new_pipeline

    def to_config(self, path: Optional[str] = None) -> List[Dict[str, Any]]:
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
                'type': step['type'],
                'component': step['component'].alias,
                **step['kwargs']
            }
            config.append(step_config)

        # Save to file if path provided
        if path:
            import json
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)

        return config

    def describe(self) -> str:
        """Get a human-readable description of the pipeline.

        Returns:
            String description of the pipeline steps

        """
        if not self._steps:
            return "Empty pipeline"

        descriptions = []
        for step in self._steps:
            component = step["component"]
            step_type = step["type"]
            descriptions.append(f"{step_type}({component.alias})")

        return " -> ".join(descriptions)

    def __repr__(self) -> str:
        """Return string representation of the pipeline."""
        return f"Pipeline({self.describe()})"