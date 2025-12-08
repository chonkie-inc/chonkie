"""Component for pipeline."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ComponentType(Enum):
    """Types of pipeline components.

    These represent the stages in the CHOMP pipeline:
    - FETCHER: Retrieves raw data from sources
    - CHEF: Preprocesses and transforms data
    - CHUNKER: Splits text into chunks
    - REFINERY: Post-processes chunks (e.g., add embeddings, merge)
    - PORTER: Exports chunks to storage formats
    - HANDSHAKE: Ingests chunks into vector databases
    """

    FETCHER = "fetcher"
    CHEF = "chef"
    CHUNKER = "chunker"
    REFINERY = "refinery"
    PORTER = "porter"
    HANDSHAKE = "handshake"


@dataclass
class Component:
    """Minimal metadata about a pipeline component.

    This class stores the essential information needed to identify
    and instantiate components in a Chonkie pipeline.

    Attributes:
        name: Full class name (e.g., "RecursiveChunker")
        alias: Short alias for string-based configs (e.g., "recursive")
        component_class: The actual class to instantiate
        component_type: Which CHOMP stage this component belongs to

    """

    name: str
    alias: str
    component_class: type[Any]
    component_type: ComponentType

    def __post_init__(self) -> None:
        """Validate component after creation."""
        if not self.name:
            raise ValueError("Component name cannot be empty")
        if not self.alias:
            raise ValueError("Component alias cannot be empty")
        if not self.component_class:
            raise ValueError("Component class cannot be None")
