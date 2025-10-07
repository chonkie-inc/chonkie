"""Component registry for pipeline components."""

from typing import Callable, Dict, List, Optional, Type

from .component import Component, ComponentType


class _ComponentRegistry:
    """Internal component registry class."""

    def __init__(self) -> None:
        """Initialize the component registry."""
        self._components: Dict[str, Component] = {}
        self._aliases: Dict[str, str] = {}  # alias -> name mapping
        self._component_types: Dict[ComponentType, List[str]] = {
            ct: [] for ct in ComponentType
        }
        self._initialized = False

    def register(
        self,
        name: str,
        alias: str,
        component_class: Type,
        component_type: ComponentType,
    ) -> None:
        """Register a component in the registry.

        Args:
            name: Full name of the component (usually class name)
            alias: Short alias for the component (used in string pipelines)
            component_class: The actual component class
            component_type: Type of component (fetcher, chunker, etc.)

        Raises:
            ValueError: If component name/alias conflicts exist

        """
        # Check for name conflicts
        if name in self._components:
            existing = self._components[name]
            if existing.component_class is component_class:
                # Same class, same registration - this is fine (idempotent)
                return
            else:
                raise ValueError(
                    f"Component name '{name}' already registered with different class"
                )

        # Check for alias conflicts
        if alias in self._aliases:
            existing_name = self._aliases[alias]
            if existing_name != name:
                raise ValueError(
                    f"Alias '{alias}' already used by component '{existing_name}'"
                )

        # Create component info
        info = Component(
            name=name,
            alias=alias,
            component_class=component_class,
            component_type=component_type,
        )

        # Register the component
        self._components[name] = info
        self._aliases[alias] = name
        self._component_types[component_type].append(name)

    def get_component(self, name_or_alias: str) -> Component:
        """Get component info by name or alias.

        Args:
            name_or_alias: Component name or alias

        Returns:
            Component for the requested component

        Raises:
            ValueError: If component is not found

        """
        # Try alias first, then name
        if name_or_alias in self._aliases:
            name = self._aliases[name_or_alias]
        else:
            name = name_or_alias

        if name not in self._components:
            available_aliases = list(self._aliases.keys())
            raise ValueError(
                f"Unknown component: '{name_or_alias}'. "
                f"Available components: {available_aliases}"
            )

        return self._components[name]

    def list_components(
        self, component_type: Optional[ComponentType] = None
    ) -> List[Component]:
        """List all registered components, optionally filtered by type.

        Args:
            component_type: Optional filter by component type

        Returns:
            List of Component objects

        """
        if component_type:
            names = self._component_types[component_type]
            return [self._components[name] for name in names]
        return list(self._components.values())

    def get_aliases(self, component_type: Optional[ComponentType] = None) -> List[str]:
        """Get all available aliases, optionally filtered by type.

        Args:
            component_type: Optional filter by component type

        Returns:
            List of component aliases

        """
        if component_type:
            names = self._component_types[component_type]
            return [self._components[name].alias for name in names]
        return list(self._aliases.keys())

    def get_fetcher(self, alias: str) -> Component:
        """Get a fetcher component by alias.

        Args:
            alias: Fetcher alias

        Returns:
            Component info for the fetcher

        Raises:
            ValueError: If fetcher not found

        """
        component = self.get_component(alias)
        if component.component_type != ComponentType.FETCHER:
            raise ValueError(f"'{alias}' is not a fetcher component")
        return component

    def get_chef(self, alias: str) -> Component:
        """Get a chef component by alias.

        Args:
            alias: Chef alias

        Returns:
            Component info for the chef

        Raises:
            ValueError: If chef not found

        """
        component = self.get_component(alias)
        if component.component_type != ComponentType.CHEF:
            raise ValueError(f"'{alias}' is not a chef component")
        return component

    def get_chunker(self, alias: str) -> Component:
        """Get a chunker component by alias.

        Args:
            alias: Chunker alias

        Returns:
            Component info for the chunker

        Raises:
            ValueError: If chunker not found

        """
        component = self.get_component(alias)
        if component.component_type != ComponentType.CHUNKER:
            raise ValueError(f"'{alias}' is not a chunker component")
        return component

    def get_refinery(self, alias: str) -> Component:
        """Get a refinery component by alias.

        Args:
            alias: Refinery alias

        Returns:
            Component info for the refinery

        Raises:
            ValueError: If refinery not found

        """
        component = self.get_component(alias)
        if component.component_type != ComponentType.REFINERY:
            raise ValueError(f"'{alias}' is not a refinery component")
        return component

    def get_porter(self, alias: str) -> Component:
        """Get a porter component by alias.

        Args:
            alias: Porter alias

        Returns:
            Component info for the porter

        Raises:
            ValueError: If porter not found

        """
        component = self.get_component(alias)
        if component.component_type != ComponentType.PORTER:
            raise ValueError(f"'{alias}' is not a porter component")
        return component

    def get_handshake(self, alias: str) -> Component:
        """Get a handshake component by alias.

        Args:
            alias: Handshake alias

        Returns:
            Component info for the handshake

        Raises:
            ValueError: If handshake not found

        """
        component = self.get_component(alias)
        if component.component_type != ComponentType.HANDSHAKE:
            raise ValueError(f"'{alias}' is not a handshake component")
        return component

    def is_registered(self, name_or_alias: str) -> bool:
        """Check if a component is registered.

        Args:
            name_or_alias: Component name or alias

        Returns:
            True if component is registered, False otherwise

        """
        return name_or_alias in self._aliases or name_or_alias in self._components

    def unregister(self, name_or_alias: str) -> None:
        """Unregister a component (mainly for testing).

        Args:
            name_or_alias: Component name or alias to unregister

        """
        if name_or_alias in self._aliases:
            name = self._aliases[name_or_alias]
            alias = name_or_alias
        elif name_or_alias in self._components:
            name = name_or_alias
            alias = self._components[name].alias
        else:
            return  # Component not registered

        # Remove from all tracking structures
        component_info = self._components[name]
        component_type = component_info.component_type

        del self._components[name]
        del self._aliases[alias]
        self._component_types[component_type].remove(name)

    def clear(self) -> None:
        """Clear all registered components (mainly for testing)."""
        self._components.clear()
        self._aliases.clear()
        for component_list in self._component_types.values():
            component_list.clear()


def pipeline_component(
    alias: str,
    component_type: ComponentType,
) -> Callable[[Type], Type]:
    """Register a class as a pipeline component.

    Args:
        alias: Short name for the component (used in string pipelines)
        component_type: Type of component (fetcher, chunker, etc.)

    Returns:
        Decorator function

    Example:
        @pipeline_component("recursive", ComponentType.CHUNKER)
        class RecursiveChunker(BaseChunker):
            pass

    Raises:
        ValueError: If the class doesn't implement required methods

    """

    def decorator(cls: Type) -> Type:
        # Validate that the class has required methods
        required_methods = {
            ComponentType.FETCHER: ["fetch"],
            ComponentType.CHEF: ["process"],
            ComponentType.CHUNKER: ["chunk"],
            ComponentType.REFINERY: ["refine"],
            ComponentType.PORTER: ["export"],
            ComponentType.HANDSHAKE: ["write"],
        }

        required = required_methods.get(component_type, [])
        for method_name in required:
            if not hasattr(cls, method_name):
                raise ValueError(
                    f"{cls.__name__} must implement {method_name}() method "
                    f"to be registered as {component_type.value}"
                )

        # Register the component
        ComponentRegistry.register(
            name=cls.__name__,
            alias=alias,
            component_class=cls,
            component_type=component_type,
        )

        # Add metadata to the class for introspection
        cls._pipeline_component_info = {
            "alias": alias,
            "component_type": component_type,
        }

        return cls

    return decorator


# Specialized decorators for each component type
def fetcher(alias: str) -> Callable[[Type], Type]:
    """Register a fetcher component.

    Args:
        alias: Short name for the component

    Returns:
        Decorator function

    Example:
        @fetcher("file")
        class FileFetcher(BaseFetcher):
            pass

    """
    return pipeline_component(alias=alias, component_type=ComponentType.FETCHER)


def chef(alias: str) -> Callable[[Type], Type]:
    """Register a chef component.

    Args:
        alias: Short name for the component

    Returns:
        Decorator function

    Example:
        @chef("markdown")
        class MarkdownChef(BaseChef):
            pass

    """
    return pipeline_component(alias=alias, component_type=ComponentType.CHEF)


def chunker(alias: str) -> Callable[[Type], Type]:
    """Register a chunker component.

    Args:
        alias: Short name for the component

    Returns:
        Decorator function

    Example:
        @chunker("recursive")
        class RecursiveChunker(BaseChunker):
            pass

    """
    return pipeline_component(alias=alias, component_type=ComponentType.CHUNKER)


def refinery(alias: str) -> Callable[[Type], Type]:
    """Register a refinery component.

    Args:
        alias: Short name for the component

    Returns:
        Decorator function

    Example:
        @refinery("embeddings")
        class EmbeddingsRefinery(BaseRefinery):
            pass

    """
    return pipeline_component(alias=alias, component_type=ComponentType.REFINERY)


def porter(alias: str) -> Callable[[Type], Type]:
    """Register a porter component.

    Args:
        alias: Short name for the component

    Returns:
        Decorator function

    Example:
        @porter("json")
        class JSONPorter(BasePorter):
            pass

    """
    return pipeline_component(alias=alias, component_type=ComponentType.PORTER)


def handshake(alias: str) -> Callable[[Type], Type]:
    """Register a handshake component.

    Args:
        alias: Short name for the component

    Returns:
        Decorator function

    Example:
        @handshake("chroma")
        class ChromaHandshake(BaseHandshake):
            pass

    """
    return pipeline_component(alias=alias, component_type=ComponentType.HANDSHAKE)


# Global registry instance
ComponentRegistry = _ComponentRegistry()