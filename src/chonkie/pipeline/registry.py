"""Component registry for pipeline components."""

from typing import Any, Callable, Optional, Type, TypeVar

from .component import Component, ComponentType

ComponentT = TypeVar("ComponentT", bound=Type[Any])


class _ComponentRegistry:
    """Internal component registry class."""

    def __init__(self) -> None:
        """Initialize the component registry."""
        self._components: dict[str, Component] = {}
        # Scoped aliases: (component_type, alias) -> name mapping
        self._aliases: dict[tuple[ComponentType, str], str] = {}
        self._component_types: dict[ComponentType, list[str]] = {ct: [] for ct in ComponentType}

    def register(
        self,
        name: str,
        alias: str,
        component_class: ComponentT,
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
                    f"Component name '{name}' already registered with different class",
                )

        # Check for alias conflicts within the same component type
        alias_key = (component_type, alias)
        if alias_key in self._aliases:
            existing_name = self._aliases[alias_key]
            if existing_name != name:
                raise ValueError(
                    f"Alias '{alias}' already used by {component_type.value} component '{existing_name}'",
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
        self._aliases[alias_key] = name
        self._component_types[component_type].append(name)

    def get_component(
        self,
        name_or_alias: str,
        component_type: Optional[ComponentType] = None,
    ) -> Component:
        """Get component info by name or alias.

        Args:
            name_or_alias: Component name or alias
            component_type: Optional component type to scope alias lookup

        Returns:
            Component for the requested component

        Raises:
            ValueError: If component is not found

        """
        # If component_type provided, try scoped alias lookup first
        if component_type:
            alias_key = (component_type, name_or_alias)
            if alias_key in self._aliases:
                name = self._aliases[alias_key]
                return self._components[name]

        # Try unscoped: check if it's a direct name match
        if name_or_alias in self._components:
            comp = self._components[name_or_alias]
            # If type specified, verify it matches
            if component_type and comp.component_type != component_type:
                raise ValueError(
                    f"Component '{name_or_alias}' is a {comp.component_type.value}, "
                    f"not a {component_type.value}",
                )
            return comp

        # Try to find by alias across all types (ambiguous lookup)
        matches = []
        for (ctype, alias), name in self._aliases.items():
            if alias == name_or_alias:
                matches.append((ctype, name))

        if len(matches) == 1:
            return self._components[matches[0][1]]
        elif len(matches) > 1:
            types = [m[0].value for m in matches]
            raise ValueError(
                f"Ambiguous alias '{name_or_alias}' found in multiple types: {types}. "
                f"Specify component_type to disambiguate.",
            )

        # Not found
        available = [f"{ct.value}:{alias}" for (ct, alias) in self._aliases.keys()]
        raise ValueError(
            f"Unknown component: '{name_or_alias}'. Available: {sorted(available)[:10]}...",
        )

    def list_components(self, component_type: Optional[ComponentType] = None) -> list[Component]:
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

    def get_aliases(self, component_type: Optional[ComponentType] = None) -> list[str]:
        """Get all available aliases, optionally filtered by type.

        Args:
            component_type: Optional filter by component type

        Returns:
            List of component aliases

        """
        if component_type:
            return [alias for (ctype, alias) in self._aliases.keys() if ctype == component_type]
        return [alias for (_, alias) in self._aliases.keys()]

    def get_fetcher(self, alias: str) -> Component:
        """Get a fetcher component by alias.

        Args:
            alias: Fetcher alias

        Returns:
            Component info for the fetcher

        Raises:
            ValueError: If fetcher not found

        """
        return self.get_component(alias, ComponentType.FETCHER)

    def get_chef(self, alias: str) -> Component:
        """Get a chef component by alias.

        Args:
            alias: Chef alias

        Returns:
            Component info for the chef

        Raises:
            ValueError: If chef not found

        """
        return self.get_component(alias, ComponentType.CHEF)

    def get_chunker(self, alias: str) -> Component:
        """Get a chunker component by alias.

        Args:
            alias: Chunker alias

        Returns:
            Component info for the chunker

        Raises:
            ValueError: If chunker not found

        """
        return self.get_component(alias, ComponentType.CHUNKER)

    def get_refinery(self, alias: str) -> Component:
        """Get a refinery component by alias.

        Args:
            alias: Refinery alias

        Returns:
            Component info for the refinery

        Raises:
            ValueError: If refinery not found

        """
        return self.get_component(alias, ComponentType.REFINERY)

    def get_porter(self, alias: str) -> Component:
        """Get a porter component by alias.

        Args:
            alias: Porter alias

        Returns:
            Component info for the porter

        Raises:
            ValueError: If porter not found

        """
        return self.get_component(alias, ComponentType.PORTER)

    def get_handshake(self, alias: str) -> Component:
        """Get a handshake component by alias.

        Args:
            alias: Handshake alias

        Returns:
            Component info for the handshake

        Raises:
            ValueError: If handshake not found

        """
        return self.get_component(alias, ComponentType.HANDSHAKE)

    def is_registered(self, name_or_alias: str) -> bool:
        """Check if a component is registered.

        Args:
            name_or_alias: Component name or alias

        Returns:
            True if component is registered, False otherwise

        """
        # Check if exists in components dict or in any alias tuple
        if name_or_alias in self._components:
            return True
        # Check aliases - need to check if name_or_alias matches any alias value
        for _, alias in self._aliases.keys():
            if alias == name_or_alias:
                return True
        return False

    def unregister(
        self,
        name_or_alias: str,
        component_type: Optional[ComponentType] = None,
    ) -> None:
        """Unregister a component (mainly for testing).

        Args:
            name_or_alias: Component name or alias to unregister
            component_type: Optional component type for scoped alias lookup

        """
        # Try to find the component
        comp = None
        try:
            comp = self.get_component(name_or_alias, component_type)
        except ValueError:
            return  # Component not registered

        name = comp.name
        alias = comp.alias
        ctype = comp.component_type

        # Remove from all tracking structures
        del self._components[name]
        alias_key = (ctype, alias)
        if alias_key in self._aliases:
            del self._aliases[alias_key]
        self._component_types[ctype].remove(name)

    def clear(self) -> None:
        """Clear all registered components (mainly for testing)."""
        self._components.clear()
        self._aliases.clear()
        for component_list in self._component_types.values():
            component_list.clear()


def pipeline_component(
    alias: str,
    component_type: ComponentType,
) -> Callable[[ComponentT], ComponentT]:
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

    def decorator(cls: ComponentT) -> ComponentT:
        # Validate that the class has ALL required methods
        required_methods = {
            ComponentType.FETCHER: ["fetch"],
            ComponentType.CHEF: ["process", "parse"],  # Both required!
            ComponentType.CHUNKER: ["chunk", "chunk_document"],
            ComponentType.REFINERY: ["refine", "refine_document"],
            ComponentType.PORTER: ["export"],
            ComponentType.HANDSHAKE: ["write"],
        }

        required = required_methods.get(component_type, [])
        missing_methods = [m for m in required if not hasattr(cls, m)]

        if missing_methods:
            raise ValueError(
                f"{cls.__name__} must implement {missing_methods} method(s) "
                f"to be registered as {component_type.value}. "
                f"Required methods: {required}",
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
def fetcher(alias: str) -> Callable[[ComponentT], ComponentT]:
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


def chef(alias: str) -> Callable[[ComponentT], ComponentT]:
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


def chunker(alias: str) -> Callable[[ComponentT], ComponentT]:
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


def refinery(alias: str) -> Callable[[ComponentT], ComponentT]:
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


def porter(alias: str) -> Callable[[ComponentT], ComponentT]:
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


def handshake(alias: str) -> Callable[[ComponentT], ComponentT]:
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
