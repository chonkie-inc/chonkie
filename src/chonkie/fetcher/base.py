"""BaseFetcher is the base class for all fetchers."""

from abc import ABC, abstractmethod
from typing import Any
import asyncio


class BaseFetcher(ABC):
    """BaseFetcher is the base class for all fetchers."""

    def __init__(self) -> None:
        """Initialize the BaseFetcher."""
        pass

    @abstractmethod
    def fetch(self, *args: Any, **kwargs: Any) -> Any:
        """Fetch data from the source."""
        raise NotImplementedError("Subclasses must implement fetch()")

    async def fetch_async(self, *args: Any, **kwargs: Any) -> Any:
        """Fetch data from the source asynchronously."""
        return await asyncio.to_thread(self.fetch, *args, **kwargs)
