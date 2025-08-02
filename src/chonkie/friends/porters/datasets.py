"""DatasetsPorter to convert Chunks into datasets format for storage."""

from __future__ import annotations

import importlib
import importlib.util as importutil
from typing import TYPE_CHECKING, Any

from chonkie.types import Chunk

from .base import BasePorter

if TYPE_CHECKING:
    from datasets import Dataset


class DatasetsPorter(BasePorter):
    """Porter to convert Chunks into datasets format for storage."""

    def __init__(self) -> None:
        """Initialize the DatasetsPorter and import dependencies."""
        super().__init__()
        self._import_dependencies()

    def _import_dependencies(self) -> None:
        """Import the 'datasets' library and assign it to an instance attribute."""
        if importutil.find_spec("datasets") is None:
            raise ImportError(
                "The 'datasets' library is not installed. "
                "Please install it with 'pip install chonkie[datasets]' or 'pip install datasets'."
            )
        datasets_module = importlib.import_module("datasets")
        self.Dataset = datasets_module.Dataset

    def export(  # type: ignore[override]
        self,
        chunks: list[Chunk],
        save_to_disk: bool = True,
        path: str = "chunks",
        **kwargs: Any,
    ) -> Dataset:
        """Export a list of Chunk objects into a Hugging Face Dataset.

        Args:
            chunks (list[Chunk]): The list of Chunk objects to export.
            save_to_disk (bool, optional): If True, saves the dataset to disk.
                Defaults to True.
            path (str, optional): The path to save the dataset.
                Defaults to "chunks".
            **kwargs: Additional arguments to pass to `save_to_disk`.

        Returns:
            Dataset: The Dataset object.

        """
        dataset = self.Dataset.from_list([chunk.to_dict() for chunk in chunks])
        if save_to_disk:
            dataset.save_to_disk(path, **kwargs)
        return dataset

    def __call__(  # type: ignore[override]
        self,
        chunks: list[Chunk],
        save_to_disk: bool = True,
        path: str = "chunks",
        **kwargs: Any,
    ) -> Dataset:
        """Export a list of Chunk objects into a Hugging Face Dataset.

        This is an alias for the `export` method.

        Args:
            chunks (list[Chunk]): The list of Chunk objects to export.
            save_to_disk (bool, optional): If True, saves the dataset to disk.
                Defaults to True.
            path (str, optional): The path to save the dataset.
                Defaults to "chunks".
            **kwargs: Additional arguments to pass to `save_to_disk`.

        Returns:
            Dataset: The Dataset object.

        """
        return self.export(
            chunks, save_to_disk=save_to_disk, path=path, **kwargs
        )
