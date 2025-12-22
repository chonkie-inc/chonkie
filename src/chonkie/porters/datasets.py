"""DatasetsPorter to convert Chunks into datasets format for storage."""

from __future__ import annotations

import importlib.util as importutil
from typing import TYPE_CHECKING, Any

from chonkie.logger import get_logger
from chonkie.pipeline import porter
from chonkie.types import Chunk

from .base import BasePorter

logger = get_logger(__name__)

if TYPE_CHECKING:
    from datasets import Dataset


@porter("datasets")
class DatasetsPorter(BasePorter):
    """Porter to convert Chunks into datasets format for storage."""

    def __init__(self) -> None:
        """Initialize the DatasetsPorter and import dependencies."""
        super().__init__()
        if importutil.find_spec("datasets") is None:
            raise ImportError(
                "The 'datasets' library is not installed. "
                "Please install it with 'pip install chonkie[datasets]' or 'pip install datasets'.",
            )

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
        from datasets import Dataset

        logger.debug(f"Exporting {len(chunks)} chunks to HuggingFace Dataset")
        dataset = Dataset.from_list([chunk.to_dict() for chunk in chunks])
        if save_to_disk:
            logger.debug(f"Saving dataset to disk: {path}")
            dataset.save_to_disk(path, **kwargs)
            logger.info(
                f"Successfully exported {len(chunks)} chunks to Dataset and saved to: {path}",
            )
        else:
            logger.info(
                f"Successfully exported {len(chunks)} chunks to Dataset (not saved to disk)",
            )
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
        return self.export(chunks, save_to_disk=save_to_disk, path=path, **kwargs)
