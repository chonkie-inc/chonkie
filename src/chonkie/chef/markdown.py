"""Markdown chef for Chonkie."""

from pathlib import Path
from typing import Union

from chonkie.types import MarkdownDocument, MarkdownTable, MarkdownCode

from .base import BaseChef


class MarkdownChef(BaseChef):
    """Markdown chef for Chonkie."""

    def process(self, path: Union[str, Path]) -> MarkdownDocument:
        """Process the markdown file."""
        # TODO: Implement the markdown chef
        return MarkdownDocument(content="")