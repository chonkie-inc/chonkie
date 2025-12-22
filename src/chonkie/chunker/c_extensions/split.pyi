"""Stub file for split C extension."""

from typing import Literal, Optional, Sequence, Union

def split_text(
    text: str,
    delim: Optional[Union[str, list[str]]] = None,
    include_delim: Optional[Union[bool, Literal["prev", "next"]]] = False,
    min_characters_per_segment: int = 1,
    whitespace_mode: bool = False,
    character_fallback: bool = False,
) -> Sequence[str]:
    """Split text using given delimiters.

    Args:
        text: Text to split
        delim: Delimiter string or list of delimiter strings
        include_delim: Whether to include delimiters in output
        min_characters_per_segment: Minimum characters per segment
        whitespace_mode: Whether to use whitespace splitting
        character_fallback: Whether to fall back to character splitting

    Returns:
        Sequence of text splits

    """
    ...
