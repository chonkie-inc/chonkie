"""Custom types for recursive chunking."""

from dataclasses import dataclass
from typing import Dict, Iterator, List, Literal, Optional, Union

from chonkie.types.base import Chunk
from chonkie.utils import Hubbie


@dataclass
class RecursiveLevel:
    """RecursiveLevels express the chunking rules at a specific level for the recursive chunker.

    Attributes:
        whitespace (bool): Whether to use whitespace as a delimiter.
        delimiters (Optional[Union[str, List[str]]]): Custom delimiters for chunking.
        include_delim (Optional[Literal["prev", "next"]]): Whether to include the delimiter at all, or in the previous chunk, or the next chunk.

    """

    delimiters: Optional[Union[str, List[str]]] = None
    whitespace: bool = False
    include_delim: Optional[Literal["prev", "next"]] = "prev"

    def _validate_fields(self) -> None:
        """Validate all fields have legal values."""
        if self.delimiters is not None and self.whitespace:
            raise NotImplementedError(
                "Cannot use whitespace as a delimiter and also specify custom delimiters."
            )
        if self.delimiters is not None:
            if isinstance(self.delimiters, str) and len(self.delimiters) == 0:
                raise ValueError("Custom delimiters cannot be an empty string.")
            if isinstance(self.delimiters, list):
                if any(
                    not isinstance(delim, str) or len(delim) == 0
                    for delim in self.delimiters
                ):
                    raise ValueError("Custom delimiters cannot be an empty string.")
                if any(delim == " " for delim in self.delimiters):
                    raise ValueError(
                        "Custom delimiters cannot be whitespace only. Set whitespace to True instead."
                    )

    def __post_init__(self) -> None:
        """Validate attributes."""
        self._validate_fields()

    def __repr__(self) -> str:
        """Return a string representation of the RecursiveLevel."""
        return (
            f"RecursiveLevel(delimiters={self.delimiters}, "
            f"whitespace={self.whitespace}, include_delim={self.include_delim})"
        )

    def to_dict(self) -> dict:
        """Return the RecursiveLevel as a dictionary."""
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: dict) -> "RecursiveLevel":
        """Create RecursiveLevel object from a dictionary."""
        return cls(**data)

    @classmethod
    def from_recipe(cls, name: str, lang: Optional[str] = 'en') -> "RecursiveLevel":
        """Create RecursiveLevel object from a recipe.
        
        The recipes are registered in the [Chonkie Recipe Store](https://huggingface.co/datasets/chonkie-ai/recipes). If the recipe is not there, you can create your own recipe and share it with the community!
        
        Args:
            name (str): The name of the recipe.
            lang (Optional[str]): The language of the recipe.

        Returns:
            RecursiveLevel: The RecursiveLevel object.

        Raises:
            ValueError: If the recipe is not found.

        """
        hub = Hubbie()
        recipe = hub.get_recipe(name, lang)
        # If the recipe is not None, we can get the `recursive_rules` key
        if recipe is not None:
            return cls.from_dict({"delimiters": recipe["recipe"]["delimiters"], "include_delim": recipe["recipe"]["include_delim"]})
        else:
            raise ValueError(f"Tried getting recipe `{name}_{lang}.json` but it is not available.")

@dataclass
class RecursiveRules:
    """Expression rules for recursive chunking."""

    levels: Optional[List[RecursiveLevel]] = None

    def __post_init__(self) -> None:
        """Validate attributes."""
        if self.levels is None:
            paragraphs = RecursiveLevel(delimiters=["\n\n", "\r\n", "\n", "\r"])
            sentences = RecursiveLevel(
                delimiters=[". ", "! ", "? "],
            )
            pauses = RecursiveLevel(
                delimiters=[
                    "{",
                    "}",
                    '"',
                    "[",
                    "]",
                    "<",
                    ">",
                    "(",
                    ")",
                    ":",
                    ";",
                    ",",
                    "—",
                    "|",
                    "~",
                    "-",
                    "...",
                    "`",
                    "'",
                ],
            )
            word = RecursiveLevel(whitespace=True)
            token = RecursiveLevel()
            self.levels = [paragraphs, sentences, pauses, word, token]
        elif isinstance(self.levels, list):
            for level in self.levels:
                level._validate_fields()
        else:
            raise ValueError(
                "Levels must be a list of RecursiveLevel objects."
            )

    def __repr__(self) -> str:
        """Return a string representation of the RecursiveRules."""
        return f"RecursiveRules(levels={self.levels})"

    def __len__(self) -> int:
        """Return the number of levels."""
        return len(self.levels) if self.levels is not None else 0
            
    def __getitem__(self, index: int) -> Optional[RecursiveLevel]:
        """Return the RecursiveLevel at the specified index."""
        return self.levels[index] if self.levels is not None else None

    def __iter__(self) -> Optional[Iterator[RecursiveLevel]]:
        """Return an iterator over the RecursiveLevels."""
        return iter(self.levels) if self.levels is not None else None

    @classmethod
    def from_dict(cls, data: dict) -> "RecursiveRules":
        """Create a RecursiveRules object from a dictionary."""
        dict_levels = data.get("levels", None)
        object_levels: Optional[List[RecursiveLevel]] = None
        if dict_levels is not None:
            if isinstance(dict_levels, dict):
                object_levels = [RecursiveLevel.from_dict(dict_levels)]
            elif isinstance(dict_levels, list):
                object_levels = [RecursiveLevel.from_dict(d_level) for d_level in dict_levels]
        return cls(levels=object_levels)

    def to_dict(self) -> Dict:
        """Return the RecursiveRules as a dictionary."""
        result: Dict[str, Optional[List[Dict]]] = dict()
        result["levels"] = [level.to_dict() for level in self.levels] if self.levels is not None else None
        return result

    @classmethod
    def from_recipe(cls, 
                    name: Optional[str] = 'default', 
                    lang: Optional[str] = 'en', 
                    path: Optional[str] = None) -> "RecursiveRules":
        """Create a RecursiveRules object from a recipe.
        
        The recipes are registered in the [Chonkie Recipe Store](https://huggingface.co/datasets/chonkie-ai/recipes).
        If the recipe is not there, you can create your own recipe and share it with the community!
        
        Args:
            name (str): The name of the recipe.
            lang (Optional[str]): The language of the recipe.
            path (Optional[str]): Optionally, provide the path to the recipe.

        Returns:
            RecursiveRules: The RecursiveRules object.

        Raises:
            ValueError: If the recipe is not found.

        """
        # Create a hubbie instance
        hub = Hubbie()
        recipe = hub.get_recipe(name, lang, path) 
        return cls.from_dict(recipe["recipe"]["recursive_rules"])

@dataclass
class RecursiveChunk(Chunk):
    """Class to represent recursive chunks.

    Attributes:
        level (Optional[int]): The level of recursion for the chunk, if any.

    """

    level: Optional[int] = None

    def __repr__(self) -> str:
        """Return a string representation of the RecursiveChunk."""
        return (
            f"RecursiveChunk(text={self.text}, start_index={self.start_index}, "
            f"end_index={self.end_index}, token_count={self.token_count}, "
            f"level={self.level})"
        )

    def to_dict(self) -> Dict:
        """Return the RecursiveChunk as a dictionary."""
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: Dict) -> "RecursiveChunk":
        """Create a RecursiveChunk object from a dictionary."""
        return cls(**data)
