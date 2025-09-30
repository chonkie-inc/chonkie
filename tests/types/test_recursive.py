"""Test recursive types."""

import pytest

from chonkie import RecursiveLevel, RecursiveRules


def test_recursive_level_init() -> None:
    """Test RecursiveLevel initialization."""
    level = RecursiveLevel(delimiters=["\n", "."])
    assert level.delimiters == ["\n", "."]
    assert not level.whitespace
    assert level.include_delim == "prev"


def test_recursive_level_raises_error() -> None:
    """Test RecursiveLevel validation."""
    with pytest.raises(NotImplementedError):
        RecursiveLevel(whitespace=True, delimiters=["."])

    with pytest.raises(ValueError):
        RecursiveLevel(delimiters=[1, 2])

    with pytest.raises(ValueError):
        RecursiveLevel(delimiters=[""])

    with pytest.raises(ValueError):
        RecursiveLevel(delimiters=[" "])


def test_recursive_level_serialization() -> None:
    """Test RecursiveLevel serialization and deserialization."""
    level = RecursiveLevel(delimiters=["\n", "."])
    level_dict = level.to_dict()
    reconstructed = RecursiveLevel.from_dict(level_dict)
    assert reconstructed.delimiters == ["\n", "."]
    assert not reconstructed.whitespace
    assert reconstructed.include_delim == "prev"


# RecursiveRules Tests
def test_recursive_rules_default_init() -> None:
    """Test RecursiveRules default initialization."""
    rules = RecursiveRules()
    assert len(rules.levels) == 5
    assert all(isinstance(level, RecursiveLevel) for level in rules.levels)


def test_recursive_rules_custom_init() -> None:
    """Test RecursiveRules custom initialization."""
    levels = [
        RecursiveLevel(delimiters=["\n"]),
        RecursiveLevel(delimiters=["."]),
    ]
    rules = RecursiveRules(levels=levels)
    assert len(rules.levels) == 2
    assert rules.levels == levels


def test_recursive_rules_serialization() -> None:
    """Test RecursiveRules serialization and deserialization."""
    levels = [
        RecursiveLevel(delimiters=["\n"]),
        RecursiveLevel(delimiters=["."]),
    ]
    rules = RecursiveRules(levels=levels)
    rules_dict = rules.to_dict()
    reconstructed = RecursiveRules.from_dict(rules_dict)
    assert len(reconstructed.levels) == 2
    assert all(isinstance(level, RecursiveLevel) for level in reconstructed.levels)



def test_recursive_level_from_recipe() -> None:
    """Test RecursiveLevel from recipe."""
    level = RecursiveLevel.from_recipe("default", lang="en")
    assert isinstance(level, RecursiveLevel)
    assert level.delimiters == ['. ', '! ', '? ', '\n']
    assert not level.whitespace
    assert level.include_delim == "prev"

def test_recursive_rules_from_recipe() -> None:
    """Test RecursiveRules from recipe."""
    rules = RecursiveRules.from_recipe("default", lang="en")
    assert isinstance(rules, RecursiveRules)
    assert len(rules.levels) == 5
    assert all(isinstance(level, RecursiveLevel) for level in rules.levels)

def test_recursive_rules_from_recipe_nonexistent() -> None:
    """Test RecursiveRules from recipe with nonexistent recipe."""
    with pytest.raises(ValueError):
        RecursiveRules.from_recipe("invalid", lang="en")

    with pytest.raises(ValueError):
        RecursiveRules.from_recipe("default", lang="invalid")