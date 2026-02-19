"""Tests for handshake utility functions."""

from chonkie.handshakes.utils import ADJECTIVES, NOUNS, VERBS, generate_random_collection_name


def test_generate_name_returns_string():
    """generate_random_collection_name returns a string."""
    name = generate_random_collection_name()
    assert isinstance(name, str)


def test_generate_name_default_separator():
    """Default separator is '-', producing a three-part hyphenated name."""
    name = generate_random_collection_name()
    parts = name.split("-")
    assert len(parts) == 3


def test_generate_name_custom_separator():
    """Custom separator is used between the three parts."""
    name = generate_random_collection_name(sep="_")
    parts = name.split("_")
    assert len(parts) == 3


def test_generate_name_parts_from_word_lists():
    """Each part of the name belongs to the corresponding word list."""
    # Run several times to reduce false-positive risk from random selection.
    for _ in range(20):
        name = generate_random_collection_name()
        adjective, verb, noun = name.split("-")
        assert adjective in ADJECTIVES
        assert verb in VERBS
        assert noun in NOUNS


def test_generate_name_parts_are_non_empty():
    """All three name parts are non-empty strings."""
    name = generate_random_collection_name()
    for part in name.split("-"):
        assert len(part) > 0


def test_generate_name_is_lowercase():
    """Generated name is entirely lowercase."""
    name = generate_random_collection_name()
    assert name == name.lower()


def test_generate_name_randomness():
    """Repeated calls produce at least two distinct names out of ten."""
    names = {generate_random_collection_name() for _ in range(10)}
    assert len(names) > 1


def test_generate_name_separator_appears_exactly_twice():
    """The separator appears exactly twice in the output for a three-part name."""
    name = generate_random_collection_name(sep="-")
    assert name.count("-") == 2


def test_generate_name_empty_separator():
    """An empty separator joins the three words into one continuous string."""
    name = generate_random_collection_name(sep="")
    # The name should be non-empty and consist of letters only.
    assert len(name) > 0
    assert name.isalpha()
