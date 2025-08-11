"""Utility functions for Chonkie's Handshakes."""

import random

ADJECTIVES = [
    "happy", "chonky", "splashy", "munchy", "muddy", "groovy", "bubbly",
    "swift", "lazy", "hungry", "glowing", "radiant", "mighty", "gentle",
    "whimsical", "snug", "plump", "jovial", "sleepy", "sunny", "peppy",
    "breezy", "sneaky", "clever", "peaceful", "dreamy",
]

VERBS = [
    "chomping", "splashing", "munching", "wading", "floating", "drifting", "chunking",
    "slicing", "dancing", "wandering", "sleeping", "dreaming", "gliding", "swimming",
    "bubbling", "giggling", "jumping", "diving", "hopping", "skipping", "trotting", "sneaking",
    "exploring", "nibbling", "resting",
]

NOUNS = [
    "hippo", "river", "chunk", "lilypad", "mudbath", "stream", "pod", "chomp",
    "byte", "fragment", "slice", "splash", "nugget", "lagoon", "marsh",
    "pebble", "ripple", "cluster", "patch", "parcel", "meadow", "glade",
    "puddle", "nook", "bite", "whisper", "journey", "haven", "buddy", "pal",
    "snack", "secret"
]

def generate_random_collection_name(sep: str = "-") -> str:
    """Generate a random, fun, 3-part Chonkie-themed name (Adj-Verb-Noun).

    Combines one random adjective, one random verb, and one random noun from
    predefined lists, joined by a separator.

    Args:
        sep: The separator to use between the words. Defaults to "-".

    Returns:
        A randomly generated collection name string (e.g., "happy-splashes-hippo").

    """
    adjective = random.choice(ADJECTIVES)
    verb = random.choice(VERBS)
    noun = random.choice(NOUNS)
    return f"{adjective}{sep}{verb}{sep}{noun}"