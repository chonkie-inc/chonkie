"""Utility functions for Chonkie's Handshakes."""

import random
from typing import List, Sequence, Union

from chonkie.types import Chunk

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


def normalize_chunks(chunks: Union[Chunk, Sequence[Chunk], Sequence[Sequence[Chunk]]]) -> List[Chunk]:
    """Convert various chunk formats into a flat list of Chunk objects.
    
    This function handles different input formats:
    - A single Chunk object
    - A sequence of Chunk objects
    - A nested sequence of Chunk objects (sequence of sequences)
    
    Args:
        chunks: Input chunks in various formats (single Chunk, sequence of Chunks,
               or sequence of sequences of Chunks)
               
    Returns:
        List[Chunk]: A flat list containing all valid Chunk objects
        
    Example:
        >>> chunk = Chunk(text="example", start_index=0, end_index=7, token_count=1)
        >>> normalize_chunks(chunk)
        [Chunk(text='example', start_index=0, end_index=7, token_count=1)]
        
        >>> chunks = [chunk1, chunk2, chunk3]
        >>> normalize_chunks(chunks)
        [chunk1, chunk2, chunk3]
        
        >>> nested_chunks = [[chunk1, chunk2], [chunk3]]
        >>> normalize_chunks(nested_chunks)
        [chunk1, chunk2, chunk3]
        
    """
    # Initialize an empty list to store all valid Chunks
    chunk_list: List[Chunk] = []
    
    if isinstance(chunks, Chunk):
        # Single Chunk object
        chunk_list = [chunks]
    elif isinstance(chunks, Sequence):
        if not chunks:
            # Empty sequence
            return []
        
        if isinstance(chunks[0], Chunk):
            # Sequence[Chunk]
            for item in chunks:
                if isinstance(item, Chunk):
                    chunk_list.append(item)
        elif isinstance(chunks[0], Sequence):
            # Sequence[Sequence[Chunk]]
            for chunk_seq in chunks:
                if isinstance(chunk_seq, Sequence):
                    for item in chunk_seq:
                        if isinstance(item, Chunk):
                            chunk_list.append(item)
    
    return chunk_list