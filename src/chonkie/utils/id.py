"""Generate IDs for Chonkie types."""

from uuid import uuid4


def generate_id(prefix: str) -> str:
    """Generate a UUID for a given prefix."""
    return f"{prefix}_{uuid4().hex}"