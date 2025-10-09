"""Document type for Chonkie.

Documents allows chonkie to work together with other libraries that have their own 
document types — ensuring that the transition between libraries is as seamless as possible!

Additionally, documents are used to link together multiple sources of metadata that can be 
leveraged in downstream use-cases. One example of this would be in-line images, which are
stored as base64 encoded strings in the `metadata` field.

Lastly, documents are used by the chunkers to understand that they are working with chunks
of a document and not an assortment of text when dealing with hybrid/dual-mode chunking. 

This class is designed to be extended and might go through significant changes in the future.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .base import Chunk, generate_id


@dataclass
class Document:
    """Document type for Chonkie.
    
    Document allows us to encapsulate a text and its chunks, along with any additional 
    metadata. It becomes essential when dealing with complex chunking use-cases, such
    as dealing with in-line images, tables, or other non-text data. Documents are also 
    useful to give meaning when you want to chunk text that is already chunked, possibly
    with different chunkers.

    Args:
        id: The id of the document. If not provided, a random uuid will be generated.
        text: The complete text of the document.
        chunks: The chunks of the document.
        metadata: Any additional metadata you want to store about the document.
        
    """

    id: str = field(default_factory=lambda: generate_id("doc"))
    content: str = field(default_factory=str)
    chunks: List[Chunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)