from dataclasses import dataclass
from typing import Optional

@dataclass
class TXTExtractorConfig:
    encoding: str = "utf-8"
    extract_metadata: bool = True
    
    def __post_init__(self):
        if not isinstance(self.encoding, str):
            raise ValueError("Encoding must be a string") 