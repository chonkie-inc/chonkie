import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from ..base import BaseChef, ChefError
from .config import TXTExtractorConfig

class TXTExtractorChef(BaseChef):
    """Chef for extracting content from TXT files."""
    def __init__(self, config: Optional[TXTExtractorConfig] = None):
        self.config = config or TXTExtractorConfig()
    
    def validate(self, input_data: Union[str, Path]) -> bool:
        try:
            if isinstance(input_data, (str, Path)):
                if not os.path.exists(input_data):
                    raise ChefError(f"File not found: {input_data}")
                if not str(input_data).lower().endswith('.txt'):
                    raise ChefError(f"Not a TXT file: {input_data}")
            return True
        except Exception as e:
            raise ChefError(f"Error validating TXT: {str(e)}")
    
    def prepare(self, input_data: Union[str, Path]) -> Dict[str, Any]:
        try:
            if isinstance(input_data, (str, Path)):
                with open(input_data, 'r', encoding=self.config.encoding) as f:
                    text = f.read()
                result = {"text": text}
                if self.config.extract_metadata:
                    stat = os.stat(input_data)
                    result["metadata"] = {
                        "filename": os.path.basename(input_data),
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                    }
                return result
            else:
                raise ChefError("TXTExtractorChef only supports file paths.")
        except Exception as e:
            raise ChefError(f"Error extracting content from TXT: {str(e)}")
    
    def clean(self):
        pass 