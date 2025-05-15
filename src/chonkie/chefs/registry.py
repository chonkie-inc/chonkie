"""Chef registry for managing and instantiating chefs in Chonkie.

This module provides a registry for managing different types of chefs
and creating instances of them with appropriate configurations.
"""

from typing import Dict, Optional, Type, TypeVar, Generic, Any, Union
from pathlib import Path
import os

from .base import BaseChef, ChefConfig, ChefError
from .pdf.extractor import PDFExtractorChef, PDFExtractorConfig
from .docx.extractor import DOCXExtractorChef, DOCXExtractorConfig
from .txt.extractor import TXTExtractorChef
from .txt.config import TXTExtractorConfig

T = TypeVar('T')
R = TypeVar('R')

class ChefRegistry:
    """Registry for managing and instantiating document processing chefs."""
    
    def __init__(self):
        """Initialize the chef registry with default mappings."""
        self._chefs: Dict[str, Type[BaseChef]] = {}
        self._configs: Dict[str, Type[Any]] = {}
        self._register_default_chefs()
    
    def _register_default_chefs(self) -> None:
        """Register the default set of chefs."""
        self.register_chef(".pdf", PDFExtractorChef, PDFExtractorConfig)
        self.register_chef(".docx", DOCXExtractorChef, DOCXExtractorConfig)
        self.register_chef(".txt", TXTExtractorChef, TXTExtractorConfig)
    
    def register_chef(self, extension: str, chef_class: Type[BaseChef], config_class: Optional[Type[Any]] = None) -> None:
        """Register a new chef class for a specific file extension.
        
        Args:
            extension: File extension (e.g., '.pdf', '.docx')
            chef_class: Chef class to register
            config_class: Optional config class for the chef
            
        Raises:
            ChefError: If extension is invalid or chef class is not a BaseChef
        """
        if not extension.startswith('.'):
            raise ChefError(f"Invalid extension format: {extension}. Must start with '.'")
        
        if not issubclass(chef_class, BaseChef):
            raise ChefError(f"Invalid chef class: {chef_class}. Must inherit from BaseChef")
        
        self._chefs[extension.lower()] = chef_class
        if config_class:
            self._configs[extension.lower()] = config_class
    
    def get_chef(self, file_path: Union[str, Path], **kwargs) -> BaseChef:
        """Get the appropriate chef instance for a given file.
        
        Args:
            file_path: Path to the file to process
            **kwargs: Additional arguments to pass to the chef constructor
            
        Returns:
            An instance of the appropriate chef
            
        Raises:
            ChefError: If no chef is registered for the file type
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        extension = file_path.suffix.lower()
        if extension not in self._chefs:
            raise ChefError(f"No chef registered for file type: {extension}")
        
        chef_class = self._chefs[extension]
        
        # If we have a config class and kwargs, create a config object
        if extension in self._configs and kwargs is not None:
            config_class = self._configs[extension]
            # Only inject 'name' if the config class expects it (Pydantic-based)
            if hasattr(config_class, '__fields__') and 'name' in getattr(config_class, '__fields__', {}):
                if 'name' not in kwargs:
                    kwargs['name'] = f"auto_{extension[1:]}"
            config = config_class(**kwargs)
            return chef_class(config)
        
        return chef_class(**kwargs)
    
    def get_supported_extensions(self) -> list[str]:
        """Get a list of all supported file extensions.
        
        Returns:
            List of supported file extensions
        """
        return list(self._chefs.keys())
    
    def is_supported(self, file_path: Union[str, Path]) -> bool:
        """Check if a file type is supported.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if the file type is supported, False otherwise
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        return file_path.suffix.lower() in self._chefs
    
    # Backward compatibility methods
    def register(self, name: str, chef_class: Type[BaseChef]) -> None:
        """Register a chef by name (for backward compatibility).
        
        Args:
            name: Name to register the chef under
            chef_class: Chef class to register
            
        Raises:
            ChefError: If name is invalid or chef class is not a BaseChef
        """
        if not name:
            raise ChefError("Invalid chef name")
        
        if not issubclass(chef_class, BaseChef):
            raise ChefError(f"Invalid chef class: {chef_class}. Must inherit from BaseChef")
        
        # Map the name to a file extension
        extension = f".{name.lower()}"
        self.register_chef(extension, chef_class)
    
    def unregister(self, name: str) -> None:
        """Unregister a chef by name (for backward compatibility).
        
        Args:
            name: Name of the chef to unregister
            
        Raises:
            ChefError: If no chef is registered under the given name
        """
        extension = f".{name.lower()}"
        if extension not in self._chefs:
            raise ChefError(f"No chef registered under name '{name}'")
        
        del self._chefs[extension]
        if extension in self._configs:
            del self._configs[extension]
    
    def list_chefs(self) -> Dict[str, Type[BaseChef]]:
        """List all registered chefs (for backward compatibility).
        
        Returns:
            A dictionary mapping chef names to their classes
        """
        return {ext[1:]: chef for ext, chef in self._chefs.items()}

# Global registry instance
registry = ChefRegistry() 