"""Base class for all chef classes."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union


class BaseChef(ABC):
    """Base class for all chef classes.
    
    Chefs are responsible for preprocessing text before chunking.
    They can clean, normalize, and transform text to prepare it for chunking.
    """

    def __init__(self):
        """Initialize the chef."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the chef's dependencies are available.
        
        Returns:
            bool: True if the chef is available, False otherwise.
        """
        pass

    def _import_dependencies(self) -> None:
        """Lazy import dependencies.
        
        This method should be implemented by subclasses that have dependencies.
        """
        pass

    @abstractmethod
    def preprocess(self, text: str) -> str:
        """Preprocess the text.
        
        Args:
            text: The text to preprocess.
            
        Returns:
            str: The preprocessed text.
        """
        pass
    
    def __call__(self, text: str) -> str:
        """Call the chef to preprocess the text.
        
        Args:
            text: The text to preprocess.
            
        Returns:
            str: The preprocessed text.
        """
        return self.preprocess(text)
    
    def __repr__(self) -> str:
        """Return the string representation of the chef."""
        return f"{self.__class__.__name__}()" 