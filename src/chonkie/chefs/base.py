"""Base Chef interface for data preparation in Chonkie.

This module provides the core interfaces and base classes for implementing
data preparation chefs in Chonkie.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypeVar, Generic

from pydantic import BaseModel, Field

T = TypeVar('T')
R = TypeVar('R')

class ChefConfig(BaseModel):
    """Base configuration for chefs."""
    name: str = Field(..., description="Name of the chef")
    description: Optional[str] = Field(None, description="Description of the chef's functionality")
    enabled: bool = Field(True, description="Whether the chef is enabled")
    config: Dict[str, Any] = Field(default_factory=dict, description="Additional configuration options")

class BaseChef(Generic[T, R], ABC):
    """Base class for all chefs in Chonkie.
    
    This abstract base class defines the interface that all chefs must implement.
    Chefs are responsible for preparing and cleaning data before it is chunked.
    
    Type Parameters:
        T: The input type that the chef processes
        R: The output type that the chef produces
    """
    
    def __init__(self, config: Optional[ChefConfig] = None):
        """Initialize the chef with optional configuration.
        
        Args:
            config: Optional configuration for the chef
        """
        self.config = config or ChefConfig(name=self.__class__.__name__)
    
    @abstractmethod
    def prepare(self, data: T) -> R:
        """Prepare the data for chunking.
        
        This is the main method that processes the input data and returns
        the prepared data ready for chunking.
        
        Args:
            data: The input data to prepare
            
        Returns:
            The prepared data ready for chunking
            
        Raises:
            ChefError: If there is an error during preparation
        """
        pass
    
    @abstractmethod
    def validate(self, data: T) -> bool:
        """Validate the input data.
        
        Args:
            data: The data to validate
            
        Returns:
            True if the data is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def clean(self, data: R) -> R:
        """Clean the prepared data.
        
        Args:
            data: The prepared data to clean
            
        Returns:
            The cleaned data
        """
        pass
    
    def __call__(self, data: T) -> R:
        """Process the data through the chef's pipeline.
        
        This method implements the full pipeline:
        1. Validate the input
        2. Prepare the data
        3. Clean the prepared data
        
        Args:
            data: The input data to process
            
        Returns:
            The processed data ready for chunking
            
        Raises:
            ChefError: If there is an error during processing
        """
        if not self.validate(data):
            raise ChefError(f"Invalid input data for chef {self.config.name}")
        
        prepared_data = self.prepare(data)
        return self.clean(prepared_data)

class ChefError(Exception):
    """Base exception for chef-related errors."""
    pass 