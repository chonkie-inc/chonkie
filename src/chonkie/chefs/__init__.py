"""Chonkie Chefs - Data Preparation Framework.

This package provides a framework for preparing and cleaning data before chunking.
"""

from .base import BaseChef, ChefConfig, ChefError
from .registry import registry, ChefRegistry

__all__ = [
    'BaseChef',
    'ChefConfig',
    'ChefError',
    'registry',
    'ChefRegistry',
] 