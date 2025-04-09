"""Utility functions for Chonkie."""

import importlib.util as importutil
import json
from typing import Dict, Optional


class Hubbie: 
    """Hubbie is a Huggingface hub manager for Chonkie."""

    def __init__(self) -> None:
        """Initialize Hubbie."""
        # Lazy import the dependencies (huggingface_hub)
        self._import_dependencies()

        # define the path to the recipes
        self.recipes = {
            "repo": "chonkie-ai/recipes",
            "subfolder": "recipes",
            "repo_type": "dataset",
        }

    def _import_dependencies(self) -> None:
        """Check if the required dependencies are available and import them."""
        try:
            if self._check_dependencies():
                global hfhub
                import huggingface_hub as hfhub
        except ImportError as e:
            raise ImportError(f"Tried importing huggingface_hub but {e}.")

    def _check_dependencies(self) -> Optional[bool]:
        """Check if the required dependencies are available."""
        if importutil.find_spec("huggingface_hub") is not None:
            return True
        else:
            raise ImportError("Tried initializing Hubbie but `huggingface_hub` is not installed. Please install it via `pip install chonkie[hub]`")
        
    def get_recipe(self, recipe_name: str, lang: Optional[str] = 'en') -> Optional[Dict]:
        """Get a recipe from the hub."""
        try:
            recipe = hfhub.hf_hub_download( # type: ignore
                repo_id=self.recipes["repo"],
                repo_type=self.recipes["repo_type"],
                subfolder=self.recipes["subfolder"],
                filename=f"{recipe_name}_{lang}.json",
            )
            with open(recipe, "r") as f:
                return dict(json.loads(f.read()))
        except Exception as e:
            raise ValueError(f"Tried getting recipe `{recipe_name}_{lang}.json` but {e}.")