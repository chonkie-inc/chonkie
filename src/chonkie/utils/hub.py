"""Module for managing access to the Chonkie hub."""

import importlib.util as importutil
import json
from pathlib import Path
from typing import Dict, Optional


class Hubbie: 
    """Hubbie is a Huggingface hub manager for Chonkie.
    
    Methods:
        get_recipe(recipe_name: str, lang: Optional[str] = 'en') -> Optional[Dict]:
            Get a recipe from the hub.
        get_recipe_schema() -> Dict:
            Get the current recipe schema from the hub.
    
    """

    SCHEMA_VERSION = "v1"

    def __init__(self) -> None:
        """Initialize Hubbie."""
        # Lazy import the dependencies (huggingface_hub)
        self._import_dependencies()

        # define the path to the recipes
        self.get_recipe_config = {
            "repo": "chonkie-ai/recipes",
            "subfolder": "recipes",
            "repo_type": "dataset",
        }

        # Fetch the current recipe schema from the hub
        self.recipe_schema = self.get_recipe_schema()

    def _import_dependencies(self) -> None:
        """Check if the required dependencies are available and import them."""
        try:
            if self._check_dependencies():
                global hfhub, jsonschema
                import huggingface_hub as hfhub
                import jsonschema
        except ImportError as error:
            raise ImportError(f"Tried importing dependencies but got error: {error}.")

    def _check_dependencies(self) -> Optional[bool]:
        """Check if the required dependencies are available."""
        dependencies = ["huggingface_hub", "jsonschema"]
        for dependency in dependencies:
            if importutil.find_spec(dependency) is None:
                raise ImportError(f"Tried importing {dependency} but it is not installed. Please install it via `pip install chonkie[hub]`")
        return True

    def get_recipe_schema(self) -> Dict:
        """Get the current recipe schema from the hub."""
        path = hfhub.hf_hub_download( # type: ignore
            repo_id="chonkie-ai/recipes",
            repo_type="dataset",
            filename=f"{self.SCHEMA_VERSION}.schema.json",
        )
        with Path(path).open("r") as f:
            return dict(json.loads(f.read()))

    def _validate_recipe(self, recipe: Dict) -> Optional[bool]:
        """Validate a recipe against the current schema."""
        try:
            jsonschema.validate(recipe, self.recipe_schema) # type: ignore
            return True
        except jsonschema.ValidationError as error: # type: ignore
            raise ValueError(f"Recipe is invalid. Please check the recipe and try again. Error: {error}")

    def get_recipe(self,
                  name: Optional[str] = 'default',
                  lang: Optional[str] = 'en',
                  path: Optional[str] = None) -> Dict:
        """Get a recipe from the hub.
        
        Args:
            name (Optional[str]): The name of the recipe to get.
            lang (Optional[str]): The language of the recipe to get.
            path (Optional[str]): Optionally, provide the path to the recipe.

        Returns:
            Optional[Dict]: The recipe.

        Raises:
            ValueError: If the recipe is not found.
            ValueError: If neither (name, lang) nor path are provided.
            ValueError: If the recipe is invalid.
            
        """
        # Check if either (name & lang) or path is provided
        if (name is None or lang is None) and path is None:
            raise ValueError("Either (name & lang) or path must be provided.")
        
        # If path is not provided, download the recipe from the hub
        if path is None and (name is not None and lang is not None):
            try:
                path = hfhub.hf_hub_download( # type: ignore
                    repo_id=self.get_recipe_config["repo"],
                    repo_type=self.get_recipe_config["repo_type"],
                    subfolder=self.get_recipe_config["subfolder"],
                    filename=f"{name}_{lang}.json",
                )
            except Exception as error:
                raise ValueError(f"Could not download recipe '{name}_{lang}'. Ensure name and lang are correct or provide a valid path. Error: {error}")

        # If we couldn't get the path or download the recipe, raise error
        if path is None:
            raise ValueError(f"Could not determine path for recipe '{name}_{lang}'. Ensure name and lang are correct or provide a valid path.")

        # using Pathlib to check if the file exists 
        path_obj = Path(path)
        if not path_obj.exists():
            raise ValueError(f"Failed to get the file {path} —— please check if this file exists and if the path is correct.")
        
        # Path exists, now open it and load the recipe
        try:
            with path_obj.open("r") as f:
                recipe = dict(json.loads(f.read()))
        except Exception as error:
            raise ValueError(f"Failed to read the file {path} —— please check if the file is valid JSON and if the path is correct. Error: {error}")
        
        # Validate the recipe with jsonschema
        assert self._validate_recipe(recipe), "Recipe is invalid. Please check the recipe and try again."
        
        # Return the recipe
        return recipe