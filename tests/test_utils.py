"""Test the utils module."""

import pytest

from chonkie.utils import Hubbie


@pytest.fixture
def hubbie() -> Hubbie:
    """Fixture to create a Hubbie instance."""
    return Hubbie()

def test_hubbie_initialization() -> None:
    """Test the Hubbie initialization."""
    hubbie = Hubbie()
    assert hubbie is not None
    assert isinstance(hubbie, Hubbie)
    
    # Check that the get_recipe_config is not None
    assert hubbie.get_recipe_config is not None

    # Check that the recipe_schema is not None
    assert hubbie.recipe_schema is not None
    assert isinstance(hubbie.recipe_schema, dict)
    assert "$schema" in hubbie.recipe_schema
    
    # Check that the dependencies are imported
    assert hubbie._check_dependencies() is True
    

def test_hubbie_get_recipe_hub(hubbie: Hubbie) -> None:
    """Test the Hubbie.get_recipe method."""
    recipe = hubbie.get_recipe("default", lang="en")
    assert recipe is not None
    assert isinstance(recipe, dict)
    assert "recipe" in recipe
    assert "recursive_rules" in recipe["recipe"]
    assert "levels" in recipe["recipe"]["recursive_rules"]

def test_hubbie_get_recipe_path(hubbie: Hubbie) -> None:
    """Test the Hubbie.get_recipe method with a path."""
    recipe = hubbie.get_recipe(path="tests/samples/recipe.json")
    assert recipe is not None
    assert isinstance(recipe, dict)
    assert "recipe" in recipe
    assert "recursive_rules" in recipe["recipe"]
    assert "levels" in recipe["recipe"]["recursive_rules"]

def test_hubbie_get_recipe_invalid(hubbie: Hubbie) -> None:
    """Test the Hubbie.get_recipe method with an invalid recipe."""
    # Check for the case where path is provided
    with pytest.raises(ValueError):
        hubbie.get_recipe(path="tests/samples/invalid_recipe.json")

    # Check for the case where name and lang are provided
    with pytest.raises(ValueError):
        hubbie.get_recipe(name="invalid", lang="en")

    # Check for the case where path is None
    with pytest.raises(ValueError):
        hubbie.get_recipe(name="invalid", lang="en", path="tests/samples/invalid_recipe.json")
    
    # Check for the case where lang is None
    with pytest.raises(ValueError):
        hubbie.get_recipe(lang=None)
    
    # Check for the case where name is None
    with pytest.raises(ValueError):
        hubbie.get_recipe(name=None)

def test_hubbie_validate_recipe(hubbie: Hubbie) -> None:
    """Test the Hubbie.validate_recipe method."""
    recipe = hubbie.get_recipe(path="tests/samples/recipe.json")
    assert recipe is not None
    assert hubbie._validate_recipe(recipe) is True

    with pytest.raises(ValueError):
        hubbie._validate_recipe({"recipe": {"recursive_rules": {"levels": "invalid"}}})


def test_hubbie_get_recipe_schema(hubbie: Hubbie) -> None:
    """Test the Hubbie.get_recipe_schema method."""
    schema = hubbie.get_recipe_schema()
    assert schema is not None
    assert isinstance(schema, dict)
    assert "$schema" in schema