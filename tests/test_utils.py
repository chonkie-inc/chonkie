"""Test the utils module."""

import json
import os

import pytest

from chonkie.utils import Hubbie
from chonkie.utils._api import get_config_path, load_token, login

try:
    import huggingface_hub  # noqa: F401

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

requires_hf_hub = pytest.mark.skipif(not HF_HUB_AVAILABLE, reason="huggingface_hub not installed")


@pytest.fixture
def hubbie() -> Hubbie:
    """Fixture to create a Hubbie instance."""
    return Hubbie()


@requires_hf_hub
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


@requires_hf_hub
def test_hubbie_get_recipe_hub(hubbie: Hubbie) -> None:
    """Test the Hubbie.get_recipe method."""
    recipe = hubbie.get_recipe("default", lang="en")
    assert recipe is not None
    assert isinstance(recipe, dict)
    assert "recipe" in recipe
    assert "recursive_rules" in recipe["recipe"]
    assert "levels" in recipe["recipe"]["recursive_rules"]


@requires_hf_hub
def test_hubbie_get_recipe_path(hubbie: Hubbie) -> None:
    """Test the Hubbie.get_recipe method with a path."""
    recipe = hubbie.get_recipe(path="tests/samples/recipe.json")
    assert recipe is not None
    assert isinstance(recipe, dict)
    assert "recipe" in recipe
    assert "recursive_rules" in recipe["recipe"]
    assert "levels" in recipe["recipe"]["recursive_rules"]


@requires_hf_hub
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


@requires_hf_hub
def test_hubbie_validate_recipe(hubbie: Hubbie) -> None:
    """Test the Hubbie.validate_recipe method."""
    recipe = hubbie.get_recipe(path="tests/samples/recipe.json")
    assert recipe is not None
    assert hubbie._validate_recipe(recipe) is True

    with pytest.raises(ValueError):
        hubbie._validate_recipe({"recipe": {"recursive_rules": {"levels": "invalid"}}})


@requires_hf_hub
def test_hubbie_get_recipe_schema(hubbie: Hubbie) -> None:
    """Test the Hubbie.get_recipe_schema method."""
    schema = hubbie.get_recipe_schema()
    assert schema is not None
    assert isinstance(schema, dict)
    assert "$schema" in schema


# ---------------------------------------------------------------------------
# _api.py tests
# ---------------------------------------------------------------------------


def test_get_config_path_returns_string(tmp_path, monkeypatch):
    """get_config_path returns a path string ending in config.json."""
    monkeypatch.setenv("HOME", str(tmp_path))
    path = get_config_path()
    assert isinstance(path, str)
    assert path.endswith("config.json")


def test_get_config_path_creates_directory(tmp_path, monkeypatch):
    """get_config_path creates the ~/.chonkie directory if it does not exist."""
    monkeypatch.setenv("HOME", str(tmp_path))
    chonkie_dir = tmp_path / ".chonkie"
    assert not chonkie_dir.exists()
    get_config_path()
    assert chonkie_dir.exists()


def test_login_writes_api_key(tmp_path, monkeypatch):
    """login() writes the api_key to the config file as valid JSON."""
    monkeypatch.setenv("HOME", str(tmp_path))
    login("test-key-123")
    config_path = get_config_path()
    with open(config_path) as f:
        data = json.load(f)
    assert data["api_key"] == "test-key-123"


def test_login_updates_existing_key(tmp_path, monkeypatch):
    """login() replaces an existing api_key in the config file."""
    monkeypatch.setenv("HOME", str(tmp_path))
    login("first-key")
    login("second-key")
    config_path = get_config_path()
    with open(config_path) as f:
        data = json.load(f)
    assert data["api_key"] == "second-key"


def test_login_preserves_other_config_keys(tmp_path, monkeypatch):
    """login() preserves pre-existing keys in the config file."""
    monkeypatch.setenv("HOME", str(tmp_path))
    config_path = get_config_path()
    with open(config_path, "w") as f:
        json.dump({"other_key": "other_value"}, f)
    login("my-key")
    with open(config_path) as f:
        data = json.load(f)
    assert data["api_key"] == "my-key"
    assert data["other_key"] == "other_value"


def test_login_handles_malformed_json(tmp_path, monkeypatch):
    """login() overwrites a malformed config file gracefully."""
    monkeypatch.setenv("HOME", str(tmp_path))
    config_path = get_config_path()
    with open(config_path, "w") as f:
        f.write("not valid json {{{{")
    login("recovery-key")
    with open(config_path) as f:
        data = json.load(f)
    assert data["api_key"] == "recovery-key"


def test_load_token_from_env_variable(tmp_path, monkeypatch):
    """load_token() returns the value of CHONKIE_API_KEY when set."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CHONKIE_API_KEY", "env-token")
    token = load_token()
    assert token == "env-token"


def test_load_token_from_config_file(tmp_path, monkeypatch):
    """load_token() reads the token from the config file when env var is absent."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CHONKIE_API_KEY", raising=False)
    login("file-token")
    token = load_token()
    assert token == "file-token"


def test_load_token_env_takes_priority(tmp_path, monkeypatch):
    """load_token() prefers CHONKIE_API_KEY over the config file."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CHONKIE_API_KEY", "env-wins")
    login("file-token")
    token = load_token()
    assert token == "env-wins"


def test_load_token_raises_when_no_config(tmp_path, monkeypatch):
    """load_token() raises ValueError when neither env var nor config file exists."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CHONKIE_API_KEY", raising=False)
    # Ensure no config file exists
    config_path = tmp_path / ".chonkie" / "config.json"
    if config_path.exists():
        config_path.unlink()
    # Remove the directory too so get_config_path recreates it
    chonkie_dir = tmp_path / ".chonkie"
    if chonkie_dir.exists():
        import shutil
        shutil.rmtree(str(chonkie_dir))
    with pytest.raises(ValueError):
        load_token()


def test_load_token_raises_when_api_key_missing_from_config(tmp_path, monkeypatch):
    """load_token() raises ValueError when config exists but has no api_key."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CHONKIE_API_KEY", raising=False)
    config_path = get_config_path()
    with open(config_path, "w") as f:
        json.dump({"other_key": "value"}, f)
    with pytest.raises(ValueError):
        load_token()
