import json
import os
from typing import Union


def get_config_path() -> str:
    """Get the path to the configuration file."""
    home_dir = os.path.expanduser("~")
    config_dir = os.path.join(home_dir, ".chonkie")
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    return os.path.join(config_dir, "config.json")


def login(api_key: str) -> None:
    """Set the API token in the configuration file."""
    config_path = get_config_path()
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # Config file is empty, malformed, or was deleted after check.
            # It will be overwritten.
            pass
    config["api_key"] = api_key
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f)
        print(f"token saved successfully in {config_path}")


def load_token() -> Union[str, None]:
    """Load the API token from a given key or environment variable."""
    api_key = os.getenv("CHONKIE_API_KEY", None)
    if api_key is not None:
        return api_key
    else:
        # TODO: load token from colab secrets if colab [WIP]

        # load token from local config file
        config_path = get_config_path()
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                api_key = config.get("api_key", None)
                if api_key:
                    return api_key
                else:
                    raise ValueError(
                        "API key not found in config file, consider logging in."
                    )
        else:
            raise ValueError("config file not found, consider logging in.")
