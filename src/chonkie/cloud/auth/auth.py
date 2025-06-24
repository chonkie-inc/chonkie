"""Cloud Token Chunking for Chonkie API."""

import os
from typing import Optional

import requests


class Auth:
    """Validate API key for Chonkie Cloud."""

    BASE_URL = "https://api.chonkie.ai"
    VERSION = "v1"

    @staticmethod
    def validate(api_key: Optional[str] = None) -> bool:
        """Validate the API key."""
        # Define the payload for the request
        if api_key is None:
            api_key = os.getenv("CHONKIE_API_KEY")
        if api_key is None:
            raise ValueError(
                "No API key provided. Please set the CHONKIE_API_KEY environment variable "
                + "or pass an API key to the 'validate' method."
            )
        # Make the request to the Chonkie API
        response = requests.post(
            f"{Auth.BASE_URL}/{Auth.VERSION}/auth/validate",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        # Check if the response is successful
        if response.status_code != 200:
            raise ValueError(
                f"Error from the Chonkie API: {response.status_code} {response.text}"
            )
        return True
