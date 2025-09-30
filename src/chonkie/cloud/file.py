"""File management functions for Chonkie API."""

import os
from dataclasses import dataclass
from typing import Optional

import requests

BASE_URL = "https://api.chonkie.ai"
VERSION = "v1"

@dataclass
class File:
    """File type for Chonkie API."""
    
    name: str
    size: str

    @classmethod
    def from_dict(cls, data: dict) -> "File":
        """Create a File object from a dictionary."""
        return cls(name=data["name"], size=data["size"])

class FileManager:
    """File management functions for Chonkie API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the FileManager."""
        self.api_key = api_key or os.getenv("CHONKIE_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided. Please set the CHONKIE_API_KEY environment variable or pass an API key to the FileManager constructor.")

    def upload(self, path: str) -> File:
        """Upload a file to the Chonkie API."""
        with open(path, "rb") as file:
            response = requests.post(f"{BASE_URL}/{VERSION}/files", files={"file": file}, headers={"Authorization": f"Bearer {self.api_key}"})
        if response.status_code != 201:
            raise ValueError(f"Error uploading file: {response.status_code} {response.text}")
        try:     
            return File.from_dict(response.json())
        except Exception as e:
            raise ValueError(f"Error uploading file: {e}")