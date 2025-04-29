"""BaseGenie is the base class for all genies."""

from abc import ABC, abstractmethod
from typing import Any, List


class BaseGenie(ABC):
  """Abstract Base Class for Genies.

  This class defines the common interface for all Genie implementations. Genies
  are responsible for generating text or structured JSON responses based on
  input prompts.

  Subclasses must implement the `generate` method. If structured JSON output
  is required, subclasses should also implement the `generate_json` method.
  The batch generation methods (`generate_batch`, `generate_json_batch`) are
  provided for convenience and typically do not need to be overridden.

  Methods:
      generate(prompt: str) -> str:
          Generates a text response for a single prompt. Must be implemented by subclasses.
      generate_batch(prompts: List[str]) -> List[str]:
          Generates text responses for a batch of prompts. Uses `generate`.
      generate_json(prompt: str, schema: Any) -> Any:
          Generates a structured JSON response conforming to the provided schema
          for a single prompt. Should be implemented by subclasses if JSON output
          is needed.
      generate_json_batch(prompts: List[str], schema: Any) -> List[Any]:
          Generates structured JSON responses for a batch of prompts. Uses `generate_json`.
        
  """

  @abstractmethod
  def generate(self, prompt: str) -> str:
    """Generate a response based on the given prompt."""
    raise NotImplementedError

  def generate_batch(self, prompts: List[str]) -> List[str]:
    """Generate a batch of responses based on the given prompts."""
    return [self.generate(prompt) for prompt in prompts]

  def generate_json(self, prompt: str, schema: Any) -> Any:
    """Generate a JSON response based on the given prompt and BaseModel schema."""
    raise NotImplementedError

  def generate_json_batch(self, prompts: List[str], schema: Any) -> List[Any]:
    """Generate a batch of JSON responses based on the given prompts and BaseModel schema."""
    return [self.generate_json(prompt, schema) for prompt in prompts]