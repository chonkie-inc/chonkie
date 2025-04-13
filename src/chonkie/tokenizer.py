"""Module for abstracting tokeinization logic."""

import importlib
import inspect
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, Sequence, Union

if TYPE_CHECKING:
    # check if we can import tiktoken
    try:
        import tiktoken
    except ImportError:
        tiktoken = Any  # type: ignore # fallback to Any
    #  check if we can import tokenizers
    try:
        import tokenizers
    except ImportError:
        tokenizers = Any  # type: ignore # fallback to Any

    # check if we can import transformers
    try:
        import transformers
    except ImportError:
        transformers = Any  # type: ignore # fallback to Any


class BaseTokenizer(ABC):
    """Base class for Character and Word tokenizers."""

    def __init__(self) -> None:
        """Initialize the BaseTokenizer."""
        self.vocab: list[str] = []
        self.token2id: defaultdict[str, int] = defaultdict(lambda: len(self.vocab))
        self.token2id[" "]  # Add space to the vocabulary
        self.vocab.append(" ")  # Add space to the vocabulary

    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of the BaseTokenizer."""
        return f"{self.__class__.__name__}(vocab_size={len(self.vocab)})"

    def get_vocab(self) -> Sequence[str]:
        """Return the vocabulary."""
        return self.vocab

    def get_token2id(self) -> Dict:
        """Return token-to-id mapping."""
        return self.token2id

    @abstractmethod
    def encode(self, text: str) -> Sequence[int]:
        """Encode the given text into tokens.

        Args:
            text (str): The text to encode.

        Returns:
            Encoded sequence

        """
        raise NotImplementedError("Encoding not implemented for base tokenizer.")

    @abstractmethod
    def decode(self, tokens: Sequence[int]) -> str:
        """Decode the given tokens back into text.

        Args:
            tokens (Sequence[int]): The tokens to decode.

        Returns:
            Decoded text

        """
        raise NotImplementedError("Decoding not implemented for base tokenizer.")

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text.

        Args:
            text (str): The text to count tokens in.

        Returns:
            Number of tokens

        """
        raise NotImplementedError("Counting tokens not implemented for base tokenizer.")

    def encode_batch(self, texts: Sequence[str]) -> Sequence[Sequence[int]]:
        """Batch encode a list of texts into tokens.

        Args:
            texts (Sequence[str]): The texts to encode.

        Returns:
            List of encoded sequences

        """
        return [self.encode(text) for text in texts]

    def decode_batch(self, token_sequences: Sequence[Sequence[int]]) -> Sequence[str]:
        """Batch decode a list of tokens back into text.

        Args:
            token_sequences (Sequence[Sequence[int]]): The tokens to decode.

        Returns:
            List of decoded texts

        """
        return [self.decode(tokens) for tokens in token_sequences]

    def count_tokens_batch(self, texts: Sequence[str]) -> Sequence[int]:
        """Count the number of tokens in a batch of texts.

        Args:
            texts (Sequence[str]): The texts to count tokens in.

        Returns:
            List of token counts

        """
        return [self.count_tokens(text) for text in texts]


class CharacterTokenizer(BaseTokenizer):
    """Character-based tokenizer."""

    def __repr__(self) -> str:
        """Return a string representation of the CharacterTokenizer."""
        return f"CharacterTokenizer(vocab_size={len(self.vocab)})"

    def encode(self, text: str) -> Sequence[int]:
        """Encode the given text into tokens.

        Args:
            text (str): The text to encode.

        Returns:
            Encoded sequence

        """
        encoded = []
        for token in text:
            id = self.token2id[token]
            if id >= len(self.vocab):
                self.vocab.append(token)
            encoded.append(id)
        return encoded

    def decode(self, tokens: Sequence[int]) -> str:
        """Decode the given tokens back into text.

        Args:
            tokens (Sequence[int]): The tokens to decode.

        Returns:
            Decoded text

        """
        try:
            return "".join([self.vocab[token] for token in tokens])
        except Exception as e:
            raise ValueError(
                f"Decoding failed. Tokens: {tokens} not found in vocab."
            ) from e

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text.

        Args:
            text (str): The text to count tokens in.

        Returns:
            Number of tokens

        """
        return len(text)


class WordTokenizer(BaseTokenizer):
    """Word-based tokenizer."""

    def __repr__(self) -> str:
        """Return a string representation of the WordTokenizer."""
        return f"WordTokenizer(vocab_size={len(self.vocab)})"

    def tokenize(self, text: str) -> Sequence[str]:
        """Tokenize the given text into words.

        Args:
            text (str): The text to tokenize.

        Returns:
            List of tokens

        """
        return text.split(" ")

    def encode(self, text: str) -> Sequence[int]:
        """Encode the given text into tokens.

        Args:
            text (str): The text to encode.

        Returns:
            Encoded sequence

        """
        encoded = []
        for token in self.tokenize(text):
            id = self.token2id[token]
            if id >= len(self.vocab):
                self.vocab.append(token)
            encoded.append(id)
        return encoded

    def decode(self, tokens: Sequence[int]) -> str:
        """Decode token ids back to text."""
        try:
            return " ".join([self.vocab[token] for token in tokens])
        except Exception as e:
            raise ValueError(
                f"Decoding failed. Tokens: {tokens} not found in vocab."
            ) from e

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text.

        Args:
            text (str): The text to count tokens in.

        Returns:
            Number of tokens

        """
        return len(self.tokenize(text))


class Tokenizer:
    """Unified tokenizer interface for Chonkie.

    Args:
        tokenizer: Tokenizer identifier or instance.

    Raises:
        ImportError: If the specified tokenizer is not available.

    """

    def __init__(self, tokenizer: Union[str, Callable, Any] = "gpt2"):
        """Initialize the Tokenizer with a specified tokenizer."""
        if isinstance(tokenizer, str):
            self.tokenizer = self._load_tokenizer(tokenizer)
        else:
            self.tokenizer = tokenizer

        self._backend = self._get_backend()

    def _load_tokenizer(
        self, tokenizer: str
    ) -> Union[
        CharacterTokenizer,
        WordTokenizer,
        "tokenizers.Tokenizer",
        "tiktoken.Encoding",
        "transformers.PreTrainedTokenizer",
        "transformers.PreTrainedTokenizerFast",
        Callable[[str], int],
    ]:
        """Load the tokenizer based on the identifier."""
        if tokenizer == "character":
            return CharacterTokenizer()
        elif tokenizer == "word":
            return WordTokenizer()

        # Try tokenizers first
        if importlib.util.find_spec("tokenizers") is not None:
            try:
                from tokenizers import Tokenizer

                return Tokenizer.from_pretrained(tokenizer)
            except Exception:
                warnings.warn(
                    "Could not find 'tokenizers'. Falling back to 'tiktoken'."
                )
        else:
            warnings.warn("Could not find 'tokenizers'. Falling back to 'tiktoken'.")

        # Try tiktoken
        if importlib.util.find_spec("tiktoken") is not None:
            try:
                from tiktoken import get_encoding

                return get_encoding(tokenizer)
            except Exception:
                warnings.warn(
                    "Could not find 'tiktoken'. Falling back to 'transformers'."
                )
        else:
            warnings.warn("Could not find 'tiktoken'. Falling back to 'transformers'.")

        # Try transformers as last resort
        if importlib.util.find_spec("transformers") is not None:
            try:
                from transformers import AutoTokenizer

                return AutoTokenizer.from_pretrained(tokenizer)
            except Exception:
                raise ValueError(
                    "Tokenizer not found in transformers, tokenizers, or tiktoken"
                )
        raise ValueError("Tokenizer not found in transformers, tokenizers, or tiktoken")

    def _get_backend(self) -> str:
        """Get the tokenizer instance based on the identifier."""
        supported_backends = [
            "chonkie",
            "transformers",
            "tokenizers",
            "tiktoken",
        ]
        for backend in supported_backends:
            if backend in str(type(self.tokenizer)):
                return backend
        if (
            callable(self.tokenizer)
            or inspect.isfunction(self.tokenizer)
            or inspect.ismethod(self.tokenizer)
        ):
            return "callable"
        raise ValueError(f"Unsupported tokenizer backend: {type(self.tokenizer)}")

    def encode(self, text: str) -> Sequence[int]:
        """Encode the text into tokens.

        Args:
            text (str): The text to encode.

        Returns:
            Encoded sequence

        """
        # Supported backends
        if self._backend == "chonkie":
            return self.tokenizer.encode(text)  # type: ignore
        elif self._backend == "tiktoken":
            return self.tokenizer.encode(text)  # type: ignore
        elif self._backend == "transformers":
            return self.tokenizer.encode(text, add_special_tokens=False)  # type: ignore
        elif self._backend == "tokenizers":
            return self.tokenizer.encode(text, add_special_tokens=False).ids  # type: ignore

        # Not yet implemented backends
        if self._backend == "callable":
            raise NotImplementedError("Encoding not implemented for callable tokenizers.")

        raise ValueError(f"Unsupported tokenizer backend: {self._backend}")

    def decode(self, tokens: Sequence[int]) -> str:
        """Decode the tokens back into text.

        Args:
            tokens (Sequence[int]): The tokens to decode.

        Returns:
            Decoded text

        """
        if self._backend == "callable":
            raise NotImplementedError(
                "Decoding not implemented for callable tokenizers."
            )
        return self.tokenizer.decode(tokens)  # type: ignore

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text.

        Args:
            text (str): The text to count tokens in.

        Returns:
            Number of tokens

        """
        if self._backend == "chonkie":
            return self.tokenizer.count_tokens(text)  # type: ignore
        elif self._backend == "tiktoken":
            return len(self.tokenizer.encode(text))  # type: ignore
        elif self._backend == "transformers":
            return len(self.tokenizer.encode(text, add_special_tokens=False))  # type: ignore
        elif self._backend == "tokenizers":
            return len(self.tokenizer.encode(text, add_special_tokens=False).ids)  # type: ignore
        elif self._backend == "callable":
            return self.tokenizer(text)  # type: ignore
        raise ValueError(f"Unsupported tokenizer backend: {self._backend}")

    def encode_batch(self, texts: Sequence[str]) -> Sequence[Sequence[int]]:
        """Batch encode a list of texts into tokens.

        Args:
            texts (Sequence[str]): The texts to encode.

        Returns:
            List of encoded sequences

        """
        if self._backend == "chonkie":
            return self.tokenizer.encode_batch(texts)  # type: ignore
        elif self._backend == "tiktoken":
            return self.tokenizer.encode_batch(texts)  # type: ignore
        elif self._backend == "transformers":
            encoded = self.tokenizer.batch_encode_plus(texts, add_special_tokens=False)  # type: ignore
            return encoded["input_ids"]  # type: ignore
        elif self._backend == "tokenizers":
            return [encoding.ids for encoding in self.tokenizer.encode_batch(texts)]  # type: ignore
        if self._backend == "callable":
            raise NotImplementedError(
                "Batch encoding not implemented for callable tokenizers."
            )
        raise ValueError(f"Unsupported tokenizer backend: {self._backend}")

    def decode_batch(self, token_sequences: Sequence[Sequence[int]]) -> Sequence[str]:
        """Batch decode a list of tokens back into text.

        Args:
            token_sequences (Sequence[Sequence[int]]): The tokens to decode.

        Returns:
            List of decoded texts

        """
        if self._backend == "chonkie":
            return self.tokenizer.decode_batch(token_sequences)  # type: ignore
        elif self._backend in "tiktoken":
            return self.tokenizer.decode_batch(token_sequences)  # type: ignore
        elif self._backend in "tokenizers":
            return self.tokenizer.decode_batch(token_sequences)  # type: ignore
        elif self._backend == "transformers":
            return self.tokenizer.batch_decode(token_sequences, skip_special_tokens=True)  # type: ignore

        if self._backend == "callable":
            raise NotImplementedError(
                "Batch decoding not implemented for callable tokenizers."
            )
        else:
            raise ValueError(f"Unsupported tokenizer backend: {self._backend}")

    def count_tokens_batch(self, texts: Sequence[str]) -> Sequence[int]:
        """Count the number of tokens in a batch of texts.

        Args:
            texts (Sequence[str]): The texts to count tokens in.

        Returns:
            List of token counts

        """
        if self._backend == "chonkie":
            return self.tokenizer.count_tokens_batch(texts)  # type: ignore
        elif self._backend == "tiktoken":
            return [len(token_list) for token_list in self.tokenizer.encode_batch(texts)]  # type: ignore
        elif self._backend == "transformers":
            return [len(token_list) for token_list in self.tokenizer.batch_encode_plus(texts, add_special_tokens=False)["input_ids"]]  # type: ignore
        elif self._backend == "tokenizers":
            return [len(t.ids) for t in self.tokenizer.encode_batch(texts, add_special_tokens=False)]  # type: ignore
        elif self._backend == "callable":
            return [self.tokenizer(text) for text in texts]  # type: ignore

        raise ValueError(f"Tokenizer backend {self._backend} not supported.")
