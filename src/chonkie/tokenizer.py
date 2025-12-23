"""Module for abstracting tokeinization logic."""

import importlib
import inspect
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Protocol, Sequence, Union

if TYPE_CHECKING:
    import tiktoken
    import tokenizers
    import transformers


class TokenizerProtocol(Protocol):
    """Protocol defining the interface for tokenizers.

    Any object implementing these methods can be used as a tokenizer in Chonkie.
    """

    def encode(self, text: str) -> Sequence[int]:
        """Encode text into token IDs.

        Args:
            text: The text to encode.

        Returns:
            Sequence of token IDs.

        """
        ...

    def decode(self, tokens: Sequence[int]) -> str:
        """Decode token IDs back to text.

        Args:
            tokens: Sequence of token IDs.

        Returns:
            Decoded text string.

        """
        ...

    def tokenize(self, text: str) -> Sequence[Union[str, int]]:
        """Tokenize text into tokens.

        Args:
            text: The text to tokenize.

        Returns:
            Sequence of tokens (strings or IDs).

        """
        ...


class Tokenizer(ABC):
    """Base class for tokenizers implementing the TokenizerProtocol.

    This class provides a foundation for creating custom tokenizers with
    vocabulary management and default implementations of common methods.
    """

    def __init__(self) -> None:
        """Initialize the Tokenizer."""
        self.vocab: list[str] = []
        self.token2id: dict[str, int] = defaultdict(self.defaulttoken2id)
        # Note: Using a lambda here would cause pickling issues:
        # self.token2id: dict[str, int] = defaultdict(lambda: len(self.vocab))
        self.token2id[" "]  # Add space to the vocabulary
        self.vocab.append(" ")  # Add space to the vocabulary

    def defaulttoken2id(self) -> int:
        """Return the default token ID.

        This method is used as the default_factory for defaultdict.
        Using a named method instead of a lambda ensures the object can be pickled.
        """
        return len(self.vocab)

    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of the Tokenizer."""
        return f"{self.__class__.__name__}(vocab_size={len(self.vocab)})"

    def get_vocab(self) -> Sequence[str]:
        """Return the vocabulary."""
        return self.vocab

    def get_token2id(self) -> dict:
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
    def tokenize(self, text: str) -> Sequence[Union[str, int]]:
        """Tokenize the given text.

        Args:
            text (str): The text to tokenize.

        Returns:
            Sequence of tokens (strings or token IDs)

        """
        raise NotImplementedError("Tokenization not implemented for base tokenizer.")

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text.

        Default implementation uses tokenize() method.

        Args:
            text (str): The text to count tokens in.

        Returns:
            Number of tokens

        """
        return len(self.tokenize(text))

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


class CharacterTokenizer(Tokenizer):
    """Character-based tokenizer."""

    def __repr__(self) -> str:
        """Return a string representation of the CharacterTokenizer."""
        return f"CharacterTokenizer(vocab_size={len(self.vocab)})"

    def tokenize(self, text: str) -> Sequence[str]:
        """Tokenize text into individual characters.

        Args:
            text (str): The text to tokenize.

        Returns:
            List of characters

        """
        return list(text)

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
            raise ValueError(f"Decoding failed. Tokens: {tokens} not found in vocab.") from e

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text.

        Args:
            text (str): The text to count tokens in.

        Returns:
            Number of tokens

        """
        return len(text)


class WordTokenizer(Tokenizer):
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
            raise ValueError(f"Decoding failed. Tokens: {tokens} not found in vocab.") from e

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text.

        Args:
            text (str): The text to count tokens in.

        Returns:
            Number of tokens

        """
        return len(self.tokenize(text))


class ByteTokenizer(Tokenizer):
    """Byte-based tokenizer that operates on UTF-8 encoded bytes."""

    def __repr__(self) -> str:
        """Return a string representation of the ByteTokenizer."""
        return f"ByteTokenizer(vocab_size={len(self.vocab)})"

    def tokenize(self, text: str) -> Sequence[int]:
        """Tokenize text into individual bytes.

        Args:
            text (str): The text to tokenize.

        Returns:
            List of byte values

        """
        return list(text.encode("utf-8"))

    def encode(self, text: str) -> Sequence[int]:
        """Encode the given text into byte tokens.

        Args:
            text (str): The text to encode.

        Returns:
            Encoded sequence of byte values

        """
        return list(text.encode("utf-8"))

    def decode(self, tokens: Sequence[int]) -> str:
        """Decode byte tokens back into text.

        Args:
            tokens (Sequence[int]): The byte tokens to decode.

        Returns:
            Decoded text

        """
        try:
            return bytes(tokens).decode("utf-8")
        except Exception as e:
            raise ValueError(
                f"Decoding failed. Tokens: {tokens} cannot be decoded as UTF-8.",
            ) from e

    def count_tokens(self, text: str) -> int:
        """Count the number of byte tokens in the given text.

        Args:
            text (str): The text to count tokens in.

        Returns:
            Number of byte tokens

        """
        return len(text.encode("utf-8"))


class RowTokenizer(Tokenizer):
    """Row-based tokenizer that counts lines/rows in text.

    This tokenizer treats each line (separated by newlines) as a token.
    It is primarily useful for table chunking where you want to chunk
    by number of rows rather than by character or subword tokens.
    """

    def __repr__(self) -> str:
        """Return a string representation of the RowTokenizer."""
        return f"RowTokenizer(vocab_size={len(self.vocab)})"

    def tokenize(self, text: str) -> Sequence[str]:
        """Tokenize text into individual lines/rows.

        Args:
            text (str): The text to tokenize.

        Returns:
            List of lines/rows

        """
        if not text:
            return []
        return text.split("\n")

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
        """Decode the given tokens back into text.

        Args:
            tokens (Sequence[int]): The tokens to decode.

        Returns:
            Decoded text

        """
        try:
            return "\n".join([self.vocab[token] for token in tokens])
        except Exception as e:
            raise ValueError(f"Decoding failed. Tokens: {tokens} not found in vocab.") from e

    def count_tokens(self, text: str) -> int:
        """Count the number of rows/lines in the given text.

        Args:
            text (str): The text to count tokens in.

        Returns:
            Number of rows/lines

        """
        if not text:
            return 0
        return len(text.split("\n"))


class AutoTokenizer:
    """Auto-loading tokenizer interface for Chonkie.

    This class provides automatic loading of tokenizers from various sources
    (string identifiers, HuggingFace models, tiktoken, etc.) and wraps them
    with a unified interface.

    Args:
        tokenizer: Tokenizer identifier or instance.

    Raises:
        ImportError: If the specified tokenizer is not available.

    """

    def __init__(self, tokenizer: Union[str, Callable, Any] = "character"):
        """Initialize the AutoTokenizer with a specified tokenizer."""
        if isinstance(tokenizer, AutoTokenizer):
            self.tokenizer = tokenizer.tokenizer  # type: ignore[has-type]
            self._backend = tokenizer._backend  # type: ignore[has-type]
        elif isinstance(tokenizer, str):
            self.tokenizer = self._load_tokenizer(tokenizer)
            self._backend = self._get_backend()
        else:
            self.tokenizer = tokenizer
            self._backend = self._get_backend()

    def _load_tokenizer(
        self,
        tokenizer: str,
    ) -> Union[
        CharacterTokenizer,
        WordTokenizer,
        ByteTokenizer,
        RowTokenizer,
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
        elif tokenizer == "byte":
            return ByteTokenizer()
        elif tokenizer == "row":
            return RowTokenizer()

        # Try tokenizers first
        if importlib.util.find_spec("tokenizers") is not None:
            try:
                from tokenizers import Tokenizer as HFTokenizer

                return HFTokenizer.from_pretrained(tokenizer)
            except Exception:
                warnings.warn(
                    "Could not load tokenizer with 'tokenizers'. Falling back to 'tiktoken'.",
                )
        else:
            warnings.warn("'tokenizers' library not found. Falling back to 'tiktoken'.")

        # Try tiktoken
        if importlib.util.find_spec("tiktoken") is not None:
            try:
                from tiktoken import get_encoding

                return get_encoding(tokenizer)
            except Exception:
                warnings.warn(
                    "Could not load tokenizer with 'tiktoken'. Falling back to 'transformers'.",
                )
        else:
            warnings.warn("'tiktoken' library not found. Falling back to 'transformers'.")

        # Try transformers as last resort
        if importlib.util.find_spec("transformers") is not None:
            try:
                from transformers import AutoTokenizer

                return AutoTokenizer.from_pretrained(tokenizer)
            except Exception:
                raise ValueError("Tokenizer not found in transformers, tokenizers, or tiktoken")
        raise ValueError("Tokenizer not found in transformers, tokenizers, or tiktoken")

    def _get_backend(self) -> str:
        """Get the tokenizer instance based on the identifier."""
        if isinstance(self.tokenizer, Tokenizer):
            return "chonkie"

        supported_backends = [
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
            return self.tokenizer.encode(text)
        elif self._backend == "tiktoken":
            return self.tokenizer.encode(text)
        elif self._backend == "transformers":
            return self.tokenizer.encode(text, add_special_tokens=False)
        elif self._backend == "tokenizers":
            return self.tokenizer.encode(text, add_special_tokens=False).ids

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
            raise NotImplementedError("Decoding not implemented for callable tokenizers.")
        return self.tokenizer.decode(tokens)

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text.

        Args:
            text (str): The text to count tokens in.

        Returns:
            Number of tokens

        """
        if self._backend == "chonkie":
            return self.tokenizer.count_tokens(text)
        elif self._backend == "tiktoken":
            return len(self.tokenizer.encode(text))
        elif self._backend == "transformers":
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        elif self._backend == "tokenizers":
            return len(self.tokenizer.encode(text, add_special_tokens=False).ids)
        elif self._backend == "callable":
            return self.tokenizer(text)
        raise ValueError(f"Unsupported tokenizer backend: {self._backend}")

    def encode_batch(self, texts: Sequence[str]) -> Sequence[Sequence[int]]:
        """Batch encode a list of texts into tokens.

        Args:
            texts (Sequence[str]): The texts to encode.

        Returns:
            List of encoded sequences

        """
        if self._backend == "chonkie":
            return self.tokenizer.encode_batch(texts)
        elif self._backend == "tiktoken":
            return self.tokenizer.encode_batch(texts)
        elif self._backend == "transformers":
            encoded = self.tokenizer(texts, add_special_tokens=False)
            return encoded["input_ids"]
        elif self._backend == "tokenizers":
            return [encoding.ids for encoding in self.tokenizer.encode_batch(texts)]
        if self._backend == "callable":
            raise NotImplementedError("Batch encoding not implemented for callable tokenizers.")
        raise ValueError(f"Unsupported tokenizer backend: {self._backend}")

    def decode_batch(self, token_sequences: Sequence[Sequence[int]]) -> Sequence[str]:
        """Batch decode a list of tokens back into text.

        Args:
            token_sequences (Sequence[Sequence[int]]): The tokens to decode.

        Returns:
            List of decoded texts

        """
        if self._backend == "chonkie":
            return self.tokenizer.decode_batch(token_sequences)  # type: ignore[union-attr]
        elif self._backend in "tiktoken":
            return self.tokenizer.decode_batch(token_sequences)  # type: ignore[union-attr]
        elif self._backend in "tokenizers":
            return self.tokenizer.decode_batch(token_sequences)  # type: ignore[union-attr]
        elif self._backend == "transformers":
            return self.tokenizer.batch_decode(token_sequences, skip_special_tokens=True)  # type: ignore[union-attr]

        if self._backend == "callable":
            raise NotImplementedError("Batch decoding not implemented for callable tokenizers.")
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
            return self.tokenizer.count_tokens_batch(texts)  # type: ignore[union-attr]
        elif self._backend == "tiktoken":
            return [len(token_list) for token_list in self.tokenizer.encode_batch(texts)]  # type: ignore[union-attr,arg-type]
        elif self._backend == "transformers":
            encoded = self.tokenizer(texts, add_special_tokens=False)  # type: ignore[operator,call-arg,index,arg-type]
            return [len(token_list) for token_list in encoded["input_ids"]]  # type: ignore[index]
        elif self._backend == "tokenizers":
            return [
                len(t.ids) for t in self.tokenizer.encode_batch(texts, add_special_tokens=False)
            ]  # type: ignore[union-attr,call-arg,arg-type,attr-defined]
        elif self._backend == "callable":
            return [self.tokenizer(text) for text in texts]  # type: ignore[operator,misc]

        raise ValueError(f"Tokenizer backend {self._backend} not supported.")
