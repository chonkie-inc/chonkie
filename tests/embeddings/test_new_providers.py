"""Test suite for new provider wrapper classes (Mistral, Together, Mixedbread, Nomic, DeepInfra, Cloudflare)."""

import contextlib
import dataclasses
from unittest.mock import patch

import numpy as np
import pytest

from chonkie.embeddings.base import BaseEmbeddings
from chonkie.embeddings.cloudflare import CloudflareEmbeddings
from chonkie.embeddings.deepinfra import DeepInfraEmbeddings
from chonkie.embeddings.mistral import MistralEmbeddings
from chonkie.embeddings.mixedbread import MixedbreadEmbeddings
from chonkie.embeddings.nomic import NomicEmbeddings
from chonkie.embeddings.together import TogetherEmbeddings
from tests.embeddings.utils import make_mock_catsu_client


@dataclasses.dataclass(frozen=True)
class ProviderTestParams:  # noqa: D101
    provider_class: type[BaseEmbeddings]
    default_model: str
    dimension: int
    api_key_env: str
    provider_name: str

    @contextlib.contextmanager
    def patch_catsu_client(self):
        catsu_client = make_mock_catsu_client(
            dimension=self.dimension,
            model_name=self.provider_name,
        )
        with patch("catsu.Client", return_value=catsu_client) as mock_client:
            yield mock_client

    def build_embeddings(self) -> BaseEmbeddings:
        with self.patch_catsu_client():
            return self.provider_class(api_key="test-key")


PROVIDER_TEST_PARAMS = [
    ProviderTestParams(
        MistralEmbeddings,
        "mistral-embed",
        1024,
        "MISTRAL_API_KEY",
        "mistral",
    ),
    ProviderTestParams(
        TogetherEmbeddings,
        "togethercomputer/m2-bert-80M-8k-retrieval",
        768,
        "TOGETHER_API_KEY",
        "together",
    ),
    ProviderTestParams(
        MixedbreadEmbeddings,
        "mxbai-embed-large-v1",
        1024,
        "MIXEDBREAD_API_KEY",
        "mixedbread",
    ),
    ProviderTestParams(
        NomicEmbeddings,
        "nomic-embed-text-v1.5",
        768,
        "NOMIC_API_KEY",
        "nomic",
    ),
    ProviderTestParams(
        DeepInfraEmbeddings,
        "BAAI/bge-large-en-v1.5",
        1024,
        "DEEPINFRA_API_KEY",
        "deepinfra",
    ),
    ProviderTestParams(
        CloudflareEmbeddings,
        "@cf/baai/bge-base-en-v1.5",
        768,
        "CLOUDFLARE_API_KEY",
        "cloudflare",
    ),
]


@pytest.mark.parametrize("ptp", PROVIDER_TEST_PARAMS, ids=lambda ptp: ptp.provider_name)
class TestNewProviderEmbeddings:
    """Parametrized tests for all new provider wrapper classes."""

    def test_initialization(self, ptp: ProviderTestParams) -> None:
        embeddings = ptp.build_embeddings()
        assert embeddings.model == ptp.default_model
        assert embeddings._catsu is not None

    def test_default_model(self, ptp: ProviderTestParams) -> None:
        assert ptp.provider_class.DEFAULT_MODEL == ptp.default_model

    def test_initialization_with_env_var(self, ptp: ProviderTestParams, monkeypatch):
        monkeypatch.setenv(ptp.api_key_env, "env-key")
        with ptp.patch_catsu_client():
            assert ptp.provider_class().model == ptp.default_model

    def test_embed(self, ptp: ProviderTestParams) -> None:
        result = ptp.build_embeddings().embed("test text")
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1

    def test_embed_batch(self, ptp: ProviderTestParams) -> None:
        results = ptp.build_embeddings().embed_batch(["a", "b", "c"])
        assert isinstance(results, list)
        assert len(results) == 3

    def test_embed_batch_empty(self, ptp: ProviderTestParams) -> None:
        assert ptp.build_embeddings().embed_batch([]) == []

    def test_dimension(self, ptp: ProviderTestParams) -> None:
        embeddings = ptp.build_embeddings()
        assert isinstance(embeddings.dimension, int)
        assert embeddings.dimension == ptp.dimension

    def test_get_tokenizer(self, ptp: ProviderTestParams) -> None:
        assert ptp.build_embeddings().get_tokenizer() is not None

    def test_repr(self, ptp: ProviderTestParams) -> None:
        class_name = ptp.provider_class.__name__
        repr_str = repr(ptp.build_embeddings())
        assert class_name in repr_str
        assert ptp.default_model in repr_str

    def test_is_available(self, ptp: ProviderTestParams) -> None:
        assert isinstance(ptp.provider_class._is_available(), bool)  # type: ignore

    def test_missing_catsu(self, ptp: ProviderTestParams) -> None:
        with patch.object(ptp.provider_class, "_is_available", return_value=False):
            with pytest.raises(ImportError, match=r"One \(or more\) of the following packages"):
                ptp.build_embeddings()

    def test_correct_provider(self, ptp: ProviderTestParams) -> None:
        with ptp.patch_catsu_client() as mock_cls:
            ptp.provider_class(api_key="my-key")  # type: ignore
            assert mock_cls.call_args[1].get("api_keys") == {ptp.provider_name: "my-key"}
