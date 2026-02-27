"""Test for the DatasetsPorter class."""

from unittest.mock import patch

import pytest
from datasets import Dataset

from chonkie.porters.datasets import DatasetsPorter
from chonkie.types import Chunk


@pytest.fixture
def sample_chunks():  # noqa
    return [
        Chunk(text="Hello world", start_index=0, end_index=11, token_count=2),
        Chunk(text="Another chunk", start_index=12, end_index=25, token_count=2),
    ]


@pytest.mark.parametrize("method", ["export", "__call__"])
def test_export_and_save_to_disk(tmp_path, sample_chunks, method):  # noqa
    porter = DatasetsPorter()
    ds = getattr(porter, method)(sample_chunks, save_to_disk=True, path=tmp_path)
    assert isinstance(ds, Dataset)
    assert any(tmp_path.glob("*")), "Dataset directory should not be empty."
    reloaded_ds = Dataset.load_from_disk(tmp_path)
    assert len(reloaded_ds) == len(sample_chunks)


def test_export_and_return_dataset(sample_chunks):  # noqa
    porter = DatasetsPorter()
    ds = porter.export(sample_chunks, save_to_disk=False)
    assert ds is not None
    assert isinstance(ds, Dataset)
    assert len(ds) == len(sample_chunks)


def test_export_empty_chunks():  # noqa
    porter = DatasetsPorter()
    ds = porter.export([], save_to_disk=False)
    assert ds is not None
    assert isinstance(ds, Dataset)
    assert len(ds) == 0


def test_dataset_structure_and_content(sample_chunks):  # noqa
    porter = DatasetsPorter()
    ds = porter.export(sample_chunks, save_to_disk=False)
    # Check column names - now includes embedding field
    expected_columns = {
        "id",
        "text",
        "start_index",
        "end_index",
        "token_count",
        "context",
        "embedding",
    }
    assert set(ds.column_names) == expected_columns
    # Check content
    for i, chunk in enumerate(sample_chunks):
        row = ds[i]
        assert row["text"] == chunk.text
        assert row["start_index"] == chunk.start_index
        assert row["end_index"] == chunk.end_index
        assert row["token_count"] == chunk.token_count
        assert row["context"] is None


@patch("datasets.Dataset.save_to_disk")
def test_save_to_disk_kwargs(mock_save_to_disk, sample_chunks):  # noqa
    porter = DatasetsPorter()
    porter.export(
        sample_chunks,
        save_to_disk=True,
        path="dummy_path",
        num_shards=2,
        num_proc=4,
    )
    mock_save_to_disk.assert_called_once_with("dummy_path", num_shards=2, num_proc=4)
