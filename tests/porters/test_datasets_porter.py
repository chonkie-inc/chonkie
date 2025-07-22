"""Test for the DatasetsPorter class."""

import os
import shutil
import tempfile
from unittest.mock import patch

import pytest
from datasets import Dataset

from chonkie.friends.porters.datasets import DatasetsPorter
from chonkie.types.base import Chunk


@pytest.fixture
def sample_chunks():  # noqa
    return [
        Chunk(text="Hello world", start_index=0, end_index=11, token_count=2),
        Chunk(text="Another chunk", start_index=12, end_index=25, token_count=2),
    ]


def test_export_and_save_to_disk(sample_chunks):  # noqa
    porter = DatasetsPorter()
    tmp_dir = tempfile.mkdtemp()
    try:
        ds = porter.export(sample_chunks, save_to_disk=True, path=tmp_dir)
        assert ds is not None
        assert isinstance(ds, Dataset)
        # Check if dataset files exist in tmp_dir
        assert os.path.exists(tmp_dir)
        assert any(os.listdir(tmp_dir)), "Dataset directory should not be empty."
        # Check if we can load the dataset
        reloaded_ds = Dataset.load_from_disk(tmp_dir)
        assert len(reloaded_ds) == len(sample_chunks)
    finally:
        shutil.rmtree(tmp_dir)


def test_export_and_return_dataset(sample_chunks):  # noqa
    porter = DatasetsPorter()
    ds = porter.export(sample_chunks, save_to_disk=False)
    assert ds is not None
    assert isinstance(ds, Dataset)
    assert hasattr(ds, "__len__")
    assert len(ds) == len(sample_chunks)


def test_export_empty_chunks():  # noqa
    porter = DatasetsPorter()
    ds = porter.export([], save_to_disk=False)
    assert ds is not None
    assert isinstance(ds, Dataset)
    assert hasattr(ds, "__len__")
    assert len(ds) == 0


def test_dataset_structure_and_content(sample_chunks):  # noqa
    porter = DatasetsPorter()
    ds = porter.export(sample_chunks, save_to_disk=False)
    # Check column names
    expected_columns = {"text", "start_index", "end_index", "token_count", "context"}
    assert set(ds.column_names) == expected_columns
    # Check content
    for i, chunk in enumerate(sample_chunks):
        row = ds[i]
        assert row["text"] == chunk.text
        assert row["start_index"] == chunk.start_index
        assert row["end_index"] == chunk.end_index
        assert row["token_count"] == chunk.token_count
        assert row["context"] is None


def test_call_method(sample_chunks):  # noqa
    porter = DatasetsPorter()
    # Test with save_to_disk=False
    ds = porter(sample_chunks, save_to_disk=False)
    assert ds is not None
    assert isinstance(ds, Dataset)
    assert len(ds) == len(sample_chunks)

    # Test with save_to_disk=True
    tmp_dir = tempfile.mkdtemp()
    try:
        result = porter(sample_chunks, save_to_disk=True, path=tmp_dir)
        assert result is not None
        assert isinstance(result, Dataset)
        assert os.path.exists(tmp_dir)
        assert any(os.listdir(tmp_dir))
    finally:
        shutil.rmtree(tmp_dir)


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
    mock_save_to_disk.assert_called_once_with(
        "dummy_path", num_shards=2, num_proc=4
    )
