"""Test for the DatasetsPorter class."""

import os
import shutil
import tempfile

import pytest

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
        porter.export(sample_chunks, return_ds=False, dataset_path=tmp_dir)
        # Check if dataset files exist in tmp_dir
        assert os.path.exists(tmp_dir)
        assert any(os.listdir(tmp_dir)), "Dataset directory should not be empty."
    finally:
        shutil.rmtree(tmp_dir)


def test_export_and_return_dataset(sample_chunks):  # noqa
    porter = DatasetsPorter()
    ds = porter.export(sample_chunks, return_ds=True)
    assert ds is not None
    assert hasattr(ds, "__len__")
    assert len(ds) == len(sample_chunks)


def test_export_empty_chunks():  # noqa
    porter = DatasetsPorter()
    ds = porter.export([], return_ds=True)
    assert ds is not None
    assert hasattr(ds, "__len__")
    assert len(ds) == 0


def test_dataset_structure_and_content(sample_chunks):  # noqa
    porter = DatasetsPorter()
    ds = porter.export(sample_chunks, return_ds=True)
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
