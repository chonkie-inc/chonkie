import os
import pytest
from pathlib import Path
from chonkie.chefs.txt.extractor import TXTExtractorChef
from chonkie.chefs.txt.config import TXTExtractorConfig
from chonkie.chefs.base import ChefError

TEST_DIR = Path(__file__).parent
SAMPLE_TXT = TEST_DIR / "sample.txt"

@pytest.fixture(scope="module", autouse=True)
def create_sample_txt():
    SAMPLE_TXT.write_text("This is a test file.\nWith multiple lines.\nEnd.", encoding="utf-8")
    yield
    SAMPLE_TXT.unlink(missing_ok=True)

def test_txt_chef_initialization():
    chef = TXTExtractorChef()
    assert isinstance(chef.config, TXTExtractorConfig)
    assert chef.config.encoding == "utf-8"
    assert chef.config.extract_metadata is True
    chef = TXTExtractorChef(TXTExtractorConfig(encoding="ascii", extract_metadata=False))
    assert chef.config.encoding == "ascii"
    assert chef.config.extract_metadata is False

def test_txt_chef_validation():
    chef = TXTExtractorChef()
    with pytest.raises(ChefError):
        chef.validate("nonexistent.txt")
    with pytest.raises(ChefError):
        chef.validate("test.pdf")
    assert chef.validate(SAMPLE_TXT) is True

def test_txt_chef_prepare():
    chef = TXTExtractorChef()
    result = chef.prepare(SAMPLE_TXT)
    assert isinstance(result, dict)
    assert "text" in result
    assert "This is a test file." in result["text"]
    assert "With multiple lines." in result["text"]
    assert "End." in result["text"]
    assert "metadata" in result
    assert result["metadata"]["filename"] == SAMPLE_TXT.name
    assert result["metadata"]["size"] > 0

def test_txt_chef_prepare_no_metadata():
    chef = TXTExtractorChef(TXTExtractorConfig(extract_metadata=False))
    result = chef.prepare(SAMPLE_TXT)
    assert "text" in result
    assert "metadata" not in result

def test_txt_chef_encoding_error():
    chef = TXTExtractorChef(TXTExtractorConfig(encoding="ascii"))
    # Write a file with a non-ascii character
    bad_txt = TEST_DIR / "bad.txt"
    bad_txt.write_text("Café", encoding="utf-8")
    with pytest.raises(ChefError):
        chef.prepare(bad_txt)
    bad_txt.unlink(missing_ok=True) 