"""Tests for the DOCX extractor chef."""

import os
import pytest
from pathlib import Path
from typing import Dict, Any

from chonkie.chefs.docx import DOCXExtractorChef, DOCXExtractorConfig

# Path to a test DOCX file
TEST_DOCX_PATH = "tests/data_prep/chefs/docx/sample.docx"

@pytest.fixture
def sample_docx_file():
    """Provide a valid DOCX file for testing, or skip if not available."""
    try:
        import docx
        docx_path = Path(TEST_DOCX_PATH)
        if not docx_path.exists():
            pytest.skip("No valid DOCX file available for testing.")
        # Try to open with python-docx to ensure it's valid
        docx.Document(docx_path)
        return docx_path
    except Exception:
        pytest.skip("No valid DOCX file available for testing.")

@pytest.fixture
def docx_chef():
    """Create a DOCX extractor chef instance."""
    return DOCXExtractorChef()

@pytest.fixture
def temp_image_dir(tmp_path):
    """Create a temporary directory for image output."""
    return tmp_path / "images"

def test_docx_chef_initialization():
    """Test DOCX chef initialization."""
    chef = DOCXExtractorChef()
    assert chef.config.extract_metadata is True
    assert chef.config.extract_images is True
    assert chef.config.image_format == "png"
    assert chef.config.image_quality == 85
    
    config = DOCXExtractorConfig(
        extract_metadata=False,
        extract_images=True,
        image_format="jpeg",
        image_quality=90,
        extract_tables=True,
        table_format="markdown",
        extract_headers_footers=True,
        extract_comments=True,
        extract_styles=True
    )
    chef = DOCXExtractorChef(config)
    assert chef.config.extract_metadata is False
    assert chef.config.extract_images is True
    assert chef.config.image_format == "jpeg"
    assert chef.config.image_quality == 90
    assert chef.config.extract_tables is True
    assert chef.config.table_format == "markdown"
    assert chef.config.extract_headers_footers is True
    assert chef.config.extract_comments is True
    assert chef.config.extract_styles is True

def test_docx_chef_validation(sample_docx_file, docx_chef):
    """Test DOCX validation."""
    assert docx_chef.validate(sample_docx_file) is True

def test_docx_chef_prepare(sample_docx_file, docx_chef):
    """Test DOCX preparation."""
    result = docx_chef.prepare(sample_docx_file)
    assert isinstance(result, dict)
    assert "text" in result
    assert "tables" in result
    assert "images" in result
    assert "metadata" in result

def test_docx_chef_table_extraction(sample_docx_file):
    """Test table extraction with different formats."""
    # Test markdown format
    config = DOCXExtractorConfig(
        extract_tables=True,
        table_format="markdown"
    )
    chef = DOCXExtractorChef(config)
    result = chef.prepare(sample_docx_file)
    for table in result["tables"]:
        assert "markdown" in table
        assert isinstance(table["markdown"], str)
        assert "|" in table["markdown"]
    
    # Test HTML format
    config.table_format = "html"
    chef = DOCXExtractorChef(config)
    result = chef.prepare(sample_docx_file)
    for table in result["tables"]:
        assert "html" in table
        assert isinstance(table["html"], str)
        assert "<table>" in table["html"]
    
    # Test JSON format
    config.table_format = "json"
    chef = DOCXExtractorChef(config)
    result = chef.prepare(sample_docx_file)
    for table in result["tables"]:
        assert "rows" in table
        assert isinstance(table["rows"], list)

def test_docx_chef_style_extraction(sample_docx_file):
    """Test style extraction."""
    config = DOCXExtractorConfig(
        extract_styles=True,
        extract_paragraph_styles=True,
        extract_character_styles=True
    )
    chef = DOCXExtractorChef(config)
    result = chef.prepare(sample_docx_file)
    
    assert "styles" in result
    assert "paragraph_styles" in result["styles"]
    assert "character_styles" in result["styles"]
    
    # Check paragraph styles
    for style_name, style_data in result["styles"]["paragraph_styles"].items():
        assert isinstance(style_name, str)
        assert isinstance(style_data, dict)
        assert all(key in style_data for key in ["font", "size", "bold", "italic", "alignment"])
    
    # Check character styles
    for style_name, style_data in result["styles"]["character_styles"].items():
        assert isinstance(style_name, str)
        assert isinstance(style_data, dict)
        assert all(key in style_data for key in ["font", "size", "bold", "italic"])

def test_docx_chef_headers_footers(sample_docx_file):
    """Test headers and footers extraction."""
    config = DOCXExtractorConfig(
        extract_headers_footers=True,
        extract_styles=True
    )
    chef = DOCXExtractorChef(config)
    result = chef.prepare(sample_docx_file)
    
    assert "headers_footers" in result
    assert "headers" in result["headers_footers"]
    assert "footers" in result["headers_footers"]
    assert isinstance(result["headers_footers"]["headers"], list)
    assert isinstance(result["headers_footers"]["footers"], list)

def test_docx_chef_comments(sample_docx_file):
    """Test comments extraction."""
    config = DOCXExtractorConfig(
        extract_comments=True
    )
    chef = DOCXExtractorChef(config)
    result = chef.prepare(sample_docx_file)
    
    assert "comments" in result
    assert isinstance(result["comments"], list)
    for comment in result["comments"]:
        assert all(key in comment for key in ["id", "author", "date", "text"])

def test_docx_chef_image_extraction(sample_docx_file, temp_image_dir):
    """Test image extraction."""
    config = DOCXExtractorConfig(
        extract_images=True,
        image_output_dir=str(temp_image_dir),
        image_format="png",
        extract_image_metadata=True
    )
    chef = DOCXExtractorChef(config)
    result = chef.prepare(sample_docx_file)
    
    assert "images" in result
    for image in result["images"]:
        assert all(key in image for key in ["format", "mode", "size", "width", "height"])
        if "path" in image:
            assert os.path.exists(image["path"])
            assert image["format"] == "png"

def test_docx_chef_image_extraction_jpeg(sample_docx_file, temp_image_dir):
    """Test image extraction with JPEG format."""
    config = DOCXExtractorConfig(
        extract_images=True,
        image_output_dir=str(temp_image_dir),
        image_format="jpeg",
        image_quality=90,
        extract_image_metadata=True
    )
    chef = DOCXExtractorChef(config)
    result = chef.prepare(sample_docx_file)
    
    assert "images" in result
    for image in result["images"]:
        assert all(key in image for key in ["format", "mode", "size", "width", "height"])
        if "path" in image:
            assert os.path.exists(image["path"])
            assert image["format"] == "jpeg"

def test_docx_chef_image_extraction_memory(sample_docx_file):
    """Test image extraction without saving to disk."""
    config = DOCXExtractorConfig(
        extract_images=True,
        image_format="png",
        extract_image_metadata=True
    )
    chef = DOCXExtractorChef(config)
    result = chef.prepare(sample_docx_file)
    
    assert "images" in result
    for image in result["images"]:
        assert all(key in image for key in ["format", "mode", "size", "width", "height"])
        assert "data" in image
        assert isinstance(image["data"], bytes)

def test_docx_chef_cleanup(sample_docx_file, temp_image_dir):
    """Test cleanup of temporary files."""
    config = DOCXExtractorConfig(
        extract_images=True,
        image_output_dir=str(temp_image_dir)
    )
    chef = DOCXExtractorChef(config)
    result = chef.prepare(sample_docx_file)
    
    # Get list of temporary files
    temp_files = [image["path"] for image in result["images"] if "path" in image]
    
    # Clean up
    chef.clean()
    
    # Verify files are removed
    for temp_file in temp_files:
        assert not os.path.exists(temp_file)

def test_docx_chef_list_extraction(sample_docx_file):
    """Test list structure extraction."""
    config = DOCXExtractorConfig(
        extract_list_structure=True,
        extract_styles=True
    )
    chef = DOCXExtractorChef(config)
    result = chef.prepare(sample_docx_file)
    
    # Find paragraphs with list data
    list_paragraphs = [p for p in result["text"] if "list" in p]
    assert len(list_paragraphs) > 0
    
    for para in list_paragraphs:
        assert "list" in para
        list_data = para["list"]
        assert "level" in list_data
        assert "num_id" in list_data
        assert "list_type" in list_data
        assert list_data["list_type"] in ["bullet", "number"]
        assert "format" in list_data
        assert "start" in list_data

def test_docx_chef_hyperlink_extraction(sample_docx_file):
    """Test hyperlink extraction."""
    config = DOCXExtractorConfig(
        extract_hyperlinks=True,
        extract_styles=True
    )
    chef = DOCXExtractorChef(config)
    result = chef.prepare(sample_docx_file)
    
    # Find paragraphs with hyperlinks
    hyperlink_paragraphs = [p for p in result["text"] if "hyperlinks" in p]
    assert len(hyperlink_paragraphs) > 0
    
    for para in hyperlink_paragraphs:
        assert "hyperlinks" in para
        for link in para["hyperlinks"]:
            assert "text" in link
            assert "url" in link
            assert "start" in link
            assert "end" in link
            assert link["start"] < link["end"]

def test_docx_chef_section_properties(sample_docx_file):
    """Test section properties extraction."""
    config = DOCXExtractorConfig(
        extract_section_properties=True
    )
    chef = DOCXExtractorChef(config)
    result = chef.prepare(sample_docx_file)
    
    assert "sections" in result
    assert len(result["sections"]) > 0
    
    for section in result["sections"]:
        assert "page_width" in section
        assert "page_height" in section
        assert "left_margin" in section
        assert "right_margin" in section
        assert "top_margin" in section
        assert "bottom_margin" in section
        assert "header_distance" in section
        assert "footer_distance" in section
        assert "gutter" in section
        assert "orientation" in section
        assert section["orientation"] in ["portrait", "landscape"]
        assert "columns" in section
        assert isinstance(section["columns"], int)

def test_docx_chef_bookmarks(sample_docx_file):
    """Test bookmark extraction."""
    config = DOCXExtractorConfig(
        extract_bookmarks=True
    )
    chef = DOCXExtractorChef(config)
    result = chef.prepare(sample_docx_file)
    
    assert "bookmarks" in result
    assert isinstance(result["bookmarks"], list)
    
    for bookmark in result["bookmarks"]:
        assert "name" in bookmark
        assert "id" in bookmark
        assert "col_first" in bookmark
        assert "col_last" in bookmark
        assert "has_end" in bookmark

def test_docx_chef_footnotes_endnotes(sample_docx_file):
    """Test footnotes and endnotes extraction."""
    config = DOCXExtractorConfig(
        extract_footnotes_endnotes=True,
        extract_styles=True
    )
    chef = DOCXExtractorChef(config)
    result = chef.prepare(sample_docx_file)
    
    # Verify that footnotes_endnotes key exists
    assert "footnotes_endnotes" in result
    assert "footnotes" in result["footnotes_endnotes"]
    assert "endnotes" in result["footnotes_endnotes"]
    
    # Note: The sample.docx may not contain footnotes or endnotes,
    # so we just check for the structure, not the content

def test_docx_chef_equations(sample_docx_file):
    """Test mathematical equations extraction."""
    config = DOCXExtractorConfig(
        extract_equations=True,
        extract_styles=True
    )
    chef = DOCXExtractorChef(config)
    result = chef.prepare(sample_docx_file)
    
    # Verify that equations key exists
    assert "equations" in result
    assert isinstance(result["equations"], list)
    
    # Note: The sample.docx may not contain equations,
    # so we just check for the structure, not the content

def test_docx_chef_form_fields(sample_docx_file):
    """Test form fields extraction."""
    config = DOCXExtractorConfig(
        extract_form_fields=True,
        extract_styles=True
    )
    chef = DOCXExtractorChef(config)
    result = chef.prepare(sample_docx_file)
    
    # Verify that form_fields key exists
    assert "form_fields" in result
    assert isinstance(result["form_fields"], list)
    
    # Note: The sample.docx may not contain form fields,
    # so we just check for the structure, not the content 