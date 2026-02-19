"""Tests for Markdown types (MarkdownTable, MarkdownCode, MarkdownImage, MarkdownDocument)."""

from chonkie.types import Document
from chonkie.types.markdown import MarkdownCode, MarkdownDocument, MarkdownImage, MarkdownTable


# ---------------------------------------------------------------------------
# MarkdownTable tests
# ---------------------------------------------------------------------------


def test_markdown_table_stores_content():
    """MarkdownTable stores the table content string."""
    table = MarkdownTable(content="| a | b |\n|---|---|\n| 1 | 2 |", start_index=0, end_index=30)
    assert "| a | b |" in table.content


def test_markdown_table_stores_indices():
    """MarkdownTable stores start and end indices."""
    table = MarkdownTable(content="| x |", start_index=5, end_index=10)
    assert table.start_index == 5
    assert table.end_index == 10


def test_markdown_table_default_values():
    """MarkdownTable defaults to empty content and zero indices."""
    table = MarkdownTable()
    assert table.content == ""
    assert table.start_index == 0
    assert table.end_index == 0


# ---------------------------------------------------------------------------
# MarkdownCode tests
# ---------------------------------------------------------------------------


def test_markdown_code_stores_content():
    """MarkdownCode stores the code content."""
    code = MarkdownCode(content="x = 1", start_index=0, end_index=5)
    assert code.content == "x = 1"


def test_markdown_code_with_language():
    """MarkdownCode stores an optional language string."""
    code = MarkdownCode(content="fn main() {}", language="rust", start_index=0, end_index=12)
    assert code.language == "rust"


def test_markdown_code_without_language():
    """MarkdownCode language defaults to None."""
    code = MarkdownCode(content="x = 1", start_index=0, end_index=5)
    assert code.language is None


def test_markdown_code_stores_indices():
    """MarkdownCode stores start and end indices."""
    code = MarkdownCode(content="...", start_index=20, end_index=30)
    assert code.start_index == 20
    assert code.end_index == 30


def test_markdown_code_default_values():
    """MarkdownCode defaults to empty content, None language, zero indices."""
    code = MarkdownCode()
    assert code.content == ""
    assert code.language is None
    assert code.start_index == 0
    assert code.end_index == 0


# ---------------------------------------------------------------------------
# MarkdownImage tests
# ---------------------------------------------------------------------------


def test_markdown_image_stores_alias_and_content():
    """MarkdownImage stores alias and base64 content."""
    img = MarkdownImage(alias="logo", content="abc123", start_index=0, end_index=20)
    assert img.alias == "logo"
    assert img.content == "abc123"


def test_markdown_image_with_link():
    """MarkdownImage stores an optional URL link."""
    img = MarkdownImage(
        alias="logo", content="abc", start_index=0, end_index=10, link="https://example.com"
    )
    assert img.link == "https://example.com"


def test_markdown_image_without_link():
    """MarkdownImage link defaults to None."""
    img = MarkdownImage(alias="logo", content="abc", start_index=0, end_index=10)
    assert img.link is None


def test_markdown_image_stores_indices():
    """MarkdownImage stores start and end indices."""
    img = MarkdownImage(alias="a", content="b", start_index=7, end_index=42)
    assert img.start_index == 7
    assert img.end_index == 42


def test_markdown_image_default_values():
    """MarkdownImage defaults to empty strings, None link, and zero indices."""
    img = MarkdownImage()
    assert img.alias == ""
    assert img.content == ""
    assert img.link is None
    assert img.start_index == 0
    assert img.end_index == 0


# ---------------------------------------------------------------------------
# MarkdownDocument tests
# ---------------------------------------------------------------------------


def test_markdown_document_is_subclass_of_document():
    """MarkdownDocument is a subclass of Document."""
    assert issubclass(MarkdownDocument, Document)


def test_markdown_document_default_lists_are_empty():
    """MarkdownDocument defaults to empty lists for tables, code, and images."""
    doc = MarkdownDocument()
    assert doc.tables == []
    assert doc.code == []
    assert doc.images == []


def test_markdown_document_inherits_document_fields():
    """MarkdownDocument inherits id, content, chunks, and metadata from Document."""
    doc = MarkdownDocument(content="# Hello")
    assert doc.content == "# Hello"
    assert doc.id.startswith("doc_")
    assert doc.chunks == []
    assert doc.metadata == {}


def test_markdown_document_stores_tables():
    """MarkdownDocument stores MarkdownTable objects."""
    tables = [MarkdownTable(content="| a |", start_index=0, end_index=5)]
    doc = MarkdownDocument(content="| a |", tables=tables)
    assert len(doc.tables) == 1
    assert doc.tables[0].content == "| a |"


def test_markdown_document_stores_code_blocks():
    """MarkdownDocument stores MarkdownCode objects."""
    code_blocks = [MarkdownCode(content="x = 1", language="python", start_index=0, end_index=5)]
    doc = MarkdownDocument(content="```python\nx = 1\n```", code=code_blocks)
    assert len(doc.code) == 1
    assert doc.code[0].language == "python"


def test_markdown_document_stores_images():
    """MarkdownDocument stores MarkdownImage objects."""
    images = [MarkdownImage(alias="img", content="b64data", start_index=0, end_index=15)]
    doc = MarkdownDocument(content="![img]()", images=images)
    assert len(doc.images) == 1
    assert doc.images[0].alias == "img"


def test_markdown_document_lists_are_mutable():
    """MarkdownDocument allows appending to tables, code, and images after creation."""
    doc = MarkdownDocument()
    doc.tables.append(MarkdownTable(content="| x |", start_index=0, end_index=5))
    doc.code.append(MarkdownCode(content="y = 2", start_index=6, end_index=11))
    doc.images.append(MarkdownImage(alias="pic", content="data", start_index=12, end_index=20))
    assert len(doc.tables) == 1
    assert len(doc.code) == 1
    assert len(doc.images) == 1


def test_markdown_document_default_lists_are_independent():
    """Two MarkdownDocuments created with defaults have independent lists."""
    doc1 = MarkdownDocument()
    doc2 = MarkdownDocument()
    doc1.tables.append(MarkdownTable(content="t", start_index=0, end_index=1))
    assert doc2.tables == []
