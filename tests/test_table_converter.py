"""Tests for table converter utilities."""

from chonkie.utils.table_converter import (
    HTMLTableParser,
    html_table_to_json,
    markdown_table_to_json,
)


class TestMarkdownTableToJson:
    """Tests for markdown_table_to_json function."""

    def test_basic(self) -> None:
        """Test basic conversion with numeric type inference."""
        table = """| Name | Age | Score |
|------|-----|-------|
| Alice | 30 | 95.5 |
| Bob | 25 | 100 |"""

        result = markdown_table_to_json(table)

        assert len(result) == 2
        assert result[0] == {"Name": "Alice", "Age": 30, "Score": 95.5}
        assert result[1] == {"Name": "Bob", "Age": 25, "Score": 100}
        assert isinstance(result[0]["Age"], int)
        assert isinstance(result[0]["Score"], float)

    def test_negative_and_scientific_numbers(self) -> None:
        """Test negative numbers and scientific notation."""
        table = """| Value | Change |
|------|--------|
| 100 | -10 |
| 1e10 | 2.5e-3 |"""

        result = markdown_table_to_json(table)

        assert result[0]["Value"] == 100
        assert result[0]["Change"] == -10
        assert result[1]["Value"] == "1e10"

    def test_empty_and_missing_cells(self) -> None:
        """Test handling of empty cells and mismatched columns."""
        table = """| Name | Value | Notes |
|------|-------|-------|
| Alice | 100 | |
| Bob | | Extra |"""

        result = markdown_table_to_json(table)

        assert result[0]["Name"] == "Alice"
        assert result[0]["Value"] == 100
        assert result[1]["Name"] == "Bob"

    def test_special_content(self) -> None:
        """Test special characters, unicode, and whitespace."""
        table = """| City | Symbol |
|------|--------|
| 北京 | @ |
| Paris | $ |"""

        result = markdown_table_to_json(table)

        assert result[0]["City"] == "北京"
        assert result[1]["Symbol"] == "$"

    def test_empty_table(self) -> None:
        """Test empty and header-only tables."""
        assert markdown_table_to_json("") == []
        assert markdown_table_to_json("| Name |\n|------|") == []


class TestHTMLTableToJson:
    """Tests for html_table_to_json function."""

    def test_basic(self) -> None:
        """Test basic HTML conversion with numeric inference."""
        html = """<table>
  <thead>
    <tr><th>Name</th><th>Age</th></tr>
  </thead>
  <tbody>
    <tr><td>Alice</td><td>30</td></tr>
    <tr><td></td><td>100</td></tr>
  </tbody>
</table>"""

        result = html_table_to_json(html)

        assert result is not None
        assert result[0] == {"Name": "Alice", "Age": 30}
        assert result[1]["Name"] == ""

    def test_without_thead(self) -> None:
        """Test HTML table without thead uses first row as headers."""
        html = "<table><tr><th>Name</th><th>Age</th></tr><tr><td>Bob</td><td>25</td></tr></table>"

        result = html_table_to_json(html)

        assert result is not None
        assert result[0] == {"Name": "Bob", "Age": 25}

    def test_no_headers_returns_none(self) -> None:
        """Test HTML table without headers returns None."""
        assert html_table_to_json("<table><tr><td>Alice</td></tr></table>") is None
        assert html_table_to_json("not a table") is None


class TestHTMLTableParser:
    """Tests for HTMLTableParser class."""

    def test_basic(self) -> None:
        """Test basic parsing."""
        parser = HTMLTableParser()
        parser.feed("<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>")

        assert len(parser.tables) == 1
        assert parser.tables[0]["headers"] == ["A", "B"]
        assert parser.tables[0]["rows"] == [["1", "2"]]

    def test_multiple_tables(self) -> None:
        """Test parsing multiple tables."""
        parser = HTMLTableParser()
        parser.feed(
            "<table><tr><th>A</th></tr><tr><td>1</td></tr></table><table><tr><th>B</th></tr><tr><td>2</td></tr></table>"
        )

        assert len(parser.tables) == 2

    def test_nested_tags(self) -> None:
        """Test that nested tags in cells are handled."""
        parser = HTMLTableParser()
        parser.feed("<table><tr><th>Name</th></tr><tr><td>Hello <b>World</b></td></tr></table>")

        assert parser.tables[0]["rows"][0] == ["Hello World"]
