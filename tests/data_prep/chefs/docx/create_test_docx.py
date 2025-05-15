"""Script to create a test DOCX file with various features."""

import os
from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

def create_test_docx(output_path: str) -> None:
    """Create a test DOCX file with various features.
    
    Args:
        output_path: Path where to save the test document.
    """
    doc = Document()
    
    # Set document properties
    core_props = doc.core_properties
    core_props.title = "Test Document"
    core_props.author = "Test Author"
    core_props.subject = "Test Subject"
    core_props.keywords = "test, document, python-docx"
    core_props.category = "Test"
    core_props.comments = "This is a test document"
    
    # Add header
    header = doc.sections[0].header
    header_para = header.paragraphs[0]
    header_para.text = "Test Document Header"
    header_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Add footer
    footer = doc.sections[0].footer
    footer_para = footer.paragraphs[0]
    footer_para.text = "Page 1"
    footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Add title
    title = doc.add_heading("Test Document", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Add paragraph with different styles
    p1 = doc.add_paragraph()
    p1.add_run("This is a ").bold = True
    p1.add_run("test ").italic = True
    p1.add_run("paragraph with ").underline = True
    p1.add_run("different styles").strike = True
    
    # Add paragraph with custom font and color
    p2 = doc.add_paragraph()
    run = p2.add_run("This text has custom font and color")
    run.font.name = "Arial"
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(255, 0, 0)
    
    # Add a table
    table = doc.add_table(rows=3, cols=3)
    table.style = "Table Grid"
    
    # Add header row
    header_cells = table.rows[0].cells
    header_cells[0].text = "Header 1"
    header_cells[1].text = "Header 2"
    header_cells[2].text = "Header 3"
    
    # Add data rows
    for i in range(1, 3):
        row_cells = table.rows[i].cells
        for j in range(3):
            row_cells[j].text = f"Cell {i},{j}"
    
    # Add a bullet list with proper numbering properties
    doc.add_paragraph("This is a bullet list:")
    for i in range(3):
        p = doc.add_paragraph(f"Item {i+1}", style="List Bullet")
        p.paragraph_format.numbering_level = 0
        p.paragraph_format.numbering_style = "List Bullet"  # Explicitly set numbering style
    
    # Add a numbered list with proper numbering properties
    doc.add_paragraph("This is a numbered list:")
    for i in range(3):
        p = doc.add_paragraph(f"Item {i+1}", style="List Number")
        p.paragraph_format.numbering_level = 0
        p.paragraph_format.numbering_style = "List Number"  # Explicitly set numbering style
    
    # Add a comment
    comment = doc.add_paragraph("This is a commented paragraph")
    comment._p.append(create_comment("This is a test comment", "Test Author"))
    
    # Add a hyperlink using the standard method
    p3 = doc.add_paragraph()
    p3.add_run("This is a ")
    add_hyperlink(p3, "hyperlink", "https://example.com")
    p3.add_run(" to example.com")
    
    # Save the document
    doc.save(output_path)

def create_comment(text: str, author: str) -> OxmlElement:
    """Create a comment element.
    
    Args:
        text: The comment text.
        author: The comment author.
        
    Returns:
        The comment element.
    """
    comment = OxmlElement("w:comment")
    comment.set(qn("w:id"), "1")
    comment.set(qn("w:author"), author)
    comment.set(qn("w:date"), "2024-01-01T00:00:00Z")
    comment.text = text
    return comment

def add_hyperlink(paragraph, text: str, url: str) -> None:
    """Add a hyperlink to a paragraph using python-docx relationships."""
    # This is the official way to add a hyperlink with python-docx
    # Get the document part
    part = paragraph.part
    # Create a new relationship id
    r_id = part.relate_to(url, 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink', is_external=True)
    # Create the w:hyperlink tag and set the relationship id
    hyperlink = OxmlElement('w:hyperlink')
    hyperlink.set(qn('r:id'), r_id)
    # Create a w:r element
    new_run = OxmlElement('w:r')
    # Create a w:rPr element
    rPr = OxmlElement('w:rPr')
    # Add the hyperlink style
    rStyle = OxmlElement('w:rStyle')
    rStyle.set(qn('w:val'), 'Hyperlink')
    rPr.append(rStyle)
    new_run.append(rPr)
    # Create a w:t element
    t = OxmlElement('w:t')
    t.text = text
    new_run.append(t)
    hyperlink.append(new_run)
    # Append the hyperlink to the paragraph
    paragraph._p.append(hyperlink)

if __name__ == "__main__":
    # Create the test document
    output_path = Path(__file__).parent / "sample.docx"
    create_test_docx(str(output_path))
    print(f"Created test document at: {output_path}") 