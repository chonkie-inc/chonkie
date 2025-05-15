#!/usr/bin/env python
"""Script to generate enhanced PDF sample for testing.

This script creates a PDF file with various elements:
- Text content
- Tables
- Form fields
- Vector graphics

The PDF is saved to tests/data_prep/chefs/pdf/enhanced_sample.pdf
"""

import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    Image, PageBreak, Frame, BaseDocTemplate, PageTemplate
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfform
from reportlab.lib.units import inch

def create_enhanced_sample_pdf(output_path):
    """Create a sample PDF with various elements for testing the enhanced PDF extractor."""
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create the PDF document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        title="Enhanced PDF Sample",
        author="Chonkie Test Suite",
        subject="Test PDF with tables, forms, and more"
    )
    
    # Styles for document elements
    styles = getSampleStyleSheet()
    
    # Content elements
    elements = []
    
    # Add title
    elements.append(Paragraph("Enhanced PDF Sample for Testing", styles["Title"]))
    elements.append(Spacer(1, 0.25*inch))
    
    # Add introduction text
    elements.append(Paragraph(
        """This PDF document contains various elements to test the enhanced PDF extraction 
        features in Chonkie. It includes tables, form fields, and vector graphics.""",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 0.25*inch))
    
    # Add section for tables
    elements.append(Paragraph("1. Tables for Testing", styles["Heading2"]))
    elements.append(Spacer(1, 0.15*inch))
    elements.append(Paragraph(
        """The table below contains sample data for testing table extraction capabilities.""",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 0.15*inch))
    
    # Create a simple table
    data = [
        ['Product', 'Price', 'Quantity', 'Total'],
        ['Widget A', '$10.00', '5', '$50.00'],
        ['Widget B', '$15.00', '3', '$45.00'],
        ['Widget C', '$20.00', '2', '$40.00'],
        ['', '', 'Total', '$135.00'],
    ]
    table = Table(data, colWidths=[2*inch, inch, inch, inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (3, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (3, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.25*inch))
    
    # Add another table with different structure
    elements.append(Paragraph("Another table with different structure:", styles["Normal"]))
    elements.append(Spacer(1, 0.15*inch))
    
    # Create a complex table with merged cells
    data = [
        ['Category', 'Q1', 'Q2', 'Q3', 'Q4'],
        ['Region 1', '100', '120', '130', '150'],
        ['Region 2', '90', '115', '140', '155'],
        ['Region 3', '105', '110', '125', '145'],
    ]
    table2 = Table(data, colWidths=[1.5*inch, inch, inch, inch, inch])
    table2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.blue),
        ('BACKGROUND', (0, 0), (0, -1), colors.blue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(table2)
    elements.append(Spacer(1, 0.5*inch))
    
    # Add a section break
    elements.append(PageBreak())
    
    # Add section for vector graphics and form fields
    elements.append(Paragraph("2. Form Fields and Vector Graphics", styles["Heading2"]))
    elements.append(Spacer(1, 0.25*inch))
    elements.append(Paragraph(
        """The form fields below can be used to test form field extraction. 
        Vector graphics will be added directly to the canvas.""",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 0.25*inch))
    
    # Add a note about annotations
    elements.append(Paragraph(
        """Note: PDF annotations cannot be added with reportlab directly. 
        For a complete test PDF with annotations, use PyMuPDF (fitz) to add 
        annotations to this PDF after creation.""",
        styles["Italic"]
    ))
    elements.append(Spacer(1, 0.25*inch))
    
    # Build the document with the elements we've added so far
    doc.build(elements)
    
    # Now add form fields and vector graphics using direct canvas operations
    add_form_fields_and_graphics(output_path)
    
    print(f"Enhanced PDF sample created at: {output_path}")

def add_form_fields_and_graphics(pdf_path):
    """Add form fields and vector graphics to an existing PDF."""
    
    # Create a temporary file path
    temp_path = pdf_path + ".temp"
    
    # Open the existing PDF and add form fields
    c = canvas.Canvas(temp_path, pagesize=letter)
    
    # Set up to add form fields to the second page
    c.setFont("Helvetica", 12)
    c.showPage()  # Move to page 2
    
    # Add form fields
    form = c.acroForm
    
    # Text field
    c.drawString(1*inch, 9*inch, "Name:")
    form.textfield(
        name="name_field",
        tooltip="Enter your name",
        x=2*inch,
        y=9*inch,
        width=4*inch,
        height=0.4*inch,
        value="John Doe",
        borderStyle="solid",
        borderWidth=1,
        borderColor=colors.black,
        fillColor=colors.white,
        fontSize=12,
    )
    
    # Checkbox
    c.drawString(1*inch, 8*inch, "Agree to terms:")
    form.checkbox(
        name="agree_checkbox",
        tooltip="Check to agree",
        x=2.5*inch,
        y=8*inch,
        buttonStyle="check",
        borderStyle="solid",
        borderWidth=1,
        borderColor=colors.black,
        fillColor=colors.white,
        checked=True,
    )
    
    # Radio buttons
    c.drawString(1*inch, 7*inch, "Choose an option:")
    form.radio(
        name="radio_group",
        tooltip="Option 1",
        x=2*inch,
        y=7*inch,
        buttonStyle="circle",
        borderStyle="solid",
        borderWidth=1,
        borderColor=colors.black,
        fillColor=colors.white,
        value="option1",
        selected=True,
    )
    c.drawString(2.2*inch, 7*inch, "Option 1")
    
    form.radio(
        name="radio_group",
        tooltip="Option 2",
        x=3.5*inch,
        y=7*inch,
        buttonStyle="circle",
        borderStyle="solid",
        borderWidth=1,
        borderColor=colors.black,
        fillColor=colors.white,
        value="option2",
        selected=False,
    )
    c.drawString(3.7*inch, 7*inch, "Option 2")
    
    # Add a vector graphic
    c.drawString(1*inch, 6*inch, "Vector Graphics Examples:")
    
    # Draw a rectangle
    c.setStrokeColor(colors.blue)
    c.setFillColor(colors.lightblue)
    c.rect(1*inch, 5*inch, 2*inch, 1*inch, fill=True)
    
    # Draw a circle
    c.setStrokeColor(colors.red)
    c.setFillColor(colors.pink)
    c.circle(5*inch, 5.5*inch, 0.5*inch, fill=True)
    
    # Draw lines
    c.setStrokeColor(colors.green)
    c.setLineWidth(2)
    c.line(1*inch, 4*inch, 3*inch, 4.5*inch)
    c.line(3*inch, 4.5*inch, 5*inch, 4*inch)
    
    # Draw a bezier curve
    c.setStrokeColor(colors.purple)
    c.bezier(1*inch, 3*inch, 2*inch, 3.5*inch, 4*inch, 2.5*inch, 5*inch, 3*inch)
    
    # Add a polygon
    c.setStrokeColor(colors.black)
    c.setFillColor(colors.yellow)
    p = c.beginPath()
    p.moveTo(2*inch, 2*inch)
    p.lineTo(3*inch, 2.5*inch)
    p.lineTo(3.5*inch, 1.5*inch)
    p.lineTo(2.5*inch, 1*inch)
    p.close()
    c.drawPath(p, fill=True)
    
    # We can't add annotations with reportlab easily
    # This would need to be done with PyMuPDF or similar library
    c.drawString(4*inch, 2*inch, "Annotation would go here")
    
    # Save the enhanced PDF
    c.save()
    
    # Replace the original file with our enhanced version
    import os
    import shutil
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    shutil.move(temp_path, pdf_path)

if __name__ == "__main__":
    output_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "enhanced_sample.pdf"
    )
    create_enhanced_sample_pdf(output_file) 