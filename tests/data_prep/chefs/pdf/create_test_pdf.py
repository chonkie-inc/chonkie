"""Create a test PDF file with text and images for testing."""

from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from PIL import Image
import io
import os

def create_test_pdf(output_path: str = "sample.pdf"):
    """Create a test PDF with text and images.
    
    Args:
        output_path: Path where to save the PDF
    """
    # Create a PDF with ReportLab
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    
    # Add some text to page 1
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 100, "This is a test PDF file (Page 1)")
    c.drawString(100, height - 120, "It contains both text and images")
    
    # Create a simple image and save it temporarily
    img = Image.new('RGB', (200, 200), color='red')
    temp_img_path = "temp_test_image.png"
    img.save(temp_img_path)
    
    try:
        # Add the image to page 1
        c.drawImage(temp_img_path, 100, height - 350, width=200, height=200)
        c.setFont("Helvetica", 10)
        c.drawString(100, height - 400, "This is text below the image on page 1")
        c.showPage()
        
        # Add content to page 2
        c.setFont("Helvetica", 12)
        c.drawString(100, height - 100, "This is page 2 of the test PDF file")
        c.drawString(100, height - 120, "It also contains text and an image")
        # Create a different image for page 2
        img2 = Image.new('RGB', (200, 200), color='blue')
        temp_img_path2 = "temp_test_image2.png"
        img2.save(temp_img_path2)
        c.drawImage(temp_img_path2, 100, height - 350, width=200, height=200)
        c.setFont("Helvetica", 10)
        c.drawString(100, height - 400, "This is text below the image on page 2")
        # Save the PDF
        c.save()
    finally:
        # Clean up temporary image files
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        if os.path.exists('temp_test_image2.png'):
            os.remove('temp_test_image2.png')

if __name__ == "__main__":
    # Create the test PDF in the same directory as this script
    script_dir = Path(__file__).parent
    output_path = script_dir / "sample.pdf"
    create_test_pdf(str(output_path))
    print(f"Created test PDF at: {output_path}") 