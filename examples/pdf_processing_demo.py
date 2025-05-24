"""Demo script for PDF processing with Chonkie Chefs.

This script demonstrates how to use the PDF processing functionality
provided by Chonkie Chefs.
"""

import argparse
from pathlib import Path
from typing import Optional

from chonkie.chefs.pdf import PDFExtractorChef, PDFExtractorConfig
from chonkie.chefs import registry

def process_pdf(
    pdf_path: str,
    output_path: Optional[str] = None,
    extract_metadata: bool = True,
    extract_images: bool = False,
    image_output_dir: Optional[str] = None,
    image_format: str = "png",
    image_quality: int = 85,
    page_range: Optional[tuple[int, int]] = None
) -> None:
    """Process a PDF file using Chonkie Chefs.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Optional path to save the processed text
        extract_metadata: Whether to extract PDF metadata
        extract_images: Whether to extract images from the PDF
        image_output_dir: Directory to save extracted images
        image_format: Format to save images in (png/jpeg)
        image_quality: Image quality for JPEG (1-100)
        page_range: Optional range of pages to process
    """
    # Create PDF extractor configuration
    config = PDFExtractorConfig(
        name="pdf_demo",
        extract_metadata=extract_metadata,
        extract_images=extract_images,
        image_output_dir=image_output_dir,
        image_format=image_format,
        image_quality=image_quality,
        page_range=page_range
    )
    
    # Create and register the PDF extractor chef
    chef = PDFExtractorChef(config)
    registry.register("pdf_demo", PDFExtractorChef)
    
    try:
        # Process the PDF
        print(f"Processing PDF: {pdf_path}")
        result = chef(pdf_path)
        
        # Print processing results
        print("\nProcessing Results:")
        print(f"Number of pages: {result['num_pages']}")
        print(f"Processed pages: {result['processed_pages']}")
        
        if extract_metadata and "metadata" in result:
            print("\nPDF Metadata:")
            for key, value in result["metadata"].items():
                print(f"{key}: {value}")
        
        if extract_images and "images" in result:
            print("\nExtracted Images:")
            for i, image in enumerate(result["images"]):
                print(f"\nImage {i + 1}:")
                if "path" in image:
                    print(f"Saved to: {image['path']}")
                    print(f"Size: {image['size']} bytes")
                print(f"Format: {image['format']}")
                print(f"Dimensions: {image['dimensions']}")
        
        # Save processed text if output path is provided
        if output_path:
            output_file = Path(output_path)
            output_file.write_text(result["text"])
            print(f"\nProcessed text saved to: {output_path}")
        
        # Print a sample of the processed text
        print("\nSample of processed text:")
        sample_text = result["text"][:500] + "..." if len(result["text"]) > 500 else result["text"]
        print(sample_text)
        
    finally:
        # Clean up
        registry.unregister("pdf_demo")

def main():
    """Main entry point for the demo script."""
    parser = argparse.ArgumentParser(description="Process PDF files with Chonkie Chefs")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output", "-o", help="Path to save the processed text")
    parser.add_argument("--no-metadata", action="store_true", help="Skip metadata extraction")
    parser.add_argument("--extract-images", action="store_true", help="Extract images from PDF")
    parser.add_argument("--image-dir", help="Directory to save extracted images")
    parser.add_argument("--image-format", choices=["png", "jpeg"], default="png",
                      help="Format to save images in (default: png)")
    parser.add_argument("--image-quality", type=int, default=85,
                      help="Image quality for JPEG (1-100, default: 85)")
    parser.add_argument("--pages", help="Page range to process (e.g., '0-5')")
    
    args = parser.parse_args()
    
    # Parse page range if provided
    page_range = None
    if args.pages:
        try:
            start, end = map(int, args.pages.split("-"))
            page_range = (start, end)
        except ValueError:
            parser.error("Page range must be in format 'start-end'")
    
    process_pdf(
        args.pdf_path,
        args.output,
        not args.no_metadata,
        args.extract_images,
        args.image_dir,
        args.image_format,
        args.image_quality,
        page_range
    )

if __name__ == "__main__":
    main() 