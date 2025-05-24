"""PDF Extractor Chef for Chonkie.

This module provides a chef for extracting text and images from PDF documents.
"""

import io
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

from pydantic import Field

from ..base import BaseChef, ChefConfig, ChefError

logger = logging.getLogger(__name__)

class PDFExtractorConfig(ChefConfig):
    """Configuration for the PDF Extractor Chef."""
    extract_metadata: bool = Field(True, description="Whether to extract PDF metadata")
    extract_images: bool = Field(False, description="Whether to extract images from PDF")
    image_output_dir: Optional[str] = Field(None, description="Directory to save extracted images")
    image_format: str = Field("png", description="Format to save images in (png/jpeg)")
    image_quality: int = Field(85, description="Image quality for JPEG (1-100)")
    password: Optional[str] = Field(None, description="Password for encrypted PDFs")
    page_range: Optional[tuple[int, int]] = Field(None, description="Range of pages to extract")
    
    # New configuration options
    extract_tables: bool = Field(False, description="Whether to extract tables from PDF")
    table_format: str = Field("markdown", description="Format for extracted tables (markdown, html, csv)")
    extract_form_fields: bool = Field(False, description="Whether to extract form fields from PDF")
    extract_annotations: bool = Field(False, description="Whether to extract annotations and comments from PDF")
    extract_vector_graphics: bool = Field(False, description="Whether to extract vector graphics from PDF")
    
    # OCR configuration
    use_ocr: bool = Field(False, description="Whether to use OCR for text extraction when needed")
    ocr_language: str = Field("eng", description="Language for OCR (ISO 639-2 code)")
    ocr_dpi: int = Field(300, description="DPI to use for OCR processing")
    ocr_only_if_needed: bool = Field(True, description="Only use OCR if no text is extracted by normal means")

class PDFExtractorChef(BaseChef[Union[str, Path, bytes], Dict[str, Any]]):
    """Chef for extracting text and images from PDF documents.
    
    This chef handles the extraction of text, images, and metadata from PDF documents,
    preparing them for further processing by other chefs or chunkers.
    """
    
    def __init__(self, config: Optional[PDFExtractorConfig] = None):
        """Initialize the PDF Extractor Chef.
        
        Args:
            config: Optional configuration for the chef
        """
        super().__init__(config or PDFExtractorConfig(name="pdf_extractor"))
        self._load_dependencies()
    
    def _load_dependencies(self) -> None:
        """Load required PDF processing dependencies."""
        try:
            import PyPDF2
            from PIL import Image
            self.PyPDF2 = PyPDF2
            self.Image = Image
            
            # Load optional dependencies based on configuration
            if self.config.extract_tables:
                try:
                    import tabula
                    import pandas as pd
                    self.tabula = tabula
                    self.pd = pd
                except ImportError:
                    logger.warning("tabula-py is required for table extraction. "
                                  "Install it with: pip install tabula-py pandas")
                    self.tabula = None
            
            if self.config.extract_form_fields:
                # PyPDF2 already handles basic form fields
                pass
            
            if self.config.use_ocr:
                try:
                    import pytesseract
                    import cv2
                    self.pytesseract = pytesseract
                    self.cv2 = cv2
                except ImportError:
                    logger.warning("pytesseract and opencv-python are required for OCR. "
                                  "Install them with: pip install pytesseract opencv-python")
                    self.pytesseract = None
                    
            if self.config.extract_vector_graphics:
                try:
                    import fitz  # PyMuPDF
                    self.fitz = fitz
                except ImportError:
                    logger.warning("PyMuPDF is required for vector graphics extraction. "
                                  "Install it with: pip install PyMuPDF")
                    self.fitz = None
                    
        except ImportError:
            raise ChefError(
                "PyPDF2 and Pillow are required for PDF extraction. "
                "Install them with: pip install chonkie[pdf]"
            )
    
    def validate(self, data: Union[str, Path, bytes]) -> bool:
        """Validate the input PDF data.
        
        Args:
            data: The PDF data to validate (file path or bytes)
            
        Returns:
            True if the data is a valid PDF, False otherwise
        """
        try:
            if isinstance(data, (str, Path)):
                with open(data, 'rb') as f:
                    self.PyPDF2.PdfReader(f)
            else:
                self.PyPDF2.PdfReader(io.BytesIO(data))
            return True
        except Exception:
            return False
    
    def _extract_images_from_page(self, page: Any, page_num: int) -> List[Dict[str, Any]]:
        """Extract images from a PDF page.
        
        Args:
            page: The PDF page object
            page_num: The page number
            
        Returns:
            List of dictionaries containing image data and metadata
        """
        images = []
        if "/Resources" in page and "/XObject" in page["/Resources"]:
            xObject = page["/Resources"]["/XObject"]
            for obj in xObject:
                if xObject[obj]["/Subtype"] == "/Image":
                    try:
                        image = xObject[obj]
                        image_data = image.get_data()
                        image_format = image["/Filter"]
                        
                        # Convert image data to PIL Image
                        if image_format == "/DCTDecode":
                            img = self.Image.open(io.BytesIO(image_data))
                        else:
                            img = self.Image.open(io.BytesIO(image_data))
                        
                        # Save image if output directory is specified
                        if self.config.image_output_dir:
                            output_dir = Path(self.config.image_output_dir)
                            output_dir.mkdir(parents=True, exist_ok=True)
                            
                            filename = f"page_{page_num}_image_{len(images)}.{self.config.image_format}"
                            output_path = output_dir / filename
                            
                            if self.config.image_format.lower() == "jpeg":
                                img = img.convert("RGB")
                                img.save(output_path, "JPEG", quality=self.config.image_quality)
                            else:
                                img.save(output_path, "PNG")
                            
                            images.append({
                                "path": str(output_path),
                                "format": self.config.image_format,
                                "size": os.path.getsize(output_path),
                                "dimensions": img.size
                            })
                        else:
                            # Store image data in memory
                            img_byte_arr = io.BytesIO()
                            if self.config.image_format.lower() == "jpeg":
                                img = img.convert("RGB")
                                img.save(img_byte_arr, "JPEG", quality=self.config.image_quality)
                            else:
                                img.save(img_byte_arr, "PNG")
                            
                            images.append({
                                "data": img_byte_arr.getvalue(),
                                "format": self.config.image_format,
                                "dimensions": img.size
                            })
                    except Exception as e:
                        print(f"Warning: Failed to extract image from page {page_num}: {str(e)}")
                        continue
        
        return images
    
    def prepare(self, data: Union[str, Path, bytes]) -> Dict[str, Any]:
        """Extract text, images, and metadata from the PDF.
        
        Args:
            data: The PDF data to process (file path or bytes)
            
        Returns:
            A dictionary containing the extracted text, images, and metadata
            
        Raises:
            ChefError: If there is an error during extraction
        """
        try:
            # Handle both file paths and raw bytes
            is_file_path = isinstance(data, (str, Path))
            file_path = None
            
            if is_file_path:
                file_path = data
                with open(data, 'rb') as f:
                    pdf = self.PyPDF2.PdfReader(f)
                    
                    # Extract text and images from each page
                    text_pages = []
                    all_images = []
                    start_page = 0
                    end_page = len(pdf.pages)
                    if self.config.page_range:
                        start_page, end_page = self.config.page_range
                        end_page = min(end_page, len(pdf.pages))
                        
                    for page_num in range(start_page, end_page):
                        page = pdf.pages[page_num]
                        
                        # Extract text, with OCR if configured
                        if self.config.use_ocr:
                            text_pages.append(self._extract_text_with_ocr(page, page_num, file_path))
                        else:
                            text_pages.append(page.extract_text())
                            
                        # Extract images if configured
                        if self.config.extract_images:
                            page_images = self._extract_images_from_page(page, page_num)
                            all_images.extend(page_images)
                            
                    # Create result dictionary
                    result = {
                        "text": "\n".join(text_pages),
                        "num_pages": len(pdf.pages),
                        "processed_pages": list(range(start_page, end_page))
                    }
                    
                    # Add images to result if any were extracted
                    if self.config.extract_images and all_images:
                        result["images"] = all_images
                        
                    # Extract metadata if configured
                    if self.config.extract_metadata:
                        result["metadata"] = pdf.metadata
                        
                    # Extract tables if configured
                    if self.config.extract_tables:
                        result["tables"] = self._extract_tables(file_path)
                        
                    # Extract form fields if configured
                    if self.config.extract_form_fields:
                        result["form_fields"] = self._extract_form_fields(pdf)
                        
                    # Extract annotations if configured
                    if self.config.extract_annotations:
                        result["annotations"] = self._extract_annotations(file_path)
                        
                    # Extract vector graphics if configured
                    if self.config.extract_vector_graphics:
                        result["vector_graphics"] = self._extract_vector_graphics(file_path)
                        
                    return result
            else:
                # Handle PDF as bytes
                pdf = self.PyPDF2.PdfReader(io.BytesIO(data))
                
                # Need to save to a temporary file for some extraction methods that require a file path
                temp_file_path = None
                if any([
                    self.config.extract_tables, 
                    self.config.extract_annotations, 
                    self.config.extract_vector_graphics,
                    self.config.use_ocr
                ]):
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                        temp_file.write(data)
                        temp_file_path = temp_file.name
                
                text_pages = []
                all_images = []
                start_page = 0
                end_page = len(pdf.pages)
                if self.config.page_range:
                    start_page, end_page = self.config.page_range
                    end_page = min(end_page, len(pdf.pages))
                    
                for page_num in range(start_page, end_page):
                    page = pdf.pages[page_num]
                    
                    # Extract text, with OCR if configured
                    if self.config.use_ocr and temp_file_path:
                        text_pages.append(self._extract_text_with_ocr(page, page_num, temp_file_path))
                    else:
                        text_pages.append(page.extract_text())
                        
                    # Extract images if configured
                    if self.config.extract_images:
                        page_images = self._extract_images_from_page(page, page_num)
                        all_images.extend(page_images)
                        
                # Create result dictionary
                result = {
                    "text": "\n".join(text_pages),
                    "num_pages": len(pdf.pages),
                    "processed_pages": list(range(start_page, end_page))
                }
                
                # Add images to result if any were extracted
                if self.config.extract_images and all_images:
                    result["images"] = all_images
                    
                # Extract metadata if configured
                if self.config.extract_metadata:
                    result["metadata"] = pdf.metadata
                    
                # Extract tables if configured and temp file is available
                if self.config.extract_tables and temp_file_path:
                    result["tables"] = self._extract_tables(temp_file_path)
                    
                # Extract form fields if configured
                if self.config.extract_form_fields:
                    result["form_fields"] = self._extract_form_fields(pdf)
                    
                # Extract annotations if configured and temp file is available
                if self.config.extract_annotations and temp_file_path:
                    result["annotations"] = self._extract_annotations(temp_file_path)
                    
                # Extract vector graphics if configured and temp file is available
                if self.config.extract_vector_graphics and temp_file_path:
                    result["vector_graphics"] = self._extract_vector_graphics(temp_file_path)
                    
                # Clean up temporary file if it was created
                if temp_file_path:
                    try:
                        os.unlink(temp_file_path)
                    except Exception:
                        pass
                        
                return result
                
        except Exception as e:
            raise ChefError(f"Error extracting content from PDF: {str(e)}")
    
    def clean(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean the extracted text and metadata.
        
        Args:
            data: The extracted data to clean
            
        Returns:
            The cleaned data
        """
        # Basic cleaning of extracted text
        if "text" in data:
            # Remove multiple newlines
            data["text"] = "\n".join(
                line for line in data["text"].split("\n")
                if line.strip()
            )
        
        return data
    
    def _extract_tables(self, pdf_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Extract tables from a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing table data and metadata
        """
        if not self.tabula:
            logger.warning("Table extraction requested but tabula-py not available")
            return []
            
        tables = []
        try:
            # Determine pages to process
            pages = None
            if self.config.page_range:
                start, end = self.config.page_range
                pages = list(range(start + 1, end + 1))  # tabula uses 1-based indexing
                
            # Extract tables from PDF
            raw_tables = self.tabula.read_pdf(
                pdf_path,
                pages=pages,
                multiple_tables=True,
                guess=True,
                pandas_options={'header': None}
            )
            
            # Process extracted tables
            for i, df in enumerate(raw_tables):
                if df.empty:
                    continue
                    
                # Convert table to requested format
                table_data = None
                if self.config.table_format == "markdown":
                    table_data = df.to_markdown(index=False)
                elif self.config.table_format == "html":
                    table_data = df.to_html(index=False, header=False)
                elif self.config.table_format == "csv":
                    table_data = df.to_csv(index=False, header=False)
                else:
                    # Default to JSON-serializable format
                    table_data = df.fillna("").values.tolist()
                    
                # Add table to results
                table_info = {
                    "table_index": i,
                    "format": self.config.table_format,
                    "num_rows": len(df),
                    "num_columns": len(df.columns),
                    "data": table_data
                }
                tables.append(table_info)
                
        except Exception as e:
            logger.warning(f"Error extracting tables from PDF: {str(e)}")
            
        return tables 

    def _extract_form_fields(self, pdf: Any) -> Dict[str, Any]:
        """Extract form fields from a PDF document.
        
        Args:
            pdf: PyPDF2 PdfReader object
            
        Returns:
            Dictionary containing form field data
        """
        form_fields = {}
        try:
            # Check if PDF has form fields
            if pdf.get_form_text_fields():
                # Extract and organize form fields
                form_fields["fields"] = pdf.get_form_text_fields()
                
                # Try to extract more detailed form data if available
                if hasattr(pdf, 'get_fields') and callable(getattr(pdf, 'get_fields')):
                    all_fields = pdf.get_fields()
                    detailed_fields = []
                    
                    for field_name, field_data in all_fields.items():
                        field_type = None
                        field_value = None
                        field_options = None
                        
                        # Extract field type, value and options if available
                        if hasattr(field_data, 'field_type') and field_data.field_type:
                            field_type = str(field_data.field_type)
                        
                        if '/V' in field_data:
                            field_value = str(field_data['/V'])
                            
                        if '/Opt' in field_data:
                            field_options = [str(opt) for opt in field_data['/Opt']]
                            
                        detailed_fields.append({
                            "name": field_name,
                            "type": field_type,
                            "value": field_value,
                            "options": field_options
                        })
                    
                    form_fields["detailed_fields"] = detailed_fields
            else:
                form_fields["fields"] = {}
                form_fields["detailed_fields"] = []
                
        except Exception as e:
            logger.warning(f"Error extracting form fields from PDF: {str(e)}")
            form_fields["error"] = str(e)
            
        return form_fields 

    def _extract_annotations(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Extract annotations and comments from a PDF document.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing annotation data
        """
        if not hasattr(self, 'fitz') or self.fitz is None:
            logger.warning("Annotation extraction requested but PyMuPDF not available")
            return []
            
        annotations = []
        try:
            # Open PDF with PyMuPDF
            doc = self.fitz.open(file_path)
            
            # Determine pages to process
            start_page = 0
            end_page = len(doc)
            if self.config.page_range:
                start_page, end_page = self.config.page_range
                end_page = min(end_page, len(doc))
                
            # Extract annotations from each page
            for page_num in range(start_page, end_page):
                page = doc[page_num]
                page_annots = page.annots()
                
                if not page_annots:
                    continue
                    
                for annot in page_annots:
                    # Extract basic annotation data
                    annot_data = {
                        "page": page_num,
                        "type": annot.type[1],  # [0] is code, [1] is human-readable type
                        "rect": list(annot.rect),  # bounding rectangle
                        "content": annot.info.get("content", ""),
                        "modified": annot.info.get("modDate", ""),
                        "created": annot.info.get("creationDate", "")
                    }
                    
                    # Extract annotation-specific data
                    if annot.type[0] == 0:  # Text annotation
                        annot_data["title"] = annot.info.get("title", "")
                        annot_data["subject"] = annot.info.get("subject", "")
                    elif annot.type[0] == 1:  # Link annotation
                        if annot.is_external:
                            annot_data["url"] = annot.uri
                        else:
                            annot_data["internal_link"] = annot.uri
                    elif annot.type[0] == 8:  # Highlight annotation
                        annot_data["color"] = annot.colors["stroke"] if annot.colors["stroke"] else []
                        annot_data["highlighted_text"] = annot.info.get("content", "")
                        
                    annotations.append(annot_data)
                    
            # Close document
            doc.close()
            
        except Exception as e:
            logger.warning(f"Error extracting annotations from PDF: {str(e)}")
            
        return annotations 

    def _extract_text_with_ocr(self, page: Any, page_num: int, pdf_path: Union[str, Path]) -> str:
        """Extract text from a PDF page using OCR.
        
        Args:
            page: PyPDF2 page object
            page_num: Page number (0-indexed)
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text from the page
        """
        if not hasattr(self, 'pytesseract') or self.pytesseract is None or not hasattr(self, 'cv2'):
            logger.warning("OCR requested but pytesseract or opencv-python not available")
            return ""
            
        try:
            # First try to extract text using PyPDF2
            text = page.extract_text()
            
            # If there's text and we only want to use OCR when needed, return the extracted text
            if text.strip() and self.config.ocr_only_if_needed:
                return text
                
            # Use PyMuPDF to render the page to an image for OCR
            if hasattr(self, 'fitz') and self.fitz is not None:
                # Open the document with PyMuPDF
                doc = self.fitz.open(pdf_path)
                # Get the page
                mupdf_page = doc[page_num]
                
                # Render page to a pixmap (image)
                pix = mupdf_page.get_pixmap(matrix=self.fitz.Matrix(self.config.ocr_dpi/72, self.config.ocr_dpi/72))
                
                # Convert to OpenCV format (numpy array)
                img = self.cv2.cvtColor(
                    self.cv2.imread(pix.tobytes("png")), 
                    self.cv2.COLOR_BGR2RGB
                )
                
                # Use Tesseract to extract text
                ocr_text = self.pytesseract.image_to_string(
                    img, 
                    lang=self.config.ocr_language
                )
                
                # Close document
                doc.close()
                
                return ocr_text.strip()
            else:
                logger.warning("OCR requires PyMuPDF for rendering pages")
                return text if text else ""
                
        except Exception as e:
            logger.warning(f"Error performing OCR on page {page_num}: {str(e)}")
            # Fall back to any extracted text
            return page.extract_text() 

    def _extract_vector_graphics(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Extract vector graphics from a PDF document.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing vector graphics data
        """
        if not hasattr(self, 'fitz') or self.fitz is None:
            logger.warning("Vector graphics extraction requested but PyMuPDF not available")
            return []
            
        vector_graphics = []
        try:
            # Open PDF with PyMuPDF
            doc = self.fitz.open(file_path)
            
            # Determine pages to process
            start_page = 0
            end_page = len(doc)
            if self.config.page_range:
                start_page, end_page = self.config.page_range
                end_page = min(end_page, len(doc))
                
            # Extract vector graphics from each page
            for page_num in range(start_page, end_page):
                page = doc[page_num]
                
                # Extract all paths on the page as SVG
                svg_content = page.get_svg_image()
                
                # Extract all raw paths
                paths = []
                for item in page.get_drawings():
                    item_data = {
                        "type": item["type"],
                        "rect": item.get("rect"),
                        "color": item.get("color"),
                        "fill": item.get("fill"),
                        "stroke_width": item.get("width"),
                        "items": []
                    }
                    
                    # Include all path items while keeping structure manageable
                    for path_item in item.get("items", []):
                        if isinstance(path_item, dict):
                            # Keep only key information to avoid extremely large results
                            clean_item = {
                                "type": path_item.get("type"),
                                "points": path_item.get("points")[:10] if path_item.get("points") else []  # Limit points sample
                            }
                            item_data["items"].append(clean_item)
                    
                    paths.append(item_data)
                
                # Add vector graphic data for this page
                vector_graphics.append({
                    "page": page_num,
                    "num_paths": len(paths),
                    "paths_sample": paths[:5] if paths else [],  # Include only a sample of paths
                    "svg": svg_content[:10000] if svg_content else ""  # Limit SVG string length
                })
                
            # Close document
            doc.close()
            
        except Exception as e:
            logger.warning(f"Error extracting vector graphics from PDF: {str(e)}")
            
        return vector_graphics 