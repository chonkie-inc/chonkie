import os
import io
import json
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import docx
from docx.document import Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph
from docx.oxml.text.run import CT_R
from docx.text.run import Run
from docx.oxml.shared import qn
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.shared import RGBColor, Inches
from PIL import Image
import html
import base64
import logging
import tempfile
from collections import defaultdict
from lxml import etree

from ..base import BaseChef, ChefError
from .config import DOCXExtractorConfig

logger = logging.getLogger(__name__)

class DOCXExtractorChef(BaseChef):
    """Chef for extracting content from DOCX documents."""
    
    def __init__(self, config: Optional[DOCXExtractorConfig] = None):
        """Initialize the DOCX extractor chef.
        
        Args:
            config: Configuration for DOCX extraction. If None, uses default config.
        """
        self.config = config or DOCXExtractorConfig()
        self._doc: Optional[Document] = None
        self._temp_files: List[str] = []
        self._list_levels: Dict[int, Dict[str, Any]] = {}
    
    def validate(self, input_data: Union[str, bytes, Path]) -> bool:
        """Validate if the input is a valid DOCX document.
        
        Args:
            input_data: Path to DOCX file or bytes content.
            
        Returns:
            bool: True if valid DOCX, False otherwise.
            
        Raises:
            ChefError: If validation fails.
        """
        try:
            if isinstance(input_data, (str, Path)):
                if not os.path.exists(input_data):
                    raise ChefError(f"File not found: {input_data}")
                if not str(input_data).lower().endswith('.docx'):
                    raise ChefError(f"Not a DOCX file: {input_data}")
            return True
        except Exception as e:
            raise ChefError(f"Error validating DOCX: {str(e)}")
    
    def _extract_table(self, table: Table) -> Dict[str, Any]:
        """Extract table content with structure.
        
        Args:
            table: The table to extract.
            
        Returns:
            Dict containing table data in the configured format.
        """
        rows = []
        for row in table.rows:
            cells = []
            for cell in row.cells:
                cell_text = self._extract_cell_content(cell)
                cells.append(cell_text)
            rows.append(cells)
        
        if self.config.table_format == "markdown":
            return self._format_table_markdown(rows)
        elif self.config.table_format == "html":
            return self._format_table_html(rows)
        else:  # json
            return {"rows": rows}
    
    def _extract_cell_content(self, cell: _Cell) -> str:
        """Extract content from a table cell.
        
        Args:
            cell: The cell to extract content from.
            
        Returns:
            Extracted text content.
        """
        text = []
        for paragraph in cell.paragraphs:
            if self.config.extract_styles:
                para = self._extract_paragraph_with_styles(paragraph)
                # Serialize dict to JSON string for table cell
                text.append(json.dumps(para, ensure_ascii=False))
            else:
                text.append(paragraph.text)
        return "\n".join(text)
    
    def _format_table_markdown(self, rows: List[List[str]]) -> Dict[str, Any]:
        """Format table as markdown.
        
        Args:
            rows: Table rows.
            
        Returns:
            Dict containing markdown formatted table.
        """
        if not rows:
            return {"markdown": ""}
        
        # Create header separator
        header_sep = ["---" for _ in rows[0]]
        
        # Format rows
        formatted_rows = []
        for row in rows:
            formatted_row = [cell.replace("|", "\\|") for cell in row]
            formatted_rows.append("| " + " | ".join(formatted_row) + " |")
        
        # Combine all parts
        markdown = "\n".join([
            formatted_rows[0],  # Header
            "| " + " | ".join(header_sep) + " |",  # Separator
            *formatted_rows[1:]  # Data rows
        ])
        
        return {"markdown": markdown}
    
    def _format_table_html(self, rows: List[List[str]]) -> Dict[str, Any]:
        """Format table as HTML.
        
        Args:
            rows: Table rows.
            
        Returns:
            Dict containing HTML formatted table.
        """
        if not rows:
            return {"html": ""}
        
        # Format rows
        formatted_rows = []
        for row in rows:
            cells = [f"<td>{html.escape(cell)}</td>" for cell in row]
            formatted_rows.append(f"<tr>{''.join(cells)}</tr>")
        
        # Combine all parts
        html_table = f"<table>\n{''.join(formatted_rows)}\n</table>"
        return {"html": html_table}
    
    def _extract_paragraph_with_styles(self, paragraph: Paragraph) -> Dict[str, Any]:
        """Extract paragraph with style information.
        
        Args:
            paragraph: The paragraph to extract.
            
        Returns:
            Dict containing paragraph text and style information.
        """
        runs = []
        for run in paragraph.runs:
            run_data = {
                "text": run.text,
                "bold": run.bold,
                "italic": run.italic,
                "underline": run.underline,
                "strike": getattr(run.font, "strike", None),
                "font": run.font.name if run.font else None,
                "size": run.font.size.pt if run.font and run.font.size else None,
                "color": str(run.font.color.rgb) if run.font and run.font.color else None
            }
            runs.append(run_data)
        
        return {
            "text": paragraph.text,
            "style": paragraph.style.name,
            "alignment": str(paragraph.alignment),
            "runs": runs
        }
    
    def _extract_headers_footers(self) -> Dict[str, Any]:
        """Extract headers and footers from the document.
        
        Returns:
            Dict containing headers and footers.
        """
        headers = []
        footers = []
        
        for section in self._doc.sections:
            # Extract header
            if section.header:
                header_text = []
                for paragraph in section.header.paragraphs:
                    if self.config.extract_styles:
                        para = self._extract_paragraph_with_styles(paragraph)
                        header_text.append(json.dumps(para, ensure_ascii=False))
                    else:
                        header_text.append(paragraph.text)
                headers.append("\n".join(header_text))
            
            # Extract footer
            if section.footer:
                footer_text = []
                for paragraph in section.footer.paragraphs:
                    if self.config.extract_styles:
                        para = self._extract_paragraph_with_styles(paragraph)
                        footer_text.append(json.dumps(para, ensure_ascii=False))
                    else:
                        footer_text.append(paragraph.text)
                footers.append("\n".join(footer_text))
        
        return {
            "headers": headers,
            "footers": footers
        }
    
    def _extract_comments(self) -> List[Dict[str, Any]]:
        """Extract comments and track changes from the document.
        
        Returns:
            List of comments with their metadata.
        """
        comments = []
        
        # Extract comments from the document
        for comment in self._doc.element.findall(".//w:comment", {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}):
            comment_data = {
                "id": comment.get(qn("w:id")),
                "author": comment.get(qn("w:author")),
                "date": comment.get(qn("w:date")),
                "text": comment.text
            }
            comments.append(comment_data)
        
        return comments
    
    def _extract_image(self, rel) -> Dict[str, Any]:
        """Extract image with metadata.
        
        Args:
            rel: The relationship object containing the image.
            
        Returns:
            Dict containing image data and metadata.
        """
        image_data = rel.target_part.blob
        image = Image.open(io.BytesIO(image_data))
        
        # Prepare image metadata
        metadata = {
            "format": image.format,
            "mode": image.mode,
            "size": image.size,
            "width": image.width,
            "height": image.height
        }
        
        # Save image if output directory is specified
        if self.config.image_output_dir:
            output_dir = Path(self.config.image_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filename = self.config.image_naming_pattern.format(
                index=len(self._temp_files)
            ) + f".{self.config.image_format}"
            output_path = output_dir / filename
            
            if self.config.image_format == "jpeg":
                image = image.convert("RGB")
                image.save(output_path, format="JPEG", quality=self.config.image_quality)
            else:
                image.save(output_path, format="PNG")
            
            self._temp_files.append(str(output_path))
            metadata["path"] = str(output_path)
        else:
            # Store image data in memory
            img_byte_arr = io.BytesIO()
            if self.config.image_format == "jpeg":
                image = image.convert("RGB")
                image.save(img_byte_arr, format="JPEG", quality=self.config.image_quality)
            else:
                image.save(img_byte_arr, format="PNG")
            metadata["data"] = img_byte_arr.getvalue()
        
        return metadata
    
    def _extract_list_structure(self, paragraph: Paragraph) -> Dict[str, Any]:
        """Extract list structure with nesting levels."""
        if not paragraph.style.name.startswith(('List', 'Number')):
            return None
        
        # Get list level and properties
        num_pr = paragraph._p.pPr.numPr if paragraph._p.pPr is not None else None
        
        # If numbering properties are not present, create a fallback based on style name
        if not num_pr:
            # Fallback mechanism for lists without numbering properties
            is_bullet = paragraph.style.name.startswith("List")
            level = 0
            num_id = hash(paragraph.style.name) % 10000  # Use a hash of style name as a pseudo-ID
            
            return {
                "level": level,
                "num_id": num_id,
                "list_type": "bullet" if is_bullet else "number",
                "start": 1,
                "format": "bullet" if is_bullet else "decimal"
            }
        
        level = num_pr.ilvl.val if num_pr.ilvl else 0
        num_id = num_pr.numId.val if num_pr.numId else None
        
        if not num_id:
            return None
        
        # Get or create list level info
        if num_id not in self._list_levels:
            self._list_levels[num_id] = {
                "type": "bullet" if paragraph.style.name.startswith("List") else "number",
                "levels": {}
            }
        
        # Store level info
        if level not in self._list_levels[num_id]["levels"]:
            self._list_levels[num_id]["levels"][level] = {
                "start": num_pr.start.val if hasattr(num_pr, 'start') and num_pr.start else 1,
                "format": num_pr.format.val if hasattr(num_pr, 'format') and num_pr.format else "decimal"
            }
        
        return {
            "level": level,
            "num_id": num_id,
            "list_type": self._list_levels[num_id]["type"],
            "start": self._list_levels[num_id]["levels"][level]["start"],
            "format": self._list_levels[num_id]["levels"][level]["format"]
        }

    def _extract_hyperlinks(self, paragraph: Paragraph) -> List[Dict[str, Any]]:
        """Extract hyperlinks from a paragraph."""
        hyperlinks = []
        
        # Check for hyperlinks in runs
        for run in paragraph.runs:
            if run._r.xpath('.//w:hyperlink'):
                for hyperlink in run._r.xpath('.//w:hyperlink'):
                    rel_id = hyperlink.get(qn('r:id'))
                    if rel_id:
                        target = self._doc.part.rels[rel_id].target_ref
                        hyperlinks.append({
                            "text": run.text,
                            "url": target,
                            "start": run._r.getparent().index(run._r),
                            "end": run._r.getparent().index(run._r) + 1
                        })
        
        # If no hyperlinks found in runs, check directly at paragraph level
        if not hyperlinks:
            for hyperlink in paragraph._p.xpath('.//w:hyperlink'):
                rel_id = hyperlink.get(qn('r:id'))
                if rel_id:
                    try:
                        target = self._doc.part.rels[rel_id].target_ref
                        
                        # Extract text from the hyperlink
                        text_elements = hyperlink.xpath('.//w:t')
                        text = ''.join([t.text for t in text_elements if hasattr(t, 'text') and t.text])
                        
                        hyperlinks.append({
                            "text": text,
                            "url": target,
                            "start": 0,  # Approximation when we don't know exact position
                            "end": 1
                        })
                    except KeyError:
                        pass  # Relationship not found
        
        return hyperlinks

    def _extract_section_properties(self, section) -> Dict[str, Any]:
        """Extract section properties.
        
        Args:
            section: The section to extract properties from.
            
        Returns:
            Dict containing section properties.
        """
        # Try to get the number of columns, fallback to 1 if not available
        columns = 1
        try:
            columns = section._sectPr.cols.count
        except Exception:
            columns = 1
        return {
            "page_width": section.page_width.inches,
            "page_height": section.page_height.inches,
            "left_margin": section.left_margin.inches,
            "right_margin": section.right_margin.inches,
            "top_margin": section.top_margin.inches,
            "bottom_margin": section.bottom_margin.inches,
            "header_distance": section.header_distance.inches,
            "footer_distance": section.footer_distance.inches,
            "gutter": section.gutter.inches,
            "orientation": "landscape" if section.orientation == 1 else "portrait",
            "columns": columns
        }

    def _extract_bookmarks(self) -> List[Dict[str, Any]]:
        """Extract bookmarks from document.
        
        Returns:
            List of bookmark data
        """
        bookmarks = []
        
        try:
            # Try to get document part and any bookmark elements
            for element in self._doc.element.body.iter():
                # Look for bookmark start elements
                if element.tag.endswith('bookmarkStart'):
                    bookmark = {
                        'name': element.get('name', ''),
                        'id': element.get('id', ''),
                        'col_first': element.get('colFirst', 0),
                        'col_last': element.get('colLast', 0),
                        'has_end': False
                    }
                    
                    # Check if there's a corresponding bookmark end
                    for end_element in self._doc.element.body.iter():
                        if (end_element.tag.endswith('bookmarkEnd') and 
                            end_element.get('id') == bookmark['id']):
                            bookmark['has_end'] = True
                            break
                    
                    bookmarks.append(bookmark)
                    
        except Exception as e:
            logger.warning(f"Error extracting bookmarks: {e}")
            
        return bookmarks
        
    def _extract_footnotes_endnotes(self) -> Dict[str, List[Dict[str, Any]]]:
        """Extract footnotes and endnotes from document.
        
        Returns:
            Dictionary containing 'footnotes' and 'endnotes' lists
        """
        result = {
            'footnotes': [],
            'endnotes': []
        }
        
        try:
            # Access footnotes part if available
            if hasattr(self._doc, 'part') and hasattr(self._doc.part, 'footnotes_part'):
                footnotes_part = self._doc.part.footnotes_part
                if footnotes_part is not None:
                    for footnote in footnotes_part.footnote_list:
                        if footnote.id >= 2:  # Skip IDs 0 and 1 (separator and continuation)
                            fn_data = {
                                'id': footnote.id,
                                'text': [],
                                'paragraphs': []
                            }
                            
                            # Extract content from footnote
                            for paragraph in footnote.paragraphs:
                                text_dict = {"text": paragraph.text}
                                if self.config.extract_styles:
                                    text_dict.update(self._extract_paragraph_with_styles(paragraph))
                                fn_data['paragraphs'].append(text_dict)
                                fn_data['text'].append(paragraph.text)
                            
                            result['footnotes'].append(fn_data)
            
            # Access endnotes part if available
            if hasattr(self._doc, 'part') and hasattr(self._doc.part, 'endnotes_part'):
                endnotes_part = self._doc.part.endnotes_part
                if endnotes_part is not None:
                    for endnote in endnotes_part.endnote_list:
                        if endnote.id >= 2:  # Skip IDs 0 and 1 (separator and continuation)
                            en_data = {
                                'id': endnote.id,
                                'text': [],
                                'paragraphs': []
                            }
                            
                            # Extract content from endnote
                            for paragraph in endnote.paragraphs:
                                text_dict = {"text": paragraph.text}
                                if self.config.extract_styles:
                                    text_dict.update(self._extract_paragraph_with_styles(paragraph))
                                en_data['paragraphs'].append(text_dict)
                                en_data['text'].append(paragraph.text)
                            
                            result['endnotes'].append(en_data)
        
        except Exception as e:
            logger.warning(f"Error extracting footnotes/endnotes: {e}")
            
        return result
        
    def _extract_equations(self) -> List[Dict[str, Any]]:
        """Extract mathematical equations from document.
        
        Returns:
            List of equation data
        """
        equations = []
        
        try:
            # Search for OMML (Office Math Markup Language) elements
            for element in self._doc.element.body.iter():
                # Look for oMath elements (equations)
                if element.tag.endswith('oMath'):
                    eq_data = {
                        'text': "".join(element.itertext()),  # Extract raw text
                        'xml': etree.tostring(element, encoding='unicode'),  # Get XML representation
                    }
                    equations.append(eq_data)
                    
                # Also look for equation fields
                elif element.tag.endswith('fldChar') and element.get('fldCharType') == 'begin':
                    # Try to find parent run and check if it's an equation
                    parent = element.getparent()
                    if parent is not None and parent.tag.endswith('r'):
                        run = parent
                        instr = None
                        
                        # Find the instruction text element
                        for sibling in run.itersiblings():
                            if sibling.tag.endswith('r'):
                                for child in sibling:
                                    if child.tag.endswith('instrText'):
                                        instr = child.text
                                        break
                            if instr:
                                break
                        
                        if instr and 'EQ' in instr:
                            eq_data = {
                                'text': instr,
                                'type': 'field'
                            }
                            equations.append(eq_data)
        
        except Exception as e:
            logger.warning(f"Error extracting equations: {e}")
            
        return equations
        
    def _extract_form_fields(self) -> List[Dict[str, Any]]:
        """Extract form fields from document.
        
        Returns:
            List of form field data
        """
        form_fields = []
        
        try:
            # Look for form fields in the document
            for element in self._doc.element.body.iter():
                # Check for legacy form fields
                if element.tag.endswith('ffData'):
                    field_data = {
                        'type': 'legacy',
                        'name': '',
                        'status': '',
                        'properties': {}
                    }
                    
                    # Try to extract field properties
                    name_elem = element.find('.//' + element.tag.split('}')[0] + '}name')
                    if name_elem is not None:
                        field_data['name'] = name_elem.text
                    
                    # Check field type and extract relevant properties
                    if element.find('.//' + element.tag.split('}')[0] + '}textInput') is not None:
                        field_data['type'] = 'text'
                        default = element.find('.//' + element.tag.split('}')[0] + '}default')
                        if default is not None:
                            field_data['properties']['default'] = default.text
                            
                    elif element.find('.//' + element.tag.split('}')[0] + '}checkBox') is not None:
                        field_data['type'] = 'checkbox'
                        checked = element.find('.//' + element.tag.split('}')[0] + '}checked')
                        if checked is not None:
                            field_data['properties']['checked'] = checked.get('val', '') == '1'
                            
                    elif element.find('.//' + element.tag.split('}')[0] + '}ddList') is not None:
                        field_data['type'] = 'dropdown'
                        items = []
                        for item in element.findall('.//' + element.tag.split('}')[0] + '}listEntry'):
                            items.append(item.get('val', ''))
                        field_data['properties']['items'] = items
                        
                    form_fields.append(field_data)
                
                # Check for content controls (newer form elements)
                elif element.tag.endswith('sdt'):
                    field_data = {
                        'type': 'content_control',
                        'name': '',
                        'tag': '',
                        'properties': {}
                    }
                    
                    # Extract properties from sdtPr element
                    props = element.find('.//' + element.tag.split('}')[0] + '}sdtPr')
                    if props is not None:
                        # Get field title
                        alias = props.find('.//' + props.tag.split('}')[0] + '}alias')
                        if alias is not None:
                            field_data['name'] = alias.get('val', '')
                            
                        # Get field tag
                        tag = props.find('.//' + props.tag.split('}')[0] + '}tag')
                        if tag is not None:
                            field_data['tag'] = tag.get('val', '')
                            
                        # Determine control type
                        if props.find('.//' + props.tag.split('}')[0] + '}text') is not None:
                            field_data['type'] = 'text'
                        elif props.find('.//' + props.tag.split('}')[0] + '}checkbox') is not None:
                            field_data['type'] = 'checkbox'
                            checked = props.find('.//' + props.tag.split('}')[0] + '}checked')
                            if checked is not None:
                                field_data['properties']['checked'] = checked.get('val', '') == '1'
                        elif props.find('.//' + props.tag.split('}')[0] + '}comboBox') is not None:
                            field_data['type'] = 'combobox'
                            items = []
                            for item in props.findall('.//' + props.tag.split('}')[0] + '}listItem'):
                                items.append(item.get('displayText', ''))
                            field_data['properties']['items'] = items
                        elif props.find('.//' + props.tag.split('}')[0] + '}dropDownList') is not None:
                            field_data['type'] = 'dropdown'
                            items = []
                            for item in props.findall('.//' + props.tag.split('}')[0] + '}listItem'):
                                items.append(item.get('displayText', ''))
                            field_data['properties']['items'] = items
                        elif props.find('.//' + props.tag.split('}')[0] + '}date') is not None:
                            field_data['type'] = 'date'
                        else:
                            field_data['type'] = 'generic'
                    
                    # Extract current content
                    content_elem = element.find('.//' + element.tag.split('}')[0] + '}sdtContent')
                    if content_elem is not None:
                        field_data['properties']['content'] = "".join(content_elem.itertext())
                        
                    form_fields.append(field_data)
        
        except Exception as e:
            logger.warning(f"Error extracting form fields: {e}")
            
        return form_fields

    def prepare(self, input_data: Union[str, bytes, Path]) -> Dict[str, Any]:
        """Extract content from DOCX document.
        
        Args:
            input_data: Path to DOCX file or bytes content.
            
        Returns:
            Dict containing extracted content and metadata.
            
        Raises:
            ChefError: If extraction fails.
        """
        try:
            # Load document
            if isinstance(input_data, (str, Path)):
                self._doc = docx.Document(input_data)
            else:
                self._doc = docx.Document(io.BytesIO(input_data))
            
            result = {
                "text": [],
                "tables": [],
                "images": [],
                "metadata": {},
                "headers_footers": {},
                "comments": [],
                "styles": {},
                "lists": [],
                "hyperlinks": [],
                "sections": [],
                "bookmarks": [],
                "footnotes_endnotes": {},
                "equations": [],
                "form_fields": []
            }
            
            # Extract text, tables, and styles from paragraphs
            for element in self._doc.element.body:
                if isinstance(element, CT_P):
                    paragraph = Paragraph(element, self._doc)
                    if paragraph.text.strip():
                        para_data = {}
                        
                        # Extract basic text
                        if self.config.extract_styles:
                            para_data = self._extract_paragraph_with_styles(paragraph)
                        else:
                            para_data = {"text": paragraph.text}
                        
                        # Extract list structure
                        if self.config.extract_list_structure:
                            list_data = self._extract_list_structure(paragraph)
                            if list_data:
                                para_data["list"] = list_data
                        
                        # Extract hyperlinks
                        if self.config.extract_hyperlinks:
                            hyperlinks = self._extract_hyperlinks(paragraph)
                            if hyperlinks:
                                para_data["hyperlinks"] = hyperlinks
                        
                        result["text"].append(para_data)
                
                elif isinstance(element, CT_Tbl) and self.config.extract_tables:
                    table = Table(element, self._doc)
                    result["tables"].append(self._extract_table(table))
            
            # Extract headers and footers
            if self.config.extract_headers_footers:
                result["headers_footers"] = self._extract_headers_footers()
            
            # Extract comments
            if self.config.extract_comments:
                result["comments"] = self._extract_comments()
            
            # Extract images
            if self.config.extract_images:
                for rel in self._doc.part.rels.values():
                    if "image" in rel.target_ref:
                        image_data = self._extract_image(rel)
                        result["images"].append(image_data)
            
            # Extract metadata
            if self.config.extract_metadata:
                core_props = self._doc.core_properties
                result["metadata"] = {
                    "title": core_props.title,
                    "author": core_props.author,
                    "created": core_props.created,
                    "modified": core_props.modified,
                    "last_modified_by": core_props.last_modified_by,
                    "revision": core_props.revision,
                    "subject": core_props.subject,
                    "keywords": core_props.keywords,
                    "category": core_props.category,
                    "comments": core_props.comments,
                    "content_status": core_props.content_status,
                    "version": core_props.version
                }
            
            # Extract styles
            if self.config.extract_styles:
                if self.config.extract_paragraph_styles:
                    result["styles"]["paragraph_styles"] = {
                        style.name: {
                            "font": style.font.name if style.font else None,
                            "size": style.font.size.pt if style.font and style.font.size else None,
                            "bold": style.font.bold if style.font else None,
                            "italic": style.font.italic if style.font else None,
                            "alignment": str(style.paragraph_format.alignment) if style.paragraph_format else None
                        }
                        for style in self._doc.styles
                        if style.type == 1  # Paragraph style
                    }
                
                if self.config.extract_character_styles:
                    result["styles"]["character_styles"] = {
                        style.name: {
                            "font": style.font.name if style.font else None,
                            "size": style.font.size.pt if style.font and style.font.size else None,
                            "bold": style.font.bold if style.font else None,
                            "italic": style.font.italic if style.font else None
                        }
                        for style in self._doc.styles
                        if style.type == 2  # Character style
                    }
            
            # Extract section properties
            if self.config.extract_section_properties:
                for section in self._doc.sections:
                    result["sections"].append(self._extract_section_properties(section))
            
            # Extract bookmarks
            if self.config.extract_bookmarks:
                result["bookmarks"] = self._extract_bookmarks()
            
            # Extract footnotes and endnotes
            if self.config.extract_footnotes_endnotes:
                result["footnotes_endnotes"] = self._extract_footnotes_endnotes()
            
            # Extract equations
            if self.config.extract_equations:
                result["equations"] = self._extract_equations()
            
            # Extract form fields
            if self.config.extract_form_fields:
                result["form_fields"] = self._extract_form_fields()
            
            return result
            
        except Exception as e:
            raise ChefError(f"Error extracting content from DOCX: {str(e)}")
    
    def clean(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean up temporary files and resources and return the data.
        
        Args:
            data: The data to clean and return
            
        Returns:
            The cleaned data
        """
        try:
            # Remove temporary image files
            for temp_file in self._temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            self._temp_files = []
            
            # Clear document reference
            self._doc = None
            
            return data
            
        except Exception as e:
            raise ChefError(f"Error cleaning up DOCX extractor: {str(e)}")
    
    def __del__(self):
        """Ensure cleanup on object destruction."""
        try:
            # Call clean with empty dict to ensure resources are cleaned
            self.clean({}) 
        except:
            # Ignore errors during destruction
            pass 