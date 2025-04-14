"""Module for visualizing Chonkie."""

import base64
import html
import os
import warnings
from typing import List, Optional, Union

from chonkie.types import Chunk



# Add the theme for the Visualizer
# Pastel colored rainbow theme
PASTEL_THEME = [
    "#FFADAD",
    "#FFD6A5",  
    "#FDFFB6",
    "#CAFFBF",
    "#9BF6FF",
    "#A0C4FF",
    "#BDB2FF",
    "#FFC6FF",
]


# Add all the HTML template content here
# TODO: Make this prettier in the future — I'm not a fan of the current design
# But to keep it simple and minimal, I'm keeping it like this for now
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    {favicon_link_tag}
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol"; line-height: 1.6; padding: 0; margin: 0; background-color: #f0f2f5; color: #333; display: flex; flex-direction: column; min-height: 100vh; }}
        .content-box {{ max-width: 900px; width: 100%; margin: 30px auto; padding: 30px 20px 20px 20px; background-color: #fff; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); box-sizing: border-box; }}
        .text-display {{ white-space: pre-wrap; word-wrap: break-word; font-family: "Consolas", "Monaco", "Courier New", monospace; font-size: 0.95em; padding: 0; }}
        .text-display span[style*="background-color"] {{ border-radius: 3px; padding: 0.1em 0; cursor: help; }}
        .text-display br {{ display: block; content: ""; margin-top: 0.6em; }}
        footer {{ text-align: center; margin-top: auto; padding: 15px 0; font-size: 0.8em; color: #888; border-top: 1px solid #eee; background-color: #f0f2f5; width: 100%; }}
        footer a {{ color: #666; text-decoration: none; }}
        footer a:hover {{ text-decoration: underline; }}
        footer .heart {{ color: #d63384; display: inline-block; }}
    </style>
</head>
<body>
    {main_content}
    {footer_content}
</body>
</html>
"""

MAIN_TEMPLATE = """
<div class="content-box">
    <div class="text-display">{html_parts}</div>
</div>
"""

FOOTER_TEMPLATE = """
<footer>
    Made with <span class="heart">🤎</span> by <a href="https://github.com/chonkie-inc/chonkie" target="_blank" rel="noopener noreferrer">🦛 Chonkie</a>
</footer>
"""


class Visualizer:
    """Visualizer class for Chonkie.
    
    This class can take in Chonkie Chunks and visualize them on the terminal 
    or save them as a standalone HTML file.

    Attributes:
        theme (str): The theme to use for the visualizer (default is "pastel")

    Methods:
        print(chunks: List[Chunk], full_text: Optional[str] = None) -> None:
            Print the chunks to the terminal, with rich highlights!
        save(filename: str, chunks: List[Chunk], full_text: Optional[str] = None, title: str = "Chunk Visualization") -> None:
            Save the chunks as a standalone HTML file, always embedding a hippo emoji SVG favicon.
    
    """
    
    # Store the hippo SVG content as a class attribute
    HIPPO_SVG_CONTENT = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100"><text x="50" y="55" font-size="90" text-anchor="middle" dominant-baseline="middle">🦛</text></svg>"""

    def __init__(self, theme: Union[str, List[str]] = "pastel") -> None:
        """Initialize the Visualizer.
        
        Args:
            theme (Union[str, List[str]]): The theme to use for the visualizer (default is PASTEL_THEME)
        
        """
        # Lazy import the dependencies
        self._import_dependencies()

        # Initialize the console
        self.console = Console() # type: ignore

        # If the theme is a string, get the theme
        if isinstance(theme, str):
            self.theme = self._get_theme(theme)
        else:
            self.theme = theme

    def _import_dependencies(self) -> None:
        """Import the dependencies."""
        try:
            global Console, Text
            from rich.console import Console
            from rich.text import Text
        except ImportError as e:
            raise ImportError(f"Could not import dependencies with error: {e}. Please install the dependencies with `pip install chonkie[viz]`")

    # NOTE: This is a helper function to manage the theme
    def _get_theme(self, theme: str) -> List[str]:
        """Get the theme from the theme name."""
        if theme == "pastel":
            return PASTEL_THEME
        else:
            raise ValueError(f"Invalid theme: {theme}")

    def _get_color(self, index: int) -> str:
        """Cycles through the appropriate color list."""
        return self.theme[index % len(self.theme)]


    # NOTE: This is a helper function to manage overlapping chunk visualizations
    # At the moment, it doesn't work as expected, so we're not using it.
    def _darken_color(self, hex_color: str, amount: float = 0.7) -> str:
        """Darkens a hex color by a multiplier (0 to 1)."""
        try:
            hex_color = hex_color.lstrip('#')
            if len(hex_color) != 6:
                 if len(hex_color) == 3: hex_color = "".join([c*2 for c in hex_color])
                 else: raise ValueError("Invalid hex color format")
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            darker_rgb = tuple(max(0, int(c * amount)) for c in rgb)
            return "#{:02x}{:02x}{:02x}".format(*darker_rgb)
        except Exception as e:
            print(f"Warning: Could not darken color {hex_color}: {e}")
            return "#808080"

    def print(self, chunks: List[Chunk], full_text: Optional[str] = None) -> None:
        """Print the chunks to the terminal, with rich highlights."""
        # Check if there are any chunks to visualize
        if not chunks: 
            self.console.print("No chunks to visualize.")
            return
        # If the full text is not provided, we'll try to reconstruct it (assuming the chunks are reconstructable)
        if full_text is None:
            try:
                # Reconstruct the full text from the chunks
                full_text = "".join([chunk.text for chunk in chunks])
            except AttributeError:
                raise ValueError("Error: Chunks must have 'text', 'start_index', and 'end_index' attributes for automatic text reconstruction.")
            except Exception as e:
                raise ValueError(f"Error reconstructing full text: {e}")

        # Create a Text object to manage the text and its styles
        text = Text(full_text) # type: ignore
        text_length = len(full_text)
        spans = []
        for i, chunk in enumerate(chunks):
             try: 
                spans.append({"id": i, "start": int(chunk.start_index), "end": int(chunk.end_index)})
             except (AttributeError, TypeError, ValueError):
                warnings.warn(f"Warning: Skipping chunk with invalid start/end index: {chunk}")
                continue
        
        # Apply the styles to the text
        for span_data in spans:
            start, end = span_data["start"], span_data["end"]
            chunk_id = span_data["id"]
            if start < end and start < text_length:
                effective_end = min(end, text_length)
                color = self._get_color(chunk_id)
                style = f"on {color}"
                try: 
                    text.stylize(style, start, effective_end)
                except Exception as e:
                    warnings.warn(f"Warning: Could not apply style '{style}' to span ({start}, {effective_end}). Error: {e}")
        # Print the text with rich highlights
        self.console.print(text)

    def save(
        self,
        filename: str,
        chunks: List[Chunk],
        full_text: Optional[str] = None,
        title: str = "Chunk Visualization"
        # Removed embed_hippo_favicon parameter
    ) -> None:
        """Save the chunk visualization as a standalone, minimal HTML file, always embedding a hippo emoji SVG favicon.

        Args:
            filename (str): The path to save the HTML file.
            chunks (List[Chunk]): A list of chunk objects with 'start_index'
                                   and 'end_index'.
            full_text (Optional[str]): The complete original text. If None, it
                                       attempts reconstruction.
            title (str): The title for the browser tab.

        """
        # (Input validation and text reconstruction logic remains the same)
        if not chunks: 
            print("No chunks to visualize. HTML file not saved.")
            return
        # If the full text is not provided, we'll try to reconstruct it (assuming the chunks are reconstructable)
        if full_text is None:
            try:
                 full_text = "".join([chunk.text for chunk in chunks])
            except AttributeError: 
                raise AttributeError("Error: Chunks must have 'text', 'start_index', and 'end_index' attributes for automatic text reconstruction. HTML not saved.")
            except Exception as e: 
                raise ValueError(f"Error reconstructing full text: {e}. HTML not saved.")

        # If the filename doesn't end with ".html", add it
        if not filename.endswith(".html"):
            filename = f"{filename}.html"

        # --- 1. Validate Spans and Prepare Data ---
        validated_spans = []
        text_length = len(full_text)
        for i, chunk in enumerate(chunks):
            try:
                start, end = int(chunk.start_index), int(chunk.end_index)
                start = max(0, start); end = max(0, end)
                if start < end and start < text_length:
                    effective_end = min(end, text_length)
                    token_count = chunk.token_count
                    validated_spans.append({"id": i, "start": start, "end": effective_end, "tokens": token_count})
            except (AttributeError, TypeError, ValueError):
                warnings.warn(f"Warning: Skipping chunk with invalid start/end index: {chunk}")
                continue

        # --- 2. Generate HTML Parts (Event-based with Overlap Detection) ---
        html_parts = []
        last_processed_idx = 0
        events = []
        
        # Create events for each span
        for span_data in validated_spans:
            events.append((span_data["start"], 1, span_data["id"]))
            events.append((span_data["end"], -1, span_data["id"]))
        events.sort()
        
        # Initialize the active chunk IDs set
        active_chunk_ids: set[int] = set()

        # Iterate through the events
        for i in range(len(events)):
            event_idx, event_type, chunk_id = events[i]
            num_active = len(active_chunk_ids)
            current_bg_color = "transparent"
            # If there are active chunks, determine the primary chunk and its color
            hover_title = ""
            if num_active > 0:
                min_active_chunk_id = min(active_chunk_ids)
                primary_chunk_data = next((s for s in validated_spans if s["id"] == min_active_chunk_id), None)
                if primary_chunk_data:
                    base_color = self._get_color(primary_chunk_data["id"])
                    current_bg_color = base_color if num_active == 1 else self._darken_color(base_color, 0.65)
                    hover_title = (f"Chunk {primary_chunk_data['id']} | Start: {primary_chunk_data['start']} | End: {primary_chunk_data['end']} | Tokens: {primary_chunk_data['tokens']}{' (Overlap)' if num_active > 1 else ''}")
            # Get the text segment to process
            text_segment = full_text[last_processed_idx:event_idx]

            # If there is text to process, escape it and add it to the HTML parts
            if text_segment:
                escaped_segment = html.escape(text_segment).replace("\n", "<br>")
                # If there is a background color, add the title attribute and the span tags
                if current_bg_color != "transparent":
                    title_attr = f' title="{html.escape(hover_title)}"' if hover_title else ''
                    html_parts.append(f'<span style="background-color: {current_bg_color};"{title_attr}>')
                    html_parts.append(escaped_segment)
                    html_parts.append('</span>')
                else: html_parts.append(escaped_segment)
            last_processed_idx = event_idx
            if event_type == 1: active_chunk_ids.add(chunk_id)
            elif event_type == -1: active_chunk_ids.discard(chunk_id)
        # Process final segment
        if last_processed_idx < text_length:
            text_segment = full_text[last_processed_idx:]
            escaped_segment = html.escape(text_segment).replace("\n", "<br>")
            num_active = len(active_chunk_ids)
            current_bg_color = "transparent"; hover_title = ""
            if num_active > 0:
                 min_active_chunk_id = min(active_chunk_ids)
                 primary_chunk_data = next((s for s in validated_spans if s["id"] == min_active_chunk_id), None)
                 if primary_chunk_data:
                     base_color = self._get_color(primary_chunk_data["id"])
                     current_bg_color = base_color if num_active == 1 else self._darken_color(base_color, 0.65)
                     hover_title = (f"Chunk {primary_chunk_data['id']} | Start: {primary_chunk_data['start']} | End: {primary_chunk_data['end']} | Tokens: {primary_chunk_data['tokens']}{' (Overlap)' if num_active > 1 else ''}")
            if current_bg_color != "transparent":
                title_attr = f' title="{html.escape(hover_title)}"' if hover_title else ''
                html_parts.append(f'<span style="background-color: {current_bg_color};"{title_attr}>')
                html_parts.append(escaped_segment)
                html_parts.append('</span>')
            else: html_parts.append(escaped_segment)


        # --- 3. Assemble the final HTML page ---

        # --- Always Generate Hippo Favicon ---
        favicon_link_tag = "" # Default to empty in case of error
        try:
            encoded_svg = base64.b64encode(
                self.HIPPO_SVG_CONTENT.encode('utf-8')
            ).decode('utf-8')
            favicon_data_uri = f"data:image/svg+xml;base64,{encoded_svg}"
            favicon_link_tag = f'<link rel="icon" type="image/svg+xml" href="{favicon_data_uri}">'
        except Exception as e:
            print(f"Warning: Could not encode embedded hippo favicon: {e}")

        # Footer and Main Content (remain the same)
        footer_content = FOOTER_TEMPLATE
        main_content = MAIN_TEMPLATE.format(html_parts="".join(html_parts))

        # Assemble HTML, including the favicon tag
        html_content = HTML_TEMPLATE.format(
            title=html.escape(title),
            favicon_link_tag=favicon_link_tag,
            main_content=main_content,
            footer_content=footer_content
        )

        # --- 4. Write to file ---
        # (Remains the same)
        try:
            filepath = os.path.abspath(filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"HTML visualization saved to: file://{filepath}")
        except IOError as e:
            raise IOError(f"Error: Could not write file '{filename}': {e}")
        except Exception as e:
             raise Exception(f"An unexpected error occurred during file saving: {e}")
            

    def __call__(self, chunks: List[Chunk], full_text: Optional[str] = None) -> None:
        """Call the visualizer as a function.

        Prints the chunks to the terminal, with rich highlights.
        
        Args:
            chunks (List[Chunk]): A list of chunk objects with 'start_index'
                                   and 'end_index'.
            full_text (Optional[str]): The complete original text. If None, it
                                       attempts reconstruction.

        """
        self.print(chunks, full_text)

    def __repr__(self) -> str:
        """Return the string representation of the Visualizer."""
        return f"Visualizer(theme={self.theme})"
    
    