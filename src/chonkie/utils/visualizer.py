"""Visualization utilities for Chonkie."""

from typing import Dict, List, Optional, Union
import json


class Visualizer:
    """Visualizer for Chonkie chunks and pipelines."""
    
    def __init__(self, use_color: bool = True):
        """Initialize the visualizer.
        
        Args:
            use_color: Whether to use color in terminal output.
        """
        self.use_color = use_color
        
    def visualize_chunks(self, chunks: List, output_format: str = "text") -> str:
        """Visualize chunks.
        
        Args:
            chunks: List of chunks to visualize.
            output_format: Format to output the visualization in ("text", "html", or "json").
            
        Returns:
            The visualization as a string.
        """
        if output_format == "text":
            return self._chunks_to_text(chunks)
        elif output_format == "html":
            return self._chunks_to_html(chunks)
        elif output_format == "json":
            return self._chunks_to_json(chunks)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _chunks_to_text(self, chunks: List) -> str:
        """Convert chunks to a text representation.
        
        Args:
            chunks: List of chunks to convert.
            
        Returns:
            Text representation of the chunks.
        """
        result = []
        
        for i, chunk in enumerate(chunks):
            result.append(f"Chunk {i+1}:")
            result.append(f"  Text: {chunk.text}")
            result.append(f"  Tokens: {chunk.token_count}")
            result.append(f"  Indices: [{chunk.start_index}, {chunk.end_index}]")
            result.append("")
            
        return "\n".join(result)
    
    def _chunks_to_html(self, chunks: List) -> str:
        """Convert chunks to an HTML representation.
        
        Args:
            chunks: List of chunks to convert.
            
        Returns:
            HTML representation of the chunks.
        """
        html = ["<html><body><h1>Chunks Visualization</h1>"]
        
        for i, chunk in enumerate(chunks):
            html.append(f"<div class='chunk'>")
            html.append(f"<h3>Chunk {i+1}</h3>")
            html.append(f"<p><strong>Text:</strong> {chunk.text}</p>")
            html.append(f"<p><strong>Tokens:</strong> {chunk.token_count}</p>")
            html.append(f"<p><strong>Indices:</strong> [{chunk.start_index}, {chunk.end_index}]</p>")
            html.append("</div>")
            
        html.append("</body></html>")
        return "\n".join(html)
    
    def _chunks_to_json(self, chunks: List) -> str:
        """Convert chunks to a JSON representation.
        
        Args:
            chunks: List of chunks to convert.
            
        Returns:
            JSON representation of the chunks.
        """
        result = [chunk.to_dict() for chunk in chunks]
        return json.dumps(result, indent=2)
    
    def visualize_pipeline(self, pipeline, output_format: str = "text") -> str:
        """Visualize a pipeline.
        
        Args:
            pipeline: Pipeline to visualize.
            output_format: Format to output the visualization in ("text", "html", or "json").
            
        Returns:
            The visualization as a string.
        """
        if output_format == "text":
            return self._pipeline_to_text(pipeline)
        elif output_format == "html":
            return self._pipeline_to_html(pipeline)
        elif output_format == "json":
            return self._pipeline_to_json(pipeline)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
            
    def _pipeline_to_text(self, pipeline) -> str:
        """Convert a pipeline to a text representation.
        
        Args:
            pipeline: Pipeline to convert.
            
        Returns:
            Text representation of the pipeline.
        """
        result = ["Pipeline:"]
        
        if hasattr(pipeline, "chefs") and pipeline.chefs:
            result.append("  Chefs:")
            for chef in pipeline.chefs:
                result.append(f"    - {chef}")
                
        if hasattr(pipeline, "chunker") and pipeline.chunker:
            result.append(f"  Chunker: {pipeline.chunker}")
            
        if hasattr(pipeline, "refineries") and pipeline.refineries:
            result.append("  Refineries:")
            for refinery in pipeline.refineries:
                result.append(f"    - {refinery}")
                
        if hasattr(pipeline, "porter") and pipeline.porter:
            result.append(f"  Porter: {pipeline.porter}")
            
        if hasattr(pipeline, "handshake") and pipeline.handshake:
            result.append(f"  Handshake: {pipeline.handshake}")
            
        return "\n".join(result)
    
    def _pipeline_to_html(self, pipeline) -> str:
        """Convert a pipeline to an HTML representation.
        
        Args:
            pipeline: Pipeline to convert.
            
        Returns:
            HTML representation of the pipeline.
        """
        html = ["<html><body><h1>Pipeline Visualization</h1>"]
        
        if hasattr(pipeline, "chefs") and pipeline.chefs:
            html.append("<h2>Chefs</h2><ul>")
            for chef in pipeline.chefs:
                html.append(f"<li>{chef}</li>")
            html.append("</ul>")
                
        if hasattr(pipeline, "chunker") and pipeline.chunker:
            html.append(f"<h2>Chunker</h2><p>{pipeline.chunker}</p>")
            
        if hasattr(pipeline, "refineries") and pipeline.refineries:
            html.append("<h2>Refineries</h2><ul>")
            for refinery in pipeline.refineries:
                html.append(f"<li>{refinery}</li>")
            html.append("</ul>")
                
        if hasattr(pipeline, "porter") and pipeline.porter:
            html.append(f"<h2>Porter</h2><p>{pipeline.porter}</p>")
            
        if hasattr(pipeline, "handshake") and pipeline.handshake:
            html.append(f"<h2>Handshake</h2><p>{pipeline.handshake}</p>")
            
        html.append("</body></html>")
        return "\n".join(html)
    
    def _pipeline_to_json(self, pipeline) -> str:
        """Convert a pipeline to a JSON representation.
        
        Args:
            pipeline: Pipeline to convert.
            
        Returns:
            JSON representation of the pipeline.
        """
        result = {}
        
        if hasattr(pipeline, "chefs") and pipeline.chefs:
            result["chefs"] = [str(chef) for chef in pipeline.chefs]
                
        if hasattr(pipeline, "chunker") and pipeline.chunker:
            result["chunker"] = str(pipeline.chunker)
            
        if hasattr(pipeline, "refineries") and pipeline.refineries:
            result["refineries"] = [str(refinery) for refinery in pipeline.refineries]
                
        if hasattr(pipeline, "porter") and pipeline.porter:
            result["porter"] = str(pipeline.porter)
            
        if hasattr(pipeline, "handshake") and pipeline.handshake:
            result["handshake"] = str(pipeline.handshake)
            
        return json.dumps(result, indent=2) 