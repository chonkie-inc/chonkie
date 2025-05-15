#!/usr/bin/env python3
"""Demo of Chonkie's Multi-step Pipeline (CHOMP)."""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.chonkie import (
    Chomp,
    CSVCleanerChef,
    HTMLCleanerChef,
    JSONCleanerChef,
    MarkdownCleanerChef, 
    RecursiveChunker,
    TextCleanerChef,
    OverlapRefinery,
    JSONPorter,
)


def basic_pipeline_demo():
    """Demonstrate a basic pipeline with text cleaning and chunking."""
    print("\n=== Basic Pipeline Demo ===\n")
    
    # Create the pipeline components
    text_cleaner = TextCleanerChef(
        normalize_whitespace=True,
        strip=True,
        lowercase=False,
        remove_urls=False,
    )
    
    chunker = RecursiveChunker()
    
    # Create and build the pipeline
    pipeline = Chomp().add_chef(text_cleaner).set_chunker(chunker).build()
    
    # Process some text
    text = """
    This is   a  sample   text with   irregular  spacing
    that needs to be    chunked properly.
    
    Let's see how the CHOMP pipeline handles it!
    """
    
    chunks = pipeline(text)
    
    # Print the results
    print(f"Input text:\n{text}\n")
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:")
        print(f"  Text: {chunk.text}")
        print(f"  Tokens: {chunk.token_count}")
        print(f"  Indices: [{chunk.start_index}, {chunk.end_index}]")
        print()


def html_processing_demo():
    """Demonstrate HTML processing in a pipeline."""
    print("\n=== HTML Processing Demo ===\n")
    
    # Sample HTML content
    html_content = """
    <html>
      <body>
        <h1>Welcome to Chonkie Demo</h1>
        <p>This is a <b>demonstration</b> of the HTML cleaning capabilities.</p>
        <p>Chonkie can process <a href="https://example.com">HTML content</a> easily.</p>
        <ul>
          <li>Item 1</li>
          <li>Item 2</li>
          <li>Item 3</li>
        </ul>
      </body>
    </html>
    """
    
    # Create pipeline components
    html_cleaner = HTMLCleanerChef(preserve_line_breaks=True)
    text_cleaner = TextCleanerChef(normalize_whitespace=True)
    chunker = RecursiveChunker()
    
    # Create and build the pipeline
    pipeline = (
        Chomp()
        .add_chef(html_cleaner)
        .add_chef(text_cleaner)
        .set_chunker(chunker)
        .build()
    )
    
    # Process the HTML
    chunks = pipeline(html_content)
    
    # Print the results
    print(f"Input HTML:\n{html_content}\n")
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:")
        print(f"  Text: {chunk.text}")
        print(f"  Tokens: {chunk.token_count}")
        print()


def json_processing_demo():
    """Demonstrate JSON processing in a pipeline."""
    print("\n=== JSON Processing Demo ===\n")
    
    # Sample JSON content
    json_content = """
    {
      "article": {
        "title": "Understanding JSON Processing",
        "author": {
          "name": "Jane Smith",
          "email": "jane@example.com"
        },
        "content": "JSON (JavaScript Object Notation) is a lightweight data interchange format. It is easy for humans to read and write and easy for machines to parse and generate.",
        "tags": ["json", "data", "format", "processing"],
        "published": true,
        "views": 1250
      },
      "metadata": {
        "timestamp": "2023-07-15T10:30:00Z",
        "source": "Chonkie Blog"
      }
    }
    """
    
    # Create pipeline components
    # Flatten the nested JSON and extract only specific fields
    json_cleaner = JSONCleanerChef(
        flatten=True,
        extract_fields=[
            "article_title", 
            "article_author_name", 
            "article_content", 
            "metadata_source"
        ],
        join_text_fields=True
    )
    text_cleaner = TextCleanerChef(normalize_whitespace=True)
    chunker = RecursiveChunker()
    
    # Create and build the pipeline
    pipeline = (
        Chomp()
        .add_chef(json_cleaner)
        .add_chef(text_cleaner)
        .set_chunker(chunker)
        .build()
    )
    
    # Process the JSON
    chunks = pipeline(json_content)
    
    # Print the results
    print(f"Input JSON:\n{json_content}\n")
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:")
        print(f"  Text: {chunk.text}")
        print(f"  Tokens: {chunk.token_count}")
        print()


def csv_processing_demo():
    """Demonstrate CSV processing in a pipeline."""
    print("\n=== CSV Processing Demo ===\n")
    
    # Sample CSV content
    csv_content = """id,name,age,occupation,city,country
1,John Doe,32,Software Engineer,New York,USA
2,Jane Smith,28,Data Scientist,San Francisco,USA
3,Robert Johnson,45,Project Manager,London,UK
4,Maria Garcia,36,UX Designer,Barcelona,Spain
5,Hiroshi Tanaka,41,Product Manager,Tokyo,Japan
"""
    
    # Create pipeline components
    # Extract only specific columns and join them
    csv_cleaner = CSVCleanerChef(
        extract_columns=["name", "occupation", "city", "country"],
        join_columns=True,
        join_separator=" - "
    )
    text_cleaner = TextCleanerChef(normalize_whitespace=True)
    chunker = RecursiveChunker()
    
    # Create and build the pipeline
    pipeline = (
        Chomp()
        .add_chef(csv_cleaner)
        .add_chef(text_cleaner)
        .set_chunker(chunker)
        .build()
    )
    
    # Process the CSV
    chunks = pipeline(csv_content)
    
    # Print the results
    print(f"Input CSV:\n{csv_content}\n")
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:")
        print(f"  Text: {chunk.text}")
        print(f"  Tokens: {chunk.token_count}")
        print()


def full_pipeline_demo():
    """Demonstrate a complete pipeline with all components."""
    print("\n=== Full Pipeline Demo ===\n")
    
    # Sample markdown content
    markdown_content = """
    # Chonkie Demo
    
    This is a **demonstration** of the full CHOMP pipeline with:
    
    1. Markdown cleaning
    2. Text normalization
    3. Chunking
    4. Overlap refinement
    5. JSON export
    
    Visit [Chonkie website](https://docs.chonkie.ai) for more information!
    """
    
    # Create pipeline components
    markdown_cleaner = MarkdownCleanerChef(preserve_headings=True)
    text_cleaner = TextCleanerChef(normalize_whitespace=True, remove_urls=False)
    chunker = RecursiveChunker(chunk_size=50)  # Smaller chunk size for demonstration
    refinery = OverlapRefinery(context_size=10, method="suffix")
    porter = JSONPorter(filepath="chunked_output.json", pretty=True)
    
    # Create and build the pipeline
    pipeline = (
        Chomp()
        .add_chef(markdown_cleaner)
        .add_chef(text_cleaner)
        .set_chunker(chunker)
        .add_refinery(refinery)
        .set_porter(porter)
        .build()
    )
    
    # Process the markdown
    result = pipeline(markdown_content)
    
    # Print information about the processing
    print(f"Input markdown:\n{markdown_content}\n")
    print("Pipeline processed the text and exported results to 'chunked_output.json'")
    print(f"Export result: {result}")
    

if __name__ == "__main__":
    print("=== CHOMP Demo: Chonkie's Multi-step Pipeline ===")
    
    # Run the basic demo
    basic_pipeline_demo()
    
    try:
        # This may fail if bs4 is not installed or if the chef is not available
        html_processing_demo()
    except (ImportError, ValueError) as e:
        print(f"\nHTML processing demo skipped: {e}")
    
    try:
        # Run JSON demo
        json_processing_demo()
    except Exception as e:
        print(f"\nJSON processing demo skipped: {e}")
        
    try:
        # Run CSV demo
        csv_processing_demo()
    except Exception as e:
        print(f"\nCSV processing demo skipped: {e}")
    
    try:
        # This may fail if markdown is not installed
        full_pipeline_demo()
    except (ImportError, ValueError) as e:
        print(f"\nFull pipeline demo skipped: {e}")
        
    print("\nDemo completed!")
