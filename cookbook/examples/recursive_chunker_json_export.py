"""RecursiveChunker with JSON Export Example.

This example demonstrates how to use the RecursiveChunker to split text into chunks
and export the results to JSON format using the JSONPorter. This is useful for
saving chunked data for later processing or analysis.

The RecursiveChunker uses a hierarchical approach, trying larger separators first
(like paragraphs) and falling back to smaller ones (like sentences, then words)
to create semantically meaningful chunks while respecting size limits.
"""

from chonkie import RecursiveChunker, JSONPorter
import json

if __name__ == "__main__":
    # Sample text for demonstration
    sample_text = """
    Artificial Intelligence has revolutionized many aspects of modern life. From healthcare 
    to transportation, AI systems are becoming increasingly sophisticated and capable.
    
    Machine learning algorithms can now process vast amounts of data to identify patterns 
    and make predictions with remarkable accuracy. Natural language processing has enabled 
    computers to understand and generate human language with unprecedented fluency.
    
    Computer vision systems can recognize objects, faces, and scenes in images and videos. 
    This technology powers applications like autonomous vehicles, medical imaging analysis, 
    and security systems.
    
    However, the rapid advancement of AI also raises important ethical considerations. 
    Issues such as bias in algorithms, privacy concerns, and the potential displacement 
    of human workers need careful attention as we continue to develop and deploy AI systems.
    
    The future of AI holds both tremendous promise and significant challenges. Responsible 
    development and deployment of AI technologies will be crucial for ensuring that their 
    benefits are realized while minimizing potential risks.
    """
    
    # Initialize the RecursiveChunker with custom parameters
    chunker = RecursiveChunker(
        chunk_size=128         # Target chunk size in tokens
    )
    
    # Chunk the text
    print("Chunking text...")
    chunks = chunker(sample_text)
    
    print(f"Created {len(chunks)} chunks")
    print("\nChunk details:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {len(chunk.text)} chars, {chunk.token_count} tokens")
    
    # Export to JSON using JSONPorter
    print("\nExporting to JSON...")
    
    # Export as JSON (not JSONL)
    porter_json = JSONPorter(lines=False)
    output_file = "chunked_ai_text.json"
    porter_json(chunks, output_file)
    print(f"Exported chunks to {output_file}")
    
    # Also export as JSONL 
    porter_jsonl = JSONPorter(lines=True)
    output_file_lines = "chunked_ai_text.jsonl"
    porter_jsonl(chunks, output_file_lines)
    print(f"Exported chunks to {output_file_lines}")
    
    # Display a preview of the JSON structure
    print("\nJSON structure preview:")
    with open(output_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    print(f"  Total chunks: {len(json_data)}")
    
    # Show first chunk details
    if json_data:
        first_chunk = json_data[0]
        print(f"  First chunk keys: {list(first_chunk.keys())}")
        print(f"  First chunk text preview: {first_chunk['text'][:100]}...")
    
    # Example of custom JSON export with additional metadata
    print("\nCreating enhanced JSON export...")
    
    # Add some custom metadata to chunks
    for i, chunk in enumerate(chunks):
        chunk.metadata = {
            'chunk_id': f"ai_text_chunk_{i+1:03d}",
            'source': 'AI overview article',
            'processing_timestamp': '2024-01-01T12:00:00Z'
        }
    
    # Export with enhanced data
    enhanced_file = "chunked_ai_text_enhanced.json"
    porter_json(chunks, enhanced_file)
    print(f"Enhanced export saved to {enhanced_file}")
    
    # Load and verify the exported data
    print("\nVerifying export integrity...")
    with open(output_file, 'r', encoding='utf-8') as f:
        loaded_data = json.load(f)
    
    original_text_length = sum(len(chunk.text) for chunk in chunks)
    loaded_text_length = sum(len(chunk['text']) for chunk in loaded_data)
    
    print(f"Original total text length: {original_text_length} chars")
    print(f"Loaded total text length: {loaded_text_length} chars")
    print(f"Data integrity: {'✓ PASSED' if original_text_length == loaded_text_length else '✗ FAILED'}")