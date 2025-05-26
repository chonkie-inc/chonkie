#!/usr/bin/env python3
"""
Gemini Embeddings with Recursive Chunking Example
================================================

This example demonstrates how to use Google's Gemini embedding models with Chonkie's 
RecursiveChunker to chunk text and generate embeddings. We'll use AutoEmbeddings 
for easy model loading and showcase different ways to specify Gemini models.

Prerequisites:
- Set your GEMINI_API_KEY environment variable
- Install chonkie with Gemini support: pip install "chonkie[gemini]"

Author: Chonkie Team
Date: 2025
"""

import os
from typing import List

import numpy as np

from chonkie import AutoEmbeddings, RecursiveChunker
from chonkie.types import Chunk


if __name__ == "__main__":
    """
    Example demonstrating Gemini embeddings with recursive chunking.
    
    This example shows how to:
    1. Load Gemini embeddings using AutoEmbeddings
    2. Create a RecursiveChunker for intelligent text splitting
    3. Chunk a sample document
    4. Generate embeddings for each chunk
    5. Compute similarity between chunks
    """
    
    # Check if API key is available
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY environment variable is required!")
        print("Set it with: export GEMINI_API_KEY='your-api-key-here'")
        exit(1)
    
    print("🦛 Chonkie + Google Gemini Embeddings Example")
    print("=" * 50)
    
    # Sample text - a short article about artificial intelligence
    sample_text = """
    Artificial Intelligence (AI) has revolutionized the way we interact with technology. 
    From simple chatbots to complex machine learning algorithms, AI systems are becoming 
    increasingly sophisticated and capable.
    
    Machine learning, a subset of AI, focuses on enabling computers to learn and improve 
    from experience without being explicitly programmed. This field has seen tremendous 
    growth in recent years, with applications ranging from image recognition to natural 
    language processing.
    
    Deep learning, which utilizes neural networks with multiple layers, has been 
    particularly successful in tasks such as computer vision and speech recognition. 
    These models can process vast amounts of data and identify patterns that would be 
    difficult for humans to detect.
    
    The future of AI holds great promise, with potential applications in healthcare, 
    autonomous vehicles, climate modeling, and scientific research. However, it also 
    raises important questions about ethics, privacy, and the impact on employment.
    
    As we continue to develop AI technologies, it's crucial to ensure they are designed 
    and deployed responsibly, with consideration for their societal implications and 
    the need for human oversight and control.
    """
    
    # Step 1: Initialize Gemini embeddings using AutoEmbeddings
    print("\n🔧 Step 1: Loading Gemini Embeddings")
    print("-" * 30)
    
    # Different ways to specify Gemini models:
    # Option 1: Using provider syntax
    embeddings_provider = AutoEmbeddings.get_embeddings("gemini://gemini-embedding-exp-03-07")
    print(f"✅ Loaded via provider syntax: {embeddings_provider}")
    
    # Option 2: Direct model name (uses pattern matching)
    embeddings_direct = AutoEmbeddings.get_embeddings("gemini-embedding-exp-03-07")
    print(f"✅ Loaded via direct model name: {embeddings_direct}")
    
    # Option 3: Default model using provider (experimental model with higher dimensions)
    embeddings_default = AutoEmbeddings.get_embeddings("gemini://")
    print(f"✅ Loaded default Gemini model: {embeddings_default}")
    print(f"   Dimensions: {embeddings_default.dimension}")
    
    # We'll use the default experimental model for the rest of this example
    embeddings = embeddings_default
    
    # Step 2: Initialize RecursiveChunker
    print("\n🔧 Step 2: Setting up RecursiveChunker")
    print("-" * 30)
    
    chunker = RecursiveChunker(
        chunk_size=256,  # Target chunk size in tokens (power of 2)
    )
    print(f"✅ RecursiveChunker configured:")
    print(f"   - Chunk size: {chunker.chunk_size} tokens")
    
    # Step 3: Chunk the text
    print("\n🔧 Step 3: Chunking the Text")
    print("-" * 30)
    
    chunks: List[Chunk] = chunker.chunk(sample_text)
    print(f"✅ Created {len(chunks)} chunks")
    
    # Display chunk information
    for i, chunk in enumerate(chunks):
        print(f"\n📄 Chunk {i+1}:")
        print(f"   Length: {len(chunk.text)} characters")
        print(f"   Preview: {chunk.text[:100]}...")
    
    # Step 4: Generate embeddings for each chunk
    print("\n🔧 Step 4: Generating Embeddings")
    print("-" * 30)
    
    # Extract text from chunks for embedding
    chunk_texts = [chunk.text for chunk in chunks]
    
    # Generate embeddings using batch processing for efficiency
    print("🔄 Generating embeddings (this may take a few seconds)...")
    chunk_embeddings = embeddings.embed_batch(chunk_texts)
    
    print(f"✅ Generated embeddings for {len(chunk_embeddings)} chunks")
    print(f"   Embedding dimension: {chunk_embeddings[0].shape[0]}")
    print(f"   Embedding type: {type(chunk_embeddings[0])}")
    
    # Step 5: Demonstrate similarity computation
    print("\n🔧 Step 5: Computing Chunk Similarities")
    print("-" * 30)
    
    # Compute similarity matrix between all chunks
    print("📊 Chunk similarity matrix:")
    print("   " + "".join(f"C{i+1:2}" for i in range(len(chunks))))
    
    for i, emb_i in enumerate(chunk_embeddings):
        similarities = []
        for j, emb_j in enumerate(chunk_embeddings):
            similarity = embeddings.similarity(emb_i, emb_j)
            similarities.append(similarity)
        
        # Display similarity row
        similarity_str = "".join(f"{sim:5.2f}" for sim in similarities)
        print(f"C{i+1}: {similarity_str}")
    
    # Step 6: Find most similar chunk pairs
    print("\n🔧 Step 6: Finding Most Similar Chunks")
    print("-" * 30)
    
    max_similarity = 0.0
    most_similar_pair = (0, 0)
    
    for i in range(len(chunk_embeddings)):
        for j in range(i + 1, len(chunk_embeddings)):
            similarity = embeddings.similarity(chunk_embeddings[i], chunk_embeddings[j])
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_pair = (i, j)
    
    print(f"🏆 Most similar chunks: Chunk {most_similar_pair[0]+1} and Chunk {most_similar_pair[1]+1}")
    print(f"   Similarity score: {max_similarity:.4f}")
    print(f"\n📄 Chunk {most_similar_pair[0]+1} preview:")
    print(f"   {chunks[most_similar_pair[0]].text[:150]}...")
    print(f"\n📄 Chunk {most_similar_pair[1]+1} preview:")
    print(f"   {chunks[most_similar_pair[1]].text[:150]}...")
    
    # Step 7: Demonstrate token counting
    print("\n🔧 Step 7: Token Analysis")
    print("-" * 30)
    
    total_tokens = sum(embeddings.count_tokens(chunk.text) for chunk in chunks)
    print(f"📈 Token analysis:")
    print(f"   Total tokens across all chunks: {total_tokens}")
    print(f"   Average tokens per chunk: {total_tokens / len(chunks):.1f}")
    
    # Display token count for each chunk
    for i, chunk in enumerate(chunks):
        token_count = embeddings.count_tokens(chunk.text)
        print(f"   Chunk {i+1}: {token_count} tokens")
    
    print("\n🎉 Example completed successfully!")
    print("\n💡 Next steps you could try:")
    print("   - Experiment with different chunk sizes (try 128, 512, 1024)")
    print("   - Try other Gemini models like 'text-embedding-004' for faster processing")
    print("   - Use different task types: 'CLASSIFICATION', 'CLUSTERING', etc.")
    print("   - Integrate with vector databases using Chonkie's handshakes")