#!/usr/bin/env python3
"""PostgreSQL + pgvector with PgvectorHandshake Example.

===================================================

This example demonstrates how to use Chonkie's PgvectorHandshake to store and 
search text chunks in PostgreSQL with pgvector using the vecs client library.
We'll chunk a sample document, store it in PostgreSQL, and perform vector 
similarity searches with metadata filtering.

Prerequisites:
- PostgreSQL running with pgvector extension
- Install dependencies: pip install "chonkie[pgvector]"
- Set up your PostgreSQL connection details

What you'll learn:
1. How to connect to PostgreSQL using individual connection parameters
2. Store chunked text with embeddings using PgvectorHandshake
3. Perform semantic similarity searches with metadata filtering
4. Create vector indexes for better performance
5. Use environment variables for connection management

Author: Chonkie Team
Date: 2025

"""

import importlib.util
import os
import sys
import time
from typing import Dict, Union

if importlib.util.find_spec("vecs") is None:
    print("❌ Required dependencies not found!")
    print("Install with: pip install 'chonkie[pgvector]'")
    sys.exit(1)

from chonkie import PgvectorHandshake, RecursiveChunker


def get_database_config() -> Dict[str, Union[str, int]]:
    """Get database configuration from environment variables or defaults."""
    # Database connection parameters
    # You can modify these or use environment variables
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "database": os.getenv("POSTGRES_DB", "chonkie_demo"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
    }


def test_connection(handshake: PgvectorHandshake) -> bool:
    """Test the database connection by getting collection info."""
    try:
        info = handshake.get_collection_info()
        print(f"✅ Connected to PostgreSQL collection: {info['name']}")
        print(f"   Vector dimensions: {info['dimension']}")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to PostgreSQL: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   1. Make sure PostgreSQL is running")
        print("   2. Check your connection parameters")
        print("   3. Ensure the database exists")
        print("   4. Verify user permissions")
        print("   5. Make sure pgvector extension is installed")
        print("\n📚 Quick setup commands:")
        print("   createdb chonkie_demo")
        print("   psql -d chonkie_demo -c 'CREATE EXTENSION vector;'")
        return False


if __name__ == "__main__":
    """
    Example demonstrating PgvectorHandshake with PostgreSQL and pgvector using vecs.
    
    This example shows how to:
    1. Connect to PostgreSQL with individual connection parameters
    2. Initialize PgvectorHandshake with embeddings
    3. Chunk and store documents with vector embeddings
    4. Perform semantic similarity searches with metadata filtering
    5. Create indexes for optimal performance
    """
    
    print("🦛 Chonkie + PostgreSQL + pgvector (vecs) Example")
    print("=" * 55)
    
    # Step 1: Set up database configuration
    print("\n📋 Step 1: Database Configuration")
    db_config = get_database_config()
    print(f"   Host: {db_config['host']}:{db_config['port']}")
    print(f"   Database: {db_config['database']}")
    print(f"   User: {db_config['user']}")
    
    # Step 2: Initialize PgvectorHandshake with individual connection parameters
    print("\n🔌 Step 2: Initializing PgvectorHandshake")
    try:
        handshake = PgvectorHandshake(
            host=db_config["host"],
            port=db_config["port"],
            database=db_config["database"],
            user=db_config["user"],
            password=db_config["password"],
            collection_name="chonkie_recursive_demo",
            embedding_model="minishlab/potion-retrieval-32M"  # Fast, lightweight model
        )
        print("✅ PgvectorHandshake initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize PgvectorHandshake: {e}")
        sys.exit(1)
    
    # Step 3: Test connection
    print("\n🔗 Step 3: Testing Connection")
    if not test_connection(handshake):
        sys.exit(1)
    
    # Step 4: Prepare sample text for chunking
    print("\n📄 Step 4: Preparing Sample Text")
    sample_text = """
    Vector databases are revolutionizing how we store and search unstructured data. 
    PostgreSQL with pgvector extension provides excellent vector similarity search capabilities.
    
    The vecs library from Supabase makes it incredibly easy to work with pgvector in Python.
    It provides a high-level API for creating collections, storing vectors, and performing searches.
    
    Chonkie's PgvectorHandshake integrates seamlessly with vecs to provide powerful text chunking
    and storage capabilities. You can chunk your documents and store them with embeddings for
    semantic search.
    
    Metadata filtering is another powerful feature that allows you to filter search results
    based on custom attributes like document type, date, author, or any other metadata.
    
    Vector indexes like HNSW (Hierarchical Navigable Small World) significantly improve
    search performance for large datasets. Creating proper indexes is crucial for production
    deployments.
    """
    
    print(f"   Sample text length: {len(sample_text)} characters")
    
    # Step 5: Initialize chunker and create chunks
    print("\n🔪 Step 5: Chunking Text")
    chunker = RecursiveChunker(
        chunk_size=200  # Smaller chunks for this example
    )
    
    chunks = chunker.chunk(sample_text)
    print(f"   Created {len(chunks)} chunks")
    print(f"   Average chunk size: {sum(c.token_count for c in chunks) / len(chunks):.1f} tokens")
    
    # Step 6: Store chunks in PostgreSQL
    print("\n💾 Step 6: Storing Chunks in PostgreSQL")
    start_time = time.time()
    try:
        chunk_ids = handshake.write(chunks)
        storage_time = time.time() - start_time
        print(f"✅ Stored {len(chunk_ids)} chunks in {storage_time:.2f} seconds")
        print(f"   First chunk ID: {chunk_ids[0]}")
    except Exception as e:
        print(f"❌ Failed to store chunks: {e}")
        sys.exit(1)
    
    # Step 7: Create vector index for better performance
    print("\n🚀 Step 7: Creating Vector Index")
    try:
        handshake.create_index(method="hnsw")
        print("✅ Created HNSW index for improved search performance")
    except Exception as e:
        print(f"⚠️  Warning: Could not create index: {e}")
        print("   Search will still work but may be slower")
    
    # Step 8: Perform similarity searches
    print("\n🔍 Step 8: Performing Similarity Searches")
    
    # Search 1: Basic similarity search
    print("\n   Search 1: 'vector databases'")
    try:
        results = handshake.search("vector databases", limit=3)
        print(f"   Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            text_preview = result['text'][:100].replace('\n', ' ')
            print(f"     {i}. Similarity: {result['similarity']:.3f}")
            print(f"        Text: {text_preview}...")
            print(f"        Tokens: {result.get('token_count', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Search failed: {e}")
    
    # Search 2: Search with metadata filtering
    print("\n   Search 2: 'PostgreSQL' with metadata filtering")
    try:
        results = handshake.search(
            "PostgreSQL pgvector", 
            limit=2,
            filters={"chunk_type": {"$eq": "RecursiveChunk"}}
        )
        print(f"   Found {len(results)} results with metadata filtering:")
        for i, result in enumerate(results, 1):
            text_preview = result['text'][:80].replace('\n', ' ')
            print(f"     {i}. Similarity: {result['similarity']:.3f}")
            print(f"        Text: {text_preview}...")
            print(f"        Chunk Type: {result.get('chunk_type', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Filtered search failed: {e}")
    
    # Search 3: Search for technical concepts
    print("\n   Search 3: 'HNSW indexing performance'")
    try:
        results = handshake.search("HNSW indexing performance", limit=2)
        print(f"   Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            text_preview = result['text'][:100].replace('\n', ' ')
            print(f"     {i}. Similarity: {result['similarity']:.3f}")
            print(f"        Text: {text_preview}...")
    except Exception as e:
        print(f"   ❌ Technical search failed: {e}")
    
    # Step 9: Display collection statistics
    print("\n📊 Step 9: Collection Statistics")
    try:
        info = handshake.get_collection_info()
        print(f"   Collection Name: {info['name']}")
        print(f"   Vector Dimensions: {info['dimension']}")
        print(f"   Total Chunks Stored: {len(chunk_ids)}")
    except Exception as e:
        print(f"   ❌ Could not get collection info: {e}")
    
    # Step 10: Cleanup (optional)
    print("\n🧹 Step 10: Cleanup Options")
    cleanup = os.getenv("CHONKIE_CLEANUP", "false").lower() == "true"
    if cleanup:
        try:
            handshake.delete_collection()
            print("✅ Collection deleted successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not delete collection: {e}")
    else:
        print("   Set CHONKIE_CLEANUP=true to automatically delete the collection")
        print(f"   Collection '{info['name']}' will persist for future runs")
    
    print("\n🎉 Example completed successfully!")
    print("=" * 55)
    print("💡 Next steps:")
    print("   - Try with your own documents")
    print("   - Experiment with different embedding models")
    print("   - Use metadata filtering for complex queries")
    print("   - Explore different chunking strategies")
    print("   - Scale up with larger datasets")