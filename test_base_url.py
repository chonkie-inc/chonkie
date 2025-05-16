"""
Simple test script to verify BASE_URL functionality in OpenAIEmbeddings.
"""

from chonkie.embeddings import OpenAIEmbeddings
import os
import sys

def main():
    # Test 1: Default usage (uses OpenAI's standard endpoint)
    print("Test 1: Default endpoint")
    try:
        embeddings = OpenAIEmbeddings()
        print(f"- Initialized successfully: {embeddings}")
        print(f"- Base URL: {embeddings.base_url}")
        
        # Test embedding a simple text
        result = embeddings.embed("This is a test of the OpenAI embeddings with default endpoint")
        print(f"- Embedding shape: {result.shape}")
        print("✅ Test 1 passed: Successfully used default endpoint\n")
    except Exception as e:
        print(f"❌ Test 1 failed: {e}\n")
    
    # Test 2: With explicit base_url parameter
    print("Test 2: Custom endpoint via parameter")
    custom_base_url = "https://api.openai.com/v1"  # Same as default but explicitly set
    try:
        embeddings = OpenAIEmbeddings(base_url=custom_base_url)
        print(f"- Initialized successfully: {embeddings}")
        print(f"- Base URL: {embeddings.base_url}")
        
        # Test embedding a simple text
        result = embeddings.embed("This is a test of the OpenAI embeddings with custom endpoint via parameter")
        print(f"- Embedding shape: {result.shape}")
        print("✅ Test 2 passed: Successfully used custom endpoint via parameter\n")
    except Exception as e:
        print(f"❌ Test 2 failed: {e}\n")
    
    # Test 3: With environment variable
    print("Test 3: Custom endpoint via environment variable")
    try:
        # Backup current env var
        old_base_url = os.environ.get("OPENAI_BASE_URL")
        
        # Set env var
        os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"  # Same as default but via env var
        
        embeddings = OpenAIEmbeddings()
        print(f"- Initialized successfully: {embeddings}")
        print(f"- Base URL: {embeddings.base_url}")
        
        # Test embedding a simple text
        result = embeddings.embed("This is a test of the OpenAI embeddings with custom endpoint via env var")
        print(f"- Embedding shape: {result.shape}")
        
        # Restore env var
        if old_base_url:
            os.environ["OPENAI_BASE_URL"] = old_base_url
        else:
            del os.environ["OPENAI_BASE_URL"]
            
        print("✅ Test 3 passed: Successfully used custom endpoint via env var\n")
    except Exception as e:
        # Restore env var on failure
        if old_base_url:
            os.environ["OPENAI_BASE_URL"] = old_base_url
        elif "OPENAI_BASE_URL" in os.environ:
            del os.environ["OPENAI_BASE_URL"]
        print(f"❌ Test 3 failed: {e}\n")

    # Test 4: Parameter overrides environment variable
    print("Test 4: Parameter overrides environment variable")
    try:
        # Backup current env var
        old_base_url = os.environ.get("OPENAI_BASE_URL")
        
        # Set env var
        os.environ["OPENAI_BASE_URL"] = "https://env-var-should-not-be-used.example.com/v1"
        
        # Parameter should override env var
        embeddings = OpenAIEmbeddings(base_url="https://api.openai.com/v1")
        print(f"- Initialized successfully: {embeddings}")
        print(f"- Base URL: {embeddings.base_url}")
        
        # Test embedding a simple text
        result = embeddings.embed("This is a test to verify parameter overrides environment variable")
        print(f"- Embedding shape: {result.shape}")
        
        # Verify that we're NOT using the env var URL
        assert embeddings.base_url != "https://env-var-should-not-be-used.example.com/v1"
        assert embeddings.base_url == "https://api.openai.com/v1"
        
        # Restore env var
        if old_base_url:
            os.environ["OPENAI_BASE_URL"] = old_base_url
        else:
            del os.environ["OPENAI_BASE_URL"]
            
        print("✅ Test 4 passed: Parameter successfully overrode environment variable\n")
    except Exception as e:
        # Restore env var on failure
        if old_base_url:
            os.environ["OPENAI_BASE_URL"] = old_base_url
        elif "OPENAI_BASE_URL" in os.environ:
            del os.environ["OPENAI_BASE_URL"]
        print(f"❌ Test 4 failed: {e}\n")
    
    # Test 5: URL normalization (trailing slashes)
    print("Test 5: URL normalization (trailing slashes)")
    try:
        embeddings = OpenAIEmbeddings(base_url="https://api.openai.com/v1/////")
        print(f"- Initialized successfully: {embeddings}")
        print(f"- Base URL: {embeddings.base_url}")
        
        # Verify trailing slashes are removed
        assert embeddings.base_url == "https://api.openai.com/v1"
        
        # Test embedding a simple text 
        result = embeddings.embed("This is a test to verify URL normalization")
        print(f"- Embedding shape: {result.shape}")
        print("✅ Test 5 passed: URL successfully normalized\n")
    except Exception as e:
        print(f"❌ Test 5 failed: {e}\n")

if __name__ == "__main__":
    main() 