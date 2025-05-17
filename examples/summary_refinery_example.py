"""Example usage of SummaryRefinery."""

from chonkie.chunker import TokenChunker
from chonkie.refinery import SummaryRefinery
from chonkie.types import Chunk

# Sample text with multiple paragraphs for demonstration
sample_text = """
Artificial Intelligence (AI) is transforming industries across the globe. From healthcare to finance, transportation to entertainment, AI technologies are revolutionizing how we work, live, and interact with the world around us. At its core, AI refers to the simulation of human intelligence in machines programmed to think and learn like humans. These systems can analyze vast amounts of data, recognize patterns, make decisions, and continuously improve their performance over time.

Machine learning, a subset of AI, focuses on developing algorithms that allow computers to learn from and make predictions based on data. Rather than following explicitly programmed instructions, these systems build models from sample data to make data-driven predictions or decisions. Deep learning, a more specialized branch of machine learning, utilizes artificial neural networks with multiple layers (hence "deep") to progressively extract higher-level features from raw input. For example, in image processing, lower layers might identify edges, while higher layers might identify concepts relevant to humans such as digits, letters, or faces.

Natural Language Processing (NLP) is another critical area of AI research and application. NLP focuses on the interaction between computers and human language, particularly how to program computers to process and analyze large amounts of natural language data. With recent advancements in transformer architectures like BERT, GPT, and T5, NLP has seen remarkable improvements in tasks such as translation, summarization, question answering, and conversational AI.

Computer vision enables machines to interpret and make decisions based on visual input from the world. This field has advanced significantly with the development of convolutional neural networks, allowing systems to recognize objects, understand scenes, track movement, and even generate new images. Applications range from facial recognition systems and autonomous vehicles to medical image analysis and augmented reality experiences.

Reinforcement learning, where agents learn to make decisions by taking actions in an environment to maximize some notion of cumulative reward, has enabled breakthroughs in areas like game playing (chess, Go, video games) and robotic control. By combining reinforcement learning with deep neural networks, systems can learn complex strategies and behaviors through trial and error, often exceeding human performance in specific domains.

Ethical considerations in AI development are becoming increasingly important as these technologies become more integrated into society. Issues such as algorithmic bias, privacy concerns, job displacement, security vulnerabilities, and questions of autonomy and control must be addressed thoughtfully. The field of AI ethics seeks to develop frameworks for responsible AI development and deployment, ensuring these technologies benefit humanity while minimizing potential harms.

As AI continues to evolve, we can expect more sophisticated systems that can reason across domains, require less data for learning, better understand context and nuance in human communication, and collaborate more effectively with human partners. The future of AI promises both exciting opportunities and significant challenges as we navigate this transformative technology.
"""

def test_method(method_name):
    """Test a specific summarization method."""
    print(f"\n===== Testing {method_name.upper()} Summarization =====")
    
    # First, create chunks using a simple TokenChunker
    chunker = TokenChunker(chunk_size=300, chunk_overlap=0)
    chunks = chunker.chunk(sample_text)
    
    print(f"Original chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} (first 80 chars):")
        print(f"  {chunk.text[:80]}...")
        print(f"  Length: {len(chunk.text)} chars")
    
    # Now, apply the summary refinery
    refinery = SummaryRefinery(
        method=method_name,
        model_name="t5-small" if method_name in ["abstractive", "hybrid"] else None,
        max_summary_length=80,
        min_length=20,
        summary_location="context",  # Store in metadata only
        inplace=True,
    )
    
    # Check if the refinery is available
    if not refinery.is_available():
        print(f"\n{method_name.capitalize()} summarization requires additional dependencies.")
        if method_name in ["abstractive", "hybrid"]:
            print("Please install: pip install transformers torch")
        print("Also ensure spaCy is installed: pip install spacy")
        print("And download the English model: python -m spacy download en_core_web_sm")
        return None
    
    # Refine the chunks
    refined_chunks = refinery.refine(chunks)
    
    print(f"\nRefined chunks with summaries: {len(refined_chunks)}")
    for i, chunk in enumerate(refined_chunks):
        print(f"\nChunk {i+1} Summary:")
        summary = chunk.context["summary"]
        print(f"  {summary}")
        print(f"  Summary length: {len(summary)} chars")
    
    return refined_chunks

def test_summary_location():
    """Test different summary location options."""
    print("\n===== Testing Different Summary Locations =====")
    
    # Create a single chunk for demonstration
    chunk = Chunk(
        text=sample_text,
        start_index=0,
        end_index=len(sample_text),
        token_count=len(sample_text),
    )
    
    # Test different summary locations
    locations = ["context", "prepend", "append"]
    
    for location in locations:
        print(f"\nTesting summary_location='{location}'")
        
        refinery = SummaryRefinery(
            method="extractive",
            max_summary_length=80,
            min_length=20,
            summary_location=location,
            summary_separator=" | SUMMARY: ",
            inplace=False,
        )
        
        if not refinery.is_available():
            print("Extractive summarization requires additional dependencies.")
            continue
        
        # Refine the chunk
        refined_chunk = refinery.refine([chunk.copy()])[0]
        
        # Print the summary
        print(f"Summary: {refined_chunk.context['summary']}")
        
        # Show how the chunk was modified based on location
        if location == "context":
            print("Text modification: None (summary stored in metadata only)")
        elif location == "prepend":
            print(f"Text starts with: {refined_chunk.text[:100]}...")
            print(f"Original text starts at position: {refined_chunk.text.find(chunk.text[:20])}")
        elif location == "append":
            print(f"Text ends with: ...{refined_chunk.text[-100:]}")
            print(f"Summary starts at position: {refined_chunk.text.find(refinery.summary_separator)}")

def main():
    """Run the example."""
    print("SummaryRefinery Example\n")
    print("This example demonstrates how the SummaryRefinery enhances chunks with summaries")
    print("using different summarization methods and configurations.")
    
    # Test different summarization methods
    extractive_chunks = test_method("extractive")
    
    try:
        # These methods require transformer models
        abstractive_chunks = test_method("abstractive")
        hybrid_chunks = test_method("hybrid")
        
        # Compare results if all methods were available
        if extractive_chunks and abstractive_chunks and hybrid_chunks:
            print("\n===== Comparison of Methods =====")
            
            print("\nExtractive summary (chunk 1):")
            print(f"  {extractive_chunks[0].context['summary']}")
            
            print("\nAbstractive summary (chunk 1):")
            print(f"  {abstractive_chunks[0].context['summary']}")
            
            print("\nHybrid summary (chunk 1):")
            print(f"  {hybrid_chunks[0].context['summary']}")
    except Exception as e:
        print(f"\nError when testing transformer-based methods: {e}")
        print("This may be due to missing dependencies.")
    
    # Test different summary location options
    test_summary_location()

if __name__ == "__main__":
    main() 