"""Example usage of ContextualRefinery."""

from chonkie.chunker import TokenChunker
from chonkie.refinery import ContextualRefinery
from chonkie.types import Chunk

# Sample text for demonstration
sample_text = """
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines programmed to think and learn like humans. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.

The core problems of artificial intelligence include programming computers for certain traits such as:
- Knowledge
- Reasoning
- Problem solving
- Perception
- Learning
- Planning
- Ability to manipulate and move objects

Machine learning is a core part of AI. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves. The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide.

Deep learning is a subset of machine learning. Deep learning algorithms are inspired by the structure and function of the brain, particularly the interconnections between neurons, called artificial neural networks. It uses multiple layers to progressively extract higher-level features from the raw input. For example, in image processing, lower layers may identify edges, while higher layers may identify the concepts relevant to a human such as digits or letters or faces.

Natural Language Processing (NLP) is a field of AI that gives computers the ability to understand text and spoken words in much the same way human beings can. NLP combines computational linguistics—rule-based modeling of human language—with statistical, machine learning, and deep learning models.
"""

def process_with_refinery(scoring_method, min_score=0.3):
    """Process the sample text with the contextual refinery using specified method."""
    print(f"\n===== Using {scoring_method.upper()} Scoring (min_score={min_score}) =====")
    
    # First, create chunks using a simple TokenChunker
    chunker = TokenChunker(chunk_size=100, chunk_overlap=0)
    chunks = chunker.chunk(sample_text)
    
    print(f"Original chunks: {len(chunks)}")
    
    # Now, refine the chunks using ContextualRefinery
    refinery = ContextualRefinery(
        context_window=50,        # 50 character context window
        min_context_score=min_score,    # Minimum score to include context
        scoring_method=scoring_method,  # Scoring method
        inplace=False,            # Create new chunks instead of modifying existing ones
    )
    
    # Check if the refinery is available (requires dependencies to be installed)
    if not refinery.is_available():
        print(f"\n{scoring_method} scoring requires dependencies to be installed.")
        if scoring_method in ["semantic", "hybrid"]:
            print("Please install them with: pip install spacy sentence-transformers")
        else:
            print("Please install spacy with: pip install spacy")
        print("And download the English model with: python -m spacy download en_core_web_sm")
        return None
    
    # Refine the chunks
    refined_chunks = refinery.refine(chunks)
    
    print(f"\nRefined chunks: {len(refined_chunks)}")
    for i, chunk in enumerate(refined_chunks):
        print(f"\nRefined Chunk {i+1}:")
        print(f"Text: {chunk.text[:50]}...")  # Show just the beginning
        print(f"Length: {len(chunk.text)} chars, {chunk.token_count} tokens")
        
        # Calculate how much context was added
        original_chunk = chunks[i]
        added_context = len(chunk.text) - len(original_chunk.text)
        if added_context > 0:
            print(f"Added context: {added_context} chars")
        else:
            print("No context added (below relevance threshold)")
    
    # Compare overall context addition
    total_original_chars = sum(len(c.text) for c in chunks)
    total_refined_chars = sum(len(c.text) for c in refined_chunks)
    context_increase = total_refined_chars - total_original_chars
    context_percentage = (context_increase / total_original_chars) * 100
    
    print(f"\nTotal original text: {total_original_chars} chars")
    print(f"Total refined text: {total_refined_chars} chars")
    print(f"Total added context: {context_increase} chars ({context_percentage:.1f}%)")
    
    return refined_chunks

def main():
    """Run the example."""
    print("ContextualRefinery Example\n")
    print("This example demonstrates how the ContextualRefinery enhances chunks with relevant context")
    print("using different scoring methods to determine which surrounding text to include.")
    
    # Try different scoring methods
    frequency_chunks = process_with_refinery("frequency")
    
    try:
        # Try semantic scoring with a higher threshold
        semantic_chunks = process_with_refinery("semantic", min_score=0.5)
        
        # Try hybrid scoring (combination of frequency and semantic)
        hybrid_chunks = process_with_refinery("hybrid", min_score=0.4)
        
        # Compare results
        if frequency_chunks and semantic_chunks and hybrid_chunks:
            print("\n===== Comparison of Scoring Methods =====")
            
            freq_increase = sum(len(c.text) for c in frequency_chunks) - sum(len(c.text) for c in frequency_chunks[:1])
            sem_increase = sum(len(c.text) for c in semantic_chunks) - sum(len(c.text) for c in semantic_chunks[:1])
            hybrid_increase = sum(len(c.text) for c in hybrid_chunks) - sum(len(c.text) for c in hybrid_chunks[:1])
            
            print(f"Frequency scoring added: {freq_increase} total chars of context")
            print(f"Semantic scoring added: {sem_increase} total chars of context")
            print(f"Hybrid scoring added: {hybrid_increase} total chars of context")
            
    except Exception as e:
        print(f"\nError when trying semantic or hybrid scoring: {e}")
        print("This may be due to missing dependencies.")

if __name__ == "__main__":
    main() 