"""Example usage of PropositionalRefinery."""

from chonkie.chunker import TokenChunker
from chonkie.refinery import PropositionalRefinery
from chonkie.types import Chunk

# Sample text with multiple propositions
sample_text = """
The Dense X Retrieval paper introduces a novel approach to information retrieval.
Traditional methods typically use passages or sentences as retrieval units.
Propositions are defined as atomic expressions within text, each encapsulating a distinct factoid.
These propositions are presented in a concise, self-contained natural language format.
Studies show that proposition-based retrieval significantly outperforms traditional methods.
This is because the retrieved texts are more condensed with question-relevant information.
Using propositions reduces the need for lengthy input tokens and minimizes irrelevant information.
"""

def main():
    """Run the example."""
    print("PropositionalRefinery Example\n")
    
    # First, create chunks using a simple TokenChunker
    chunker = TokenChunker(chunk_size=200, chunk_overlap=0)
    chunks = chunker.chunk(sample_text)
    
    print(f"Original chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"Text: {chunk.text!r}")
        print(f"Length: {len(chunk.text)} chars, {chunk.token_count} tokens")
        print(f"Indices: {chunk.start_index}:{chunk.end_index}")
    
    # Now, refine the chunks using PropositionalRefinery
    refinery = PropositionalRefinery(
        method="rule-based",
        min_prop_length=10,    # Shorter minimum length to capture more propositions
        max_prop_length=200,
        min_token_count=3,     # Shorter minimum token count
        max_token_count=80,
        merge_short=True,
        inplace=False,  # Create new chunks instead of modifying existing ones
    )
    
    # Check if the refinery is available (requires spaCy to be installed)
    if not refinery.is_available():
        print("\nPropositionalRefinery requires spaCy to be installed.")
        print("Please install it with: pip install spacy")
        print("And download the English model with: python -m spacy download en_core_web_sm")
        return
    
    # Refine the chunks
    refined_chunks = refinery.refine(chunks)
    
    print(f"\nRefined chunks: {len(refined_chunks)}")
    for i, chunk in enumerate(refined_chunks):
        print(f"\nProposition {i+1}:")
        print(f"Text: {chunk.text!r}")
        print(f"Length: {len(chunk.text)} chars, {chunk.token_count} tokens")
        print(f"Indices: {chunk.start_index}:{chunk.end_index}")
    
    # Compare average chunk lengths
    avg_orig_len = sum(len(c.text) for c in chunks) / len(chunks)
    avg_refined_len = sum(len(c.text) for c in refined_chunks) / len(refined_chunks)
    
    print(f"\nAverage original chunk length: {avg_orig_len:.1f} chars")
    print(f"Average proposition length: {avg_refined_len:.1f} chars")
    print(f"Ratio: {avg_refined_len/avg_orig_len:.2f}")

if __name__ == "__main__":
    main() 