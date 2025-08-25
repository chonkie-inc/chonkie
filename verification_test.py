from chonkie.refinery import PropositionalRefinery
from chonkie.types import Chunk

def test_refinery(text, title):
    print(f"\n=== {title} ===")
    print(f"Original text ({len(text)} chars):")
    print(f"{text}")
    
    # Create a chunk with the text
    chunk = Chunk(
        text=text,
        start_index=0,
        end_index=len(text),
        token_count=len(text),  # Simple character count as approximation
    )
    
    # Setup refinery with balanced constraints
    refinery = PropositionalRefinery(
        method="rule-based",
        min_prop_length=10,
        max_prop_length=200,
        min_token_count=3,
        max_token_count=60,
        merge_short=True,
        inplace=False,
    )
    
    # Extract propositions
    refined_chunks = refinery.refine([chunk])
    
    print(f"\nExtracted {len(refined_chunks)} propositions:")
    for i, prop in enumerate(refined_chunks):
        print(f"{i+1}. {prop.text!r}")
    
    print("-" * 80)

# Test with various examples
examples = [
    {
        "title": "Simple Example",
        "text": "The cat sat on the mat. It was looking for mice."
    },
    {
        "title": "Complex Sentence",
        "text": "Although the economy was struggling, the company reported record profits, which surprised analysts who had predicted significant losses."
    },
    {
        "title": "Multiple Sentences with Details",
        "text": "AI systems are trained on vast amounts of data. They learn patterns from this data to make predictions. Recent advances have improved their accuracy significantly."
    },
    {
        "title": "Text with Lists",
        "text": "The benefits of exercise include: improved cardiovascular health, better mental well-being, and increased strength. Regular physical activity can also help maintain a healthy weight."
    },
    {
        "title": "Technical Content",
        "text": "Transformers architecture relies on self-attention mechanisms. Unlike RNNs, transformers process all tokens in parallel, which allows for faster training on modern hardware. This architecture has become the foundation for most state-of-the-art NLP models."
    }
]

# Run tests
for example in examples:
    test_refinery(example["text"], example["title"])

print("\nVerification test completed.") 