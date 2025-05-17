from chonkie.refinery.propositional import PropositionalRefinery
import spacy

# Setup refinery with even looser constraints for testing
refinery = PropositionalRefinery(
    method="rule-based",
    min_prop_length=5,  # Very low threshold
    max_prop_length=500,  # Very high threshold
    min_token_count=1,   # Very low threshold
    max_token_count=100,  # Very high threshold
)

# Sample text
sample_text = (
    "The Dense X Retrieval paper introduces propositions as atomic expressions. "
    "Propositions encapsulate distinct factoids in a concise format. "
    "Studies show that proposition-based retrieval outperforms traditional methods, "
    "because the retrieved texts are more condensed with relevant information."
)

# Load spacy directly to see what sentences are found
nlp = spacy.load("en_core_web_sm")
doc = nlp(sample_text)
print("Spacy detected sentences:")
for sent in doc.sents:
    print(f"  - {sent.text}")
    print(f"    Length: {len(sent.text)} chars, Token count: {refinery.tokenizer.count_tokens(sent.text)}")

# Test split_text_into_clauses
print("\nClauses from _split_text_into_clauses:")
clauses = refinery._split_text_into_clauses(sample_text)
for clause in clauses:
    print(f"  - {clause}")
    print(f"    Length: {len(clause)} chars, Token count: {refinery.tokenizer.count_tokens(clause)}")
    print(f"    Min length required: {refinery.min_prop_length}, Max length allowed: {refinery.max_prop_length}")
    print(f"    Min tokens required: {refinery.min_token_count}, Max tokens allowed: {refinery.max_token_count}")

# Add direct test of is_valid_proposition with verbose output
for clause in clauses:
    print(f"\nValidation check for: {clause}")
    
    # Check lengths
    if len(clause) < refinery.min_prop_length:
        print(f"  ❌ Length {len(clause)} is less than min_prop_length {refinery.min_prop_length}")
    elif len(clause) > refinery.max_prop_length:
        print(f"  ❌ Length {len(clause)} is more than max_prop_length {refinery.max_prop_length}")
    else:
        print(f"  ✅ Length {len(clause)} is valid")
    
    # Check token counts
    token_count = refinery.tokenizer.count_tokens(clause)
    if token_count < refinery.min_token_count:
        print(f"  ❌ Token count {token_count} is less than min_token_count {refinery.min_token_count}")
    elif token_count > refinery.max_token_count:
        print(f"  ❌ Token count {token_count} is more than max_token_count {refinery.max_token_count}")
    else:
        print(f"  ✅ Token count {token_count} is valid")
    
    # Final validation result
    is_valid = refinery._is_valid_proposition(clause)
    print(f"  Final validation result: {'✅ VALID' if is_valid else '❌ INVALID'}")

# Test extract_propositions directly
print("\nFinal propositions from _extract_propositions:")
props = refinery._extract_propositions(sample_text)
for prop in props:
    print(f"  - {prop}")

print(f"\nFound {len(props)} propositions")

# Try a simpler text to test if proposition extraction works at all
simple_text = "This is a simple test sentence."
print(f"\nTesting with simple text: '{simple_text}'")
print(f"Is valid proposition: {refinery._is_valid_proposition(simple_text)}")
simple_props = refinery._extract_propositions(simple_text)
print(f"Extracted propositions: {simple_props}")
print(f"Found {len(simple_props)} propositions from simple text") 