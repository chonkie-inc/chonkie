"""
Simple Direct Test for Issue #317
"""

def simple_verify():
    """Simple verification without complex imports"""
    print("🔍 Simple Verification of Issue #317 Fix")
    print("-" * 40)
    
    # Check the file directly
    with open('sentence_transformer.py', 'r') as f:
        content = f.read()
    
    # The key fixes that should be present
    fixes = {
        'add_special_tokens=False removed': 'add_special_tokens=False' not in content,
        'Tokenizer call fixed': 'self.model.tokenizer(text)' in content,
        'Encode call fixed': 'output_value="token_embeddings"' in content and 'add_special_tokens=False' not in content,
    }
    
    print("Checking fixes in sentence_transformer.py:")
    all_good = True
    for desc, status in fixes.items():
        icon = "✅" if status else "❌"
        print(f"   {icon} {desc}")
        if not status:
            all_good = False
    
    if all_good:
        print("\n🎉 Issue #317 should be FIXED!")
        print("The problematic 'add_special_tokens' parameters have been removed.")
    else:
        print("\n💥 Issue #317 may still exist.")
        print("Some fixes are missing from the code.")
    
    return all_good

if __name__ == "__main__":
    simple_verify()