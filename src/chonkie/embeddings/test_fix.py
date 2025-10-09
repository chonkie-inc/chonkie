"""
Test and Verify Issue #317 Fix
"""

import os
import sys

def verify_code_fix():
    """Verify the code changes without imports"""
    print("🔍 STEP 1: Verifying Code Fix...")
    print("-" * 40)
    
    try:
        with open('sentence_transformer.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if fixes are applied
        checks = [
            ('add_special_tokens=False' not in content, '❌ add_special_tokens parameters removed'),
            ('self.model.tokenizer(text)' in content, '✅ Tokenizer call fixed'),
            ('output_value="token_embeddings"' in content, '✅ Encode functionality preserved'),
        ]
        
        all_passed = True
        for check, message in checks:
            if check:
                print(message.replace('❌', '✅'))
            else:
                print(message)
                all_passed = False
        
        if all_passed:
            print("\n✅ CODE VERIFICATION: All fixes applied correctly!")
            return True
        else:
            print("\n❌ CODE VERIFICATION: Some fixes missing!")
            return False
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

def test_functionality():
    """Test if the fix actually works"""
    print("\n🧪 STEP 2: Testing Functionality...")
    print("-" * 40)
    
    try:
        # Add parent directory to path to avoid circular imports
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        # Now try to import
        from chonkie import LateChunker, RecursiveRules
        
        print("✅ Imports successful - no circular import issues")
        
        # Test the exact scenario from Issue #317
        chunker = LateChunker(
            embedding_model="all-MiniLM-L6-v2",
            chunk_size=10,
            rules=RecursiveRules(),
            min_characters_per_chunk=24,
        )
        
        print("✅ LateChunker created successfully")
        
        text = """First paragraph about a specific topic.
Second paragraph continuing the same topic.
Third paragraph switching to a different topic.
Fourth paragraph expanding on the new topic."""
        
        chunks = chunker(text)
        
        print(f"✅ Chunking successful - created {len(chunks)} chunks")
        print("✅ No 'add_special_tokens' parameter errors!")
        
        # Show some results
        for i, chunk in enumerate(chunks[:2]):
            print(f"   Chunk {i}: {chunk.text[:30]}... (tokens: {chunk.token_count})")
        
        return True
        
    except ValueError as e:
        if "add_special_tokens" in str(e):
            print(f"❌ ISSUE #317 STILL EXISTS: {e}")
            return False
        else:
            print(f"❌ Different ValueError: {e}")
            return False
    except ImportError as e:
        print(f"❌ Import error (circular imports): {e}")
        print("   This might be due to project structure issues")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Complete test and verification"""
    print("=" * 50)
    print("ISSUE #317 - COMPLETE TEST & VERIFICATION")
    print("=" * 50)
    
    # Step 1: Verify code changes
    code_ok = verify_code_fix()
    
    # Step 2: Test functionality (only if code is fixed)
    if code_ok:
        functionality_ok = test_functionality()
    else:
        functionality_ok = False
        print("\n⏩ Skipping functionality test - code needs fixing first")
    
    # Final result
    print("\n" + "=" * 50)
    if code_ok and functionality_ok:
        print("🎉 SUCCESS: Issue #317 COMPLETELY FIXED!")
        print("   ✓ Code changes verified")
        print("   ✓ Functionality tested")
        print("   ✓ No parameter errors")
    elif code_ok and not functionality_ok:
        print("⚠️  PARTIAL: Code is fixed but testing failed")
        print("   This might be due to environment/circular import issues")
    else:
        print("💥 FAILED: Issue #317 not fixed")
        print("   Code changes are missing or incorrect")
    print("=" * 50)

if __name__ == "__main__":
    main()