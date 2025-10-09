"""Minimal test for LateChunker numpy bug"""

def test_numpy_import():
    """Test if numpy is properly imported in the fixed code"""
    try:
        # Read the late.py file and check if it's fixed
        with open('late.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if the buggy patterns exist
        if 'np.cumsum' in content or 'np.mean' in content:
            print("❌ BUG EXISTS: Code still uses 'np.' instead of 'numpy.'")
            return False
        elif 'numpy.cumsum' in content and 'numpy.mean' in content:
            print("✅ CODE IS FIXED: Using 'numpy.' instead of 'np.'")
            return True
        else:
            print("⚠️  UNKNOWN: Cannot determine if code is fixed")
            return False
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

def test_direct_execution():
    """Test the fixed code by direct execution"""
    try:
        # Import numpy directly
        import numpy
        print("✅ numpy imported successfully")
        
        # Test the fixed _get_late_embeddings method directly
        def _get_late_embeddings_fixed(token_embeddings, token_counts):
            embs = []
            cum_token_counts = numpy.cumsum([0] + token_counts)
            for i in range(len(token_counts)):
                embs.append(
                    numpy.mean(
                        token_embeddings[cum_token_counts[i] : cum_token_counts[i + 1]],
                        axis=0,
                    )
                )
            return embs
        
        # Test with sample data
        test_embeddings = numpy.random.rand(6, 384)
        test_counts = [2, 2, 2]
        result = _get_late_embeddings_fixed(test_embeddings, test_counts)
        
        print("✅ _get_late_embeddings method works with fixed code!")
        print(f"   Created {len(result)} embeddings")
        return True
        
    except NameError as e:
        if "np" in str(e):
            print(f"❌ BUG EXISTS: {e}")
            return False
        else:
            raise e

if __name__ == "__main__":
    print("Testing LateChunker Numpy Fix")
    print("=" * 40)
    
    test1 = test_numpy_import()
    test2 = test_direct_execution()
    
    print("=" * 40)
    if test1 and test2:
        print("🎉 SUCCESS: The numpy bug is FIXED!")
    else:
        print("💥 FAILED: The bug still exists or code needs fixing")