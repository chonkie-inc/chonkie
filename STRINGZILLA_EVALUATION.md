# StringZilla Optimization Evaluation for Chonkie

## Executive Summary

**Verdict**: StringZilla provides **limited benefits** for RecursiveChunker due to API limitations.

- **Best case speedup**: 3-15x for multi-delimiter splitting (without delimiter inclusion)
- **Common case**: Cannot optimize (lacks `replace()` functionality needed for `include_delim`)
- **Recommendation**: Keep existing Cython extensions; consider StringZilla only as a fallback

## Benchmark Results

### Test 1: Multi-Delimiter Splitting (StringZilla's Strength)
Splitting on sentence delimiters `.!?` without delimiter inclusion:

| Text Size | Python (replace + split) | StringZilla (split_byteset) | Speedup |
|-----------|-------------------------|----------------------------|---------|
| Small (1KB) | 0.002ms | 0.001ms | **3.2x** |
| Medium (100KB) | 0.140ms | 0.019ms | **7.5x** |
| Large (1MB) | 2.754ms | 0.188ms | **14.6x** |

**Result**: StringZilla's `split_byteset()` significantly outperforms Python's multi-replace approach.

### Test 2: Single Delimiter Splitting
Splitting on paragraph boundaries `\n\n`:

| Text Size | Python (str.split) | StringZilla (Str.split) | Speedup |
|-----------|-------------------|------------------------|---------|
| Small (1KB) | 0.001ms | 0.000ms | **2.0x** |
| Medium (100KB) | 0.039ms | 0.007ms | **5.3x** |
| Large (1MB) | 0.391ms | 0.073ms | **5.4x** |

**Result**: StringZilla provides consistent 2-5x speedup for single delimiter splits.

### Test 3: Find Operations
Finding delimiter positions in text:

| Text Size | Pattern | Speedup |
|-----------|---------|---------|
| Small (1KB) | `\n\n` | 1.5x |
| Small (1KB) | `. ` | 1.6x |
| Small (1KB) | `the` | 1.8x |
| Large (1MB) | `\n\n` | 1.5x |
| Large (1MB) | `. ` | 1.6x |
| Large (1MB) | `the` | 1.8x |

**Result**: Modest 1.5-1.8x speedup for find operations.

## Critical Limitation: No `replace()` Support

### The Problem

RecursiveChunker's most common use case requires delimiter inclusion:

```python
# Typical RecursiveLevel configuration
RecursiveLevel(delimiters=[". ", "! ", "? "], include_delim="prev")
```

This requires replacing delimiters with markers:
```python
for delimiter in delimiters:
    text = text.replace(delimiter, delimiter + separator)  # ❌ StringZilla doesn't have replace()
```

**StringZilla's Str object does NOT have a `replace()` method.**

### Impact

Looking at common RecursiveChunker recipes (src/chonkie/chunker/recursive.py:132-191):

1. **WITH include_delim** (most common): ❌ Cannot optimize
   - Sentence boundaries: `include_delim="prev"`
   - Paragraph markers: `include_delim="next"`
   - Estimated ~70-80% of use cases

2. **WITHOUT include_delim**: ✓ Can optimize (3-15x speedup)
   - Estimated ~20-30% of use cases

3. **Whitespace splitting**: ✓ Minimal benefit (~1.1x)

4. **Token-based**: N/A (uses tokenizer, not string ops)

## Existing Optimizations

Chonkie already has Cython extensions that provide excellent performance:

### split.pyx (140 lines)
- Uses `PyUnicode_Replace()` C API
- Handles `include_delim` correctly
- ~2-3x faster than Python fallback
- ✓ Production-ready and battle-tested

### merge.pyx (249 lines)
- Uses C arrays for token counting
- Inline binary search
- ~48% improvement (most critical bottleneck)
- ✓ Targets the actual performance hotspot

### Coverage
```
✓ Split extension (Cython) is available
✓ Merge extension (Cython) is available
```

## Recommendation

### DO NOT integrate StringZilla as primary optimization because:

1. **API mismatch**: Lacks `replace()` needed for common use cases
2. **Limited scope**: Only helps 20-30% of operations
3. **Existing solution**: Cython extensions already provide 2-3x speedup
4. **Integration cost**: Would require significant refactoring for minimal gain

### CONSIDER StringZilla for:

1. **Pure-Python fallback**: When Cython unavailable (better than nothing)
   - Use `split_byteset()` for multi-delimiter splits without inclusion
   - Use `Str.split()` for simple splits
   - Provides 3-15x speedup over pure Python

2. **Specific operations**:
   - Find operations (1.5-2x speedup)
   - Character set operations (`find_first_of`, `find_last_of`)

3. **Future enhancements**:
   - If StringZilla adds `replace()` support, re-evaluate
   - Could be valuable for MarkdownChef regex operations

## Alternative Optimizations

Instead of StringZilla, focus on:

1. **Enhance Cython extensions**:
   - Add SIMD-optimized replace operation to split.pyx
   - Optimize merge.pyx further with AVX2/NEON
   - Better instruction than generic string library

2. **Profile-guided optimization**:
   - Identify actual bottlenecks in production workloads
   - The merge phase (token accumulation) is more critical than split

3. **Algorithmic improvements**:
   - Cache optimization (already using LRU with 4096 entries)
   - Reduce redundant tokenizer calls
   - Batch processing optimizations

## Conclusion

StringZilla is an impressive library with significant performance benefits (3-15x for certain operations), but **its lack of `replace()` functionality makes it unsuitable as a primary optimization for RecursiveChunker**.

The existing Cython extensions are better suited for Chonkie's needs:
- ✓ Support all required operations (including `include_delim`)
- ✓ Provide comparable performance (2-3x)
- ✓ Target the actual bottlenecks (merge phase)
- ✓ Already integrated and tested

**Recommended Action**: Close this evaluation branch without merging StringZilla integration. Consider it only as a pure-Python fallback if needed in the future.

---

## Benchmark Files

- `benchmark_stringzilla.py` - Initial exploration
- `benchmark_final.py` - Comprehensive analysis
- `test_split_byteset.py` - API testing
- `check_sz_methods.py` - Method availability check

Run the final benchmark:
```bash
python benchmark_final.py
```
