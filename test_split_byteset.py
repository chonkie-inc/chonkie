#!/usr/bin/env python3
"""Test StringZilla's split_byteset functionality."""

from stringzilla import Str
import time

print("Testing StringZilla split_byteset")
print("=" * 80)

# Test text with multiple delimiters
text = Str("Hello\n\nWorld\nThis is a test. Another sentence! And more? Yes.")

print(f"Original text: {text}\n")

# Test split_byteset with newlines
print("1. split_byteset with newlines:")
parts = text.split_byteset("\n")
print(f"   Delimiters: \\n")
print(f"   Result: {parts}")
print(f"   Count: {len(parts)}")

# Test with multiple delimiters
print("\n2. split_byteset with sentence delimiters:")
parts = text.split_byteset(".!?")
print(f"   Delimiters: .!?")
print(f"   Result: {parts}")
print(f"   Count: {len(parts)}")

# Performance comparison: split on multiple delimiters
print("\n3. Performance comparison:")
long_text = "The quick brown fox. Jumped over! The lazy dog? " * 1000
sz_text = Str(long_text)

iterations = 1000

# Python approach: multiple replace + split
start = time.perf_counter()
for _ in range(iterations):
    temp = long_text
    temp = temp.replace(".", "✄")
    temp = temp.replace("!", "✄")
    temp = temp.replace("?", "✄")
    parts = temp.split("✄")
end = time.perf_counter()
python_time = end - start
print(f"   Python (replace + split): {python_time*1000:.3f}ms")

# StringZilla split_byteset
start = time.perf_counter()
for _ in range(iterations):
    parts = sz_text.split_byteset(".!?")
end = time.perf_counter()
sz_time = end - start
print(f"   StringZilla (split_byteset): {sz_time*1000:.3f}ms")

speedup = python_time / sz_time
print(f"\n   Speedup: {speedup:.2f}x")

# Test if we can emulate the RecursiveChunker behavior
print("\n4. Emulating RecursiveChunker behavior:")
test_text = "First paragraph.\n\nSecond paragraph.\nSame paragraph continued.\n\nThird paragraph."
sz_test = Str(test_text)

print(f"   Text: {test_text}")

# Try splitting on \n\n first (paragraph level)
print("\n   Splitting on \\n\\n:")
paragraphs = sz_test.split("\n\n")
for i, p in enumerate(paragraphs):
    print(f"   [{i}]: {p}")

# Now split each paragraph on \n
print("\n   Splitting first paragraph on \\n:")
if paragraphs:
    lines = paragraphs[0].split("\n")
    for i, line in enumerate(lines):
        print(f"   [{i}]: {line}")

print("\n5. Checking split_byteset behavior with delim inclusion:")
# StringZilla doesn't seem to have include_delim option, test what it returns
test = Str("a.b.c")
result = test.split_byteset(".")
print(f"   'a.b.c'.split_byteset('.'): {result}")
print(f"   Note: Delimiters are NOT included in results")
