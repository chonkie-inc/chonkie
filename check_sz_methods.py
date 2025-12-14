#!/usr/bin/env python3
"""Check StringZilla available methods."""

from stringzilla import Str
import stringzilla

print("StringZilla module attributes:")
print("=" * 80)
sz_attrs = [attr for attr in dir(stringzilla) if not attr.startswith('_')]
for attr in sz_attrs:
    print(f"  - {attr}")

print("\nStringZilla Str methods:")
print("=" * 80)
sz = Str("test")
methods = [m for m in dir(sz) if not m.startswith('_')]
for method in methods:
    print(f"  - {method}")

# Check for split variations
print("\nTesting split methods:")
text = Str("a:b:c,d,e")
print(f"  split(':'): {text.split(':')}")
print(f"  split(','): {text.split(',')}")

# Check if there's a way to split on multiple delimiters
print("\nChecking documentation...")
print(f"  Str.__doc__: {Str.__doc__}")

# Test what the module offers
print("\nStringZilla module functions:")
for attr in dir(stringzilla):
    if not attr.startswith('_'):
        obj = getattr(stringzilla, attr)
        if callable(obj):
            print(f"  - {attr}: {type(obj)}")
