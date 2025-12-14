#!/usr/bin/env python3
"""Explore StringZilla API and test different usage patterns."""

from stringzilla import Str
import time

# Test text
text = "Hello world. This is a test. Multiple sentences here! And more? Yes indeed."

print("StringZilla API Exploration")
print("=" * 80)

# Create a Str object
sz = Str(text)

print(f"\nOriginal text: {sz}")
print(f"Type: {type(sz)}")

# Test split
print("\n1. Testing split():")
parts = sz.split(". ")
print(f"   Result: {parts}")
print(f"   Type: {type(parts)}")
if parts:
    print(f"   First element type: {type(parts[0])}")

# Test find
print("\n2. Testing find():")
idx = sz.find("test")
print(f"   Position of 'test': {idx}")

# Test replace
print("\n3. Testing replace():")
replaced = sz.replace("test", "experiment")
print(f"   Result: {replaced}")
print(f"   Type: {type(replaced)}")

# Check available methods
print("\n4. Available methods:")
methods = [m for m in dir(sz) if not m.startswith('_')]
print(f"   {methods}")

# Performance test: conversion overhead
print("\n5. Testing conversion overhead:")
test_str = "x" * 100000

iterations = 1000

# Just Python string split
start = time.perf_counter()
for _ in range(iterations):
    parts = test_str.split(" ")
end = time.perf_counter()
python_time = end - start
print(f"   Python str.split(): {python_time*1000:.3f}ms for {iterations} iterations")

# StringZilla with conversion
start = time.perf_counter()
for _ in range(iterations):
    sz_temp = Str(test_str)
    parts = sz_temp.split(" ")
end = time.perf_counter()
sz_time = end - start
print(f"   StringZilla (with conversion): {sz_time*1000:.3f}ms for {iterations} iterations")

# StringZilla pre-converted
sz_pre = Str(test_str)
start = time.perf_counter()
for _ in range(iterations):
    parts = sz_pre.split(" ")
end = time.perf_counter()
sz_pre_time = end - start
print(f"   StringZilla (pre-converted): {sz_pre_time*1000:.3f}ms for {iterations} iterations")

print(f"\n   Conversion overhead makes it {sz_time/python_time:.2f}x slower")
print(f"   Pre-converted is {sz_pre_time/python_time:.2f}x vs Python")
