from chonkie.chefs import MarkitdownChef, DoclingChef

# Create a sample markdown file
md_content = """# Demo Markdown

This is a demo markdown file.

## Features
- Easy to use
- Fast

```python
def hello():
    print("Hello, Markdown!")
```
"""
with open("demo.md", "w", encoding="utf-8") as f:
    f.write(md_content)

# Create a sample reStructuredText file
rst_content = """Demo Documentation
==================

This is a demo reStructuredText file.

Features
--------

- Easy to use
- Fast

Usage
-----

Here's how to use it::

    print("Hello, reStructuredText!")
"""
with open("demo.rst", "w", encoding="utf-8") as f:
    f.write(rst_content)

# Process with MarkitdownChef
md_chef = MarkitdownChef()
md_result = md_chef.process("demo.md")
print("=== MarkitdownChef ===")
print("Status:", md_result.status)
print("Text:", md_result.document.text)
print("HTML Content:", md_result.document.metadata["html_content"][:100], "...")  # Print first 100 chars

# Process with DoclingChef
doc_chef = DoclingChef()
doc_result = doc_chef.process("demo.md")
print("\n=== DoclingChef (Markdown) ===")
print("Sections:", doc_result.document.metadata["sections"])
print("Code Blocks:", doc_result.document.metadata["code_blocks"])

rst_result = doc_chef.process("demo.rst")
print("\n=== DoclingChef (reStructuredText) ===")
print("Sections:", rst_result.document.metadata["sections"])
print("HTML Content:", rst_result.document.metadata["html_content"][:100], "...")  # Print first 100 chars

# Clean up
import os
os.remove("demo.md")
os.remove("demo.rst") 