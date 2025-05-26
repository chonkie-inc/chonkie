"""Setup script for Chonkie's Cython extensions.

This script configures the Cython extensions used in the Chonkie library.
It includes the token_chunker, split, and merge extensions.
"""
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "chonkie.chunker.c_extensions.split",
        ["src/chonkie/chunker/c_extensions/split.pyx"],
    ),
    Extension(
        "chonkie.chunker.c_extensions.merge",
        ["src/chonkie/chunker/c_extensions/merge.pyx"],
    )
]

setup(
    ext_modules=cythonize(extensions),
)
