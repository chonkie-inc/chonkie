"""Setup script for Chonkie's Cython extensions.

This script configures the Cython extensions used in the Chonkie library.
It includes the token_chunker, split, merge, and NumPy-free Savitzky-Golay extensions.
"""
import os

from Cython.Build import cythonize
from setuptools import Extension, setup

# Get the c_extensions directory
c_extensions_dir = os.path.join("src", "chonkie", "chunker", "c_extensions")

extensions = [
    Extension(
        "chonkie.chunker.c_extensions.split",
        [os.path.join(c_extensions_dir, "split.pyx")],
    ),
    Extension(
        "chonkie.chunker.c_extensions.merge",
        [os.path.join(c_extensions_dir, "merge.pyx")],
    ),
    # The -O3 compile flag was removed as it caused issues with re-installing
    # the package in editable mode.
    Extension(
        "chonkie.chunker.c_extensions.savgol",
        [os.path.join(c_extensions_dir, "savgol.pyx")],
    ),
]

setup(
    ext_modules=cythonize(extensions, language_level=3),
)
