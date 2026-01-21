"""Setup script for Chonkie's Cython extensions.

This script configures the Cython extensions used in the Chonkie library.
The split and merge extensions have been replaced by chonkie-core (Rust).
Only the NumPy-free Savitzky-Golay extension remains.
"""

import os

from Cython.Build import cythonize
from setuptools import Extension, setup

# Get the c_extensions directory
c_extensions_dir = os.path.join("src", "chonkie", "chunker", "c_extensions")

extensions = [
    Extension(
        "chonkie.chunker.c_extensions.savgol",
        [os.path.join(c_extensions_dir, "savgol.pyx")],
    ),
]

setup(
    ext_modules=cythonize(extensions, language_level=3),
)
