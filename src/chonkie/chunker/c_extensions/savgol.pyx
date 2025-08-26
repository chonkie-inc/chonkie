# distutils: language = c
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""
NumPy-free Cython wrapper for pure C Savitzky-Golay implementation.

This module provides Python bindings for the pure C implementation without NumPy dependencies.
It handles conversion between Python lists/arrays and C arrays.
"""

import cython
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

# External C declarations
cdef extern from "savgol_pure.h":
    ctypedef struct ArrayResult:
        double* data
        size_t size
    
    ctypedef struct MinimaResult:
        int* indices
        double* values
        size_t count
    
    ArrayResult* create_array_result(size_t size)
    void free_array_result(ArrayResult* result)
    MinimaResult* create_minima_result(size_t size)
    void free_minima_result(MinimaResult* result)
    
    ArrayResult* savgol_filter_pure(const double* data, size_t n,
                                    int window_length, int polyorder, int deriv)
    
    MinimaResult* find_local_minima_interpolated_pure(const double* data, size_t n,
                                                      int window_size, int poly_order,
                                                      double tolerance)
    
    ArrayResult* windowed_cross_similarity_pure(const double* embeddings, size_t n, size_t d,
                                               int window_size)
    
    MinimaResult* filter_split_indices_pure(const int* indices, const double* values,
                                           size_t n_indices, double threshold,
                                           int min_distance)

# Helper function to convert Python list to C array
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double* list_to_c_array(list data, size_t* length):
    """Convert Python list to C double array."""
    cdef size_t n = len(data)
    cdef double* c_array = <double*>malloc(n * sizeof(double))
    if not c_array:
        raise MemoryError("Failed to allocate memory for C array")
    
    for i in range(n):
        c_array[i] = float(data[i])
    
    length[0] = n
    return c_array

# Helper function to convert Python list of ints to C array
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int* int_list_to_c_array(list data, size_t* length):
    """Convert Python list of ints to C int array."""
    cdef size_t n = len(data)
    cdef int* c_array = <int*>malloc(n * sizeof(int))
    if not c_array:
        raise MemoryError("Failed to allocate memory for C array")
    
    for i in range(n):
        c_array[i] = int(data[i])
    
    length[0] = n
    return c_array

# Helper to convert C array to Python list
@cython.boundscheck(False)
@cython.wraparound(False)
cdef list c_array_to_list(double* data, size_t n):
    """Convert C double array to Python list."""
    result = []
    for i in range(n):
        result.append(data[i])
    return result

# Helper to convert C int array to Python list
@cython.boundscheck(False)
@cython.wraparound(False)
cdef list c_int_array_to_list(int* data, size_t n):
    """Convert C int array to Python list."""
    result = []
    for i in range(n):
        result.append(data[i])
    return result

# Python-accessible functions

def savgol_filter(data, int window_length=5, int polyorder=3, int deriv=0, bint use_float32=False):
    """
    Apply Savitzky-Golay filter without NumPy.
    
    Args:
        data: Input data (list or array-like)
        window_length: Length of the filter window (must be odd and > polyorder)
        polyorder: Order of the polynomial
        deriv: Derivative order (0=smoothing, 1=first, 2=second)
        use_float32: Ignored (kept for compatibility)
    
    Returns:
        Filtered data as a list
    """
    # Convert input to list if needed
    if not isinstance(data, list):
        data = list(data)
    
    cdef size_t n
    cdef double* c_data = list_to_c_array(data, &n)
    cdef ArrayResult* result
    
    try:
        # Call C function
        result = savgol_filter_pure(c_data, n, window_length, polyorder, deriv)
        if not result:
            raise ValueError("Invalid parameters for Savitzky-Golay filter")
        
        # Convert result to Python list
        py_result = c_array_to_list(result.data, result.size)
        
        # Free C result
        free_array_result(result)
        
        return py_result
    finally:
        free(c_data)

def find_local_minima_interpolated(data, int window_size=11, int poly_order=2,
                                  double tolerance=0.2, bint use_float32=False):
    """
    Find local minima with sub-sample accuracy without NumPy.
    
    Args:
        data: Input data (list or array-like)
        window_size: Savitzky-Golay window size
        poly_order: Polynomial order
        tolerance: Tolerance for considering derivative as zero
        use_float32: Ignored (kept for compatibility)
    
    Returns:
        Tuple of (indices, values) as lists
    """
    # Convert input to list if needed
    if not isinstance(data, list):
        data = list(data)
    
    cdef size_t n
    cdef double* c_data = list_to_c_array(data, &n)
    cdef MinimaResult* result
    
    try:
        # Call C function
        result = find_local_minima_interpolated_pure(
            c_data, n, window_size, poly_order, tolerance
        )
        if not result:
            raise ValueError("Invalid parameters for minima detection")
        
        # Convert results to Python lists
        indices = c_int_array_to_list(result.indices, result.count)
        values = c_array_to_list(result.values, result.count)
        
        # Free C result
        free_minima_result(result)
        
        return (indices, values)
    finally:
        free(c_data)

def windowed_cross_similarity(embeddings, int window_size):
    """
    Compute windowed cross-similarity without NumPy.
    
    Args:
        embeddings: List of embedding vectors (list of lists)
        window_size: Size of sliding window (must be odd and >= 3)
    
    Returns:
        List of average similarities for each position
    """
    # Handle embeddings as list of lists
    if not embeddings:
        return []
    
    cdef size_t n = len(embeddings)
    cdef size_t d = len(embeddings[0]) if n > 0 else 0
    cdef size_t idx = 0
    cdef double* c_embeddings
    cdef ArrayResult* result
    
    if d == 0:
        return []
    
    # Flatten embeddings to C array (row-major)
    c_embeddings = <double*>malloc(n * d * sizeof(double))
    if not c_embeddings:
        raise MemoryError("Failed to allocate memory for embeddings")
    
    idx = 0
    for i in range(n):
        emb = embeddings[i]
        if len(emb) != d:
            free(c_embeddings)
            raise ValueError("All embeddings must have the same dimension")
        for j in range(d):
            c_embeddings[idx] = float(emb[j])
            idx += 1
    
    try:
        # Call C function
        result = windowed_cross_similarity_pure(
            c_embeddings, n, d, window_size
        )
        if not result:
            raise ValueError("Invalid parameters for windowed cross-similarity")
        
        # Convert result to Python list
        py_result = c_array_to_list(result.data, result.size)
        
        # Free C result
        free_array_result(result)
        
        return py_result
    finally:
        free(c_embeddings)

def filter_split_indices(indices, values, double threshold, int min_distance):
    """
    Filter split indices by percentile threshold and minimum distance without NumPy.
    
    Args:
        indices: Candidate split indices (list)
        values: Values at those indices (list)
        threshold: Percentile threshold (0-1)
        min_distance: Minimum distance between splits
    
    Returns:
        Filtered indices as a list
    """
    # Convert inputs to lists if needed
    if not isinstance(indices, list):
        indices = list(indices)
    if not isinstance(values, list):
        values = list(values)
    
    if len(indices) != len(values):
        raise ValueError("indices and values must have the same length")
    
    if len(indices) == 0:
        return []
    
    cdef size_t n
    cdef int* c_indices = int_list_to_c_array(indices, &n)
    cdef double* c_values = list_to_c_array(values, &n)
    cdef MinimaResult* result
    
    try:
        # Call C function
        result = filter_split_indices_pure(
            c_indices, c_values, n, threshold, min_distance
        )
        if not result:
            raise ValueError("Invalid parameters for filter_split_indices")
        
        # Convert result to Python list (only indices needed)
        py_result = c_int_array_to_list(result.indices, result.count)
        
        # Free C result
        free_minima_result(result)
        
        return py_result
    finally:
        free(c_indices)
        free(c_values)

def get_cached_coeffs(int window_size, int poly_order, dtype=None):
    """
    Stub for compatibility - caching is handled internally in C.
    Returns None as coefficients are computed as needed.
    """
    return None
