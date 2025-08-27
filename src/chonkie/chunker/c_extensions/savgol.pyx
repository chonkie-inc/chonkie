# distutils: language = c
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""
Self-contained Cython implementation of Savitzky-Golay filter.

This module provides optimized filtering and similarity operations without NumPy.
All C implementations are included directly in this file.
"""

import cython
from libc.stdlib cimport malloc, free, calloc, qsort
from libc.string cimport memcpy
from libc.math cimport fabs, pow, sqrt

# Structure definitions
cdef struct ArrayResult:
    double* data
    size_t size

cdef struct MinimaResult:
    int* indices
    double* values
    size_t count

# Memory management helpers
cdef ArrayResult* create_array_result(size_t size):
    cdef ArrayResult* result = <ArrayResult*>malloc(sizeof(ArrayResult))
    if not result:
        return NULL
    
    result.data = <double*>calloc(size, sizeof(double))
    if not result.data:
        free(result)
        return NULL
    result.size = size
    return result

cdef void free_array_result(ArrayResult* result):
    if result:
        free(result.data)
        free(result)

cdef MinimaResult* create_minima_result(size_t size):
    cdef MinimaResult* result = <MinimaResult*>malloc(sizeof(MinimaResult))
    if not result:
        return NULL
    
    if size > 0:
        result.indices = <int*>malloc(size * sizeof(int))
        result.values = <double*>malloc(size * sizeof(double))
        if not result.indices or not result.values:
            free(result.indices)
            free(result.values)
            free(result)
            return NULL
    else:
        result.indices = NULL
        result.values = NULL
    result.count = size
    return result

cdef void free_minima_result(MinimaResult* result):
    if result:
        free(result.indices)
        free(result.values)
        free(result)

# Matrix operations
cdef void matrix_multiply(const double* A, const double* B, double* C,
                         size_t m, size_t n, size_t p):
    cdef size_t i, j, k
    for i in range(m):
        for j in range(p):
            C[i * p + j] = 0
            for k in range(n):
                C[i * p + j] += A[i * n + k] * B[k * p + j]

cdef void matrix_transpose(const double* A, double* AT, size_t m, size_t n):
    cdef size_t i, j
    for i in range(m):
        for j in range(n):
            AT[j * m + i] = A[i * n + j]

cdef int matrix_inverse(double* A, double* A_inv, size_t n):
    cdef size_t i, j, k, max_row
    cdef double max_val, pivot, factor, temp
    
    # Create identity matrix in A_inv
    for i in range(n):
        for j in range(n):
            A_inv[i * n + j] = 1.0 if i == j else 0.0
    
    # Make a copy of A to work with
    cdef double* work = <double*>malloc(n * n * sizeof(double))
    memcpy(work, A, n * n * sizeof(double))
    
    # Gaussian elimination with partial pivoting
    for i in range(n):
        # Find pivot
        max_row = i
        max_val = fabs(work[i * n + i])
        for k in range(i + 1, n):
            if fabs(work[k * n + i]) > max_val:
                max_val = fabs(work[k * n + i])
                max_row = k
        
        # Swap rows if needed
        if max_row != i:
            for j in range(n):
                temp = work[i * n + j]
                work[i * n + j] = work[max_row * n + j]
                work[max_row * n + j] = temp
                temp = A_inv[i * n + j]
                A_inv[i * n + j] = A_inv[max_row * n + j]
                A_inv[max_row * n + j] = temp
        
        # Check for singular matrix
        pivot = work[i * n + i]
        if fabs(pivot) < 1e-10:
            free(work)
            return 0
        
        # Normalize pivot row
        for j in range(n):
            work[i * n + j] /= pivot
            A_inv[i * n + j] /= pivot
        
        # Eliminate column
        for k in range(n):
            if k != i:
                factor = work[k * n + i]
                for j in range(n):
                    work[k * n + j] -= factor * work[i * n + j]
                    A_inv[k * n + j] -= factor * A_inv[i * n + j]
    
    free(work)
    return 1

# Compute Savitzky-Golay coefficients
cdef double* compute_savgol_coeffs(int window_size, int poly_order, int deriv):
    cdef int half_window = (window_size - 1) // 2
    cdef size_t mat_size = window_size * (poly_order + 1)
    cdef double* A = <double*>malloc(mat_size * sizeof(double))
    cdef double* AT = <double*>malloc((poly_order + 1) * window_size * sizeof(double))
    cdef double* ATA = <double*>malloc((poly_order + 1) * (poly_order + 1) * sizeof(double))
    cdef double* ATA_inv = <double*>malloc((poly_order + 1) * (poly_order + 1) * sizeof(double))
    cdef double* coeffs = <double*>malloc(window_size * sizeof(double))
    
    if not A or not AT or not ATA or not ATA_inv or not coeffs:
        free(A)
        free(AT)
        free(ATA)
        free(ATA_inv)
        free(coeffs)
        return NULL
    
    # Build Vandermonde matrix
    cdef int i, j
    cdef double val
    for i in range(window_size):
        val = i - half_window
        for j in range(poly_order + 1):
            A[i * (poly_order + 1) + j] = pow(val, j)
    
    # Compute A^T * A
    matrix_transpose(A, AT, window_size, poly_order + 1)
    matrix_multiply(AT, A, ATA, poly_order + 1, window_size, poly_order + 1)
    
    # Invert (A^T * A)
    if not matrix_inverse(ATA, ATA_inv, poly_order + 1):
        free(A)
        free(AT)
        free(ATA)
        free(ATA_inv)
        free(coeffs)
        return NULL
    
    # Compute coefficients for the requested derivative
    cdef double factorial = 1.0
    for i in range(1, deriv + 1):
        factorial *= i
    
    # Extract the appropriate row for the derivative
    cdef double sum_val
    cdef int k
    for i in range(window_size):
        coeffs[i] = 0
        for j in range(poly_order + 1):
            if j == deriv:
                sum_val = 0
                for k in range(poly_order + 1):
                    sum_val += ATA_inv[deriv * (poly_order + 1) + k] * A[i * (poly_order + 1) + k]
                coeffs[i] = factorial * sum_val
    
    free(A)
    free(AT)
    free(ATA)
    free(ATA_inv)
    
    return coeffs

# Apply convolution
cdef void apply_convolution(const double* data, size_t n, const double* kernel,
                           size_t kernel_size, double* output):
    cdef int half = kernel_size // 2
    cdef size_t i, j
    cdef int idx
    cdef double sum_val
    
    for i in range(n):
        sum_val = 0
        for j in range(kernel_size):
            idx = <int>i - half + <int>j
            # Handle boundaries with reflection
            if idx < 0:
                idx = -idx
            elif idx >= <int>n:
                idx = 2 * <int>n - idx - 2
            sum_val += data[idx] * kernel[j]
        output[i] = sum_val

# Dot product
cdef double dot_product(const double* a, const double* b, size_t n):
    cdef double result = 0
    cdef size_t i
    for i in range(n):
        result += a[i] * b[i]
    return result

# Comparison function for qsort
cdef int compare_doubles(const void* a, const void* b) noexcept nogil:
    cdef double val_a = (<double*>a)[0]
    cdef double val_b = (<double*>b)[0]
    if val_a < val_b:
        return -1
    elif val_a > val_b:
        return 1
    else:
        return 0

# Percentile calculation
cdef double percentile(const double* data, size_t n, double p):
    if n == 0:
        return 0
    
    # Create sorted copy
    cdef double* sorted_data = <double*>malloc(n * sizeof(double))
    memcpy(sorted_data, data, n * sizeof(double))
    
    # Use qsort for efficient sorting
    qsort(sorted_data, n, sizeof(double), compare_doubles)
    
    cdef double idx = p * (n - 1)
    cdef size_t lower = <size_t>idx
    cdef size_t upper = lower + 1 if lower < n - 1 else lower
    cdef double weight = idx - lower
    cdef double result = sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight
    
    free(sorted_data)
    return result

# Main Savitzky-Golay filter implementation
cdef ArrayResult* savgol_filter_pure(const double* data, size_t n,
                                    int window_length, int polyorder, int deriv):
    if window_length % 2 == 0 or window_length <= polyorder:
        return NULL
    
    cdef ArrayResult* result = create_array_result(n)
    if not result:
        return NULL
    
    cdef double* coeffs = compute_savgol_coeffs(window_length, polyorder, deriv)
    if not coeffs:
        free_array_result(result)
        return NULL
    
    apply_convolution(data, n, coeffs, window_length, result.data)
    
    free(coeffs)
    return result

# Find local minima with interpolation
cdef MinimaResult* find_local_minima_interpolated_pure(const double* data, size_t n,
                                                      int window_size, int poly_order,
                                                      double tolerance):
    # Get first and second derivatives
    cdef ArrayResult* first_deriv = savgol_filter_pure(data, n, window_size, poly_order, 1)
    cdef ArrayResult* second_deriv = savgol_filter_pure(data, n, window_size, poly_order, 2)
    
    if not first_deriv or not second_deriv:
        free_array_result(first_deriv)
        free_array_result(second_deriv)
        return NULL
    
    # Count minima
    cdef size_t count = 0
    cdef size_t i
    for i in range(n):
        if fabs(first_deriv.data[i]) < tolerance and second_deriv.data[i] > 0:
            count += 1
    
    # Create result
    cdef MinimaResult* result = create_minima_result(count)
    if not result:
        free_array_result(first_deriv)
        free_array_result(second_deriv)
        return NULL
    
    # Find minima
    cdef size_t j = 0
    for i in range(n):
        if fabs(first_deriv.data[i]) < tolerance and second_deriv.data[i] > 0:
            result.indices[j] = <int>i
            result.values[j] = data[i]
            j += 1
    
    free_array_result(first_deriv)
    free_array_result(second_deriv)
    return result

# Windowed cross similarity
cdef ArrayResult* windowed_cross_similarity_pure(const double* embeddings, size_t n, size_t d,
                                                int window_size):
    if window_size % 2 == 0 or window_size < 3:
        return NULL
    
    cdef ArrayResult* result = create_array_result(n - 1)
    if not result:
        return NULL
    
    cdef int half_window = window_size // 2
    cdef size_t i, j, k
    cdef int start, end, win_idx
    cdef double sim, norm1, norm2, dot
    cdef int count
    
    for i in range(n - 1):
        # Define window boundaries
        start = <int>i - half_window
        if start < 0:
            start = 0
        end = <int>i + half_window + 2
        if end > <int>n:
            end = <int>n
        
        # Calculate average similarity in window
        sim = 0
        count = 0
        
        for j in range(<size_t>start, <size_t>end - 1):
            # Compute cosine similarity between consecutive embeddings
            norm1 = 0
            norm2 = 0
            dot = 0
            
            for k in range(d):
                dot += embeddings[j * d + k] * embeddings[(j + 1) * d + k]
                norm1 += embeddings[j * d + k] * embeddings[j * d + k]
                norm2 += embeddings[(j + 1) * d + k] * embeddings[(j + 1) * d + k]
            
            if norm1 > 0 and norm2 > 0:
                sim += dot / (sqrt(norm1) * sqrt(norm2))
                count += 1
        
        if count > 0:
            result.data[i] = sim / count
        else:
            result.data[i] = 0
    
    return result

# Filter split indices
cdef MinimaResult* filter_split_indices_pure(const int* indices, const double* values,
                                            size_t n_indices, double threshold,
                                            int min_distance):
    if n_indices == 0:
        return create_minima_result(0)
    
    # Calculate threshold value
    cdef double threshold_val = percentile(values, n_indices, threshold)
    
    # Count valid splits
    cdef size_t count = 0
    cdef size_t i
    cdef int last_idx = -min_distance - 1
    
    for i in range(n_indices):
        if values[i] <= threshold_val and indices[i] - last_idx >= min_distance:
            count += 1
            last_idx = indices[i]
    
    # Create result
    cdef MinimaResult* result = create_minima_result(count)
    if not result:
        return NULL
    
    # Fill result
    cdef size_t j = 0
    last_idx = -min_distance - 1
    
    for i in range(n_indices):
        if values[i] <= threshold_val and indices[i] - last_idx >= min_distance:
            result.indices[j] = indices[i]
            result.values[j] = values[i]
            j += 1
            last_idx = indices[i]
    
    return result

# Python interface functions

# Helper to convert Python list to C array
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double* list_to_c_array(list data, size_t* length):
    """Convert Python list to C double array."""
    cdef size_t n = len(data)
    cdef double* c_array = <double*>malloc(n * sizeof(double))
    if not c_array:
        return NULL
    
    cdef size_t i
    for i in range(n):
        c_array[i] = float(data[i])
    
    length[0] = n
    return c_array

# Helper to convert C array to Python list
@cython.boundscheck(False)
@cython.wraparound(False)
cdef list c_array_to_list(const double* array, size_t n):
    """Convert C array to Python list."""
    cdef list result = []
    cdef size_t i
    for i in range(n):
        result.append(array[i])
    return result

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
    
    cdef size_t n, i
    cdef double* c_data = list_to_c_array(data, &n)
    cdef MinimaResult* result
    
    try:
        # Call C function
        result = find_local_minima_interpolated_pure(c_data, n, window_size, poly_order, tolerance)
        if not result:
            return [], []
        
        # Convert results to Python lists
        indices = []
        values = []
        for i in range(result.count):
            indices.append(result.indices[i])
            values.append(result.values[i])
        
        # Free C result
        free_minima_result(result)
        
        return indices, values
    finally:
        free(c_data)

def windowed_cross_similarity(embeddings, int window_size=3, bint use_float32=False):
    """
    Compute windowed cross-similarity for semantic chunking without NumPy.
    
    Args:
        embeddings: 2D array of embeddings (n_sentences x embedding_dim)
        window_size: Size of sliding window (must be odd and >= 3)
        use_float32: Ignored (kept for compatibility)
    
    Returns:
        Average similarities for each position as a list
    """
    # Convert embeddings to flat list
    cdef list flat_embeddings = []
    cdef size_t n = len(embeddings)
    cdef size_t d = len(embeddings[0]) if n > 0 else 0
    
    for emb in embeddings:
        for val in emb:
            flat_embeddings.append(float(val))
    
    cdef size_t total_size = n * d
    cdef double* c_embeddings = <double*>malloc(total_size * sizeof(double))
    if not c_embeddings:
        raise MemoryError("Failed to allocate memory for embeddings")
    
    cdef size_t i
    for i in range(total_size):
        c_embeddings[i] = flat_embeddings[i]
    
    cdef ArrayResult* result
    try:
        # Call C function
        result = windowed_cross_similarity_pure(c_embeddings, n, d, window_size)
        if not result:
            raise ValueError("Invalid parameters for windowed cross-similarity")
        
        # Convert result to Python list
        py_result = c_array_to_list(result.data, result.size)
        
        # Free C result
        free_array_result(result)
        
        return py_result
    finally:
        free(c_embeddings)

def filter_split_indices(indices, values, double threshold=0.5, int min_distance=2):
    """
    Filter split indices by percentile threshold and minimum distance.
    
    Args:
        indices: Candidate split indices
        values: Values at those indices
        threshold: Percentile threshold (0-1)
        min_distance: Minimum distance between splits
    
    Returns:
        Tuple of (filtered_indices, filtered_values) as lists
    """
    if not indices or not values:
        return [], []
    
    cdef size_t n = len(indices)
    cdef int* c_indices = <int*>malloc(n * sizeof(int))
    cdef double* c_values = <double*>malloc(n * sizeof(double))
    
    if not c_indices or not c_values:
        free(c_indices)
        free(c_values)
        raise MemoryError("Failed to allocate memory")
    
    cdef size_t i
    for i in range(n):
        c_indices[i] = int(indices[i])
        c_values[i] = float(values[i])
    
    cdef MinimaResult* result
    try:
        # Call C function
        result = filter_split_indices_pure(c_indices, c_values, n, threshold, min_distance)
        if not result:
            return [], []
        
        # Convert results to Python lists
        filtered_indices = []
        filtered_values = []
        for i in range(result.count):
            filtered_indices.append(result.indices[i])
            filtered_values.append(result.values[i])
        
        # Free C result
        free_minima_result(result)
        
        return filtered_indices, filtered_values
    finally:
        free(c_indices)
        free(c_values)