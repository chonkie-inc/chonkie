from typing import Tuple, List, Union, Optional, Any

def savgol_filter(
    data: Union[List[float], Any],
    window_length: int = 5,
    polyorder: int = 3,
    deriv: int = 0,
    use_float32: bool = False
) -> List[float]: 
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
    ...

def find_local_minima_interpolated(
    data: Union[List[float], Any],
    window_size: int = 11,
    poly_order: int = 2,
    tolerance: float = 0.2,
    use_float32: bool = False
) -> Tuple[List[int], List[float]]: 
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
    ...

def windowed_cross_similarity(
    embeddings: List[List[float]],
    window_size: int
) -> List[float]: 
    """
    Compute windowed cross-similarity without NumPy.
    
    Args:
        embeddings: List of embedding vectors (list of lists)
        window_size: Size of sliding window (must be odd and >= 3)
    
    Returns:
        List of average similarities for each position
    """
    ...

def filter_split_indices(
    indices: Union[List[int], Any],
    values: Union[List[float], Any],
    threshold: float,
    min_distance: int
) -> List[int]: 
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
    ...

def get_cached_coeffs(
    window_size: int,
    poly_order: int,
    dtype: Optional[type] = None
) -> None: 
    """
    Stub for compatibility - caching is handled internally in C.
    Returns None as coefficients are computed as needed.
    """
    ...
