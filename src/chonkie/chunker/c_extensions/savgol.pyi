import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Union, Literal

def savgol_filter(
    data: NDArray[Union[np.float32, np.float64]],
    window_length: int = 5,
    polyorder: int = 3,
    deriv: int = 0,
    use_float32: bool = False
) -> NDArray[Union[np.float32, np.float64]]: ...

def find_local_minima_interpolated(
    y: NDArray[Union[np.float32, np.float64]],
    window_size: int = 11,
    poly_order: int = 2,
    tolerance: float = 0.2,
    use_float32: bool = False
) -> Tuple[NDArray[np.int32], NDArray[Union[np.float32, np.float64]]]: ...

def windowed_cross_similarity(
    embeddings: NDArray[Union[np.float32, np.float64]],
    window_size: int
) -> NDArray[Union[np.float32, np.float64]]: ...

def filter_split_indices(
    indices: NDArray[np.int32],
    values: NDArray[Union[np.float32, np.float64]],
    threshold: float,
    min_distance: int
) -> NDArray[np.int32]: ...

def get_cached_coeffs(
    window_size: int,
    poly_order: int,
    dtype: type = np.float32
) -> NDArray[Union[np.float32, np.float64]]: ...
