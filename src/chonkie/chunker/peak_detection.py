"""Peak detection module for semantic chunking.

This module implements peak detection algorithms for finding optimal chunk boundaries
in semantic text chunking, similar to WordLlama's approach.
"""

from typing import List, Optional, Tuple
import numpy as np
from scipy.signal import savgol_filter


class PeakDetector:
    """Peak detection for semantic chunking using smoothing and derivatives.
    
    This class implements peak detection algorithms that use smoothing filters
    and derivative calculations to find optimal chunk boundaries in semantic text.
    
    Args:
        window_length: Length of the smoothing window (must be odd)
        polyorder: Order of the polynomial for Savitzky-Golay filter
        threshold: Threshold for peak detection (optional)
        find: 'minima' (default) or 'maxima' to control which extrema to detect
    """
    
    def __init__(
        self,
        window_length: int = 5,
        polyorder: int = 2,
        threshold: Optional[float] = None,
        find: str = 'minima',
    ) -> None:
        """Initialize the PeakDetector.
        
        Args:
            window_length: Length of the smoothing window (must be odd)
            polyorder: Order of the polynomial for Savitzky-Golay filter
            threshold: Threshold for peak detection (optional)
            find: 'minima' (default) or 'maxima' to control which extrema to detect
        
        Raises:
            ValueError: If window_length is even or polyorder >= window_length
        """
        if window_length % 2 == 0:
            raise ValueError("window_length must be odd")
        if polyorder >= window_length:
            raise ValueError("polyorder must be less than window_length")
        if find not in ('minima', 'maxima'):
            raise ValueError("find must be 'minima' or 'maxima'")
        
        self.window_length = window_length
        self.polyorder = polyorder
        self.threshold = threshold
        self.find = find

    def smooth_signal(self, signal: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay filter to smooth the signal.
        
        Args:
            signal: Input signal array
            
        Returns:
            Smoothed signal array
        """
        return savgol_filter(signal, self.window_length, self.polyorder)

    def compute_derivatives(
        self, signal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute first and second derivatives of the signal.
        
        Args:
            signal: Input signal array
            
        Returns:
            Tuple of (first_derivative, second_derivative) arrays
        """
        # First derivative
        first_deriv = savgol_filter(
            signal, self.window_length, self.polyorder, deriv=1
        )
        
        # Second derivative
        second_deriv = savgol_filter(
            signal, self.window_length, self.polyorder, deriv=2
        )
        
        return first_deriv, second_deriv

    def find_extrema(
        self,
        signal: np.ndarray,
        first_deriv: np.ndarray,
        second_deriv: np.ndarray
    ) -> List[int]:
        """Find local minima or maxima in the signal using derivatives.
        
        Args:
            signal: Input signal array
            first_deriv: First derivative array
            second_deriv: Second derivative array
        
        Returns:
            List of indices where local minima or maxima occur
        """
        # Find zero crossings in first derivative
        zero_crossings = np.where(np.diff(np.signbit(first_deriv)))[0]
        
        if self.find == 'minima':
            # Find indices where second derivative is positive at zero crossings
            # A positive second derivative at a zero crossing indicates a local minimum:
            # - First derivative (slope) changes from negative to positive
            # - Second derivative > 0 means the curve is concave up at this point
            # - This combination identifies points where the function reaches a local minimum
            minima_indices = [
                idx for idx in zero_crossings
                if second_deriv[idx] > 0
            ]
            # Apply threshold if specified (minima: signal <= threshold)
            if self.threshold is not None:
                minima_indices = [
                    idx for idx in minima_indices
                    if signal[idx] <= self.threshold
                ]
        else:
            # Maxima: negative second derivative
            extrema = [
                idx for idx in zero_crossings
                if second_deriv[idx] < 0
            ]
            # Apply threshold if specified (maxima: signal >= threshold)
            if self.threshold is not None:
                extrema = [
                    idx for idx in extrema
                    if signal[idx] >= self.threshold
                ]
        return extrema

    def detect_peaks(self, signal: np.ndarray) -> List[int]:
        """Detect peaks (minima or maxima) in the signal using smoothing and derivatives.
        
        Args:
            signal: Input signal array
        
        Returns:
            List of indices where peaks (minima or maxima) occur
        """
        # Smooth the signal
        smoothed = self.smooth_signal(signal)
        
        # Compute derivatives
        first_deriv, second_deriv = self.compute_derivatives(smoothed)
        
        # Find extrema
        peaks = self.find_extrema(smoothed, first_deriv, second_deriv)
        
        return peaks 