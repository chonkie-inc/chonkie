"""Tests for the peak detection module."""

import numpy as np
import pytest
from chonkie.chunker.peak_detection import PeakDetector


def test_peak_detector_initialization():
    """Test PeakDetector initialization with valid and invalid parameters."""
    # Test valid initialization
    detector = PeakDetector(window_length=5, polyorder=2)
    assert detector.window_length == 5
    assert detector.polyorder == 2
    assert detector.threshold is None

    # Test invalid window_length
    with pytest.raises(ValueError, match="window_length must be odd"):
        PeakDetector(window_length=4)

    # Test invalid polyorder
    with pytest.raises(ValueError, match="polyorder must be less than window_length"):
        PeakDetector(window_length=5, polyorder=5)


def test_smooth_signal():
    """Test signal smoothing functionality."""
    detector = PeakDetector(window_length=5, polyorder=2)
    
    # Create a noisy sine wave
    x = np.linspace(0, 10, 100)
    signal = np.sin(x) + np.random.normal(0, 0.1, 100)
    
    # Smooth the signal
    smoothed = detector.smooth_signal(signal)
    
    # Check that smoothing worked
    assert len(smoothed) == len(signal)
    assert np.all(np.isfinite(smoothed))
    assert np.std(smoothed) < np.std(signal)  # Smoothed signal should have less variance


def test_compute_derivatives():
    """Test derivative computation."""
    detector = PeakDetector(window_length=5, polyorder=2)
    
    # Create a simple signal
    x = np.linspace(0, 10, 100)
    signal = np.sin(x)
    
    # Compute derivatives
    first_deriv, second_deriv = detector.compute_derivatives(signal)
    
    # Check derivatives
    assert len(first_deriv) == len(signal)
    assert len(second_deriv) == len(signal)
    assert np.all(np.isfinite(first_deriv))
    assert np.all(np.isfinite(second_deriv))


def test_find_local_minima():
    """Test local minima detection (now uses find_extrema for minima)."""
    detector = PeakDetector(window_length=5, polyorder=2, find='minima')
    
    # Create a signal with known minima
    x = np.linspace(0, 10, 100)
    signal = np.sin(x)
    
    # Compute derivatives
    first_deriv, second_deriv = detector.compute_derivatives(signal)
    
    # Find minima (using new method)
    minima = detector.find_extrema(signal, first_deriv, second_deriv)
    
    # Check that we found minima
    assert len(minima) > 0
    for idx in minima:
        # Check that we're at a local minimum
        assert second_deriv[idx] > 0


def test_detect_peaks():
    """Test complete peak detection pipeline for maxima."""
    detector = PeakDetector(window_length=5, polyorder=2, find='maxima')
    
    # Create a signal with known peaks
    x = np.linspace(0, 10, 100)
    signal = np.sin(x)
    
    # Detect peaks
    peaks = detector.detect_peaks(signal)
    
    # Check that we found peaks
    assert len(peaks) > 0
    
    # Empirically, for this configuration, maxima are detected at indices 15 and 77
    expected_peaks = np.array([15, 77])
    for expected in expected_peaks:
        assert any(abs(peak - expected) < 3 for peak in peaks), f"No detected peak near expected index {expected} (detected: {peaks})"


def test_peak_detection_with_threshold():
    """Test peak detection with threshold."""
    detector = PeakDetector(window_length=5, polyorder=2, threshold=0.5)
    
    # Create a signal with varying amplitudes
    x = np.linspace(0, 10, 100)
    signal = np.sin(x) * np.exp(-x/10)  # Decaying sine wave
    
    # Detect peaks
    peaks = detector.detect_peaks(signal)
    
    # Check that peaks respect the threshold
    for peak in peaks:
        assert signal[peak] <= 0.5 