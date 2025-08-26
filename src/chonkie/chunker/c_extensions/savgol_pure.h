#ifndef SAVGOL_PURE_H
#define SAVGOL_PURE_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Result structure for functions that return arrays
typedef struct {
    double* data;
    size_t size;
} ArrayResult;

// Result structure for minima detection (returns indices and values)
typedef struct {
    int* indices;
    double* values;
    size_t count;
} MinimaResult;

// Memory management helpers
ArrayResult* create_array_result(size_t size);
void free_array_result(ArrayResult* result);
MinimaResult* create_minima_result(size_t size);
void free_minima_result(MinimaResult* result);

// Core Savitzky-Golay functions

/**
 * Apply Savitzky-Golay filter to data
 * @param data Input data array
 * @param n Length of input data
 * @param window_length Length of the filter window (must be odd and > polyorder)
 * @param polyorder Order of the polynomial
 * @param deriv Derivative order (0=smoothing, 1=first, 2=second)
 * @return Filtered data (caller must free)
 */
ArrayResult* savgol_filter_pure(const double* data, size_t n,
                                int window_length, int polyorder, int deriv);

/**
 * Find local minima with sub-sample accuracy using zero-crossing interpolation
 * @param data Input data array
 * @param n Length of input data
 * @param window_size Savitzky-Golay window size
 * @param poly_order Polynomial order
 * @param tolerance Tolerance for considering derivative as zero
 * @return Minima indices and values (caller must free)
 */
MinimaResult* find_local_minima_interpolated_pure(const double* data, size_t n,
                                                  int window_size, int poly_order,
                                                  double tolerance);

/**
 * Compute windowed cross-similarity for semantic chunking
 * @param embeddings Flattened n×d matrix of embeddings (row-major)
 * @param n Number of embeddings
 * @param d Dimension of each embedding
 * @param window_size Size of sliding window (must be odd and >= 3)
 * @return Array of average similarities for each position (caller must free)
 */
ArrayResult* windowed_cross_similarity_pure(const double* embeddings, size_t n, size_t d,
                                           int window_size);

/**
 * Filter split indices by percentile threshold and minimum distance
 * @param indices Candidate split indices
 * @param values Values at those indices  
 * @param n_indices Number of indices
 * @param threshold Percentile threshold (0-1)
 * @param min_distance Minimum distance between splits
 * @return Filtered indices (caller must free)
 */
MinimaResult* filter_split_indices_pure(const int* indices, const double* values,
                                       size_t n_indices, double threshold,
                                       int min_distance);

// Helper math functions
double* compute_savgol_coeffs(int window_size, int poly_order, int deriv);
void apply_convolution(const double* data, size_t n, const double* kernel, 
                      size_t kernel_size, double* output);
double dot_product(const double* a, const double* b, size_t n);
double percentile(const double* data, size_t n, double p);

#ifdef __cplusplus
}
#endif

#endif // SAVGOL_PURE_H
