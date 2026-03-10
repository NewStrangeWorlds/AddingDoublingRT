/// @file cuda_warp_matrix.cuh
/// @brief Warp-cooperative fixed-size matrix library for the CUDA RT solver.
///
/// Instead of one thread owning an entire N×N matrix (N² registers),
/// N threads cooperate: each thread owns one row (N registers per matrix)
/// and one element of each vector (1 register per vector).
/// Cross-row communication uses __shfl_sync with width=N.
///
/// Designed for N=8 and N=16 where the per-thread approach causes
/// excessive register spilling.

#pragma once

#include <cmath>

namespace adrt {
namespace cuda {

// ============================================================================
//  WarpRow: one row of a distributed N×N matrix
// ============================================================================

template<int N>
struct WarpRow {
  float data[N];

  __device__ __forceinline__ float& operator[](int j) { return data[j]; }
  __device__ __forceinline__ float  operator[](int j) const { return data[j]; }
};


// ============================================================================
//  Group mask: only includes the N threads in this thread's group
//  This is critical for correctness when groups within a warp may diverge.
// ============================================================================

template<int N>
__device__ __forceinline__ unsigned warp_group_mask() {
  unsigned lane = threadIdx.x & 31u;
  unsigned group = lane / N;
  return ((1u << N) - 1u) << (group * N);
}


// ============================================================================
//  Double-precision shuffle helpers
// ============================================================================

__device__ __forceinline__ double shfl_double(
    unsigned mask, double val, int src, int width)
{
  int lo = __shfl_sync(mask, __double2loint(val), src, width);
  int hi = __shfl_sync(mask, __double2hiint(val), src, width);
  return __hiloint2double(hi, lo);
}

__device__ __forceinline__ double shfl_xor_double(
    unsigned mask, double val, int lane_mask, int width)
{
  int lo = __shfl_xor_sync(mask, __double2loint(val), lane_mask, width);
  int hi = __shfl_xor_sync(mask, __double2hiint(val), lane_mask, width);
  return __hiloint2double(hi, lo);
}

__device__ __forceinline__ double shfl_down_double(
    unsigned mask, double val, int delta, int width)
{
  int lo = __shfl_down_sync(mask, __double2loint(val), delta, width);
  int hi = __shfl_down_sync(mask, __double2hiint(val), delta, width);
  return __hiloint2double(hi, lo);
}


// ============================================================================
//  Matrix initialisation
// ============================================================================

template<int N>
__device__ __forceinline__ void wmat_set_zero(WarpRow<N>& A) {
  #pragma unroll
  for (int j = 0; j < N; ++j)
    A[j] = 0.0f;
}

template<int N>
__device__ __forceinline__ void wmat_set_identity(WarpRow<N>& A, int row_id) {
  #pragma unroll
  for (int j = 0; j < N; ++j)
    A[j] = (row_id == j) ? 1.0f : 0.0f;
}

template<int N>
__device__ __forceinline__ void wmat_set_diagonal(
    WarpRow<N>& A, float diag_val, int row_id) {
  #pragma unroll
  for (int j = 0; j < N; ++j)
    A[j] = (row_id == j) ? diag_val : 0.0f;
}

template<int N>
__device__ __forceinline__ void wmat_copy(WarpRow<N>& dst, const WarpRow<N>& src) {
  #pragma unroll
  for (int j = 0; j < N; ++j)
    dst[j] = src[j];
}


// ============================================================================
//  Element-wise matrix arithmetic (no shuffles needed)
// ============================================================================

/// C = A + alpha * B
template<int N>
__device__ __forceinline__ void wmat_add(
    WarpRow<N>& C, const WarpRow<N>& A, const WarpRow<N>& B,
    float alpha = 1.0f) {
  #pragma unroll
  for (int j = 0; j < N; ++j)
    C[j] = A[j] + alpha * B[j];
}

/// A += alpha * B
template<int N>
__device__ __forceinline__ void wmat_add_inplace(
    WarpRow<N>& A, const WarpRow<N>& B, float alpha = 1.0f) {
  #pragma unroll
  for (int j = 0; j < N; ++j)
    A[j] += alpha * B[j];
}

/// C = alpha * A
template<int N>
__device__ __forceinline__ void wmat_scale(
    WarpRow<N>& C, const WarpRow<N>& A, float alpha) {
  #pragma unroll
  for (int j = 0; j < N; ++j)
    C[j] = alpha * A[j];
}


// ============================================================================
//  Matrix multiply: C = A * B  (N² shuffles + N² FMAs per thread)
// ============================================================================

template<int N>
__device__ __forceinline__ void wmat_multiply(
    WarpRow<N>& C, const WarpRow<N>& A, const WarpRow<N>& B) {
  unsigned gmask = warp_group_mask<N>();

  #pragma unroll
  for (int j = 0; j < N; ++j)
    C[j] = 0.0f;

  #pragma unroll
  for (int k = 0; k < N; ++k) {
    float a_ik = A[k];
    #pragma unroll
    for (int j = 0; j < N; ++j) {
      float b_kj = __shfl_sync(gmask, B[j], k, N);
      C[j] += a_ik * b_kj;
    }
  }
}


// ============================================================================
//  Matrix-vector multiply: returns y[row_id] = (A * x)[row_id]
//  x is distributed: thread i holds x[i]
// ============================================================================

template<int N>
__device__ __forceinline__ float wmat_vec_multiply(
    const WarpRow<N>& A, float x_elem) {
  unsigned gmask = warp_group_mask<N>();
  float sum = 0.0f;
  #pragma unroll
  for (int j = 0; j < N; ++j) {
    float xj = __shfl_sync(gmask, x_elem, j, N);
    sum += A[j] * xj;
  }
  return sum;
}


// ============================================================================
//  Vector reduction (sum across threads in a group)
// ============================================================================

template<int N>
__device__ __forceinline__ float wvec_reduce_sum(float val) {
  unsigned gmask = warp_group_mask<N>();
  #pragma unroll
  for (int offset = N / 2; offset > 0; offset >>= 1)
    val += __shfl_down_sync(gmask, val, offset, N);
  return val;  // result valid on lane 0 of each group
}


// ============================================================================
//  Distributed LU factorisation (double precision)
//  Each thread holds one row: dA[0..N-1]
//  my_pivot stores the original row index for this thread's row
// ============================================================================

template<int N>
__device__ __forceinline__ void wdmat_lu_factor(
    double dA[N], int& my_pivot, int row_id)
{
  unsigned gmask = warp_group_mask<N>();
  my_pivot = row_id;

  for (int k = 0; k < N; ++k) {
    // --- Step 1: Pivot search (find max |A(i,k)| for i >= k) ---
    double my_col_val = (row_id >= k) ? fabs(dA[k]) : 0.0;
    int my_candidate = row_id;

    // Pairwise reduction to find argmax
    #pragma unroll
    for (int offset = N / 2; offset > 0; offset >>= 1) {
      double other_val = shfl_xor_double(gmask, my_col_val, offset, N);
      int other_lane = __shfl_xor_sync(gmask, my_candidate, offset, N);
      if (other_val > my_col_val ||
          (other_val == my_col_val && other_lane < my_candidate)) {
        my_col_val = other_val;
        my_candidate = other_lane;
      }
    }
    int max_lane = my_candidate;

    // --- Step 2: Row swap between rows k and max_lane ---
    if (max_lane != k) {
      int pivot_k = __shfl_sync(gmask, my_pivot, k, N);
      int pivot_max = __shfl_sync(gmask, my_pivot, max_lane, N);
      if (row_id == k) my_pivot = pivot_max;
      if (row_id == max_lane) my_pivot = pivot_k;

      #pragma unroll
      for (int j = 0; j < N; ++j) {
        double val_k = shfl_double(gmask, dA[j], k, N);
        double val_max = shfl_double(gmask, dA[j], max_lane, N);
        if (row_id == k) dA[j] = val_max;
        if (row_id == max_lane) dA[j] = val_k;
      }
    }

    // --- Step 3: Broadcast pivot element and compute multiplier ---
    double akk = shfl_double(gmask, dA[k], k, N);
    double inv_kk = (fabs(akk) > 1e-35) ? 1.0 / akk : 0.0;

    // --- Step 4: Elimination ---
    // All threads must participate in shuffles; only row_id > k updates.
    double lik = 0.0;
    if (row_id > k) {
      dA[k] *= inv_kk;
      lik = dA[k];
    }

    #pragma unroll
    for (int j = 0; j < N; ++j) {
      double akj = shfl_double(gmask, dA[j], k, N);
      if (row_id > k && j > k)
        dA[j] -= lik * akj;
    }
  }
}


// ============================================================================
//  Distributed forward/back substitution (double precision)
//  Returns x[row_id]
// ============================================================================

template<int N>
__device__ __forceinline__ double wdmat_lu_solve_vec(
    const double dLU[N], int my_pivot, double b_elem, int row_id)
{
  unsigned gmask = warp_group_mask<N>();

  // Apply row permutation
  double x = shfl_double(gmask, b_elem, my_pivot, N);

  // Forward substitution: L * y = Pb
  for (int j = 0; j < N - 1; ++j) {
    double xj = shfl_double(gmask, x, j, N);
    if (row_id > j)
      x -= dLU[j] * xj;
  }

  // Back substitution: U * x = y
  for (int j = N - 1; j >= 0; --j) {
    if (row_id == j)
      x /= dLU[j];
    double xj = shfl_double(gmask, x, j, N);
    if (row_id < j)
      x -= dLU[j] * xj;
  }

  return x;
}


// ============================================================================
//  High-level solve: A * x = b (float interface, double LU internals)
//  Returns x[row_id]
// ============================================================================

template<int N>
__device__ __forceinline__ float wmat_solve(
    const WarpRow<N>& A, float b_elem, int row_id)
{
  double dA[N];
  #pragma unroll
  for (int j = 0; j < N; ++j)
    dA[j] = static_cast<double>(A[j]);

  int pivot;
  wdmat_lu_factor<N>(dA, pivot, row_id);

  double db = static_cast<double>(b_elem);
  double dx = wdmat_lu_solve_vec<N>(dA, pivot, db, row_id);

  return static_cast<float>(dx);
}


// ============================================================================
//  Solve A * X = B for matrix X (float interface, double LU internals)
//  Each thread holds row[row_id] of X and B
// ============================================================================

template<int N>
__device__ __forceinline__ void wmat_solve_matrix(
    WarpRow<N>& X_row, const WarpRow<N>& A_row, const WarpRow<N>& B_row,
    int row_id)
{
  double dA[N];
  #pragma unroll
  for (int j = 0; j < N; ++j)
    dA[j] = static_cast<double>(A_row[j]);

  int pivot;
  wdmat_lu_factor<N>(dA, pivot, row_id);

  #pragma unroll
  for (int c = 0; c < N; ++c) {
    double b_elem = static_cast<double>(B_row[c]);
    double x_elem = wdmat_lu_solve_vec<N>(dA, pivot, b_elem, row_id);
    X_row[c] = static_cast<float>(x_elem);
  }
}


// ============================================================================
//  Solve X * A = B for matrix X (right solve)
//  This is equivalent to A^T * X^T = B^T
//  Each thread holds row[row_id] of X and B
// ============================================================================

template<int N>
__device__ __forceinline__ void wmat_right_solve_matrix(
    WarpRow<N>& X_row, const WarpRow<N>& A_row, const WarpRow<N>& B_row,
    int row_id)
{
  unsigned gmask = warp_group_mask<N>();

  // Transpose A: AT(row_id, src) = A(src, row_id)
  double dAT[N];
  #pragma unroll
  for (int src = 0; src < N; ++src) {
    double at_val = 0.0;
    #pragma unroll
    for (int k = 0; k < N; ++k) {
      float a_src_k = __shfl_sync(gmask, A_row[k], src, N);
      if (row_id == k)
        at_val = static_cast<double>(a_src_k);
    }
    dAT[src] = at_val;
  }

  // Factor A^T
  int pivot;
  wdmat_lu_factor<N>(dAT, pivot, row_id);

  // Solve N systems
  #pragma unroll
  for (int c = 0; c < N; ++c) {
    // Scatter: thread k needs B(c, k)
    double b_elem = 0.0;
    #pragma unroll
    for (int k = 0; k < N; ++k) {
      float bval = __shfl_sync(gmask, B_row[k], c, N);
      if (row_id == k)
        b_elem = static_cast<double>(bval);
    }

    double x_elem = wdmat_lu_solve_vec<N>(dAT, pivot, b_elem, row_id);

    // Gather: thread c needs X(c, 0..N-1)
    #pragma unroll
    for (int k = 0; k < N; ++k) {
      float xval = __shfl_sync(gmask, static_cast<float>(x_elem), k, N);
      if (row_id == c)
        X_row[k] = xval;
    }
  }
}


} // namespace cuda
} // namespace adrt
