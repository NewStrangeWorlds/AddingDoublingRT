/// @file cuda_matrix.cuh
/// @brief Device-side fixed-size dense matrix library for the CUDA RT solver.
///
/// Provides GpuMatrix<N> and GpuVec<N> with all matrix operations needed by the
/// adding-doubling algorithm, implemented as __device__ __forceinline__ functions.
/// Designed for small matrices (N=2..32) where per-thread execution is efficient.

#pragma once

#include <cmath>

namespace adrt {
namespace cuda {

// ============================================================================
//  Fixed-size vector (thread-local storage)
// ============================================================================

template<int N>
struct GpuVec {
  double data[N];

  __device__ __forceinline__ double& operator[](int i) { return data[i]; }
  __device__ __forceinline__ double  operator[](int i) const { return data[i]; }
};

// ============================================================================
//  Fixed-size square matrix (thread-local storage, row-major)
// ============================================================================

template<int N>
struct GpuMatrix {
  double data[N * N];

  __device__ __forceinline__ double& operator()(int i, int j) {
    return data[i * N + j];
  }
  __device__ __forceinline__ double operator()(int i, int j) const {
    return data[i * N + j];
  }
};


// ============================================================================
//  Vector operations
// ============================================================================

template<int N>
__device__ __forceinline__ void vec_set_zero(GpuVec<N>& v) {
  #pragma unroll
  for (int i = 0; i < N; ++i)
    v[i] = 0.0;
}

template<int N>
__device__ __forceinline__ void vec_set_scalar(GpuVec<N>& v, double s) {
  #pragma unroll
  for (int i = 0; i < N; ++i)
    v[i] = s;
}

template<int N>
__device__ __forceinline__ void vec_copy(GpuVec<N>& dst, const GpuVec<N>& src) {
  #pragma unroll
  for (int i = 0; i < N; ++i)
    dst[i] = src[i];
}

/// dst = a + b
template<int N>
__device__ __forceinline__ void vec_add(
    GpuVec<N>& dst, const GpuVec<N>& a, const GpuVec<N>& b) {
  #pragma unroll
  for (int i = 0; i < N; ++i)
    dst[i] = a[i] + b[i];
}

/// dst = a + alpha * b
template<int N>
__device__ __forceinline__ void vec_add_scaled(
    GpuVec<N>& dst, const GpuVec<N>& a, const GpuVec<N>& b, double alpha) {
  #pragma unroll
  for (int i = 0; i < N; ++i)
    dst[i] = a[i] + alpha * b[i];
}


// ============================================================================
//  Matrix initialisation
// ============================================================================

template<int N>
__device__ __forceinline__ void mat_set_zero(GpuMatrix<N>& A) {
  #pragma unroll
  for (int i = 0; i < N * N; ++i)
    A.data[i] = 0.0;
}

template<int N>
__device__ __forceinline__ void mat_set_identity(GpuMatrix<N>& A) {
  #pragma unroll
  for (int i = 0; i < N; ++i)
    #pragma unroll
    for (int j = 0; j < N; ++j)
      A(i, j) = (i == j) ? 1.0 : 0.0;
}

template<int N>
__device__ __forceinline__ void mat_set_diagonal(GpuMatrix<N>& A, const GpuVec<N>& v) {
  #pragma unroll
  for (int i = 0; i < N; ++i)
    #pragma unroll
    for (int j = 0; j < N; ++j)
      A(i, j) = (i == j) ? v[i] : 0.0;
}

template<int N>
__device__ __forceinline__ void mat_copy(GpuMatrix<N>& dst, const GpuMatrix<N>& src) {
  #pragma unroll
  for (int i = 0; i < N * N; ++i)
    dst.data[i] = src.data[i];
}


// ============================================================================
//  Matrix arithmetic
// ============================================================================

/// C = A * B
template<int N>
__device__ __forceinline__ void mat_multiply(
    GpuMatrix<N>& C, const GpuMatrix<N>& A, const GpuMatrix<N>& B) {
  #pragma unroll
  for (int i = 0; i < N; ++i) {
    #pragma unroll
    for (int j = 0; j < N; ++j) {
      double sum = 0.0;
      #pragma unroll
      for (int k = 0; k < N; ++k)
        sum += A(i, k) * B(k, j);
      C(i, j) = sum;
    }
  }
}

/// y = A * x
template<int N>
__device__ __forceinline__ void mat_vec_multiply(
    GpuVec<N>& y, const GpuMatrix<N>& A, const GpuVec<N>& x) {
  #pragma unroll
  for (int i = 0; i < N; ++i) {
    double sum = 0.0;
    #pragma unroll
    for (int j = 0; j < N; ++j)
      sum += A(i, j) * x[j];
    y[i] = sum;
  }
}

/// C = A + alpha * B
template<int N>
__device__ __forceinline__ void mat_add(
    GpuMatrix<N>& C, const GpuMatrix<N>& A, const GpuMatrix<N>& B,
    double alpha = 1.0) {
  #pragma unroll
  for (int i = 0; i < N * N; ++i)
    C.data[i] = A.data[i] + alpha * B.data[i];
}

/// A += alpha * B  (in-place)
template<int N>
__device__ __forceinline__ void mat_add_inplace(
    GpuMatrix<N>& A, const GpuMatrix<N>& B, double alpha = 1.0) {
  #pragma unroll
  for (int i = 0; i < N * N; ++i)
    A.data[i] += alpha * B.data[i];
}

/// C = alpha * A
template<int N>
__device__ __forceinline__ void mat_scale(
    GpuMatrix<N>& C, const GpuMatrix<N>& A, double alpha) {
  #pragma unroll
  for (int i = 0; i < N * N; ++i)
    C.data[i] = alpha * A.data[i];
}


// ============================================================================
//  LU decomposition with partial pivoting (in-place)
// ============================================================================
//
//  Factorises A into P*A = L*U, overwriting A with L\U (unit diagonal for L
//  is implicit). The pivot array records row swaps.

template<int N>
__device__ __forceinline__ void mat_lu_factor(
    GpuMatrix<N>& A, int pivot[N]) {
  #pragma unroll
  for (int i = 0; i < N; ++i)
    pivot[i] = i;

  for (int k = 0; k < N; ++k) {
    // Partial pivot: find row with largest |A(i,k)| for i >= k
    int max_row = k;
    double max_val = fabs(A(k, k));

    for (int i = k + 1; i < N; ++i) {
      double v = fabs(A(i, k));
      if (v > max_val) {
        max_val = v;
        max_row = i;
      }
    }

    // Swap rows k and max_row
    if (max_row != k) {
      int tmp_p = pivot[k];
      pivot[k] = pivot[max_row];
      pivot[max_row] = tmp_p;

      #pragma unroll
      for (int j = 0; j < N; ++j) {
        double tmp = A(k, j);
        A(k, j) = A(max_row, j);
        A(max_row, j) = tmp;
      }
    }

    // Eliminate below
    double inv_kk = (max_val > 1e-300) ? 1.0 / A(k, k) : 0.0;

    for (int i = k + 1; i < N; ++i) {
      A(i, k) *= inv_kk;
      #pragma unroll
      for (int j = k + 1; j < N; ++j)
        A(i, j) -= A(i, k) * A(k, j);
    }
  }
}


// ============================================================================
//  Solve A x = b via pre-factored LU
// ============================================================================

template<int N>
__device__ __forceinline__ void mat_lu_solve_vec(
    const GpuMatrix<N>& LU, const int pivot[N],
    GpuVec<N>& x, const GpuVec<N>& b) {
  // Apply pivot permutation: x[i] = b[pivot[i]]
  #pragma unroll
  for (int i = 0; i < N; ++i)
    x[i] = b[pivot[i]];

  // Forward substitution (L y = Pb)
  for (int i = 1; i < N; ++i)
    for (int j = 0; j < i; ++j)
      x[i] -= LU(i, j) * x[j];

  // Back substitution (U x = y)
  for (int i = N - 1; i >= 0; --i) {
    for (int j = i + 1; j < N; ++j)
      x[i] -= LU(i, j) * x[j];
    x[i] /= LU(i, i);
  }
}


// ============================================================================
//  High-level solve/inverse operations
// ============================================================================

/// Solve A x = b for x. Modifies a temporary copy of A.
template<int N>
__device__ __forceinline__ void mat_solve(
    GpuVec<N>& x, const GpuMatrix<N>& A, const GpuVec<N>& b) {
  GpuMatrix<N> LU;
  mat_copy<N>(LU, A);
  int pivot[N];
  mat_lu_factor<N>(LU, pivot);
  mat_lu_solve_vec<N>(LU, pivot, x, b);
}

/// Solve A X = B for matrix X (multiple RHS). X = A^{-1} B.
template<int N>
__device__ __forceinline__ void mat_solve_matrix(
    GpuMatrix<N>& X, const GpuMatrix<N>& A, const GpuMatrix<N>& B) {
  GpuMatrix<N> LU;
  mat_copy<N>(LU, A);
  int pivot[N];
  mat_lu_factor<N>(LU, pivot);

  // Solve for each column of B
  GpuVec<N> col_b, col_x;
  #pragma unroll
  for (int j = 0; j < N; ++j) {
    #pragma unroll
    for (int i = 0; i < N; ++i)
      col_b[i] = B(i, j);

    mat_lu_solve_vec<N>(LU, pivot, col_x, col_b);

    #pragma unroll
    for (int i = 0; i < N; ++i)
      X(i, j) = col_x[i];
  }
}

/// Solve X A = B for X, i.e. X = B A^{-1}.
/// Equivalent to solving A^T X^T = B^T.
template<int N>
__device__ __forceinline__ void mat_right_solve_matrix(
    GpuMatrix<N>& X, const GpuMatrix<N>& A, const GpuMatrix<N>& B) {
  // Transpose A
  GpuMatrix<N> AT;
  #pragma unroll
  for (int i = 0; i < N; ++i)
    #pragma unroll
    for (int j = 0; j < N; ++j)
      AT(i, j) = A(j, i);

  GpuMatrix<N> LU;
  mat_copy<N>(LU, AT);
  int pivot[N];
  mat_lu_factor<N>(LU, pivot);

  // Solve A^T * col(X^T, j) = col(B^T, j) for each column j
  // col(X^T, j) = row(X, j); col(B^T, j) = row(B, j)
  GpuVec<N> row_b, row_x;
  #pragma unroll
  for (int i = 0; i < N; ++i) {
    #pragma unroll
    for (int j = 0; j < N; ++j)
      row_b[j] = B(i, j);

    mat_lu_solve_vec<N>(LU, pivot, row_x, row_b);

    #pragma unroll
    for (int j = 0; j < N; ++j)
      X(i, j) = row_x[j];
  }
}

/// Compute A^{-1} in-place (result written to Ainv).
template<int N>
__device__ __forceinline__ void mat_inverse(
    GpuMatrix<N>& Ainv, const GpuMatrix<N>& A) {
  GpuMatrix<N> I;
  mat_set_identity<N>(I);
  mat_solve_matrix<N>(Ainv, A, I);
}


} // namespace cuda
} // namespace adrt
