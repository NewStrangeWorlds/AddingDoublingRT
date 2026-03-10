/// @file cuda_matrix.cuh
/// @brief Device-side fixed-size dense matrix library for the CUDA RT solver.
///
/// Provides GpuMatrix<N> (float) and GpuVec<N> (float) for the main solver,
/// with double-precision LU solves for numerical stability.
/// Designed for small matrices (N=2..32) where per-thread execution is efficient.

#pragma once

#include <cmath>

namespace adrt {
namespace cuda {

// ============================================================================
//  Fixed-size vector (thread-local, float storage)
// ============================================================================

template<int N>
struct GpuVec {
  float data[N];

  __device__ __forceinline__ float& operator[](int i) { return data[i]; }
  __device__ __forceinline__ float  operator[](int i) const { return data[i]; }
};

// ============================================================================
//  Fixed-size square matrix (thread-local, float storage, row-major)
// ============================================================================

template<int N>
struct GpuMatrix {
  float data[N * N];

  __device__ __forceinline__ float& operator()(int i, int j) {
    return data[i * N + j];
  }
  __device__ __forceinline__ float operator()(int i, int j) const {
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
    v[i] = 0.0f;
}

template<int N>
__device__ __forceinline__ void vec_set_scalar(GpuVec<N>& v, float s) {
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
    GpuVec<N>& dst, const GpuVec<N>& a, const GpuVec<N>& b, float alpha) {
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
    A.data[i] = 0.0f;
}

template<int N>
__device__ __forceinline__ void mat_set_identity(GpuMatrix<N>& A) {
  #pragma unroll
  for (int i = 0; i < N; ++i)
    #pragma unroll
    for (int j = 0; j < N; ++j)
      A(i, j) = (i == j) ? 1.0f : 0.0f;
}

template<int N>
__device__ __forceinline__ void mat_set_diagonal(GpuMatrix<N>& A, const GpuVec<N>& v) {
  #pragma unroll
  for (int i = 0; i < N; ++i)
    #pragma unroll
    for (int j = 0; j < N; ++j)
      A(i, j) = (i == j) ? v[i] : 0.0f;
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
      float sum = 0.0f;
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
    float sum = 0.0f;
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
    float alpha = 1.0f) {
  #pragma unroll
  for (int i = 0; i < N * N; ++i)
    C.data[i] = A.data[i] + alpha * B.data[i];
}

/// A += alpha * B  (in-place)
template<int N>
__device__ __forceinline__ void mat_add_inplace(
    GpuMatrix<N>& A, const GpuMatrix<N>& B, float alpha = 1.0f) {
  #pragma unroll
  for (int i = 0; i < N * N; ++i)
    A.data[i] += alpha * B.data[i];
}

/// C += A * B  (C must not alias A or B; uses N-element row buffer instead of N² temp)
template<int N>
__device__ __forceinline__ void mat_multiply_addto(
    GpuMatrix<N>& C, const GpuMatrix<N>& A, const GpuMatrix<N>& B) {
  #pragma unroll
  for (int i = 0; i < N; ++i) {
    #pragma unroll
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      #pragma unroll
      for (int k = 0; k < N; ++k)
        sum += A(i, k) * B(k, j);
      C(i, j) += sum;
    }
  }
}

/// C = A * B  where C may alias A (NOT B). Computes row-by-row with an N-element buffer.
template<int N>
__device__ __forceinline__ void mat_multiply_into(
    GpuMatrix<N>& C, const GpuMatrix<N>& A, const GpuMatrix<N>& B) {
  #pragma unroll
  for (int i = 0; i < N; ++i) {
    float row[N];
    #pragma unroll
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      #pragma unroll
      for (int k = 0; k < N; ++k)
        sum += A(i, k) * B(k, j);
      row[j] = sum;
    }
    #pragma unroll
    for (int j = 0; j < N; ++j)
      C(i, j) = row[j];
  }
}

/// C = alpha * A
template<int N>
__device__ __forceinline__ void mat_scale(
    GpuMatrix<N>& C, const GpuMatrix<N>& A, float alpha) {
  #pragma unroll
  for (int i = 0; i < N * N; ++i)
    C.data[i] = alpha * A.data[i];
}


// ============================================================================
//  Double-precision LU decomposition (for numerical stability)
// ============================================================================
//
//  The matrix solve operations (I - R²)⁻¹ etc. are ill-conditioned for
//  high albedo, so we promote to double for the factorisation/solve only.

/// Double-precision LU factorisation with partial pivoting.
template<int N>
__device__ __forceinline__ void dmat_lu_factor(
    double A[N * N], int pivot[N]) {
  #pragma unroll
  for (int i = 0; i < N; ++i)
    pivot[i] = i;

  for (int k = 0; k < N; ++k) {
    int max_row = k;
    double max_val = fabs(A[k * N + k]);

    for (int i = k + 1; i < N; ++i) {
      double v = fabs(A[i * N + k]);
      if (v > max_val) {
        max_val = v;
        max_row = i;
      }
    }

    if (max_row != k) {
      int tmp_p = pivot[k];
      pivot[k] = pivot[max_row];
      pivot[max_row] = tmp_p;

      #pragma unroll
      for (int j = 0; j < N; ++j) {
        double tmp = A[k * N + j];
        A[k * N + j] = A[max_row * N + j];
        A[max_row * N + j] = tmp;
      }
    }

    double inv_kk = (max_val > 1e-35) ? 1.0 / A[k * N + k] : 0.0;

    for (int i = k + 1; i < N; ++i) {
      A[i * N + k] *= inv_kk;
      #pragma unroll
      for (int j = k + 1; j < N; ++j)
        A[i * N + j] -= A[i * N + k] * A[k * N + j];
    }
  }
}

/// Double-precision forward/back solve using pre-factored LU.
template<int N>
__device__ __forceinline__ void dmat_lu_solve_vec(
    const double LU[N * N], const int pivot[N],
    double x[N], const double b[N]) {
  #pragma unroll
  for (int i = 0; i < N; ++i)
    x[i] = b[pivot[i]];

  for (int i = 1; i < N; ++i)
    for (int j = 0; j < i; ++j)
      x[i] -= LU[i * N + j] * x[j];

  for (int i = N - 1; i >= 0; --i) {
    for (int j = i + 1; j < N; ++j)
      x[i] -= LU[i * N + j] * x[j];
    x[i] /= LU[i * N + i];
  }
}


// ============================================================================
//  High-level solve operations (float interface, double internals)
// ============================================================================

/// Solve A x = b for x. A and b are float; solve done in double.
template<int N>
__device__ __forceinline__ void mat_solve(
    GpuVec<N>& x, const GpuMatrix<N>& A, const GpuVec<N>& b) {
  double dA[N * N], db[N], dx[N];
  int pivot[N];

  #pragma unroll
  for (int i = 0; i < N * N; ++i)
    dA[i] = static_cast<double>(A.data[i]);
  #pragma unroll
  for (int i = 0; i < N; ++i)
    db[i] = static_cast<double>(b[i]);

  dmat_lu_factor<N>(dA, pivot);
  dmat_lu_solve_vec<N>(dA, pivot, dx, db);

  #pragma unroll
  for (int i = 0; i < N; ++i)
    x[i] = static_cast<float>(dx[i]);
}

/// Solve A X = B for matrix X. Float interface, double internals.
template<int N>
__device__ __forceinline__ void mat_solve_matrix(
    GpuMatrix<N>& X, const GpuMatrix<N>& A, const GpuMatrix<N>& B) {
  double dA[N * N];
  int pivot[N];

  #pragma unroll
  for (int i = 0; i < N * N; ++i)
    dA[i] = static_cast<double>(A.data[i]);

  dmat_lu_factor<N>(dA, pivot);

  double col_b[N], col_x[N];
  #pragma unroll
  for (int j = 0; j < N; ++j) {
    #pragma unroll
    for (int i = 0; i < N; ++i)
      col_b[i] = static_cast<double>(B(i, j));

    dmat_lu_solve_vec<N>(dA, pivot, col_x, col_b);

    #pragma unroll
    for (int i = 0; i < N; ++i)
      X(i, j) = static_cast<float>(col_x[i]);
  }
}

/// Solve X A = B for X, i.e. X = B A^{-1}. Float interface, double internals.
template<int N>
__device__ __forceinline__ void mat_right_solve_matrix(
    GpuMatrix<N>& X, const GpuMatrix<N>& A, const GpuMatrix<N>& B) {
  // Transpose A into double
  double dAT[N * N];
  #pragma unroll
  for (int i = 0; i < N; ++i)
    #pragma unroll
    for (int j = 0; j < N; ++j)
      dAT[i * N + j] = static_cast<double>(A(j, i));

  int pivot[N];
  dmat_lu_factor<N>(dAT, pivot);

  // Solve A^T * col(X^T, i) = col(B^T, i) for each row i of X
  double row_b[N], row_x[N];
  #pragma unroll
  for (int i = 0; i < N; ++i) {
    #pragma unroll
    for (int j = 0; j < N; ++j)
      row_b[j] = static_cast<double>(B(i, j));

    dmat_lu_solve_vec<N>(dAT, pivot, row_x, row_b);

    #pragma unroll
    for (int j = 0; j < N; ++j)
      X(i, j) = static_cast<float>(row_x[j]);
  }
}

/// Compute A^{-1}. Float interface, double internals.
template<int N>
__device__ __forceinline__ void mat_inverse(
    GpuMatrix<N>& Ainv, const GpuMatrix<N>& A) {
  GpuMatrix<N> I;
  mat_set_identity<N>(I);
  mat_solve_matrix<N>(Ainv, A, I);
}


} // namespace cuda
} // namespace adrt
