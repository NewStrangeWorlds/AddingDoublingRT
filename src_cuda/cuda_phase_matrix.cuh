/// @file cuda_phase_matrix.cuh
/// @brief Device-side phase matrix construction from Legendre coefficients.
///
/// Builds azimuthally-averaged phase matrices Ppp and Ppm for the
/// adding-doubling algorithm.

#pragma once

#include "cuda_matrix.cuh"
#include "cuda_quadrature.cuh"

namespace adrt {
namespace cuda {

/// Build phase matrices from Legendre moments using precomputed polynomials
/// in constant memory (d_Pl).
///
/// @param chi        Legendre moments chi[0..L-1] (in global memory)
/// @param L          Number of moments
/// @param nmu        Number of quadrature points (== N for template path)
/// @param Ppp        Output: forward-scattering phase matrix
/// @param Ppm        Output: back-scattering phase matrix
template<int N>
__device__ __forceinline__ void compute_phase_matrices(
    const float* chi, int L,
    GpuMatrix<N>& Ppp, GpuMatrix<N>& Ppm)
{
  constexpr float inv_2pi = 1.0f / (2.0f * 3.14159265f);

  mat_set_zero<N>(Ppp);
  mat_set_zero<N>(Ppm);

  // Sum Legendre expansion
  #pragma unroll
  for (int i = 0; i < N; ++i) {
    #pragma unroll
    for (int j = 0; j < N; ++j) {
      float sum_pp = 0.0f, sum_pm = 0.0f;
      float sign = 1.0f;

      for (int l = 0; l < L; ++l) {
        float Pl_i = d_Pl[l * MAX_NMU + i];
        float Pl_j = d_Pl[l * MAX_NMU + j];
        float term = (2 * l + 1) * chi[l] * Pl_i * Pl_j;
        sum_pp += term;
        sum_pm += sign * term;
        sign = -sign;
      }

      Ppp(i, j) = sum_pp * inv_2pi;
      Ppm(i, j) = sum_pm * inv_2pi;
    }
  }

  // Hansen normalisation: ensure energy conservation per column
  #pragma unroll
  for (int j = 0; j < N; ++j) {
    float sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < N; ++i)
      sum += (Ppp(i, j) + Ppm(i, j)) * d_wt[i];

    if (sum > 0.0f) {
      float correction = inv_2pi / sum;

      #pragma unroll
      for (int i = 0; i < N; ++i) {
        Ppp(i, j) *= correction;
        Ppm(i, j) *= correction;
      }
    }
  }
}


/// Build solar phase vectors from Legendre moments.
///
/// @param chi          Legendre moments chi[0..L-1]
/// @param L            Number of moments
/// @param mu0          Solar zenith cosine
/// @param p_plus       Output: forward solar phase vector
/// @param p_minus      Output: backward solar phase vector
template<int N>
__device__ __forceinline__ void compute_solar_phase_vectors(
    const float* chi, int L, float mu0,
    GpuVec<N>& p_plus, GpuVec<N>& p_minus)
{
  constexpr float inv_2pi = 1.0f / (2.0f * 3.14159265f);

  vec_set_zero<N>(p_plus);
  vec_set_zero<N>(p_minus);

  // Legendre polynomials at solar angle (computed on the fly — small cost)
  float Pl_mu0[MAX_NMOM];
  Pl_mu0[0] = 1.0f;
  if (L > 1) Pl_mu0[1] = mu0;
  for (int l = 2; l < L; ++l)
    Pl_mu0[l] = ((2 * l - 1) * mu0 * Pl_mu0[l - 1] - (l - 1) * Pl_mu0[l - 2]) / l;

  #pragma unroll
  for (int i = 0; i < N; ++i) {
    float sum_p = 0.0f, sum_m = 0.0f;
    float sign = 1.0f;

    for (int l = 0; l < L; ++l) {
      float Pl_i = d_Pl[l * MAX_NMU + i];
      float term = (2 * l + 1) * chi[l] * Pl_i * Pl_mu0[l];
      sum_p += term;
      sum_m += sign * term;
      sign = -sign;
    }

    p_plus[i]  = sum_p * inv_2pi;
    p_minus[i] = sum_m * inv_2pi;
  }

  // Hansen normalisation
  float sum = 0.0f;
  #pragma unroll
  for (int i = 0; i < N; ++i)
    sum += (p_plus[i] + p_minus[i]) * d_wt[i];

  if (sum > 0.0f) {
    float correction = inv_2pi / sum;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
      p_plus[i]  *= correction;
      p_minus[i] *= correction;
    }
  }
}

} // namespace cuda
} // namespace adrt
