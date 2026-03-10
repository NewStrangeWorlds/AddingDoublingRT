/// @file cuda_doubling.cuh
/// @brief Device-side doubling algorithm for the CUDA adding-doubling solver.
///
/// Computes R, T, and source vectors for a single homogeneous layer
/// via iterative doubling. Direct port of doubling.hpp.

#pragma once

#include "cuda_layer.cuh"
#include "cuda_matrix.cuh"
#include "cuda_quadrature.cuh"

#include <cmath>

namespace adrt {
namespace cuda {

/// Adaptive number of initial doublings based on single-scattering albedo.
/// Reduced from (4, 10, 16) thanks to second-order thin-layer initialization
/// which gives O(tau0^3) starting error instead of O(tau0^2).
__device__ __forceinline__ int compute_ipow0(float omega) {
  if (omega < 0.01f) return 2;
  if (omega < 0.1f)  return 5;
  return 8;
}

/// Doubling algorithm for a single homogeneous layer.
///
/// @param layer          Output: reflection/transmission matrices + source vectors
/// @param tau            Layer optical depth
/// @param omega          Single-scattering albedo
/// @param B_top          Planck function at layer top
/// @param B_bottom       Planck function at layer bottom
/// @param Ppp            Forward phase matrix (N×N)
/// @param Ppm            Backward phase matrix (N×N)
/// @param solar_flux     Incident solar flux (0 if no solar)
/// @param solar_mu       Solar zenith cosine
/// @param tau_cumulative Cumulative optical depth above this layer
/// @param p_plus_solar   Solar forward phase vector (or nullptr)
/// @param p_minus_solar  Solar backward phase vector (or nullptr)
/// @param has_solar_phase Whether solar phase vectors are valid
template<int N>
__device__ __forceinline__ void doubling(
    GpuLayerMatrices<N>& layer,
    float tau, float omega,
    float B_top, float B_bottom,
    const GpuMatrix<N>& Ppp, const GpuMatrix<N>& Ppm,
    float solar_flux, float solar_mu, float tau_cumulative,
    const GpuVec<N>* p_plus_solar, const GpuVec<N>* p_minus_solar,
    bool has_solar_phase)
{
  constexpr float PI = 3.14159265f;

  layer.set_transparent();

  float B_bar = (B_bottom + B_top) * 0.5f;
  float B_d = (tau > 0.0f) ? (B_bottom - B_top) / tau : 0.0f;

  if (tau <= 0.0f)
    return;

  // --- Pure absorption (no scattering) ---
  if (omega <= 0.0f) {
    mat_set_zero<N>(layer.T_ab);
    mat_set_zero<N>(layer.T_ba);

    #pragma unroll
    for (int i = 0; i < N; ++i) {
      float tex = -tau / d_mu[i];
      float trans = (tex > -87.0f) ? expf(tex) : 0.0f;
      layer.T_ab(i, i) = trans;
      layer.T_ba(i, i) = trans;
      float one_minus_t = 1.0f - trans;
      float slope_term = d_mu[i] * one_minus_t - 0.5f * tau * (1.0f + trans);
      layer.s_up[i]   = B_bar * one_minus_t + B_d * slope_term;
      layer.s_down[i] = B_bar * one_minus_t - B_d * slope_term;
    }
    return;
  }

  // --- General case: scattering layer ---
  layer.is_scattering = true;
  if (omega > 1.0f) omega = 1.0f;

  float con = 2.0f * omega * PI;

  // Build Gpp = (I - 2*omega*pi*Ppp*C) / diag(mu)
  // Build Gpm = 2*omega*pi*Ppm*C / diag(mu)
  GpuMatrix<N> Gpp, Gpm;

  // C = diag(wt), so (P*C)(i,j) = P(i,j)*wt[j]
  #pragma unroll
  for (int i = 0; i < N; ++i) {
    float inv_mu = 1.0f / d_mu[i];
    #pragma unroll
    for (int j = 0; j < N; ++j) {
      float ppc = Ppp(i, j) * d_wt[j];
      float pmc = Ppm(i, j) * d_wt[j];
      float delta_ij = (i == j) ? 1.0f : 0.0f;
      Gpp(i, j) = (delta_ij - con * ppc) * inv_mu;
      Gpm(i, j) = con * pmc * inv_mu;
    }
  }

  // Adaptive doubling count
  int nn = static_cast<int>(logf(tau) / logf(2.0f)) + compute_ipow0(omega);
  if (nn < 2) nn = 2;

  float xfac = 1.0f / exp2f(static_cast<float>(nn));
  float tau0 = tau * xfac;

  bool has_solar = (solar_flux > 0.0f && solar_mu > 0.0f && has_solar_phase);
  float F_top = has_solar ? solar_flux * expf(-tau_cumulative / solar_mu) : 0.0f;

  // Second-order thin-layer initialization (error O(tau0^3) instead of O(tau0^2)).
  // From the Taylor expansion of the interaction principle:
  //   T = I - tau0*Gpp + (tau0^2/2)*(Gpp^2 + Gpm^2)
  //   R = tau0*Gpm - (tau0^2/2)*(Gpp*Gpm + Gpm*Gpp)
  // Computed sequentially to reuse temporaries (A, B declared later for doubling loop).
  float half_tau0_sq = 0.5f * tau0 * tau0;

  GpuMatrix<N> R_k, T_k;

  // T_k = I - tau0*Gpp
  #pragma unroll
  for (int i = 0; i < N; ++i)
    #pragma unroll
    for (int j = 0; j < N; ++j)
      T_k(i, j) = ((i == j) ? 1.0f : 0.0f) - tau0 * Gpp(i, j);

  // R_k = tau0*Gpm
  mat_scale<N>(R_k, Gpm, tau0);

  {
    // Use a scoped temporary to add second-order corrections
    GpuMatrix<N> tmp;

    // T_k += half_tau0_sq * Gpp²
    mat_multiply<N>(tmp, Gpp, Gpp);
    mat_add_inplace<N>(T_k, tmp, half_tau0_sq);

    // T_k += half_tau0_sq * Gpm²
    mat_multiply<N>(tmp, Gpm, Gpm);
    mat_add_inplace<N>(T_k, tmp, half_tau0_sq);

    // R_k -= half_tau0_sq * Gpp*Gpm
    mat_multiply<N>(tmp, Gpp, Gpm);
    mat_add_inplace<N>(R_k, tmp, -half_tau0_sq);

    // R_k -= half_tau0_sq * Gpm*Gpp
    mat_multiply<N>(tmp, Gpm, Gpp);
    mat_add_inplace<N>(R_k, tmp, -half_tau0_sq);
  }

  // Initial source vectors
  GpuVec<N> y_k, z_k;
  vec_set_zero<N>(z_k);
  #pragma unroll
  for (int i = 0; i < N; ++i)
    y_k[i] = (1.0f - omega) * tau0 / d_mu[i];

  GpuVec<N> s_up_sol_k, s_down_sol_k;
  vec_set_zero<N>(s_up_sol_k);
  vec_set_zero<N>(s_down_sol_k);

  if (has_solar) {
    #pragma unroll
    for (int i = 0; i < N; ++i) {
      float base = omega * tau0 / d_mu[i] * F_top;
      s_up_sol_k[i]   = base * (*p_minus_solar)[i];
      s_down_sol_k[i] = base * (*p_plus_solar)[i];
    }
  }

  float g_k = 0.5f * tau0;
  float gamma_sol = has_solar ? expf(-tau0 / solar_mu) : 0.0f;

  // --- Doubling iteration ---
  // Uses only 2 temporary N×N matrices (A, B) to reduce peak register pressure.
  // Mapping: B = TG = (I - R²)⁻¹ T,  A = TGR = TG * R
  GpuMatrix<N> A, B;

  for (int k = 0; k < nn; ++k) {
    // A = R_k²
    mat_multiply<N>(A, R_k, R_k);

    // A = I - R_k²  (in-place reuse of A)
    #pragma unroll
    for (int i = 0; i < N * N; ++i)
      A.data[i] = -A.data[i];
    #pragma unroll
    for (int i = 0; i < N; ++i)
      A(i, i) += 1.0f;

    // B = TG = T_k * (I - R²)⁻¹  via right solve: B * A = T_k
    mat_right_solve_matrix<N>(B, A, T_k);

    // A = TGR = B * R_k  (A is now free to reuse)
    mat_multiply<N>(A, B, R_k);

    // Thermal source update (uses B=TG and A=TGR, before R_k/T_k are modified)
    GpuVec<N> zpgy;
    #pragma unroll
    for (int i = 0; i < N; ++i)
      zpgy[i] = z_k[i] + g_k * y_k[i];

    GpuVec<N> TG_zpgy, TGR_zpgy, TG_y, TGR_y;
    mat_vec_multiply<N>(TG_zpgy, B, zpgy);
    mat_vec_multiply<N>(TGR_zpgy, A, zpgy);
    mat_vec_multiply<N>(TG_y, B, y_k);
    mat_vec_multiply<N>(TGR_y, A, y_k);

    GpuVec<N> z_new, y_new;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
      z_new[i] = (TG_zpgy[i] - TGR_zpgy[i]) + z_k[i] - g_k * y_k[i];
      y_new[i] = TG_y[i] + TGR_y[i] + y_k[i];
    }

    // Solar source update (uses B=TG and R_k before they are modified)
    if (has_solar) {
      GpuVec<N> R_sdown, R_sup;
      mat_vec_multiply<N>(R_sdown, R_k, s_down_sol_k);
      mat_vec_multiply<N>(R_sup, R_k, s_up_sol_k);

      GpuVec<N> rhs_up, rhs_down;
      #pragma unroll
      for (int i = 0; i < N; ++i) {
        rhs_up[i]   = R_sdown[i] + gamma_sol * s_up_sol_k[i];
        rhs_down[i] = gamma_sol * R_sup[i] + s_down_sol_k[i];
      }

      GpuVec<N> TG_rhs_up, TG_rhs_down;
      mat_vec_multiply<N>(TG_rhs_up, B, rhs_up);
      mat_vec_multiply<N>(TG_rhs_down, B, rhs_down);

      #pragma unroll
      for (int i = 0; i < N; ++i) {
        s_up_sol_k[i]   = TG_rhs_up[i] + s_up_sol_k[i];
        s_down_sol_k[i] = TG_rhs_down[i] + gamma_sol * s_down_sol_k[i];
      }

      gamma_sol = gamma_sol * gamma_sol;
    }

    // Update R_k and T_k (A=TGR, B=TG still valid)
    // R_k += A * T_k  (i.e. R_k += TGR * T_k)
    mat_multiply_addto<N>(R_k, A, T_k);
    // T_k = TG * T_k — compute into A (now dead), then copy back
    mat_multiply<N>(A, B, T_k);
    mat_copy<N>(T_k, A);

    vec_copy<N>(y_k, y_new);
    vec_copy<N>(z_k, z_new);
    g_k = 2.0f * g_k;
  }

  // --- Assemble result ---
  layer.is_scattering = true;
  mat_copy<N>(layer.R_ab, R_k);
  mat_copy<N>(layer.R_ba, R_k);
  mat_copy<N>(layer.T_ab, T_k);
  mat_copy<N>(layer.T_ba, T_k);

  #pragma unroll
  for (int i = 0; i < N; ++i) {
    layer.s_up[i]   = y_k[i] * B_bar + z_k[i] * B_d;
    layer.s_down[i] = y_k[i] * B_bar - z_k[i] * B_d;
    layer.s_up_solar[i]   = s_up_sol_k[i];
    layer.s_down_solar[i] = s_down_sol_k[i];
  }
}

} // namespace cuda
} // namespace adrt
