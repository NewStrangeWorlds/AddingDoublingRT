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
__device__ __forceinline__ int compute_ipow0(float omega) {
  if (omega < 0.01f) return 4;
  if (omega < 0.1f)  return 8;
  return 12;
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
  if (nn < 1) nn = 1;

  float xfac = 1.0f / exp2f(static_cast<float>(nn));
  float tau0 = tau * xfac;

  bool has_solar = (solar_flux > 0.0f && solar_mu > 0.0f && has_solar_phase);
  float F_top = has_solar ? solar_flux * expf(-tau_cumulative / solar_mu) : 0.0f;

  // Initial R_k, T_k for thin sub-layer
  GpuMatrix<N> R_k, T_k;
  #pragma unroll
  for (int i = 0; i < N; ++i) {
    #pragma unroll
    for (int j = 0; j < N; ++j) {
      T_k(i, j) = ((i == j) ? 1.0f : 0.0f) - tau0 * Gpp(i, j);
      R_k(i, j) = tau0 * Gpm(i, j);
    }
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
  for (int k = 0; k < nn; ++k) {
    GpuMatrix<N> R_sq, I_minus_R2;
    mat_multiply<N>(R_sq, R_k, R_k);

    // I - R_k^2
    GpuMatrix<N> I_mat;
    mat_set_identity<N>(I_mat);
    mat_add<N>(I_minus_R2, I_mat, R_sq, -1.0f);

    // TG = (I - R^2)^{-1} * T_k  via rightSolve: TG * (I - R^2) = T_k
    // i.e. solve X * A = B where A = I_minus_R2, B = T_k
    GpuMatrix<N> TG;
    mat_right_solve_matrix<N>(TG, I_minus_R2, T_k);

    GpuMatrix<N> TGR;
    mat_multiply<N>(TGR, TG, R_k);

    // R_new = R_k + TGR * T_k
    GpuMatrix<N> TGR_T, R_new;
    mat_multiply<N>(TGR_T, TGR, T_k);
    mat_add<N>(R_new, R_k, TGR_T, 1.0f);

    // T_new = TG * T_k
    GpuMatrix<N> T_new;
    mat_multiply<N>(T_new, TG, T_k);

    // Thermal source update
    GpuVec<N> zpgy;
    #pragma unroll
    for (int i = 0; i < N; ++i)
      zpgy[i] = z_k[i] + g_k * y_k[i];

    GpuVec<N> TG_zpgy, TGR_zpgy, TG_y, TGR_y;
    mat_vec_multiply<N>(TG_zpgy, TG, zpgy);
    mat_vec_multiply<N>(TGR_zpgy, TGR, zpgy);
    mat_vec_multiply<N>(TG_y, TG, y_k);
    mat_vec_multiply<N>(TGR_y, TGR, y_k);

    GpuVec<N> z_new, y_new;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
      z_new[i] = (TG_zpgy[i] - TGR_zpgy[i]) + z_k[i] - g_k * y_k[i];
      y_new[i] = TG_y[i] + TGR_y[i] + y_k[i];
    }

    // Solar source update
    GpuVec<N> s_up_sol_new, s_down_sol_new;
    vec_set_zero<N>(s_up_sol_new);
    vec_set_zero<N>(s_down_sol_new);

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
      mat_vec_multiply<N>(TG_rhs_up, TG, rhs_up);
      mat_vec_multiply<N>(TG_rhs_down, TG, rhs_down);

      #pragma unroll
      for (int i = 0; i < N; ++i) {
        s_up_sol_new[i]   = TG_rhs_up[i] + s_up_sol_k[i];
        s_down_sol_new[i] = TG_rhs_down[i] + gamma_sol * s_down_sol_k[i];
      }

      gamma_sol = gamma_sol * gamma_sol;
    }

    // Advance to next doubling level
    mat_copy<N>(R_k, R_new);
    mat_copy<N>(T_k, T_new);
    vec_copy<N>(y_k, y_new);
    vec_copy<N>(z_k, z_new);
    vec_copy<N>(s_up_sol_k, s_up_sol_new);
    vec_copy<N>(s_down_sol_k, s_down_sol_new);
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
