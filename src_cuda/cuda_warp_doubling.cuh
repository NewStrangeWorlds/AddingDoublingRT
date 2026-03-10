/// @file cuda_warp_doubling.cuh
/// @brief Warp-cooperative doubling algorithm for the CUDA adding-doubling solver.
///
/// Distributed version of cuda_doubling.cuh: N threads cooperate on one layer,
/// each owning one row of every matrix and one element of every vector.

#pragma once

#include "cuda_warp_layer.cuh"
#include "cuda_warp_matrix.cuh"
#include "cuda_quadrature.cuh"

#include <cmath>

namespace adrt {
namespace cuda {

/// Adaptive number of initial doublings based on single-scattering albedo.
/// Same values as cuda_doubling.cuh.
__device__ __forceinline__ int warp_compute_ipow0(float omega) {
  if (omega < 0.01f) return 2;
  if (omega < 0.1f)  return 5;
  return 8;
}

/// Warp-cooperative doubling algorithm for a single homogeneous layer.
///
/// @param layer          Output: reflection/transmission matrices + source vectors
/// @param tau            Layer optical depth
/// @param omega          Single-scattering albedo
/// @param B_top          Planck function at layer top
/// @param B_bottom       Planck function at layer bottom
/// @param Ppp            Forward phase matrix (distributed: thread owns row row_id)
/// @param Ppm            Backward phase matrix (distributed)
/// @param solar_flux     Incident solar flux (0 if no solar)
/// @param solar_mu       Solar zenith cosine
/// @param tau_cumulative Cumulative optical depth above this layer
/// @param p_plus_solar   Solar forward phase element (this thread's element), or 0
/// @param p_minus_solar  Solar backward phase element (this thread's element), or 0
/// @param has_solar_phase Whether solar phase data is valid
/// @param row_id         This thread's row index (threadIdx.x % N)
template<int N>
__device__ __forceinline__ void warp_doubling(
    WarpLayerMatrices<N>& layer,
    float tau, float omega,
    float B_top, float B_bottom,
    const WarpRow<N>& Ppp, const WarpRow<N>& Ppm,
    float solar_flux, float solar_mu, float tau_cumulative,
    float p_plus_solar, float p_minus_solar,
    bool has_solar_phase,
    int row_id)
{
  constexpr float PI = 3.14159265f;

  layer.set_transparent(row_id);

  float B_bar = (B_bottom + B_top) * 0.5f;
  float B_d = (tau > 0.0f) ? (B_bottom - B_top) / tau : 0.0f;

  if (tau <= 0.0f)
    return;

  // --- Pure absorption (no scattering) ---
  if (omega <= 0.0f) {
    wmat_set_zero<N>(layer.T_ab);
    wmat_set_zero<N>(layer.T_ba);

    // Each thread computes its own row (diagonal transmission)
    float tex = -tau / d_mu[row_id];
    float trans = (tex > -87.0f) ? expf(tex) : 0.0f;

    // Set diagonal element in this thread's row
    #pragma unroll
    for (int j = 0; j < N; ++j) {
      layer.T_ab[j] = (row_id == j) ? trans : 0.0f;
      layer.T_ba[j] = (row_id == j) ? trans : 0.0f;
    }

    float one_minus_t = 1.0f - trans;
    float slope_term = d_mu[row_id] * one_minus_t - 0.5f * tau * (1.0f + trans);
    layer.s_up   = B_bar * one_minus_t + B_d * slope_term;
    layer.s_down = B_bar * one_minus_t - B_d * slope_term;

    return;
  }

  // --- General case: scattering layer ---
  layer.is_scattering = true;
  if (omega > 1.0f) omega = 1.0f;

  float con = 2.0f * omega * PI;

  // Build Gpp and Gpm (each thread computes its own row)
  // Gpp(i,j) = (delta_ij - con * Ppp(i,j) * wt[j]) / mu[i]
  // Gpm(i,j) = con * Ppm(i,j) * wt[j] / mu[i]
  WarpRow<N> Gpp_row, Gpm_row;
  float inv_mu = 1.0f / d_mu[row_id];

  #pragma unroll
  for (int j = 0; j < N; ++j) {
    float ppc = Ppp[j] * d_wt[j];
    float pmc = Ppm[j] * d_wt[j];
    float delta_ij = (row_id == j) ? 1.0f : 0.0f;
    Gpp_row[j] = (delta_ij - con * ppc) * inv_mu;
    Gpm_row[j] = con * pmc * inv_mu;
  }

  // Adaptive doubling count
  int nn = static_cast<int>(logf(tau) / logf(2.0f)) + warp_compute_ipow0(omega);
  if (nn < 2) nn = 2;

  float xfac = 1.0f / exp2f(static_cast<float>(nn));
  float tau0 = tau * xfac;

  bool has_solar = (solar_flux > 0.0f && solar_mu > 0.0f && has_solar_phase);
  float F_top = has_solar ? solar_flux * expf(-tau_cumulative / solar_mu) : 0.0f;

  // Second-order thin-layer initialization (error O(tau0^3))
  float half_tau0_sq = 0.5f * tau0 * tau0;

  WarpRow<N> GppGpp, GpmGpm, GppGpm, GpmGpp;
  wmat_multiply<N>(GppGpp, Gpp_row, Gpp_row);
  wmat_multiply<N>(GpmGpm, Gpm_row, Gpm_row);
  wmat_multiply<N>(GppGpm, Gpp_row, Gpm_row);
  wmat_multiply<N>(GpmGpp, Gpm_row, Gpp_row);

  WarpRow<N> R_k, T_k;
  #pragma unroll
  for (int j = 0; j < N; ++j) {
    float delta_ij = (row_id == j) ? 1.0f : 0.0f;
    T_k[j] = delta_ij - tau0 * Gpp_row[j]
             + half_tau0_sq * (GppGpp[j] + GpmGpm[j]);
    R_k[j] = tau0 * Gpm_row[j]
             - half_tau0_sq * (GppGpm[j] + GpmGpp[j]);
  }

  // Initial source vectors (distributed: each thread holds its element)
  float y_k = (1.0f - omega) * tau0 / d_mu[row_id];
  float z_k = 0.0f;

  float s_up_sol_k = 0.0f;
  float s_down_sol_k = 0.0f;

  if (has_solar) {
    float base = omega * tau0 / d_mu[row_id] * F_top;
    s_up_sol_k   = base * p_minus_solar;
    s_down_sol_k = base * p_plus_solar;
  }

  float g_k = 0.5f * tau0;
  float gamma_sol = has_solar ? expf(-tau0 / solar_mu) : 0.0f;

  // --- Doubling iteration ---
  for (int iter = 0; iter < nn; ++iter) {
    WarpRow<N> R_sq, I_minus_R2;
    wmat_multiply<N>(R_sq, R_k, R_k);

    // I - R_k^2
    #pragma unroll
    for (int j = 0; j < N; ++j) {
      float delta_ij = (row_id == j) ? 1.0f : 0.0f;
      I_minus_R2[j] = delta_ij - R_sq[j];
    }

    // TG = (I - R^2)^{-1} * T_k  via rightSolve: TG * (I - R^2) = T_k
    WarpRow<N> TG;
    wmat_right_solve_matrix<N>(TG, I_minus_R2, T_k, row_id);

    WarpRow<N> TGR;
    wmat_multiply<N>(TGR, TG, R_k);

    // R_new = R_k + TGR * T_k
    WarpRow<N> TGR_T, R_new;
    wmat_multiply<N>(TGR_T, TGR, T_k);
    wmat_add<N>(R_new, R_k, TGR_T, 1.0f);

    // T_new = TG * T_k
    WarpRow<N> T_new;
    wmat_multiply<N>(T_new, TG, T_k);

    // Thermal source update
    float zpgy = z_k + g_k * y_k;

    float TG_zpgy  = wmat_vec_multiply<N>(TG, zpgy);
    float TGR_zpgy = wmat_vec_multiply<N>(TGR, zpgy);
    float TG_y     = wmat_vec_multiply<N>(TG, y_k);
    float TGR_y    = wmat_vec_multiply<N>(TGR, y_k);

    float z_new = (TG_zpgy - TGR_zpgy) + z_k - g_k * y_k;
    float y_new = TG_y + TGR_y + y_k;

    // Solar source update
    float s_up_sol_new = 0.0f;
    float s_down_sol_new = 0.0f;

    if (has_solar) {
      float R_sdown = wmat_vec_multiply<N>(R_k, s_down_sol_k);
      float R_sup   = wmat_vec_multiply<N>(R_k, s_up_sol_k);

      float rhs_up   = R_sdown + gamma_sol * s_up_sol_k;
      float rhs_down = gamma_sol * R_sup + s_down_sol_k;

      float TG_rhs_up   = wmat_vec_multiply<N>(TG, rhs_up);
      float TG_rhs_down = wmat_vec_multiply<N>(TG, rhs_down);

      s_up_sol_new   = TG_rhs_up + s_up_sol_k;
      s_down_sol_new = TG_rhs_down + gamma_sol * s_down_sol_k;

      gamma_sol = gamma_sol * gamma_sol;
    }

    // Advance to next doubling level
    wmat_copy<N>(R_k, R_new);
    wmat_copy<N>(T_k, T_new);
    y_k = y_new;
    z_k = z_new;
    s_up_sol_k = s_up_sol_new;
    s_down_sol_k = s_down_sol_new;
    g_k = 2.0f * g_k;
  }

  // --- Assemble result ---
  layer.is_scattering = true;
  wmat_copy<N>(layer.R_ab, R_k);
  wmat_copy<N>(layer.R_ba, R_k);
  wmat_copy<N>(layer.T_ab, T_k);
  wmat_copy<N>(layer.T_ba, T_k);

  layer.s_up         = y_k * B_bar + z_k * B_d;
  layer.s_down       = y_k * B_bar - z_k * B_d;
  layer.s_up_solar   = s_up_sol_k;
  layer.s_down_solar = s_down_sol_k;
}

} // namespace cuda
} // namespace adrt
