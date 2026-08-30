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
/// The GPU backends use fewer doublings than the CPU (4, 10, 16): the start is
/// exact through single scattering with an O(tau0^3) multiple-scattering
/// truncation, and single precision limits the attainable accuracy to ~1e-4
/// anyway. (2, 8, 12) keeps the flux error of the thin-layer start at or below
/// that level; the former (2, 5, 8) gave 1e-2 level thermal-source errors for
/// strongly scattering thick layers.
__device__ __forceinline__ int compute_ipow0(float omega) {
  if (omega < 0.01f) return 2;
  if (omega < 0.1f)  return 8;
  return 12;
}

/// Number of doublings: omega-adaptive rule plus the omega-scaled extinction
/// floor nn >= log2(tau/mu_min) + 0.5*log2(omega) + c (mirrors
/// adrt::computeDoublingCount; c = 6.6 here vs 7.6 on the CPU, i.e. a start
/// tolerance of ~1e-4 matching single precision).
///
/// The floor is capped at the strong-scattering count log2(tau) + 12: the
/// doubling squares T at every step, so in single precision the round-off
/// error grows as ~2^nn * eps and exceeds the start's truncation error beyond
/// ~14 steps. Uncapped, the floor pushed N = 16 layers with omega >= 0.5 to
/// 17 doublings and *degraded* the flux from <5e-4 to 1e-3 relative.
template<int N>
__device__ __forceinline__ int compute_doubling_count(float tau, float omega) {
  int log2tau = static_cast<int>(logf(tau) / logf(2.0f));
  int nn = log2tau + compute_ipow0(omega);

  float mu_min = d_mu[0];
  #pragma unroll
  for (int i = 1; i < N; ++i) mu_min = fminf(mu_min, d_mu[i]);

  float om = fmaxf(omega, 1e-8f);
  int n_ext = static_cast<int>(ceilf(log2f(tau / mu_min) + 0.5f * log2f(om) + 6.6f));
  n_ext = min(n_ext, log2tau + 12);

  return max(1, max(nn, n_ext));
}

/// phi(u) = expm1(u) / u, phi(0) = 1.
__device__ __forceinline__ float doubling_phi(float u) {
  return (fabsf(u) < 1e-6f) ? 1.0f + 0.5f * u : expm1f(u) / u;
}

/// int_0^tau0 exp(-(tau0-t)/mu_i) exp(-t/mu_j) dt = (e_j - e_i) / d, d = 1/mu_i - 1/mu_j.
/// tau0 e_i phi(tau0 d) for small |tau0 d|, difference form otherwise (no overflow).
__device__ __forceinline__ float doubling_path_integral(float tau0, float d, float e_i, float e_j) {
  float u = tau0 * d;
  return (fabsf(u) < 1.0f) ? tau0 * e_i * doubling_phi(u) : (e_j - e_i) / d;
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

  // Scattering operators (C = diag(wt), so (P*C)(i,j) = P(i,j)*wt[j]):
  //   Spp = 2*omega*pi*Ppp*C / diag(mu)   (forward,  same hemisphere)
  //   Spm = 2*omega*pi*Ppm*C / diag(mu)   (backward, opposite hemisphere)
  GpuMatrix<N> Spp, Spm;

  #pragma unroll
  for (int i = 0; i < N; ++i) {
    float inv_mu = 1.0f / d_mu[i];
    #pragma unroll
    for (int j = 0; j < N; ++j) {
      Spp(i, j) = con * Ppp(i, j) * d_wt[j] * inv_mu;
      Spm(i, j) = con * Ppm(i, j) * d_wt[j] * inv_mu;
    }
  }

  // Adaptive doubling count
  int nn = compute_doubling_count<N>(tau, omega);

  float xfac = 1.0f / exp2f(static_cast<float>(nn));
  float tau0 = tau * xfac;

  bool has_solar = (solar_flux > 0.0f && solar_mu > 0.0f && has_solar_phase);
  float F_top = has_solar ? solar_flux * expf(-tau_cumulative / solar_mu) : 0.0f;

  // Thin-layer initialisation (direct port of adrt::initDoublingStart):
  //   exact extinction e_i = exp(-tau0/mu_i), exact single scattering,
  //   Taylor double scattering tau0^2/2 * S^2, thermal source consistent with
  //   the absorbed fraction of the operators (Kirchhoff) through O(tau0^2).
  // Extinction and single scattering are exact for any tau0/mu, so the start
  // never leaves the physical range (the former polynomial starts blew up for
  // tau0 > mu_min under doubling).
  float half_tau0_sq = 0.5f * tau0 * tau0;

  GpuMatrix<N> R_k, T_k;

  #pragma unroll
  for (int i = 0; i < N; ++i) {
    float inv_mu_i = 1.0f / d_mu[i];
    float e_i = expf(-tau0 * inv_mu_i);
    #pragma unroll
    for (int j = 0; j < N; ++j) {
      float inv_mu_j = 1.0f / d_mu[j];
      float e_j = expf(-tau0 * inv_mu_j);
      // T: int_0^tau0 exp(-(tau0-t)/mu_i) exp(-t/mu_j) dt ; R: int_0^tau0 exp(-t/mu_i) exp(-t/mu_j) dt
      float int_T = doubling_path_integral(tau0, inv_mu_i - inv_mu_j, e_i, e_j);
      float int_R = tau0 * doubling_phi(-tau0 * (inv_mu_i + inv_mu_j));
      T_k(i, j) = ((i == j) ? e_i : 0.0f) + Spp(i, j) * int_T;
      R_k(i, j) = Spm(i, j) * int_R;
    }
  }

  {
    // Taylor double scattering: T += h (Spp^2 + Spm^2), R += h (Spp Spm + Spm Spp)
    GpuMatrix<N> tmp;

    mat_multiply<N>(tmp, Spp, Spp);
    mat_add_inplace<N>(T_k, tmp, half_tau0_sq);

    mat_multiply<N>(tmp, Spm, Spm);
    mat_add_inplace<N>(T_k, tmp, half_tau0_sq);

    mat_multiply<N>(tmp, Spp, Spm);
    mat_add_inplace<N>(R_k, tmp, half_tau0_sq);

    mat_multiply<N>(tmp, Spm, Spp);
    mat_add_inplace<N>(R_k, tmp, half_tau0_sq);
  }

  // Initial source vectors
  GpuVec<N> y_k, z_k;
  GpuVec<N> s_up_sol_k, s_down_sol_k;
  vec_set_zero<N>(s_up_sol_k);
  vec_set_zero<N>(s_down_sol_k);

  #pragma unroll
  for (int i = 0; i < N; ++i) {
    float inv_mu_i = 1.0f / d_mu[i];
    float x = tau0 * inv_mu_i;
    float e_i = expf(-x);
    float a_i = -expm1f(-x);            // 1 - e_i

    // Kirchhoff-consistent emission: absorbed fraction of the start through O(tau0^2)
    float d2 = 0.0f;
    #pragma unroll
    for (int k = 0; k < N; ++k)
      d2 += (Spp(i, k) + Spm(i, k)) / d_mu[k];

    y_k[i] = (1.0f - omega) * (a_i + half_tau0_sq * d2);

    // z_i = (1-omega) [ mu_i (1 - e_i) - tau0/2 (1 + e_i) ]  (= -(1-omega) mu_i x^3/12 + ...)
    float slope = (x < 3e-2f)
        ? d_mu[i] * x * x * x * (-1.0f / 12.0f + x * (1.0f / 24.0f - x / 80.0f))
        : d_mu[i] * a_i - 0.5f * tau0 * (1.0f + e_i);
    z_k[i] = (1.0f - omega) * slope;
  }

  if (has_solar) {
    float inv_mu0 = 1.0f / solar_mu;
    float e_0 = expf(-tau0 * inv_mu0);

    #pragma unroll
    for (int i = 0; i < N; ++i) {
      float inv_mu_i = 1.0f / d_mu[i];
      float e_i = expf(-tau0 * inv_mu_i);

      // single scattering of the direct beam, attenuated along both legs
      float int_up = tau0 * doubling_phi(-tau0 * (inv_mu0 + inv_mu_i));
      float int_dn = doubling_path_integral(tau0, inv_mu_i - inv_mu0, e_i, e_0);

      float src_up_i = omega * F_top * (*p_minus_solar)[i] * inv_mu_i;
      float src_dn_i = omega * F_top * (*p_plus_solar)[i]  * inv_mu_i;

      // Taylor double scattering
      float d_up = 0.0f, d_dn = 0.0f;
      #pragma unroll
      for (int k = 0; k < N; ++k) {
        float inv_mu_k = 1.0f / d_mu[k];
        float src_up_k = omega * F_top * (*p_minus_solar)[k] * inv_mu_k;
        float src_dn_k = omega * F_top * (*p_plus_solar)[k]  * inv_mu_k;
        d_up += Spp(i, k) * src_up_k + Spm(i, k) * src_dn_k;
        d_dn += Spp(i, k) * src_dn_k + Spm(i, k) * src_up_k;
      }

      s_up_sol_k[i]   = src_up_i * int_up + half_tau0_sq * d_up;
      s_down_sol_k[i] = src_dn_i * int_dn + half_tau0_sq * d_dn;
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
