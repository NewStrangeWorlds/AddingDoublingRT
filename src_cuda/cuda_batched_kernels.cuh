/// @file cuda_batched_kernels.cuh
/// @brief Custom CUDA kernels for the batched cuBLAS adding-doubling solver.
///
/// Element-wise operations that cuBLAS cannot express: phase matrix construction,
/// thin-layer initialization, source vector updates, flux reduction, etc.

#pragma once

#include "cuda_quadrature.cuh"
#include "cuda_planck.cuh"

#include <cmath>

namespace adrt {
namespace cuda {
namespace batched {

// ============================================================================
//  Grid helpers
// ============================================================================

inline int divUp(int a, int b) { return (a + b - 1) / b; }
constexpr int BLOCK = 256;

// ============================================================================
//  Phase matrix construction (one thread per layer)
// ============================================================================

/// Build Ppp, Ppm matrices for all layers from Legendre moments.
/// Also applies delta-M truncation to the moments themselves.
/// Layout: out_Ppp[l * N * N + i * N + j], row-major per layer.
__global__ void batchedPhaseMatrixKernel(
    int nlay, int N, int nmom_max, bool use_delta_m,
    float solar_mu, bool has_solar,
    const float* __restrict__ phase_moments,  // [nlay * nmom_max]
    const float* __restrict__ mu,             // [N]
    const float* __restrict__ wt,             // [N]
    const float* __restrict__ Pl,             // [nmom_max * N] — Pl[l*N+i]
    float* __restrict__ out_Ppp,              // [nlay * N * N]
    float* __restrict__ out_Ppm,              // [nlay * N * N]
    float* __restrict__ out_f_trunc,          // [nlay]
    float* __restrict__ out_solar_pp,         // [nlay * N] or nullptr
    float* __restrict__ out_solar_pm)         // [nlay * N] or nullptr
{
  int l = blockIdx.x * blockDim.x + threadIdx.x;
  if (l >= nlay) return;

  constexpr float inv_2pi = 1.0f / (2.0f * 3.14159265f);
  int two_M = 2 * N;

  // Load chi into local buffer
  float chi_buf[MAX_NMOM];
  int nmom = nmom_max;
  for (int m = 0; m < nmom; ++m)
    chi_buf[m] = phase_moments[l * nmom_max + m];

  // Delta-M scaling
  float f = 0.0f;
  if (use_delta_m) {
    f = (nmom > two_M) ? chi_buf[two_M] : 0.0f;
    if (f > 1e-12f && f < 1.0f - 1e-12f) {
      for (int m = 0; m < two_M; ++m)
        chi_buf[m] = (chi_buf[m] - f) / (1.0f - f);
      nmom = two_M;
    } else {
      f = 0.0f;
      if (nmom > two_M) nmom = two_M;
    }
  }
  out_f_trunc[l] = f;

  // Build phase matrices
  int base = l * N * N;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum_pp = 0.0f, sum_pm = 0.0f;
      float sign = 1.0f;
      for (int ll = 0; ll < nmom; ++ll) {
        float Pl_i = Pl[ll * N + i];
        float Pl_j = Pl[ll * N + j];
        float term = (2 * ll + 1) * chi_buf[ll] * Pl_i * Pl_j;
        sum_pp += term;
        sum_pm += sign * term;
        sign = -sign;
      }
      out_Ppp[base + i * N + j] = sum_pp * inv_2pi;
      out_Ppm[base + i * N + j] = sum_pm * inv_2pi;
    }
  }

  // Hansen normalisation
  for (int j = 0; j < N; ++j) {
    float sum = 0.0f;
    for (int i = 0; i < N; ++i)
      sum += (out_Ppp[base + i * N + j] + out_Ppm[base + i * N + j]) * wt[i];
    if (sum > 0.0f) {
      float correction = inv_2pi / sum;
      for (int i = 0; i < N; ++i) {
        out_Ppp[base + i * N + j] *= correction;
        out_Ppm[base + i * N + j] *= correction;
      }
    }
  }

  // Solar phase vectors
  if (has_solar && out_solar_pp != nullptr) {
    float Pl_mu0[MAX_NMOM];
    Pl_mu0[0] = 1.0f;
    if (nmom > 1) Pl_mu0[1] = solar_mu;
    for (int ll = 2; ll < nmom; ++ll)
      Pl_mu0[ll] = ((2 * ll - 1) * solar_mu * Pl_mu0[ll - 1]
                     - (ll - 1) * Pl_mu0[ll - 2]) / ll;

    int vbase = l * N;
    for (int i = 0; i < N; ++i) {
      float sum_p = 0.0f, sum_m = 0.0f;
      float sign = 1.0f;
      for (int ll = 0; ll < nmom; ++ll) {
        float Pl_i = Pl[ll * N + i];
        float term = (2 * ll + 1) * chi_buf[ll] * Pl_i * Pl_mu0[ll];
        sum_p += term;
        sum_m += sign * term;
        sign = -sign;
      }
      out_solar_pp[vbase + i] = sum_p * inv_2pi;
      out_solar_pm[vbase + i] = sum_m * inv_2pi;
    }

    // Hansen normalisation for solar
    float sv_sum = 0.0f;
    for (int i = 0; i < N; ++i)
      sv_sum += (out_solar_pp[l * N + i] + out_solar_pm[l * N + i]) * wt[i];
    if (sv_sum > 0.0f) {
      float correction = inv_2pi / sv_sum;
      for (int i = 0; i < N; ++i) {
        out_solar_pp[l * N + i] *= correction;
        out_solar_pm[l * N + i] *= correction;
      }
    }
  }
}


// ============================================================================
//  Build Gpp, Gpm from phase matrices (one thread per matrix element per wav)
// ============================================================================

/// Gpp(i,j) = (delta_ij - 2*omega*pi * Ppp(i,j)*wt[j]) / mu[i]
/// Gpm(i,j) = 2*omega*pi * Ppm(i,j)*wt[j] / mu[i]
/// Indexed: element [w * N*N + i*N + j]
__global__ void batchedBuildGppGpmKernel(
    int nwav, int N,
    const float* __restrict__ Ppp_layer,  // [N*N] (shared across wavenumbers)
    const float* __restrict__ Ppm_layer,  // [N*N]
    const float* __restrict__ omega,      // [nwav] per-wavenumber omega
    const float* __restrict__ mu,         // [N]
    const float* __restrict__ wt,         // [N]
    float* __restrict__ Gpp,              // [nwav * N * N]
    float* __restrict__ Gpm)              // [nwav * N * N]
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = nwav * N * N;
  if (idx >= total) return;

  int w = idx / (N * N);
  int rem = idx % (N * N);
  int i = rem / N;
  int j = rem % N;

  float om = omega[w];
  float con = 2.0f * 3.14159265f * om;
  float inv_mu = 1.0f / mu[i];
  float ppc = Ppp_layer[i * N + j] * wt[j];
  float pmc = Ppm_layer[i * N + j] * wt[j];
  float delta_ij = (i == j) ? 1.0f : 0.0f;

  Gpp[idx] = (delta_ij - con * ppc) * inv_mu;
  Gpm[idx] = con * pmc * inv_mu;
}


// ============================================================================
//  Second-order thin-layer initialization
// ============================================================================

/// T_k = I - tau0*Gpp, R_k = tau0*Gpm  (first order, applied element-wise)
/// Second-order corrections added via cuBLAS GEMM calls from the host.
__global__ void batchedFirstOrderInitKernel(
    int nwav, int N,
    const float* __restrict__ tau0,   // [nwav]
    const float* __restrict__ Gpp,    // [nwav * N*N]
    const float* __restrict__ Gpm,    // [nwav * N*N]
    float* __restrict__ T_k,          // [nwav * N*N]
    float* __restrict__ R_k)          // [nwav * N*N]
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = nwav * N * N;
  if (idx >= total) return;

  int w = idx / (N * N);
  int rem = idx % (N * N);
  int i = rem / N;
  int j = rem % N;

  float t0 = tau0[w];
  float delta_ij = (i == j) ? 1.0f : 0.0f;

  T_k[idx] = delta_ij - t0 * Gpp[idx];
  R_k[idx] = t0 * Gpm[idx];
}

/// Add second-order correction: T_k += half_tau0^2 * tmp, or R_k -= half_tau0^2 * tmp
/// Called after cuBLAS computes tmp = Gpp*Gpp, Gpm*Gpm, etc.
__global__ void batchedAddScaledMatrixKernel(
    int nwav, int N,
    const float* __restrict__ half_tau0_sq, // [nwav]
    float sign,                              // +1 or -1
    const float* __restrict__ src,           // [nwav * N*N]
    float* __restrict__ dst)                 // [nwav * N*N]
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = nwav * N * N;
  if (idx >= total) return;

  int w = idx / (N * N);
  dst[idx] += sign * half_tau0_sq[w] * src[idx];
}


// ============================================================================
//  I - A  (negate and add identity)
// ============================================================================

__global__ void batchedNegateAddIdentityKernel(
    int nwav, int N,
    float* __restrict__ A)  // [nwav * N*N], modified in-place
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = nwav * N * N;
  if (idx >= total) return;

  int rem = idx % (N * N);
  int i = rem / N;
  int j = rem % N;

  float val = -A[idx];
  if (i == j) val += 1.0f;
  A[idx] = val;
}


// ============================================================================
//  Float <-> Double conversion for LU solve
// ============================================================================

__global__ void batchedFloat2DoubleKernel(
    int count,
    const float* __restrict__ src,
    double* __restrict__ dst)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  dst[idx] = static_cast<double>(src[idx]);
}

__global__ void batchedDouble2FloatKernel(
    int count,
    const double* __restrict__ src,
    float* __restrict__ dst)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  dst[idx] = static_cast<float>(src[idx]);
}


// ============================================================================
//  Build pointer arrays for batched LU
// ============================================================================

__global__ void batchedBuildPointerArrayKernel(
    int nwav, int stride,
    double* base,
    double** ptrs)
{
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= nwav) return;
  ptrs[w] = base + w * stride;
}

// Overload for float
__global__ void batchedBuildFloatPointerArrayKernel(
    int nwav, int stride,
    float* base,
    float** ptrs)
{
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= nwav) return;
  ptrs[w] = base + w * stride;
}


// ============================================================================
//  Source vector initialization (doubling)
// ============================================================================

/// y_k[w*N+i] = (1 - omega) * tau0 / mu[i]
/// z_k = 0
/// Solar: s_up_sol[w*N+i] = omega * tau0 / mu[i] * F_top * p_minus[i]
///        s_down_sol[w*N+i] = omega * tau0 / mu[i] * F_top * p_plus[i]
__global__ void batchedSourceInitKernel(
    int nwav, int N,
    const float* __restrict__ tau0,     // [nwav]
    const float* __restrict__ omega,    // [nwav]
    const float* __restrict__ mu,       // [N]
    float solar_flux, float solar_mu,
    const float* __restrict__ tau_above, // [nwav] cumulative tau above this layer
    bool has_solar,
    const float* __restrict__ solar_pp, // [N] or nullptr
    const float* __restrict__ solar_pm, // [N] or nullptr
    float* __restrict__ y_k,            // [nwav * N]
    float* __restrict__ z_k,            // [nwav * N]
    float* __restrict__ s_up_sol,       // [nwav * N]
    float* __restrict__ s_down_sol,     // [nwav * N]
    float* __restrict__ g_k,            // [nwav]
    float* __restrict__ gamma_sol)      // [nwav]
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // First pass: per-wavenumber scalars (g_k, gamma_sol)
  // Handle both scalars and vectors in one kernel
  if (idx < nwav) {
    g_k[idx] = 0.5f * tau0[idx];
    if (has_solar)
      gamma_sol[idx] = expf(-tau0[idx] / solar_mu);
    else
      gamma_sol[idx] = 0.0f;
  }

  // Vector elements
  if (idx >= nwav * N) return;

  int w = idx / N;
  int i = idx % N;

  float t0 = tau0[w];
  float om = omega[w];
  float inv_mu = 1.0f / mu[i];

  y_k[idx] = (1.0f - om) * t0 * inv_mu;
  z_k[idx] = 0.0f;

  if (has_solar && solar_pp != nullptr) {
    float F_top = solar_flux * expf(-tau_above[w] / solar_mu);
    float base_val = om * t0 * inv_mu * F_top;
    s_up_sol[idx]   = base_val * solar_pm[i];
    s_down_sol[idx] = base_val * solar_pp[i];
  } else {
    s_up_sol[idx]   = 0.0f;
    s_down_sol[idx] = 0.0f;
  }
}


// ============================================================================
//  Thermal source update (after each doubling step)
// ============================================================================

/// Given TG (B matrix from solve) and TGR (A matrix = TG * R_k), update sources.
/// y_new[i] = TG_y[i] + TGR_y[i] + y_k[i]
/// z_new[i] = (TG_zpgy[i] - TGR_zpgy[i]) + z_k[i] - g_k * y_k[i]
/// where zpgy[i] = z_k[i] + g_k * y_k[i]
///
/// This kernel computes zpgy, then the host calls cuBLAS for TG*zpgy etc.,
/// then this kernel combines results.

__global__ void batchedComputeZpgyKernel(
    int nwav, int N,
    const float* __restrict__ z_k,    // [nwav * N]
    const float* __restrict__ y_k,    // [nwav * N]
    const float* __restrict__ g_k,    // [nwav]
    float* __restrict__ zpgy)         // [nwav * N]
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nwav * N) return;
  int w = idx / N;
  zpgy[idx] = z_k[idx] + g_k[w] * y_k[idx];
}

/// Combine TG/TGR results into updated y, z
__global__ void batchedThermalSourceCombineKernel(
    int nwav, int N,
    const float* __restrict__ TG_zpgy,   // [nwav * N]
    const float* __restrict__ TGR_zpgy,  // [nwav * N]
    const float* __restrict__ TG_y,      // [nwav * N]
    const float* __restrict__ TGR_y,     // [nwav * N]
    const float* __restrict__ y_k_old,   // [nwav * N]
    const float* __restrict__ z_k_old,   // [nwav * N]
    const float* __restrict__ g_k,       // [nwav]
    float* __restrict__ y_k_new,         // [nwav * N]
    float* __restrict__ z_k_new)         // [nwav * N]
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nwav * N) return;
  int w = idx / N;

  float gk = g_k[w];
  float y_old = y_k_old[idx];
  float z_old = z_k_old[idx];

  z_k_new[idx] = (TG_zpgy[idx] - TGR_zpgy[idx]) + z_old - gk * y_old;
  y_k_new[idx] = TG_y[idx] + TGR_y[idx] + y_old;
}

/// Update g_k *= 2
__global__ void batchedDoubleGkKernel(int nwav, float* __restrict__ g_k) {
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= nwav) return;
  g_k[w] *= 2.0f;
}


// ============================================================================
//  Solar source update (after each doubling step)
// ============================================================================

/// Compute rhs_up[i] = R_k * s_down_sol[i] + gamma_sol * s_up_sol[i]   (via TG)
/// Compute rhs_down[i] = gamma_sol * R_k * s_up_sol[i] + s_down_sol[i] (via TG)
/// Then: s_up_new = TG * rhs_up + s_up_old
///        s_down_new = TG * rhs_down + gamma_sol * s_down_old
///
/// This kernel builds the rhs vectors; cuBLAS does R_k * vec and TG * vec.

__global__ void batchedSolarRhsKernel(
    int nwav, int N,
    const float* __restrict__ R_sdown,     // [nwav * N] = R_k * s_down_sol
    const float* __restrict__ R_sup,       // [nwav * N] = R_k * s_up_sol
    const float* __restrict__ s_up_sol,    // [nwav * N]
    const float* __restrict__ s_down_sol,  // [nwav * N]
    const float* __restrict__ gamma_sol,   // [nwav]
    float* __restrict__ rhs_up,            // [nwav * N]
    float* __restrict__ rhs_down)          // [nwav * N]
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nwav * N) return;
  int w = idx / N;
  float gs = gamma_sol[w];
  rhs_up[idx]   = R_sdown[idx] + gs * s_up_sol[idx];
  rhs_down[idx] = gs * R_sup[idx] + s_down_sol[idx];
}

/// Combine: s_up_new = TG_rhs_up + s_up_old, s_down_new = TG_rhs_down + gamma * s_down_old
/// Also update gamma_sol = gamma_sol^2
__global__ void batchedSolarCombineKernel(
    int nwav, int N,
    const float* __restrict__ TG_rhs_up,
    const float* __restrict__ TG_rhs_down,
    float* __restrict__ s_up_sol,    // in/out
    float* __restrict__ s_down_sol,  // in/out
    float* __restrict__ gamma_sol)   // in/out [nwav]
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Update gamma_sol (per-wavenumber)
  if (idx < nwav)
    gamma_sol[idx] = gamma_sol[idx] * gamma_sol[idx];

  if (idx >= nwav * N) return;
  int w = idx / N;
  // Note: gamma_sol was already squared above, so use sqrt for this step
  // Actually we need old gamma. Let's re-read: the combine happens BEFORE gamma update.
  // We'll fix by doing gamma update separately.
}

// Better: separate gamma update
__global__ void batchedSolarCombineKernel2(
    int nwav, int N,
    const float* __restrict__ TG_rhs_up,
    const float* __restrict__ TG_rhs_down,
    const float* __restrict__ gamma_sol,  // [nwav] (not yet squared)
    float* __restrict__ s_up_sol,         // in/out
    float* __restrict__ s_down_sol)       // in/out
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nwav * N) return;
  int w = idx / N;
  float gs = gamma_sol[w];
  s_up_sol[idx]   = TG_rhs_up[idx] + s_up_sol[idx];
  s_down_sol[idx] = TG_rhs_down[idx] + gs * s_down_sol[idx];
}

__global__ void batchedSquareGammaSolKernel(int nwav, float* __restrict__ gamma_sol) {
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= nwav) return;
  gamma_sol[w] = gamma_sol[w] * gamma_sol[w];
}


// ============================================================================
//  Assemble final layer from R_k, T_k, y_k, z_k
// ============================================================================

__global__ void batchedAssembleLayerKernel(
    int nwav, int N,
    const float* __restrict__ R_k,          // [nwav * N*N]
    const float* __restrict__ T_k,          // [nwav * N*N]
    float* __restrict__ R_ab,               // [nwav * N*N]
    float* __restrict__ R_ba,               // [nwav * N*N]
    float* __restrict__ T_ab,               // [nwav * N*N]
    float* __restrict__ T_ba)               // [nwav * N*N]
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = nwav * N * N;
  if (idx >= total) return;
  // For a homogeneous layer: R_ab = R_ba = R_k, T_ab = T_ba = T_k
  R_ab[idx] = R_k[idx];
  R_ba[idx] = R_k[idx];
  T_ab[idx] = T_k[idx];
  T_ba[idx] = T_k[idx];
}

__global__ void batchedAssembleSourceKernel(
    int nwav, int N,
    const float* __restrict__ y_k,
    const float* __restrict__ z_k,
    const float* __restrict__ s_up_sol,
    const float* __restrict__ s_down_sol,
    const float* __restrict__ B_bar,    // [nwav]
    const float* __restrict__ B_d,      // [nwav]
    float* __restrict__ s_up,           // [nwav * N]
    float* __restrict__ s_down,         // [nwav * N]
    float* __restrict__ s_up_solar,     // [nwav * N]
    float* __restrict__ s_down_solar)   // [nwav * N]
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nwav * N) return;
  int w = idx / N;
  float bb = B_bar[w];
  float bd = B_d[w];
  s_up[idx]         = y_k[idx] * bb + z_k[idx] * bd;
  s_down[idx]       = y_k[idx] * bb - z_k[idx] * bd;
  s_up_solar[idx]   = s_up_sol[idx];
  s_down_solar[idx] = s_down_sol[idx];
}


// ============================================================================
//  Pure absorption layer (no scattering)
// ============================================================================

__global__ void batchedPureAbsorptionKernel(
    int nwav, int N,
    const float* __restrict__ tau,    // [nwav]
    const float* __restrict__ B_top,  // [nwav]
    const float* __restrict__ B_bot,  // [nwav]
    const float* __restrict__ mu,     // [N]
    float* __restrict__ T_ab,         // [nwav * N*N]
    float* __restrict__ T_ba,         // [nwav * N*N]
    float* __restrict__ R_ab,         // [nwav * N*N]
    float* __restrict__ R_ba,         // [nwav * N*N]
    float* __restrict__ s_up,         // [nwav * N]
    float* __restrict__ s_down,       // [nwav * N]
    float* __restrict__ s_up_solar,   // [nwav * N]
    float* __restrict__ s_down_solar) // [nwav * N]
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Zero out all matrices
  int mat_total = nwav * N * N;
  if (idx < mat_total) {
    R_ab[idx] = 0.0f;
    R_ba[idx] = 0.0f;
    T_ab[idx] = 0.0f;
    T_ba[idx] = 0.0f;

    // Set diagonal of T
    int w = idx / (N * N);
    int rem = idx % (N * N);
    int i = rem / N;
    int j = rem % N;
    if (i == j) {
      float tex = -tau[w] / mu[i];
      float trans = (tex > -87.0f) ? expf(tex) : 0.0f;
      T_ab[idx] = trans;
      T_ba[idx] = trans;
    }
  }

  // Source vectors
  if (idx < nwav * N) {
    int w = idx / N;
    int i = idx % N;
    float t = tau[w];
    float B_bar = (B_bot[w] + B_top[w]) * 0.5f;
    float B_d_val = (t > 0.0f) ? (B_bot[w] - B_top[w]) / t : 0.0f;
    float tex = -t / mu[i];
    float trans = (tex > -87.0f) ? expf(tex) : 0.0f;
    float one_minus_t = 1.0f - trans;
    float slope_term = mu[i] * one_minus_t - 0.5f * t * (1.0f + trans);
    s_up[idx]   = B_bar * one_minus_t + B_d_val * slope_term;
    s_down[idx] = B_bar * one_minus_t - B_d_val * slope_term;
    s_up_solar[idx]   = 0.0f;
    s_down_solar[idx] = 0.0f;
  }
}


// ============================================================================
//  Transparent layer init
// ============================================================================

__global__ void batchedSetTransparentKernel(
    int nwav, int N,
    float* __restrict__ R_ab,
    float* __restrict__ R_ba,
    float* __restrict__ T_ab,
    float* __restrict__ T_ba,
    float* __restrict__ s_up,
    float* __restrict__ s_down,
    float* __restrict__ s_up_solar,
    float* __restrict__ s_down_solar)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int mat_total = nwav * N * N;
  if (idx < mat_total) {
    int rem = idx % (N * N);
    int i = rem / N;
    int j = rem % N;
    float diag = (i == j) ? 1.0f : 0.0f;
    R_ab[idx] = 0.0f;
    R_ba[idx] = 0.0f;
    T_ab[idx] = diag;
    T_ba[idx] = diag;
  }
  if (idx < nwav * N) {
    s_up[idx] = 0.0f;
    s_down[idx] = 0.0f;
    s_up_solar[idx] = 0.0f;
    s_down_solar[idx] = 0.0f;
  }
}


// ============================================================================
//  Surface layer construction
// ============================================================================

__global__ void batchedSurfaceLayerKernel(
    int nwav, int N,
    float surface_albedo,
    float surface_emission_scalar,
    float solar_flux, float solar_mu,
    bool has_solar,
    bool use_thermal_emission,
    const float* __restrict__ mu,   // [N]
    const float* __restrict__ wt,   // [N]
    float xfac,
    const float* __restrict__ per_wav_surface_emission, // [nwav] or nullptr
    const float* __restrict__ tau_total,                // [nwav]
    const float* __restrict__ B_levels,                 // [nwav * nlev] or nullptr
    int nlev,
    float* __restrict__ R_ab,         // [nwav * N*N]
    float* __restrict__ R_ba,
    float* __restrict__ T_ab,
    float* __restrict__ T_ba,
    float* __restrict__ s_up,
    float* __restrict__ s_down,
    float* __restrict__ s_up_solar,
    float* __restrict__ s_down_solar)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int mat_total = nwav * N * N;
  float A = surface_albedo;
  constexpr float PI = 3.14159265f;

  if (idx < mat_total) {
    int w = idx / (N * N);
    int rem = idx % (N * N);
    int i = rem / N;
    int j = rem % N;

    T_ab[idx] = 0.0f;
    T_ba[idx] = 0.0f;
    float r = 2.0f * A * mu[j] * wt[j] * xfac;
    R_ab[idx] = r;
    R_ba[idx] = r;
  }

  if (idx < nwav * N) {
    int w = idx / N;
    int i = idx % N;
    float B_surf;
    if (use_thermal_emission && B_levels != nullptr)
      B_surf = B_levels[w * nlev + (nlev - 1)];
    else if (per_wav_surface_emission != nullptr)
      B_surf = per_wav_surface_emission[w];
    else
      B_surf = surface_emission_scalar;
    s_up[idx]   = (1.0f - A) * B_surf;
    s_down[idx] = 0.0f;

    if (has_solar && A > 0.0f)
      s_up_solar[idx] = (A / PI) * solar_flux * solar_mu
                         * expf(-tau_total[w] / solar_mu);
    else
      s_up_solar[idx] = 0.0f;
    s_down_solar[idx] = 0.0f;
  }
}


// ============================================================================
//  Adding: source combination
// ============================================================================

/// Combined source update for adding.
/// s_up_new = top.s_up + T_ba_D1 * (bot.s_up + bot.R_ab * top.s_down)
/// s_down_new = bot.s_down + T_bc_D2 * (top.s_down + top.R_ba * bot.s_up)
/// The matrix-vector products are done via cuBLAS; this kernel does the vec adds.

__global__ void batchedVecAddKernel(
    int count,
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  c[idx] = a[idx] + b[idx];
}

__global__ void batchedVecCopyKernel(
    int count,
    const float* __restrict__ src,
    float* __restrict__ dst)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  dst[idx] = src[idx];
}

__global__ void batchedMatCopyKernel(
    int count,
    const float* __restrict__ src,
    float* __restrict__ dst)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  dst[idx] = src[idx];
}


// ============================================================================
//  Flux reduction
// ============================================================================

__global__ void batchedFluxReductionKernel(
    int nwav, int N,
    const float* __restrict__ R_ab,        // [nwav * N*N]  composite
    const float* __restrict__ T_ba,        // [nwav * N*N]  composite
    const float* __restrict__ s_up,        // [nwav * N]
    const float* __restrict__ s_up_solar,  // [nwav * N]
    const float* __restrict__ I_top_down,  // [nwav * N]  boundary intensity
    const float* __restrict__ I_bot_up,    // [nwav * N]  boundary intensity
    const float* __restrict__ mu,          // [N]
    const float* __restrict__ wt,          // [N]
    float* __restrict__ flux_up,           // [nwav]
    float* __restrict__ flux_down,         // [nwav]
    float* __restrict__ flux_direct,       // [nwav] or nullptr
    bool has_solar, float solar_flux, float solar_mu,
    const float* __restrict__ tau_total)   // [nwav]
{
  constexpr float PI = 3.14159265f;
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= nwav) return;

  float F_up = 0.0f;
  float F_down = 0.0f;

  for (int i = 0; i < N; ++i) {
    // I_up = R_ab * I_top_down + T_ba * I_bot_up + s_up + s_up_solar
    float Iup_R = 0.0f;
    for (int j = 0; j < N; ++j)
      Iup_R += R_ab[w * N * N + i * N + j] * I_top_down[w * N + j];

    float Iup_T = 0.0f;
    for (int j = 0; j < N; ++j)
      Iup_T += T_ba[w * N * N + i * N + j] * I_bot_up[w * N + j];

    float Iup_i = Iup_R + Iup_T + s_up[w * N + i] + s_up_solar[w * N + i];
    F_up   += 2.0f * PI * wt[i] * mu[i] * Iup_i;
    F_down += 2.0f * PI * wt[i] * mu[i] * I_top_down[w * N + i];
  }

  flux_up[w] = F_up;
  flux_down[w] = F_down;

  if (flux_direct != nullptr) {
    if (has_solar)
      flux_direct[w] = solar_flux * solar_mu * expf(-tau_total[w] / solar_mu);
    else
      flux_direct[w] = 0.0f;
  }
}


// ============================================================================
//  Compute per-wavenumber tau0 and nn (doubling count)
// ============================================================================

__global__ void batchedComputeTau0Kernel(
    int nwav,
    const float* __restrict__ tau,      // [nwav]
    const float* __restrict__ omega,    // [nwav]
    int nn_max,
    float* __restrict__ tau0,           // [nwav]
    float* __restrict__ half_tau0_sq)   // [nwav]
{
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= nwav) return;
  float xfac = 1.0f / exp2f(static_cast<float>(nn_max));
  float t0 = tau[w] * xfac;
  tau0[w] = t0;
  half_tau0_sq[w] = 0.5f * t0 * t0;
}

/// Compute nn_max across all wavenumbers (host-side reduction helper)
inline int computeNnMax(const std::vector<float>& tau_host,
                        const std::vector<float>& omega_host,
                        int nwav)
{
  int nn_max = 2;
  for (int w = 0; w < nwav; ++w) {
    float t = tau_host[w];
    float om = omega_host[w];
    if (t <= 0.0f || om <= 0.0f) continue;

    int ipow0;
    if (om < 0.01f) ipow0 = 2;
    else if (om < 0.1f) ipow0 = 5;
    else ipow0 = 8;

    int nn = static_cast<int>(logf(t) / logf(2.0f)) + ipow0;
    if (nn < 2) nn = 2;
    if (nn > nn_max) nn_max = nn;
  }
  return nn_max;
}


// ============================================================================
//  Compute Planck values at layer boundaries
// ============================================================================

__global__ void batchedPlanckKernel(
    int nwav, int nlev,
    bool use_thermal_emission,
    double wavenumber_low, double wavenumber_high,
    const float* __restrict__ temperature,    // [nwav * nlev] or nullptr
    const float* __restrict__ planck_levels,  // [nwav * nlev] or nullptr
    float* __restrict__ B_levels)             // [nwav * nlev]
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nwav * nlev) return;

  if (use_thermal_emission && temperature != nullptr) {
    B_levels[idx] = planck_function(wavenumber_low, wavenumber_high,
                                    static_cast<double>(temperature[idx]));
  } else if (planck_levels != nullptr) {
    B_levels[idx] = planck_levels[idx];
  } else {
    B_levels[idx] = 0.0f;
  }
}


// ============================================================================
//  Apply delta-M scaling to tau and omega per wavenumber
// ============================================================================

/// Subtract per-wavenumber tau from tau_above: tau_above[w] -= tau_layer[w]
__global__ void batchedSubtractTauKernel(
    int nwav,
    const float* __restrict__ tau_layer,  // [nwav]
    float* __restrict__ tau_above)        // [nwav] in/out
{
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= nwav) return;
  tau_above[w] -= tau_layer[w];
}

__global__ void batchedDeltaMScaleKernel(
    int nwav,
    const float* __restrict__ f_trunc,     // [1] for this layer
    const float* __restrict__ delta_tau_in, // [nwav] original tau for this layer
    float* __restrict__ tau_scaled,         // [nwav]
    float* __restrict__ omega_scaled,       // [nwav] (in/out — read omega, write scaled)
    const float* __restrict__ omega_in)     // [nwav] original omega
{
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= nwav) return;

  float f = f_trunc[0];
  float om = omega_in[w];
  float dt = delta_tau_in[w];

  if (f > 0.0f && om > 0.0f && dt > 0.0f) {
    float omega_f = om * f;
    tau_scaled[w]   = (1.0f - omega_f) * dt;
    omega_scaled[w] = om * (1.0f - f) / (1.0f - omega_f);
  } else {
    tau_scaled[w]   = dt;
    omega_scaled[w] = om;
  }
}


// ============================================================================
//  Boundary intensity setup
// ============================================================================

__global__ void batchedBoundaryIntensityKernel(
    int nwav, int N,
    const float* __restrict__ B_levels,     // [nwav * nlev]
    int nlev,
    float top_emission_scalar,
    float surface_emission_scalar,
    bool use_thermal_emission,
    bool use_diffusion_lower_bc,
    bool has_surface,
    const float* __restrict__ per_wav_top_emission,     // [nwav] or nullptr
    const float* __restrict__ per_wav_surface_emission, // [nwav] or nullptr
    const float* __restrict__ delta_tau,                // [nwav * nlay]
    int nlay,
    const float* __restrict__ mu,   // [N]
    float* __restrict__ I_top_down, // [nwav * N]
    float* __restrict__ I_bot_up)   // [nwav * N]
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nwav * N) return;

  int w = idx / N;
  int i = idx % N;

  // Top boundary
  float B_top;
  if (use_thermal_emission) {
    B_top = B_levels[w * nlev + 0];
  } else if (per_wav_top_emission != nullptr) {
    B_top = per_wav_top_emission[w];
  } else {
    B_top = top_emission_scalar;
  }
  I_top_down[idx] = B_top;

  // Bottom boundary
  if (use_diffusion_lower_bc) {
    float B_bottom = B_levels[w * nlev + nlay];
    float B_second_last = B_levels[w * nlev + nlay - 1];
    float dtau_last = delta_tau[w * nlay + (nlay - 1)];
    float dB_dtau = (dtau_last > 0.0f) ? (B_bottom - B_second_last) / dtau_last : 0.0f;
    I_bot_up[idx] = B_bottom + mu[i] * dB_dtau;
  } else if (!has_surface) {
    float B_surf;
    if (use_thermal_emission) {
      B_surf = B_levels[w * nlev + nlay];
    } else if (per_wav_surface_emission != nullptr) {
      B_surf = per_wav_surface_emission[w];
    } else {
      B_surf = surface_emission_scalar;
    }
    I_bot_up[idx] = B_surf;
  } else {
    I_bot_up[idx] = 0.0f;
  }
}


} // namespace batched
} // namespace cuda
} // namespace adrt
