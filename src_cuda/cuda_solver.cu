/// @file cuda_solver.cu
/// @brief CUDA kernel and host wrappers for the batched adding-doubling solver.

#include "cuda_solver.cuh"
#include "cuda_adding.cuh"
#include "cuda_doubling.cuh"
#include "cuda_layer.cuh"
#include "cuda_matrix.cuh"
#include "cuda_phase_matrix.cuh"
#include "cuda_planck.cuh"
#include "cuda_quadrature.cuh"

#include <cmath>
#include <cstdio>
#include <vector>

namespace adrt {
namespace cuda {

// ============================================================================
//  Device-side config struct (uploaded via kernel parameter)
// ============================================================================

struct DeviceBatchConfig {
  int nwav;
  int nlay;
  int nmu;
  int nmom_max;

  bool use_delta_m;
  bool use_thermal_emission;
  bool use_diffusion_lower_bc;
  bool phase_moments_shared;

  float surface_albedo;
  float surface_emission;
  float top_emission;
  float solar_flux;
  float solar_mu;
  float wavenumber_low;
  float wavenumber_high;
};


// ============================================================================
//  Precompute kernel: phase matrices for shared moments (one thread per layer)
// ============================================================================

template<int N>
__global__ void precomputePhaseKernel(
    int nlay, int nmom_max, bool use_delta_m,
    float solar_mu, bool has_solar,
    const float* __restrict__ phase_moments,
    float* __restrict__ out_Ppp,        // [nlay * N * N]
    float* __restrict__ out_Ppm,        // [nlay * N * N]
    float* __restrict__ out_f_trunc,    // [nlay]
    float* __restrict__ out_solar_pp,   // [nlay * N] or nullptr
    float* __restrict__ out_solar_pm)   // [nlay * N] or nullptr
{
  int l = blockIdx.x * blockDim.x + threadIdx.x;
  if (l >= nlay) return;

  constexpr int two_M = 2 * N;
  int nmom = nmom_max;

  // Load chi into local buffer
  float chi_buf[MAX_NMOM];
  for (int m = 0; m < nmom; ++m)
    chi_buf[m] = phase_moments[l * nmom_max + m];

  // Delta-M scaling of chi (tau/omega adjustment is per-wavenumber, done in main kernel)
  float f = 0.0f;
  if (use_delta_m) {
    f = (nmom > two_M) ? chi_buf[two_M] : 0.0f;

    if (f > 1e-12f && f < 1.0f - 1e-12f) {
      for (int m = 0; m < two_M; ++m)
        chi_buf[m] = (chi_buf[m] - f) / (1.0f - f);
      nmom = two_M;
    }
    else {
      f = 0.0f;  // no actual truncation
      if (nmom > two_M) nmom = two_M;
    }
  }

  out_f_trunc[l] = f;

  // Compute phase matrices
  GpuMatrix<N> Ppp, Ppm;
  compute_phase_matrices<N>(chi_buf, nmom, Ppp, Ppm);

  // Write to global memory
  int base = l * N * N;
  #pragma unroll
  for (int i = 0; i < N; ++i)
    #pragma unroll
    for (int j = 0; j < N; ++j) {
      out_Ppp[base + i * N + j] = Ppp(i, j);
      out_Ppm[base + i * N + j] = Ppm(i, j);
    }

  // Solar phase vectors
  if (has_solar && out_solar_pp != nullptr && out_solar_pm != nullptr) {
    GpuVec<N> sp, sm;
    compute_solar_phase_vectors<N>(chi_buf, nmom, solar_mu, sp, sm);

    int vbase = l * N;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
      out_solar_pp[vbase + i] = sp[i];
      out_solar_pm[vbase + i] = sm[i];
    }
  }
}


// ============================================================================
//  Main solver kernel: one thread per wavenumber, TOA flux output
// ============================================================================

template<int N>
__global__ void solveKernel(
    DeviceBatchConfig cfg,
    const float* __restrict__ delta_tau,
    const float* __restrict__ single_scat_albedo,
    const float* __restrict__ phase_moments,
    const float* __restrict__ temperature,
    const float* __restrict__ planck_levels,
    const float* __restrict__ per_wav_surface_emission,
    const float* __restrict__ per_wav_top_emission,
    float* __restrict__ flux_up_out,
    float* __restrict__ flux_down_out,
    float* __restrict__ flux_direct_out,
    // Precomputed phase matrices (nullptr if not precomputed)
    const float* __restrict__ precomp_Ppp,
    const float* __restrict__ precomp_Ppm,
    const float* __restrict__ precomp_f_trunc,
    const float* __restrict__ precomp_solar_pp,
    const float* __restrict__ precomp_solar_pm)
{
  constexpr float PI = 3.14159265f;

  int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= cfg.nwav) return;

  int nlay = cfg.nlay;
  int nlev = nlay + 1;
  int two_M = 2 * N;
  bool has_solar = (cfg.solar_flux > 0.0f && cfg.solar_mu > 0.0f);

  // --- 1. Planck / thermal source setup ---
  // Instead of storing all B[nlev] on the stack, we load values on the fly
  // during the layer loop. Here we only determine the surface/top values.

  // Helper lambda to load a single Planck level value
  auto loadB = [&](int level) -> float {
    if (cfg.use_thermal_emission && temperature != nullptr)
      return planck_function(
          static_cast<double>(cfg.wavenumber_low),
          static_cast<double>(cfg.wavenumber_high),
          static_cast<double>(temperature[w * nlev + level]));
    else if (planck_levels != nullptr)
      return planck_levels[w * nlev + level];
    else
      return 0.0f;
  };

  float B_surface = 0.0f;
  float B_top_emission = 0.0f;

  if (cfg.use_thermal_emission && temperature != nullptr) {
    B_surface = loadB(nlay);
    // B_top_emission will be set after the layer loop (= B at level 0)
  }
  else {
    B_surface = (per_wav_surface_emission != nullptr)
                  ? per_wav_surface_emission[w] : cfg.surface_emission;
    B_top_emission = (per_wav_top_emission != nullptr)
                       ? per_wav_top_emission[w] : cfg.top_emission;
  }

  // --- 2. Process layers: doubling + bottom-up adding (fused) ---
  GpuLayerMatrices<N> rbase;
  rbase.set_transparent();
  bool rbase_empty = true;

  bool has_precomp = (precomp_Ppp != nullptr);

  // Temporary for Legendre moments (only needed when not precomputed)
  float chi_buf[MAX_NMOM];

  // Compute total optical depth (needed for surface solar beam)
  float tau_total = 0.0f;
  for (int l = 0; l < nlay; ++l)
    tau_total += delta_tau[w * nlay + l];

  // Surface layer (if applicable)
  bool has_surface = !cfg.use_diffusion_lower_bc
                     && (cfg.surface_albedo > 0.0f || B_surface > 0.0f);

  if (has_surface) {
    GpuLayerMatrices<N> surf;
    mat_set_zero<N>(surf.T_ab);
    mat_set_zero<N>(surf.T_ba);
    mat_set_zero<N>(surf.R_ab);
    mat_set_zero<N>(surf.R_ba);
    vec_set_zero<N>(surf.s_up);
    vec_set_zero<N>(surf.s_down);
    vec_set_zero<N>(surf.s_up_solar);
    vec_set_zero<N>(surf.s_down_solar);

    float A = cfg.surface_albedo;

    #pragma unroll
    for (int i = 0; i < N; ++i) {
      #pragma unroll
      for (int j = 0; j < N; ++j) {
        float r = 2.0f * A * d_mu[j] * d_wt[j] * d_xfac;
        surf.R_ab(i, j) = r;
        surf.R_ba(i, j) = r;
      }
      surf.s_up[i]   = (1.0f - A) * B_surface;
      surf.s_down[i] = 0.0f;

      if (has_solar && A > 0.0f) {
        surf.s_up_solar[i] = (A / PI) * cfg.solar_flux * cfg.solar_mu
                             * expf(-tau_total / cfg.solar_mu);
      }
    }

    surf.is_scattering = (A > 0.0f);
    rbase = surf;
    rbase_empty = false;
  }

  // Process layers from bottom to top.
  float tau_above = tau_total;
  float B_prev = loadB(nlay);

  for (int l = nlay - 1; l >= 0; --l) {
    float tau_layer = delta_tau[w * nlay + l];
    float omega_layer = single_scat_albedo[w * nlay + l];
    float B_layer_top = loadB(l);
    float B_layer_bot = B_prev;
    B_prev = B_layer_top;

    tau_above -= tau_layer;
    float tau_cumulative = tau_above;

    // Build phase matrices (precomputed or on-the-fly)
    GpuMatrix<N> Ppp, Ppm;
    GpuVec<N> p_plus_solar, p_minus_solar;
    bool has_solar_phase = false;

    if (has_precomp) {
      float f_trunc = precomp_f_trunc[l];
      if (cfg.use_delta_m && f_trunc > 0.0f && omega_layer > 0.0f && tau_layer > 0.0f) {
        float omega_f = omega_layer * f_trunc;
        tau_layer   = (1.0f - omega_f) * delta_tau[w * nlay + l];
        omega_layer = omega_layer * (1.0f - f_trunc) / (1.0f - omega_f);
      }

      if (omega_layer > 0.0f && tau_layer > 0.0f) {
        int base = l * N * N;
        #pragma unroll
        for (int i = 0; i < N; ++i)
          #pragma unroll
          for (int j = 0; j < N; ++j) {
            Ppp(i, j) = precomp_Ppp[base + i * N + j];
            Ppm(i, j) = precomp_Ppm[base + i * N + j];
          }

        if (has_solar && precomp_solar_pp != nullptr) {
          int vbase = l * N;
          #pragma unroll
          for (int i = 0; i < N; ++i) {
            p_plus_solar[i]  = precomp_solar_pp[vbase + i];
            p_minus_solar[i] = precomp_solar_pm[vbase + i];
          }
          has_solar_phase = true;
        }
      }
      else {
        mat_set_zero<N>(Ppp);
        mat_set_zero<N>(Ppm);
        vec_set_zero<N>(p_plus_solar);
        vec_set_zero<N>(p_minus_solar);
      }
    }
    else {
      int nmom = cfg.nmom_max;
      for (int m = 0; m < nmom; ++m)
        chi_buf[m] = phase_moments[w * nlay * nmom + l * nmom + m];

      if (cfg.use_delta_m && omega_layer > 0.0f && tau_layer > 0.0f) {
        float f_trunc = (nmom > two_M) ? chi_buf[two_M] : 0.0f;

        if (f_trunc > 1e-12f && f_trunc < 1.0f - 1e-12f) {
          float omega_f = omega_layer * f_trunc;
          tau_layer   = (1.0f - omega_f) * delta_tau[w * nlay + l];
          omega_layer = omega_layer * (1.0f - f_trunc) / (1.0f - omega_f);

          for (int m = 0; m < two_M; ++m)
            chi_buf[m] = (chi_buf[m] - f_trunc) / (1.0f - f_trunc);
          nmom = two_M;
        }
        else {
          if (nmom > two_M) nmom = two_M;
        }
      }

      if (omega_layer > 0.0f && tau_layer > 0.0f) {
        compute_phase_matrices<N>(chi_buf, nmom, Ppp, Ppm);

        if (has_solar) {
          compute_solar_phase_vectors<N>(chi_buf, nmom, cfg.solar_mu,
                                         p_plus_solar, p_minus_solar);
          has_solar_phase = true;
        }
      }
      else {
        mat_set_zero<N>(Ppp);
        mat_set_zero<N>(Ppm);
        vec_set_zero<N>(p_plus_solar);
        vec_set_zero<N>(p_minus_solar);
      }
    }

    // Doubling
    GpuLayerMatrices<N> layer;
    doubling<N>(layer, tau_layer, omega_layer, B_layer_top, B_layer_bot,
                Ppp, Ppm, cfg.solar_flux, cfg.solar_mu, tau_cumulative,
                has_solar_phase ? &p_plus_solar : nullptr,
                has_solar_phase ? &p_minus_solar : nullptr,
                has_solar_phase);

    // Fold into bottom-up composite via adding
    if (rbase_empty) {
      rbase = layer;
      rbase_empty = false;
    }
    else {
      GpuLayerMatrices<N> combined;
      add_layers<N>(combined, layer, rbase);
      rbase = combined;
    }
  }

  // B_prev now holds B[0] (the topmost level value)
  if (cfg.use_thermal_emission && temperature != nullptr)
    B_top_emission = B_prev;

  // --- 3. Boundary intensities ---
  GpuVec<N> I_top_down;
  vec_set_scalar<N>(I_top_down, B_top_emission);

  GpuVec<N> I_bot_up;
  vec_set_zero<N>(I_bot_up);

  if (cfg.use_diffusion_lower_bc) {
    float B_bottom = loadB(nlay);
    float B_second_last = loadB(nlay - 1);
    float dtau_last = delta_tau[w * nlay + (nlay - 1)];
    float dB_dtau = (dtau_last > 0.0f) ? (B_bottom - B_second_last) / dtau_last : 0.0f;
    #pragma unroll
    for (int i = 0; i < N; ++i)
      I_bot_up[i] = B_bottom + d_mu[i] * dB_dtau;
  }
  else if (!has_surface) {
    vec_set_scalar<N>(I_bot_up, B_surface);
  }

  // --- 4. Compute TOA upward flux ---
  GpuVec<N> Iup_term1, Iup_term2;
  mat_vec_multiply<N>(Iup_term1, rbase.R_ab, I_top_down);
  mat_vec_multiply<N>(Iup_term2, rbase.T_ba, I_bot_up);

  float F_up = 0.0f;
  float F_down = 0.0f;

  #pragma unroll
  for (int i = 0; i < N; ++i) {
    float Iup_i = Iup_term1[i] + Iup_term2[i]
                   + rbase.s_up[i] + rbase.s_up_solar[i];
    F_up   += 2.0f * PI * d_wt[i] * d_mu[i] * Iup_i;
    F_down += 2.0f * PI * d_wt[i] * d_mu[i] * I_top_down[i];
  }

  if (flux_up_out != nullptr)
    flux_up_out[w] = F_up;
  if (flux_down_out != nullptr)
    flux_down_out[w] = F_down;

  // Direct solar flux at surface
  if (flux_direct_out != nullptr) {
    if (has_solar)
      flux_direct_out[w] = cfg.solar_flux * cfg.solar_mu
                           * expf(-tau_total / cfg.solar_mu);
    else
      flux_direct_out[w] = 0.0f;
  }
}


// ============================================================================
//  Host-side kernel launch
// ============================================================================

void solveBatch(
    const BatchConfig& config,
    const DeviceData& data,
    cudaStream_t stream)
{
  // Upload quadrature data to constant memory
  int L = config.num_moments_max;
  if (L > MAX_NMOM) L = MAX_NMOM;
  uploadQuadratureData(config.num_quadrature, L);

  // Build device config
  DeviceBatchConfig dcfg;
  dcfg.nwav = config.num_wavenumbers;
  dcfg.nlay = config.num_layers;
  dcfg.nmu  = config.num_quadrature;
  dcfg.nmom_max = config.num_moments_max;
  dcfg.use_delta_m = config.use_delta_m;
  dcfg.use_thermal_emission = config.use_thermal_emission;
  dcfg.use_diffusion_lower_bc = config.use_diffusion_lower_bc;
  dcfg.phase_moments_shared = data.phase_moments_shared;
  dcfg.surface_albedo = static_cast<float>(config.surface_albedo);
  dcfg.surface_emission = static_cast<float>(config.surface_emission);
  dcfg.top_emission = static_cast<float>(config.top_emission);
  dcfg.solar_flux = static_cast<float>(config.solar_flux);
  dcfg.solar_mu = static_cast<float>(config.solar_mu);
  dcfg.wavenumber_low = static_cast<float>(config.wavenumber_low);
  dcfg.wavenumber_high = static_cast<float>(config.wavenumber_high);

  // --- Precompute phase matrices for shared moments ---
  float *d_precomp_Ppp = nullptr, *d_precomp_Ppm = nullptr;
  float *d_precomp_f_trunc = nullptr;
  float *d_precomp_solar_pp = nullptr, *d_precomp_solar_pm = nullptr;

  int nlay = config.num_layers;
  int N = config.num_quadrature;
  bool has_solar = (config.solar_flux > 0.0 && config.solar_mu > 0.0);

  if (data.phase_moments_shared) {
    cudaMalloc(&d_precomp_Ppp, nlay * N * N * sizeof(float));
    cudaMalloc(&d_precomp_Ppm, nlay * N * N * sizeof(float));
    cudaMalloc(&d_precomp_f_trunc, nlay * sizeof(float));

    if (has_solar) {
      cudaMalloc(&d_precomp_solar_pp, nlay * N * sizeof(float));
      cudaMalloc(&d_precomp_solar_pm, nlay * N * sizeof(float));
    }

    int precomp_threads = 64;
    int precomp_blocks = (nlay + precomp_threads - 1) / precomp_threads;

    float solar_mu_f = static_cast<float>(config.solar_mu);

    switch (config.num_quadrature) {
      case 2:
        precomputePhaseKernel<2><<<precomp_blocks, precomp_threads, 0, stream>>>(
            nlay, config.num_moments_max, config.use_delta_m,
            solar_mu_f, has_solar, data.phase_moments,
            d_precomp_Ppp, d_precomp_Ppm, d_precomp_f_trunc,
            d_precomp_solar_pp, d_precomp_solar_pm);
        break;
      case 4:
        precomputePhaseKernel<4><<<precomp_blocks, precomp_threads, 0, stream>>>(
            nlay, config.num_moments_max, config.use_delta_m,
            solar_mu_f, has_solar, data.phase_moments,
            d_precomp_Ppp, d_precomp_Ppm, d_precomp_f_trunc,
            d_precomp_solar_pp, d_precomp_solar_pm);
        break;
      case 8:
        precomputePhaseKernel<8><<<precomp_blocks, precomp_threads, 0, stream>>>(
            nlay, config.num_moments_max, config.use_delta_m,
            solar_mu_f, has_solar, data.phase_moments,
            d_precomp_Ppp, d_precomp_Ppm, d_precomp_f_trunc,
            d_precomp_solar_pp, d_precomp_solar_pm);
        break;
      case 16:
        precomputePhaseKernel<16><<<precomp_blocks, precomp_threads, 0, stream>>>(
            nlay, config.num_moments_max, config.use_delta_m,
            solar_mu_f, has_solar, data.phase_moments,
            d_precomp_Ppp, d_precomp_Ppm, d_precomp_f_trunc,
            d_precomp_solar_pp, d_precomp_solar_pm);
        break;
      case 32:
        precomputePhaseKernel<32><<<precomp_blocks, precomp_threads, 0, stream>>>(
            nlay, config.num_moments_max, config.use_delta_m,
            solar_mu_f, has_solar, data.phase_moments,
            d_precomp_Ppp, d_precomp_Ppm, d_precomp_f_trunc,
            d_precomp_solar_pp, d_precomp_solar_pm);
        break;
    }
  }

  int threads_per_block = 128;
  int num_blocks = (config.num_wavenumbers + threads_per_block - 1) / threads_per_block;

  // Dispatch to template specialisation
  #define LAUNCH_SOLVE_KERNEL(NQ) \
    solveKernel<NQ><<<num_blocks, threads_per_block, 0, stream>>>( \
        dcfg, data.delta_tau, data.single_scat_albedo, data.phase_moments, \
        data.temperature, data.planck_levels, \
        data.surface_emission, data.top_emission, \
        data.flux_up, data.flux_down, data.flux_direct, \
        d_precomp_Ppp, d_precomp_Ppm, d_precomp_f_trunc, \
        d_precomp_solar_pp, d_precomp_solar_pm)

  switch (config.num_quadrature) {
    case 2:  LAUNCH_SOLVE_KERNEL(2);  break;
    case 4:  LAUNCH_SOLVE_KERNEL(4);  break;
    case 8:  LAUNCH_SOLVE_KERNEL(8);  break;
    case 16: LAUNCH_SOLVE_KERNEL(16); break;
    case 32: LAUNCH_SOLVE_KERNEL(32); break;
    default: break;
  }

  #undef LAUNCH_SOLVE_KERNEL

  // Free precomputed arrays
  if (stream != 0)
    cudaStreamSynchronize(stream);

  if (d_precomp_Ppp) cudaFree(d_precomp_Ppp);
  if (d_precomp_Ppm) cudaFree(d_precomp_Ppm);
  if (d_precomp_f_trunc) cudaFree(d_precomp_f_trunc);
  if (d_precomp_solar_pp) cudaFree(d_precomp_solar_pp);
  if (d_precomp_solar_pm) cudaFree(d_precomp_solar_pm);
}


// ============================================================================
//  Convenience: solve from host data
// ============================================================================

HostResult solveBatchHost(
    const BatchConfig& config,
    const std::vector<float>& delta_tau,
    const std::vector<float>& single_scat_albedo,
    const std::vector<float>& phase_moments,
    bool phase_moments_shared,
    const std::vector<float>& planck_levels,
    const std::vector<float>& temperature)
{
  int nwav = config.num_wavenumbers;

  // Allocate device memory
  float *d_dtau = nullptr, *d_ssa = nullptr, *d_pmom = nullptr;
  float *d_planck = nullptr, *d_temp = nullptr;
  float *d_flux_up = nullptr, *d_flux_down = nullptr, *d_flux_direct = nullptr;

  cudaMalloc(&d_dtau, delta_tau.size() * sizeof(float));
  cudaMalloc(&d_ssa, single_scat_albedo.size() * sizeof(float));
  cudaMalloc(&d_pmom, phase_moments.size() * sizeof(float));

  cudaMemcpy(d_dtau, delta_tau.data(), delta_tau.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_ssa, single_scat_albedo.data(), single_scat_albedo.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_pmom, phase_moments.data(), phase_moments.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  if (!planck_levels.empty()) {
    cudaMalloc(&d_planck, planck_levels.size() * sizeof(float));
    cudaMemcpy(d_planck, planck_levels.data(), planck_levels.size() * sizeof(float),
               cudaMemcpyHostToDevice);
  }

  if (!temperature.empty()) {
    cudaMalloc(&d_temp, temperature.size() * sizeof(float));
    cudaMemcpy(d_temp, temperature.data(), temperature.size() * sizeof(float),
               cudaMemcpyHostToDevice);
  }

  cudaMalloc(&d_flux_up, nwav * sizeof(float));
  cudaMalloc(&d_flux_down, nwav * sizeof(float));
  cudaMalloc(&d_flux_direct, nwav * sizeof(float));

  // Set up DeviceData
  DeviceData data;
  data.delta_tau = d_dtau;
  data.single_scat_albedo = d_ssa;
  data.phase_moments = d_pmom;
  data.phase_moments_shared = phase_moments_shared;
  data.temperature = d_temp;
  data.planck_levels = d_planck;
  data.flux_up = d_flux_up;
  data.flux_down = d_flux_down;
  data.flux_direct = d_flux_direct;

  // Launch
  solveBatch(config, data);
  cudaDeviceSynchronize();

  // Copy results back
  HostResult result;
  result.flux_up.resize(nwav);
  result.flux_down.resize(nwav);
  result.flux_direct.resize(nwav);

  cudaMemcpy(result.flux_up.data(), d_flux_up, nwav * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(result.flux_down.data(), d_flux_down, nwav * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(result.flux_direct.data(), d_flux_direct, nwav * sizeof(float),
             cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_dtau);
  cudaFree(d_ssa);
  cudaFree(d_pmom);
  if (d_planck) cudaFree(d_planck);
  if (d_temp) cudaFree(d_temp);
  cudaFree(d_flux_up);
  cudaFree(d_flux_down);
  cudaFree(d_flux_direct);

  return result;
}

} // namespace cuda
} // namespace adrt
