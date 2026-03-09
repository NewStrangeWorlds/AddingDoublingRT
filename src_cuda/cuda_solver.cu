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

  double surface_albedo;
  double surface_emission;
  double top_emission;
  double solar_flux;
  double solar_mu;
  double wavenumber_low;
  double wavenumber_high;
};


// ============================================================================
//  Main solver kernel: one thread per wavenumber, TOA flux output
// ============================================================================

template<int N>
__global__ void solveKernel(
    DeviceBatchConfig cfg,
    const double* __restrict__ delta_tau,
    const double* __restrict__ single_scat_albedo,
    const double* __restrict__ phase_moments,
    const double* __restrict__ temperature,
    const double* __restrict__ planck_levels,
    const double* __restrict__ per_wav_surface_emission,
    const double* __restrict__ per_wav_top_emission,
    double* __restrict__ flux_up_out,
    double* __restrict__ flux_down_out,
    double* __restrict__ flux_direct_out)
{
  constexpr double PI = 3.14159265358979323846;

  int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= cfg.nwav) return;

  int nlay = cfg.nlay;
  int nlev = nlay + 1;
  int two_M = 2 * N;
  bool has_solar = (cfg.solar_flux > 0.0 && cfg.solar_mu > 0.0);

  // --- 1. Planck values at each level ---
  // Use a small stack array (nlay+1 levels). For 100 layers this is 808 bytes.
  double B[128];  // Support up to 127 layers
  double B_surface = 0.0;
  double B_top_emission = 0.0;

  if (cfg.use_thermal_emission && temperature != nullptr) {
    for (int l = 0; l <= nlay; ++l)
      B[l] = planck_function(cfg.wavenumber_low, cfg.wavenumber_high,
                             temperature[w * nlev + l]);
    B_surface = B[nlay];
    B_top_emission = B[0];
  }
  else if (planck_levels != nullptr) {
    for (int l = 0; l <= nlay; ++l)
      B[l] = planck_levels[w * nlev + l];
    B_surface = (per_wav_surface_emission != nullptr)
                  ? per_wav_surface_emission[w] : cfg.surface_emission;
    B_top_emission = (per_wav_top_emission != nullptr)
                       ? per_wav_top_emission[w] : cfg.top_emission;
  }
  else {
    for (int l = 0; l <= nlay; ++l)
      B[l] = 0.0;
    B_surface = (per_wav_surface_emission != nullptr)
                  ? per_wav_surface_emission[w] : cfg.surface_emission;
    B_top_emission = (per_wav_top_emission != nullptr)
                       ? per_wav_top_emission[w] : cfg.top_emission;
  }

  // --- 2. Process layers: doubling + bottom-up adding (fused) ---
  // We build the bottom-up composite incrementally:
  //   rbase starts empty, then we fold in layers from bottom to top.
  //
  // But for TOA flux we need the composite of ALL layers viewed from the top.
  // The adding order is: start with bottom layer, add the next layer above, etc.
  // So we process layers from bottom (l=nlay-1) to top (l=0).

  GpuLayerMatrices<N> rbase;
  rbase.set_transparent();
  bool rbase_empty = true;

  // Temporary for Legendre moments (per-layer, loaded from global memory)
  double chi_buf[MAX_NMOM];

  double tau_total = 0.0;

  // First pass: compute cumulative tau for solar beam
  double tau_cumulative_arr[128];
  {
    double tc = 0.0;
    for (int l = 0; l < nlay; ++l) {
      tau_cumulative_arr[l] = tc;
      tc += delta_tau[w * nlay + l];
    }
    tau_total = tc;
  }

  // Surface layer (if applicable)
  bool has_surface = !cfg.use_diffusion_lower_bc
                     && (cfg.surface_albedo > 0.0 || B_surface > 0.0);

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

    double A = cfg.surface_albedo;

    #pragma unroll
    for (int i = 0; i < N; ++i) {
      #pragma unroll
      for (int j = 0; j < N; ++j) {
        double r = 2.0 * A * d_mu[j] * d_wt[j] * d_xfac;
        surf.R_ab(i, j) = r;
        surf.R_ba(i, j) = r;
      }
      surf.s_up[i]   = (1.0 - A) * B_surface;
      surf.s_down[i] = 0.0;

      if (has_solar && A > 0.0) {
        surf.s_up_solar[i] = (A / PI) * cfg.solar_flux * cfg.solar_mu
                             * exp(-tau_total / cfg.solar_mu);
      }
    }

    surf.is_scattering = (A > 0.0);
    rbase = surf;
    rbase_empty = false;
  }

  // Process layers from bottom to top
  for (int l = nlay - 1; l >= 0; --l) {
    double tau_layer = delta_tau[w * nlay + l];
    double omega_layer = single_scat_albedo[w * nlay + l];
    double B_layer_top = B[l];
    double B_layer_bot = B[l + 1];
    double tau_cumulative = tau_cumulative_arr[l];

    // Load phase function moments
    int nmom = cfg.nmom_max;
    if (cfg.phase_moments_shared) {
      for (int m = 0; m < nmom; ++m)
        chi_buf[m] = phase_moments[l * nmom + m];
    }
    else {
      for (int m = 0; m < nmom; ++m)
        chi_buf[m] = phase_moments[w * nlay * nmom + l * nmom + m];
    }

    // Delta-M scaling
    if (cfg.use_delta_m && omega_layer > 0.0 && tau_layer > 0.0) {
      double f_trunc = (nmom > two_M) ? chi_buf[two_M] : 0.0;

      if (f_trunc > 1e-12 && f_trunc < 1.0 - 1e-12) {
        double omega_f = omega_layer * f_trunc;
        tau_layer   = (1.0 - omega_f) * delta_tau[w * nlay + l];
        omega_layer = omega_layer * (1.0 - f_trunc) / (1.0 - omega_f);

        // Truncate and scale moments
        for (int m = 0; m < two_M; ++m)
          chi_buf[m] = (chi_buf[m] - f_trunc) / (1.0 - f_trunc);
        nmom = two_M;
      }
      else {
        if (nmom > two_M) nmom = two_M;
      }
    }

    // Build phase matrices
    GpuMatrix<N> Ppp, Ppm;
    GpuVec<N> p_plus_solar, p_minus_solar;
    bool has_solar_phase = false;

    if (omega_layer > 0.0 && tau_layer > 0.0) {
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

  // --- 3. Boundary intensities ---
  GpuVec<N> I_top_down;
  vec_set_scalar<N>(I_top_down, B_top_emission);

  GpuVec<N> I_bot_up;
  vec_set_zero<N>(I_bot_up);

  if (cfg.use_diffusion_lower_bc) {
    double B_bottom = B[nlay];
    double dtau_last = delta_tau[w * nlay + (nlay - 1)];
    double dB_dtau = (dtau_last > 0.0) ? (B_bottom - B[nlay - 1]) / dtau_last : 0.0;
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

  double F_up = 0.0;
  double F_down = 0.0;

  #pragma unroll
  for (int i = 0; i < N; ++i) {
    double Iup_i = Iup_term1[i] + Iup_term2[i]
                   + rbase.s_up[i] + rbase.s_up_solar[i];
    F_up   += 2.0 * PI * d_wt[i] * d_mu[i] * Iup_i;
    F_down += 2.0 * PI * d_wt[i] * d_mu[i] * I_top_down[i];
  }

  if (flux_up_out != nullptr)
    flux_up_out[w] = F_up;
  if (flux_down_out != nullptr)
    flux_down_out[w] = F_down;

  // Direct solar flux at surface
  if (flux_direct_out != nullptr) {
    if (has_solar)
      flux_direct_out[w] = cfg.solar_flux * cfg.solar_mu
                           * exp(-tau_total / cfg.solar_mu);
    else
      flux_direct_out[w] = 0.0;
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
  dcfg.surface_albedo = config.surface_albedo;
  dcfg.surface_emission = config.surface_emission;
  dcfg.top_emission = config.top_emission;
  dcfg.solar_flux = config.solar_flux;
  dcfg.solar_mu = config.solar_mu;
  dcfg.wavenumber_low = config.wavenumber_low;
  dcfg.wavenumber_high = config.wavenumber_high;

  int threads_per_block = 128;
  int num_blocks = (config.num_wavenumbers + threads_per_block - 1) / threads_per_block;

  // Dispatch to template specialisation
  switch (config.num_quadrature) {
    case 2:
      solveKernel<2><<<num_blocks, threads_per_block, 0, stream>>>(
          dcfg, data.delta_tau, data.single_scat_albedo, data.phase_moments,
          data.temperature, data.planck_levels,
          data.surface_emission, data.top_emission,
          data.flux_up, data.flux_down, data.flux_direct);
      break;
    case 4:
      solveKernel<4><<<num_blocks, threads_per_block, 0, stream>>>(
          dcfg, data.delta_tau, data.single_scat_albedo, data.phase_moments,
          data.temperature, data.planck_levels,
          data.surface_emission, data.top_emission,
          data.flux_up, data.flux_down, data.flux_direct);
      break;
    case 8:
      solveKernel<8><<<num_blocks, threads_per_block, 0, stream>>>(
          dcfg, data.delta_tau, data.single_scat_albedo, data.phase_moments,
          data.temperature, data.planck_levels,
          data.surface_emission, data.top_emission,
          data.flux_up, data.flux_down, data.flux_direct);
      break;
    case 16:
      solveKernel<16><<<num_blocks, threads_per_block, 0, stream>>>(
          dcfg, data.delta_tau, data.single_scat_albedo, data.phase_moments,
          data.temperature, data.planck_levels,
          data.surface_emission, data.top_emission,
          data.flux_up, data.flux_down, data.flux_direct);
      break;
    case 32:
      solveKernel<32><<<num_blocks, threads_per_block, 0, stream>>>(
          dcfg, data.delta_tau, data.single_scat_albedo, data.phase_moments,
          data.temperature, data.planck_levels,
          data.surface_emission, data.top_emission,
          data.flux_up, data.flux_down, data.flux_direct);
      break;
    default:
      // Unsupported quadrature order
      break;
  }
}


// ============================================================================
//  Convenience: solve from host data
// ============================================================================

HostResult solveBatchHost(
    const BatchConfig& config,
    const std::vector<double>& delta_tau,
    const std::vector<double>& single_scat_albedo,
    const std::vector<double>& phase_moments,
    bool phase_moments_shared,
    const std::vector<double>& planck_levels,
    const std::vector<double>& temperature)
{
  int nwav = config.num_wavenumbers;

  // Allocate device memory
  double *d_dtau = nullptr, *d_ssa = nullptr, *d_pmom = nullptr;
  double *d_planck = nullptr, *d_temp = nullptr;
  double *d_flux_up = nullptr, *d_flux_down = nullptr, *d_flux_direct = nullptr;

  cudaMalloc(&d_dtau, delta_tau.size() * sizeof(double));
  cudaMalloc(&d_ssa, single_scat_albedo.size() * sizeof(double));
  cudaMalloc(&d_pmom, phase_moments.size() * sizeof(double));

  cudaMemcpy(d_dtau, delta_tau.data(), delta_tau.size() * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_ssa, single_scat_albedo.data(), single_scat_albedo.size() * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_pmom, phase_moments.data(), phase_moments.size() * sizeof(double),
             cudaMemcpyHostToDevice);

  if (!planck_levels.empty()) {
    cudaMalloc(&d_planck, planck_levels.size() * sizeof(double));
    cudaMemcpy(d_planck, planck_levels.data(), planck_levels.size() * sizeof(double),
               cudaMemcpyHostToDevice);
  }

  if (!temperature.empty()) {
    cudaMalloc(&d_temp, temperature.size() * sizeof(double));
    cudaMemcpy(d_temp, temperature.data(), temperature.size() * sizeof(double),
               cudaMemcpyHostToDevice);
  }

  cudaMalloc(&d_flux_up, nwav * sizeof(double));
  cudaMalloc(&d_flux_down, nwav * sizeof(double));
  cudaMalloc(&d_flux_direct, nwav * sizeof(double));

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

  cudaMemcpy(result.flux_up.data(), d_flux_up, nwav * sizeof(double),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(result.flux_down.data(), d_flux_down, nwav * sizeof(double),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(result.flux_direct.data(), d_flux_direct, nwav * sizeof(double),
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
