/// @file test_cuda_solver.cu
/// @brief Validation tests for the CUDA adding-doubling solver.
///
/// Compares CUDA batch solver output against the CPU reference implementation
/// for known benchmark cases.

#include "adding_doubling.hpp"
#include "cuda_solver.cuh"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

// ============================================================================
//  Minimal test framework
// ============================================================================

static int g_passed = 0;
static int g_failed = 0;

static void check_near(
    const char* name, double actual, double expected, double tol,
    const char* file, int line)
{
  double diff = std::abs(actual - expected);
  if (diff <= tol) {
    g_passed++;
  }
  else {
    std::cerr << file << ":" << line << ": FAIL " << name
              << ": |" << actual << " - " << expected << "| = "
              << diff << " > " << tol << "\n";
    g_failed++;
  }
}

#define CHECK_NEAR(name, actual, expected, tol) \
  check_near(name, actual, expected, tol, __FILE__, __LINE__)


// ============================================================================
//  Helper: run CPU solver and return TOA flux
// ============================================================================

static void cpuSolve(
    const adrt::ADConfig& config,
    double& flux_up, double& flux_down)
{
  auto result = adrt::solve(config);
  flux_up = result.flux_up[0];
  flux_down = result.flux_down[0];
}


// ============================================================================
//  Helper: run CUDA batch solver for a single wavenumber
// ============================================================================

static void cudaSolveSingle(
    const adrt::ADConfig& config,
    double& flux_up, double& flux_down)
{
  int nlay = config.num_layers;
  int nlev = nlay + 1;
  int nmom_max = 0;

  for (int l = 0; l < nlay; ++l) {
    int nm = static_cast<int>(config.phase_function_moments[l].size());
    if (nm > nmom_max) nmom_max = nm;
  }

  // Flatten data into SoA format (single wavenumber, so trivial)
  // Convert double → float for the CUDA solver
  std::vector<float> delta_tau(nlay);
  std::vector<float> ssa(nlay);
  for (int l = 0; l < nlay; ++l) {
    delta_tau[l] = static_cast<float>(config.delta_tau[l]);
    ssa[l] = static_cast<float>(config.single_scat_albedo[l]);
  }

  // Flatten phase moments: [nlay * nmom_max], zero-padded
  std::vector<float> pmom(nlay * nmom_max, 0.0f);
  for (int l = 0; l < nlay; ++l) {
    const auto& chi = config.phase_function_moments[l];
    for (int m = 0; m < static_cast<int>(chi.size()); ++m)
      pmom[l * nmom_max + m] = static_cast<float>(chi[m]);
  }

  // Planck levels
  std::vector<float> planck(nlev, 0.0f);
  if (!config.planck_levels.empty()) {
    for (int l = 0; l < nlev; ++l)
      planck[l] = static_cast<float>(config.planck_levels[l]);
  }

  // Temperature
  std::vector<float> temperature;
  if (config.use_thermal_emission) {
    temperature.resize(config.temperature.size());
    for (size_t i = 0; i < config.temperature.size(); ++i)
      temperature[i] = static_cast<float>(config.temperature[i]);
  }

  adrt::cuda::BatchConfig bcfg;
  bcfg.num_wavenumbers = 1;
  bcfg.num_layers = nlay;
  bcfg.num_quadrature = config.num_quadrature;
  bcfg.num_moments_max = nmom_max;
  bcfg.use_delta_m = config.use_delta_m;
  bcfg.use_thermal_emission = config.use_thermal_emission;
  bcfg.use_diffusion_lower_bc = config.use_diffusion_lower_bc;
  bcfg.surface_albedo = config.surface_albedo;
  bcfg.surface_emission = config.surface_emission;
  bcfg.top_emission = config.top_emission;
  bcfg.solar_flux = config.solar_flux;
  bcfg.solar_mu = config.solar_mu;
  bcfg.wavenumber_low = config.wavenumber_low;
  bcfg.wavenumber_high = config.wavenumber_high;

  auto result = adrt::cuda::solveBatchHost(
      bcfg, delta_tau, ssa, pmom, true, planck, temperature);

  flux_up = static_cast<double>(result.flux_up[0]);
  flux_down = static_cast<double>(result.flux_down[0]);
}


// ============================================================================
//  Test cases
// ============================================================================

/// Pure absorption: single layer, no scattering.
void test_pure_absorption() {
  std::cout << "  test_pure_absorption ... ";

  adrt::ADConfig config;
  config.num_layers = 1;
  config.num_quadrature = 8;
  config.allocate();
  config.delta_tau[0] = 1.0;
  config.single_scat_albedo[0] = 0.0;
  config.setIsotropic();
  config.planck_levels = {1.0, 1.0};

  double cpu_up, cpu_down, cuda_up, cuda_down;
  cpuSolve(config, cpu_up, cpu_down);
  cudaSolveSingle(config, cuda_up, cuda_down);

  CHECK_NEAR("flux_up", cuda_up, cpu_up, 1e-6);
  CHECK_NEAR("flux_down", cuda_down, cpu_down, 1e-6);
  std::cout << "done\n";
}


/// Isotropic scattering, conservative (omega = 1).
void test_isotropic_conservative() {
  std::cout << "  test_isotropic_conservative ... ";

  adrt::ADConfig config;
  config.num_layers = 1;
  config.num_quadrature = 8;
  config.allocate();
  config.delta_tau[0] = 0.5;
  config.single_scat_albedo[0] = 1.0;
  config.setIsotropic();
  config.planck_levels = {1.0, 1.0};

  double cpu_up, cpu_down, cuda_up, cuda_down;
  cpuSolve(config, cpu_up, cpu_down);
  cudaSolveSingle(config, cuda_up, cuda_down);

  CHECK_NEAR("flux_up", cuda_up, cpu_up, 1e-4);
  CHECK_NEAR("flux_down", cuda_down, cpu_down, 1e-4);
  std::cout << "done\n";
}


/// Henyey-Greenstein scattering with thermal source.
void test_hg_thermal() {
  std::cout << "  test_hg_thermal ... ";

  adrt::ADConfig config;
  config.num_layers = 5;
  config.num_quadrature = 8;
  config.allocate();

  for (int l = 0; l < 5; ++l) {
    config.delta_tau[l] = 0.5;
    config.single_scat_albedo[l] = 0.8;
  }
  config.setHenyeyGreenstein(0.5);
  config.planck_levels = {100.0, 120.0, 140.0, 160.0, 180.0, 200.0};

  double cpu_up, cpu_down, cuda_up, cuda_down;
  cpuSolve(config, cpu_up, cpu_down);
  cudaSolveSingle(config, cuda_up, cuda_down);

  CHECK_NEAR("flux_up", cuda_up, cpu_up, 1.0);
  CHECK_NEAR("flux_down", cuda_down, cpu_down, 1.0);
  std::cout << "done\n";
}


/// Solar beam with Lambertian surface.
void test_solar_surface() {
  std::cout << "  test_solar_surface ... ";

  adrt::ADConfig config;
  config.num_layers = 3;
  config.num_quadrature = 8;
  config.allocate();

  for (int l = 0; l < 3; ++l) {
    config.delta_tau[l] = 0.2;
    config.single_scat_albedo[l] = 0.9;
  }
  config.setRayleigh();
  config.solar_flux = 1.0;
  config.solar_mu = 0.6;
  config.surface_albedo = 0.3;

  double cpu_up, cpu_down, cuda_up, cuda_down;
  cpuSolve(config, cpu_up, cpu_down);
  cudaSolveSingle(config, cuda_up, cuda_down);

  CHECK_NEAR("flux_up", cuda_up, cpu_up, 1e-3);
  CHECK_NEAR("flux_down", cuda_down, cpu_down, 1e-3);
  std::cout << "done\n";
}


/// Multi-layer atmosphere with varying optical properties.
void test_multilayer() {
  std::cout << "  test_multilayer ... ";

  int nlay = 20;
  adrt::ADConfig config;
  config.num_layers = nlay;
  config.num_quadrature = 8;
  config.allocate();

  for (int l = 0; l < nlay; ++l) {
    config.delta_tau[l] = 0.1 + 0.05 * l;
    config.single_scat_albedo[l] = 0.3 + 0.03 * l;
  }
  config.setHenyeyGreenstein(0.7);

  std::vector<double> planck(nlay + 1);
  for (int l = 0; l <= nlay; ++l)
    planck[l] = 100.0 + 10.0 * l;
  config.planck_levels = planck;
  config.surface_albedo = 0.1;

  double cpu_up, cpu_down, cuda_up, cuda_down;
  cpuSolve(config, cpu_up, cpu_down);
  cudaSolveSingle(config, cuda_up, cuda_down);

  CHECK_NEAR("flux_up", cuda_up, cpu_up, 1.0);
  CHECK_NEAR("flux_down", cuda_down, cpu_down, 1.0);
  std::cout << "done\n";
}


/// Batch consistency: same input across multiple wavenumbers.
void test_batch_consistency() {
  std::cout << "  test_batch_consistency ... ";

  int nlay = 5;
  int nwav = 100;
  int nmom = 3;

  // Set up identical inputs for all wavenumbers
  std::vector<float> delta_tau(nwav * nlay);
  std::vector<float> ssa(nwav * nlay);
  std::vector<float> pmom(nlay * nmom, 0.0f);  // shared
  std::vector<float> planck(nwav * (nlay + 1));

  // Isotropic phase function
  pmom[0 * nmom + 0] = 1.0f;  // chi[0] = 1 for all layers
  for (int l = 1; l < nlay; ++l)
    pmom[l * nmom + 0] = 1.0f;

  for (int w = 0; w < nwav; ++w) {
    for (int l = 0; l < nlay; ++l) {
      delta_tau[w * nlay + l] = 0.5f;
      ssa[w * nlay + l] = 0.8f;
    }
    for (int l = 0; l <= nlay; ++l)
      planck[w * (nlay + 1) + l] = 100.0f + 20.0f * l;
  }

  adrt::cuda::BatchConfig bcfg;
  bcfg.num_wavenumbers = nwav;
  bcfg.num_layers = nlay;
  bcfg.num_quadrature = 8;
  bcfg.num_moments_max = nmom;
  bcfg.surface_albedo = 0.1;

  auto result = adrt::cuda::solveBatchHost(
      bcfg, delta_tau, ssa, pmom, true, planck);

  // All wavenumbers should produce the same result
  bool all_same = true;
  for (int w = 1; w < nwav; ++w) {
    if (std::abs(result.flux_up[w] - result.flux_up[0]) > 1e-6f ||
        std::abs(result.flux_down[w] - result.flux_down[0]) > 1e-6f) {
      all_same = false;
      break;
    }
  }

  if (all_same) {
    g_passed++;
    std::cout << "done\n";
  }
  else {
    g_failed++;
    std::cerr << __FILE__ << ":" << __LINE__
              << ": FAIL batch_consistency: outputs differ across wavenumbers\n";
    std::cout << "FAILED\n";
  }
}


/// Optically thick layer (tau >> 1).
void test_optically_thick() {
  std::cout << "  test_optically_thick ... ";

  adrt::ADConfig config;
  config.num_layers = 1;
  config.num_quadrature = 8;
  config.allocate();
  config.delta_tau[0] = 100.0;
  config.single_scat_albedo[0] = 0.5;
  config.setIsotropic();
  config.planck_levels = {100.0, 200.0};

  double cpu_up, cpu_down, cuda_up, cuda_down;
  cpuSolve(config, cpu_up, cpu_down);
  cudaSolveSingle(config, cuda_up, cuda_down);

  CHECK_NEAR("flux_up", cuda_up, cpu_up, 1.0);
  CHECK_NEAR("flux_down", cuda_down, cpu_down, 1.0);
  std::cout << "done\n";
}


// ============================================================================
//  Main
// ============================================================================

int main() {
  std::cout << "CUDA Adding-Doubling Solver Validation Tests\n";
  std::cout << "=============================================\n\n";

  // Check for CUDA device
  int device_count = 0;
  cudaGetDeviceCount(&device_count);

  if (device_count == 0) {
    std::cerr << "No CUDA devices found. Skipping tests.\n";
    return 0;
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "GPU: " << prop.name << " (compute " << prop.major << "."
            << prop.minor << ")\n\n";

  test_pure_absorption();
  test_isotropic_conservative();
  test_hg_thermal();
  test_solar_surface();
  test_multilayer();
  test_batch_consistency();
  test_optically_thick();

  int total = g_passed + g_failed;
  std::cout << "\n" << g_passed << "/" << total << " checks passed";
  if (g_failed > 0) std::cout << ", " << g_failed << " FAILED";
  std::cout << "\n";

  return g_failed;
}
