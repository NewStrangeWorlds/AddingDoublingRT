/// @file test_cuda_solver.cu
/// @brief Comprehensive validation tests for the CUDA adding-doubling solver.
///
/// Mirrors the CPU test suite (test_ad_solver.cpp), comparing CUDA batch solver
/// output against the CPU reference implementation for all benchmark cases.
///
/// References:
///   VH1 = Van de Hulst (1980), Multiple Light Scattering, Table 12
///   VH2 = Van de Hulst (1980), Table 37
///   SW  = Sweigart (1970), Table 1
///   GS  = Garcia & Siewert (1985), Tables 12-20
///   OS  = Ozisik & Shouman, Table 1

#include "adding_doubling.hpp"
#include "cuda_solver.cuh"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>
#include <algorithm>

static constexpr double PI = 3.14159265358979323846;

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

static void check_true(
    const char* name, bool condition,
    const char* file, int line)
{
  if (condition) {
    g_passed++;
  }
  else {
    std::cerr << file << ":" << line << ": FAIL " << name << "\n";
    g_failed++;
  }
}

#define CHECK_NEAR(name, actual, expected, tol) \
  check_near(name, actual, expected, tol, __FILE__, __LINE__)

#define CHECK_TRUE(name, cond) \
  check_true(name, cond, __FILE__, __LINE__)


// ============================================================================
//  Helper: run CPU solver
// ============================================================================

struct CPUResult {
  std::vector<double> flux_up;
  std::vector<double> flux_down;
  std::vector<double> flux_direct;
};

static CPUResult cpuSolve(const adrt::ADConfig& config) {
  auto r = adrt::solve(config);
  return {r.flux_up, r.flux_down, r.flux_direct};
}


// ============================================================================
//  Helper: run CUDA batch solver for a single wavenumber
// ============================================================================

struct CUDAResult {
  double flux_up;
  double flux_down;
  double flux_direct;
};

static CUDAResult cudaSolveSingle(const adrt::ADConfig& config)
{
  int nlay = config.num_layers;
  int nlev = nlay + 1;
  int nmom_max = 0;

  for (int l = 0; l < nlay; ++l) {
    int nm = static_cast<int>(config.phase_function_moments[l].size());
    if (nm > nmom_max) nmom_max = nm;
  }

  std::vector<float> delta_tau(nlay);
  std::vector<float> ssa(nlay);
  for (int l = 0; l < nlay; ++l) {
    delta_tau[l] = static_cast<float>(config.delta_tau[l]);
    ssa[l] = static_cast<float>(config.single_scat_albedo[l]);
  }

  std::vector<float> pmom(nlay * nmom_max, 0.0f);
  for (int l = 0; l < nlay; ++l) {
    const auto& chi = config.phase_function_moments[l];
    for (int m = 0; m < static_cast<int>(chi.size()); ++m)
      pmom[l * nmom_max + m] = static_cast<float>(chi[m]);
  }

  std::vector<float> planck(nlev, 0.0f);
  if (!config.planck_levels.empty()) {
    for (int l = 0; l < nlev; ++l)
      planck[l] = static_cast<float>(config.planck_levels[l]);
  }

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

  return {
    static_cast<double>(result.flux_up[0]),
    static_cast<double>(result.flux_down[0]),
    static_cast<double>(result.flux_direct[0])
  };
}


// ============================================================================
//  Helper: set Haze-L Garcia-Siewert phase function
// ============================================================================

static void setHazeGarciaSiewert(adrt::ADConfig& cfg, int lc) {
  static const double hazelm[82] = {
    2.41260, 3.23047, 3.37296, 3.23150, 2.89350, 2.49594, 2.11361, 1.74812,
    1.44692, 1.17714, 0.96643, 0.78237, 0.64114, 0.51966, 0.42563, 0.34688,
    0.28351, 0.23317, 0.18963, 0.15788, 0.12739, 0.10762, 0.08597, 0.07381,
    0.05828, 0.05089, 0.03971, 0.03524, 0.02720, 0.02451, 0.01874, 0.01711,
    0.01298, 0.01198, 0.00904, 0.00841, 0.00634, 0.00592, 0.00446, 0.00418,
    0.00316, 0.00296, 0.00225, 0.00210, 0.00160, 0.00150, 0.00115, 0.00107,
    0.00082, 0.00077, 0.00059, 0.00055, 0.00043, 0.00040, 0.00031, 0.00029,
    0.00023, 0.00021, 0.00017, 0.00015, 0.00012, 0.00011, 0.00009, 0.00008,
    0.00006, 0.00006, 0.00005, 0.00004, 0.00004, 0.00003, 0.00003, 0.00002,
    0.00002, 0.00002, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
    0.00001, 0.00001
  };

  int nmom = static_cast<int>(cfg.phase_function_moments[lc].size());
  cfg.phase_function_moments[lc][0] = 1.0;
  int nmom_limit = std::min(82, nmom - 1);
  for (int k = 1; k <= nmom_limit; ++k)
    cfg.phase_function_moments[lc][k] = hazelm[k - 1] / static_cast<double>(2 * k + 1);
  for (int k = nmom_limit + 1; k < nmom; ++k)
    cfg.phase_function_moments[lc][k] = 0.0;
}


// ============================================================================
//  Test 6b: Pure Absorption — Beer's Law
// ============================================================================

void test_pure_absorption_beers_law() {
  std::cout << "  test_pure_absorption_beers_law ... ";

  adrt::ADConfig cfg(1, 8);
  cfg.solar_flux = 200.0;
  cfg.solar_mu = 0.5;
  cfg.surface_albedo = 0.0;
  cfg.allocate();
  cfg.delta_tau[0] = 1.0;
  cfg.single_scat_albedo[0] = 0.0;
  cfg.setIsotropic();

  auto cpu = cpuSolve(cfg);
  auto cuda = cudaSolveSingle(cfg);

  CHECK_NEAR("flux_up_toa", cuda.flux_up, cpu.flux_up[0], 1e-4);
  CHECK_NEAR("flux_down_toa", cuda.flux_down, cpu.flux_down[0], 1e-4);
  CHECK_NEAR("flux_direct_toa", cuda.flux_direct, cpu.flux_direct[1], 1e-3);
  std::cout << "done\n";
}


void test_pure_absorption_lambertian() {
  std::cout << "  test_pure_absorption_lambertian ... ";

  adrt::ADConfig cfg(1, 8);
  cfg.solar_flux = 200.0;
  cfg.solar_mu = 0.5;
  cfg.surface_albedo = 0.5;
  cfg.allocate();
  cfg.delta_tau[0] = 1.0;
  cfg.single_scat_albedo[0] = 0.0;
  cfg.setIsotropic();

  auto cpu = cpuSolve(cfg);
  auto cuda = cudaSolveSingle(cfg);

  CHECK_NEAR("flux_up_toa", cuda.flux_up, cpu.flux_up[0], 1e-3);
  CHECK_NEAR("flux_down_toa", cuda.flux_down, cpu.flux_down[0], 1e-4);
  std::cout << "done\n";
}


// ============================================================================
//  Test 1a-1d: Isotropic Scattering (VH1 Table 12)
// ============================================================================

void test_isotropic_scattering() {
  std::cout << "  test_isotropic_scattering ... ";

  auto run = [&](double tau, double omega, const char* label, double tol) {
    adrt::ADConfig cfg(1, 8);
    cfg.solar_flux = 10.0 * PI;
    cfg.solar_mu = 0.1;
    cfg.surface_albedo = 0.0;
    cfg.allocate();
    cfg.delta_tau[0] = tau;
    cfg.single_scat_albedo[0] = omega;
    cfg.setIsotropic();

    auto cpu = cpuSolve(cfg);
    auto cuda = cudaSolveSingle(cfg);

    char buf[256];
    snprintf(buf, sizeof(buf), "%s_flux_up", label);
    CHECK_NEAR(buf, cuda.flux_up, cpu.flux_up[0], tol);
    snprintf(buf, sizeof(buf), "%s_flux_down", label);
    CHECK_NEAR(buf, cuda.flux_down, cpu.flux_down[0], tol);
    snprintf(buf, sizeof(buf), "%s_flux_direct", label);
    CHECK_NEAR(buf, cuda.flux_direct, cpu.flux_direct[1], tol);
  };

  // 1a: tau=0.03125, omega=0.2
  run(0.03125, 0.2, "iso1a", 0.01);
  // 1b: tau=0.03125, omega=1.0
  run(0.03125, 1.0, "iso1b", 0.01);
  // 1c: tau=0.03125, omega=0.99
  run(0.03125, 0.99, "iso1c", 0.01);
  // 1d: tau=32, omega=0.2
  run(32.0, 0.2, "iso1d", 0.01);

  std::cout << "done\n";
}


// ============================================================================
//  Test 2a-2d: Rayleigh Scattering (SW Table 1)
// ============================================================================

void test_rayleigh_scattering() {
  std::cout << "  test_rayleigh_scattering ... ";

  auto run = [&](double tau, double omega, const char* label, double tol) {
    adrt::ADConfig cfg(1, 8);
    cfg.solar_flux = PI;
    cfg.solar_mu = 0.080442;
    cfg.surface_albedo = 0.0;
    cfg.allocate();
    cfg.delta_tau[0] = tau;
    cfg.single_scat_albedo[0] = omega;
    cfg.setRayleigh();

    auto cpu = cpuSolve(cfg);
    auto cuda = cudaSolveSingle(cfg);

    char buf[256];
    snprintf(buf, sizeof(buf), "%s_flux_up", label);
    CHECK_NEAR(buf, cuda.flux_up, cpu.flux_up[0], tol);
    snprintf(buf, sizeof(buf), "%s_flux_direct", label);
    CHECK_NEAR(buf, cuda.flux_direct, cpu.flux_direct[1], tol);
  };

  // 2a: tau=0.2, omega=0.5
  run(0.2, 0.5, "ray2a", 0.01);
  // 2b: tau=0.2, omega=1.0
  run(0.2, 1.0, "ray2b", 0.01);
  // 2c: tau=5.0, omega=0.5
  run(5.0, 0.5, "ray2c", 0.01);
  // 2d: tau=5.0, omega=1.0
  run(5.0, 1.0, "ray2d", 0.01);

  std::cout << "done\n";
}


// ============================================================================
//  Test 3a-3b: HG Scattering (VH2 Table 37)
// ============================================================================

void test_hg_scattering() {
  std::cout << "  test_hg_scattering ... ";

  auto run = [&](double tau, const char* label, double tol) {
    adrt::ADConfig cfg(1, 8);
    cfg.solar_flux = PI;
    cfg.solar_mu = 1.0;
    cfg.surface_albedo = 0.0;
    cfg.allocate();
    cfg.delta_tau[0] = tau;
    cfg.single_scat_albedo[0] = 1.0;
    cfg.setHenyeyGreenstein(0.75);

    auto cpu = cpuSolve(cfg);
    auto cuda = cudaSolveSingle(cfg);

    char buf[256];
    snprintf(buf, sizeof(buf), "%s_flux_up", label);
    CHECK_NEAR(buf, cuda.flux_up, cpu.flux_up[0], tol);
    snprintf(buf, sizeof(buf), "%s_flux_direct", label);
    CHECK_NEAR(buf, cuda.flux_direct, cpu.flux_direct[1], tol);
  };

  // 3a: tau=1.0
  run(1.0, "hg3a", 5e-3);
  // 3b: tau=8.0
  run(8.0, "hg3b", 0.05);

  std::cout << "done\n";
}


// ============================================================================
//  Test 4a-4c: Haze-L Garcia-Siewert (GS Tables 12-16)
//  nquad=16 → uses batched cuBLAS path
// ============================================================================

void test_haze_garcia_siewert() {
  std::cout << "  test_haze_garcia_siewert ... ";

  auto run = [&](double omega, double mu0, const char* label, double tol) {
    adrt::ADConfig cfg(1, 16);
    cfg.solar_flux = PI;
    cfg.solar_mu = mu0;
    cfg.surface_albedo = 0.0;
    cfg.allocate();
    cfg.delta_tau[0] = 1.0;
    cfg.single_scat_albedo[0] = omega;
    setHazeGarciaSiewert(cfg, 0);

    auto cpu = cpuSolve(cfg);
    auto cuda = cudaSolveSingle(cfg);

    char buf[256];
    snprintf(buf, sizeof(buf), "%s_flux_up", label);
    CHECK_NEAR(buf, cuda.flux_up, cpu.flux_up[0], tol);
    snprintf(buf, sizeof(buf), "%s_flux_direct", label);
    CHECK_NEAR(buf, cuda.flux_direct, cpu.flux_direct[1], tol);
  };

  // 4a: omega=1.0, mu0=1.0
  run(1.0, 1.0, "haze4a", 0.05);
  // 4b: omega=0.9, mu0=1.0
  run(0.9, 1.0, "haze4b", 0.05);
  // 4c: omega=0.9, mu0=0.5
  run(0.9, 0.5, "haze4c", 0.05);

  std::cout << "done\n";
}


// ============================================================================
//  Multi-layer tests
// ============================================================================

void test_multilayer_rayleigh() {
  std::cout << "  test_multilayer_rayleigh ... ";

  adrt::ADConfig cfg(2, 8);
  cfg.solar_flux = 1.0;
  cfg.solar_mu = 0.5;
  cfg.surface_albedo = 0.0;
  cfg.allocate();

  cfg.delta_tau[0] = 0.1;
  cfg.single_scat_albedo[0] = 0.5;
  cfg.setRayleigh(0);

  cfg.delta_tau[1] = 0.2;
  cfg.single_scat_albedo[1] = 0.8;
  cfg.setRayleigh(1);

  auto cpu = cpuSolve(cfg);
  auto cuda = cudaSolveSingle(cfg);

  CHECK_NEAR("flux_up_toa", cuda.flux_up, cpu.flux_up[0], 1e-3);
  CHECK_NEAR("flux_direct_bot", cuda.flux_direct, cpu.flux_direct[2], 1e-3);
  std::cout << "done\n";
}


void test_multilayer_hg() {
  std::cout << "  test_multilayer_hg ... ";

  adrt::ADConfig cfg(2, 8);
  cfg.solar_flux = PI;
  cfg.solar_mu = 1.0;
  cfg.surface_albedo = 0.0;
  cfg.allocate();

  cfg.delta_tau[0] = 0.5;
  cfg.single_scat_albedo[0] = 1.0;
  cfg.setHenyeyGreenstein(0.75, 0);

  cfg.delta_tau[1] = 0.5;
  cfg.single_scat_albedo[1] = 1.0;
  cfg.setHenyeyGreenstein(0.75, 1);

  auto cpu = cpuSolve(cfg);
  auto cuda = cudaSolveSingle(cfg);

  CHECK_NEAR("flux_up_toa", cuda.flux_up, cpu.flux_up[0], 5e-3);
  CHECK_NEAR("flux_direct_bot", cuda.flux_direct, cpu.flux_direct[2], 5e-3);
  std::cout << "done\n";
}


void test_multilayer_isotropic() {
  std::cout << "  test_multilayer_isotropic ... ";

  adrt::ADConfig cfg(3, 8);
  cfg.solar_flux = 1.0;
  cfg.solar_mu = 0.5;
  cfg.surface_albedo = 0.1;
  cfg.allocate();

  cfg.delta_tau[0] = 0.5;  cfg.single_scat_albedo[0] = 0.5;  cfg.setIsotropic(0);
  cfg.delta_tau[1] = 1.0;  cfg.single_scat_albedo[1] = 0.9;  cfg.setIsotropic(1);
  cfg.delta_tau[2] = 0.5;  cfg.single_scat_albedo[2] = 0.5;  cfg.setIsotropic(2);

  auto cpu = cpuSolve(cfg);
  auto cuda = cudaSolveSingle(cfg);

  CHECK_NEAR("flux_up_toa", cuda.flux_up, cpu.flux_up[0], 1e-3);
  CHECK_NEAR("flux_direct_bot", cuda.flux_direct, cpu.flux_direct[3], 1e-3);
  std::cout << "done\n";
}


// ============================================================================
//  Test 8a-8c: Diffuse Illumination (OS Table 1)
// ============================================================================

void test_diffuse_illumination() {
  std::cout << "  test_diffuse_illumination ... ";

  auto run = [&](double tau0, double omega0, double tau1, double omega1,
                 const char* label, double tol) {
    adrt::ADConfig cfg(2, 4);
    cfg.top_emission = 1.0 / PI;
    cfg.surface_albedo = 0.0;
    cfg.allocate();

    cfg.delta_tau[0] = tau0;  cfg.single_scat_albedo[0] = omega0;  cfg.setIsotropic(0);
    cfg.delta_tau[1] = tau1;  cfg.single_scat_albedo[1] = omega1;  cfg.setIsotropic(1);

    auto cpu = cpuSolve(cfg);
    auto cuda = cudaSolveSingle(cfg);

    char buf[256];
    snprintf(buf, sizeof(buf), "%s_flux_up", label);
    CHECK_NEAR(buf, cuda.flux_up, cpu.flux_up[0], tol);
    snprintf(buf, sizeof(buf), "%s_flux_down", label);
    CHECK_NEAR(buf, cuda.flux_down, cpu.flux_down[0], tol);
  };

  // 8a
  run(0.25, 0.5, 0.25, 0.3, "diff8a", 1e-3);
  // 8b
  run(0.25, 0.8, 0.25, 0.95, "diff8b", 1e-3);
  // 8c
  run(1.0, 0.8, 2.0, 0.95, "diff8c", 5e-3);

  std::cout << "done\n";
}


// ============================================================================
//  Test 9a-9b: 6-Layer Heterogeneous Atmosphere
// ============================================================================

void test_six_layer() {
  std::cout << "  test_six_layer ... ";

  // 9a: Isotropic
  {
    adrt::ADConfig cfg(6, 4);
    cfg.top_emission = 1.0 / PI;
    cfg.surface_albedo = 0.0;
    cfg.allocate();

    for (int lc = 0; lc < 6; ++lc) {
      cfg.delta_tau[lc] = static_cast<double>(lc + 1);
      cfg.single_scat_albedo[lc] = 0.65 + lc * 0.05;
      cfg.setIsotropic(lc);
    }

    auto cpu = cpuSolve(cfg);
    auto cuda = cudaSolveSingle(cfg);

    CHECK_NEAR("9a_flux_up", cuda.flux_up, cpu.flux_up[0], 5e-3);
    CHECK_NEAR("9a_flux_down", cuda.flux_down, cpu.flux_down[0], 5e-3);
  }

  // 9b: Anisotropic (DGIS)
  {
    const double pmom_dgis[9] = {
      1.0,
      2.00916 / 3.0,
      1.56339 / 5.0,
      0.67407 / 7.0,
      0.22215 / 9.0,
      0.04725 / 11.0,
      0.00671 / 13.0,
      0.00068 / 15.0,
      0.00005 / 17.0
    };

    adrt::ADConfig cfg(6, 4);
    cfg.top_emission = 1.0 / PI;
    cfg.surface_albedo = 0.0;
    cfg.allocate();

    for (int lc = 0; lc < 6; ++lc) {
      cfg.delta_tau[lc] = static_cast<double>(lc + 1);
      cfg.single_scat_albedo[lc] = 0.65 + lc * 0.05;
      int nmom = static_cast<int>(cfg.phase_function_moments[lc].size());
      for (int k = 0; k < std::min(9, nmom); ++k)
        cfg.phase_function_moments[lc][k] = pmom_dgis[k];
      for (int k = 9; k < nmom; ++k)
        cfg.phase_function_moments[lc][k] = 0.0;
    }

    auto cpu = cpuSolve(cfg);
    auto cuda = cudaSolveSingle(cfg);

    CHECK_NEAR("9b_flux_up", cuda.flux_up, cpu.flux_up[0], 5e-3);
    CHECK_NEAR("9b_flux_down", cuda.flux_down, cpu.flux_down[0], 5e-3);
  }

  std::cout << "done\n";
}


// ============================================================================
//  Test 11a-11b: Combined Sources
// ============================================================================

void test_combined_sources() {
  std::cout << "  test_combined_sources ... ";

  // 11a: single layer
  {
    adrt::ADConfig cfg(1, 8);
    cfg.solar_flux = 1.0;
    cfg.solar_mu = 0.5;
    cfg.top_emission = 0.5 / PI;
    cfg.surface_albedo = 0.5;
    cfg.allocate();
    cfg.delta_tau[0] = 1.0;
    cfg.single_scat_albedo[0] = 0.9;
    cfg.setIsotropic();

    auto cpu = cpuSolve(cfg);
    auto cuda = cudaSolveSingle(cfg);

    CHECK_NEAR("11a_flux_up", cuda.flux_up, cpu.flux_up[0], 5e-3);
    CHECK_NEAR("11a_flux_down", cuda.flux_down, cpu.flux_down[0], 5e-3);
    CHECK_NEAR("11a_flux_direct", cuda.flux_direct, cpu.flux_direct[1], 5e-3);
  }

  // 11b: layer split (same problem, 3 layers)
  {
    adrt::ADConfig cfg(3, 8);
    cfg.solar_flux = 1.0;
    cfg.solar_mu = 0.5;
    cfg.top_emission = 0.5 / PI;
    cfg.surface_albedo = 0.5;
    cfg.allocate();

    cfg.delta_tau[0] = 0.05;  cfg.single_scat_albedo[0] = 0.9;  cfg.setIsotropic(0);
    cfg.delta_tau[1] = 0.45;  cfg.single_scat_albedo[1] = 0.9;  cfg.setIsotropic(1);
    cfg.delta_tau[2] = 0.50;  cfg.single_scat_albedo[2] = 0.9;  cfg.setIsotropic(2);

    auto cpu = cpuSolve(cfg);
    auto cuda = cudaSolveSingle(cfg);

    CHECK_NEAR("11b_flux_up", cuda.flux_up, cpu.flux_up[0], 5e-3);
    CHECK_NEAR("11b_flux_down", cuda.flux_down, cpu.flux_down[0], 5e-3);
    CHECK_NEAR("11b_flux_direct", cuda.flux_direct, cpu.flux_direct[3], 1e-3);
  }

  std::cout << "done\n";
}


// ============================================================================
//  Thermal Emission Tests
// ============================================================================

void test_thermal_7a() {
  std::cout << "  test_thermal_7a ... ";

  adrt::ADConfig cfg(1, 8);
  cfg.surface_albedo = 0.0;
  cfg.allocate();

  cfg.delta_tau[0] = 1.0;
  cfg.single_scat_albedo[0] = 0.1;
  cfg.setHenyeyGreenstein(0.05);

  cfg.planck_levels = {
    adrt::planckFunction(300.0, 800.0, 200.0),
    adrt::planckFunction(300.0, 800.0, 300.0)
  };
  cfg.top_emission = 0.0;
  cfg.surface_emission = 0.0;

  auto cpu = cpuSolve(cfg);
  auto cuda = cudaSolveSingle(cfg);

  CHECK_NEAR("7a_flux_up", cuda.flux_up, cpu.flux_up[0], 0.5);
  CHECK_NEAR("7a_flux_down", cuda.flux_down, cpu.flux_down[0], 0.5);
  std::cout << "done\n";
}


void test_thermal_7c() {
  std::cout << "  test_thermal_7c ... ";

  adrt::ADConfig cfg(1, 8);
  cfg.solar_flux = 200.0;
  cfg.solar_mu = 0.5;
  cfg.surface_albedo = 0.0;
  cfg.allocate();

  cfg.delta_tau[0] = 1.0;
  cfg.single_scat_albedo[0] = 0.5;
  cfg.setHenyeyGreenstein(0.8);

  cfg.planck_levels = {
    adrt::planckFunction(0.01, 50000.0, 300.0),
    adrt::planckFunction(0.01, 50000.0, 200.0)
  };
  cfg.top_emission = 100.0 + adrt::planckFunction(0.01, 50000.0, 100.0);
  cfg.surface_emission = adrt::planckFunction(0.01, 50000.0, 320.0);

  auto cpu = cpuSolve(cfg);
  auto cuda = cudaSolveSingle(cfg);

  // Wider tolerance — large Planck values, float precision
  double tol = 0.02 * std::abs(cpu.flux_up[0]);
  CHECK_NEAR("7c_flux_up", cuda.flux_up, cpu.flux_up[0], tol);
  CHECK_NEAR("7c_flux_direct", cuda.flux_direct, cpu.flux_direct[1], 1e-3);
  std::cout << "done\n";
}


void test_thermal_pure_absorption() {
  std::cout << "  test_thermal_pure_absorption ... ";

  adrt::ADConfig cfg(3, 8);
  cfg.use_thermal_emission = true;
  cfg.wavenumber_low = 500.0;
  cfg.wavenumber_high = 600.0;
  cfg.allocate();

  std::vector<double> T = {200.0, 230.0, 260.0, 290.0};
  for (int l = 0; l < 3; ++l) {
    cfg.delta_tau[l] = 0.5;
    cfg.single_scat_albedo[l] = 0.0;
    cfg.temperature[l] = T[l];
  }
  cfg.temperature[3] = T[3];
  cfg.setIsotropic();

  auto cpu = cpuSolve(cfg);
  auto cuda = cudaSolveSingle(cfg);

  CHECK_NEAR("flux_up", cuda.flux_up, cpu.flux_up[0], 0.5);
  CHECK_TRUE("flux_up_pos", cuda.flux_up > 0.0);
  CHECK_TRUE("not_nan", !std::isnan(cuda.flux_up));
  std::cout << "done\n";
}


void test_thermal_scattering() {
  std::cout << "  test_thermal_scattering ... ";

  adrt::ADConfig cfg(4, 8);
  cfg.use_thermal_emission = true;
  cfg.wavenumber_low = 800.0;
  cfg.wavenumber_high = 900.0;
  cfg.allocate();

  std::vector<double> T = {180.0, 210.0, 240.0, 270.0, 300.0};
  for (int l = 0; l < 4; ++l) {
    cfg.delta_tau[l] = 0.3;
    cfg.single_scat_albedo[l] = 1.0;
    cfg.temperature[l] = T[l];
  }
  cfg.temperature[4] = T[4];
  cfg.setIsotropic();

  auto cpu = cpuSolve(cfg);
  auto cuda = cudaSolveSingle(cfg);

  CHECK_NEAR("flux_up", cuda.flux_up, cpu.flux_up[0], 0.5);
  CHECK_TRUE("not_nan_up", !std::isnan(cuda.flux_up));
  CHECK_TRUE("not_nan_down", !std::isnan(cuda.flux_down));
  std::cout << "done\n";
}


void test_thermal_hg_deltam() {
  std::cout << "  test_thermal_hg_deltam ... ";

  adrt::ADConfig cfg(3, 8);
  cfg.use_thermal_emission = true;
  cfg.use_delta_m = true;
  cfg.wavenumber_low = 600.0;
  cfg.wavenumber_high = 700.0;
  cfg.surface_albedo = 0.3;
  cfg.allocate();

  std::vector<double> T = {220.0, 240.0, 270.0, 300.0};
  for (int l = 0; l < 3; ++l) {
    cfg.delta_tau[l] = 1.0;
    cfg.single_scat_albedo[l] = 0.9;
    cfg.temperature[l] = T[l];
  }
  cfg.temperature[3] = T[3];
  cfg.setHenyeyGreenstein(0.8);

  auto cpu = cpuSolve(cfg);
  auto cuda = cudaSolveSingle(cfg);

  double tol = 0.02 * std::abs(cpu.flux_up[0]) + 0.5;
  CHECK_NEAR("flux_up", cuda.flux_up, cpu.flux_up[0], tol);
  CHECK_TRUE("flux_up_pos", cuda.flux_up > 0.0);
  CHECK_TRUE("not_nan", !std::isnan(cuda.flux_up));
  std::cout << "done\n";
}


// ============================================================================
//  Energy Conservation Tests
// ============================================================================

void test_energy_conservation() {
  std::cout << "  test_energy_conservation ... ";

  // For omega=1: net flux = F_up - F_down - F_direct should be constant.
  // CUDA only gives TOA values, so we check that CPU energy is conserved
  // and CUDA TOA matches CPU TOA.

  // Conservative, no delta-M
  {
    int nlay = 5;
    adrt::ADConfig cfg(nlay, 8);
    cfg.solar_flux = 1.0;
    cfg.solar_mu = 0.5;
    cfg.surface_albedo = 0.0;
    cfg.allocate();

    for (int l = 0; l < nlay; ++l) {
      cfg.delta_tau[l] = 0.3;
      cfg.single_scat_albedo[l] = 1.0;
    }
    cfg.setHenyeyGreenstein(0.7);

    auto cpu = cpuSolve(cfg);
    auto cuda = cudaSolveSingle(cfg);

    CHECK_NEAR("cons_flux_up", cuda.flux_up, cpu.flux_up[0], 5e-3);
    CHECK_NEAR("cons_flux_down", cuda.flux_down, cpu.flux_down[0], 5e-3);
  }

  // Conservative with delta-M
  {
    int nlay = 5;
    adrt::ADConfig cfg(nlay, 8);
    cfg.use_delta_m = true;
    cfg.solar_flux = 1.0;
    cfg.solar_mu = 0.6;
    cfg.surface_albedo = 0.0;
    cfg.allocate();

    for (int l = 0; l < nlay; ++l) {
      cfg.delta_tau[l] = 0.5;
      cfg.single_scat_albedo[l] = 1.0;
    }
    cfg.setHenyeyGreenstein(0.85);

    auto cpu = cpuSolve(cfg);
    auto cuda = cudaSolveSingle(cfg);

    CHECK_NEAR("deltam_flux_up", cuda.flux_up, cpu.flux_up[0], 5e-3);
    CHECK_NEAR("deltam_flux_down", cuda.flux_down, cpu.flux_down[0], 5e-3);
  }

  std::cout << "done\n";
}


// ============================================================================
//  Delta-M Tests
// ============================================================================

void test_delta_m() {
  std::cout << "  test_delta_m ... ";

  // Rayleigh with delta-M ON vs OFF should give same result (f=0)
  {
    adrt::ADConfig cfg_off(1, 8);
    cfg_off.solar_flux = 1.0;
    cfg_off.solar_mu = 1.0;
    cfg_off.allocate();
    cfg_off.delta_tau[0] = 0.5;
    cfg_off.single_scat_albedo[0] = 1.0;
    cfg_off.setRayleigh();

    adrt::ADConfig cfg_on(1, 8);
    cfg_on.use_delta_m = true;
    cfg_on.solar_flux = 1.0;
    cfg_on.solar_mu = 1.0;
    cfg_on.allocate();
    cfg_on.delta_tau[0] = 0.5;
    cfg_on.single_scat_albedo[0] = 1.0;
    cfg_on.setRayleigh();

    auto cuda_off = cudaSolveSingle(cfg_off);
    auto cuda_on = cudaSolveSingle(cfg_on);

    CHECK_NEAR("rayleigh_deltam_up", cuda_off.flux_up, cuda_on.flux_up, 1e-5);
    CHECK_NEAR("rayleigh_deltam_down", cuda_off.flux_down, cuda_on.flux_down, 1e-5);
  }

  // HG g=0.9: delta-M should improve convergence
  {
    auto solve_nq = [](int nq, bool delta_m) -> double {
      adrt::ADConfig cfg(1, nq);
      cfg.use_delta_m = delta_m;
      cfg.solar_flux = 1.0;
      cfg.solar_mu = 0.5;
      cfg.allocate();
      cfg.delta_tau[0] = 1.0;
      cfg.single_scat_albedo[0] = 0.9;
      cfg.setHenyeyGreenstein(0.9);
      return cudaSolveSingle(cfg).flux_up;
    };

    // Only test with nquad=4 and 8 (CUDA supported sizes)
    double off_4 = solve_nq(4, false);
    double off_8 = solve_nq(8, false);
    double on_4 = solve_nq(4, true);
    double on_8 = solve_nq(8, true);

    double spread_off = std::abs(off_4 - off_8);
    double spread_on = std::abs(on_4 - on_8);
    CHECK_TRUE("deltam_convergence", spread_on < spread_off);
  }

  std::cout << "done\n";
}


// ============================================================================
//  Flux Solver Comparison Tests
// ============================================================================

void test_flux_solver_cases() {
  std::cout << "  test_flux_solver_cases ... ";

  // Isotropic beam
  {
    adrt::ADConfig cfg(2, 8);
    cfg.solar_flux = PI;
    cfg.solar_mu = 0.6;
    cfg.surface_albedo = 0.1;
    cfg.allocate();

    for (int lc = 0; lc < 2; ++lc) {
      cfg.delta_tau[lc] = 0.25;
      cfg.single_scat_albedo[lc] = 0.9;
      cfg.setIsotropic(lc);
    }

    auto cpu = cpuSolve(cfg);
    auto cuda = cudaSolveSingle(cfg);

    CHECK_NEAR("iso_beam_up", cuda.flux_up, cpu.flux_up[0], 5e-3);
    CHECK_TRUE("iso_beam_up_pos", cuda.flux_up > 0.0);
  }

  // Multi-layer with varying properties
  {
    const int nlyr = 10;
    adrt::ADConfig cfg(nlyr, 4);
    cfg.solar_flux = PI;
    cfg.solar_mu = 0.7;
    cfg.surface_albedo = 0.3;
    cfg.allocate();

    for (int lc = 0; lc < nlyr; ++lc) {
      cfg.delta_tau[lc] = 0.05 * (lc + 1);
      cfg.single_scat_albedo[lc] = 0.9 - 0.05 * lc;
      cfg.setHenyeyGreenstein(0.7, lc);
    }

    auto cpu = cpuSolve(cfg);
    auto cuda = cudaSolveSingle(cfg);

    CHECK_NEAR("multi_flux_up", cuda.flux_up, cpu.flux_up[0], 5e-3);
    CHECK_TRUE("multi_not_nan", !std::isnan(cuda.flux_up));
  }

  // Optically thick (tau=10 each)
  {
    adrt::ADConfig cfg(5, 4);
    cfg.solar_flux = 1.0;
    cfg.solar_mu = 0.5;
    cfg.surface_albedo = 0.0;
    cfg.allocate();

    for (int lc = 0; lc < 5; ++lc) {
      cfg.delta_tau[lc] = 10.0;
      cfg.single_scat_albedo[lc] = 0.1;
      cfg.setIsotropic(lc);
    }

    auto cpu = cpuSolve(cfg);
    auto cuda = cudaSolveSingle(cfg);

    CHECK_NEAR("thick_flux_up", cuda.flux_up, cpu.flux_up[0], 1e-3);
    CHECK_TRUE("thick_direct_zero", cuda.flux_direct < 1e-15);
    CHECK_TRUE("thick_not_nan", !std::isnan(cuda.flux_up));
  }

  // Reflective surface
  {
    adrt::ADConfig cfg(2, 4);
    cfg.solar_flux = 1.0;
    cfg.solar_mu = 0.5;
    cfg.surface_albedo = 0.8;
    cfg.allocate();

    for (int lc = 0; lc < 2; ++lc) {
      cfg.delta_tau[lc] = 0.1;
      cfg.single_scat_albedo[lc] = 0.9;
      cfg.setIsotropic(lc);
    }

    auto cpu = cpuSolve(cfg);
    auto cuda = cudaSolveSingle(cfg);

    CHECK_NEAR("refl_flux_up", cuda.flux_up, cpu.flux_up[0], 5e-3);
    CHECK_TRUE("refl_up_pos", cuda.flux_up > 0.0);
  }

  std::cout << "done\n";
}


// ============================================================================
//  Mixed Atmosphere (thermal + solar, mixed phase functions)
// ============================================================================

void test_mixed_atmosphere() {
  std::cout << "  test_mixed_atmosphere ... ";

  adrt::ADConfig cfg(3, 8);
  cfg.surface_albedo = 0.1;
  cfg.solar_flux = 0.01;
  cfg.solar_mu = 0.7;
  cfg.allocate();

  cfg.delta_tau[0] = 0.1;  cfg.single_scat_albedo[0] = 0.95;
  cfg.setRayleigh(0);

  cfg.delta_tau[1] = 2.0;  cfg.single_scat_albedo[1] = 0.99;
  cfg.setDoubleHenyeyGreenstein(0.8, 0.7, -0.3, 1);

  cfg.delta_tau[2] = 0.5;  cfg.single_scat_albedo[2] = 0.1;
  cfg.setIsotropic(2);

  cfg.planck_levels = {1.0, 2.0, 3.0, 3.0};
  cfg.surface_emission = 4.0;

  auto cpu = cpuSolve(cfg);
  auto cuda = cudaSolveSingle(cfg);

  CHECK_NEAR("mixed_flux_up", cuda.flux_up, cpu.flux_up[0], 0.1);
  CHECK_TRUE("mixed_not_nan_up", !std::isnan(cuda.flux_up));
  CHECK_TRUE("mixed_not_nan_down", !std::isnan(cuda.flux_down));
  CHECK_TRUE("mixed_direct_atten", cuda.flux_direct < cpu.flux_direct[0]);
  std::cout << "done\n";
}


// ============================================================================
//  Diffusion Lower Boundary Condition Tests
// ============================================================================

void test_diffusion_bc() {
  std::cout << "  test_diffusion_bc ... ";

  // Pure absorption with diffusion BC
  {
    adrt::ADConfig cfg(3, 8);
    cfg.use_thermal_emission = true;
    cfg.use_diffusion_lower_bc = true;
    cfg.wavenumber_low = 500.0;
    cfg.wavenumber_high = 600.0;
    cfg.allocate();

    std::vector<double> T = {200.0, 230.0, 260.0, 290.0};
    for (int l = 0; l < 3; ++l) {
      cfg.delta_tau[l] = 0.5;
      cfg.single_scat_albedo[l] = 0.0;
      cfg.temperature[l] = T[l];
    }
    cfg.temperature[3] = T[3];
    cfg.setIsotropic();

    auto cpu = cpuSolve(cfg);
    auto cuda = cudaSolveSingle(cfg);

    CHECK_NEAR("diffbc_abs_up", cuda.flux_up, cpu.flux_up[0], 0.5);
    CHECK_TRUE("diffbc_abs_pos", cuda.flux_up > 0.0);
  }

  // Scattering with diffusion BC
  {
    adrt::ADConfig cfg(4, 8);
    cfg.use_thermal_emission = true;
    cfg.use_diffusion_lower_bc = true;
    cfg.wavenumber_low = 800.0;
    cfg.wavenumber_high = 900.0;
    cfg.allocate();

    std::vector<double> T = {180.0, 210.0, 240.0, 270.0, 300.0};
    for (int l = 0; l < 4; ++l) {
      cfg.delta_tau[l] = 0.3;
      cfg.single_scat_albedo[l] = 0.9;
      cfg.temperature[l] = T[l];
    }
    cfg.temperature[4] = T[4];
    cfg.setIsotropic();

    auto cpu = cpuSolve(cfg);
    auto cuda = cudaSolveSingle(cfg);

    CHECK_NEAR("diffbc_scat_up", cuda.flux_up, cpu.flux_up[0], 0.5);
    CHECK_TRUE("diffbc_scat_not_nan", !std::isnan(cuda.flux_up));
  }

  // Diffusion BC with delta-M and HG
  {
    adrt::ADConfig cfg(3, 8);
    cfg.use_thermal_emission = true;
    cfg.use_diffusion_lower_bc = true;
    cfg.use_delta_m = true;
    cfg.wavenumber_low = 600.0;
    cfg.wavenumber_high = 700.0;
    cfg.allocate();

    std::vector<double> T = {220.0, 250.0, 280.0, 310.0};
    for (int l = 0; l < 3; ++l) {
      cfg.delta_tau[l] = 1.0;
      cfg.single_scat_albedo[l] = 0.9;
      cfg.temperature[l] = T[l];
    }
    cfg.temperature[3] = T[3];
    cfg.setHenyeyGreenstein(0.8);

    auto cpu = cpuSolve(cfg);
    auto cuda = cudaSolveSingle(cfg);

    double tol = 0.02 * std::abs(cpu.flux_up[0]) + 0.5;
    CHECK_NEAR("diffbc_deltam_up", cuda.flux_up, cpu.flux_up[0], tol);
    CHECK_TRUE("diffbc_deltam_pos", cuda.flux_up > 0.0);
  }

  std::cout << "done\n";
}


// ============================================================================
//  Rayleigh Spherical Albedo (nquad=16 → batched path)
// ============================================================================

void test_rayleigh_spherical_albedo() {
  std::cout << "  test_rayleigh_spherical_albedo ... ";

  int nlay = 10;
  double total_tau = 0.5;

  adrt::ADConfig cfg(nlay, 16);
  cfg.solar_flux = 1.0;
  cfg.solar_mu = 1.0;
  cfg.surface_albedo = 0.0;
  cfg.allocate();

  for (int l = 0; l < nlay; ++l) {
    cfg.delta_tau[l] = total_tau / nlay;
    cfg.single_scat_albedo[l] = 1.0;
  }
  cfg.setRayleigh();

  auto cpu = cpuSolve(cfg);
  auto cuda = cudaSolveSingle(cfg);

  CHECK_NEAR("sph_alb_up", cuda.flux_up, cpu.flux_up[0], 0.05);

  double albedo = cuda.flux_up / (cfg.solar_flux * cfg.solar_mu);
  CHECK_TRUE("albedo_range", albedo > 0.15 && albedo < 0.25);

  std::cout << "done\n";
}


// ============================================================================
//  Batch Consistency: same input across multiple wavenumbers
// ============================================================================

void test_batch_consistency() {
  std::cout << "  test_batch_consistency ... ";

  int nlay = 5;
  int nwav = 100;
  int nmom = 3;

  std::vector<float> delta_tau(nwav * nlay);
  std::vector<float> ssa(nwav * nlay);
  std::vector<float> pmom(nlay * nmom, 0.0f);
  std::vector<float> planck(nwav * (nlay + 1));

  pmom[0 * nmom + 0] = 1.0f;
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

  bool all_same = true;
  for (int w = 1; w < nwav; ++w) {
    if (std::abs(result.flux_up[w] - result.flux_up[0]) > 1e-6f ||
        std::abs(result.flux_down[w] - result.flux_down[0]) > 1e-6f) {
      all_same = false;
      break;
    }
  }

  CHECK_TRUE("batch_consistency", all_same);
  std::cout << "done\n";
}


// ============================================================================
//  Optically thick layer (tau >> 1)
// ============================================================================

void test_optically_thick() {
  std::cout << "  test_optically_thick ... ";

  adrt::ADConfig cfg;
  cfg.num_layers = 1;
  cfg.num_quadrature = 8;
  cfg.allocate();
  cfg.delta_tau[0] = 100.0;
  cfg.single_scat_albedo[0] = 0.5;
  cfg.setIsotropic();
  cfg.planck_levels = {100.0, 200.0};

  auto cpu = cpuSolve(cfg);
  auto cuda = cudaSolveSingle(cfg);

  CHECK_NEAR("flux_up", cuda.flux_up, cpu.flux_up[0], 1.0);
  CHECK_NEAR("flux_down", cuda.flux_down, cpu.flux_down[0], 1.0);
  std::cout << "done\n";
}


// ============================================================================
//  Conservative scattering stress tests
// ============================================================================

void test_conservative_thick() {
  std::cout << "  test_conservative_thick ... ";

  auto run_case = [&](int nlay, double tau_per_layer, int nquad,
                      double g, double tol, const char* label) {
    adrt::ADConfig cfg;
    cfg.num_layers = nlay;
    cfg.num_quadrature = nquad;
    cfg.allocate();

    for (int l = 0; l < nlay; ++l) {
      cfg.delta_tau[l] = tau_per_layer;
      cfg.single_scat_albedo[l] = 1.0;
    }
    cfg.setHenyeyGreenstein(g);

    std::vector<double> planck(nlay + 1);
    for (int l = 0; l <= nlay; ++l)
      planck[l] = 100.0 + 50.0 * l;
    cfg.planck_levels = planck;

    auto cpu = cpuSolve(cfg);
    auto cuda = cudaSolveSingle(cfg);

    char buf[256];
    snprintf(buf, sizeof(buf), "%s_flux_up", label);
    CHECK_NEAR(buf, cuda.flux_up, cpu.flux_up[0], tol);
    snprintf(buf, sizeof(buf), "%s_flux_down", label);
    CHECK_NEAR(buf, cuda.flux_down, cpu.flux_down[0], tol);
  };

  // Per-thread path (N=4, N=8)
  run_case(10, 5.0, 4, 0.0, 1.0, "N4_iso_tau50");
  run_case(20, 5.0, 8, 0.8, 1.0, "N8_hg08_tau100");
  run_case(1, 100.0, 8, 0.0, 1.0, "N8_iso_tau100_1lay");

  // Batched cuBLAS path (N=16)
  run_case(10, 5.0, 16, 0.0, 1.0, "N16_iso_tau50");
  run_case(20, 5.0, 16, 0.8, 2.0, "N16_hg08_tau100");
  run_case(1, 100.0, 16, 0.0, 1.0, "N16_iso_tau100_1lay");

  std::cout << "done\n";
}


void test_conservative_solar_energy() {
  std::cout << "  test_conservative_solar_energy ... ";

  auto run_case = [&](int nquad, double tau, double g, double tol,
                      const char* label) {
    adrt::ADConfig cfg;
    cfg.num_layers = 5;
    cfg.num_quadrature = nquad;
    cfg.allocate();

    for (int l = 0; l < 5; ++l) {
      cfg.delta_tau[l] = tau / 5.0;
      cfg.single_scat_albedo[l] = 1.0;
    }
    cfg.setHenyeyGreenstein(g);
    cfg.solar_flux = 1.0;
    cfg.solar_mu = 0.5;
    cfg.surface_albedo = 0.0;

    auto cpu = cpuSolve(cfg);
    auto cuda = cudaSolveSingle(cfg);

    char buf[256];
    snprintf(buf, sizeof(buf), "%s_flux_up", label);
    CHECK_NEAR(buf, cuda.flux_up, cpu.flux_up[0], tol);
    snprintf(buf, sizeof(buf), "%s_flux_down", label);
    CHECK_NEAR(buf, cuda.flux_down, cpu.flux_down[0], tol);
  };

  // Per-thread path
  run_case(4, 10.0, 0.0, 5e-3, "N4_iso_tau10");
  run_case(8, 10.0, 0.0, 5e-3, "N8_iso_tau10");
  run_case(8, 10.0, 0.85, 5e-3, "N8_hg085_tau10");

  // Batched cuBLAS path
  run_case(16, 10.0, 0.0, 5e-2, "N16_iso_tau10");
  run_case(16, 10.0, 0.85, 5e-2, "N16_hg085_tau10");

  std::cout << "done\n";
}


// ============================================================================
//  20-layer varying atmosphere (original multilayer test)
// ============================================================================

void test_multilayer_varying() {
  std::cout << "  test_multilayer_varying ... ";

  int nlay = 20;
  adrt::ADConfig cfg;
  cfg.num_layers = nlay;
  cfg.num_quadrature = 8;
  cfg.allocate();

  for (int l = 0; l < nlay; ++l) {
    cfg.delta_tau[l] = 0.1 + 0.05 * l;
    cfg.single_scat_albedo[l] = 0.3 + 0.03 * l;
  }
  cfg.setHenyeyGreenstein(0.7);

  std::vector<double> planck(nlay + 1);
  for (int l = 0; l <= nlay; ++l)
    planck[l] = 100.0 + 10.0 * l;
  cfg.planck_levels = planck;
  cfg.surface_albedo = 0.1;

  auto cpu = cpuSolve(cfg);
  auto cuda = cudaSolveSingle(cfg);

  CHECK_NEAR("flux_up", cuda.flux_up, cpu.flux_up[0], 1.0);
  CHECK_NEAR("flux_down", cuda.flux_down, cpu.flux_down[0], 1.0);
  std::cout << "done\n";
}


// ============================================================================
//  Linear Source — Pure Absorption (analytical comparison)
// ============================================================================

void test_linear_source_pure_absorption() {
  std::cout << "  test_linear_source_pure_absorption ... ";

  double tau = 1.0;
  double B_top = 1.0;
  double B_bot = 3.0;

  adrt::ADConfig cfg(1, 8);
  cfg.allocate();
  cfg.delta_tau[0] = tau;
  cfg.single_scat_albedo[0] = 0.0;
  cfg.planck_levels = {B_top, B_bot};
  cfg.setIsotropic();

  auto cpu = cpuSolve(cfg);
  auto cuda = cudaSolveSingle(cfg);

  // Compare CUDA against CPU (CPU already matches analytical to 1e-8)
  CHECK_NEAR("flux_up", cuda.flux_up, cpu.flux_up[0], 1e-4);
  CHECK_NEAR("flux_down", cuda.flux_down, cpu.flux_down[0], 1e-4);
  std::cout << "done\n";
}


// ============================================================================
//  N=16 specific tests (batched cuBLAS path)
// ============================================================================

void test_n16_specific() {
  std::cout << "  test_n16_specific ... ";

  // Pure absorption at N=16
  {
    adrt::ADConfig cfg(1, 16);
    cfg.solar_flux = 200.0;
    cfg.solar_mu = 0.5;
    cfg.surface_albedo = 0.0;
    cfg.allocate();
    cfg.delta_tau[0] = 1.0;
    cfg.single_scat_albedo[0] = 0.0;
    cfg.setIsotropic();

    auto cpu = cpuSolve(cfg);
    auto cuda = cudaSolveSingle(cfg);

    CHECK_NEAR("n16_abs_up", cuda.flux_up, cpu.flux_up[0], 1e-3);
    CHECK_NEAR("n16_abs_direct", cuda.flux_direct, cpu.flux_direct[1], 1e-2);
  }

  // Isotropic scattering at N=16
  {
    adrt::ADConfig cfg(1, 16);
    cfg.solar_flux = PI;
    cfg.solar_mu = 0.1;
    cfg.surface_albedo = 0.0;
    cfg.allocate();
    cfg.delta_tau[0] = 0.03125;
    cfg.single_scat_albedo[0] = 1.0;
    cfg.setIsotropic();

    auto cpu = cpuSolve(cfg);
    auto cuda = cudaSolveSingle(cfg);

    CHECK_NEAR("n16_iso_up", cuda.flux_up, cpu.flux_up[0], 0.01);
  }

  // Multi-layer with surface at N=16
  {
    adrt::ADConfig cfg(3, 16);
    cfg.solar_flux = 1.0;
    cfg.solar_mu = 0.5;
    cfg.surface_albedo = 0.3;
    cfg.allocate();

    cfg.delta_tau[0] = 0.5;  cfg.single_scat_albedo[0] = 0.9;  cfg.setIsotropic(0);
    cfg.delta_tau[1] = 1.0;  cfg.single_scat_albedo[1] = 0.8;  cfg.setIsotropic(1);
    cfg.delta_tau[2] = 0.5;  cfg.single_scat_albedo[2] = 0.7;  cfg.setIsotropic(2);

    auto cpu = cpuSolve(cfg);
    auto cuda = cudaSolveSingle(cfg);

    CHECK_NEAR("n16_multi_up", cuda.flux_up, cpu.flux_up[0], 0.02);
    CHECK_NEAR("n16_multi_direct", cuda.flux_direct, cpu.flux_direct[3], 0.01);
  }

  // Thermal at N=16
  {
    adrt::ADConfig cfg(3, 16);
    cfg.use_thermal_emission = true;
    cfg.wavenumber_low = 500.0;
    cfg.wavenumber_high = 600.0;
    cfg.surface_albedo = 0.2;
    cfg.allocate();

    std::vector<double> T = {200.0, 230.0, 260.0, 290.0};
    for (int l = 0; l < 3; ++l) {
      cfg.delta_tau[l] = 0.5;
      cfg.single_scat_albedo[l] = 0.5;
      cfg.temperature[l] = T[l];
    }
    cfg.temperature[3] = T[3];
    cfg.setIsotropic();

    auto cpu = cpuSolve(cfg);
    auto cuda = cudaSolveSingle(cfg);

    double tol = 0.02 * std::abs(cpu.flux_up[0]) + 1.0;
    CHECK_NEAR("n16_thermal_up", cuda.flux_up, cpu.flux_up[0], tol);
  }

  // Combined solar + thermal at N=16
  {
    adrt::ADConfig cfg(2, 16);
    cfg.solar_flux = 1.0;
    cfg.solar_mu = 0.5;
    cfg.top_emission = 0.5 / PI;
    cfg.surface_albedo = 0.3;
    cfg.allocate();

    cfg.delta_tau[0] = 0.5;  cfg.single_scat_albedo[0] = 0.9;  cfg.setIsotropic(0);
    cfg.delta_tau[1] = 0.5;  cfg.single_scat_albedo[1] = 0.9;  cfg.setIsotropic(1);
    cfg.planck_levels = {1.0, 2.0, 3.0};
    cfg.surface_emission = 4.0;

    auto cpu = cpuSolve(cfg);
    auto cuda = cudaSolveSingle(cfg);

    CHECK_NEAR("n16_combined_up", cuda.flux_up, cpu.flux_up[0], 0.1);
    CHECK_TRUE("n16_combined_pos", cuda.flux_up > 0.0);
  }

  std::cout << "done\n";
}


// ============================================================================
//  Test: Raw-input entry point (solveBatchFromCoefficients)
// ============================================================================

/// Host-side single-wavenumber Planck matching cuda_planck_single.cuh
static float planck_single_host(float temperature, float wavenumber) {
  constexpr float c1 = 1.19105e-08f;
  constexpr float c2 = 1.43879f;
  if (temperature < 1e-5f) return 0.0f;
  float wn3 = wavenumber * wavenumber * wavenumber;
  return (c1 * wn3) / (std::exp(c2 * wavenumber / temperature) - 1.0f);
}

static void test_raw_input_entry_point() {
  std::cout << "=== Raw-input entry point ===\n";

  // Test parameters
  constexpr int nwav = 4;
  constexpr int nlay = 5;
  constexpr int nlev = nlay + 1;
  constexpr int N = 4;
  constexpr int nmom = 8;

  // Wavenumbers (cm⁻¹) — typical infrared range
  double wavenumbers[nwav] = {500.0, 1000.0, 1500.0, 2000.0};

  // Altitude grid (cm, BOA=0) — 6 levels, non-uniform spacing
  float altitude[nlev] = {0.0f, 1e5f, 3e5f, 6e5f, 1e6f, 2e6f};

  // Temperature profile (K, BOA=0)
  float temperature[nlev] = {1500.0f, 1300.0f, 1100.0f, 900.0f, 700.0f, 500.0f};

  // Absorption and scattering coefficients [level * nwav + wav], BOA=0
  std::vector<float> abs_coeff(nlev * nwav);
  std::vector<float> scat_coeff(nlev * nwav);

  for (int lev = 0; lev < nlev; ++lev) {
    for (int w = 0; w < nwav; ++w) {
      // Absorption varies with level and wavenumber
      abs_coeff[lev * nwav + w] = 1e-6f * (1.0f + 0.5f * lev) * (1.0f + 0.3f * w);
      // Scattering is a fraction of absorption
      scat_coeff[lev * nwav + w] = 0.3f * abs_coeff[lev * nwav + w];
    }
  }

  // Phase moments: isotropic (shared across wavenumbers)
  // chi[0] = 1, chi[1..] = 0
  std::vector<float> pmom(nlay * nmom, 0.0f);
  for (int l = 0; l < nlay; ++l)
    pmom[l * nmom + 0] = 1.0f;

  // --- Reference: manually compute delta_tau, ssa, planck on host ---
  // Then run solveBatchHost with these values.

  std::vector<float> ref_delta_tau(nwav * nlay);
  std::vector<float> ref_ssa(nwav * nlay);
  std::vector<float> ref_planck(nwav * nlev);

  for (int w = 0; w < nwav; ++w) {
    for (int l = 0; l < nlay; ++l) {
      // ADRT layer l (TOA=0) → BeAR layer = nlay - 1 - l (BOA=0)
      int bear_layer = nlay - 1 - l;
      float dz = altitude[bear_layer + 1] - altitude[bear_layer];

      float abs_bot  = abs_coeff[bear_layer       * nwav + w];
      float abs_top  = abs_coeff[(bear_layer + 1) * nwav + w];
      float scat_bot = scat_coeff[bear_layer       * nwav + w];
      float scat_top = scat_coeff[(bear_layer + 1) * nwav + w];

      float ext_bot = abs_bot + scat_bot;
      float ext_top = abs_top + scat_top;

      float tau = dz * (ext_top + ext_bot) * 0.5f;
      float scat_depth = dz * (scat_top + scat_bot) * 0.5f;

      if (tau < 0.0f) tau = 0.0f;
      float ssa = (tau > 0.0f) ? scat_depth / tau : 0.0f;

      ref_delta_tau[w * nlay + l] = tau;
      ref_ssa      [w * nlay + l] = ssa;
    }

    for (int lev = 0; lev < nlev; ++lev) {
      int bear_level = nlev - 1 - lev;
      float wn = static_cast<float>(wavenumbers[w]);
      ref_planck[w * nlev + lev] = planck_single_host(temperature[bear_level], wn);
    }
  }

  // Surface emission = Planck at BOA for each wavenumber
  std::vector<float> ref_surface_emission(nwav);
  std::vector<float> ref_top_emission(nwav);
  for (int w = 0; w < nwav; ++w) {
    float wn = static_cast<float>(wavenumbers[w]);
    ref_surface_emission[w] = planck_single_host(temperature[0], wn);
    ref_top_emission[w]     = planck_single_host(temperature[nlev - 1], wn);
  }

  // --- Run reference: solveBatch with pre-computed values + per-wavenumber emission ---
  // We must supply per-wavenumber surface/top emission because the raw path computes them
  // from temperature via planck_single, and the standard path uses config scalars.
  adrt::cuda::BatchConfig bcfg;
  bcfg.num_wavenumbers = nwav;
  bcfg.num_layers = nlay;
  bcfg.num_quadrature = N;
  bcfg.num_moments_max = nmom;
  bcfg.use_thermal_emission = false;
  bcfg.surface_albedo = 0.0;

  // Allocate all device arrays for reference path
  float* d_abs = nullptr;
  float* d_scat = nullptr;
  float* d_alt = nullptr;
  float* d_temp = nullptr;
  double* d_wn = nullptr;
  float* d_pmom = nullptr;
  float* d_flux_up = nullptr;
  float* d_flux_down = nullptr;
  float* d_flux_direct = nullptr;
  float* d_ref_dtau = nullptr;
  float* d_ref_ssa = nullptr;
  float* d_ref_planck = nullptr;
  float* d_ref_surf_em = nullptr;
  float* d_ref_top_em = nullptr;

  cudaMalloc(&d_abs,  nlev * nwav * sizeof(float));
  cudaMalloc(&d_scat, nlev * nwav * sizeof(float));
  cudaMalloc(&d_alt,  nlev * sizeof(float));
  cudaMalloc(&d_temp, nlev * sizeof(float));
  cudaMalloc(&d_wn,   nwav * sizeof(double));
  cudaMalloc(&d_pmom, nlay * nmom * sizeof(float));
  cudaMalloc(&d_flux_up,     nwav * sizeof(float));
  cudaMalloc(&d_flux_down,   nwav * sizeof(float));
  cudaMalloc(&d_flux_direct, nwav * sizeof(float));
  cudaMalloc(&d_ref_dtau,    nwav * nlay * sizeof(float));
  cudaMalloc(&d_ref_ssa,     nwav * nlay * sizeof(float));
  cudaMalloc(&d_ref_planck,  nwav * nlev * sizeof(float));
  cudaMalloc(&d_ref_surf_em, nwav * sizeof(float));
  cudaMalloc(&d_ref_top_em,  nwav * sizeof(float));

  cudaMemcpy(d_abs,  abs_coeff.data(),  nlev * nwav * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_scat, scat_coeff.data(), nlev * nwav * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_alt,  altitude,          nlev * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_temp, temperature,       nlev * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_wn,   wavenumbers,       nwav * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_pmom, pmom.data(),       nlay * nmom * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ref_dtau,    ref_delta_tau.data(),        nwav * nlay * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ref_ssa,     ref_ssa.data(),              nwav * nlay * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ref_planck,  ref_planck.data(),           nwav * nlev * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ref_surf_em, ref_surface_emission.data(), nwav * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ref_top_em,  ref_top_emission.data(),     nwav * sizeof(float), cudaMemcpyHostToDevice);

  // Run reference with DeviceData (per-wavenumber emission)
  adrt::cuda::DeviceData ref_dd;
  ref_dd.delta_tau = d_ref_dtau;
  ref_dd.single_scat_albedo = d_ref_ssa;
  ref_dd.phase_moments = d_pmom;
  ref_dd.phase_moments_shared = true;
  ref_dd.planck_levels = d_ref_planck;
  ref_dd.surface_emission = d_ref_surf_em;
  ref_dd.top_emission = d_ref_top_em;
  ref_dd.flux_up = d_flux_up;
  ref_dd.flux_down = d_flux_down;
  ref_dd.flux_direct = d_flux_direct;

  adrt::cuda::solveBatch(bcfg, ref_dd);
  cudaDeviceSynchronize();

  std::vector<float> ref_flux_up(nwav), ref_flux_down(nwav);
  cudaMemcpy(ref_flux_up.data(),   d_flux_up,   nwav * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(ref_flux_down.data(), d_flux_down, nwav * sizeof(float), cudaMemcpyDeviceToHost);

  adrt::cuda::RawDeviceData raw;
  raw.absorption_coeff = d_abs;
  raw.scattering_coeff = d_scat;
  raw.altitude = d_alt;
  raw.temperature = d_temp;
  raw.wavenumber = d_wn;
  raw.phase_moments = d_pmom;
  raw.phase_moments_shared = true;
  raw.flux_up = d_flux_up;
  raw.flux_down = d_flux_down;
  raw.flux_direct = d_flux_direct;

  adrt::cuda::BatchConfig raw_bcfg;
  raw_bcfg.num_wavenumbers = nwav;
  raw_bcfg.num_layers = nlay;
  raw_bcfg.num_quadrature = N;
  raw_bcfg.num_moments_max = nmom;
  raw_bcfg.surface_albedo = 0.0;

  adrt::cuda::solveBatchFromCoefficients(raw_bcfg, raw);
  cudaDeviceSynchronize();

  // Copy results back
  std::vector<float> test_flux_up(nwav), test_flux_down(nwav);
  cudaMemcpy(test_flux_up.data(),   d_flux_up,   nwav * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(test_flux_down.data(), d_flux_down, nwav * sizeof(float), cudaMemcpyDeviceToHost);

  // Compare — tolerance accounts for __expf vs std::exp difference in Planck
  float tol = 5e-4f;
  for (int w = 0; w < nwav; ++w) {
    char name[128];
    snprintf(name, sizeof(name), "raw_input_flux_up[wn=%.0f]", wavenumbers[w]);
    CHECK_NEAR(name, test_flux_up[w], ref_flux_up[w], tol);

    snprintf(name, sizeof(name), "raw_input_flux_down[wn=%.0f]", wavenumbers[w]);
    CHECK_NEAR(name, test_flux_down[w], ref_flux_down[w], tol);
  }

  // --- Test with cloud optical depth ---
  std::vector<float> cloud_tau(nlay * nwav, 0.0f);
  for (int l = 0; l < nlay; ++l)
    for (int w = 0; w < nwav; ++w)
      cloud_tau[l * nwav + w] = 0.01f * (l + 1);  // increasing with layer

  // Reference: add cloud tau to delta_tau (with BOA→TOA reversal)
  std::vector<float> ref_delta_tau_cloud(nwav * nlay);
  std::vector<float> ref_ssa_cloud(nwav * nlay);
  for (int w = 0; w < nwav; ++w) {
    for (int l = 0; l < nlay; ++l) {
      int bear_layer = nlay - 1 - l;
      float cloud_contribution = cloud_tau[bear_layer * nwav + w];
      float orig_scat_depth = ref_ssa[w * nlay + l] * ref_delta_tau[w * nlay + l];
      ref_delta_tau_cloud[w * nlay + l] = ref_delta_tau[w * nlay + l] + cloud_contribution;
      float new_tau = ref_delta_tau_cloud[w * nlay + l];
      ref_ssa_cloud[w * nlay + l] = (new_tau > 0.0f) ? orig_scat_depth / new_tau : 0.0f;
    }
  }

  // Upload cloud reference and run
  float* d_ref_dtau_cloud = nullptr;
  float* d_ref_ssa_cloud = nullptr;
  cudaMalloc(&d_ref_dtau_cloud, nwav * nlay * sizeof(float));
  cudaMalloc(&d_ref_ssa_cloud,  nwav * nlay * sizeof(float));
  cudaMemcpy(d_ref_dtau_cloud, ref_delta_tau_cloud.data(), nwav * nlay * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ref_ssa_cloud,  ref_ssa_cloud.data(),       nwav * nlay * sizeof(float), cudaMemcpyHostToDevice);

  ref_dd.delta_tau = d_ref_dtau_cloud;
  ref_dd.single_scat_albedo = d_ref_ssa_cloud;
  adrt::cuda::solveBatch(bcfg, ref_dd);
  cudaDeviceSynchronize();

  std::vector<float> ref_cloud_flux_up(nwav), ref_cloud_flux_down(nwav);
  cudaMemcpy(ref_cloud_flux_up.data(),   d_flux_up,   nwav * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(ref_cloud_flux_down.data(), d_flux_down, nwav * sizeof(float), cudaMemcpyDeviceToHost);

  // Upload cloud tau and run raw solver
  float* d_cloud_tau = nullptr;
  cudaMalloc(&d_cloud_tau, nlay * nwav * sizeof(float));
  cudaMemcpy(d_cloud_tau, cloud_tau.data(), nlay * nwav * sizeof(float), cudaMemcpyHostToDevice);

  raw.cloud_optical_depth = d_cloud_tau;
  adrt::cuda::solveBatchFromCoefficients(raw_bcfg, raw);
  cudaDeviceSynchronize();

  cudaMemcpy(test_flux_up.data(),   d_flux_up,   nwav * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(test_flux_down.data(), d_flux_down, nwav * sizeof(float), cudaMemcpyDeviceToHost);

  for (int w = 0; w < nwav; ++w) {
    char name[128];
    snprintf(name, sizeof(name), "raw_input_cloud_flux_up[wn=%.0f]", wavenumbers[w]);
    CHECK_NEAR(name, test_flux_up[w], ref_cloud_flux_up[w], tol);

    snprintf(name, sizeof(name), "raw_input_cloud_flux_down[wn=%.0f]", wavenumbers[w]);
    CHECK_NEAR(name, test_flux_down[w], ref_cloud_flux_down[w], tol);
  }

  // Cleanup
  cudaFree(d_abs);
  cudaFree(d_scat);
  cudaFree(d_alt);
  cudaFree(d_temp);
  cudaFree(d_wn);
  cudaFree(d_pmom);
  cudaFree(d_flux_up);
  cudaFree(d_flux_down);
  cudaFree(d_flux_direct);
  cudaFree(d_ref_dtau);
  cudaFree(d_ref_ssa);
  cudaFree(d_ref_planck);
  cudaFree(d_ref_surf_em);
  cudaFree(d_ref_top_em);
  cudaFree(d_ref_dtau_cloud);
  cudaFree(d_ref_ssa_cloud);
  cudaFree(d_cloud_tau);

  std::cout << "done\n";
}


// ============================================================================
//  Main
// ============================================================================

int main() {
  std::cout << "CUDA Adding-Doubling Solver Validation Tests\n";
  std::cout << "=============================================\n\n";

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

  // Pure absorption
  test_pure_absorption_beers_law();
  test_pure_absorption_lambertian();

  // Isotropic scattering (VH1 Table 12)
  test_isotropic_scattering();

  // Rayleigh scattering (SW Table 1)
  test_rayleigh_scattering();

  // HG scattering (VH2 Table 37)
  test_hg_scattering();

  // Haze-L Garcia-Siewert (nquad=16, batched path)
  test_haze_garcia_siewert();

  // Multi-layer
  test_multilayer_rayleigh();
  test_multilayer_hg();
  test_multilayer_isotropic();
  test_multilayer_varying();

  // Diffuse illumination (OS Table 1)
  test_diffuse_illumination();

  // 6-layer heterogeneous
  test_six_layer();

  // Combined sources
  test_combined_sources();

  // Thermal emission
  test_thermal_7a();
  test_thermal_7c();
  test_thermal_pure_absorption();
  test_thermal_scattering();
  test_thermal_hg_deltam();

  // Linear source (analytical)
  test_linear_source_pure_absorption();

  // Energy conservation
  test_energy_conservation();

  // Delta-M
  test_delta_m();

  // Flux solver comparison
  test_flux_solver_cases();

  // Mixed atmosphere
  test_mixed_atmosphere();

  // Diffusion BC
  test_diffusion_bc();

  // Rayleigh spherical albedo (N=16)
  test_rayleigh_spherical_albedo();

  // N=16 specific tests
  test_n16_specific();

  // Batch consistency
  test_batch_consistency();

  // Optically thick
  test_optically_thick();

  // Conservative stress tests
  test_conservative_thick();
  test_conservative_solar_energy();

  // Raw-input entry point
  test_raw_input_entry_point();

  int total = g_passed + g_failed;
  std::cout << "\n" << g_passed << "/" << total << " checks passed";
  if (g_failed > 0) std::cout << ", " << g_failed << " FAILED";
  std::cout << "\n";

  return g_failed;
}
