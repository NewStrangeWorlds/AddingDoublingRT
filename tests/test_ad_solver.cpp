/// @file test_ad_solver.cpp
/// @brief Comprehensive test suite for the adding-doubling RT solver.
///
/// Tests ported from the DisORT test suite (test_solver.cpp, test_flux_solver.cpp)
/// using the same CDISORT reference values from published benchmark tables.
///
/// References:
///   VH1 = Van de Hulst (1980), Multiple Light Scattering, Table 12
///   VH2 = Van de Hulst (1980), Table 37
///   SW  = Sweigart (1970), Table 1
///   GS  = Garcia & Siewert (1985), Tables 12-20
///   OS  = Ozisik & Shouman, Table 1

#include "testing.hpp"
#include "adding_doubling.hpp"

#include <cmath>
#include <vector>

static constexpr double PI = 3.14159265358979323846;


// ============================================================================
//  Helper: set Haze-L Garcia-Siewert phase function on ADConfig
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
//  Helper: set Cloud C.1 Garcia-Siewert phase function on ADConfig
// ============================================================================

static void setCloudGarciaSiewert(adrt::ADConfig& cfg, int lc) {
  static const double cldmom[299] = {
    2.544,3.883,4.568,5.235,5.887,6.457,7.177,7.859,8.494,9.286,9.856,10.615,11.229,11.851,12.503,
    13.058,13.626,14.209,14.660,15.231,15.641,16.126,16.539,16.934,17.325,17.673,17.999,18.329,18.588,
    18.885,19.103,19.345,19.537,19.721,19.884,20.024,20.145,20.251,20.330,20.401,20.444,20.477,20.489,
    20.483,20.467,20.427,20.382,20.310,20.236,20.136,20.036,19.909,19.785,19.632,19.486,19.311,19.145,
    18.949,18.764,18.551,18.348,18.119,17.901,17.659,17.428,17.174,16.931,16.668,16.415,16.144,15.883,
    15.606,15.338,15.058,14.784,14.501,14.225,13.941,13.662,13.378,13.098,12.816,12.536,12.257,11.978,
    11.703,11.427,11.156,10.884,10.618,10.350,10.090,9.827,9.574,9.318,9.072,8.822,8.584,8.340,8.110,
    7.874,7.652,7.424,7.211,6.990,6.785,6.573,6.377,6.173,5.986,5.790,5.612,5.424,5.255,5.075,4.915,
    4.744,4.592,4.429,4.285,4.130,3.994,3.847,3.719,3.580,3.459,3.327,3.214,3.090,2.983,2.866,2.766,
    2.656,2.562,2.459,2.372,2.274,2.193,2.102,2.025,1.940,1.869,1.790,1.723,1.649,1.588,1.518,1.461,
    1.397,1.344,1.284,1.235,1.179,1.134,1.082,1.040,0.992,0.954,0.909,0.873,0.832,0.799,0.762,0.731,
    0.696,0.668,0.636,0.610,0.581,0.557,0.530,0.508,0.483,0.463,0.440,0.422,0.401,0.384,0.364,0.349,
    0.331,0.317,0.301,0.288,0.273,0.262,0.248,0.238,0.225,0.215,0.204,0.195,0.185,0.177,0.167,0.160,
    0.151,0.145,0.137,0.131,0.124,0.118,0.112,0.107,0.101,0.097,0.091,0.087,0.082,0.079,0.074,0.071,
    0.067,0.064,0.060,0.057,0.054,0.052,0.049,0.047,0.044,0.042,0.039,0.038,0.035,0.034,0.032,0.030,
    0.029,0.027,0.026,0.024,0.023,0.022,0.021,0.020,0.018,0.018,0.017,0.016,0.015,0.014,0.013,0.013,
    0.012,0.011,0.011,0.010,0.009,0.009,0.008,0.008,0.008,0.007,0.007,0.006,0.006,0.006,0.005,0.005,
    0.005,0.005,0.004,0.004,0.004,0.004,0.003,0.003,0.003,0.003,0.003,0.003,0.002,0.002,0.002,0.002,
    0.002,0.002,0.002,0.002,0.002,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,
    0.001,0.001,0.001,0.001,0.001,0.001,0.001
  };

  int nmom = static_cast<int>(cfg.phase_function_moments[lc].size());
  cfg.phase_function_moments[lc][0] = 1.0;
  int nmom_limit = std::min(298, nmom - 1);
  for (int k = 1; k <= nmom_limit; ++k)
    cfg.phase_function_moments[lc][k] = cldmom[k - 1] / static_cast<double>(2 * k + 1);
  for (int k = nmom_limit + 1; k < nmom; ++k)
    cfg.phase_function_moments[lc][k] = 0.0;
}


// ============================================================================
//  Utility Tests
// ============================================================================

TEST(Utility, GaussLegendreWeightSum) {
  // Gauss-Legendre weights on [0,1] should sum to 1.
  for (int n : {2, 4, 8, 16, 32}) {
    std::vector<double> nodes, weights;
    adrt::gaussLegendre(n, nodes, weights);
    EXPECT_EQ(static_cast<int>(nodes.size()), n);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += weights[i];
    EXPECT_NEAR(sum, 1.0, 1e-14);
  }
}

TEST(Utility, GaussLegendreNodesInRange) {
  // All nodes should be in (0, 1).
  std::vector<double> nodes, weights;
  adrt::gaussLegendre(16, nodes, weights);
  for (int i = 0; i < 16; ++i) {
    EXPECT_GT(nodes[i], 0.0);
    EXPECT_LT(nodes[i], 1.0);
  }
}

TEST(Utility, PlanckFunctionBasic) {
  // Planck function at 300K, 500-600 cm^-1 should be positive.
  double B = adrt::planckFunction(500.0, 600.0, 300.0);
  EXPECT_GT(B, 0.0);

  // Higher temperature -> more emission.
  double B_hot = adrt::planckFunction(500.0, 600.0, 400.0);
  EXPECT_GT(B_hot, B);

  // Planck at T=0 should be zero.
  EXPECT_NEAR(adrt::planckFunction(500.0, 600.0, 0.0), 0.0, 1e-30);
}

TEST(Utility, PlanckStefanBoltzmann) {
  // Integral over all wavenumbers should approximate sigma*T^4/pi.
  double T = 300.0;
  double B = adrt::planckFunction(0.01, 100000.0, T);
  double sigma = 5.670374419e-8;
  double expected = sigma * T * T * T * T / PI;
  EXPECT_NEAR(B, expected, 0.01 * expected);  // 1% tolerance
}


// ============================================================================
//  Test 6b: Pure Absorption — Beer's Law (no scattering)
//  Ref: CDISORT test06 case b. RFLDIR = mu0*F0*exp(-tau/mu0). No diffuse.
// ============================================================================

TEST(ADSolver, PureAbsorption_BeersLaw) {
  // 1 layer, ssalb=0, beam source. Direct flux follows Beer's law exactly.
  adrt::ADConfig cfg(1, 8);
  cfg.solar_flux = 200.0;
  cfg.solar_mu = 0.5;
  cfg.surface_albedo = 0.0;
  cfg.allocate();
  cfg.delta_tau[0] = 1.0;
  cfg.single_scat_albedo[0] = 0.0;
  cfg.setIsotropic();

  auto r = adrt::solve(cfg);

  // At top: flux_direct = mu0 * F0 = 0.5 * 200 = 100
  EXPECT_NEAR(r.flux_direct[0], 100.0, 1e-6);
  EXPECT_NEAR(r.flux_down[0], 0.0, 1e-6);
  EXPECT_NEAR(r.flux_up[0], 0.0, 1e-6);

  // At bottom: flux_direct = 100 * exp(-1/0.5) = 100*exp(-2) = 13.5335
  EXPECT_NEAR(r.flux_direct[1], 100.0 * std::exp(-2.0), 1e-4);
  EXPECT_NEAR(r.flux_down[1], 0.0, 1e-6);
  EXPECT_NEAR(r.flux_up[1], 0.0, 1e-6);
}

TEST(ADSolver, PureAbsorption_LambertianSurface) {
  // Pure absorption with Lambertian surface (albedo=0.5).
  // flux_up at bottom = albedo * (flux_direct + flux_down) at bottom.
  // Upward flux attenuates through the layer.
  adrt::ADConfig cfg(1, 8);
  cfg.solar_flux = 200.0;
  cfg.solar_mu = 0.5;
  cfg.surface_albedo = 0.5;
  cfg.allocate();
  cfg.delta_tau[0] = 1.0;
  cfg.single_scat_albedo[0] = 0.0;
  cfg.setIsotropic();

  auto r = adrt::solve(cfg);

  // Direct beam at bottom
  double rfldir_bot = 100.0 * std::exp(-2.0);
  EXPECT_NEAR(r.flux_direct[1], rfldir_bot, 1e-4);

  // Upward flux at bottom = albedo * direct
  double flup_bot = 0.5 * rfldir_bot;
  EXPECT_NEAR(r.flux_up[1], flup_bot, 0.02 * flup_bot);

  // Upward flux at top should be attenuated surface reflection
  EXPECT_GT(r.flux_up[0], 0.0);
  EXPECT_LT(r.flux_up[0], r.flux_up[1]);
}


// ============================================================================
//  Test 1a-1d: Isotropic Scattering (VH1 Table 12)
//  Ref: CDISORT disotest cases 1a-1d
//  F0 = 10*pi, mu0 = 0.1, surface_albedo = 0, Lambertian
// ============================================================================

TEST(ADSolver, Isotropic1a) {
  // tau=0.03125, omega=0.2
  adrt::ADConfig cfg(1, 8);
  cfg.solar_flux = 10.0 * PI;
  cfg.solar_mu = 0.1;
  cfg.surface_albedo = 0.0;
  cfg.allocate();
  cfg.delta_tau[0] = 0.03125;
  cfg.single_scat_albedo[0] = 0.2;
  cfg.setIsotropic();

  auto r = adrt::solve(cfg);

  const double tol = 2e-2;
  EXPECT_NEAR(r.flux_direct[0], 3.1416, tol * 3.1416);
  EXPECT_NEAR(r.flux_direct[1], 2.2984, tol * 2.2984);
  EXPECT_NEAR(r.flux_down[0], 0.0, 1e-4);
  EXPECT_NEAR(r.flux_up[0], 7.9945e-02, tol * 7.9945e-02);
  EXPECT_NEAR(r.flux_down[1], 7.9411e-02, tol * 7.9411e-02);
  EXPECT_NEAR(r.flux_up[1], 0.0, 1e-4);
}

TEST(ADSolver, Isotropic1b) {
  // tau=0.03125, omega=1.0 (conservative)
  adrt::ADConfig cfg(1, 8);
  cfg.solar_flux = 10.0 * PI;
  cfg.solar_mu = 0.1;
  cfg.surface_albedo = 0.0;
  cfg.allocate();
  cfg.delta_tau[0] = 0.03125;
  cfg.single_scat_albedo[0] = 1.0;
  cfg.setIsotropic();

  auto r = adrt::solve(cfg);

  const double tol = 2e-2;
  EXPECT_NEAR(r.flux_direct[0], 3.1416, tol * 3.1416);
  EXPECT_NEAR(r.flux_direct[1], 2.2984, tol * 2.2984);
  EXPECT_NEAR(r.flux_up[0], 0.42292, tol * 0.42292);
  EXPECT_NEAR(r.flux_down[1], 0.42023, tol * 0.42023);
}

TEST(ADSolver, Isotropic1c) {
  // tau=0.03125, omega=0.99 (nearly conservative)
  adrt::ADConfig cfg(1, 8);
  cfg.solar_flux = 10.0 * PI;
  cfg.solar_mu = 0.1;
  cfg.surface_albedo = 0.0;
  cfg.allocate();
  cfg.delta_tau[0] = 0.03125;
  cfg.single_scat_albedo[0] = 0.99;
  cfg.setIsotropic();

  auto r = adrt::solve(cfg);

  const double tol = 2e-2;
  EXPECT_NEAR(r.flux_direct[0], 3.1416, tol * 3.1416);
  EXPECT_NEAR(r.flux_direct[1], 2.2984, tol * 2.2984);
  // Upward flux should be between 1a (omega=0.2) and 1b (omega=1.0)
  EXPECT_GT(r.flux_up[0], 0.07);
  EXPECT_LT(r.flux_up[0], 0.43);
}

TEST(ADSolver, Isotropic1d) {
  // tau=32, omega=0.2 (optically thick)
  adrt::ADConfig cfg(1, 8);
  cfg.solar_flux = 10.0 * PI;
  cfg.solar_mu = 0.1;
  cfg.surface_albedo = 0.0;
  cfg.allocate();
  cfg.delta_tau[0] = 32.0;
  cfg.single_scat_albedo[0] = 0.2;
  cfg.setIsotropic();

  auto r = adrt::solve(cfg);

  const double tol = 2e-2;
  EXPECT_NEAR(r.flux_direct[0], 3.1416, tol * 3.1416);
  // At bottom, beam is heavily attenuated: exp(-32/0.1) ~ 0
  EXPECT_LT(r.flux_direct[1], 1e-6);
  EXPECT_FALSE(std::isnan(r.flux_up[0]));
  EXPECT_FALSE(std::isnan(r.flux_up[1]));
}


// ============================================================================
//  Test 2a-2d: Rayleigh Scattering (SW Table 1)
//  Ref: CDISORT disotest cases 2a-2d
//  F0 = pi, mu0 = 0.080442, surface_albedo = 0
// ============================================================================

TEST(ADSolver, Rayleigh2a) {
  // tau=0.2, omega=0.5
  adrt::ADConfig cfg(1, 8);
  cfg.solar_flux = PI;
  cfg.solar_mu = 0.080442;
  cfg.surface_albedo = 0.0;
  cfg.allocate();
  cfg.delta_tau[0] = 0.2;
  cfg.single_scat_albedo[0] = 0.5;
  cfg.setRayleigh();

  auto r = adrt::solve(cfg);

  const double tol = 2e-2;
  EXPECT_NEAR(r.flux_direct[0], 2.52716e-01, tol * 2.52716e-01);
  EXPECT_NEAR(r.flux_direct[1], 2.10311e-02, tol * 2.10311e-02);
  EXPECT_NEAR(r.flux_down[0], 0.0, 1e-6);
  EXPECT_NEAR(r.flux_up[1], 0.0, 1e-6);
  EXPECT_NEAR(r.flux_up[0], 5.35063e-02, tol * 5.35063e-02);
  EXPECT_NEAR(r.flux_down[1], 4.41794e-02, tol * 4.41794e-02);
}

TEST(ADSolver, Rayleigh2b) {
  // tau=0.2, omega=1.0 (conservative Rayleigh)
  adrt::ADConfig cfg(1, 8);
  cfg.solar_flux = PI;
  cfg.solar_mu = 0.080442;
  cfg.surface_albedo = 0.0;
  cfg.allocate();
  cfg.delta_tau[0] = 0.2;
  cfg.single_scat_albedo[0] = 1.0;
  cfg.setRayleigh();

  auto r = adrt::solve(cfg);

  const double tol = 2e-2;
  EXPECT_NEAR(r.flux_direct[0], 2.52716e-01, tol * 2.52716e-01);
  EXPECT_NEAR(r.flux_direct[1], 2.10311e-02, tol * 2.10311e-02);
  EXPECT_NEAR(r.flux_up[0], 1.25561e-01, tol * 1.25561e-01);
  EXPECT_NEAR(r.flux_down[1], 1.06123e-01, tol * 1.06123e-01);
}

TEST(ADSolver, Rayleigh2c) {
  // tau=5.0, omega=0.5 (optically thick Rayleigh)
  adrt::ADConfig cfg(1, 8);
  cfg.solar_flux = PI;
  cfg.solar_mu = 0.080442;
  cfg.surface_albedo = 0.0;
  cfg.allocate();
  cfg.delta_tau[0] = 5.0;
  cfg.single_scat_albedo[0] = 0.5;
  cfg.setRayleigh();

  auto r = adrt::solve(cfg);

  const double tol = 2e-2;
  EXPECT_NEAR(r.flux_direct[0], 2.52716e-01, tol * 2.52716e-01);
  EXPECT_LT(r.flux_direct[1], 1e-20);
  EXPECT_NEAR(r.flux_up[0], 6.24730e-02, tol * 6.24730e-02);
  EXPECT_NEAR(r.flux_down[1], 2.51683e-04, 3e-5);
}

TEST(ADSolver, Rayleigh2d) {
  // tau=5.0, omega=1.0 (optically thick, conservative Rayleigh)
  adrt::ADConfig cfg(1, 8);
  cfg.solar_flux = PI;
  cfg.solar_mu = 0.080442;
  cfg.surface_albedo = 0.0;
  cfg.allocate();
  cfg.delta_tau[0] = 5.0;
  cfg.single_scat_albedo[0] = 1.0;
  cfg.setRayleigh();

  auto r = adrt::solve(cfg);

  const double tol = 2e-2;
  EXPECT_NEAR(r.flux_direct[0], 2.52716e-01, tol * 2.52716e-01);
  EXPECT_LT(r.flux_direct[1], 1e-20);
  EXPECT_NEAR(r.flux_up[0], 2.25915e-01, tol * 2.25915e-01);
  EXPECT_NEAR(r.flux_down[1], 2.68008e-02, tol * 2.68008e-02);
}


// ============================================================================
//  Test 3a-3b: HG Scattering (VH2 Table 37)
//  g=0.75, omega=1.0, F0=pi, mu0=1.0, surface_albedo=0
// ============================================================================

TEST(ADSolver, HenyeyGreenstein3a) {
  // tau=1.0, g=0.75, omega=1.0
  adrt::ADConfig cfg(1, 8);
  cfg.solar_flux = PI;
  cfg.solar_mu = 1.0;
  cfg.surface_albedo = 0.0;
  cfg.allocate();
  cfg.delta_tau[0] = 1.0;
  cfg.single_scat_albedo[0] = 1.0;
  cfg.setHenyeyGreenstein(0.75);

  auto r = adrt::solve(cfg);

  const double tol = 2e-2;
  EXPECT_NEAR(r.flux_direct[0], 3.14159, tol * 3.14159);
  EXPECT_NEAR(r.flux_direct[1], 1.15573, tol * 1.15573);
  EXPECT_NEAR(r.flux_up[0], 2.47374e-01, tol * 2.47374e-01);
  EXPECT_NEAR(r.flux_down[1], 1.73849, tol * 1.73849);
  EXPECT_NEAR(r.flux_up[1], 0.0, 1e-6);
}

TEST(ADSolver, HenyeyGreenstein3b) {
  // tau=8.0, g=0.75, omega=1.0 (optically thick HG)
  adrt::ADConfig cfg(1, 8);
  cfg.solar_flux = PI;
  cfg.solar_mu = 1.0;
  cfg.surface_albedo = 0.0;
  cfg.allocate();
  cfg.delta_tau[0] = 8.0;
  cfg.single_scat_albedo[0] = 1.0;
  cfg.setHenyeyGreenstein(0.75);

  auto r = adrt::solve(cfg);

  const double tol = 5e-2;  // Wider for challenging case
  EXPECT_NEAR(r.flux_direct[0], 3.14159, 2e-2 * 3.14159);
  EXPECT_NEAR(r.flux_direct[1], 1.05389e-03, 2e-2 * 1.05389e-03);
  EXPECT_NEAR(r.flux_up[0], 1.59096, tol * 1.59096);
  EXPECT_NEAR(r.flux_down[1], 1.54958, tol * 1.54958);
}


// ============================================================================
//  Test 4a-4c: Haze-L Garcia-Siewert (GS Tables 12-16)
//  Realistic aerosol phase function, nstr=32
// ============================================================================

TEST(ADSolver, HazeGarciaSiewert4a) {
  // tau=1.0, omega=1.0, mu0=1.0, nquad=16 (32 streams)
  adrt::ADConfig cfg(1, 16);
  cfg.solar_flux = PI;
  cfg.solar_mu = 1.0;
  cfg.surface_albedo = 0.0;
  cfg.allocate();
  cfg.delta_tau[0] = 1.0;
  cfg.single_scat_albedo[0] = 1.0;
  setHazeGarciaSiewert(cfg, 0);

  auto r = adrt::solve(cfg);

  const double tol = 2e-2;
  EXPECT_NEAR(r.flux_direct[0], 3.14159, tol * 3.14159);
  EXPECT_NEAR(r.flux_direct[1], 1.15573, tol * 1.15573);
  EXPECT_NEAR(r.flux_up[0], 1.73223e-01, tol * 1.73223e-01);
  EXPECT_NEAR(r.flux_down[1], 1.81264, tol * 1.81264);
}

TEST(ADSolver, HazeGarciaSiewert4b) {
  // tau=1.0, omega=0.9 (absorbing Haze-L)
  adrt::ADConfig cfg(1, 16);
  cfg.solar_flux = PI;
  cfg.solar_mu = 1.0;
  cfg.surface_albedo = 0.0;
  cfg.allocate();
  cfg.delta_tau[0] = 1.0;
  cfg.single_scat_albedo[0] = 0.9;
  setHazeGarciaSiewert(cfg, 0);

  auto r = adrt::solve(cfg);

  const double tol = 2e-2;
  EXPECT_NEAR(r.flux_direct[0], 3.14159, tol * 3.14159);
  EXPECT_NEAR(r.flux_direct[1], 1.15573, tol * 1.15573);
  EXPECT_NEAR(r.flux_up[0], 1.23665e-01, tol * 1.23665e-01);
  EXPECT_NEAR(r.flux_down[1], 1.51554, tol * 1.51554);
}

TEST(ADSolver, HazeGarciaSiewert4c) {
  // tau=1.0, omega=0.9, mu0=0.5 (tilted sun)
  adrt::ADConfig cfg(1, 16);
  cfg.solar_flux = PI;
  cfg.solar_mu = 0.5;
  cfg.surface_albedo = 0.0;
  cfg.allocate();
  cfg.delta_tau[0] = 1.0;
  cfg.single_scat_albedo[0] = 0.9;
  setHazeGarciaSiewert(cfg, 0);

  auto r = adrt::solve(cfg);

  const double tol = 2e-2;
  EXPECT_NEAR(r.flux_direct[0], 1.57080, tol * 1.57080);
  EXPECT_NEAR(r.flux_direct[1], 2.12584e-01, tol * 2.12584e-01);
  EXPECT_NEAR(r.flux_up[0], 2.25487e-01, tol * 2.25487e-01);
  EXPECT_NEAR(r.flux_down[1], 8.03294e-01, tol * 8.03294e-01);
}


// ============================================================================
//  Multi-Layer Validation Tests
//  Reference values from CDISORT
// ============================================================================

TEST(ADSolver, MultiLayer1_TwoLayerRayleigh) {
  // 2-layer Rayleigh-like scattering with different properties per layer.
  // Layer 0: tau=0.1, omega=0.5, Rayleigh P2=0.1
  // Layer 1: tau=0.2, omega=0.8, Rayleigh P2=0.1
  // F0=1, mu0=0.5, albedo=0
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

  auto r = adrt::solve(cfg);

  const double tol = 2e-2;
  // Top (tau=0)
  EXPECT_NEAR(r.flux_direct[0], 5.0e-01, tol * 5.0e-01);
  EXPECT_NEAR(r.flux_down[0], 0.0, 1e-6);
  EXPECT_NEAR(r.flux_up[0], 6.593572e-02, tol * 6.593572e-02);

  // Interface (tau=0.1)
  EXPECT_NEAR(r.flux_direct[1], 4.093654e-01, tol * 4.093654e-01);
  EXPECT_NEAR(r.flux_down[1], 2.327019e-02, tol * 2.327019e-02);
  EXPECT_NEAR(r.flux_up[1], 5.456810e-02, tol * 5.456810e-02);

  // Bottom (tau=0.3)
  EXPECT_NEAR(r.flux_direct[2], 2.744058e-01, tol * 2.744058e-01);
  EXPECT_NEAR(r.flux_down[2], 6.715049e-02, tol * 6.715049e-02);
  EXPECT_NEAR(r.flux_up[2], 0.0, 1e-6);
}

TEST(ADSolver, MultiLayer2_TwoLayerHG) {
  // 2-layer HG, equivalent to test 3a split at tau=0.5.
  // Both layers: tau=0.5, omega=1.0, g=0.75. Total tau=1.0.
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

  auto r = adrt::solve(cfg);

  const double tol = 2e-2;
  const double tol5 = 5e-2;

  // Top must match test 3a top values
  EXPECT_NEAR(r.flux_direct[0], 3.141593e+00, tol * 3.141593e+00);
  EXPECT_NEAR(r.flux_down[0], 0.0, 1e-5);
  EXPECT_NEAR(r.flux_up[0], 2.473744e-01, tol * 2.473744e-01);

  // Middle (tau=0.5)
  EXPECT_NEAR(r.flux_direct[1], 1.905472e+00, tol * 1.905472e+00);
  EXPECT_NEAR(r.flux_down[1], 1.149116e+00, tol5 * 1.149116e+00);
  EXPECT_NEAR(r.flux_up[1], 1.603704e-01, tol5 * 1.603704e-01);

  // Bottom must match test 3a bottom values
  EXPECT_NEAR(r.flux_direct[2], 1.155727e+00, tol * 1.155727e+00);
  EXPECT_NEAR(r.flux_down[2], 1.738491e+00, tol * 1.738491e+00);
  EXPECT_NEAR(r.flux_up[2], 0.0, 1e-5);
}

TEST(ADSolver, MultiLayer3_ThreeLayerIsotropic) {
  // 3-layer isotropic with heterogeneous layers and reflective surface.
  // Layer 0: tau=0.5, omega=0.5
  // Layer 1: tau=1.0, omega=0.9
  // Layer 2: tau=0.5, omega=0.5
  // F0=1, mu0=0.5, albedo=0.1
  adrt::ADConfig cfg(3, 8);
  cfg.solar_flux = 1.0;
  cfg.solar_mu = 0.5;
  cfg.surface_albedo = 0.1;
  cfg.allocate();

  cfg.delta_tau[0] = 0.5;  cfg.single_scat_albedo[0] = 0.5;  cfg.setIsotropic(0);
  cfg.delta_tau[1] = 1.0;  cfg.single_scat_albedo[1] = 0.9;  cfg.setIsotropic(1);
  cfg.delta_tau[2] = 0.5;  cfg.single_scat_albedo[2] = 0.5;  cfg.setIsotropic(2);

  auto r = adrt::solve(cfg);

  const double tol = 2e-2;

  EXPECT_NEAR(r.flux_direct[0], 5.0e-01, tol * 5.0e-01);
  EXPECT_NEAR(r.flux_down[0], 0.0, 1e-6);
  EXPECT_NEAR(r.flux_up[0], 1.133613e-01, tol * 1.133613e-01);

  EXPECT_NEAR(r.flux_direct[1], 1.839397e-01, tol * 1.839397e-01);
  EXPECT_NEAR(r.flux_down[1], 6.416506e-02, tol * 6.416506e-02);
  EXPECT_NEAR(r.flux_up[1], 1.031752e-01, tol * 1.031752e-01);

  EXPECT_NEAR(r.flux_direct[2], 2.489353e-02, tol * 2.489353e-02);
  EXPECT_NEAR(r.flux_down[2], 8.605320e-02, tol * 8.605320e-02);
  EXPECT_NEAR(r.flux_up[2], 1.536433e-02, tol * 1.536433e-02);

  EXPECT_NEAR(r.flux_direct[3], 9.157819e-03, tol * 9.157819e-03);
  EXPECT_NEAR(r.flux_down[3], 4.944670e-02, tol * 4.944670e-02);
  EXPECT_NEAR(r.flux_up[3], 5.860452e-03, tol * 5.860452e-03);
}


// ============================================================================
//  Test 8a-8c: Two-Layer Isotropic, Diffuse Illumination (OS Table 1)
//  isotropic_flux_top = 1/pi, no beam, surface_albedo=0
// ============================================================================

TEST(ADSolver, DiffuseIllumination8a) {
  // ssalb=[0.5, 0.3], dtau=[0.25, 0.25]
  adrt::ADConfig cfg(2, 4);
  cfg.top_emission = 1.0 / PI;
  cfg.surface_albedo = 0.0;
  cfg.allocate();

  cfg.delta_tau[0] = 0.25;  cfg.single_scat_albedo[0] = 0.5;  cfg.setIsotropic(0);
  cfg.delta_tau[1] = 0.25;  cfg.single_scat_albedo[1] = 0.3;  cfg.setIsotropic(1);

  auto r = adrt::solve(cfg);

  const double tol = 2e-2;

  // Top: flux_down = pi * (1/pi) = 1.0
  EXPECT_NEAR(r.flux_direct[0], 0.0, 1e-8);
  EXPECT_NEAR(r.flux_down[0], 1.0, tol * 1.0);
  EXPECT_NEAR(r.flux_up[0], 9.29633e-02, tol * 9.29633e-02);

  // Interface (tau=0.25)
  EXPECT_NEAR(r.flux_direct[1], 0.0, 1e-8);
  EXPECT_NEAR(r.flux_down[1], 7.22235e-01, tol * 7.22235e-01);
  EXPECT_NEAR(r.flux_up[1], 2.78952e-02, tol * 2.78952e-02);

  // Bottom (tau=0.5)
  EXPECT_NEAR(r.flux_direct[2], 0.0, 1e-8);
  EXPECT_NEAR(r.flux_down[2], 5.13132e-01, tol * 5.13132e-01);
  EXPECT_NEAR(r.flux_up[2], 0.0, 1e-6);
}

TEST(ADSolver, DiffuseIllumination8b) {
  // ssalb=[0.8, 0.95], dtau=[0.25, 0.25]
  adrt::ADConfig cfg(2, 4);
  cfg.top_emission = 1.0 / PI;
  cfg.surface_albedo = 0.0;
  cfg.allocate();

  cfg.delta_tau[0] = 0.25;  cfg.single_scat_albedo[0] = 0.8;  cfg.setIsotropic(0);
  cfg.delta_tau[1] = 0.25;  cfg.single_scat_albedo[1] = 0.95; cfg.setIsotropic(1);

  auto r = adrt::solve(cfg);

  const double tol = 2e-2;

  EXPECT_NEAR(r.flux_down[0], 1.0, tol * 1.0);
  EXPECT_NEAR(r.flux_up[0], 2.25136e-01, tol * 2.25136e-01);

  EXPECT_NEAR(r.flux_down[1], 7.95332e-01, tol * 7.95332e-01);
  EXPECT_NEAR(r.flux_up[1], 1.26349e-01, tol * 1.26349e-01);

  EXPECT_NEAR(r.flux_down[2], 6.50417e-01, tol * 6.50417e-01);
  EXPECT_NEAR(r.flux_up[2], 0.0, 1e-6);
}

TEST(ADSolver, DiffuseIllumination8c) {
  // ssalb=[0.8, 0.95], dtau=[1.0, 2.0] (thicker layers)
  adrt::ADConfig cfg(2, 4);
  cfg.top_emission = 1.0 / PI;
  cfg.surface_albedo = 0.0;
  cfg.allocate();

  cfg.delta_tau[0] = 1.0;  cfg.single_scat_albedo[0] = 0.8;  cfg.setIsotropic(0);
  cfg.delta_tau[1] = 2.0;  cfg.single_scat_albedo[1] = 0.95; cfg.setIsotropic(1);

  auto r = adrt::solve(cfg);

  const double tol = 2e-2;

  EXPECT_NEAR(r.flux_down[0], 1.0, tol * 1.0);
  EXPECT_NEAR(r.flux_up[0], 3.78578e-01, tol * 3.78578e-01);

  EXPECT_NEAR(r.flux_down[1], 4.86157e-01, tol * 4.86157e-01);
  EXPECT_NEAR(r.flux_up[1], 2.43397e-01, tol * 2.43397e-01);

  EXPECT_NEAR(r.flux_down[2], 1.59984e-01, tol * 1.59984e-01);
  EXPECT_NEAR(r.flux_up[2], 0.0, 1e-6);
}


// ============================================================================
//  Test 9a-9b: 6-Layer Heterogeneous Atmosphere (CDISORT reference)
//  isotropic_flux_top = 1/pi, no beam, surface_albedo=0
// ============================================================================

TEST(ADSolver, SixLayer9a_Isotropic) {
  // dtau[lc] = lc+1, ssalb[lc] = 0.65 + lc*0.05, isotropic phase function
  adrt::ADConfig cfg(6, 4);
  cfg.top_emission = 1.0 / PI;
  cfg.surface_albedo = 0.0;
  cfg.allocate();

  for (int lc = 0; lc < 6; ++lc) {
    cfg.delta_tau[lc] = static_cast<double>(lc + 1);
    cfg.single_scat_albedo[lc] = 0.65 + lc * 0.05;
    cfg.setIsotropic(lc);
  }

  auto r = adrt::solve(cfg);

  const double tol = 2e-2;

  const double tol5 = 5e-2;

  // tau=0 (top)
  EXPECT_NEAR(r.flux_down[0], 1.0, tol * 1.0);
  EXPECT_NEAR(r.flux_up[0], 2.279734e-01, tol5 * 2.279734e-01);

  // tau=1 (layer 0/1 interface) — wider tolerance for nquad=4
  EXPECT_NEAR(r.flux_up[1], 8.750978e-02, tol5 * 8.750978e-02);

  // tau=21 (bottom)
  EXPECT_NEAR(r.flux_up[6], 0.0, 1e-6);
}

TEST(ADSolver, SixLayer9b_Anisotropic) {
  // Same as 9a but with DGIS anisotropic phase function.
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

  auto r = adrt::solve(cfg);

  const double tol = 2e-2;

  EXPECT_NEAR(r.flux_down[0], 1.0, tol * 1.0);
  EXPECT_NEAR(r.flux_up[0], 1.000789e-01, tol * 1.000789e-01);
  EXPECT_NEAR(r.flux_up[6], 0.0, 1e-6);
}


// ============================================================================
//  Test 11a-11b: Combined Beam + Isotropic Illumination + Surface Albedo
//  Ref: CDISORT disotest case 11
// ============================================================================

TEST(ADSolver, CombinedSources11a) {
  // 1 layer, beam + isotropic + surface albedo.
  // F0=1, mu0=0.5, isotropic_flux_top=0.5/pi, albedo=0.5
  // tau=1.0, omega=0.9, isotropic
  adrt::ADConfig cfg(1, 8);
  cfg.solar_flux = 1.0;
  cfg.solar_mu = 0.5;
  cfg.top_emission = 0.5 / PI;
  cfg.surface_albedo = 0.5;
  cfg.allocate();
  cfg.delta_tau[0] = 1.0;
  cfg.single_scat_albedo[0] = 0.9;
  cfg.setIsotropic();

  auto r = adrt::solve(cfg);

  const double tol = 2e-2;

  // tau=0: flux_direct=0.5, flux_down=0.5 (isotropic), flux_up=0.5014
  EXPECT_NEAR(r.flux_direct[0], 5.000000e-01, tol * 5.0e-01);
  EXPECT_NEAR(r.flux_down[0], 5.000000e-01, tol * 5.0e-01);
  EXPECT_NEAR(r.flux_up[0], 5.013753e-01, tol * 5.013753e-01);

  // tau=1: flux_direct=exp(-2)*0.5=0.06767
  EXPECT_NEAR(r.flux_direct[1], 6.766764e-02, tol * 6.766764e-02);
  EXPECT_NEAR(r.flux_down[1], 4.723631e-01, tol * 4.723631e-01);
  EXPECT_NEAR(r.flux_up[1], 2.700154e-01, tol * 2.700154e-01);
}

TEST(ADSolver, CombinedSources11b_LayerSplit) {
  // Same as 11a but split into 3 layers: [0,0.05], [0.05,0.5], [0.5,1.0].
  // Layer-splitting invariance: top/bottom fluxes must match 11a.
  adrt::ADConfig cfg(3, 8);
  cfg.solar_flux = 1.0;
  cfg.solar_mu = 0.5;
  cfg.top_emission = 0.5 / PI;
  cfg.surface_albedo = 0.5;
  cfg.allocate();

  cfg.delta_tau[0] = 0.05;  cfg.single_scat_albedo[0] = 0.9;  cfg.setIsotropic(0);
  cfg.delta_tau[1] = 0.45;  cfg.single_scat_albedo[1] = 0.9;  cfg.setIsotropic(1);
  cfg.delta_tau[2] = 0.50;  cfg.single_scat_albedo[2] = 0.9;  cfg.setIsotropic(2);

  auto r = adrt::solve(cfg);

  const double tol = 2e-2;

  // Top: must match 11a
  EXPECT_NEAR(r.flux_direct[0], 5.000000e-01, tol * 5.0e-01);
  EXPECT_NEAR(r.flux_down[0], 5.000000e-01, tol * 5.0e-01);
  EXPECT_NEAR(r.flux_up[0], 5.013753e-01, tol * 5.013753e-01);

  // Bottom: must match 11a
  EXPECT_NEAR(r.flux_direct[3], 6.766764e-02, tol * 6.766764e-02);
  EXPECT_NEAR(r.flux_down[3], 4.723631e-01, tol * 4.723631e-01);
  EXPECT_NEAR(r.flux_up[3], 2.700154e-01, tol * 2.700154e-01);
}


// ============================================================================
//  Thermal Emission Tests
//  Ref: CDISORT disotest cases 7a, 7c
// ============================================================================

TEST(ADSolver, Thermal7a) {
  // Planck-only emission, no beam.
  // 1 layer, HG g=0.05, tau=1, omega=0.1, T=[200,300], wvnm=[300,800]
  // DisORT: temperature_top=0, temperature_bottom=0 → no boundary emission,
  //         only internal Planck emission from the temperature profile.
  adrt::ADConfig cfg(1, 8);
  cfg.surface_albedo = 0.0;
  cfg.allocate();

  cfg.delta_tau[0] = 1.0;
  cfg.single_scat_albedo[0] = 0.1;
  cfg.setHenyeyGreenstein(0.05);

  // Use planck_levels directly to separate boundary emission from internal emission.
  // DisORT's temperature_top=0, temperature_bottom=0 means no boundary Planck emission.
  cfg.planck_levels = {
    adrt::planckFunction(300.0, 800.0, 200.0),
    adrt::planckFunction(300.0, 800.0, 300.0)
  };
  cfg.top_emission = 0.0;
  cfg.surface_emission = 0.0;

  auto r = adrt::solve(cfg);

  const double tol = 2e-2;
  EXPECT_NEAR(r.flux_up[0], 8.62935618e+01, tol * 8.62935618e+01);
  EXPECT_NEAR(r.flux_down[1], 1.21203517e+02, tol * 1.21203517e+02);
}

TEST(ADSolver, Thermal7c) {
  // Planck + beam + isotropic, HG g=0.8, nstr=12 (nquad=6).
  // tau=1, omega=0.5, F0=200, mu0=0.5
  // DisORT: isotropic_flux_top=100, temperature_top=100, temperature_bottom=320
  //         TEMPER=[300,200], wvnm=[0,50000], albedo=0
  adrt::ADConfig cfg(1, 6);
  cfg.solar_flux = 200.0;
  cfg.solar_mu = 0.5;
  cfg.surface_albedo = 0.0;
  cfg.allocate();

  cfg.delta_tau[0] = 1.0;
  cfg.single_scat_albedo[0] = 0.5;
  cfg.setHenyeyGreenstein(0.8);

  // Use planck_levels directly to separate boundary vs internal emission.
  // DisORT: temperature=[300,200] (internal), temperature_top=100 (boundary),
  //         temperature_bottom=320 (surface boundary), isotropic_flux_top=100
  cfg.planck_levels = {
    adrt::planckFunction(0.01, 50000.0, 300.0),
    adrt::planckFunction(0.01, 50000.0, 200.0)
  };
  // TOA boundary: isotropic diffuse (100) + Planck at temperature_top (100K)
  cfg.top_emission = 100.0 + adrt::planckFunction(0.01, 50000.0, 100.0);
  // Surface boundary: Planck at temperature_bottom (320K)
  cfg.surface_emission = adrt::planckFunction(0.01, 50000.0, 320.0);

  auto r = adrt::solve(cfg);

  const double tol = 2e-2;
  EXPECT_NEAR(r.flux_direct[0], 1.00000000e+02, tol * 1.00000000e+02);
  EXPECT_NEAR(r.flux_up[0], 4.29571753e+02, tol * 4.29571753e+02);
}


// ============================================================================
//  Thermal Emission: Cross-Validation with DisORT FluxSolver
//  These match the compare_solvers tests that showed ~1e-6 agreement.
// ============================================================================

TEST(ADSolver, ThermalPureAbsorption) {
  // 3-layer pure absorption, thermal emission.
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

  auto r = adrt::solve(cfg);

  // All fluxes should be positive and physical
  for (int l = 0; l <= 3; ++l) {
    EXPECT_GT(r.flux_up[l], 0.0);
    EXPECT_FALSE(std::isnan(r.flux_up[l]));
    EXPECT_FALSE(std::isnan(r.flux_down[l]));
  }
  // Upward flux should generally increase toward surface (warmer)
  EXPECT_GT(r.flux_up[3], r.flux_up[0]);
}

TEST(ADSolver, ThermalScattering) {
  // Thermal + scattering: 4-layer isotropic, omega=1.0
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

  auto r = adrt::solve(cfg);

  for (int l = 0; l <= 4; ++l) {
    EXPECT_FALSE(std::isnan(r.flux_up[l]));
    EXPECT_FALSE(std::isnan(r.flux_down[l]));
  }
}

TEST(ADSolver, ThermalHGDeltaM) {
  // Thermal + HG g=0.8, delta-M, surface albedo.
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

  auto r = adrt::solve(cfg);

  for (int l = 0; l <= 3; ++l) {
    EXPECT_GT(r.flux_up[l], 0.0);
    EXPECT_FALSE(std::isnan(r.flux_up[l]));
    EXPECT_FALSE(std::isnan(r.flux_down[l]));
  }
}


// ============================================================================
//  Energy Conservation Tests
// ============================================================================

TEST(ADSolver, EnergyConservation_Conservative) {
  // For omega=1 (no absorption), net flux should be constant through all layers.
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

  auto r = adrt::solve(cfg);

  // Net flux = F_up - F_down - F_direct should be constant
  double fnet0 = r.flux_up[0] - r.flux_down[0] - r.flux_direct[0];
  for (int l = 1; l <= nlay; ++l) {
    double fnet = r.flux_up[l] - r.flux_down[l] - r.flux_direct[l];
    EXPECT_NEAR(fnet, fnet0, 1e-5);
  }
}

TEST(ADSolver, EnergyConservation_DeltaM) {
  // Energy conservation with delta-M enabled (omega=1).
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

  auto r = adrt::solve(cfg);

  double fnet0 = r.flux_up[0] - r.flux_down[0] - r.flux_direct[0];
  for (int l = 1; l <= nlay; ++l) {
    double fnet = r.flux_up[l] - r.flux_down[l] - r.flux_direct[l];
    EXPECT_NEAR(fnet, fnet0, 1e-5);
  }
}


// ============================================================================
//  Delta-M Tests
// ============================================================================

TEST(ADSolver, DeltaM_RayleighNoEffect) {
  // Rayleigh scattering has f=0, so delta-M should have no effect.
  adrt::ADConfig cfg_off(1, 8);
  cfg_off.solar_flux = 1.0;
  cfg_off.solar_mu = 1.0;
  cfg_off.allocate();
  cfg_off.delta_tau[0] = 0.5;
  cfg_off.single_scat_albedo[0] = 1.0;
  cfg_off.setRayleigh();

  adrt::ADConfig cfg_on = cfg_off;
  cfg_on.use_delta_m = true;

  // Need to reallocate for delta-M (different number of moments)
  adrt::ADConfig cfg_on2(1, 8);
  cfg_on2.use_delta_m = true;
  cfg_on2.solar_flux = 1.0;
  cfg_on2.solar_mu = 1.0;
  cfg_on2.allocate();
  cfg_on2.delta_tau[0] = 0.5;
  cfg_on2.single_scat_albedo[0] = 1.0;
  cfg_on2.setRayleigh();

  auto r_off = adrt::solve(cfg_off);
  auto r_on = adrt::solve(cfg_on2);

  EXPECT_NEAR(r_off.flux_up[0], r_on.flux_up[0], 1e-10);
  EXPECT_NEAR(r_off.flux_down[1], r_on.flux_down[1], 1e-10);
}

TEST(ADSolver, DeltaM_ConvergenceFasterForHG) {
  // With delta-M, convergence w.r.t. quadrature order should be faster
  // for forward-peaked HG (g=0.9).
  double results_off[2], results_on[2];

  for (int idx = 0; idx < 2; ++idx) {
    int nq = (idx == 0) ? 4 : 16;

    adrt::ADConfig cfg_off(1, nq);
    cfg_off.solar_flux = 1.0;
    cfg_off.solar_mu = 0.5;
    cfg_off.allocate();
    cfg_off.delta_tau[0] = 1.0;
    cfg_off.single_scat_albedo[0] = 0.9;
    cfg_off.setHenyeyGreenstein(0.9);
    auto r_off = adrt::solve(cfg_off);
    results_off[idx] = r_off.flux_up[0];

    adrt::ADConfig cfg_on(1, nq);
    cfg_on.use_delta_m = true;
    cfg_on.solar_flux = 1.0;
    cfg_on.solar_mu = 0.5;
    cfg_on.allocate();
    cfg_on.delta_tau[0] = 1.0;
    cfg_on.single_scat_albedo[0] = 0.9;
    cfg_on.setHenyeyGreenstein(0.9);
    auto r_on = adrt::solve(cfg_on);
    results_on[idx] = r_on.flux_up[0];
  }

  // The spread between nq=4 and nq=16 should be smaller with delta-M
  double spread_off = std::abs(results_off[0] - results_off[1]);
  double spread_on = std::abs(results_on[0] - results_on[1]);
  EXPECT_LT(spread_on, spread_off);
}


// ============================================================================
//  Flux Solver Comparison Tests (from test_flux_solver.cpp)
//  These test AD against the same scenarios as DisortFluxSolver tests.
// ============================================================================

TEST(ADSolver, FluxSolver_IsotropicBeam) {
  // 2 layers, F0=pi, mu0=0.6, albedo=0.1, omega=0.9
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

  auto r = adrt::solve(cfg);

  // Basic sanity: fluxes are physical
  EXPECT_GT(r.flux_direct[0], 0.0);
  EXPECT_GT(r.flux_up[0], 0.0);
  EXPECT_GT(r.flux_down[2], 0.0);
  EXPECT_GT(r.flux_up[2], 0.0);  // Non-zero due to surface albedo
}

TEST(ADSolver, FluxSolver_MultiLayer) {
  // 10 layers with varying properties
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

  auto r = adrt::solve(cfg);

  // Check all fluxes are finite and physical
  for (int l = 0; l <= nlyr; ++l) {
    EXPECT_FALSE(std::isnan(r.flux_up[l]));
    EXPECT_FALSE(std::isnan(r.flux_down[l]));
    EXPECT_FALSE(std::isnan(r.flux_direct[l]));
    EXPECT_FALSE(std::isnan(r.mean_intensity[l]));
  }

  // Direct beam should monotonically decrease
  for (int l = 0; l < nlyr; ++l) {
    EXPECT_GT(r.flux_direct[l], r.flux_direct[l + 1]);
  }
}

TEST(ADSolver, FluxSolver_OpticallyThick) {
  // Very thick layers (tau=10 each), mostly absorbing.
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

  auto r = adrt::solve(cfg);

  // Direct beam at bottom should be essentially zero
  EXPECT_LT(r.flux_direct[5], 1e-20);
  // All fluxes should be finite
  for (int l = 0; l <= 5; ++l) {
    EXPECT_FALSE(std::isnan(r.flux_up[l]));
    EXPECT_FALSE(std::isnan(r.flux_down[l]));
  }
}

TEST(ADSolver, FluxSolver_ReflectiveSurface) {
  // Highly reflective surface (albedo=0.8)
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

  auto r = adrt::solve(cfg);

  // High surface albedo should produce significant upward flux
  EXPECT_GT(r.flux_up[2], 0.3 * (r.flux_direct[2] + r.flux_down[2]));
}


// ============================================================================
//  Configuration Validation Tests
// ============================================================================

TEST(ADConfig, ValidateRejectsInvalid) {
  adrt::ADConfig cfg(0, 8);  // num_layers = 0
  EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

TEST(ADConfig, ValidateRejectsNegativeTau) {
  adrt::ADConfig cfg(1, 8);
  cfg.allocate();
  cfg.delta_tau[0] = -0.1;
  EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

TEST(ADConfig, ValidateRejectsBadAlbedo) {
  adrt::ADConfig cfg(1, 8);
  cfg.allocate();
  cfg.delta_tau[0] = 1.0;
  cfg.single_scat_albedo[0] = 1.5;
  EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

TEST(ADConfig, AllocateSetsCorrectSizes) {
  adrt::ADConfig cfg(3, 8);
  cfg.use_thermal_emission = true;
  cfg.allocate();

  EXPECT_EQ(static_cast<int>(cfg.delta_tau.size()), 3);
  EXPECT_EQ(static_cast<int>(cfg.single_scat_albedo.size()), 3);
  EXPECT_EQ(static_cast<int>(cfg.phase_function_moments.size()), 3);
  EXPECT_EQ(static_cast<int>(cfg.temperature.size()), 4);
  // Phase function moments: 2*nquad = 16
  EXPECT_EQ(static_cast<int>(cfg.phase_function_moments[0].size()), 16);
}

TEST(ADConfig, HenyeyGreensteinMoments) {
  adrt::ADConfig cfg(1, 4);
  cfg.allocate();
  double g = 0.5;
  cfg.setHenyeyGreenstein(g);

  // chi_k = g^k
  for (int k = 0; k < static_cast<int>(cfg.phase_function_moments[0].size()); ++k) {
    double expected = std::pow(g, k);
    EXPECT_NEAR(cfg.phase_function_moments[0][k], expected, 1e-14);
  }
}

TEST(ADConfig, RayleighMoments) {
  adrt::ADConfig cfg(1, 4);
  cfg.allocate();
  cfg.setRayleigh();

  EXPECT_NEAR(cfg.phase_function_moments[0][0], 1.0, 1e-14);
  EXPECT_NEAR(cfg.phase_function_moments[0][1], 0.0, 1e-14);
  EXPECT_NEAR(cfg.phase_function_moments[0][2], 0.1, 1e-14);
  for (int k = 3; k < static_cast<int>(cfg.phase_function_moments[0].size()); ++k)
    EXPECT_NEAR(cfg.phase_function_moments[0][k], 0.0, 1e-14);
}


// ============================================================================
//  Dynamic (non-standard quadrature) fallback test
// ============================================================================

TEST(ADSolver, DynamicFallback) {
  // Use nquad=6 which is not in the switch (2,4,8,16,32) → dynamic path.
  adrt::ADConfig cfg(1, 6);
  cfg.solar_flux = 1.0;
  cfg.solar_mu = 0.5;
  cfg.surface_albedo = 0.0;
  cfg.allocate();
  cfg.delta_tau[0] = 0.5;
  cfg.single_scat_albedo[0] = 0.9;
  cfg.setIsotropic();

  auto r = adrt::solve(cfg);

  EXPECT_GT(r.flux_up[0], 0.0);
  EXPECT_GT(r.flux_down[1], 0.0);
  EXPECT_FALSE(std::isnan(r.flux_up[0]));
}


// ============================================================================
//  Linear-in-tau thermal source (pure absorption, analytical comparison)
// ============================================================================

TEST(ADSolver, LinearSource_PureAbsorption) {
  // Pure absorption with linear Planck source.
  // Compare against analytical solution.
  double tau = 1.0;
  double B_top = 1.0;
  double B_bot = 3.0;

  adrt::ADConfig cfg(1, 8);
  cfg.allocate();
  cfg.delta_tau[0] = tau;
  cfg.single_scat_albedo[0] = 0.0;
  cfg.planck_levels = {B_top, B_bot};
  cfg.setIsotropic();

  auto r = adrt::solve(cfg);

  // Compute analytical solution using Gauss-Legendre quadrature.
  double B_bar = (B_top + B_bot) / 2.0;
  double B_d = (B_bot - B_top) / tau;

  int nmu = 8;
  std::vector<double> mu, wt;
  adrt::gaussLegendre(nmu, mu, wt);

  double F_up_exact = 0.0;
  double F_down_exact = 0.0;
  for (int i = 0; i < nmu; ++i) {
    double trans = std::exp(-tau / mu[i]);
    double one_minus_t = 1.0 - trans;
    double slope_term = mu[i] * one_minus_t - 0.5 * tau * (1.0 + trans);
    double I_up = B_bar * one_minus_t + B_d * slope_term;
    double I_down = B_bar * one_minus_t - B_d * slope_term;
    F_up_exact += 2.0 * PI * wt[i] * mu[i] * I_up;
    F_down_exact += 2.0 * PI * wt[i] * mu[i] * I_down;
  }

  EXPECT_NEAR(r.flux_up[0], F_up_exact, 1e-8);
  EXPECT_NEAR(r.flux_down[1], F_down_exact, 1e-8);
}


// ============================================================================
//  Diffusion Lower Boundary Condition Tests
// ============================================================================

TEST(ADSolver, DiffusionBC_PureAbsorption) {
  // Pure absorption with diffusion lower BC.
  // In the diffusion limit, the upward intensity at the bottom is:
  //   I_up(μ) = B(T_bot) + μ × dB/dτ
  // For pure absorption (ω=0), the upward flux at the bottom should be:
  //   F_up_bot = 2π ∫₀¹ I_up(μ) μ dμ = π B_bot + (2π/3) dB/dτ
  // The solver should propagate this through the atmosphere.
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

  auto r = adrt::solve(cfg);

  // All fluxes should be finite and positive (thermal atmosphere)
  for (int l = 0; l <= 3; ++l) {
    EXPECT_FALSE(std::isnan(r.flux_up[l]));
    EXPECT_FALSE(std::isnan(r.flux_down[l]));
    EXPECT_GT(r.flux_up[l], 0.0);
  }

  // The upward flux at the bottom should exceed B_surface * π
  // because the diffusion BC adds a gradient term.
  double B_bot = adrt::planckFunction(500.0, 600.0, 290.0);
  double B_top_last = adrt::planckFunction(500.0, 600.0, 260.0);
  double dB_dtau = (B_bot - B_top_last) / 0.5;

  // Expected bottom flux: π*B_bot + (2π/3)*dB/dτ (from I_up = B + μ*dB/dτ)
  double F_up_bot_expected = PI * B_bot + (2.0 * PI / 3.0) * dB_dtau;
  EXPECT_NEAR(r.flux_up[3], F_up_bot_expected, 0.01 * F_up_bot_expected);
}

TEST(ADSolver, DiffusionBC_Scattering) {
  // Scattering atmosphere with diffusion lower BC.
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

  auto r = adrt::solve(cfg);

  for (int l = 0; l <= 4; ++l) {
    EXPECT_FALSE(std::isnan(r.flux_up[l]));
    EXPECT_FALSE(std::isnan(r.flux_down[l]));
  }

  // Upward flux at bottom should be larger than at top (warmer below)
  EXPECT_GT(r.flux_up[4], r.flux_up[0]);
}

TEST(ADSolver, DiffusionBC_IgnoresSurface) {
  // When diffusion BC is on, surface_albedo should be ignored.
  // Two runs: one with albedo=0.5, one with albedo=0.0, both with diffusion BC.
  // Results should be identical.
  for (double albedo : {0.0, 0.5}) {
    adrt::ADConfig cfg(2, 8);
    cfg.use_thermal_emission = true;
    cfg.use_diffusion_lower_bc = true;
    cfg.wavenumber_low = 500.0;
    cfg.wavenumber_high = 600.0;
    cfg.surface_albedo = albedo;
    cfg.allocate();

    cfg.delta_tau[0] = 1.0;  cfg.single_scat_albedo[0] = 0.5;
    cfg.delta_tau[1] = 1.0;  cfg.single_scat_albedo[1] = 0.5;
    cfg.temperature = {200.0, 250.0, 300.0};
    cfg.setIsotropic();

    auto r = adrt::solve(cfg);

    // Store first result for comparison
    static double first_flux_up_0 = 0.0;
    static double first_flux_up_2 = 0.0;
    static bool first = true;

    if (first) {
      first_flux_up_0 = r.flux_up[0];
      first_flux_up_2 = r.flux_up[2];
      first = false;
    }
    else {
      EXPECT_NEAR(r.flux_up[0], first_flux_up_0, 1e-12);
      EXPECT_NEAR(r.flux_up[2], first_flux_up_2, 1e-12);
    }
  }
}

TEST(ADSolver, DiffusionBC_VsSurfaceForThickAtmosphere) {
  // For an optically very thick lowest layer, the diffusion BC and
  // the standard surface BC (with B_surface = Planck(T_bottom)) should
  // give similar results at TOA, because the thick layer "shields"
  // the boundary condition.
  double wlo = 500.0, whi = 600.0;

  // Run 1: Standard surface BC
  adrt::ADConfig cfg1(3, 8);
  cfg1.use_thermal_emission = true;
  cfg1.wavenumber_low = wlo;
  cfg1.wavenumber_high = whi;
  cfg1.allocate();
  cfg1.delta_tau[0] = 0.5;  cfg1.single_scat_albedo[0] = 0.5;
  cfg1.delta_tau[1] = 0.5;  cfg1.single_scat_albedo[1] = 0.5;
  cfg1.delta_tau[2] = 50.0; cfg1.single_scat_albedo[2] = 0.5;
  cfg1.temperature = {200.0, 230.0, 260.0, 290.0};
  cfg1.setIsotropic();

  auto r1 = adrt::solve(cfg1);

  // Run 2: Diffusion BC
  adrt::ADConfig cfg2 = cfg1;
  cfg2.use_diffusion_lower_bc = true;

  auto r2 = adrt::solve(cfg2);

  // At TOA, the two should agree well because the thick bottom layer
  // thermalizes the radiation regardless of the exact BC.
  EXPECT_NEAR(r1.flux_up[0], r2.flux_up[0], 0.05 * r1.flux_up[0]);
  EXPECT_NEAR(r1.flux_down[0], r2.flux_down[0], 0.05 * std::abs(r1.flux_down[0]) + 1e-10);
}

TEST(ADSolver, DiffusionBC_WithDeltaM) {
  // Diffusion BC with delta-M scaling should work correctly.
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

  auto r = adrt::solve(cfg);

  for (int l = 0; l <= 3; ++l) {
    EXPECT_FALSE(std::isnan(r.flux_up[l]));
    EXPECT_FALSE(std::isnan(r.flux_down[l]));
    EXPECT_GT(r.flux_up[l], 0.0);
  }
}

TEST(ADSolver, DiffusionBC_DynamicPath) {
  // Test diffusion BC through the dynamic (non-standard nquad) path.
  adrt::ADConfig cfg(2, 6);
  cfg.use_thermal_emission = true;
  cfg.use_diffusion_lower_bc = true;
  cfg.wavenumber_low = 500.0;
  cfg.wavenumber_high = 600.0;
  cfg.allocate();

  cfg.delta_tau[0] = 0.5;  cfg.single_scat_albedo[0] = 0.3;
  cfg.delta_tau[1] = 0.5;  cfg.single_scat_albedo[1] = 0.3;
  cfg.temperature = {200.0, 250.0, 300.0};
  cfg.setIsotropic();

  auto r = adrt::solve(cfg);

  for (int l = 0; l <= 2; ++l) {
    EXPECT_FALSE(std::isnan(r.flux_up[l]));
    EXPECT_FALSE(std::isnan(r.flux_down[l]));
    EXPECT_GT(r.flux_up[l], 0.0);
  }
}
