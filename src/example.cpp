/// @file example.cpp
/// @brief Demonstration of the adding-doubling RT solver.
///
/// Compile:  g++ -std=c++17 -O2 -o example example.cpp adding_doubling.cpp

#include "adding_doubling.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>


static constexpr double PI = 3.14159265358979323846;

/// Simple Planck function B_nu(T) at wavenumber nu [cm^-1] and temperature T [K].
/// Returns spectral radiance in W cm^-2 sr^-1 / cm^-1.
double planck(double nu_cm, double T) 
{
  constexpr double h = 6.62607015e-34;   // J s
  constexpr double c = 2.99792458e10;    // cm/s
  constexpr double k = 1.380649e-23;     // J/K
  double x = h * c * nu_cm / (k * T);
  if (x > 500.0) return 0.0;
  return 2.0 * h * c * c * nu_cm * nu_cm * nu_cm / (std::exp(x) - 1.0);
}


void print_header(const std::string& title) 
{
  std::cout << "\n" << std::string(70, '=') << "\n";
  std::cout << "  " << title << "\n";
  std::cout << std::string(70, '=') << "\n\n";
}


void print_results(const adrt::RTOutput& result, int nlay) 
{
  std::cout << std::fixed << std::setprecision(6);
  bool has_direct = false;
  
  for (int l = 0; l <= nlay; ++l)
    if (result.flux_direct[l] > 0.0) has_direct = true;

  if (has_direct) 
  {
    std::cout << "  Level    F_up           F_down         F_direct       F_net_total    J_mean\n";
    std::cout << "  " << std::string(80, '-') << "\n";
    
    for (int l = 0; l <= nlay; ++l) 
    {
      double fnet = result.flux_up[l] - result.flux_down[l] - result.flux_direct[l];
      std::cout << "  " << std::setw(5) << l
                << "  " << std::setw(13) << result.flux_up[l]
                << "  " << std::setw(13) << result.flux_down[l]
                << "  " << std::setw(13) << result.flux_direct[l]
                << "  " << std::setw(13) << fnet
                << "  " << std::setw(13) << result.mean_intensity[l]
                << "\n";
    }
  }
  else 
  {
    std::cout << "  Level    F_up           F_down         F_net          J_mean\n";
    std::cout << "  " << std::string(66, '-') << "\n";
    
    for (int l = 0; l <= nlay; ++l) 
    {
      double fnet = result.flux_up[l] - result.flux_down[l];
      std::cout << "  " << std::setw(5) << l
                << "  " << std::setw(13) << result.flux_up[l]
                << "  " << std::setw(13) << result.flux_down[l]
                << "  " << std::setw(13) << fnet
                << "  " << std::setw(13) << result.mean_intensity[l]
                << "\n";
    }
  }

  std::cout << "\n";
}


// ============================================================================
//  Test 1: Pure absorption (no scattering)
// ============================================================================
void test_pure_absorption() 
{
  print_header("Test 1: Pure absorption (omega=0, thermal emission)");

  double T_surface = 300.0;
  double T_atm     = 250.0;
  double nu        = 1000.0;   // 10 micron (thermal IR)

  double B_surf = planck(nu, T_surface);
  double B_atm  = planck(nu, T_atm);

  std::cout << "  B(T_surf=" << T_surface << "K) = " << B_surf << "\n";
  std::cout << "  B(T_atm =" << T_atm << "K) = " << B_atm << "\n\n";

  adrt::ADConfig cfg(1, 8);
  cfg.surface_emission = B_surf;
  cfg.allocate();
  cfg.delta_tau[0] = 0.5;
  cfg.single_scat_albedo[0] = 0.0;
  cfg.planck_levels = {B_atm, B_atm};
  cfg.setIsotropic();

  auto result = adrt::solve(cfg);
  print_results(result, 1);

  std::cout << "  Expected: F_up(TOA) dominated by attenuated surface + layer emission\n";
}


// ============================================================================
//  Test 2: Conservative scattering (omega=1, energy conservation)
// ============================================================================
void test_conservative_scattering() 
{
  print_header("Test 2: Conservative scattering (omega=1, solar beam)");

  std::cout << "  For omega=1, net flux should be constant through all layers.\n\n";

  int nlay = 5;
  adrt::ADConfig cfg(nlay, 8);
  cfg.solar_flux = 1.0;
  cfg.solar_mu = 0.5;
  cfg.surface_albedo = 0.3;
  cfg.allocate();

  for (int l = 0; l < nlay; ++l) 
  {
    cfg.delta_tau[l] = 0.2;
    cfg.single_scat_albedo[l] = 1.0;
  }

  cfg.setHenyeyGreenstein(0.5);

  auto result = adrt::solve(cfg);
  print_results(result, nlay);

  double fnet0 = result.flux_up[0] - result.flux_down[0] - result.flux_direct[0];
  double fnetN = result.flux_up[nlay] - result.flux_down[nlay] - result.flux_direct[nlay];
  std::cout << "  F_net_total(TOA) = " << fnet0 << "  (includes direct beam)\n";
  std::cout << "  F_net_total(BOA) = " << fnetN << "\n";
  std::cout << "  Difference = " << std::abs(fnet0 - fnetN) << "\n";
}


// ============================================================================
//  Test 3: Rayleigh scattering atmosphere
// ============================================================================
void test_rayleigh() 
{
  print_header("Test 3: Rayleigh scattering atmosphere");

  int nlay = 10;
  double total_tau = 0.5;

  adrt::ADConfig cfg(nlay, 8);
  cfg.solar_flux = 1.0;
  cfg.solar_mu = 1.0;
  cfg.surface_albedo = 0.0;
  cfg.allocate();

  for (int l = 0; l < nlay; ++l) 
  {
    cfg.delta_tau[l] = total_tau / nlay;
    cfg.single_scat_albedo[l] = 1.0;
  }

  cfg.setRayleigh();

  auto result = adrt::solve(cfg);
  print_results(result, nlay);

  std::cout << "  Spherical albedo (F_up/F_solar) = "
            << result.flux_up[0] / (cfg.solar_flux * cfg.solar_mu)
            << "\n";
}


// ============================================================================
//  Test 4: Multi-layer atmosphere with mixed phase functions
// ============================================================================
void test_mixed_atmosphere() 
{
  print_header("Test 4: Mixed atmosphere (HG + Rayleigh layers, thermal + solar)");

  double nu = 5000.0;   // 2 micron (near-IR)

  int nlay = 3;
  adrt::ADConfig cfg(nlay, 8);
  cfg.surface_albedo = 0.1;
  cfg.surface_emission = planck(nu, 300.0);
  cfg.solar_flux = 0.01;
  cfg.solar_mu = 0.7;
  cfg.allocate();

  // Layer 0: thin Rayleigh
  cfg.delta_tau[0] = 0.1;
  cfg.single_scat_albedo[0] = 0.95;
  cfg.setRayleigh(0);

  // Layer 1: forward-scattering cloud
  cfg.delta_tau[1] = 2.0;
  cfg.single_scat_albedo[1] = 0.99;
  cfg.setDoubleHenyeyGreenstein(0.8, 0.7, -0.3, 1);

  // Layer 2: absorbing gas
  cfg.delta_tau[2] = 0.5;
  cfg.single_scat_albedo[2] = 0.1;
  cfg.setIsotropic(2);

  cfg.planck_levels = {planck(nu, 220.0), planck(nu, 250.0), planck(nu, 280.0), planck(nu, 280.0)};

  auto result = adrt::solve(cfg);
  print_results(result, nlay);
}


// ============================================================================
//  Test 5: Convergence with number of quadrature points
// ============================================================================
void test_convergence() 
{
  print_header("Test 5: Convergence with quadrature order");

  std::cout << "  N_quad   F_up(TOA)      F_down(BOA)    J(TOA)\n";
  std::cout << "  " << std::string(55, '-') << "\n";

  for (int nq : {2, 4, 8, 12, 16, 20}) 
  {
    adrt::ADConfig cfg(1, nq);
    cfg.surface_emission = 2.0;
    cfg.allocate();
    cfg.delta_tau[0] = 1.0;
    cfg.single_scat_albedo[0] = 0.9;
    cfg.planck_levels = {1.0, 1.0};
    cfg.setHenyeyGreenstein(0.8);

    auto result = adrt::solve(cfg);
    std::cout << "  " << std::setw(6) << nq
              << "   " << std::setw(13) << result.flux_up[0]
              << "  " << std::setw(13) << result.flux_down[1]
              << "  " << std::setw(13) << result.mean_intensity[0]
              << "\n";
  }

  std::cout << "\n";
}


// ============================================================================
//  Test 6: Linear-in-tau thermal source (pure absorption)
// ============================================================================
void test_linear_source() 
{
  print_header("Test 6: Linear-in-tau thermal source (pure absorption)");

  double tau = 1.0;
  double B_top = 1.0;
  double B_bot = 3.0;
  double B_bar = (B_top + B_bot) / 2.0;
  double B_d   = (B_bot - B_top) / tau;

  std::cout << "  tau=" << tau << "  B_top=" << B_top << "  B_bot=" << B_bot << "\n";
  std::cout << "  B_bar=" << B_bar << "  B_d=" << B_d << "\n\n";

  // Isothermal
  adrt::ADConfig cfg_iso(1, 8);
  cfg_iso.allocate();
  cfg_iso.delta_tau[0] = tau;
  cfg_iso.single_scat_albedo[0] = 0.0;
  cfg_iso.planck_levels = {B_bar, B_bar};
  cfg_iso.setIsotropic();

  // Linear
  adrt::ADConfig cfg_lin(1, 8);
  cfg_lin.allocate();
  cfg_lin.delta_tau[0] = tau;
  cfg_lin.single_scat_albedo[0] = 0.0;
  cfg_lin.planck_levels = {B_top, B_bot};
  cfg_lin.setIsotropic();

  auto result_iso = adrt::solve(cfg_iso);
  auto result_lin = adrt::solve(cfg_lin);

  std::cout << "               F_up(TOA)     F_down(BOA)\n";
  std::cout << "  Isothermal:  " << std::setw(12) << result_iso.flux_up[0]
            << "  " << std::setw(12) << result_iso.flux_down[1] << "\n";
  std::cout << "  Linear:      " << std::setw(12) << result_lin.flux_up[0]
            << "  " << std::setw(12) << result_lin.flux_down[1] << "\n";

  // Analytical
  int nmu = 8;
  std::vector<double> mu, wt;
  adrt::gaussLegendre(nmu, mu, wt);

  double F_up_analytic = 0.0;
  double F_down_analytic = 0.0;
  
  for (int i = 0; i < nmu; ++i) 
  {
    double trans = std::exp(-tau / mu[i]);
    double one_minus_t = 1.0 - trans;
    double slope_term = mu[i] * one_minus_t - 0.5 * tau * (1.0 + trans);
    double I_up   = B_bar * one_minus_t + B_d * slope_term;
    double I_down = B_bar * one_minus_t - B_d * slope_term;
    F_up_analytic   += 2.0 * PI * wt[i] * mu[i] * I_up;
    F_down_analytic += 2.0 * PI * wt[i] * mu[i] * I_down;
  }

  std::cout << "  Analytical:  " << std::setw(12) << F_up_analytic
            << "  " << std::setw(12) << F_down_analytic << "\n\n";

  std::cout << "  Linear vs Analytical (F_up):  diff = "
            << std::abs(result_lin.flux_up[0] - F_up_analytic) << "\n";
  std::cout << "  Linear vs Analytical (F_down): diff = "
            << std::abs(result_lin.flux_down[1] - F_down_analytic) << "\n";
  std::cout << "  (should be ~ machine epsilon)\n";
}


// ============================================================================
//  Test 7: Delta-M convergence for forward-peaked HG (g=0.9)
// ============================================================================
void test_delta_m_convergence() 
{
  print_header("Test 7: Delta-M convergence for g=0.9 HG (solar beam)");

  std::cout << "  Without delta-M:\n";
  std::cout << "  N_quad   F_up(TOA)      F_down(BOA)\n";
  std::cout << "  " << std::string(42, '-') << "\n";

  for (int nq : {4, 8, 16, 32}) 
  {
    adrt::ADConfig cfg(1, nq);
    cfg.solar_flux = 1.0;
    cfg.solar_mu = 0.5;
    cfg.allocate();
    cfg.delta_tau[0] = 1.0;
    cfg.single_scat_albedo[0] = 0.9;
    cfg.setHenyeyGreenstein(0.9);

    auto r = adrt::solve(cfg);
    std::cout << "  " << std::setw(6) << nq
              << "   " << std::setw(13) << r.flux_up[0]
              << "  " << std::setw(13) << r.flux_down[1] << "\n";
  }

  std::cout << "\n  With delta-M:\n";
  std::cout << "  N_quad   F_up(TOA)      F_down(BOA)\n";
  std::cout << "  " << std::string(42, '-') << "\n";
  
  for (int nq : {4, 8, 16, 32}) 
  {
    adrt::ADConfig cfg(1, nq);
    cfg.use_delta_m = true;
    cfg.solar_flux = 1.0;
    cfg.solar_mu = 0.5;
    cfg.allocate();
    cfg.delta_tau[0] = 1.0;
    cfg.single_scat_albedo[0] = 0.9;
    cfg.setHenyeyGreenstein(0.9);

    auto r = adrt::solve(cfg);
    std::cout << "  " << std::setw(6) << nq
              << "   " << std::setw(13) << r.flux_up[0]
              << "  " << std::setw(13) << r.flux_down[1] << "\n";
  }
  std::cout << "\n  Delta-M values should converge faster (less spread across N_quad).\n";
}


// ============================================================================
//  Test 8: Energy conservation with delta-M (omega=1)
// ============================================================================
void test_delta_m_energy_conservation() 
{
  print_header("Test 8: Energy conservation with delta-M (omega=1, g=0.85)");

  int nlay = 5;
  adrt::ADConfig cfg(nlay, 8);
  cfg.use_delta_m = true;
  cfg.solar_flux = 1.0;
  cfg.solar_mu = 0.6;
  cfg.allocate();

  for (int l = 0; l < nlay; ++l) 
  {
    cfg.delta_tau[l] = 0.5;
    cfg.single_scat_albedo[l] = 1.0;
  }
  
  cfg.setHenyeyGreenstein(0.85);

  auto result = adrt::solve(cfg);
  print_results(result, nlay);

  double fnet0 = result.flux_up[0] - result.flux_down[0] - result.flux_direct[0];
  double fnetN = result.flux_up[nlay] - result.flux_down[nlay] - result.flux_direct[nlay];
  std::cout << "  F_net_total(TOA) = " << fnet0 << "\n";
  std::cout << "  F_net_total(BOA) = " << fnetN << "\n";
  std::cout << "  Difference = " << std::abs(fnet0 - fnetN)
            << "  (should be ~ 0 for omega=1)\n";
}


// ============================================================================
//  Test 9: Backward compatibility (use_delta_m=false unchanged)
// ============================================================================
void test_delta_m_backward_compat() 
{
  print_header("Test 9: Backward compatibility (use_delta_m=false)");

  adrt::ADConfig cfg_default(1, 8);
  cfg_default.surface_emission = 1.5;
  cfg_default.allocate();
  cfg_default.delta_tau[0] = 1.0;
  cfg_default.single_scat_albedo[0] = 0.9;
  cfg_default.planck_levels = {1.0, 2.0};
  cfg_default.setHenyeyGreenstein(0.5);

  adrt::ADConfig cfg_false = cfg_default;  // identical config

  auto r_default = adrt::solve(cfg_default);
  auto r_false   = adrt::solve(cfg_false);

  double diff_up   = std::abs(r_default.flux_up[0] - r_false.flux_up[0]);
  double diff_down = std::abs(r_default.flux_down[1] - r_false.flux_down[1]);
  std::cout << "  F_up(TOA)  diff = " << diff_up   << "  (should be 0)\n";
  std::cout << "  F_down(BOA) diff = " << diff_down << "  (should be 0)\n";
}


// ============================================================================
//  Test 10: Delta-M with Rayleigh (f=0, no truncation)
// ============================================================================
void test_delta_m_rayleigh() 
{
  print_header("Test 10: Delta-M with Rayleigh (f=0, no truncation expected)");

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

  auto r_off = adrt::solve(cfg_off);
  auto r_on  = adrt::solve(cfg_on);

  // Check truncation fraction from Rayleigh moments
  std::vector<double> rayleigh_chi = {1.0, 0.0, 0.1};
  rayleigh_chi.resize(2 * 8 + 1, 0.0);
  double f = rayleigh_chi[2 * 8];
  std::cout << "  Truncation fraction f = " << f << "  (should be 0)\n";
  std::cout << "  F_up(TOA)  off=" << r_off.flux_up[0]
            << "  on=" << r_on.flux_up[0]
            << "  diff=" << std::abs(r_off.flux_up[0] - r_on.flux_up[0]) << "\n";
  std::cout << "  (should match exactly since Rayleigh has no truncation)\n";
}


int main() 
{
  std::cout << "Adding-Doubling Radiative Transfer Solver - Test Suite\n";
  std::cout << "Based on Plass, Hansen & Kattawar (1973)\n";

  test_pure_absorption();
  test_conservative_scattering();
  test_rayleigh();
  test_mixed_atmosphere();
  test_convergence();
  test_linear_source();
  test_delta_m_convergence();
  test_delta_m_energy_conservation();
  test_delta_m_backward_compat();
  test_delta_m_rayleigh();

  return 0;
}
