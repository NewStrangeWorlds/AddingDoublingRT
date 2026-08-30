/// @file test_temperature_jacobian.cpp
/// @brief Finite-difference validation of the analytic temperature Jacobians.
///
/// The analytic Jacobians (dF_up/dT, dF_down/dT, dJ/dT, d(divF)/dT at every
/// interface w.r.t. every level temperature and the surface skin temperature)
/// are compared against central finite differences of the forward solver.
/// See tex/temperature_jacobian_plan.tex.

#include "testing.hpp"
#include "adding_doubling.hpp"
#include "constants.hpp"

#include <cmath>
#include <vector>

using adrt::ADConfig;
using adrt::RTOutput;

namespace {

// Forward solve with the Jacobian flag off (for finite differencing).
RTOutput solveForward(ADConfig cfg) {
  cfg.compute_temperature_jacobian = false;
  return adrt::solve(cfg);
}

// Magnitude scale used to set the absolute tolerance floor.
double fluxScale(const RTOutput& r) {
  double s = 1.0;
  for (size_t k = 0; k < r.flux_up.size(); ++k)
    s = std::max(s, std::fabs(r.flux_up[k]) + std::fabs(r.flux_down[k]));
  return s;
}

// Compare one analytic Jacobian column against a central finite difference taken
// by perturbing a temperature DOF. `col` is the Jacobian column; when `surface`
// is true the surface skin temperature is perturbed, otherwise temperature[col].
// `cfg` is taken by value so the knob and the solved config are the same object.
void checkColumn(ADConfig cfg, int col, bool surface,
                 const RTOutput& ana, double scale) {
  const int m = col;
  double& knob = surface ? cfg.surface_temperature : cfg.temperature[col];
  const double T0 = knob;
  const double h = 1.0e-3 * std::max(std::fabs(T0), 1.0);

  knob = T0 + h;
  RTOutput rp = solveForward(cfg);
  knob = T0 - h;
  RTOutput rm = solveForward(cfg);
  knob = T0;

  const int nint = static_cast<int>(ana.flux_up.size());
  const double rtol = 3.0e-3;
  const double atol = 1.0e-6 * scale;

  auto cmp = [&](const std::vector<double>& up, const std::vector<double>& dn,
                 const std::vector<std::vector<double>>& jac) {
    for (int k = 0; k < nint; ++k) {
      double fd = (up[k] - dn[k]) / (2.0 * h);
      double a  = jac[k][m];
      EXPECT_NEAR(a, fd, atol + rtol * std::fabs(fd));
    }
  };

  cmp(rp.flux_up,        rm.flux_up,        ana.flux_up_temperature_jac);
  cmp(rp.flux_down,      rm.flux_down,      ana.flux_down_temperature_jac);
  cmp(rp.mean_intensity, rm.mean_intensity, ana.mean_intensity_temperature_jac);
  cmp(rp.flux_divergence,rm.flux_divergence,ana.flux_divergence_temperature_jac);
}

// Run the full DOF sweep for a configuration.
void runCase(ADConfig cfg) {
  cfg.compute_temperature_jacobian = true;
  RTOutput ana = adrt::solve(cfg);

  const int nlay = cfg.num_layers;
  const int ndof = nlay + 2;
  ASSERT_TRUE(static_cast<int>(ana.flux_up_temperature_jac.size()) == nlay + 1);
  ASSERT_TRUE(static_cast<int>(ana.flux_up_temperature_jac[0].size()) == ndof);

  const double scale = fluxScale(ana);

  // Level temperature DOFs.
  for (int m = 0; m <= nlay; ++m)
    checkColumn(cfg, m, /*surface=*/false, ana, scale);

  // Surface DOF.
  const bool surf_distinct = cfg.surface_temperature >= 0.0;
  if (surf_distinct) {
    checkColumn(cfg, nlay + 1, /*surface=*/true, ana, scale);
  } else {
    // Folded case: surface column must be identically zero.
    for (int k = 0; k <= nlay; ++k)
      EXPECT_NEAR(ana.flux_up_temperature_jac[k][nlay + 1], 0.0, 1e-12);
  }
}

// Build a baseline thermal configuration with a temperature profile.
ADConfig baseThermal(int nlay, int nquad) {
  ADConfig cfg(nlay, nquad);
  cfg.use_thermal_emission = true;
  cfg.wavenumber_low = 500.0;
  cfg.wavenumber_high = 1500.0;
  cfg.allocate();
  for (int l = 0; l < nlay; ++l)
    cfg.delta_tau[l] = 0.3 + 0.1 * l;
  for (int l = 0; l <= nlay; ++l)
    cfg.temperature[l] = 200.0 + 8.0 * l;   // increasing with depth
  return cfg;
}

}  // namespace


TEST(TemperatureJacobian, AbsorptionWithSurface) {
  ADConfig cfg = baseThermal(4, 8);
  for (int l = 0; l < cfg.num_layers; ++l)
    cfg.single_scat_albedo[l] = 0.0;       // pure-absorption branch
  cfg.surface_albedo = 0.0;
  // surface_temperature left at default (-1): folded into bottom level.
  runCase(cfg);
}

TEST(TemperatureJacobian, ScatteringDistinctSurface) {
  ADConfig cfg = baseThermal(5, 8);
  for (int l = 0; l < cfg.num_layers; ++l) {
    cfg.single_scat_albedo[l] = 0.5;
    cfg.setHenyeyGreenstein(0.5, l);
  }
  cfg.surface_albedo = 0.3;
  cfg.surface_temperature = 250.0;          // distinct surface DOF
  runCase(cfg);
}

TEST(TemperatureJacobian, HighScatteringDiffusionBC) {
  ADConfig cfg = baseThermal(5, 8);
  for (int l = 0; l < cfg.num_layers; ++l) {
    cfg.single_scat_albedo[l] = 0.9;
    cfg.setHenyeyGreenstein(0.7, l);
  }
  cfg.use_diffusion_lower_bc = true;        // no surface layer
  runCase(cfg);
}

TEST(TemperatureJacobian, WithSolarBeamInvariance) {
  // The direct beam is temperature-independent; dJ/dT and d(divF)/dT must still
  // match the finite difference of the beam-inclusive outputs.
  ADConfig cfg = baseThermal(4, 8);
  for (int l = 0; l < cfg.num_layers; ++l) {
    cfg.single_scat_albedo[l] = 0.6;
    cfg.setHenyeyGreenstein(0.4, l);
  }
  cfg.surface_albedo = 0.2;
  cfg.surface_temperature = 240.0;
  cfg.solar_flux = 100.0;
  cfg.solar_mu = 0.6;
  runCase(cfg);
}

TEST(TemperatureJacobian, DeltaMScaling) {
  ADConfig cfg(5, 8);
  cfg.use_thermal_emission = true;
  cfg.use_delta_m = true;
  cfg.wavenumber_low = 500.0;
  cfg.wavenumber_high = 1500.0;
  cfg.allocate();
  for (int l = 0; l < cfg.num_layers; ++l) {
    cfg.delta_tau[l] = 0.5;
    cfg.single_scat_albedo[l] = 0.8;
    cfg.setHenyeyGreenstein(0.8, l);
  }
  for (int l = 0; l <= cfg.num_layers; ++l)
    cfg.temperature[l] = 210.0 + 6.0 * l;
  cfg.surface_albedo = 0.1;
  cfg.surface_temperature = 255.0;
  runCase(cfg);
}

TEST(TemperatureJacobian, QuadratureOrders) {
  for (int nquad : {4, 8, 16}) {
    ADConfig cfg = baseThermal(4, nquad);
    for (int l = 0; l < cfg.num_layers; ++l) {
      cfg.single_scat_albedo[l] = 0.7;
      cfg.setHenyeyGreenstein(0.5, l);
    }
    cfg.surface_albedo = 0.25;
    cfg.surface_temperature = 245.0;
    runCase(cfg);
  }
}

// Off-list quadrature counts (6, 12) take the runtime-sized solveDynamic path.
TEST(TemperatureJacobian, DynamicPathScattering) {
  for (int nquad : {6, 12}) {
    ADConfig cfg = baseThermal(5, nquad);
    for (int l = 0; l < cfg.num_layers; ++l) {
      cfg.single_scat_albedo[l] = 0.6;
      cfg.setHenyeyGreenstein(0.5, l);
    }
    cfg.surface_albedo = 0.3;
    cfg.surface_temperature = 250.0;
    runCase(cfg);
  }
}

TEST(TemperatureJacobian, DynamicPathAbsorptionAndDiffusionBC) {
  // Absorption + folded surface on the dynamic path.
  {
    ADConfig cfg = baseThermal(4, 6);
    for (int l = 0; l < cfg.num_layers; ++l) cfg.single_scat_albedo[l] = 0.0;
    cfg.surface_albedo = 0.0;
    runCase(cfg);
  }
  // High scattering + diffusion lower BC on the dynamic path.
  {
    ADConfig cfg = baseThermal(5, 6);
    for (int l = 0; l < cfg.num_layers; ++l) {
      cfg.single_scat_albedo[l] = 0.9;
      cfg.setHenyeyGreenstein(0.7, l);
    }
    cfg.use_diffusion_lower_bc = true;
    runCase(cfg);
  }
}

// Regression: with an explicit top_temperature the TOA downwelling no longer
// depends on temperature[0], so the Jacobian must not attribute a boundary
// sensitivity to column 0. (The default top boundary, B(temperature[0]), is
// already covered above; this exercises the decoupled case the other tests
// missed.)
TEST(TemperatureJacobian, ColdSpaceTopBoundary) {
  // Cold space (top_temperature = 0, matching DisORT's default), scattering,
  // distinct surface. Templated path.
  ADConfig cfg = baseThermal(4, 8);
  for (int l = 0; l < cfg.num_layers; ++l) {
    cfg.single_scat_albedo[l] = 0.6;
    cfg.setHenyeyGreenstein(0.5, l);
  }
  cfg.surface_albedo = 0.2;
  cfg.surface_temperature = 250.0;
  cfg.top_temperature = 0.0;
  runCase(cfg);
}

TEST(TemperatureJacobian, ExplicitTopBoundary) {
  // Finite, distinct top-boundary temperature. Pure absorption (templated) and
  // scattering on the dynamic path.
  {
    ADConfig cfg = baseThermal(4, 8);
    for (int l = 0; l < cfg.num_layers; ++l) cfg.single_scat_albedo[l] = 0.0;
    cfg.surface_albedo = 0.0;
    cfg.top_temperature = 180.0;
    runCase(cfg);
  }
  {
    ADConfig cfg = baseThermal(5, 6);   // dynamic path
    for (int l = 0; l < cfg.num_layers; ++l) {
      cfg.single_scat_albedo[l] = 0.6;
      cfg.setHenyeyGreenstein(0.5, l);
    }
    cfg.surface_albedo = 0.3;
    cfg.surface_temperature = 250.0;
    cfg.top_temperature = 210.0;
    runCase(cfg);
  }
}


// ============================================================================
//  Column windows and the optical-depth band
// ============================================================================

namespace {

// Deep column: cumulative tau 1e-4..1e4, T 500..3000 K, Rayleigh omega = 1e-4
// with a 5-layer scattering deck (omega = 0.9, HG g = 0.7); cold space at top.
ADConfig deepColumn(int nlay, int nquad) {
  ADConfig cfg(nlay, nquad);
  cfg.use_thermal_emission = true;
  cfg.top_temperature = 0.0;
  cfg.surface_albedo = 0.0;
  cfg.wavenumber_low = 1000.0;
  cfg.wavenumber_high = 1001.0;
  cfg.compute_temperature_jacobian = true;
  cfg.allocate();
  std::vector<double> tc(nlay + 1);
  for (int l = 0; l <= nlay; ++l) tc[l] = std::pow(10.0, -4.0 + 8.0 * l / nlay);
  for (int l = 0; l < nlay; ++l) {
    cfg.delta_tau[l] = tc[l + 1] - tc[l];
    cfg.single_scat_albedo[l] = 1e-4;
  }
  cfg.setRayleigh();
  for (int l = nlay / 3; l < nlay / 3 + 5; ++l) {
    cfg.single_scat_albedo[l] = 0.9;
    cfg.setHenyeyGreenstein(0.7, l);
  }
  for (int l = 0; l <= nlay; ++l)
    cfg.temperature[l] = 500.0 + 2500.0 * (std::log10(tc[l]) + 4.0) / 8.0;
  return cfg;
}

double jacMaxAbs(const std::vector<std::vector<double>>& J) {
  double m = 0.0;
  for (const auto& row : J) for (double v : row) m = std::max(m, std::fabs(v));
  return m;
}

double jacMaxDiff(const std::vector<std::vector<double>>& A,
                  const std::vector<std::vector<double>>& B) {
  double m = 0.0;
  for (size_t k = 0; k < A.size(); ++k)
    for (size_t j = 0; j < A[k].size(); ++j) m = std::max(m, std::fabs(A[k][j] - B[k][j]));
  return m;
}

long jacCountZeros(const std::vector<std::vector<double>>& J) {
  long n = 0;
  for (const auto& row : J) for (double v : row) if (v == 0.0) ++n;
  return n;
}

void checkBanding(int nquad) {
  auto run = [&](double band) {
    ADConfig cfg = deepColumn(80, nquad);
    cfg.temperature_jacobian_tau_band = band;
    return adrt::solve(cfg);
  };
  RTOutput dense = run(0.0);       // banding disabled
  RTOutput def   = run(40.0);      // default band: below double round-off
  RTOutput narrow = run(12.0);     // aggressive band: bounded by exp(-12)

  const std::vector<const std::vector<std::vector<double>> RTOutput::*> fields = {
      &RTOutput::flux_up_temperature_jac, &RTOutput::flux_down_temperature_jac,
      &RTOutput::mean_intensity_temperature_jac, &RTOutput::flux_divergence_temperature_jac};

  for (auto f : fields) {
    const double scale = jacMaxAbs(dense.*f);
    EXPECT_TRUE(scale > 0.0);
    EXPECT_LT(jacMaxDiff(def.*f, dense.*f), 1e-13 * scale);
    EXPECT_LT(jacMaxDiff(narrow.*f, dense.*f), 10.0 * std::exp(-12.0) * scale);
  }
  // The band must actually be active on this column (deep layers are ~tau 100s).
  EXPECT_GT(jacCountZeros(def.flux_up_temperature_jac),
            jacCountZeros(dense.flux_up_temperature_jac));
  EXPECT_GT(jacCountZeros(narrow.flux_up_temperature_jac),
            jacCountZeros(def.flux_up_temperature_jac));
}

}  // namespace

TEST(TemperatureJacobian, BandedMatchesDense_N8)  { checkBanding(8); }
TEST(TemperatureJacobian, BandedMatchesDense_N12) { checkBanding(12); }   // dynamic path

TEST(TemperatureJacobian, BandedDeepColumnMatchesFiniteDifference) {
  // End-to-end with the default band on a column whose deep layers exceed it.
  ADConfig cfg = deepColumn(30, 8);
  runCase(cfg);
}

// ============================================================================
//  Interface-averaged albedo in the flux divergence (experiment flag)
// ============================================================================

TEST(TemperatureJacobian, FluxDivergenceOmegaAverage) {
  ADConfig cfg = baseThermal(6, 8);
  cfg.surface_temperature = 320.0;
  std::vector<double> om = {0.0, 0.3, 0.8, 0.8, 0.1, 0.0};
  for (int l = 0; l < 6; ++l) {
    cfg.single_scat_albedo[l] = om[l];
    if (om[l] > 0.0) cfg.setHenyeyGreenstein(0.5, l);
  }
  cfg.compute_temperature_jacobian = true;
  RTOutput off = adrt::solve(cfg);
  cfg.flux_divergence_omega_average = true;
  RTOutput on = adrt::solve(cfg);

  // J and B do not depend on the flag: divF scales with the (1 - omega) prefactor.
  const int nlay = 6;
  for (int k = 0; k <= nlay; ++k) {
    double om_above = om[(k == 0) ? 0 : k - 1];
    double om_avg = (k == 0) ? om[0] : (k == nlay) ? om[nlay - 1] : 0.5 * (om[k - 1] + om[k]);
    EXPECT_NEAR(on.flux_divergence[k] * (1.0 - om_above),
                off.flux_divergence[k] * (1.0 - om_avg), 1e-12 * std::fabs(off.flux_divergence[k]) + 1e-14);
    EXPECT_NEAR(on.mean_intensity[k], off.mean_intensity[k], 1e-14 * std::fabs(off.mean_intensity[k]));
    for (int m = 0; m < nlay + 2; ++m)
      EXPECT_NEAR(on.flux_divergence_temperature_jac[k][m] * (1.0 - om_above),
                  off.flux_divergence_temperature_jac[k][m] * (1.0 - om_avg),
                  1e-12 * std::fabs(off.flux_divergence_temperature_jac[k][m]) + 1e-14);
  }
  // And the Jacobian stays consistent with finite differences of the flagged output.
  runCase(cfg);
}
