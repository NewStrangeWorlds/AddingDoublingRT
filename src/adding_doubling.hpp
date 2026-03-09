/// @file adding_doubling.hpp
/// @brief Adding-doubling radiative transfer solver for plane-parallel atmospheres.
///
/// Implements the matrix operator / adding-doubling method of
/// Plass, Hansen & Kattawar (1973, Appl. Optics 12, 314) to compute
/// fluxes and mean intensities in a scattering, absorbing, and emitting
/// plane-parallel atmosphere.

#pragma once

#include "matrix.hpp"
#include "workspace.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>


namespace adrt {

// ============================================================================
//  Output data structure
// ============================================================================

/// Results of the RT calculation.
///
/// All arrays are indexed by interface number:
///   index 0         = top of atmosphere (above layer 0)
///   index n_layers  = bottom of atmosphere (below last layer / surface)
struct RTOutput {
  std::vector<double> flux_up;          ///< Upward diffuse flux [same unit as B]
  std::vector<double> flux_down;        ///< Downward diffuse flux
  std::vector<double> mean_intensity;   ///< J = (1/4pi) integral I dOmega
  std::vector<double> flux_direct;      ///< Attenuated direct solar beam flux
};


// ============================================================================
//  Configuration
// ============================================================================

/// Configuration for the adding-doubling RT solver.
///
/// Mirrors the DisortFluxConfig interface: the user provides flat arrays
/// of optical depths, single-scattering albedos, temperatures, and Legendre
/// moments.  When use_thermal_emission is true, the solver computes Planck
/// functions internally from temperatures and wavenumber bounds.
class ADConfig {
public:
  // ======== Dimensions ========
  int num_layers = 0;
  int num_quadrature = 8;  ///< Gauss-Legendre quadrature points (half-range)

  // ======== Flags ========
  bool use_thermal_emission = false;  ///< Compute Planck sources from temperatures
  bool use_delta_m = false;           ///< Enable delta-M scaling (Wiscombe 1977b)
  bool use_diffusion_lower_bc = false; ///< Use diffusion approximation at lower boundary (stellar atmospheres)
  bool index_from_bottom = false;     ///< If true, user arrays indexed BOA (0) -> TOA

  // ======== Boundary conditions ========
  double surface_albedo = 0.0;      ///< Lambertian surface albedo [0, 1]
  double surface_emission = 0.0;    ///< Raw surface thermal emission (only if !use_thermal_emission)
  double top_emission = 0.0;        ///< Raw isotropic diffuse radiation at TOA (only if !use_thermal_emission)
  double solar_flux = 0.0;          ///< Collimated solar flux at TOA
  double solar_mu = 1.0;            ///< cos(solar zenith angle), must be > 0

  // ======== Thermal emission parameters ========
  double wavenumber_low = 0.0;      ///< Lower wavenumber [cm^-1]
  double wavenumber_high = 0.0;     ///< Upper wavenumber [cm^-1]

  // ======== Layer/Level arrays ========
  std::vector<double> delta_tau;           ///< Optical depth per layer [num_layers]
  std::vector<double> single_scat_albedo;  ///< SSA per layer [num_layers]
  std::vector<double> temperature;         ///< Temperature at levels [num_layers+1] (thermal only)
  std::vector<double> planck_levels;       ///< Raw Planck values at levels [num_layers+1] (alternative to temperature)
  std::vector<std::vector<double>> phase_function_moments;  ///< [num_layers][num_moments]

  // ======== Construction ========
  ADConfig() = default;

  /// Construct with basic dimensions.
  /// @param nlyr  Number of layers
  /// @param nquad Number of quadrature points (half-range, default 8)
  ADConfig(int nlyr, int nquad = 8)
    : num_layers(nlyr), num_quadrature(nquad) {}

  /// Allocate arrays based on current dimensions and flags.
  /// Must be called after setting num_layers, num_quadrature, use_delta_m, and use_thermal_emission.
  void allocate() {
    delta_tau.assign(num_layers, 0.0);
    single_scat_albedo.assign(num_layers, 0.0);

    int nmom = use_delta_m ? 2 * num_quadrature + 1 : 2 * num_quadrature;
    phase_function_moments.assign(num_layers, std::vector<double>(nmom, 0.0));

    for (auto& pm : phase_function_moments)
      pm[0] = 1.0;

    if (use_thermal_emission)
      temperature.assign(num_layers + 1, 0.0);
  }

  /// Validate configuration.
  /// @throws std::invalid_argument if configuration is invalid
  void validate() const {
    if (num_layers <= 0)
      throw std::invalid_argument("ADConfig: num_layers must be > 0");
    if (num_quadrature < 2)
      throw std::invalid_argument("ADConfig: num_quadrature must be >= 2");
    if (static_cast<int>(delta_tau.size()) != num_layers)
      throw std::invalid_argument("ADConfig: delta_tau size mismatch");
    if (static_cast<int>(single_scat_albedo.size()) != num_layers)
      throw std::invalid_argument("ADConfig: single_scat_albedo size mismatch");
    if (static_cast<int>(phase_function_moments.size()) != num_layers)
      throw std::invalid_argument("ADConfig: phase_function_moments size mismatch");

    for (int l = 0; l < num_layers; ++l) 
    {
      if (delta_tau[l] < 0.0)
        throw std::invalid_argument("ADConfig: delta_tau[" + std::to_string(l) + "] < 0");
      if (single_scat_albedo[l] < 0.0 || single_scat_albedo[l] > 1.0)
        throw std::invalid_argument("ADConfig: single_scat_albedo[" + std::to_string(l) + "] out of [0,1]");
    }

    if (surface_albedo < 0.0 || surface_albedo > 1.0)
      throw std::invalid_argument("ADConfig: surface_albedo out of [0,1]");
    if (solar_flux > 0.0 && (solar_mu <= 0.0 || solar_mu > 1.0))
      throw std::invalid_argument("ADConfig: solar_mu must be in (0,1] when solar_flux > 0");

    if (use_thermal_emission) 
    {
      if (static_cast<int>(temperature.size()) != num_layers + 1)
        throw std::invalid_argument("ADConfig: temperature size must be num_layers+1");
      for (int l = 0; l <= num_layers; ++l)
        if (temperature[l] < 0.0)
          throw std::invalid_argument("ADConfig: temperature[" + std::to_string(l) + "] < 0");
      if (wavenumber_low < 0.0 || wavenumber_high <= wavenumber_low)
        throw std::invalid_argument("ADConfig: invalid wavenumber range");
    }

    if (use_diffusion_lower_bc) 
    {
      if (!use_thermal_emission && static_cast<int>(planck_levels.size()) != num_layers + 1)
        throw std::invalid_argument("ADConfig: use_diffusion_lower_bc requires use_thermal_emission or planck_levels");
    }
  }

  // ======== Phase function helpers ========

  /// Set single Henyey-Greenstein phase function.
  /// @param g  Asymmetry parameter
  /// @param lc Layer index (-1 = all layers)
  void setHenyeyGreenstein(double g, int lc = -1) {
    int start = (lc < 0) ? 0 : lc;
    int end = (lc < 0) ? num_layers : lc + 1;
    
    for (int l = start; l < end; ++l) 
    {
      double gk = 1.0;
      for (int k = 0; k < static_cast<int>(phase_function_moments[l].size()); ++k) {
        phase_function_moments[l][k] = gk;
        gk *= g;
      }
    }
  }

  /// Set double Henyey-Greenstein phase function.
  /// @param f   Weight of first lobe
  /// @param g1  Asymmetry parameter of first lobe
  /// @param g2  Asymmetry parameter of second lobe
  /// @param lc  Layer index (-1 = all layers)
  void setDoubleHenyeyGreenstein(double f, double g1, double g2, int lc = -1) {
    int start = (lc < 0) ? 0 : lc;
    int end = (lc < 0) ? num_layers : lc + 1;
    
    for (int l = start; l < end; ++l) 
    {
      double g1k = 1.0, g2k = 1.0;
      
      for (int k = 0; k < static_cast<int>(phase_function_moments[l].size()); ++k) 
      {
        phase_function_moments[l][k] = f * g1k + (1.0 - f) * g2k;
        g1k *= g1;
        g2k *= g2;
      }
    }
  }

  /// Set isotropic scattering (all moments zero except zeroth = 1).
  /// @param lc Layer index (-1 = all layers)
  void setIsotropic(int lc = -1) {
    int start = (lc < 0) ? 0 : lc;
    int end = (lc < 0) ? num_layers : lc + 1;
    
    for (int l = start; l < end; ++l) 
    {
      std::fill(phase_function_moments[l].begin(), phase_function_moments[l].end(), 0.0);
      phase_function_moments[l][0] = 1.0;
    }
  }

  /// Set Rayleigh scattering phase function.
  /// Legendre moments: chi_0 = 1, chi_1 = 0, chi_2 = 0.1, chi_l = 0 for l > 2.
  /// (The (2l+1) factor is applied in the phase matrix construction.)
  /// @param lc Layer index (-1 = all layers)
  void setRayleigh(int lc = -1) {
    int start = (lc < 0) ? 0 : lc;
    int end = (lc < 0) ? num_layers : lc + 1;
    
    for (int l = start; l < end; ++l) 
    {
      std::fill(phase_function_moments[l].begin(), phase_function_moments[l].end(), 0.0);
      phase_function_moments[l][0] = 1.0;
      
      if (static_cast<int>(phase_function_moments[l].size()) > 2)
        phase_function_moments[l][2] = 0.1;
    }
  }
};


// ============================================================================
//  Solver
// ============================================================================

/// Solve the radiative transfer problem.
///
/// @param config  Solver configuration (layers, boundary conditions, flags).
/// @return        Fluxes and mean intensities at each layer interface.
RTOutput solve(const ADConfig& config);

/// Solve with a reusable workspace for improved performance.
///
/// The workspace caches Legendre polynomials and avoids redundant
/// recomputation across layers and across successive solve() calls.
/// Each workspace must be used by a single thread only.
///
/// @param config     Solver configuration.
/// @param workspace  Caller-owned workspace (must not be shared between threads).
/// @return           Fluxes and mean intensities at each layer interface.
RTOutput solve(const ADConfig& config, SolverWorkspace& workspace);


// ============================================================================
//  Utilities (exposed for testing)
// ============================================================================

/// Compute Gauss-Legendre quadrature nodes and weights on [0, 1].
void gaussLegendre(
    int n,
    std::vector<double>& nodes,
    std::vector<double>& weights);

/// Build azimuthally-averaged phase matrices from Legendre coefficients.
/// Uses the addition theorem for the m=0 component, followed by Hansen normalization.
void computePhaseMatricesFromLegendre(
    const std::vector<double>& chi,
    const std::vector<double>& mu,
    const std::vector<double>& weights,
    DynamicMatrix& Ppp,
    DynamicMatrix& Ppm);

/// Compute Planck function integrated between two wavenumbers.
/// @param wnumlo Lower wavenumber [cm^-1]
/// @param wnumhi Upper wavenumber [cm^-1]
/// @param temp   Temperature [K]
/// @return       Integrated Planck function [W/m^2]
double planckFunction(double wnumlo, double wnumhi, double temp);

} // namespace adrt
