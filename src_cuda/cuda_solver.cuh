/// @file cuda_solver.cuh
/// @brief Public CUDA API for the batched adding-doubling RT solver.
///
/// The primary interface is solveBatch(), which assumes all data is already
/// on the GPU (designed for integration with BeAR). A convenience function
/// solveBatchHost() handles host-device transfers for standalone use.

#pragma once

#include <cuda_runtime.h>

#include <vector>

namespace adrt {
namespace cuda {

/// Solver configuration (scalars shared across all wavenumbers).
struct BatchConfig {
  int num_wavenumbers = 0;    ///< Number of wavenumbers to solve in parallel
  int num_layers = 0;         ///< Number of atmospheric layers
  int num_quadrature = 8;     ///< Quadrature order N (2, 4, 8, 16, or 32)
  int num_moments_max = 0;    ///< Max number of Legendre moments across all layers

  bool use_delta_m = false;           ///< Apply delta-M scaling
  bool use_thermal_emission = false;  ///< Compute Planck function from temperature
  bool use_diffusion_lower_bc = false;

  double surface_albedo = 0.0;
  double surface_emission = 0.0;
  double top_emission = 0.0;
  double solar_flux = 0.0;
  double solar_mu = 1.0;

  double wavenumber_low = 0.0;   ///< For Planck function (thermal only)
  double wavenumber_high = 0.0;
};


/// Device memory pointers for batched solve.
/// All arrays use SoA layout: array[wav * stride + layer_or_level].
struct DeviceData {
  // --- Inputs (caller-owned device memory) ---

  const double* delta_tau;            ///< [nwav * nlay] optical depth per layer
  const double* single_scat_albedo;   ///< [nwav * nlay] single-scattering albedo
  const double* phase_moments;        ///< [nwav * nlay * nmom] or [nlay * nmom]
  bool phase_moments_shared = false;  ///< true if moments are the same for all wavenumbers

  /// Planck / thermal source at each level.
  /// If use_thermal_emission: temperature[nwav * nlev] (Planck computed on device).
  /// Otherwise: planck_levels[nwav * nlev] (pre-computed Planck values).
  const double* temperature = nullptr;     ///< [nwav * nlev] or nullptr
  const double* planck_levels = nullptr;   ///< [nwav * nlev] or nullptr

  /// Per-wavenumber surface/top emission (used when not computing from temperature).
  const double* surface_emission = nullptr;  ///< [nwav] or nullptr (uses BatchConfig value)
  const double* top_emission = nullptr;      ///< [nwav] or nullptr

  // --- Outputs (caller-owned device memory) ---

  double* flux_up = nullptr;       ///< [nwav] TOA upward flux
  double* flux_down = nullptr;     ///< [nwav] TOA downward flux
  double* flux_direct = nullptr;   ///< [nwav] direct solar flux at surface, or nullptr
};


/// Launch the batched solver kernel.
/// All data must be on the GPU. Call uploadQuadratureData() first.
///
/// @param config  Solver configuration (scalars)
/// @param data    Device memory pointers
/// @param stream  CUDA stream for async execution
void solveBatch(
    const BatchConfig& config,
    const DeviceData& data,
    cudaStream_t stream = 0);


/// Convenience: solve from host-side data.
/// Allocates device memory, copies data, runs kernel, copies results back.
struct HostResult {
  std::vector<double> flux_up;       ///< [nwav] TOA upward flux
  std::vector<double> flux_down;     ///< [nwav] TOA downward flux
  std::vector<double> flux_direct;   ///< [nwav] direct solar at surface
};

HostResult solveBatchHost(
    const BatchConfig& config,
    const std::vector<double>& delta_tau,           // [nwav * nlay]
    const std::vector<double>& single_scat_albedo,  // [nwav * nlay]
    const std::vector<double>& phase_moments,       // [nwav * nlay * nmom] or [nlay * nmom]
    bool phase_moments_shared,
    const std::vector<double>& planck_levels,        // [nwav * nlev] or empty
    const std::vector<double>& temperature = {});    // [nwav * nlev] or empty


} // namespace cuda
} // namespace adrt
