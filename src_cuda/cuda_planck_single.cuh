/// @file cuda_planck_single.cuh
/// @brief Single-wavenumber Planck function B(T, ν) for device-side use.
///
/// Evaluates the Planck function at a single wavenumber (no band integration).
/// Matches the implementation used in the BeAR retrieval code.
/// Units: W m⁻² cm (CGS wavenumber convention).

#pragma once

namespace adrt {
namespace cuda {

/// Single-wavenumber Planck function B(T, ν).
///
/// @param temperature  Temperature in Kelvin
/// @param wavenumber   Wavenumber in cm⁻¹
/// @return Planck function value in W m⁻² cm
__forceinline__ __device__
float planck_single(float temperature, float wavenumber)
{
  // Physical constants in CGS:
  //   h  = 6.62606896e-27  g cm² / s
  //   c  = 2.99792458e10   cm / s
  //   k  = 1.3806504e-16   g cm² / (K s²)
  //
  // c1 = 2 h c² × 1e-3 = 1.19105e-08  (unit conversion to W m⁻² cm)
  // c2 = h c / k         = 1.43879
  // Same values as the CPU adrt::planckFunction (planck.cpp) so that the raw
  // path and a CPU reference agree: c1 = 1.1911e-8, c2 = 1.438786 (the former
  // 1.19105e-8 / 1.43879 gave a systematic -1.6e-4 at c2 nu / T ~ 50).
  constexpr float c1 = 1.1911e-08f;
  constexpr float c2 = 1.438786f;

  if (temperature < 1e-5f) return 0.0f;

  float wn3 = wavenumber * wavenumber * wavenumber;
  return (c1 * wn3) / (__expf(c2 * wavenumber / temperature) - 1.0f);
}

} // namespace cuda
} // namespace adrt
