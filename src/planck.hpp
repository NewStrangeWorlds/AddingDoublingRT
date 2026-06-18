/// @file planck.hpp
/// @brief Planck function for the adding-doubling radiative transfer solver.

#pragma once

namespace adrt {

/// Compute Planck function integrated between two wavenumbers.
/// @param wnumlo Lower wavenumber [cm^-1]
/// @param wnumhi Upper wavenumber [cm^-1]
/// @param temp   Temperature [K]
/// @return       Integrated Planck function [W/m^2]
double planckFunction(double wnumlo, double wnumhi, double temp);

/// Temperature derivative dB/dT of the band-integrated Planck function.
///
/// Uses the closed form
///   dB/dT = 4 B / T  -  (sigma/pi)(15/pi^4) T^3 (v1 f(v1) - v0 f(v0)),
/// with v_i = C2 wnum_i / T and f(v) = v^3/(e^v - 1), plus a stable
/// single-wavenumber limit for wnumhi == wnumlo.  Required for the chain rule
/// in the temperature-Jacobian computation (see tex/temperature_jacobian_plan).
///
/// @param wnumlo Lower wavenumber [cm^-1]
/// @param wnumhi Upper wavenumber [cm^-1]
/// @param temp   Temperature [K]
/// @return       dB/dT [W/m^2/K]
double planckFunctionDeriv(double wnumlo, double wnumhi, double temp);

} // namespace adrt
