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

} // namespace adrt
