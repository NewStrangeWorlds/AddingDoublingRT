/// @file quadrature.hpp
/// @brief Gauss-Legendre quadrature and Legendre polynomial utilities.

#pragma once

#include <vector>

namespace adrt {

/// Compute Gauss-Legendre quadrature nodes and weights on [0, 1].
void gaussLegendre(
  int n,
  std::vector<double>& nodes,
  std::vector<double>& weights);

/// Precompute Legendre polynomials P_l(x) for l=0..L-1 at given x values.
/// @param L  Number of Legendre orders
/// @param x  Evaluation points
/// @return   Pl[l][i] = P_l(x_i)
std::vector<std::vector<double>> precomputeLegendrePolynomials(
  int L, 
  const std::vector<double>& x);

} // namespace adrt
