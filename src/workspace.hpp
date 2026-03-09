/// @file workspace.hpp
/// @brief Pre-allocated workspace for the adding-doubling solver.
///
/// Holds cached data that can be reused across multiple solve() calls,
/// avoiding redundant Legendre polynomial computation and heap allocations.
/// Each workspace instance is caller-owned and must not be shared between
/// threads, preserving the solver's thread-safety model.

#pragma once

#include "quadrature.hpp"

#include <vector>

namespace adrt {

/// Reusable workspace for the adding-doubling solver.
///
/// Create one per thread and pass it to solve() for repeated calls.
/// The workspace caches Legendre polynomials and avoids redundant
/// recomputation when the phase function doesn't change between layers
/// or between successive solve() calls.
struct SolverWorkspace {
  /// Get or compute Legendre polynomials P_l(mu_i) for l=0..L-1.
  /// Returns cached values if L and mu haven't changed since last call.
  const std::vector<std::vector<double>>& getLegendrePolynomials(
      int L, const std::vector<double>& mu) {
    if (L == cached_L_ && mu == cached_mu_)
      return cached_Pl_;

    cached_L_ = L;
    cached_mu_ = mu;
    cached_Pl_ = precomputeLegendrePolynomials(L, mu);
    return cached_Pl_;
  }

  /// Clear all cached data.
  void clear() {
    cached_L_ = 0;
    cached_mu_.clear();
    cached_Pl_.clear();
  }

private:
  int cached_L_ = 0;
  std::vector<double> cached_mu_;
  std::vector<std::vector<double>> cached_Pl_;
};

} // namespace adrt
